package main

import (
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	opus "gopkg.in/hraban/opus.v2"
)

// The outbound (agent -> user) audio plane. TTS PCM arrives from the
// orchestrator over the existing gRPC stream in per-clause bursts; this buffer
// paces it onto the WebRTC Opus track so the user hears smooth, continuous
// speech instead of raw bursts, and so barge-in can be cut LOCALLY (before any
// network round-trip) the instant the edge VAD hears the user.
//
// Bias: smoothness over latency. We accumulate a prebuffer cushion before we
// start emitting, and if we run dry mid-utterance we rebuild the cushion rather
// than dribble choppy single frames. This trades a little added delay for
// glitch-free playback, which is what a phone call should feel like.
const (
	playoutRate      = 48000 // Opus RTP clock rate (fixed by RFC 7587)
	playoutFrameMS   = 20
	playoutFrameSamp = playoutRate / 1000 * playoutFrameMS // 960 samples / 20 ms frame
)

// Tunables (env). Prebuffer is the cushion depth before playback starts / after
// an underrun; larger = smoother but more latency. Max caps memory when TTS
// outruns real time (GPU synth is often faster than playback).
var (
	playoutPrebufferMS = envInt("EDGE_PLAYOUT_PREBUFFER_MS", 200, 0, 2000)
	playoutMaxMS       = envInt("EDGE_PLAYOUT_MAX_MS", 30000, 1000, 120000)
	// Ceiling on how long buffered audio may be held back waiting for the cushion
	// to refill. Without it, a clause shorter than the cushion is never played at
	// all: the LLM streams clause-by-clause, so a short sentence ("Okay.") can be
	// all that arrives before the next synthesis gap, and the cushion it is
	// waiting on can only be filled by audio that is still seconds away.
	playoutMaxStallMS = envInt("EDGE_PLAYOUT_MAX_STALL_MS", 2*playoutPrebufferMS, 0, 5000)
)

func envInt(key string, def, lo, hi int) int {
	n, err := strconv.Atoi(os.Getenv(key))
	if err != nil || n < lo || n > hi {
		return def
	}
	return n
}

// playoutSink is the half of playout that runBridge drives. Kept an interface so
// the decode/forward loop stays decoupled from WebRTC and can be handed a nil
// (or fake) sink without pulling in a real Opus track.
type playoutSink interface {
	Write(sessionID string, turnID int64, pcm []byte, sampleRate int) // queue TTS PCM16 (little-endian, mono)
	SetPlaybackStarted(func(sessionID string, turnID int64, queueMS int64))
	Flush() // barge-in / stop-playback: drop everything now
	// MarkTurnEnd records that the orchestrator has finished producing audio for
	// turn turnID. The completion callback fires only once the buffered audio has
	// actually drained, which is the moment a UI may unmute the mic — the
	// orchestrator's own AGENT_TURN_ENDED is up to a prebuffer too early.
	MarkTurnEnd(sessionID string, turnID int64, reason string)
}

// turnEnd is a pending drain marker: the orchestrator said turn turnID is over,
// and we owe the client a PLAYBACK_COMPLETE once the tail has been emitted.
type turnEnd struct {
	sessionID string
	turnID    int64
	reason    string
}

type playout struct {
	track *webrtc.TrackLocalStaticSample
	enc   *opus.Encoder

	mu              sync.Mutex
	buf             []int16 // 48 kHz mono PCM awaiting playout
	playing         bool
	bufferSessionID string
	bufferTurnID    int64
	queuedAt        time.Time
	startedTurns    map[int64]bool
	pendingStart    *playbackStart
	pendingEnd      *turnEnd
	// When non-zero, the instant we first held sub-cushion audio without playing.
	// Bounds the wait so short clauses cannot be stranded (see playoutMaxStallMS).
	stallSince time.Time

	// onTurnComplete is invoked (off the playout goroutine's lock) when a marked
	// turn's audio has drained or been cut. nil is allowed.
	onTurnComplete func(sessionID string, turnID int64, reason string)
	onTurnStart    func(sessionID string, turnID int64, queueMS int64)

	startSamples int
	maxSamples   int
	maxStall     time.Duration

	// Lip-sync with an avatar (see avatar.go). Video can only be rendered once
	// the audio it depicts is known, so when an avatar is attached every tick
	// is shown to tap immediately and then held in delay for a fixed number of
	// ticks before it reaches the track. Both are nil/empty without an avatar,
	// and the playout then behaves exactly as it did before.
	tap       playoutTap
	delay     []delaySlot // ring; len is the delay in ticks
	delayHead int
	dueFrame  []int16

	encScratch []byte
	stop       chan struct{}
	done       chan struct{}
}

type playbackStart struct {
	sessionID string
	turnID    int64
	queueMS   int64
}

// playoutTap observes the agent audio timeline one tick at a time, before any
// lip-sync delay is applied. The avatar renders video from what it sees here.
type playoutTap interface {
	// Tick receives the 48 kHz frame scheduled for this tick, or nil when the
	// agent is silent. It runs on the playout clock under the playout lock, so
	// it must not block or call back into the playout.
	Tick(frame []int16, now time.Time)
}

// tick is what one playout tick owes the wire: a frame to emit (nil for none)
// and the turn boundaries that belong with it.
type tick struct {
	frame []int16
	start *playbackStart
	end   *turnEnd
}

// delaySlot is one tick parked in the lip-sync delay line.
type delaySlot struct {
	pcm   []int16
	live  bool // pcm holds audio (false = silence)
	start *playbackStart
	end   *turnEnd
}

func newPlayout(track *webrtc.TrackLocalStaticSample,
	onTurnComplete func(sessionID string, turnID int64, reason string)) (*playout, error) {
	enc, err := opus.NewEncoder(playoutRate, 1, opus.AppVoIP)
	if err != nil {
		return nil, err
	}
	p := &playout{
		track:          track,
		enc:            enc,
		onTurnComplete: onTurnComplete,
		startedTurns:   make(map[int64]bool),
		startSamples:   playoutPrebufferMS * playoutRate / 1000,
		maxStall:       time.Duration(playoutMaxStallMS) * time.Millisecond,
		maxSamples:     playoutMaxMS * playoutRate / 1000,
		encScratch:     make([]byte, 4000), // max Opus packet
		stop:           make(chan struct{}),
		done:           make(chan struct{}),
	}
	go p.run()
	return p, nil
}

func (p *playout) SetPlaybackStarted(callback func(sessionID string, turnID int64, queueMS int64)) {
	p.mu.Lock()
	p.onTurnStart = callback
	p.mu.Unlock()
}

// Write queues one TTS PCM chunk, resampled to 48 kHz. Under runaway overload
// (TTS far outrunning playback) it drops the oldest audio to bound memory; that
// is a last resort, not the normal path.
func (p *playout) Write(sessionID string, turnID int64, pcm []byte, sampleRate int) {
	if len(pcm) < 2 {
		return
	}
	up := resampleTo48k(bytesToInt16(pcm), sampleRate)
	if len(up) == 0 {
		return
	}
	p.mu.Lock()
	if len(p.buf) == 0 {
		p.bufferSessionID = sessionID
		p.bufferTurnID = turnID
		p.queuedAt = time.Now()
	}
	p.buf = append(p.buf, up...)
	if len(p.buf) > p.maxSamples {
		drop := len(p.buf) - p.maxSamples
		p.buf = append(p.buf[:0], p.buf[drop:]...)
		mPlayoutDropped.Add(float64(drop) / playoutFrameSamp)
	}
	p.mu.Unlock()
}

// SetTap attaches (or, with nil, detaches) the avatar's view of the timeline.
func (p *playout) SetTap(tap playoutTap) {
	p.mu.Lock()
	p.tap = tap
	p.mu.Unlock()
}

// SetDelay holds every subsequent tick back by d (rounded up to whole ticks)
// before it is emitted. Meant to be called once, when an avatar becomes ready;
// audio already in flight is simply heard d later.
func (p *playout) SetDelay(d time.Duration) {
	ticks := int((d + playoutFrameMS*time.Millisecond - 1) / (playoutFrameMS * time.Millisecond))
	p.mu.Lock()
	defer p.mu.Unlock()
	if ticks <= 0 || len(p.delay) > 0 {
		return
	}
	p.delay = make([]delaySlot, ticks)
	for i := range p.delay {
		p.delay[i].pcm = make([]int16, playoutFrameSamp)
	}
	p.dueFrame = make([]int16, playoutFrameSamp)
}

// schedule passes this tick's freshly stepped frame through the tap and the
// delay line and returns the tick that is due on the wire now. With no delay
// configured that is the same tick it was given.
func (p *playout) schedule(frame []int16, emit bool, end *turnEnd, now time.Time) tick {
	p.mu.Lock()
	defer p.mu.Unlock()
	cur := tick{end: end}
	if emit {
		cur.frame = frame
		cur.start = p.pendingStart
		p.pendingStart = nil
	}
	if p.tap != nil {
		p.tap.Tick(cur.frame, now)
	}
	if len(p.delay) == 0 {
		return cur
	}
	slot := &p.delay[p.delayHead]
	due := tick{start: slot.start, end: slot.end}
	if slot.live {
		copy(p.dueFrame, slot.pcm)
		due.frame = p.dueFrame
	}
	if due.start != nil {
		// PLAYBACK_STARTED reports how long the first audio waited at the
		// edge; the time spent in the delay line is part of that wait.
		due.start.queueMS += int64(len(p.delay) * playoutFrameMS)
	}
	slot.live = cur.frame != nil
	if slot.live {
		copy(slot.pcm, cur.frame)
	}
	slot.start, slot.end = cur.start, cur.end
	p.delayHead = (p.delayHead + 1) % len(p.delay)
	return due
}

// Flush is the local barge-in cut: empty the buffer and re-arm the prebuffer so
// the next turn starts clean. Instant and lock-cheap — this is why playback
// lives at the edge, not the browser.
//
// A pending drain marker is completed as "interrupted" rather than dropped: the
// audio is gone, so the turn is over for the listener either way, and a silently
// dropped marker would strand the client waiting for a PLAYBACK_COMPLETE.
func (p *playout) Flush() {
	p.mu.Lock()
	p.buf = p.buf[:0]
	p.playing = false
	end := p.takePendingEndLocked()
	// Audio parked in the lip-sync delay line has not been heard yet either.
	// Cut it too, and complete any turn whose end marker was riding with it.
	var parked []*turnEnd
	for i := range p.delay {
		slot := &p.delay[i]
		if slot.end != nil {
			parked = append(parked, slot.end)
		}
		slot.live, slot.start, slot.end = false, nil, nil
	}
	p.mu.Unlock()
	for _, e := range parked {
		p.fire(e, "interrupted")
	}
	p.fire(end, "interrupted")
}

// MarkTurnEnd registers the drain marker for turnID. If nothing is buffered the
// completion fires immediately — the common case for a turn that produced no
// audio at all (LLM error, empty reply).
func (p *playout) MarkTurnEnd(sessionID string, turnID int64, reason string) {
	p.mu.Lock()
	// With a delay line the turn's tail may still be parked in it even though
	// buf is empty, so the marker has to travel through step and the line too.
	if len(p.buf) == 0 && len(p.delay) == 0 {
		p.mu.Unlock()
		p.fire(&turnEnd{sessionID: sessionID, turnID: turnID, reason: reason}, "")
		return
	}
	// A second marker before the first drained (turns are sequential, so this
	// means the earlier one is already moot) replaces it rather than queueing.
	p.pendingEnd = &turnEnd{sessionID: sessionID, turnID: turnID, reason: reason}
	p.mu.Unlock()
}

// takePendingEndLocked clears and returns the marker. Caller must hold p.mu.
func (p *playout) takePendingEndLocked() *turnEnd {
	end := p.pendingEnd
	p.pendingEnd = nil
	return end
}

// fire invokes the completion callback outside p.mu. override, when non-empty,
// replaces the reason recorded at MarkTurnEnd time (a cut beats a clean drain).
func (p *playout) fire(end *turnEnd, override string) {
	if end == nil || p.onTurnComplete == nil {
		return
	}
	reason := end.reason
	if override != "" {
		reason = override
	}
	p.onTurnComplete(end.sessionID, end.turnID, reason)
}

// Close stops the ticker goroutine. Safe to call once.
func (p *playout) Close() {
	select {
	case <-p.stop:
	default:
		close(p.stop)
	}
	<-p.done
}

func (p *playout) run() {
	defer close(p.done)
	ticker := time.NewTicker(playoutFrameMS * time.Millisecond)
	defer ticker.Stop()
	frame := make([]int16, playoutFrameSamp)
	for {
		select {
		case <-p.stop:
			return
		case <-ticker.C:
			now := time.Now()
			emit, end := p.step(frame, now)
			due := p.schedule(frame, emit, end, now)
			if due.frame != nil && !p.emitFrame(due.frame) {
				return // track closed / peer gone
			}
			if due.start != nil {
				p.mu.Lock()
				callback := p.onTurnStart
				p.mu.Unlock()
				if callback != nil {
					callback(due.start.sessionID, due.start.turnID, due.start.queueMS)
				}
			}
			p.fire(due.end, "")
		}
	}
}

// step advances the buffer by one 20 ms frame under the lock and reports what
// the caller owes the wire: whether frame holds audio to emit, and whether a
// turn's drain marker has just come due. Split out from run so the buffering and
// end-of-turn rules are testable without libopus or a live track.
func (p *playout) step(frame []int16, now time.Time) (emit bool, end *turnEnd) {
	p.mu.Lock()
	defer p.mu.Unlock()
	// A marked turn is finished producing, so the cushion will never fill again
	// and a short buffer is not "wait for more" — it is the tail. Drain it
	// (zero-padding the last partial frame) and complete the marker.
	ending := p.pendingEnd != nil
	forced := false
	if !p.playing {
		if len(p.buf) == 0 {
			p.stallSince = time.Time{} // nothing held back; not a stall
			if !ending {
				return false, nil
			}
		}
		if len(p.buf) < p.startSamples && !ending {
			// Hold for the cushion, but only for so long. The clause that is
			// buffered may be the entire rest of the sentence, in which case the
			// cushion can never fill and holding means never playing it.
			if p.stallSince.IsZero() {
				p.stallSince = now
			}
			if now.Sub(p.stallSince) < p.maxStall {
				return false, nil // still filling the cushion
			}
			forced = true
			mPlayoutStalls.Inc()
		}
		p.playing = true
		if p.bufferTurnID > 0 && !p.startedTurns[p.bufferTurnID] {
			queueMS := int64(0)
			if !p.queuedAt.IsZero() {
				queueMS = time.Since(p.queuedAt).Milliseconds()
				if queueMS < 0 {
					queueMS = 0
				}
			}
			p.startedTurns[p.bufferTurnID] = true
			p.pendingStart = &playbackStart{
				sessionID: p.bufferSessionID,
				turnID:    p.bufferTurnID,
				queueMS:   queueMS,
			}
		}
		p.stallSince = time.Time{}
	}
	if len(p.buf) < playoutFrameSamp {
		if ending || forced {
			n := copy(frame, p.buf)
			for i := n; i < playoutFrameSamp; i++ {
				frame[i] = 0
			}
			p.buf = p.buf[:0]
			p.playing = false
			// A marker with nothing left to play emits no frame at all, so a
			// turn that produced no audio does not inject 20 ms of silence.
			return n > 0, p.takePendingEndLocked()
		}
		// Underrun: rebuild the cushion before resuming instead of emitting a
		// lone frame followed by silence (choppy).
		if p.playing {
			mPlayoutUnderruns.Inc()
		}
		p.playing = false
		return false, nil
	}
	copy(frame, p.buf[:playoutFrameSamp])
	p.buf = append(p.buf[:0], p.buf[playoutFrameSamp:]...)
	return true, nil
}

// emitFrame encodes and writes one 20 ms frame. false means the track is gone.
func (p *playout) emitFrame(frame []int16) bool {
	n, err := p.enc.Encode(frame, p.encScratch)
	if err != nil {
		mPlayoutEncodeErrors.Inc()
		return true
	}
	if err := p.track.WriteSample(media.Sample{
		Data:     append([]byte(nil), p.encScratch[:n]...),
		Duration: playoutFrameMS * time.Millisecond,
	}); err != nil {
		return false
	}
	mPlayoutFrames.Inc()
	return true
}

// resampleTo48k linearly resamples mono PCM16 to 48 kHz. Each chunk is treated
// standalone; for the common 24 kHz TTS rate this is an exact 2x upsample and
// the per-chunk boundary error is inaudible for speech.
func resampleTo48k(in []int16, inRate int) []int16 {
	if inRate <= 0 || inRate == playoutRate {
		return in
	}
	ratio := float64(playoutRate) / float64(inRate)
	outLen := int(float64(len(in)) * ratio)
	if outLen <= 0 {
		return nil
	}
	out := make([]int16, outLen)
	for i := range out {
		src := float64(i) / ratio
		idx := int(src)
		frac := src - float64(idx)
		s0 := float64(in[idx])
		s1 := s0
		if idx+1 < len(in) {
			s1 = float64(in[idx+1])
		}
		out[i] = int16(s0*(1-frac) + s1*frac)
	}
	return out
}

// bytesToInt16 reinterprets little-endian PCM16 bytes as samples (drops a
// trailing odd byte defensively).
func bytesToInt16(b []byte) []int16 {
	n := len(b) / 2
	out := make([]int16, n)
	for i := 0; i < n; i++ {
		out[i] = int16(b[2*i]) | int16(b[2*i+1])<<8
	}
	return out
}
