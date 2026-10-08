package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/pion/interceptor"
	"github.com/pion/opus"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	"mondee/media-edge/internal/edgepb"
	"mondee/media-edge/internal/profile"
	"mondee/media-edge/internal/silero"
)

const pcmSampleRate = 16000

// pcmFrameCap bounds the per-packet decode buffer: 120 ms (Opus max) at 16 kHz
// mono = 1920 samples. Any real client at ptime 10 ms sends 160.
const pcmFrameCap = 1920

const (
	rtpReadPollInterval = 100 * time.Millisecond
	finalReplyWait      = 12 * time.Second
)

// vad is a per-session energy gate with a silence hangover. It lives entirely
// inside one forwardTrack goroutine, so there is no cross-session state to
// guard — N concurrent callers each carry their own, scaling across cores the
// same way the Opus decode does. That locality is the whole point of doing VAD
// at the edge rather than on the orchestrator's shared event loop.
//
// The metric is either an RMS energy gate (default, cheap, no model) or the
// Silero neural VAD (when profile.VADModelPath is set). Silero tells speech from
// speech-level noise (music, other voices, fan hum) that energy cannot. Either
// way the same Schmitt trigger with debounce (enter/exit hysteresis + attack +
// hangover) runs in step(), which is what stops it flapping on room noise. See
// profile.Profile for the knobs.
//
// ponytail: one Silero session per caller (the model is stateful and not
// thread-safe, so it can't be shared without serializing inference). A tiny VAD
// model x N callers is fine for the beta; pool the sessions if 50-way memory bites.
type vad struct {
	speaking   bool
	attack     time.Duration // sustained enter before "started" fires
	hangover   time.Duration // sustained silence before "stopped" fires
	voiceStart time.Time     // when the current above-enter run began (zero: none)
	lastVoice  time.Time     // last above-exit frame while speaking
	lastLevel  float64       // most recent metric reading: Silero prob (0..1) or RMS

	// energy path (sil == nil)
	enter float64 // RMS to begin counting toward speech
	exit  float64 // RMS below which we count toward silence (< enter)

	// silero path (predict != nil)
	sil         *silero.Model
	predict     func([]float32) (float32, error) // = sil.Predict; a seam for tests
	acc         []float32                        // leftover samples awaiting a full 512-sample window
	speechProb  float64                          // prob to begin counting toward speech
	silenceProb float64                          // prob below which we count toward silence (< speechProb)
	processNS   int64                            // cumulative update cost for this session
	// processNS as of the last control event, so vadMeta can report the
	// per-event delta rather than the running total.
	reportedProcessMS int64
	windows           int64 // VAD update calls (not audio packets logged individually)
}

// sileroWindow is Silero's required chunk at 16 kHz (32 ms). The model manages
// its own 64-sample lookback context internally, so we feed exactly this many.
const sileroWindow = 512

func newVAD(p profile.Profile) (*vad, error) {
	v := &vad{
		attack:   time.Duration(p.VADAttackMs) * time.Millisecond,
		hangover: time.Duration(p.VADHangoverMs) * time.Millisecond,
		enter:    p.VADEnterRms,
		exit:     p.VADExitRms,
	}
	if p.VADModelPath != "" {
		m, err := silero.NewModel(16000, p.VADModelPath)
		if err != nil {
			return nil, fmt.Errorf("load silero model %q: %w", p.VADModelPath, err)
		}
		m.Reset()
		v.sil = m
		v.predict = m.Predict
		v.speechProb = p.VADSpeechProb
		v.silenceProb = p.VADSilenceProb
	}
	return v, nil
}

// close releases the Silero session (no-op for the energy gate).
func (v *vad) close() {
	if v.sil != nil {
		v.sil.Destroy()
	}
}

// update feeds one decoded frame and reports edge transitions: started once the
// metric has held above the enter line for the attack window, stopped once it
// has held below the exit line for the hangover window. At most one is true per
// call in practice (a 10 ms/160-sample frame yields at most one Silero window).
func (v *vad) update(pcm []int16, now time.Time) (started, stopped bool) {
	defer func(t time.Time) { v.processNS += time.Since(t).Nanoseconds(); v.windows++ }(time.Now())
	if v.predict == nil {
		r := rms16(pcm)
		v.lastLevel = r
		return v.step(r >= v.enter, r >= v.exit, now)
	}
	// Silero needs whole 512-sample windows; accumulate across frames and run
	// the state machine once per full window.
	for _, s := range pcm {
		v.acc = append(v.acc, float32(s)/32768)
	}
	consumed := 0
	for len(v.acc)-consumed >= sileroWindow {
		win := v.acc[consumed : consumed+sileroWindow]
		consumed += sileroWindow
		p, err := v.predict(win)
		if err != nil {
			mVADErrors.Inc()
			warnf("VAD: silero inference error: %v", err)
			continue // drop a bad window rather than kill the session
		}
		v.lastLevel = float64(p)
		if st, sp := v.step(float64(p) >= v.speechProb, float64(p) >= v.silenceProb, now); st {
			started = true
		} else if sp {
			stopped = true
		}
	}
	if consumed > 0 { // compact leftover to the front so acc stays bounded
		v.acc = v.acc[:copy(v.acc, v.acc[consumed:])]
	}
	return started, stopped
}

// flush pads and processes any partial Silero window at the end of a stream.
// Without this, short final utterance tails never reach the endpointer.
func (v *vad) flush(now time.Time) (started, stopped bool) {
	if v.predict == nil || len(v.acc) == 0 {
		return false, false
	}
	pcm := make([]int16, sileroWindow-len(v.acc))
	return v.update(pcm, now)
}

// step advances the Schmitt-trigger state machine from one metric reading:
// aboveEnter means the level crossed the (higher) speech line, aboveExit the
// (lower) silence line. It is identical for the energy and Silero paths.
func (v *vad) step(aboveEnter, aboveExit bool, now time.Time) (started, stopped bool) {
	if !v.speaking {
		if !aboveEnter {
			v.voiceStart = time.Time{} // spike didn't sustain; reset the attack run
			return false, false
		}
		if v.voiceStart.IsZero() {
			v.voiceStart = now
		}
		if now.Sub(v.voiceStart) >= v.attack {
			v.speaking = true
			v.lastVoice = now
			v.voiceStart = time.Time{}
			return true, false
		}
		return false, false
	}
	// speaking: hold until level stays below the (lower) exit line past hangover
	if aboveExit {
		v.lastVoice = now
		return false, false
	}
	if now.Sub(v.lastVoice) >= v.hangover {
		v.speaking = false
		return false, true
	}
	return false, false
}

// rms16 is the root-mean-square amplitude of int16 PCM samples, 0 for empty.
func rms16(s []int16) float64 {
	if len(s) == 0 {
		return 0
	}
	var sum float64
	for _, x := range s {
		f := float64(x)
		sum += f * f
	}
	return math.Sqrt(sum / float64(len(s)))
}

// dialOrchestrator opens one shared, long-lived gRPC connection to the Python
// orchestrator on localhost. All sessions multiplex their own Stream RPCs over
// it (HTTP/2 streams are cheap; the TCP/TLS handshake is paid once).
func dialOrchestrator(addr string) (*grpc.ClientConn, error) {
	return grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()), // localhost, same host
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Second,
			Timeout:             3 * time.Second,
			PermitWithoutStream: true,
		}),
	)
}

// forwardTrack decodes an inbound Opus track to 16 kHz mono PCM16 and streams
// it to the orchestrator as AudioFrames, bracketed by SESSION_OPEN/CLOSE
// control events. It returns the session id and total PCM bytes forwarded once
// the track ends (client "stop" or dropped connection), always sending
// SESSION_CLOSE so the orchestrator can finalize its side.
func forwardTrack(ctx context.Context, conn *grpc.ClientConn, track *webrtc.TrackRemote, prof profile.Profile,
	sink playoutSink, onOpen func(string), onReply func(*edgepb.AgentReply), onComplete func(string),
	onTurnEvent func(*edgepb.Control)) (sessionID string, bytesFwd int64, err error) {
	sessionID = uuid.NewString()
	if onOpen != nil {
		onOpen(sessionID)
	}
	mSessionsTotal.Inc()
	mActiveSessions.Inc()
	defer mActiveSessions.Dec()
	t0 := time.Now()
	defer func() { mSessionDuration.Observe(time.Since(t0).Seconds()) }()

	v, err := newVAD(prof)
	if err != nil {
		return sessionID, 0, fmt.Errorf("init vad: %w", err)
	}
	defer v.close()

	dec, err := opus.NewDecoderWithOutput(16000, 1)
	if err != nil {
		return sessionID, 0, fmt.Errorf("opus decoder: %w", err)
	}

	// Check connection state before opening a new stream — fail fast if the
	// orchestrator connection is dead instead of hanging on the first Send.
	if state := conn.GetState(); state == connectivity.Shutdown {
		return sessionID, 0, fmt.Errorf("orchestrator connection is shut down")
	}

	// `ctx` (trackCtx) is also how the WS "stop" handler breaks the RTP
	// read-poll loop below (via ctx.Err() once cancelled) -- but that same
	// cancellation, if it also tore down the gRPC stream's context, killed
	// the still-draining response side out from under a call that had
	// already finished sending audio and was just waiting (per runBridge's
	// own half-close design, bounded by finalReplyWait) for the final reply.
	// Observed directly: stream_wav.py's normal "stop" message after the
	// last queued turn raced this and turned an already-successful turn
	// into a reported "context canceled" failure. The stream gets an
	// independent context -- same deadline/value inheritance, but immune to
	// trackCtx's cancellation -- so ending the read loop and ending the
	// response wait are no longer the same event.
	stream, err := edgepb.NewMediaBridgeClient(conn).Stream(context.WithoutCancel(ctx))
	if err != nil {
		return sessionID, 0, fmt.Errorf("open bridge stream: %w", err)
	}

	bytesFwd, err = runBridge(ctx, stream, track, dec.DecodeToInt16, v, prof, sessionID, sink, onReply, onComplete, onTurnEvent)
	return sessionID, bytesFwd, err
}

// rtpReader is the subset of *webrtc.TrackRemote runBridge consumes; a seam so a
// scripted track can drive the read loop in tests without a live peer.
type rtpReader interface {
	ReadRTP() (*rtp.Packet, interceptor.Attributes, error)
	SetReadDeadline(time.Time) error
}

// bridgeClientStream is the subset of the gRPC bidi stream runBridge uses. The
// real edgepb.MediaBridge_StreamClient satisfies it; a mock stands in for tests.
type bridgeClientStream interface {
	Send(*edgepb.ClientEvent) error
	Recv() (*edgepb.ServerEvent, error)
	CloseSend() error
}

// micTap is implemented by sinks that also want the decoded microphone audio:
// the avatar uses it as its listening track. Optional, so a plain playout (or a
// test's recording sink) is still a complete playoutSink.
type micTap interface {
	PushMic(pcm []int16) // 16 kHz mono; the slice is reused, copy to keep it
}

// runBridge is the decode/VAD/forward loop, split out from forwardTrack so its
// timeout, endpointing, flush and half-close behavior is testable behind the
// rtpReader/bridgeClientStream/decode seams. decode maps an RTP payload to PCM16
// (dec.DecodeToInt16 in production).
func runBridge(ctx context.Context, stream bridgeClientStream, track rtpReader,
	decode func(payload []byte, out []int16) (int, error), v *vad, prof profile.Profile, sessionID string,
	sink playoutSink, onReply func(*edgepb.AgentReply), onComplete func(string),
	onTurnEvent func(*edgepb.Control)) (bytesFwd int64, err error) {
	mediaCursor := newMediaClock()
	mic, _ := sink.(micTap)
	meta := make(map[string]string, 14) // reused for vadMeta each control event
	var sendMu sync.Mutex
	var sendClosed bool // guards stream.Send after stream.CloseSend (see call site below)
	var endpointMu sync.Mutex
	endpointWall := make(map[int64]time.Time)
	var edgeTurnID int64
	sendEvent := func(event *edgepb.ClientEvent) error {
		sendMu.Lock()
		defer sendMu.Unlock()
		if sendClosed {
			// The send side is already half-closed (track ended before this
			// async reply/playback event arrived). The orchestrator already
			// has everything it needs from us; silently drop rather than
			// calling Send after CloseSend, which grpc-go treats as fatal
			// ("SendMsg called after CloseSend") instead of a normal error.
			return nil
		}
		return stream.Send(event)
	}
	sendControlSafe := func(sessionID string, t edgepb.Control_Type, ts time.Time, metadata map[string]string) error {
		sendMu.Lock()
		defer sendMu.Unlock()
		if sendClosed {
			return nil
		}
		return sendControl(stream, sessionID, t, ts, metadata)
	}
	if sink != nil {
		sink.SetPlaybackStarted(func(startSessionID string, turnID int64, queueMS int64) {
			metadata := map[string]string(nil)
			endpointMu.Lock()
			endpointAt, ok := endpointWall[turnID]
			if ok {
				delete(endpointWall, turnID)
				elapsed := nowFn().Sub(endpointAt).Milliseconds()
				if elapsed >= 0 {
					metadata = map[string]string{
						"vad_endpoint_to_first_audio_ms": strconv.FormatInt(elapsed, 10),
					}
				}
			}
			endpointMu.Unlock()
			control := controlAt(startSessionID, edgepb.Control_PLAYBACK_STARTED, nowFn(), metadata)
			control.GetControl().TurnId = turnID
			control.GetControl().ProcMs = queueMS
			_ = sendEvent(control)
		})
	}
	if err := sendControlSafe(sessionID, edgepb.Control_SESSION_OPEN, mediaCursor.Now(), vadMeta(meta, v, prof, 0, 0, 0, 0)); err != nil {
		return 0, fmt.Errorf("session_open: %w", err)
	}

	// The same bidi RPC carries agent replies back while audio still flows up. Do
	// not wait until the track ends: a normal conversational turn gets its reply
	// while the browser keeps the WebRTC call open.
	recvDone := make(chan error, 1)
	go func() {
		for {
			ev, e := stream.Recv()
			if e != nil {
				recvDone <- e
				return
			}
			if reply := ev.GetReply(); reply != nil && onReply != nil {
				onReply(reply)
			}
			if audio := ev.GetAudio(); audio != nil && len(audio.Pcm16) > 0 && sink != nil {
				sink.Write(audio.SessionId, audio.TurnId, audio.Pcm16, int(audio.SampleRateHz))
			}
			if c := ev.GetControl(); c != nil {
				switch c.Type {
				// Orchestrator-initiated stop (its interrupt controller): drop any
				// buffered agent audio. The edge also cuts locally on its own VAD
				// below, so this is the belt to that suspenders.
				case edgepb.Control_STOP_PLAYBACK:
					if sink != nil {
						sink.Flush()
					}
				case edgepb.Control_AGENT_TURN_STARTED:
					// Pure passthrough to the client's turn state machine.
					if onTurnEvent != nil {
						onTurnEvent(c)
					}
				case edgepb.Control_AGENT_TURN_ENDED:
					// NOT forwarded as-is: it means "no more audio is coming", which
					// is up to a prebuffer earlier than "the user has heard it all".
					// Hand it to the playout buffer, which reports PLAYBACK_COMPLETE
					// once the tail has actually gone out on the wire.
					reason := c.Metadata["reason"]
					if sink == nil {
						if onTurnEvent != nil {
							onTurnEvent(playbackComplete(c.SessionId, c.TurnId, reason))
						}
						break
					}
					sink.MarkTurnEnd(c.SessionId, c.TurnId, reason)
				}
			}
		}
	}()

	// Prints a throttled live status line so you can see the metric react as you
	// talk — the quickest way to confirm the VAD is alive and to tune
	// vad_speech_prob/vad_silence_prob against real levels. Off by default; set
	// VAD_DEBUG=1 when tuning.
	vadDebug := os.Getenv("VAD_DEBUG") == "1" || strings.EqualFold(os.Getenv("VAD_DEBUG"), "true")
	var lastStatus time.Time

	pcm := make([]int16, pcmFrameCap)
	leBuf := make([]byte, pcmFrameCap*2) // reused for int16sToLE each packet
	var sendErr error
	var decodeNS int64
	var decodeErrors int64
	var decodedFrames int64
	var speechStart time.Time
	for {
		if err := track.SetReadDeadline(nowFn().Add(rtpReadPollInterval)); err != nil {
			warnf("[%s] SetReadDeadline failed: %v", sessionID, err)
		}
		pkt, _, readErr := track.ReadRTP()
		if readErr != nil {
			if isTimeout(readErr) {
				if ctx.Err() != nil {
					break
				}
				// Drive the VAD state machine forward even when no packets
				// arrive, so attack/hangover timers don't stall during idle.
				//
				// The fill is sized by the wall time actually elapsed since
				// the cursor last moved, NOT by a fixed rtpReadPollInterval's
				// worth. A fixed fill double-counts: a read deadline that
				// expires shortly after a packet was consumed adds a full
				// 100 ms of synthetic silence on top of that packet's own
				// duration, so the cursor gains media time that no wall time
				// backs. Repeated over a call it made the cursor -- and every
				// control timestamp stamped from it -- run measurably ahead of
				// wall clock (up to 7% over a 41 s session), and it stretched
				// the VAD's attack/hangover windows by the same factor.
				silence := make([]int16, mediaCursor.PendingSamples(nowFn()))
				if len(silence) == 0 {
					continue // no media time to account for yet
				}
				now := mediaCursor.Advance(len(silence))
				if started, stopped := v.update(silence, now); started {
					edgeTurnID++
					mSpeechStarted.Inc()
					speechStart = now
					if sink != nil {
						sink.Flush() // local barge-in cut: kill agent audio before the network round-trip
					}
					eventf("[%s] speech started", sessionID)
					if err := sendControlSafe(sessionID, edgepb.Control_USER_SPEECH_STARTED, now, vadMeta(meta, v, prof, 0, decodeNS, decodedFrames, decodeErrors)); err != nil {
						sendErr = err
						break
					}
				} else if stopped {
					endpointMu.Lock()
					endpointWall[edgeTurnID] = nowFn()
					endpointMu.Unlock()
					mTurns.Inc()
					eventf("[%s] speech ended", sessionID)
					speechMS := int64(0)
					if !speechStart.IsZero() {
						speechMS = now.Sub(speechStart).Milliseconds()
					}
					if err := sendControlSafe(sessionID, edgepb.Control_END_TURN, now, vadMeta(meta, v, prof, speechMS, decodeNS, decodedFrames, decodeErrors)); err != nil {
						sendErr = err
						break
					}
				}
				continue
			}
			break // track ended (stop / disconnect)
		}
		if len(pkt.Payload) == 0 {
			continue // empty/keepalive
		}
		decodeAt := time.Now()
		n, decErr := decode(pkt.Payload, pcm)
		decodeNS += time.Since(decodeAt).Nanoseconds()
		if decErr != nil {
			// One bad frame shouldn't kill the session; skip it.
			mOpusErrors.Inc()
			decodeErrors++
			continue
		}
		decodedFrames++
		if mic != nil {
			mic.PushMic(pcm[:n])
		}
		now := mediaCursor.Advance(n)
		if started, stopped := v.update(pcm[:n], now); started {
			edgeTurnID++
			mSpeechStarted.Inc()
			speechStart = now
			if sink != nil {
				sink.Flush() // local barge-in cut: kill agent audio before the network round-trip
			}
			eventf("[%s] speech started", sessionID)
			if err := sendControlSafe(sessionID, edgepb.Control_USER_SPEECH_STARTED, now,
				vadMeta(meta, v, prof, 0, decodeNS, decodedFrames, decodeErrors)); err != nil {
				sendErr = err
				break
			}
		} else if stopped {
			endpointMu.Lock()
			endpointWall[edgeTurnID] = nowFn()
			endpointMu.Unlock()
			mTurns.Inc()
			eventf("[%s] speech ended", sessionID)
			speechMS := int64(0)
			if !speechStart.IsZero() {
				speechMS = now.Sub(speechStart).Milliseconds()
			}
			if err := sendControlSafe(sessionID, edgepb.Control_END_TURN, now,
				vadMeta(meta, v, prof, speechMS, decodeNS, decodedFrames, decodeErrors)); err != nil {
				sendErr = err
				break
			}
		}
		if vadDebug && now.Sub(lastStatus) >= 300*time.Millisecond {
			infof("[%s] vad speaking=%-5v level=%.3f", sessionID, v.speaking, v.lastLevel)
			lastStatus = now
		}
		buf := int16sToLEInto(pcm[:n], leBuf)
		if sendErr = sendEvent(&edgepb.ClientEvent{Msg: &edgepb.ClientEvent_Audio{Audio: &edgepb.AudioFrame{
			SessionId: sessionID,
			Pcm16:     buf,
			TsMs:      now.UnixMilli(),
		}}}); sendErr != nil {
			mGRPCSendErrors.Inc()
			break
		}
		bytesFwd += int64(len(buf))
	}

	if sendErr == nil {
		if started, stopped := v.flush(mediaCursor.Now()); started {
			sendErr = sendControlSafe(sessionID, edgepb.Control_USER_SPEECH_STARTED, mediaCursor.Now(), vadMeta(meta, v, prof, 0, decodeNS, decodedFrames, decodeErrors))
		} else if stopped {
			sendErr = sendControlSafe(sessionID, edgepb.Control_END_TURN, mediaCursor.Now(), vadMeta(meta, v, prof, 0, decodeNS, decodedFrames, decodeErrors))
		}
	}
	if sendErr == nil {
		if closeErr := sendControlSafe(sessionID, edgepb.Control_SESSION_CLOSE, mediaCursor.Now(),
			vadMeta(meta, v, prof, 0, decodeNS, decodedFrames, decodeErrors)); closeErr != nil && !errors.Is(closeErr, io.EOF) {
			// This is the client's own courtesy notice that its input is
			// done -- by the time it's sent, the orchestrator may have
			// already finished the turn and closed its receive side first
			// (a benign race between two sides both trying to end the call
			// gracefully). io.EOF here means "nothing left to tell it,"
			// not a failure: everything up to and including the turn's
			// reply already succeeded. Any other Send error still counts.
			sendErr = closeErr
		}
	}
	sendMu.Lock()
	sendClosed = true
	_ = stream.CloseSend()
	sendMu.Unlock()
	// Half-close leaves the response direction alive, allowing the orchestrator
	// to drain STT and send the last LLM reply before it closes its side.
	var recvErr error
	select {
	case recvErr = <-recvDone:
	case <-time.After(finalReplyWait):
		recvErr = fmt.Errorf("timed out waiting %s for final orchestrator response", finalReplyWait)
	}
	if onComplete != nil {
		onComplete(sessionID)
	}
	if sendErr != nil {
		return bytesFwd, sendErr
	}
	if recvErr != nil && recvErr != io.EOF {
		return bytesFwd, fmt.Errorf("receive orchestrator response: %w", recvErr)
	}
	return bytesFwd, nil
}

// playbackComplete builds the edge-authored control the client actually gates
// its mic and "agent speaking" indicator on.
func playbackComplete(sessionID string, turnID int64, reason string) *edgepb.Control {
	if reason == "" {
		reason = "complete"
	}
	return &edgepb.Control{
		SessionId: sessionID,
		Type:      edgepb.Control_PLAYBACK_COMPLETE,
		TsMs:      nowFn().UnixMilli(),
		TurnId:    turnID,
		Metadata:  map[string]string{"reason": reason},
	}
}

// controlAt builds a lifecycle control. `ts` is the MEDIA cursor position the
// event was detected at; it travels as the media_ts_ms metadata key.
//
// TsMs is wall clock, not the cursor. The orchestrator subtracts control
// timestamps from its own wall clock to derive caller-felt latency, so a
// cursor value there produced a cross-clock subtraction: negative differences
// clamped to a fabricated 0 ms, and a session-relative cursor from an older
// build landed events in 1970. Keeping the two timebases in separate,
// differently-named fields is what stops them being confused again.
func controlAt(sessionID string, t edgepb.Control_Type, ts time.Time, metadata map[string]string) *edgepb.ClientEvent {
	if metadata != nil {
		metadata["media_ts_ms"] = strconv.FormatInt(ts.UnixMilli(), 10)
	}
	return &edgepb.ClientEvent{Msg: &edgepb.ClientEvent_Control{Control: &edgepb.Control{
		SessionId: sessionID,
		Type:      t,
		TsMs:      nowFn().UnixMilli(),
		Metadata:  metadata,
		ProcMs:    metadataInt(metadata, "vad_process_ms"),
	}}}
}

type bridgeStream interface {
	Send(*edgepb.ClientEvent) error
}

func sendControl(stream bridgeStream, sessionID string, t edgepb.Control_Type, ts time.Time, metadata map[string]string) error {
	if err := stream.Send(controlAt(sessionID, t, ts, metadata)); err != nil {
		mGRPCSendErrors.Inc()
		return fmt.Errorf("%s: %w", t.String(), err)
	}
	return nil
}

// nowFn is the wall clock, seamed so tests can drive the read-deadline/idle
// path deterministically instead of sleeping through real poll intervals.
// Production always uses time.Now.
var nowFn = time.Now

// mediaClock is a sample-counting cursor over the received audio. It is the
// correct timebase for the VAD -- attack and hangover are durations of MEDIA,
// so they must not be measured against a wall clock that keeps running while
// packets are absent.
//
// It is deliberately NOT a wall clock and must never be published as one: it
// only advances when audio is accounted for, so it lags a stalled stream and
// (before PendingSamples existed) could run ahead of one. Control events carry
// wall time in TsMs; the cursor's offset travels beside it as metadata.
type mediaClock struct {
	samples int64
	start   time.Time
}

func newMediaClock() *mediaClock {
	return &mediaClock{start: nowFn()}
}

func (c *mediaClock) Now() time.Time {
	return c.start.Add(c.Elapsed())
}

// Elapsed is the media time accounted for so far.
func (c *mediaClock) Elapsed() time.Duration {
	return time.Duration(c.samples) * time.Second / pcmSampleRate
}

// PendingSamples is how many samples of silence the cursor would need to catch
// up to wall time at `now`. It is zero when the cursor is already level with or
// ahead of the wall (decoded packets having already accounted for that span),
// which is what keeps an idle fill from double-counting a consumed packet.
// The result is capped at the decode buffer so one long stall cannot allocate
// an unbounded slice or hand the VAD a single enormous frame.
func (c *mediaClock) PendingSamples(now time.Time) int {
	behind := now.Sub(c.start) - c.Elapsed()
	if behind <= 0 {
		return 0
	}
	samples := int(behind * pcmSampleRate / time.Second)
	if samples > pcmFrameCap {
		samples = pcmFrameCap
	}
	return samples
}

func (c *mediaClock) Advance(samples int) time.Time {
	c.samples += int64(samples)
	return c.Now()
}

func isTimeout(err error) bool {
	type timeout interface {
		Timeout() bool
	}
	t, ok := err.(timeout)
	return ok && t.Timeout()
}

func metadataInt(m map[string]string, key string) int64 {
	n, _ := strconv.ParseInt(m[key], 10, 64)
	return n
}

// vadMeta writes into a pre-allocated map to avoid per-control-event allocs.
func vadMeta(dst map[string]string, v *vad, p profile.Profile, speechMS, decodeNS, frames, decodeErrors int64) map[string]string {
	engine, model := "energy-rms", ""
	if v.sil != nil {
		engine, model = "silero-onnxruntime", p.VADModelPath
	}
	dst["vad_engine"] = engine
	dst["vad_model"] = model
	// Per-event, not cumulative: the orchestrator stores this in duration_ms,
	// which every other event type fills with the duration of THAT event.
	// Publishing the session counter here made VAD duration_ms climb
	// monotonically across a call, so any average or quantile over it
	// described how late in the call the event happened, not how long the
	// VAD took. The running total stays available under its own name.
	total := v.processNS / 1_000_000
	dst["vad_process_ms"] = strconv.FormatInt(total-v.reportedProcessMS, 10)
	dst["vad_process_total_ms"] = strconv.FormatInt(total, 10)
	v.reportedProcessMS = total
	dst["vad_windows"] = strconv.FormatInt(v.windows, 10)
	dst["vad_last_level"] = fmt.Sprintf("%.6f", v.lastLevel)
	dst["vad_attack_ms"] = strconv.Itoa(p.VADAttackMs)
	dst["vad_hangover_ms"] = strconv.Itoa(p.VADHangoverMs)
	dst["speech_duration_ms"] = strconv.FormatInt(speechMS, 10)
	dst["opus_decode_ms"] = strconv.FormatInt(decodeNS/1_000_000, 10)
	dst["opus_decoded_frames"] = strconv.FormatInt(frames, 10)
	dst["opus_decode_errors"] = strconv.FormatInt(decodeErrors, 10)
	return dst
}

// int16sToLEInto packs samples as little-endian PCM16 bytes into dst, which
// must be at least len(s)*2 bytes. Returns the filled portion of dst.
func int16sToLEInto(s []int16, dst []byte) []byte {
	for i, v := range s {
		dst[2*i] = byte(v)
		dst[2*i+1] = byte(v >> 8)
	}
	return dst[:len(s)*2]
}
