package main

import (
	"context"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/pion/rtcp"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	"mondee/media-edge/internal/avatarpb"
)

// The avatar plane: an optional talking-head video track rendered by a separate
// service (see avatar/README.md) from the two audio streams the edge already
// holds — what the agent is about to say, and what the user is saying.
//
// The edge is the one place both of those meet a single clock, so it owns the
// lip-sync. Every playout tick is one slice of a session timeline: the tick's
// agent audio and the microphone audio that arrived since the last tick go to
// the avatar service stamped with the slice's position, and video comes back
// stamped on the same timeline. The model has to hear ~400 ms past a frame
// before it can draw it, so the edge holds the agent audio back by a fixed
// delay (playout.SetDelay) and releases each video frame at
//
//	timeline position + delay
//
// which is exactly when the audio it depicts leaves the delay line. One
// constant, one clock, no drift between the two tracks.
//
// Everything here is best-effort. If the avatar service is absent, full, or
// dies mid-call, the session carries on as audio-only.
const (
	avatarRate     = 16000
	avatarTickSamp = avatarRate / 1000 * playoutFrameMS // 320 samples per 20 ms tick
	// Microphone audio waiting for a tick. More than this means the playout
	// clock stalled; old listening audio is not worth replaying late.
	avatarMicMaxSamp = avatarRate / 5
	avatarSendQueue  = 64
)

// Tunables (env). The service reports how much audio it needs past a frame
// (its lookahead); margin is what the edge adds on top for render, encode and
// transport. Too small and frames arrive after their slot — watch
// edge_avatar_late_frames_total — too large and every reply is simply later.
var (
	avatarID       = os.Getenv("EDGE_AVATAR_ID")
	avatarBG       = os.Getenv("EDGE_AVATAR_BG")
	avatarMarginMS = envInt("EDGE_AVATAR_RENDER_MARGIN_MS", 200, 0, 2000)
	// Non-zero overrides lookahead + margin outright.
	avatarDelayMS = envInt("EDGE_AVATAR_DELAY_MS", 0, 0, 5000)
)

// avatarVideoCodec is the one video codec the edge negotiates: H.264
// constrained baseline, which every WebRTC browser must decode and which the
// avatar service's encoder is pinned to.
var avatarVideoCodec = webrtc.RTPCodecCapability{
	MimeType:    webrtc.MimeTypeH264,
	ClockRate:   90000,
	SDPFmtpLine: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f",
}

// dialAvatar opens the shared connection to the avatar service. Lazy like
// dialOrchestrator: a service that is down fails the session's stream, not
// edge startup.
func dialAvatar(addr string) (*grpc.ClientConn, error) {
	return grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Second,
			Timeout:             3 * time.Second,
			PermitWithoutStream: true,
		}),
	)
}

// offerHasVideo reports whether the browser asked for a video track. Clients
// that did not (pipeline_test's WAV harness, an older page) stay audio-only.
func offerHasVideo(sdp string) bool {
	return strings.HasPrefix(sdp, "m=video ") || strings.Contains(sdp, "\nm=video ")
}

type avatarSession struct {
	sessionID string
	stream    avatarpb.Avatar_StreamClient
	video     *webrtc.TrackLocalStaticSample
	playout   *playout
	cancel    context.CancelFunc
	send      chan *avatarpb.ClientEvent

	mu         sync.Mutex
	mic        []int16   // 16 kHz microphone samples not yet paired with a tick
	pts        int64     // timeline position of the next tick, in 16 kHz samples
	anchor     time.Time // wall time of timeline position 0, as of the latest tick
	delay      time.Duration
	lastKeyReq time.Time
}

// startAvatar opens one avatar stream for a voice session and returns at once;
// the playout is only delayed and tapped when the service answers ready. nil
// means no avatar for this session.
func startAvatar(conn *grpc.ClientConn, sessionID string, video *webrtc.TrackLocalStaticSample, p *playout) *avatarSession {
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := avatarpb.NewAvatarClient(conn).Stream(ctx)
	if err == nil {
		err = stream.Send(&avatarpb.ClientEvent{Msg: &avatarpb.ClientEvent_Open{Open: &avatarpb.SessionOpen{
			SessionId: sessionID, AvatarId: avatarID, BackgroundId: avatarBG,
		}}})
	}
	if err != nil {
		cancel()
		mAvatarErrors.Inc()
		warnf("[%s] avatar unavailable, continuing audio-only: %v", sessionID, err)
		return nil
	}
	a := &avatarSession{
		sessionID: sessionID, stream: stream, video: video, playout: p, cancel: cancel,
		send: make(chan *avatarpb.ClientEvent, avatarSendQueue),
	}
	go a.sendLoop(ctx)
	go a.recvLoop(ctx)
	return a
}

// Close ends the stream. The playout keeps its delay: removing it mid-call
// would skip audio, and the peer is going away anyway.
func (a *avatarSession) Close() {
	a.playout.SetTap(nil)
	a.cancel()
}

func (a *avatarSession) sendLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case ev := <-a.send:
			if err := a.stream.Send(ev); err != nil {
				return // recvLoop reports the stream's real error
			}
		}
	}
}

// enqueue never blocks: its callers are the playout clock and the RTP read
// loop, and neither may wait on a slow avatar service.
func (a *avatarSession) enqueue(ev *avatarpb.ClientEvent) bool {
	select {
	case a.send <- ev:
		return true
	default:
		return false
	}
}

func (a *avatarSession) recvLoop(ctx context.Context) {
	defer a.playout.SetTap(nil)
	frameDur := 40 * time.Millisecond
	for {
		ev, err := a.stream.Recv()
		if err != nil {
			if ctx.Err() == nil {
				mAvatarErrors.Inc()
				warnf("[%s] avatar stream ended, continuing audio-only: %v", a.sessionID, err)
			}
			return
		}
		if ready := ev.GetReady(); ready != nil {
			delay := time.Duration(int(ready.LookaheadMs)+avatarMarginMS) * time.Millisecond
			if avatarDelayMS > 0 {
				delay = time.Duration(avatarDelayMS) * time.Millisecond
			}
			if ready.Fps > 0 {
				frameDur = time.Second / time.Duration(ready.Fps)
			}
			a.mu.Lock()
			a.delay = delay
			a.mu.Unlock()
			// Delay first, then tap: the timeline must not start until the
			// audio it describes is actually being held back.
			a.playout.SetDelay(delay)
			a.playout.SetTap(a)
			mAvatarSessions.Inc()
			infof("[%s] avatar ready: %s/%s %dx%d@%d %s, audio delayed %s for lip-sync",
				a.sessionID, ready.AvatarId, ready.BackgroundId, ready.Width, ready.Height,
				ready.Fps, ready.Encoder, delay)
			continue
		}
		frame := ev.GetVideo()
		if frame == nil || len(frame.Data) == 0 {
			continue
		}
		// Hold the frame until the audio it depicts leaves the delay line.
		// Sleeping here is the pacing: frames arrive in timeline order, and a
		// service that outruns us is backpressured by the stream itself.
		a.mu.Lock()
		due := a.anchor.Add(time.Duration(frame.PtsSamples) * time.Second / avatarRate).Add(a.delay)
		a.mu.Unlock()
		wait := time.Until(due)
		switch {
		case wait < -frameDur:
			mAvatarLateFrames.Inc() // shown anyway: a late mouth beats a frozen one
		case wait > 0:
			timer := time.NewTimer(wait)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
		}
		if err := a.video.WriteSample(media.Sample{Data: frame.Data, Duration: frameDur}); err != nil {
			return // track closed / peer gone
		}
		mAvatarFrames.Inc()
	}
}

// Tick implements playoutTap: one slice of the session timeline per playout
// tick. Runs under the playout lock, so it only copies and enqueues.
func (a *avatarSession) Tick(frame []int16, now time.Time) {
	speech := make([]byte, avatarTickSamp*2)
	if frame != nil {
		// 48 kHz -> 16 kHz by averaging each group of three samples: a crude
		// low-pass, but the model only reads speech-band content from it.
		for i := 0; i < avatarTickSamp; i++ {
			s := (int32(frame[3*i]) + int32(frame[3*i+1]) + int32(frame[3*i+2])) / 3
			speech[2*i], speech[2*i+1] = byte(s), byte(s>>8)
		}
	}
	listen := make([]byte, avatarTickSamp*2)
	a.mu.Lock()
	n := min(len(a.mic), avatarTickSamp)
	for i := 0; i < n; i++ {
		listen[2*i], listen[2*i+1] = byte(a.mic[i]), byte(a.mic[i]>>8)
	}
	a.mic = a.mic[:copy(a.mic, a.mic[n:])]
	pts := a.pts
	a.pts += avatarTickSamp
	// Re-anchored every tick rather than fixed at the first one: if the
	// ticker ever skips, audio slips by a tick and video must slip with it.
	a.anchor = now.Add(-time.Duration(pts) * time.Second / avatarRate)
	a.mu.Unlock()
	if !a.enqueue(&avatarpb.ClientEvent{Msg: &avatarpb.ClientEvent_Audio{Audio: &avatarpb.AudioFrame{
		SpeechPcm16: speech, ListenPcm16: listen, PtsSamples: pts,
	}}}) {
		mAvatarAudioDropped.Inc() // the service fills the hole with silence
	}
}

// PushMic queues decoded microphone audio for the next tick's listening track.
func (a *avatarSession) PushMic(pcm []int16) {
	a.mu.Lock()
	a.mic = append(a.mic, pcm...)
	if extra := len(a.mic) - avatarMicMaxSamp; extra > 0 {
		a.mic = a.mic[:copy(a.mic, a.mic[extra:])]
	}
	a.mu.Unlock()
}

// FlushSpeech tells the service the agent audio it has not rendered yet was
// cut by a barge-in, so the mouth stops with the sound.
func (a *avatarSession) FlushSpeech() {
	a.enqueue(&avatarpb.ClientEvent{Msg: &avatarpb.ClientEvent_Control{
		Control: &avatarpb.Control{Type: avatarpb.Control_FLUSH_SPEECH}}})
}

// RequestKeyframe asks for a fresh keyframe, at most twice a second: a browser
// that lost a reference repeats its PLI until one arrives.
func (a *avatarSession) RequestKeyframe() {
	a.mu.Lock()
	recent := time.Since(a.lastKeyReq) < 500*time.Millisecond
	if !recent {
		a.lastKeyReq = time.Now()
	}
	a.mu.Unlock()
	if !recent {
		a.enqueue(&avatarpb.ClientEvent{Msg: &avatarpb.ClientEvent_Control{
			Control: &avatarpb.Control{Type: avatarpb.Control_KEYFRAME}}})
	}
}

// watchKeyframeRequests drains RTCP for the video sender (pion's interceptors
// need it read regardless) and turns picture-loss reports into keyframe
// requests. Returns when the sender is closed.
func watchKeyframeRequests(sender *webrtc.RTPSender, session func() *avatarSession) {
	for {
		packets, _, err := sender.ReadRTCP()
		if err != nil {
			return
		}
		for _, packet := range packets {
			switch packet.(type) {
			case *rtcp.PictureLossIndication, *rtcp.FullIntraRequest:
				if a := session(); a != nil {
					a.RequestKeyframe()
				}
			}
		}
	}
}

// avatarSink is the playout as the bridge sees it when a video track was
// negotiated: the same sink, plus the two things the avatar needs from the
// bridge — the microphone audio, and the moment of a barge-in.
type avatarSink struct {
	*playout
	session func() *avatarSession // nil until the audio track starts
}

func (s avatarSink) Flush() {
	s.playout.Flush()
	if a := s.session(); a != nil {
		a.FlushSpeech()
	}
}

func (s avatarSink) PushMic(pcm []int16) {
	if a := s.session(); a != nil {
		a.PushMic(pcm)
	}
}
