package main

import (
	"context"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"mondee/media-edge/internal/avatarpb"
)

// --- Lip-sync delay line -------------------------------------------------
//
// These drive schedule/Flush/MarkTurnEnd directly, the same way the drain-marker
// tests drive step: no ticker, no track, no Opus.

func constFrame(v int16) []int16 {
	f := make([]int16, playoutFrameSamp)
	for i := range f {
		f[i] = v
	}
	return f
}

type recordingTap struct{ frames [][]int16 }

func (t *recordingTap) Tick(frame []int16, _ time.Time) {
	if frame == nil {
		t.frames = append(t.frames, nil)
		return
	}
	t.frames = append(t.frames, append([]int16(nil), frame...))
}

// Without a delay the playout must behave exactly as it did before the avatar
// existed: what goes in comes straight back out, and the tap is only a witness.
func TestScheduleWithoutDelayIsPassthrough(t *testing.T) {
	p, _ := newTestPlayout(playoutFrameSamp)
	tap := &recordingTap{}
	p.SetTap(tap)
	p.pendingStart = &playbackStart{sessionID: "s", turnID: 1, queueMS: 5}
	end := &turnEnd{sessionID: "s", turnID: 1, reason: "complete"}

	due := p.schedule(constFrame(7), true, end, time.Now())
	if due.frame == nil || due.frame[0] != 7 || due.start == nil || due.start.queueMS != 5 || due.end != end {
		t.Fatalf("passthrough tick = %+v", due)
	}
	if due := p.schedule(constFrame(0), false, nil, time.Now()); due.frame != nil || due.start != nil || due.end != nil {
		t.Fatalf("silent tick should owe nothing, got %+v", due)
	}
	if len(tap.frames) != 2 || tap.frames[0][0] != 7 || tap.frames[1] != nil {
		t.Fatalf("tap saw %v", tap.frames)
	}
}

// The tap must see a frame the instant it is scheduled, while the wire gets it
// a fixed number of ticks later with its turn boundaries still attached.
func TestDelayLineHoldsAudioAndItsTurnBoundaries(t *testing.T) {
	p, _ := newTestPlayout(playoutFrameSamp)
	tap := &recordingTap{}
	p.SetTap(tap)
	p.SetDelay(50 * time.Millisecond) // rounds up to 3 ticks of 20 ms
	if len(p.delay) != 3 {
		t.Fatalf("delay = %d ticks, want 3", len(p.delay))
	}
	p.pendingStart = &playbackStart{sessionID: "s", turnID: 1, queueMS: 5}
	end := &turnEnd{sessionID: "s", turnID: 1, reason: "complete"}

	if due := p.schedule(constFrame(7), true, end, time.Now()); due.frame != nil || due.start != nil || due.end != nil {
		t.Fatalf("delayed tick leaked onto the wire immediately: %+v", due)
	}
	if tap.frames[0] == nil || tap.frames[0][0] != 7 {
		t.Fatal("tap must see the frame before the delay, not after")
	}
	for i := 0; i < 2; i++ {
		if due := p.schedule(constFrame(0), false, nil, time.Now()); due.frame != nil {
			t.Fatalf("tick %d: frame came out early", i+1)
		}
	}
	due := p.schedule(constFrame(0), false, nil, time.Now())
	if due.frame == nil || due.frame[0] != 7 {
		t.Fatalf("frame did not come out after 3 ticks: %+v", due)
	}
	if due.start == nil || due.start.queueMS != 5+60 {
		t.Fatalf("playback start should report the delay as queue time, got %+v", due.start)
	}
	if due.end != end {
		t.Fatal("turn end must leave the line with the audio it belongs to")
	}
}

// Barge-in: audio parked in the delay line has not been heard, so it must be
// cut with the rest, and a turn whose end marker was parked is "interrupted".
func TestFlushCutsDelayedAudioAndCompletesParkedTurn(t *testing.T) {
	p, got := newTestPlayout(playoutFrameSamp)
	p.SetDelay(60 * time.Millisecond)
	p.schedule(constFrame(7), true, &turnEnd{sessionID: "s", turnID: 4, reason: "complete"}, time.Now())

	p.Flush()
	if len(*got) != 1 || (*got)[0] != (completion{"s", 4, "interrupted"}) {
		t.Fatalf("completions = %+v", *got)
	}
	for i := 0; i < 4; i++ {
		if due := p.schedule(constFrame(0), false, nil, time.Now()); due.frame != nil || due.end != nil {
			t.Fatalf("flushed audio or marker resurfaced on tick %d: %+v", i, due)
		}
	}
}

// With a delay line an empty buffer does not mean the listener has heard
// everything, so the completion has to wait its turn in the line.
func TestMarkTurnEndWaitsForTheDelayLine(t *testing.T) {
	p, got := newTestPlayout(playoutFrameSamp)
	p.SetDelay(40 * time.Millisecond)
	frame := make([]int16, playoutFrameSamp)

	p.MarkTurnEnd("s", 9, "complete")
	if len(*got) != 0 {
		t.Fatalf("completed before the delay elapsed: %+v", *got)
	}
	for i := 0; i < 3; i++ {
		now := time.Now()
		emit, end := p.step(frame, now)
		p.fire(p.schedule(frame, emit, end, now).end, "")
	}
	if len(*got) != 1 || (*got)[0] != (completion{"s", 9, "complete"}) {
		t.Fatalf("completions = %+v", *got)
	}
}

// --- Timeline frames -----------------------------------------------------

func TestTickPairsAgentAudioWithMicrophoneOnOneTimeline(t *testing.T) {
	a := &avatarSession{send: make(chan *avatarpb.ClientEvent, 4)}
	mic := make([]int16, avatarTickSamp+avatarTickSamp/2) // one and a half ticks
	for i := range mic {
		mic[i] = 1000
	}
	a.PushMic(mic)
	t0 := time.Unix(1700000000, 0)

	a.Tick(constFrame(3000), t0)
	a.Tick(nil, t0.Add(20*time.Millisecond))

	first, second := (<-a.send).GetAudio(), (<-a.send).GetAudio()
	if first.PtsSamples != 0 || second.PtsSamples != avatarTickSamp {
		t.Fatalf("pts = %d, %d", first.PtsSamples, second.PtsSamples)
	}
	speech, listen := bytesToInt16(first.SpeechPcm16), bytesToInt16(first.ListenPcm16)
	if len(speech) != avatarTickSamp || speech[0] != 3000 || speech[avatarTickSamp-1] != 3000 {
		t.Fatalf("48 kHz agent frame did not downsample to a full 16 kHz tick: len=%d first=%d", len(speech), speech[0])
	}
	if listen[0] != 1000 || listen[avatarTickSamp-1] != 1000 {
		t.Fatal("first tick should carry a full tick of microphone audio")
	}
	speech, listen = bytesToInt16(second.SpeechPcm16), bytesToInt16(second.ListenPcm16)
	if speech[0] != 0 {
		t.Fatal("a silent tick must send silence, not stale agent audio")
	}
	if listen[avatarTickSamp/2-1] != 1000 || listen[avatarTickSamp/2] != 0 {
		t.Fatal("leftover microphone audio should be zero-padded to a full tick")
	}
	if !a.anchor.Equal(t0) {
		t.Fatalf("anchor = %v, want the wall time of timeline position 0 (%v)", a.anchor, t0)
	}
}

func TestTickNeverBlocksWhenTheStreamIsBackedUp(t *testing.T) {
	a := &avatarSession{send: make(chan *avatarpb.ClientEvent, 1)}
	before := testutil.ToFloat64(mAvatarAudioDropped)
	a.Tick(nil, time.Now())
	a.Tick(nil, time.Now()) // queue full: must return, and must still advance the timeline
	if got := testutil.ToFloat64(mAvatarAudioDropped) - before; got != 1 {
		t.Fatalf("dropped = %v, want 1", got)
	}
	if a.pts != 2*avatarTickSamp {
		t.Fatalf("timeline stalled at %d; a dropped frame must still consume its slot", a.pts)
	}
}

func TestOfferHasVideo(t *testing.T) {
	audioOnly := "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=mid:0\r\n"
	if offerHasVideo(audioOnly) {
		t.Fatal("audio-only offer reported video")
	}
	if !offerHasVideo(audioOnly + "m=video 9 UDP/TLS/RTP/SAVPF 102\r\na=mid:1\r\n") {
		t.Fatal("video offer not detected")
	}
}

// --- Negotiation ---------------------------------------------------------

// browserOffer builds what the page sends: a sendrecv audio track, and (when
// the edge advertises an avatar) a recvonly video transceiver.
func browserOffer(t *testing.T, withVideo bool) (*webrtc.PeerConnection, webrtc.SessionDescription) {
	t.Helper()
	browser, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = browser.Close() })
	mic, err := webrtc.NewTrackLocalStaticSample(webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus}, "audio", "mic")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := browser.AddTrack(mic); err != nil {
		t.Fatal(err)
	}
	if withVideo {
		if _, err := browser.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo,
			webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
			t.Fatal(err)
		}
	}
	offer, err := browser.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := browser.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	return browser, offer
}

func videoSection(sdp string) string {
	i := strings.Index(sdp, "m=video ")
	if i < 0 {
		return ""
	}
	return sdp[i:]
}

// The whole point of the video plumbing: a recvonly video offer must be
// answered with a live, sendonly H.264 section the browser accepts.
func TestVideoOfferIsAnsweredWithSendonlyH264(t *testing.T) {
	browser, offer := browserOffer(t, true)
	edge, err := newPeerConnection(true)
	if err != nil {
		t.Fatal(err)
	}
	defer edge.Close()
	if err := edge.SetRemoteDescription(offer); err != nil {
		t.Fatal(err)
	}
	audio, _ := webrtc.NewTrackLocalStaticSample(webrtc.RTPCodecCapability{
		MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2, SDPFmtpLine: "minptime=10;useinbandfec=1",
	}, "audio", "agent")
	if _, err := edge.AddTrack(audio); err != nil {
		t.Fatal(err)
	}
	video, err := webrtc.NewTrackLocalStaticSample(avatarVideoCodec, "video", "agent")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := edge.AddTrack(video); err != nil {
		t.Fatal(err)
	}
	answer, err := edge.CreateAnswer(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := edge.SetLocalDescription(answer); err != nil {
		t.Fatal(err)
	}
	section := videoSection(answer.SDP)
	if section == "" || strings.HasPrefix(section, "m=video 0 ") {
		t.Fatalf("video section missing or rejected:\n%s", answer.SDP)
	}
	if !strings.Contains(section, "a=sendonly") || !strings.Contains(section, "H264/90000") {
		t.Fatalf("video section is not sendonly H.264:\n%s", section)
	}
	if strings.Count(answer.SDP, "m=") != strings.Count(offer.SDP, "m=") {
		t.Fatal("answer must reuse the offer's transceivers, not add media sections")
	}
	if err := browser.SetRemoteDescription(answer); err != nil {
		t.Fatalf("browser rejected the answer: %v", err)
	}
}

// An edge with no avatar must still cope with a page that asks for video
// anyway (a cached page, say): reject that one section, keep the audio.
func TestVideoOfferWithoutAvatarKeepsAudio(t *testing.T) {
	browser, offer := browserOffer(t, true)
	edge, err := newPeerConnection(false)
	if err != nil {
		t.Fatal(err)
	}
	defer edge.Close()
	if err := edge.SetRemoteDescription(offer); err != nil {
		t.Fatal(err)
	}
	answer, err := edge.CreateAnswer(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := edge.SetLocalDescription(answer); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(videoSection(answer.SDP), "m=video 0 ") {
		t.Fatalf("video should be rejected when no avatar is configured:\n%s", answer.SDP)
	}
	if !strings.Contains(answer.SDP, "opus/48000") {
		t.Fatal("audio must still be negotiated")
	}
	if err := browser.SetRemoteDescription(answer); err != nil {
		t.Fatalf("browser rejected the answer: %v", err)
	}
}

// --- Session against a fake avatar service -------------------------------

type fakeAvatarServer struct {
	avatarpb.UnimplementedAvatarServer
	mu       sync.Mutex
	open     *avatarpb.SessionOpen
	audio    []*avatarpb.AudioFrame
	controls []avatarpb.Control_Type
	gotAudio chan struct{}
}

func (s *fakeAvatarServer) Stream(stream avatarpb.Avatar_StreamServer) error {
	first, err := stream.Recv()
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.open = first.GetOpen()
	s.mu.Unlock()
	if err := stream.Send(&avatarpb.ServerEvent{Msg: &avatarpb.ServerEvent_Ready{Ready: &avatarpb.SessionReady{
		Codec: "h264", Width: 1280, Height: 720, Fps: 25, LookaheadMs: 405, Encoder: "fake",
	}}}); err != nil {
		return err
	}
	sentVideo := false
	for {
		ev, err := stream.Recv()
		if err != nil {
			return nil
		}
		s.mu.Lock()
		if a := ev.GetAudio(); a != nil {
			s.audio = append(s.audio, a)
		}
		if c := ev.GetControl(); c != nil {
			s.controls = append(s.controls, c.Type)
		}
		first := !sentVideo && len(s.audio) > 0
		s.mu.Unlock()
		if first {
			sentVideo = true
			_ = stream.Send(&avatarpb.ServerEvent{Msg: &avatarpb.ServerEvent_Video{Video: &avatarpb.VideoFrame{
				Data: []byte{0, 0, 0, 1, 0x65}, PtsSamples: 0, Keyframe: true,
			}}})
			close(s.gotAudio)
		}
	}
}

func dialFakeAvatar(t *testing.T, srv *fakeAvatarServer) *grpc.ClientConn {
	t.Helper()
	lis := bufconn.Listen(1 << 20)
	server := grpc.NewServer()
	avatarpb.RegisterAvatarServer(server, srv)
	go func() { _ = server.Serve(lis) }()
	t.Cleanup(server.Stop)
	conn, err := grpc.NewClient("passthrough:///avatar",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return lis.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestAvatarSessionDelaysPlayoutStreamsTimelineAndPlaysVideo(t *testing.T) {
	srv := &fakeAvatarServer{gotAudio: make(chan struct{})}
	conn := dialFakeAvatar(t, srv)
	video, err := webrtc.NewTrackLocalStaticSample(avatarVideoCodec, "video", "agent")
	if err != nil {
		t.Fatal(err)
	}
	p, _ := newTestPlayout(playoutFrameSamp)
	framesBefore := testutil.ToFloat64(mAvatarFrames)

	a := startAvatar(conn, "sess-1", video, p)
	if a == nil {
		t.Fatal("startAvatar returned nil against a healthy service")
	}
	defer a.Close()

	// Ready arrives: only now is the playout delayed and tapped. 405 ms of
	// lookahead + the default 200 ms margin = 605 ms -> 31 ticks.
	waitFor(t, "avatar ready", func() bool {
		p.mu.Lock()
		defer p.mu.Unlock()
		return p.tap != nil
	})
	p.mu.Lock()
	delayTicks := len(p.delay)
	p.mu.Unlock()
	if delayTicks != 31 {
		t.Fatalf("delay = %d ticks, want 31", delayTicks)
	}

	sink := avatarSink{playout: p, session: func() *avatarSession { return a }}
	sink.PushMic(make([]int16, avatarTickSamp))
	p.schedule(constFrame(900), true, nil, time.Now().Add(-time.Second)) // tick 0, a second ago
	sink.Flush()

	<-srv.gotAudio
	waitFor(t, "flush control", func() bool {
		srv.mu.Lock()
		defer srv.mu.Unlock()
		return len(srv.controls) == 1
	})
	srv.mu.Lock()
	open, first, ctl := srv.open, srv.audio[0], srv.controls[0]
	srv.mu.Unlock()
	if open.GetSessionId() != "sess-1" {
		t.Fatalf("open = %+v", open)
	}
	if first.PtsSamples != 0 || len(first.SpeechPcm16) != avatarTickSamp*2 || len(first.ListenPcm16) != avatarTickSamp*2 {
		t.Fatalf("first timeline frame = pts %d, %d/%d bytes", first.PtsSamples, len(first.SpeechPcm16), len(first.ListenPcm16))
	}
	if ctl != avatarpb.Control_FLUSH_SPEECH {
		t.Fatalf("barge-in sent %v, want FLUSH_SPEECH", ctl)
	}
	// The frame for timeline position 0 was due a second ago plus the delay,
	// i.e. already: it must be written without waiting.
	waitFor(t, "video frame written", func() bool { return testutil.ToFloat64(mAvatarFrames)-framesBefore == 1 })
}

func TestAvatarUnavailableLeavesTheSessionAudioOnly(t *testing.T) {
	conn, err := grpc.NewClient("passthrough:///nowhere",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return nil, net.ErrClosed }),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	video, _ := webrtc.NewTrackLocalStaticSample(avatarVideoCodec, "video", "agent")
	p, _ := newTestPlayout(playoutFrameSamp)
	errorsBefore := testutil.ToFloat64(mAvatarErrors)

	if a := startAvatar(conn, "sess-1", video, p); a != nil {
		// A lazy channel accepts the first Send; the failure then surfaces on
		// the receive side.
		defer a.Close()
	}
	waitFor(t, "avatar failure to be counted", func() bool { return testutil.ToFloat64(mAvatarErrors)-errorsBefore == 1 })
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.tap != nil || len(p.delay) != 0 {
		t.Fatal("a session with no avatar must not be tapped or delayed")
	}
}
