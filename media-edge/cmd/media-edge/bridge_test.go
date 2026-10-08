package main

import (
	"context"
	"errors"
	"io"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/rtp"

	"mondee/media-edge/internal/edgepb"
	"mondee/media-edge/internal/profile"
)

// --- Seam fakes for runBridge ---

// scriptedTrack drives the read loop from a fixed list of steps: each step is
// either an RTP payload or an error (a timeout to exercise the idle path, or a
// terminal error to end the loop). Exhausting the script returns io.EOF.
// scriptedTrack replays a fixed sequence of reads AND the wall time each one
// would have consumed in production. The idle branch of runBridge sizes its
// synthetic silence from elapsed wall time (so it cannot double-count a packet
// it already accounted for), which makes these steps meaningless without a
// clock that moves: a step returning instantly represents zero elapsed time.
// Each read therefore advances the fake clock by what the real track would
// have blocked for — one frame duration for a delivered packet, the full poll
// interval for an expired read deadline.
type scriptedTrack struct {
	steps []trackStep
	i     int
	clock *fakeClock
}

type trackStep struct {
	payload []byte
	err     error
}

// fakeClock drives nowFn during a test. Not safe for concurrent use; these
// tests run runBridge on the calling goroutine.
type fakeClock struct{ t time.Time }

func (c *fakeClock) Now() time.Time          { return c.t }
func (c *fakeClock) Advance(d time.Duration) { c.t = c.t.Add(d) }

// installFakeClock points nowFn at a clock the test controls and returns it
// along with the restore func.
func installFakeClock(t *testing.T) *fakeClock {
	t.Helper()
	c := &fakeClock{t: time.Unix(1700000000, 0)}
	prev := nowFn
	nowFn = c.Now
	t.Cleanup(func() { nowFn = prev })
	return c
}

func (s *scriptedTrack) SetReadDeadline(time.Time) error { return nil }

func (s *scriptedTrack) ReadRTP() (*rtp.Packet, interceptor.Attributes, error) {
	if s.i >= len(s.steps) {
		return nil, nil, io.EOF
	}
	st := s.steps[s.i]
	s.i++
	if s.clock != nil {
		if st.err != nil {
			s.clock.Advance(rtpReadPollInterval) // the read deadline expired
		} else {
			s.clock.Advance(testFrameDuration) // one decoded frame arrived
		}
	}
	if st.err != nil {
		return nil, nil, st.err
	}
	return &rtp.Packet{Payload: st.payload}, nil, nil
}

// testFrameDuration matches the 160 samples at 16 kHz that testDecode returns.
const testFrameDuration = 10 * time.Millisecond

func loudSteps(n int) []trackStep {
	out := make([]trackStep, n)
	for i := range out {
		out[i] = trackStep{payload: []byte("L")}
	}
	return out
}

func quietSteps(n int) []trackStep {
	out := make([]trackStep, n)
	for i := range out {
		out[i] = trackStep{payload: []byte("q")}
	}
	return out
}

func timeoutStep() trackStep { return trackStep{err: &timeoutError{timeout: true}} }

// testDecode stands in for opus.DecodeToInt16: payload "L" -> a loud 10 ms
// frame (above the energy enter line), anything else -> silence.
func testDecode(payload []byte, out []int16) (int, error) {
	amp := int16(0)
	if len(payload) > 0 && payload[0] == 'L' {
		amp = 3000
	}
	for i := 0; i < 160; i++ {
		out[i] = amp
	}
	return 160, nil
}

// mockBridgeStream records sent events and can be scripted to fail a Send.
// Recv blocks until CloseSend so the half-close in runBridge returns promptly
// with io.EOF instead of waiting out finalReplyWait.
type mockBridgeStream struct {
	sent   []*edgepb.ClientEvent
	failFn func(callN int, ev *edgepb.ClientEvent) bool
	callN  int
	closed chan struct{}
}

func newMockBridgeStream() *mockBridgeStream {
	return &mockBridgeStream{closed: make(chan struct{})}
}

func (m *mockBridgeStream) Send(ev *edgepb.ClientEvent) error {
	m.callN++
	if m.failFn != nil && m.failFn(m.callN, ev) {
		return errors.New("mock send failure")
	}
	m.sent = append(m.sent, ev)
	return nil
}

func (m *mockBridgeStream) Recv() (*edgepb.ServerEvent, error) {
	<-m.closed
	return nil, io.EOF
}

func (m *mockBridgeStream) CloseSend() error {
	select {
	case <-m.closed:
	default:
		close(m.closed)
	}
	return nil
}

func (m *mockBridgeStream) controls() []edgepb.Control_Type {
	var out []edgepb.Control_Type
	for _, ev := range m.sent {
		if c := ev.GetControl(); c != nil {
			out = append(out, c.Type)
		}
	}
	return out
}

func (m *mockBridgeStream) audioFrames() int {
	n := 0
	for _, ev := range m.sent {
		if ev.GetAudio() != nil {
			n++
		}
	}
	return n
}

func runTestBridge(t *testing.T, ctx context.Context, track *scriptedTrack, ms *mockBridgeStream) (int64, error, bool) {
	t.Helper()
	completed := false
	if track.clock == nil {
		// Every bridge test needs reads to consume time; see scriptedTrack.
		track.clock = installFakeClock(t)
	}
	bytes, err := runBridge(ctx, ms, track, testDecode, testVAD(), profile.Profile{}, "sess-test",
		nil /*sink*/, nil /*onReply*/, func(string) { completed = true }, nil /*onTurnEvent*/)
	return bytes, err, completed
}

// --- Tests ---

// TestRunBridgeHappyTurn walks one full turn: sustained speech crosses the
// attack window (USER_SPEECH_STARTED), then sustained silence past the hangover
// endpoints the turn (END_TURN), and the stream is always bracketed by
// SESSION_OPEN / SESSION_CLOSE with audio forwarded in between.
func TestRunBridgeHappyTurn(t *testing.T) {
	track := &scriptedTrack{steps: append(loudSteps(8), quietSteps(12)...)}
	ms := newMockBridgeStream()

	bytes, err, completed := runTestBridge(t, context.Background(), track, ms)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !completed {
		t.Fatal("onComplete was not called")
	}
	if bytes == 0 {
		t.Fatal("expected PCM bytes forwarded")
	}
	if ms.audioFrames() != 20 {
		t.Fatalf("expected 20 audio frames (8 loud + 12 quiet), got %d", ms.audioFrames())
	}
	want := []edgepb.Control_Type{
		edgepb.Control_SESSION_OPEN,
		edgepb.Control_USER_SPEECH_STARTED,
		edgepb.Control_END_TURN,
		edgepb.Control_SESSION_CLOSE,
	}
	if got := ms.controls(); !reflect.DeepEqual(got, want) {
		t.Fatalf("control sequence: got %v, want %v", got, want)
	}
}

// TestRunBridgeIdleTimeoutEndpoints verifies the read-timeout branch keeps the
// VAD state machine advancing when packets stop mid-turn: after speech starts,
// a read timeout (no RTP) must still drive the hangover to completion and emit
// END_TURN — a turn must not hang open just because the client went silent.
func TestRunBridgeIdleTimeoutEndpoints(t *testing.T) {
	track := &scriptedTrack{steps: append(loudSteps(8), timeoutStep())}
	ms := newMockBridgeStream()

	_, err, completed := runTestBridge(t, context.Background(), track, ms)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !completed {
		t.Fatal("onComplete was not called")
	}
	got := ms.controls()
	if !slices.Contains(got, edgepb.Control_USER_SPEECH_STARTED) {
		t.Fatalf("expected USER_SPEECH_STARTED, got %v", got)
	}
	if !slices.Contains(got, edgepb.Control_END_TURN) {
		t.Fatalf("idle timeout must endpoint the turn (END_TURN), got %v", got)
	}
	if got[len(got)-1] != edgepb.Control_SESSION_CLOSE {
		t.Fatalf("stream must end with SESSION_CLOSE, got %v", got)
	}
}

// TestRunBridgeSendErrorAborts: a failed upstream Send breaks the loop, returns
// the error, and skips the SESSION_CLOSE control (a dead stream can't close
// cleanly) — but onComplete still runs so the caller finalizes its side.
func TestRunBridgeSendErrorAborts(t *testing.T) {
	track := &scriptedTrack{steps: loudSteps(8)}
	ms := newMockBridgeStream()
	ms.failFn = func(_ int, ev *edgepb.ClientEvent) bool { return ev.GetAudio() != nil }

	_, err, completed := runTestBridge(t, context.Background(), track, ms)
	if err == nil {
		t.Fatal("expected error when an audio Send fails")
	}
	if !completed {
		t.Fatal("onComplete must run even after a send failure")
	}
	if slices.Contains(ms.controls(), edgepb.Control_SESSION_CLOSE) {
		t.Fatalf("SESSION_CLOSE must not be sent after a mid-stream send error, got %v", ms.controls())
	}
}

// TestRunBridgeContextCancelStops: a cancelled context observed on a read
// timeout stops the loop before any speech, still bracketing with a clean
// SESSION_OPEN/SESSION_CLOSE and returning no error.
func TestRunBridgeContextCancelStops(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	track := &scriptedTrack{steps: []trackStep{timeoutStep()}}
	ms := newMockBridgeStream()

	_, err, completed := runTestBridge(t, ctx, track, ms)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !completed {
		t.Fatal("onComplete was not called")
	}
	want := []edgepb.Control_Type{edgepb.Control_SESSION_OPEN, edgepb.Control_SESSION_CLOSE}
	if got := ms.controls(); !reflect.DeepEqual(got, want) {
		t.Fatalf("cancel path should bracket with OPEN/CLOSE only, got %v", got)
	}
}

// --- Downstream control routing -----------------------------------------

// scriptedRecvStream is a mockBridgeStream that also plays back a fixed list of
// ServerEvents to runBridge's receive goroutine before returning io.EOF, so the
// downstream (orchestrator -> edge) leg is testable without a real orchestrator.
type scriptedRecvStream struct {
	*mockBridgeStream
	events []*edgepb.ServerEvent
	i      int
}

func (m *scriptedRecvStream) Recv() (*edgepb.ServerEvent, error) {
	if m.i < len(m.events) {
		ev := m.events[m.i]
		m.i++
		return ev, nil
	}
	<-m.closed
	return nil, io.EOF
}

// recordingSink captures what the downstream leg drives the playout buffer with.
type recordingSink struct {
	mu      sync.Mutex
	writes  int
	flushes int
	ends    []turnEnd
}

func (s *recordingSink) Write(sessionID string, turnID int64, pcm []byte, sampleRate int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.writes++
}

func (s *recordingSink) SetPlaybackStarted(func(string, int64, int64)) {}

func (s *recordingSink) Flush() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.flushes++
}

func (s *recordingSink) MarkTurnEnd(sessionID string, turnID int64, reason string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ends = append(s.ends, turnEnd{sessionID, turnID, reason})
}

func serverControl(t edgepb.Control_Type, turnID int64, reason string) *edgepb.ServerEvent {
	c := &edgepb.Control{SessionId: "sess-test", Type: t, TurnId: turnID}
	if reason != "" {
		c.Metadata = map[string]string{"reason": reason}
	}
	return &edgepb.ServerEvent{Msg: &edgepb.ServerEvent_Control{Control: c}}
}

func runScriptedBridge(t *testing.T, sink playoutSink, events ...*edgepb.ServerEvent) []*edgepb.Control {
	t.Helper()
	ms := &scriptedRecvStream{mockBridgeStream: newMockBridgeStream(), events: events}
	var mu sync.Mutex
	var seen []*edgepb.Control
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	track := &scriptedTrack{steps: quietSteps(2)}
	if _, err := runBridge(ctx, ms, track, testDecode, testVAD(), profile.Profile{}, "sess-test",
		sink, nil /*onReply*/, nil /*onComplete*/, func(c *edgepb.Control) {
			mu.Lock()
			defer mu.Unlock()
			seen = append(seen, c)
		}); err != nil {
		t.Fatalf("runBridge: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	return seen
}

// AGENT_TURN_STARTED is a passthrough, but AGENT_TURN_ENDED is not: it means
// "no more audio is coming", which is up to a prebuffer earlier than "the user
// has heard it all", so it is handed to the playout buffer instead.
func TestBridgeRoutesAgentTurnLifecycle(t *testing.T) {
	sink := &recordingSink{}
	seen := runScriptedBridge(t, sink,
		serverControl(edgepb.Control_AGENT_TURN_STARTED, 2, ""),
		&edgepb.ServerEvent{Msg: &edgepb.ServerEvent_Audio{Audio: &edgepb.AudioFrame{
			SessionId: "sess-test", Pcm16: []byte{1, 0, 2, 0}, SampleRateHz: 24000, TurnId: 2}}},
		serverControl(edgepb.Control_AGENT_TURN_ENDED, 2, "complete"),
	)

	if len(seen) != 1 || seen[0].Type != edgepb.Control_AGENT_TURN_STARTED || seen[0].TurnId != 2 {
		t.Fatalf("forwarded %+v, want only AGENT_TURN_STARTED for turn 2", seen)
	}
	if sink.writes != 1 {
		t.Fatalf("sink writes=%d want 1", sink.writes)
	}
	want := turnEnd{"sess-test", 2, "complete"}
	if len(sink.ends) != 1 || sink.ends[0] != want {
		t.Fatalf("sink ends=%+v want %+v", sink.ends, want)
	}
}

// STOP_PLAYBACK still cuts the buffer, and the orchestrator's follow-up
// AGENT_TURN_ENDED(interrupted) is what closes the turn for the client.
func TestBridgeStopPlaybackThenInterruptedEnd(t *testing.T) {
	sink := &recordingSink{}
	runScriptedBridge(t, sink,
		serverControl(edgepb.Control_STOP_PLAYBACK, 3, "barge_in"),
		serverControl(edgepb.Control_AGENT_TURN_ENDED, 3, "interrupted"),
	)
	if sink.flushes != 1 {
		t.Fatalf("flushes=%d want 1", sink.flushes)
	}
	if len(sink.ends) != 1 || sink.ends[0].reason != "interrupted" {
		t.Fatalf("ends=%+v want one interrupted marker", sink.ends)
	}
}

// With no playout plane there is nothing to drain, so the edge synthesizes
// PLAYBACK_COMPLETE itself rather than leaving the client's turn state open.
func TestBridgeSynthesizesPlaybackCompleteWithoutSink(t *testing.T) {
	seen := runScriptedBridge(t, nil,
		serverControl(edgepb.Control_AGENT_TURN_ENDED, 5, "error"),
	)
	if len(seen) != 1 || seen[0].Type != edgepb.Control_PLAYBACK_COMPLETE {
		t.Fatalf("forwarded %+v, want PLAYBACK_COMPLETE", seen)
	}
	if seen[0].TurnId != 5 || seen[0].Metadata["reason"] != "error" {
		t.Fatalf("got turn=%d reason=%q want 5/error", seen[0].TurnId, seen[0].Metadata["reason"])
	}
}

// TestMediaClockNeverRunsAheadOfWall pins the defect that made every edge
// telemetry timestamp untrustworthy: the idle branch used to add a fixed
// rtpReadPollInterval of synthetic silence per expired read deadline, on top of
// whatever real packets had already been accounted for in that same span. The
// cursor gained media time no wall time backed, so control timestamps stamped
// from it ran ahead of the orchestrator's clock (up to 7% over a call) and the
// endpoint-to-first-audio subtraction went negative and was clamped to 0 ms.
func TestMediaClockNeverRunsAheadOfWall(t *testing.T) {
	clock := &fakeClock{t: time.Unix(1700000000, 0)}
	prev := nowFn
	nowFn = clock.Now
	defer func() { nowFn = prev }()

	c := newMediaClock()
	// A packet's worth of audio arrives, consuming its own duration of wall
	// time, and then the read deadline expires.
	clock.Advance(testFrameDuration)
	c.Advance(160) // 10 ms at 16 kHz
	clock.Advance(rtpReadPollInterval)

	fill := c.PendingSamples(clock.Now())
	c.Advance(fill)
	if got := c.Now(); got.After(clock.Now()) {
		t.Fatalf("media cursor ran %v ahead of wall clock", got.Sub(clock.Now()))
	}
	// And with no wall time elapsed, an expired deadline must add nothing at
	// all rather than another full interval.
	if extra := c.PendingSamples(clock.Now()); extra != 0 {
		t.Fatalf("idle fill double-counted: %d samples for 0 ms of wall time", extra)
	}
}

// TestVADProcessMsIsPerEvent guards the reporting contract for the value the
// orchestrator stores in duration_ms. It used to be the session-cumulative
// counter, which made VAD duration_ms climb monotonically across a call and
// turned any average or quantile over it into a measure of how late the event
// was, not how long the VAD took.
func TestVADProcessMsIsPerEvent(t *testing.T) {
	v := testVAD()
	meta := make(map[string]string, 14)

	v.processNS = 5 * int64(time.Millisecond)
	if got := vadMeta(meta, v, profile.Profile{}, 0, 0, 0, 0)["vad_process_ms"]; got != "5" {
		t.Fatalf("first report: vad_process_ms = %q, want 5", got)
	}
	v.processNS = 12 * int64(time.Millisecond)
	got := vadMeta(meta, v, profile.Profile{}, 0, 0, 0, 0)
	if got["vad_process_ms"] != "7" {
		t.Fatalf("second report: vad_process_ms = %q, want 7 (the delta)", got["vad_process_ms"])
	}
	if got["vad_process_total_ms"] != "12" {
		t.Fatalf("running total lost: vad_process_total_ms = %q, want 12", got["vad_process_total_ms"])
	}
}

// TestControlTimestampIsWallClock: TsMs is the field the orchestrator subtracts
// from its own wall clock, so it must never carry the media cursor. An older
// build sent a session-relative cursor here, landing events in 1970.
func TestControlTimestampIsWallClock(t *testing.T) {
	clock := &fakeClock{t: time.Unix(1700000000, 0)}
	prev := nowFn
	nowFn = clock.Now
	defer func() { nowFn = prev }()

	mediaTime := time.Unix(1600000000, 0) // deliberately far from wall
	meta := map[string]string{}
	ev := controlAt("sess", edgepb.Control_END_TURN, mediaTime, meta)

	if got := ev.GetControl().TsMs; got != clock.Now().UnixMilli() {
		t.Fatalf("TsMs = %d, want wall clock %d", got, clock.Now().UnixMilli())
	}
	if got := meta["media_ts_ms"]; got != "1600000000000" {
		t.Fatalf("media cursor must still travel as metadata, got %q", got)
	}
}
