package main

import (
	"os"
	"testing"
	"time"
)

// These cover the pure audio math behind the outbound playout plane. The buffer
// ticker and Opus encode need a live track and libopus, so they're exercised in
// integration, not here.

func TestResampleTo48k(t *testing.T) {
	in := []int16{0, 100, 200, 300}

	// 24 kHz -> 48 kHz is an exact 2x upsample; endpoints preserved, one linear
	// interpolation between each pair.
	got := resampleTo48k(in, 24000)
	if len(got) != 8 {
		t.Fatalf("24k->48k: len=%d want 8", len(got))
	}
	if got[0] != 0 || got[1] != 50 || got[2] != 100 {
		t.Fatalf("24k->48k interpolation wrong: %v", got[:3])
	}

	// Same rate is a passthrough (no allocation of a resampled copy needed).
	if got := resampleTo48k(in, 48000); len(got) != 4 {
		t.Fatalf("48k passthrough: len=%d want 4", len(got))
	}

	// A non-integer ratio still lands the right length (16k -> 48k = 3x).
	if got := resampleTo48k(in, 16000); len(got) != 12 {
		t.Fatalf("16k->48k: len=%d want 12", len(got))
	}

	// Defensive: non-positive rate and empty input don't panic.
	if resampleTo48k(nil, 24000) != nil {
		t.Fatal("empty input should resample to nil")
	}
	if got := resampleTo48k(in, 0); len(got) != 4 {
		t.Fatal("rate<=0 should pass through unchanged")
	}
}

func TestBytesToInt16(t *testing.T) {
	// Little-endian, signed: 0x0001 -> 1, 0xFFFF -> -1, 0x8000 -> -32768.
	got := bytesToInt16([]byte{0x01, 0x00, 0xFF, 0xFF, 0x00, 0x80})
	want := []int16{1, -1, -32768}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sample %d: got %d want %d", i, got[i], want[i])
		}
	}
	// A trailing odd byte is dropped rather than read out of bounds.
	if n := len(bytesToInt16([]byte{1, 0, 5})); n != 1 {
		t.Fatalf("odd trailing byte: len=%d want 1", n)
	}
}

func TestPlaybackStartReportsEdgeQueueOnce(t *testing.T) {
	p, _ := newTestPlayout(0)
	p.startedTurns = make(map[int64]bool)
	p.maxSamples = playoutRate * 30
	p.onTurnStart = func(sessionID string, turnID int64, queueMS int64) {
		t.Fatalf("callback must be dispatched by run, not step")
	}
	p.Write("session-1", 11, make([]byte, playoutFrameSamp*4), playoutRate)
	frame := make([]int16, playoutFrameSamp)
	if emit, _ := p.step(frame, time.Now()); !emit {
		t.Fatal("queued frame should start immediately with no prebuffer")
	}
	if p.pendingStart == nil || p.pendingStart.sessionID != "session-1" || p.pendingStart.turnID != 11 {
		t.Fatalf("missing playback-start marker: %+v", p.pendingStart)
	}
	p.pendingStart = nil
	if emit, _ := p.step(frame, time.Now()); !emit {
		t.Fatal("second frame should still be emitted")
	}
	if p.pendingStart != nil {
		t.Fatal("playback-start marker must be emitted only once per turn")
	}
}

func TestEnvInt(t *testing.T) {
	const key = "EDGE_TEST_ENVINT"
	os.Unsetenv(key)
	if got := envInt(key, 200, 0, 2000); got != 200 {
		t.Fatalf("unset: got %d want default 200", got)
	}
	os.Setenv(key, "500")
	defer os.Unsetenv(key)
	if got := envInt(key, 200, 0, 2000); got != 500 {
		t.Fatalf("valid: got %d want 500", got)
	}
	os.Setenv(key, "99999") // out of range -> default
	if got := envInt(key, 200, 0, 2000); got != 200 {
		t.Fatalf("out-of-range: got %d want default 200", got)
	}
	os.Setenv(key, "notanumber")
	if got := envInt(key, 200, 0, 2000); got != 200 {
		t.Fatalf("garbage: got %d want default 200", got)
	}
}

// --- Agent-turn drain markers -------------------------------------------
//
// These exercise step/MarkTurnEnd/Flush directly, so no track or libopus encoder
// is involved: run() is only a ticker around step, and emitFrame is the only
// part that touches Opus.

type completion struct {
	sessionID string
	turnID    int64
	reason    string
}

func newTestPlayout(prebufferSamples int) (*playout, *[]completion) {
	var got []completion
	p := &playout{
		startSamples: prebufferSamples,
		maxSamples:   playoutRate * 30,
		maxStall:     time.Duration(playoutMaxStallMS) * time.Millisecond,
	}
	p.onTurnComplete = func(sid string, turnID int64, reason string) {
		got = append(got, completion{sid, turnID, reason})
	}
	return p, &got
}

// A turn that produced no audio at all (LLM error, empty reply) must complete
// immediately — otherwise the client waits forever for a turn that is over.
func TestMarkTurnEndWithEmptyBufferCompletesImmediately(t *testing.T) {
	p, got := newTestPlayout(playoutFrameSamp)
	p.MarkTurnEnd("sess-1", 7, "error")
	if len(*got) != 1 || (*got)[0] != (completion{"sess-1", 7, "error"}) {
		t.Fatalf("got %+v, want one immediate error completion", *got)
	}
	if p.pendingEnd != nil {
		t.Fatal("marker should not be left pending")
	}
}

// The tail of a turn is shorter than the prebuffer cushion in the common case.
// Before the marker existed it sat in the buffer unplayed; now it is drained,
// zero-padded, and only then reported complete.
func TestPendingEndDrainsTailBelowPrebuffer(t *testing.T) {
	// A cushion far larger than the audio we queue: without a marker, step would
	// never start playing.
	p, got := newTestPlayout(playoutRate) // 1 s cushion
	p.buf = make([]int16, playoutFrameSamp/2)
	frame := make([]int16, playoutFrameSamp)

	if emit, end := p.step(frame, time.Now()); emit || end != nil {
		t.Fatal("without a marker a short buffer must keep filling the cushion")
	}

	p.MarkTurnEnd("sess-1", 3, "complete")
	if len(*got) != 0 {
		t.Fatalf("completion fired before the tail drained: %+v", *got)
	}
	emit, end := p.step(frame, time.Now())
	if !emit {
		t.Fatal("the tail frame must be emitted, not stranded in the buffer")
	}
	if frame[playoutFrameSamp-1] != 0 {
		t.Fatal("the partial final frame must be zero-padded")
	}
	if end == nil {
		t.Fatal("marker should come due once the buffer is drained")
	}
	p.fire(end, "")
	if len(*got) != 1 || (*got)[0] != (completion{"sess-1", 3, "complete"}) {
		t.Fatalf("got %+v", *got)
	}
	if len(p.buf) != 0 {
		t.Fatalf("buffer not drained: %d samples left", len(p.buf))
	}
}

// A full turn's worth of audio drains frame by frame, and the completion lands
// on the frame that empties the buffer — not before it.
func TestPendingEndCompletesOnlyAfterFullDrain(t *testing.T) {
	p, got := newTestPlayout(0)
	p.buf = make([]int16, playoutFrameSamp*3)
	p.MarkTurnEnd("sess-1", 9, "complete")
	frame := make([]int16, playoutFrameSamp)

	for i := 0; i < 3; i++ {
		if emit, end := p.step(frame, time.Now()); !emit || end != nil {
			t.Fatalf("frame %d: emit=%v end=%v; want a plain audio frame", i, emit, end)
		}
	}
	if len(*got) != 0 {
		t.Fatalf("completed while audio was still buffered: %+v", *got)
	}
	emit, end := p.step(frame, time.Now())
	if emit {
		t.Fatal("an empty tail should not emit a silent frame")
	}
	p.fire(end, "")
	if len(*got) != 1 || (*got)[0].turnID != 9 {
		t.Fatalf("got %+v, want turn 9 complete", *got)
	}
}

// Barge-in throws the audio away, so the turn is over for the listener. The
// marker must be reported as interrupted rather than silently dropped.
func TestFlushCompletesPendingEndAsInterrupted(t *testing.T) {
	p, got := newTestPlayout(0)
	p.buf = make([]int16, playoutFrameSamp*2)
	p.MarkTurnEnd("sess-1", 4, "complete")
	p.Flush()
	if len(*got) != 1 || (*got)[0] != (completion{"sess-1", 4, "interrupted"}) {
		t.Fatalf("got %+v, want turn 4 interrupted", *got)
	}
	if len(p.buf) != 0 || p.playing {
		t.Fatal("flush must empty the buffer and re-arm the prebuffer")
	}
}

// Flush with no marker pending is the ordinary barge-in path: the orchestrator
// sends STOP_PLAYBACK first and AGENT_TURN_ENDED(interrupted) after cancelling,
// so the completion comes from that later marker, not from the flush.
func TestFlushWithoutPendingEndReportsNothing(t *testing.T) {
	p, got := newTestPlayout(0)
	p.buf = make([]int16, playoutFrameSamp)
	p.Flush()
	if len(*got) != 0 {
		t.Fatalf("got %+v, want no completion", *got)
	}
	p.MarkTurnEnd("sess-1", 5, "interrupted")
	if len(*got) != 1 || (*got)[0].reason != "interrupted" {
		t.Fatalf("got %+v, want the post-cancel marker to complete the turn", *got)
	}
}

// A clause shorter than the prebuffer cushion must still play. The LLM streams
// clause by clause, so a short sentence ("Okay.") is often all that arrives
// before the next synthesis gap — and the cushion it would be waiting on can
// only be filled by audio that is still seconds away. Re-arming the cushion on
// every underrun stranded exactly those clauses: the first (long) clause of a
// turn played and the short ones after it did not.
func TestShortClauseIsNotStrandedByThePrebufferCushion(t *testing.T) {
	cushion := playoutRate / 5 // 200 ms
	p, _ := newTestPlayout(cushion)
	p.maxStall = 400 * time.Millisecond
	frame := make([]int16, playoutFrameSamp)
	now := time.Now()

	emitted := 0
	tick := func(n int) {
		for i := 0; i < n; i++ {
			now = now.Add(playoutFrameMS * time.Millisecond)
			if emit, _ := p.step(frame, now); emit {
				emitted += playoutFrameSamp
			}
		}
	}
	write := func(ms int) {
		p.mu.Lock()
		p.buf = append(p.buf, make([]int16, ms*playoutRate/1000)...)
		p.mu.Unlock()
	}

	write(1000) // clause 1: comfortably over the cushion
	tick(60)
	if emitted != playoutRate || len(p.buf) != 0 {
		t.Fatalf("clause 1: emitted=%d buf=%d, want a clean 1 s drain", emitted, len(p.buf))
	}

	// Clause 2 is 150 ms — below the 200 ms cushion, so it can never satisfy it.
	write(150)
	tick(10) // 200 ms: within the stall ceiling, still holding for the cushion
	if emitted != playoutRate {
		t.Fatalf("clause 2 played before the stall ceiling: emitted=%d", emitted)
	}
	// Past EDGE_PLAYOUT_MAX_STALL_MS the clause plays rather than being stranded.
	// The sub-frame remainder re-arms the ceiling once more, so allow for that:
	// the contract is that it drains in bounded time, not on the very next tick.
	for i := 0; i < 100 && len(p.buf) > 0; i++ {
		tick(1)
	}
	if len(p.buf) != 0 {
		t.Fatalf("STRANDED: %d samples of a short clause never played", len(p.buf))
	}
	if emitted <= playoutRate {
		t.Fatalf("short clause produced no audio: emitted=%d", emitted)
	}
}

// The stall ceiling must not defeat the cushion in the normal case: audio that
// is still arriving gets the full prebuffer before playback starts.
func TestCushionStillAppliesWhileAudioIsArriving(t *testing.T) {
	cushion := playoutRate / 5
	p, _ := newTestPlayout(cushion)
	p.maxStall = 400 * time.Millisecond
	frame := make([]int16, playoutFrameSamp)
	now := time.Now()

	// 20 ms arrives each tick: the cushion fills well inside the stall ceiling.
	for i := 0; i < 5; i++ {
		p.mu.Lock()
		p.buf = append(p.buf, make([]int16, playoutFrameSamp)...)
		p.mu.Unlock()
		now = now.Add(playoutFrameMS * time.Millisecond)
		if emit, _ := p.step(frame, now); emit {
			t.Fatalf("tick %d: played before the cushion filled", i)
		}
	}
	p.mu.Lock()
	p.buf = append(p.buf, make([]int16, cushion)...)
	p.mu.Unlock()
	now = now.Add(playoutFrameMS * time.Millisecond)
	if emit, _ := p.step(frame, now); !emit {
		t.Fatal("playback should start once the cushion is satisfied")
	}
}
