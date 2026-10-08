package main

import (
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"mondee/media-edge/internal/edgepb"
	"mondee/media-edge/internal/profile"
)

func fill(n int, amp int16) []int16 {
	s := make([]int16, n)
	for i := range s {
		s[i] = amp
	}
	return s
}

func testVAD() *vad {
	return &vad{enter: 900, exit: 500, attack: 50 * time.Millisecond, hangover: 100 * time.Millisecond}
}

// TestVADTransitions walks one utterance: silence -> sustained speech onset
// (only after the attack window) -> sustained speech (no repeat) -> brief gap
// under hangover (no stop) -> silence past hangover (stop).
func TestVADTransitions(t *testing.T) {
	v := testVAD()
	t0 := time.Now()
	loud, quiet := fill(160, 2000), fill(160, 0)

	if started, _ := v.update(quiet, t0); started {
		t.Fatal("silence must not start speech")
	}
	if started, _ := v.update(loud, t0); started {
		t.Fatal("first loud frame must not start speech before attack elapses")
	}
	if started, _ := v.update(loud, t0.Add(60*time.Millisecond)); !started {
		t.Fatal("sustained loud past attack must start speech")
	}
	if started, stopped := v.update(loud, t0.Add(70*time.Millisecond)); started || stopped {
		t.Fatal("sustained speech must emit no transition")
	}
	if _, stopped := v.update(quiet, t0.Add(120*time.Millisecond)); stopped {
		t.Fatal("must not stop before hangover elapses")
	}
	if _, stopped := v.update(quiet, t0.Add(250*time.Millisecond)); !stopped {
		t.Fatal("must stop once hangover elapses")
	}
}

// TestVADRejectsSpike is the flapping fix: a lone noise spike above enter, not
// sustained through the attack window, must never declare speech.
func TestVADRejectsSpike(t *testing.T) {
	v := testVAD()
	t0 := time.Now()
	loud, quiet := fill(160, 2000), fill(160, 0)

	if started, _ := v.update(loud, t0); started {
		t.Fatal("single spike must not start speech")
	}
	if started, _ := v.update(quiet, t0.Add(10*time.Millisecond)); started {
		t.Fatal("spike followed by silence must not start speech")
	}
	// A later sustained burst still works (attack run reset cleanly).
	v.update(loud, t0.Add(100*time.Millisecond))
	if started, _ := v.update(loud, t0.Add(160*time.Millisecond)); !started {
		t.Fatal("sustained speech after a spike must still start")
	}
}

// TestVADHysteresis: once speaking, a dip between exit and enter must not end
// the utterance — only a fall below exit, held past hangover, does.
func TestVADHysteresis(t *testing.T) {
	v := testVAD()
	t0 := time.Now()
	loud, mid := fill(160, 2000), fill(160, 700) // 700 is < enter(900) but > exit(500)

	v.update(loud, t0)
	v.update(loud, t0.Add(60*time.Millisecond)) // now speaking
	if _, stopped := v.update(mid, t0.Add(300*time.Millisecond)); stopped {
		t.Fatal("level above exit must keep the utterance alive")
	}
}

// TestVADSileroWindowing checks the Silero path's buffering: frames of any size
// accumulate into exact 512-sample windows, each fed to predict() once, in order,
// with no samples lost or duplicated across the compaction. A stub predictor
// stands in for the ONNX model (unavailable in unit tests).
func TestVADSileroWindowing(t *testing.T) {
	var windows int
	var seen float32 // running sum of every sample the predictor received
	v := &vad{
		attack: 0, hangover: 0, speechProb: 0.5, silenceProb: 0.35,
		predict: func(win []float32) (float32, error) {
			if len(win) != sileroWindow {
				t.Fatalf("predict got %d samples, want %d", len(win), sileroWindow)
			}
			for _, s := range win {
				seen += s
			}
			windows++
			return 0, nil
		},
	}

	// Feed 3 frames of 200 samples (=600), all value 1 -> 1/32768 each.
	// 600 samples => one full 512 window, 88 left over.
	for range 3 {
		v.update(fill(200, 1), time.Now())
	}
	if windows != 1 {
		t.Fatalf("600 samples => want 1 window, got %d", windows)
	}
	// One more 200-sample frame: 88 + 200 = 288, still < 512, no new window.
	v.update(fill(200, 1), time.Now())
	if windows != 1 {
		t.Fatalf("288 buffered => still 1 window, got %d", windows)
	}
	// A 512-sample frame: 288 + 512 = 800 => one more window, 288 left.
	v.update(fill(512, 1), time.Now())
	if windows != 2 {
		t.Fatalf("want 2 windows, got %d", windows)
	}
	if len(v.acc) != 288 {
		t.Fatalf("leftover must be 288, got %d", len(v.acc))
	}
	// Every predicted sample was 1/32768; 2 windows * 512 = 1024 of them.
	if want := float32(1024) / 32768; seen != want {
		t.Fatalf("predictor saw sum %v, want %v (samples lost/duplicated?)", seen, want)
	}
}

func TestVADSileroFlushesPartialWindow(t *testing.T) {
	var windows int
	v := &vad{
		attack: 0, hangover: 0, speechProb: 0.5, silenceProb: 0.35,
		predict: func(win []float32) (float32, error) {
			windows++
			if len(win) != sileroWindow {
				t.Fatalf("predict got %d samples, want %d", len(win), sileroWindow)
			}
			for i := 400; i < len(win); i++ {
				if win[i] != 0 {
					t.Fatalf("flush padding sample %d = %v, want 0", i, win[i])
				}
			}
			return 0, nil
		},
	}

	v.update(fill(400, 1), time.Now())
	if windows != 0 {
		t.Fatalf("partial window must stay buffered, got %d predictions", windows)
	}
	v.flush(time.Now())
	if windows != 1 {
		t.Fatalf("flush must process one padded window, got %d predictions", windows)
	}
	if len(v.acc) != 0 {
		t.Fatalf("flush must drain accumulator, got %d samples", len(v.acc))
	}
}

func TestVADUsesMediaClockNotArrivalDelay(t *testing.T) {
	v := testVAD()
	clock := newMediaClock()
	loud := fill(160, 2000)

	started, _ := v.update(loud, clock.Advance(len(loud)))
	if started {
		t.Fatal("first 10 ms loud packet must not start before attack")
	}
	started, _ = v.update(loud, clock.Advance(len(loud)))
	if started {
		t.Fatal("20 ms of speech must not start even if wall-clock scheduling stalls")
	}
	sawStart := false
	for i := 0; i < 8; i++ {
		started, _ = v.update(loud, clock.Advance(len(loud)))
		sawStart = sawStart || started
	}
	if !sawStart {
		t.Fatal("sustained media-time speech past attack must start")
	}
}

func TestBloatingWAVConcurrentVADSessions(t *testing.T) {
	pcm := loadBloatingPCM16(t)
	if len(pcm) < 160 {
		t.Fatal("bloating.wav produced no 16 kHz PCM")
	}

	const clients = 50
	var wg sync.WaitGroup
	errs := make(chan string, clients)
	for id := 0; id < clients; id++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			v := testVAD()
			clock := newMediaClock()
			starts, stops := 0, 0
			for off := 0; off < len(pcm); off += 160 {
				end := off + 160
				if end > len(pcm) {
					end = len(pcm)
				}
				started, stopped := v.update(pcm[off:end], clock.Advance(end-off))
				if started {
					starts++
				}
				if stopped {
					stops++
				}
			}
			if starts == 0 {
				errs <- "no speech-start events"
				return
			}
			if stops == 0 {
				errs <- "no endpoint events"
				return
			}
			if len(v.acc) != 0 {
				errs <- "energy VAD unexpectedly buffered samples"
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

func loadBloatingPCM16(t *testing.T) []int16 {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "pipeline_test", "recordings", "bloating.wav"))
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("fixture pipeline_test/recordings/bloating.wav not present (gitignored); skipping")
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < 44 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		t.Fatal("not a RIFF/WAVE file")
	}
	var channels, bits, sampleRate uint16
	var samples []int16
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		off += 8
		if off+size > len(data) {
			t.Fatal("truncated WAV chunk")
		}
		switch id {
		case "fmt ":
			if size < 16 {
				t.Fatal("short fmt chunk")
			}
			format := binary.LittleEndian.Uint16(data[off : off+2])
			channels = binary.LittleEndian.Uint16(data[off+2 : off+4])
			rate := binary.LittleEndian.Uint32(data[off+4 : off+8])
			bits = binary.LittleEndian.Uint16(data[off+14 : off+16])
			if format != 1 {
				t.Fatalf("want PCM WAV, got format %d", format)
			}
			if rate != 48000 {
				t.Fatalf("want 48 kHz fixture, got %d", rate)
			}
			sampleRate = uint16(rate / 1000)
		case "data":
			if channels != 1 || bits != 16 || sampleRate != 48 {
				t.Fatalf("want mono PCM16 48 kHz, got channels=%d bits=%d rate_khz=%d", channels, bits, sampleRate)
			}
			for i := off; i+5 < off+size; i += 6 {
				samples = append(samples, int16(binary.LittleEndian.Uint16(data[i:i+2])))
			}
		}
		off += size
		if size%2 == 1 {
			off++
		}
	}
	return samples
}

// --- New tests for the fixes ---

// mockStream is a test double for bridgeStream that records sent events and
// can be configured to fail.
type mockStream struct {
	sent   []*edgepb.ClientEvent
	failN  int // fail on the Nth Send (1-indexed), 0 = never fail
	callN  int
	sendFn func(*edgepb.ClientEvent) error
}

func (m *mockStream) Send(ev *edgepb.ClientEvent) error {
	m.callN++
	if m.sendFn != nil {
		return m.sendFn(ev)
	}
	if m.failN > 0 && m.callN == m.failN {
		return errors.New("mock send failure")
	}
	m.sent = append(m.sent, ev)
	return nil
}

func TestSendControlPropagatesError(t *testing.T) {
	ms := &mockStream{failN: 1}
	err := sendControl(ms, "sess-1", edgepb.Control_USER_SPEECH_STARTED, time.Now(), map[string]string{})
	if err == nil {
		t.Fatal("sendControl should return error on send failure")
	}
	if len(ms.sent) != 0 {
		t.Fatalf("failed send should not be recorded, got %d", len(ms.sent))
	}
}

func TestSendControlSuccess(t *testing.T) {
	ms := &mockStream{}
	err := sendControl(ms, "sess-1", edgepb.Control_SESSION_OPEN, time.Now(), map[string]string{"k": "v"})
	if err != nil {
		t.Fatalf("sendControl should succeed: %v", err)
	}
	if len(ms.sent) != 1 {
		t.Fatalf("expected 1 sent event, got %d", len(ms.sent))
	}
	if ms.sent[0].GetControl().SessionId != "sess-1" {
		t.Fatalf("wrong session id: %s", ms.sent[0].GetControl().SessionId)
	}
	if ms.sent[0].GetControl().Type != edgepb.Control_SESSION_OPEN {
		t.Fatalf("wrong control type: %v", ms.sent[0].GetControl().Type)
	}
}

func TestMediaClockMonotonic(t *testing.T) {
	c := newMediaClock()
	t0 := c.Now()
	// Wall-clock offset should be close to time.Now()
	if diff := time.Since(t0); diff > time.Second {
		t.Fatalf("mediaClock start offset too large: %v", diff)
	}
	t1 := c.Advance(160) // 10ms at 16kHz
	t2 := c.Advance(160)
	if !t2.After(t1) {
		t.Fatal("mediaClock must be monotonically increasing")
	}
	expected := t0.Add(20 * time.Millisecond)
	if t2.Sub(expected) > time.Millisecond {
		t.Fatalf("expected ~%v, got %v (diff %v)", expected, t2, t2.Sub(expected))
	}
}

func TestMediaClockZeroSamples(t *testing.T) {
	c := newMediaClock()
	if c.Now().IsZero() {
		t.Fatal("mediaClock.Now() must not return zero time")
	}
}

func TestIsTimeoutTrue(t *testing.T) {
	err := &timeoutError{timeout: true}
	if !isTimeout(err) {
		t.Fatal("expected isTimeout to return true for timeout error")
	}
}

func TestIsTimeoutFalse(t *testing.T) {
	err := &timeoutError{timeout: false}
	if isTimeout(err) {
		t.Fatal("expected isTimeout to return false for non-timeout error")
	}
}

func TestIsTimeoutNilInterface(t *testing.T) {
	if isTimeout(errors.New("plain error")) {
		t.Fatal("expected isTimeout to return false for non-timeout error type")
	}
}

type timeoutError struct{ timeout bool }

func (e *timeoutError) Error() string   { return "timeout" }
func (e *timeoutError) Timeout() bool   { return e.timeout }

func TestInt16sToLEIntoCorrectness(t *testing.T) {
	samples := []int16{0, 1, -1, 32767, -32768}
	dst := make([]byte, len(samples)*2)
	result := int16sToLEInto(samples, dst)
	if len(result) != len(samples)*2 {
		t.Fatalf("expected %d bytes, got %d", len(samples)*2, len(result))
	}
	// Verify each sample is packed correctly
	for i, want := range samples {
		got := int16(result[2*i]) | int16(result[2*i+1])<<8
		if got != want {
			t.Fatalf("sample %d: want %d, got %d", i, want, got)
		}
	}
	// Verify the underlying buffer is reused (same pointer)
	result2 := int16sToLEInto([]int16{42}, dst)
	if &result2[0] != &dst[0] {
		t.Fatal("int16sToLEInto should reuse the provided buffer")
	}
}

func TestVadMetaReusesMap(t *testing.T) {
	v := testVAD()
	prof := profile.Profile{VADAttackMs: 90, VADHangoverMs: 1500}
	meta := make(map[string]string, 14)
	result1 := vadMeta(meta, v, prof, 100, 2000000, 50, 2)
	// result1 IS meta — mutate via result1, confirm meta reflects it
	result1["speech_duration_ms"] = "999"
	result2 := vadMeta(meta, v, prof, 200, 3000000, 100, 5)
	if result2["speech_duration_ms"] != "200" {
		t.Fatalf("vadMeta should overwrite values, got speech_duration_ms=%s", result2["speech_duration_ms"])
	}
	if result2["opus_decode_errors"] != "5" {
		t.Fatalf("vadMeta should overwrite values, got opus_decode_errors=%s", result2["opus_decode_errors"])
	}
}

func TestMetadataIntParsesCorrectly(t *testing.T) {
	m := map[string]string{"vad_process_ms": "42", "missing": ""}
	if got := metadataInt(m, "vad_process_ms"); got != 42 {
		t.Fatalf("expected 42, got %d", got)
	}
	if got := metadataInt(m, "missing"); got != 0 {
		t.Fatalf("expected 0 for missing key, got %d", got)
	}
}

func TestRtcEventsPendingBounds(t *testing.T) {
	e := &rtcEvents{}
	// Fill to the limit
	for i := 0; i < maxPendingEvents; i++ {
		e.send(signalMsg{Type: "agent_reply", Text: "ok"})
	}
	if len(e.pending) != maxPendingEvents {
		t.Fatalf("expected %d pending, got %d", maxPendingEvents, len(e.pending))
	}
	// One more should be dropped (logged as error)
	e.send(signalMsg{Type: "agent_reply", Text: "overflow"})
	if len(e.pending) != maxPendingEvents {
		t.Fatalf("overflow should not increase pending, got %d", len(e.pending))
	}
}

func TestRtcEventsByteBounds(t *testing.T) {
	e := &rtcEvents{}
	// Use messages under maxAgentEventText (16KB) but large enough that a
	// few of them exceed maxPendingBytes (64KB). Each JSON message adds ~36
	// bytes of structural overhead, so 10000-byte text ≈ 10036 bytes on wire.
	// 7 such messages ≈ 70KB > 65KB budget.
	big := make([]byte, 10000)
	for i := range big {
		big[i] = 'x'
	}
	sent := 0
	for i := 0; i < 10; i++ {
		e.send(signalMsg{Type: "agent_reply", Text: string(big)})
		sent++
	}
	// Not all 10 should have been buffered — byte limit should have kicked in.
	if len(e.pending) >= sent {
		t.Fatalf("expected byte limit to reject some messages, but all %d were buffered", sent)
	}
	if len(e.pending) == 0 {
		t.Fatal("at least some messages should have been buffered before the limit")
	}
}
