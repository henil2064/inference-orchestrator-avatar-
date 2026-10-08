package main

import (
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/pion/opus"
	"github.com/pion/opus/pkg/oggreader"

	"mondee/media-edge/internal/silero"
)

// BenchmarkOpusDecode measures pure-Go Opus decode of real frames from pion's
// test vector — the other per-session CPU cost besides VAD. Reports ns per
// decoded packet; the codebase decodes one per RTP packet (10 ms at ptime 10).
func BenchmarkOpusDecode(b *testing.B) {
	ogg := os.Getenv("OPUS_OGG")
	if ogg == "" {
		b.Skip("set OPUS_OGG to a .ogg opus vector to run")
	}
	f, err := os.Open(ogg)
	if err != nil {
		b.Skipf("no opus test vector: %v", err)
	}
	defer f.Close()
	r, _, err := oggreader.NewWith(f)
	if err != nil {
		b.Skipf("ogg parse: %v", err)
	}
	var packets [][]byte
	for {
		pages, _, e := r.ParseNextPage()
		if e != nil {
			break
		}
		packets = append(packets, pages...)
	}
	if len(packets) == 0 {
		b.Skip("no opus packets in vector")
	}
	dec, err := opus.NewDecoderWithOutput(16000, 1)
	if err != nil {
		b.Fatal(err)
	}
	pcm := make([]int16, pcmFrameCap)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pkt := packets[i%len(packets)]
		if len(pkt) == 0 {
			continue
		}
		_, _ = dec.DecodeToInt16(pkt, pcm)
	}
}

// These benchmarks measure the per-session hot-path cost so we can reason about
// how many concurrent sessions the box holds. Run:
//   go test ./cmd/media-edge -run x -bench 'VAD|PCM|Silero' -benchmem
// Silero benchmarks skip automatically if the onnxruntime lib / model are absent.

func modelAndLib() (model, lib string, ok bool) {
	model = filepath.Join("..", "..", "models", "silero_vad.onnx")
	// Windows dev box uses the vendored DLL; setup_ubuntu.sh uses the .so.
	for _, l := range []string{
		filepath.Join("..", "..", "third_party", "onnxruntime", "onnxruntime.dll"),
		filepath.Join("..", "..", "third_party", "onnxruntime", "libonnxruntime.so"),
	} {
		if _, err := os.Stat(l); err == nil {
			lib = l
			break
		}
	}
	if _, err := os.Stat(model); err != nil || lib == "" {
		return "", "", false
	}
	return model, lib, true
}

// BenchmarkVADEnergyUpdate is the cheap default path: RMS over one 10 ms frame.
func BenchmarkVADEnergyUpdate(b *testing.B) {
	v := testVAD()
	frame := fill(160, 1500)
	now := time.Now()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		now = now.Add(10 * time.Millisecond)
		v.update(frame, now)
	}
}

// BenchmarkPCMPack measures int16->LE packing for one 10 ms frame.
func BenchmarkPCMPack(b *testing.B) {
	frame := fill(160, 1500)
	dst := make([]byte, 160*2)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		int16sToLEInto(frame, dst)
	}
}

// BenchmarkSileroPredict is the real cost driver: one ONNX inference per 512-
// sample (32 ms) window, single-threaded (as configured in newVAD).
func BenchmarkSileroPredict(b *testing.B) {
	model, lib, ok := modelAndLib()
	if !ok {
		b.Skip("silero model / onnxruntime lib not found")
	}
	if err := silero.Init(lib); err != nil {
		b.Skipf("onnxruntime init failed: %v", err)
	}
	defer silero.Destroy()
	m, err := silero.NewModel(16000, model)
	if err != nil {
		b.Fatalf("load model: %v", err)
	}
	defer m.Destroy()
	m.Reset()
	win := make([]float32, sileroWindow)
	for i := range win {
		win[i] = 0.1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Predict(win); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSileroConcurrent runs P independent Silero sessions in lockstep,
// each doing one Predict per iteration, to expose how inference latency
// degrades under core contention. Set with -cpu or read the per-op time at
// different GOMAXPROCS. Reports the aggregate; divide by P for per-session.
func BenchmarkSileroConcurrent(b *testing.B) {
	model, lib, ok := modelAndLib()
	if !ok {
		b.Skip("silero model / onnxruntime lib not found")
	}
	if err := silero.Init(lib); err != nil {
		b.Skipf("onnxruntime init failed: %v", err)
	}
	defer silero.Destroy()

	for _, P := range []int{1, 4, 8, 16, 32, 50} {
		b.Run("sessions="+itoa(P), func(b *testing.B) {
			models := make([]*silero.Model, P)
			for i := range models {
				m, err := silero.NewModel(16000, model)
				if err != nil {
					b.Fatalf("load model %d: %v", i, err)
				}
				m.Reset()
				models[i] = m
			}
			defer func() {
				for _, m := range models {
					m.Destroy()
				}
			}()
			win := make([]float32, sileroWindow)
			for i := range win {
				win[i] = 0.1
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var wg sync.WaitGroup
				wg.Add(P)
				for p := 0; p < P; p++ {
					go func(m *silero.Model) {
						defer wg.Done()
						_, _ = m.Predict(win)
					}(models[p])
				}
				wg.Wait()
			}
		})
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// TestReportCores just records the box the numbers were taken on.
func TestReportCores(t *testing.T) {
	t.Logf("GOMAXPROCS=%d NumCPU=%d", runtime.GOMAXPROCS(0), runtime.NumCPU())
}
