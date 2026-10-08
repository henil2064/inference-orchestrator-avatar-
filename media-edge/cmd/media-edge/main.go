// Command media-edge is the phase-1 WebRTC media edge: it serves a small
// browser page, terminates a WebRTC connection from it, decodes the inbound
// Opus to 16 kHz mono PCM, and streams it to the Python orchestrator over
// gRPC bidi (see docs/PRD.md §3.1, §8.2).
package main

import (
	"bufio"
	"context"
	"embed"
	"encoding/json"
	"flag"
	"io/fs"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"mondee/media-edge/internal/certs"
	"mondee/media-edge/internal/profile"
	"mondee/media-edge/internal/silero"
)

// loadDotenv reads KEY=VALUE lines from a .env in the working dir into the process
// environment, WITHOUT overriding vars already set in the real environment (so a
// shell export or a CLI flag still wins). Best-effort: missing file == no-op.
func loadDotenv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if i := strings.IndexByte(line, '#'); i >= 0 {
			line = line[:i]
		}
		k, v, ok := strings.Cut(strings.TrimSpace(line), "=")
		if !ok {
			continue
		}
		k = strings.TrimSpace(k)
		v = strings.Trim(strings.TrimSpace(v), `'"`)
		if _, exists := os.LookupEnv(k); !exists {
			os.Setenv(k, v)
		}
	}
}

func envOr(key, def string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return def
}

func envBool(key string, def bool) bool {
	if v, ok := os.LookupEnv(key); ok {
		return v == "1" || strings.EqualFold(v, "true")
	}
	return def
}

// version is set at build time via -ldflags "-X main.version=vX.Y.Z"
// (see scripts/build-media-edge.sh). "dev" means a plain `go run`/`go build`.
var version = "dev"

//go:embed web
var webFS embed.FS

// sessionWg tracks active forwardTrack goroutines so shutdown can wait for
// them to drain before destroying the ONNX environment.
var sessionWg sync.WaitGroup

func main() {
	// .env (working dir) feeds the flag defaults below; a CLI flag still overrides.
	loadDotenv(".env")
	configureTerminal()
	addr := flag.String("addr", envOr("EDGE_ADDR", ":8443"), "listen address, e.g. 0.0.0.0:8443")
	profilePath := flag.String("profile", envOr("EDGE_PROFILE", ""), "path to a JSON profile file (default: built-in 'voice' profile)")
	orchAddr := flag.String("orchestrator", envOr("EDGE_ORCHESTRATOR", "localhost:50051"), "orchestrator gRPC address to forward decoded PCM to")
	certDir := flag.String("cert-dir", envOr("EDGE_CERT_DIR", "certs"), "directory to cache the self-signed TLS cert/key in")
	noTLS := flag.Bool("no-tls", envBool("EDGE_NO_TLS", false), "serve plain HTTP instead of HTTPS (mic access will fail on iOS Safari / non-localhost origins)")
	onnxLib := flag.String("onnx-lib", envOr("EDGE_ONNX_LIB", ""), "path to the onnxruntime shared library (default: third_party/onnxruntime/libonnxruntime.so from setup_ubuntu.sh)")
	vadModel := flag.String("vad-model", envOr("EDGE_VAD_MODEL", ""), "path to silero_vad.onnx (default: models/silero_vad.onnx — Silero is ON by default, falls back to energy RMS if the onnx lib can't load)")
	metricsAddr := flag.String("metrics-addr", envOr("EDGE_METRICS_ADDR", ":9101"), "plain-HTTP address to serve Prometheus /metrics on")
	avatarAddr := flag.String("avatar", envOr("EDGE_AVATAR_TARGET", ""), "avatar service gRPC address, e.g. localhost:50056 (empty: audio-only, no video track)")
	flag.Parse()

	infof("media-edge %s", version)

	prof, err := profile.Load(*profilePath)
	if err != nil {
		fatalf("load profile: %v", err)
	}
	// Silero is the default VAD. Precedence: explicit --vad-model > profile's
	// vad_model_path > vendored default. Same for the onnxruntime lib path.
	if *vadModel != "" {
		prof.VADModelPath = *vadModel // CLI override so you don't need a profile JSON
	} else if prof.VADModelPath == "" {
		prof.VADModelPath = "models/silero_vad.onnx" // vendored default -> Silero on
	}
	onnxLibPath := *onnxLib
	if onnxLibPath == "" {
		onnxLibPath = "third_party/onnxruntime/libonnxruntime.so" // setup_ubuntu.sh puts it here
	}
	infof("using profile %q: %+v", prof.Name, prof)

	// Best-effort Silero: if the onnxruntime lib can't load, fall back to the energy
	// RMS gate with a loud warning rather than dying — keeps a zero-native-dep
	// checkout runnable (DECISION D-5). VAD_DEBUG (on by default) shows which is live.
	onnxReady := false
	if prof.VADModelPath != "" {
		if err := silero.Init(onnxLibPath); err != nil {
			warnf("VAD: Silero requested (%s) but onnxruntime failed to load (%s: %v) — "+
				"falling back to energy RMS gate. Run media-edge/setup_ubuntu.sh or pass --onnx-lib.",
				prof.VADModelPath, onnxLibPath, err)
			prof.VADModelPath = ""
		} else {
			m, err := silero.NewModel(16000, prof.VADModelPath)
			if err != nil {
				warnf("VAD: Silero requested (%s) but model failed to load (%v) — falling back to energy RMS gate.",
					prof.VADModelPath, err)
				prof.VADModelPath = ""
			} else {
				m.Destroy()
				onnxReady = true
			}
			if prof.VADModelPath != "" {
				infof("VAD: Silero neural (model=%s runtime=%s) — level is speech probability 0..1", prof.VADModelPath, onnxLibPath)
			}
		}
	}
	if prof.VADModelPath == "" {
		infof("VAD: energy RMS gate")
	}

	orch, err := dialOrchestrator(*orchAddr)
	if err != nil {
		fatalf("dial orchestrator %s: %v", *orchAddr, err)
	}
	defer orch.Close()
	infof("forwarding decoded PCM to orchestrator at %s", *orchAddr)

	var avatar *grpc.ClientConn
	if *avatarAddr != "" {
		avatar, err = dialAvatar(*avatarAddr)
		if err != nil {
			fatalf("dial avatar %s: %v", *avatarAddr, err)
		}
		defer avatar.Close()
		infof("avatar video from %s", *avatarAddr)
	}

	webRoot, err := fs.Sub(webFS, "web")
	if err != nil {
		fatalf("embed web assets: %v", err)
	}

	// Graceful shutdown: catch SIGINT/SIGTERM, drain active sessions, then
	// tear down ONNX and the gRPC connection in the right order. Created before
	// the handler so each session can watch it and drain itself on shutdown.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ready := make(chan struct{})
	serveMetrics(*metricsAddr, ctx, ready)
	infof("metrics on http://%s/metrics", *metricsAddr)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler(ctx, ready))
	mux.Handle("/", http.FileServer(http.FS(webRoot)))
	mux.HandleFunc("/ws", wsHandler(ctx, prof, orch, avatar))
	// The page asks before offering to receive video, so a browser talking to
	// an edge without an avatar sends the same audio-only offer as before.
	mux.HandleFunc("/config.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_ = json.NewEncoder(w).Encode(map[string]bool{"avatar": avatar != nil})
	})

	ips := localIPs()
	scheme := "https"
	if *noTLS {
		scheme = "http"
	}
	infof("listening on %s — open one of these in a browser:", *addr)
	for _, ip := range ips {
		infof("  %s://%s%s", scheme, ip, portSuffix(*addr))
	}

	srv := &http.Server{
		Addr:    *addr,
		Handler: mux,
	}

	startupBanner("MEDIA EDGE  |  READY", []string{
		strings.ToUpper(scheme) + "     " + *addr + func() string {
			if !*noTLS {
				return " (self-signed certificate)"
			}
			return ""
		}(),
		"Gateway   " + *orchAddr,
		"VAD       " + vadDescription(prof.VADModelPath),
		"Avatar    " + func() string {
			if avatar == nil {
				return "off"
			}
			return *avatarAddr
		}(),
		"Metrics   http://" + *metricsAddr + "/metrics",
		"Details   LOG_LEVEL=INFO  |  VAD_DEBUG=1",
	})

	go func() {
		<-ctx.Done()
		infof("shutting down — waiting for active sessions to drain...")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			errorf("http shutdown: %v", err)
		}
	}()

	if *noTLS {
		ln, err := net.Listen("tcp", *addr)
		if err != nil {
			fatalf("listen %s: %v", *addr, err)
		}
		close(ready)
		if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
			errorf("HTTP server stopped: %v", err)
		}
	} else {
		certFile, keyFile, err := certs.Ensure(*certDir, ips)
		if err != nil {
			fatalf("generate TLS cert: %v", err)
		}
		ln, err := net.Listen("tcp", *addr)
		if err != nil {
			fatalf("listen %s: %v", *addr, err)
		}
		infof("self-signed cert — the browser will warn once; accept it to continue (iOS: tap 'Show Details' -> 'visit this website')")
		close(ready)
		if err := srv.ServeTLS(ln, certFile, keyFile); err != nil && err != http.ErrServerClosed {
			errorf("HTTPS server stopped: %v", err)
		}
	}

	// Wait for all active forwardTrack goroutines to finish sending
	// SESSION_CLOSE before destroying the ONNX runtime. Each session's shutdown
	// watcher (in wsHandler) cancels its track so this drains promptly; the
	// timeout is a backstop against a session that won't die.
	drained := make(chan struct{})
	go func() { sessionWg.Wait(); close(drained) }()
	select {
	case <-drained:
		infof("all sessions drained")
	case <-time.After(30 * time.Second):
		warnf("session drain timed out after 30s — exiting anyway")
	}

	if onnxReady {
		silero.Destroy()
	}
	infof("media-edge stopped")
}

func vadDescription(modelPath string) string {
	if modelPath == "" {
		return "energy RMS fallback"
	}
	return "Silero neural"
}

// localIPs returns non-loopback IPv4 addresses of this machine, so a phone
// or laptop on the same network knows what to type into its browser.
func localIPs() []string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return nil
	}
	var ips []string
	for _, a := range addrs {
		ipNet, ok := a.(*net.IPNet)
		if !ok || ipNet.IP.IsLoopback() || ipNet.IP.IsLinkLocalUnicast() {
			continue
		}
		if v4 := ipNet.IP.To4(); v4 != nil {
			ips = append(ips, v4.String())
		}
	}
	return ips
}

// portSuffix turns a listen address like ":8080" or "0.0.0.0:8080" into
// ":8080" for building a browser URL against a specific host IP.
func portSuffix(addr string) string {
	if i := strings.LastIndex(addr, ":"); i >= 0 {
		return addr[i:]
	}
	return addr
}
