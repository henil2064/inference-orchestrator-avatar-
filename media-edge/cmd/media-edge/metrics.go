package main

import (
	"context"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Prometheus metrics for the media edge. Exposed on a plain-HTTP -metrics-addr,
// separate from the HTTPS signaling port so Prometheus needn't deal with the
// self-signed cert.
var (
	mActiveSessions = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "edge_active_sessions", Help: "Voice sessions currently forwarding audio",
	})
	// Offered load: incremented as the very first thing in wsHandler, before
	// admission control or the WebSocket upgrade even run. edge_sessions_total
	// (below) only counts once a session has an audio track flowing -- far
	// too late to see a rejected or failed-to-upgrade attempt. Reconcile:
	// attempts ~= rejected + upgrade_errors + sessions_total (+ WebRTC/ICE
	// negotiation failures between upgrade and the first audio track, which
	// have no dedicated counter here).
	mConnectionAttempts = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_connection_attempts_total",
		Help: "WebSocket connection attempts reaching the edge, before admission control or upgrade",
	})
	mConnectionRejected = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_connection_rejected_total",
		Help: "Connection attempts rejected immediately by EDGE_MAX_SESSIONS admission control",
	})
	mUpgradeErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_ws_upgrade_errors_total",
		Help: "WebSocket upgrade failures for attempts that passed admission control",
	})
	mSessionsTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_sessions_total", Help: "Voice sessions started",
	})
	mSessionDuration = promauto.NewHistogram(prometheus.HistogramOpts{
		Name: "edge_session_duration_seconds", Help: "Session lifetime",
		Buckets: prometheus.ExponentialBuckets(1, 2, 9), // 1s .. ~256s
	})
	mSpeechStarted = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_speech_started_total", Help: "VAD speech-start (barge-in) events",
	})
	mTurns = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_turns_total", Help: "VAD endpointer firings (turn boundaries)",
	})
	mOpusErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_opus_decode_errors_total", Help: "Opus frames that failed to decode",
	})
	mVADErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_vad_inference_errors_total", Help: "Silero VAD inference errors",
	})
	mGRPCSendErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_grpc_send_errors_total", Help: "Audio-frame sends to the orchestrator that failed",
	})
	mPlayoutFrames = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_playout_frames_total", Help: "Opus frames written to the outbound agent track",
	})
	mPlayoutEncodeErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_playout_encode_errors_total", Help: "Opus encode failures during playout",
	})
	mPlayoutUnderruns = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_playout_underruns_total", Help: "Playout buffer ran dry mid-utterance and re-buffered",
	})
	mPlayoutStalls = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_playout_stall_releases_total",
		Help: "Buffered clause played without a full prebuffer cushion because it had been held for EDGE_PLAYOUT_MAX_STALL_MS",
	})
	mPlayoutDropped = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_playout_dropped_frames_total", Help: "Playout frames dropped because TTS outran real-time playback",
	})
	mAvatarSessions = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_avatar_sessions_total", Help: "Voice sessions that got an avatar video stream",
	})
	mAvatarErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_avatar_errors_total", Help: "Avatar streams that failed to open or ended early; the session continued audio-only",
	})
	mAvatarFrames = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_avatar_frames_total", Help: "Video frames written to the outbound avatar track",
	})
	// Sustained growth means the lip-sync delay is shorter than the avatar
	// service's real latency: raise EDGE_AVATAR_RENDER_MARGIN_MS.
	mAvatarLateFrames = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_avatar_late_frames_total", Help: "Avatar frames that arrived after the audio they depict had already played",
	})
	mAvatarAudioDropped = promauto.NewCounter(prometheus.CounterOpts{
		Name: "edge_avatar_audio_dropped_total", Help: "Timeline audio frames not sent because the avatar stream was backed up",
	})
)

// serveMetrics exposes /metrics on addr in a background goroutine.
func serveMetrics(addr string, shutdownCtx context.Context, ready <-chan struct{}) {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler(shutdownCtx, ready))
	mux.Handle("/metrics", promhttp.Handler())
	go func() {
		if err := http.ListenAndServe(addr, mux); err != nil {
			errorf("metrics server on %s stopped: %v", addr, err)
		}
	}()
}
