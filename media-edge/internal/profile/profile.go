// Package profile holds tunable audio-streaming parameters for the media
// edge. Values are loaded from a JSON file (or the built-in default) so they
// can be re-tuned without a rebuild.
package profile

import (
	"encoding/json"
	"fmt"
	"os"
)

// Profile is one named set of streaming tuning knobs.
type Profile struct {
	Name          string `json:"name"`
	Desc          string `json:"desc"`
	PtimeMs       int    `json:"ptime_ms"`
	BitrateBps    int    `json:"bitrate_bps"`
	FEC           bool   `json:"fec"`
	MaxPlaybackHz int    `json:"max_playback_hz"`
	MicLatencyMs  int    `json:"mic_latency_ms"`
	HighPriority  bool   `json:"high_priority"`
	WakeLock      bool   `json:"wake_lock"`
	TuneAnswer    bool   `json:"tune_answer"`
	RingMs        int    `json:"ring_ms"`
	PlayerMs      int    `json:"player_ms"`
	DeviceMs      int    `json:"device_ms"`
	TrimMs        int    `json:"trim_ms"`

	// VAD gate (RMS = int16 amplitude, 0..32767). A Schmitt trigger with
	// debounce, to stop background noise from flapping the state:
	//   - VADEnterRms: speech must reach this to *begin* an utterance (set it
	//     comfortably above the room-noise floor).
	//   - VADExitRms: once speaking, audio must fall below this (lower than
	//     enter — the hysteresis gap) to count toward silence. Prevents chatter
	//     when level hovers near one threshold.
	//   - VADAttackMs: level must hold >= enter this long before "started" fires
	//     — rejects single-frame noise spikes.
	//   - VADHangoverMs: level must hold < exit this long before "stopped" fires
	//     — bridges natural pauses mid-utterance.
	// All four are here (not hardcoded) because the right values depend on mic
	// gain and room noise — tune per deployment without a rebuild.
	VADEnterRms   float64 `json:"vad_enter_rms"`
	VADExitRms    float64 `json:"vad_exit_rms"`
	VADAttackMs   int     `json:"vad_attack_ms"`
	VADHangoverMs int     `json:"vad_hangover_ms"`

	// Silero neural VAD (opt-in). When VADModelPath is set, the edge runs the
	// Silero ONNX model instead of the RMS energy gate — it tells speech from
	// speech-level noise (music, other voices, fan hum) that energy cannot. The
	// same attack/hangover state machine still debounces, but on speech
	// probability instead of amplitude:
	//   - VADModelPath: path to silero_vad.onnx. Empty = use the RMS gate above.
	//     Requires the onnxruntime shared library too (media-edge --onnx-lib).
	//   - VADSpeechProb: probability to *begin* an utterance (Silero default 0.5).
	//   - VADSilenceProb: once speaking, prob must fall below this (< speech — the
	//     hysteresis gap) to count toward silence (Silero default ~0.35).
	// VADAttackMs / VADHangoverMs apply unchanged; VADEnterRms/VADExitRms are
	// ignored when a model is set.
	VADModelPath   string  `json:"vad_model_path"`
	VADSpeechProb  float64 `json:"vad_speech_prob"`
	VADSilenceProb float64 `json:"vad_silence_prob"`
}

// Voice is the tested "phone-call grade, low buffers" profile.
var Voice = Profile{
	Name: "voice", Desc: "low buffers, quality degraded to phone-call grade",
	PtimeMs: 10, BitrateBps: 24000, FEC: true, MaxPlaybackHz: 16000,
	MicLatencyMs: 5, HighPriority: true, WakeLock: true, TuneAnswer: true,
	RingMs: 60, PlayerMs: 10, DeviceMs: 10, TrimMs: 12,
	VADEnterRms: 900, VADExitRms: 500, VADAttackMs: 90, VADHangoverMs: 1500,
	VADSpeechProb: 0.5, VADSilenceProb: 0.35, // used only when VADModelPath is set
}

// Load reads a profile from a JSON file at path. An empty path (or a
// nonexistent file) returns Voice unchanged so the binary runs with no
// config present.
func Load(path string) (Profile, error) {
	p := Voice
	if path == "" {
		return p, p.Validate()
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return p, p.Validate()
	}
	if err != nil {
		return p, err
	}
	if err := json.Unmarshal(data, &p); err != nil {
		return p, err
	}
	return p, p.Validate()
}

func (p Profile) Validate() error {
	if p.PtimeMs <= 0 || p.PtimeMs > 120 {
		return fmt.Errorf("ptime_ms must be 1..120, got %d", p.PtimeMs)
	}
	if p.BitrateBps <= 0 {
		return fmt.Errorf("bitrate_bps must be positive, got %d", p.BitrateBps)
	}
	if p.VADAttackMs < 0 {
		return fmt.Errorf("vad_attack_ms must be >= 0, got %d", p.VADAttackMs)
	}
	if p.VADHangoverMs < 0 {
		return fmt.Errorf("vad_hangover_ms must be >= 0, got %d", p.VADHangoverMs)
	}
	if p.VADModelPath == "" {
		if p.VADEnterRms <= 0 || p.VADEnterRms > 32767 {
			return fmt.Errorf("vad_enter_rms must be 1..32767, got %v", p.VADEnterRms)
		}
		if p.VADExitRms < 0 || p.VADExitRms >= p.VADEnterRms {
			return fmt.Errorf("vad_exit_rms must be >= 0 and lower than vad_enter_rms, got %v", p.VADExitRms)
		}
		return nil
	}
	if p.VADSpeechProb <= 0 || p.VADSpeechProb > 1 {
		return fmt.Errorf("vad_speech_prob must be > 0 and <= 1, got %v", p.VADSpeechProb)
	}
	if p.VADSilenceProb < 0 || p.VADSilenceProb >= p.VADSpeechProb {
		return fmt.Errorf("vad_silence_prob must be >= 0 and lower than vad_speech_prob, got %v", p.VADSilenceProb)
	}
	return nil
}
