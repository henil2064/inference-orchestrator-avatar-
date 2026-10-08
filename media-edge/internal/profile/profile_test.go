package profile

import "testing"

func TestValidateAcceptsDefaults(t *testing.T) {
	if err := Voice.Validate(); err != nil {
		t.Fatalf("default voice profile must validate: %v", err)
	}
}

func TestValidateRejectsBadEnergyThresholds(t *testing.T) {
	p := Voice
	p.VADModelPath = ""
	p.VADExitRms = p.VADEnterRms
	if err := p.Validate(); err == nil {
		t.Fatal("expected equal enter/exit RMS thresholds to fail")
	}
}

func TestValidateRejectsBadSileroThresholds(t *testing.T) {
	p := Voice
	p.VADModelPath = "silero.onnx"
	p.VADSilenceProb = p.VADSpeechProb
	if err := p.Validate(); err == nil {
		t.Fatal("expected equal speech/silence probabilities to fail")
	}
}

func TestValidateRejectsImpossiblePtime(t *testing.T) {
	p := Voice
	p.PtimeMs = 0
	if err := p.Validate(); err == nil {
		t.Fatal("expected ptime_ms=0 to fail")
	}
}
