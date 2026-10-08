// Package silero wraps the Silero VAD ONNX model for per-chunk speech-probability
// inference.
//
// Vendored from github.com/plandem/silero-go (MIT, commit a4c356b): model.go and
// onnx.go, merged and unchanged in behavior. Upstream ships no go.mod, so `go get`
// synthesizes a go1.16 module and the library's own generics/`max`/`unsafe.Slice`
// fail to compile — it cannot be imported directly. We only need Model.Predict
// (the streaming Detector and miniaudio deps are dropped), so the ~190 lines that
// matter live here instead. The bundled silero_vad.onnx model is under models/.
package silero

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// Init points onnxruntime at its shared library and initializes the global
// environment. Call once before creating any Model; pair with Destroy.
func Init(sharedLibPath string, opts ...ort.EnvironmentOption) error {
	ort.SetSharedLibraryPath(sharedLibPath)
	return ort.InitializeEnvironment(opts...)
}

// Destroy tears down the global onnxruntime environment.
func Destroy() error {
	return ort.DestroyEnvironment()
}

// Model is a wrapper for the ONNX Silero VAD model. Since the model is stateful,
// it maintains the internal state and context. Note that it is not thread-safe.
type Model struct {
	context        []float32 // last samples from the previous chunk (zero-initialized)
	session        *ort.AdvancedSession
	inputTensor    *ort.Tensor[float32]
	stateTensor    *ort.Tensor[float32]
	srTensor       *ort.Tensor[int64]
	outputTensor   *ort.Tensor[float32]
	newStateTensor *ort.Tensor[float32]

	windowSize int
	sampleRate int
}

// NewModel creates a new instance of the ONNX Silero VAD model at modelPath with
// sampleRate for input audio samples. Supported values are 8000 and 16000.
func NewModel(sampleRate int, modelPath string) (*Model, error) {
	if modelPath == "" {
		return nil, fmt.Errorf("invalid ModelPath: should not be empty")
	}
	if sampleRate != 8000 && sampleRate != 16000 {
		return nil, fmt.Errorf("invalid SampleRate: valid values are 8000 and 16000")
	}

	if !ort.IsInitialized() {
		return nil, fmt.Errorf("onnxruntime should be initialized before using any model")
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Destroy()

	if err = options.SetIntraOpNumThreads(1); err != nil {
		return nil, err
	}
	if err = options.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	if err = options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, err
	}

	windowSize := 512 // 16kHz, original window size (32ms == 512 samples)
	contextSize := 64 // 16kHz, 64 samples added as context
	if sampleRate == 8000 {
		windowSize = 256
		contextSize = 32
	}

	effectiveWindowSize := windowSize + contextSize // window_size_samples + context_samples

	// Input tensor: [1, effectiveWindowSize], float32
	inputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(effectiveWindowSize)))
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	// State tensor: [2, 1, 128], float32, all zeros
	stateTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128))
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create state tensor: %w", err)
	}

	// SampleRate tensor: [1], int64
	srTensor, err := ort.NewTensor(ort.NewShape(1), []int64{int64(sampleRate)})
	if err != nil {
		inputTensor.Destroy()
		stateTensor.Destroy()
		return nil, fmt.Errorf("failed to create sr tensor: %w", err)
	}

	// Output tensor: [1, 1], float32
	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		inputTensor.Destroy()
		stateTensor.Destroy()
		srTensor.Destroy()
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	// NewState tensor: [2, 1, 128], float32, all zeros
	newStateTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128))
	if err != nil {
		inputTensor.Destroy()
		stateTensor.Destroy()
		srTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("failed to create stateN tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		[]ort.Value{inputTensor, stateTensor, srTensor},
		[]ort.Value{outputTensor, newStateTensor},
		options,
	)
	if err != nil {
		inputTensor.Destroy()
		stateTensor.Destroy()
		srTensor.Destroy()
		outputTensor.Destroy()
		newStateTensor.Destroy()
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &Model{
		context:        make([]float32, contextSize),
		session:        session,
		inputTensor:    inputTensor,
		stateTensor:    stateTensor,
		srTensor:       srTensor,
		outputTensor:   outputTensor,
		newStateTensor: newStateTensor,
		windowSize:     windowSize,
		sampleRate:     sampleRate,
	}, nil
}

// Destroy releases the ONNX session and tensors.
func (m *Model) Destroy() {
	m.session.Destroy()
	m.inputTensor.Destroy()
	m.stateTensor.Destroy()
	m.srTensor.Destroy()
	m.outputTensor.Destroy()
	m.newStateTensor.Destroy()
}

// Reset clears the internal state. Call before starting a new detection session.
func (m *Model) Reset() {
	clear(m.stateTensor.GetData())
	clear(m.context)
}

// Predict runs inference on one chunk of input data and returns the probability
// of speech. chunk is expected to be at most windowSize (512 at 16 kHz) samples.
func (m *Model) Predict(chunk []float32) (float32, error) {
	if len(chunk) > m.windowSize {
		return 0, fmt.Errorf("invalid size of chunk, maximum is %d", m.windowSize)
	}

	newData := m.inputTensor.GetData()
	clear(newData) // pad with silence if the chunk is short
	copy(newData, m.context)
	copy(newData[len(m.context):], chunk)

	if err := m.session.Run(); err != nil {
		return 0, fmt.Errorf("inference failed: %w", err)
	}

	speechProbability := m.outputTensor.GetData()[0]
	copy(m.stateTensor.GetData(), m.newStateTensor.GetData())
	copy(m.context, newData[len(newData)-len(m.context):])
	return speechProbability, nil
}
