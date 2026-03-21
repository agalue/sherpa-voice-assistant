// Package tts provides text-to-speech functionality.
//
// The package defines [Synthesizer] — the primary interface — that decouples the rest
// of the voice assistant from any specific TTS implementation. The current implementation
// uses Kokoro (via sherpa-onnx); see [KokoroSynthesizer].
//
// To add a new TTS backend:
//  1. Implement [Synthesizer] in a new file.
//  2. Implement [ModelProvider] so the binary can download/verify required model files
//     without an external setup script.
//  3. Wire the new implementation in [cmd/assistant/main.go].
package tts

// AudioOutput contains generated audio data.
type AudioOutput struct {
	Samples    []float32 // Generated audio samples (mono, float32 in [-1, 1])
	SampleRate int       // Sample rate of the generated audio (Hz)
}

// Synthesizer converts text chunks into audio.
//
// Implementations are expected to be slow (100–900 ms per call) and should run on a
// dedicated goroutine, not the audio callback thread.
type Synthesizer interface {
	// Synthesize converts a single sentence of text to audio.
	// Returns an error if synthesis fails (e.g., model error, empty input).
	Synthesize(text string) (*AudioOutput, error)

	// SampleRate returns the sample rate (in Hz) of audio produced by this synthesizer.
	SampleRate() int

	// Close releases all resources held by the synthesizer.
	Close()
}

// ModelProvider manages the lifecycle of model files required by a TTS backend.
//
// Every TTS implementation must implement this interface so that the binary can
// download, verify, and report on its model files without an external script.
type ModelProvider interface {
	// EnsureModels downloads any model files that are absent from modelDir.
	// If force is true, all files are re-downloaded even if they already exist.
	EnsureModels(modelDir string, force bool) error

	// VerifyModels checks that all required model files are present and returns
	// a list of missing file paths (empty slice means all files are present).
	VerifyModels(modelDir string) []string

	// Name returns a human-readable name for the TTS implementation (e.g. "Kokoro").
	Name() string
}
