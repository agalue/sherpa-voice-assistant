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
//  3. Register the new backend in [NewSynthesizer] and [NewModelProvider].
package tts

import (
	"fmt"
	"strings"

	"github.com/agalue/sherpa-voice-assistant/internal/config"
)

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
// It also handles voice listing so that --list-voices / --voice-info are
// dispatched generically through the interface.
type ModelProvider interface {
	// EnsureModels downloads any model files that are absent from modelDir.
	// If force is true, all files are re-downloaded even if they already exist.
	EnsureModels(modelDir string, force bool) error

	// VerifyModels checks that all required model files are present and returns
	// a list of missing file paths (empty slice means all files are present).
	VerifyModels(modelDir string) []string

	// Name returns a human-readable name for the TTS implementation (e.g. "Kokoro").
	Name() string

	// PrintVoices lists all available voices for this TTS backend.
	PrintVoices()

	// PrintVoiceInfo prints detailed information about a specific voice.
	// Returns an error if the voice name is not found.
	PrintVoiceInfo(name string) error
}

// NewSynthesizer creates the [Synthesizer] for the configured TTS backend.
//
// Each backend interprets the TTS-related config fields in its own way (e.g.
// Kokoro looks up TTSVoice in a built-in catalog). To add a new backend, add
// a case here and in [NewModelProvider].
func NewSynthesizer(cfg *config.Config) (Synthesizer, error) {
	switch strings.ToLower(cfg.TTSBackend) {
	case "kokoro":
		return NewKokoroSynthesizer(&KokoroConfig{
			ModelDir:   cfg.ModelDir,
			Voice:      cfg.TTSVoice,
			SpeakerID:  cfg.TTSSpeakerID,
			Speed:      cfg.TTSSpeed,
			Provider:   cfg.TTSProvider,
			Verbose:    cfg.Verbose,
			NumThreads: cfg.TTSThreads,
		})
	default:
		return nil, fmt.Errorf("unknown TTS backend %q (available: kokoro)", cfg.TTSBackend)
	}
}

// NewModelProvider returns the [ModelProvider] for the configured TTS backend.
//
// The returned provider handles model download, verification, and voice listing
// for the selected backend. To add a new backend, add a case here.
func NewModelProvider(cfg *config.Config) (ModelProvider, error) {
	switch strings.ToLower(cfg.TTSBackend) {
	case "kokoro":
		return &KokoroModelProvider{}, nil
	default:
		return nil, fmt.Errorf("unknown TTS backend %q (available: kokoro)", cfg.TTSBackend)
	}
}
