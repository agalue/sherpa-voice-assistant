// Package stt provides speech-to-text functionality.
//
// The package defines two interfaces — [VoiceDetector] and [Transcriber] — that
// decouple the rest of the voice assistant from any specific STT implementation.
// The current implementation uses Silero VAD for voice activity detection and
// Whisper for transcription (see [WhisperRecognizer]).
//
// To add a new STT backend:
//  1. Implement [VoiceDetector] and [Transcriber] in a new file.
//  2. Implement [ModelProvider] so the binary can download/verify required model
//     files without an external setup script.
//  3. Register the new backend in [NewTranscriber] and [NewModelProvider].
package stt

import (
	"fmt"
	"strings"

	"github.com/agalue/sherpa-voice-assistant/internal/config"
)

// AudioSegment carries a completed speech segment delivered by the VAD.
type AudioSegment = []float32

// VoiceDetector handles voice activity detection (VAD).
//
// Implementations must be safe to call from the real-time audio callback thread
// (i.e., must never block). The goroutine that reads from [VoiceDetector.SegmentChannel]
// is separate from the audio callback.
type VoiceDetector interface {
	// AcceptWaveform feeds raw PCM samples from the microphone into the VAD.
	// Must be non-blocking; never hold a lock waiting on I/O inside this method.
	AcceptWaveform(samples []float32)

	// SegmentChannel returns a receive-only channel on which completed speech
	// segments are delivered. Each value is a []float32 of 16 kHz PCM samples.
	SegmentChannel() <-chan AudioSegment

	// IsSpeechDetected returns true when the VAD currently considers the
	// microphone input to be active speech.
	IsSpeechDetected() bool

	// Clear resets the internal VAD state (e.g., flushes the audio buffer).
	Clear()

	// Close releases all resources held by the detector.
	Close()
}

// Transcriber converts raw PCM audio segments into text.
//
// Implementations may be slow (100–500 ms per call) and are expected to run on
// a dedicated goroutine, not the audio callback thread.
type Transcriber interface {
	// TranscribeSegment converts a completed speech segment to text.
	// Returns an empty string when the segment contains no recognisable speech.
	TranscribeSegment(samples []float32) string

	// Close releases all resources held by the transcriber.
	Close()
}

// ModelProvider manages the lifecycle of model files required by an STT backend.
//
// Every STT implementation must implement this interface so that the binary can
// download, verify, and report on its model files without an external script.
type ModelProvider interface {
	// EnsureModels downloads any model files that are absent from modelDir.
	// If force is true, all files are re-downloaded even if they already exist.
	EnsureModels(modelDir string, force bool) error

	// VerifyModels checks that all required model files are present and returns
	// a list of missing file paths (empty slice means all files are present).
	VerifyModels(modelDir string) []string

	// Name returns a human-readable name for the STT implementation (e.g. "Whisper").
	Name() string
}

// NewTranscriber creates the [Transcriber] for the configured STT backend.
//
// Each backend interprets cfg.STTModel in its own way (e.g. Whisper strips
// the "whisper-" prefix to get the model size). To add a new backend, add
// a case here and in [NewModelProvider].
func NewTranscriber(cfg *config.Config) (Transcriber, error) {
	switch strings.ToLower(cfg.STTBackend) {
	case "whisper":
		modelSize := strings.TrimPrefix(cfg.STTModel, "whisper-")
		return NewWhisperRecognizer(&WhisperConfig{
			ModelDir:   cfg.ModelDir,
			ModelSize:  modelSize,
			SampleRate: cfg.SampleRate,
			WakeWord:   cfg.WakeWord,
			Provider:   cfg.STTProvider,
			Language:   cfg.STTLanguage,
			Verbose:    cfg.Verbose,
			NumThreads: cfg.STTThreads,
		})
	default:
		return nil, fmt.Errorf("unknown STT backend %q (available: whisper)", cfg.STTBackend)
	}
}

// NewModelProvider returns the [ModelProvider] for the configured STT backend.
//
// The returned provider handles model download and verification for the
// selected backend. To add a new backend, add a case here.
func NewModelProvider(cfg *config.Config) (ModelProvider, error) {
	switch strings.ToLower(cfg.STTBackend) {
	case "whisper":
		modelSize := strings.TrimPrefix(cfg.STTModel, "whisper-")
		return &WhisperModelProvider{ModelSize: modelSize}, nil
	default:
		return nil, fmt.Errorf("unknown STT backend %q (available: whisper)", cfg.STTBackend)
	}
}
