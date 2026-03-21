// Package tts provides text-to-speech functionality using sherpa-onnx.
// This file contains the Kokoro-based TTS implementation.
package tts

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/agalue/sherpa-voice-assistant/internal/models"
	"github.com/agalue/sherpa-voice-assistant/internal/sherpa"
)

// Compile-time interface compliance check.
var _ Synthesizer = (*KokoroSynthesizer)(nil)

// KokoroSynthesizer implements [Synthesizer] using the Kokoro multi-lingual TTS
// model via sherpa-onnx. It is safe for concurrent use; a mutex serialises calls
// to the underlying ONNX runtime.
type KokoroSynthesizer struct {
	tts        *sherpa.OfflineTts // Kokoro TTS engine
	sampleRate int                // Output sample rate (24kHz for Kokoro)
	speakerID  int                // Speaker/voice identifier
	speed      float32            // Speech speed multiplier
	verbose    bool               // Enable verbose logging
	mu         sync.Mutex         // Protects TTS engine access
}

// KokoroConfig holds configuration for the Kokoro TTS synthesizer.
type KokoroConfig struct {
	Model      string // Path to model.onnx
	Voices     string // Path to voices.bin
	Tokens     string // Path to tokens.txt
	DataDir    string // espeak-ng-data directory
	Lexicon    string // Path to lexicon.txt (optional)
	Language   string // Language code for multi-lingual models (e.g., "en-gb", "en-us")
	SpeakerID  int
	Speed      float32
	Provider   string // Hardware acceleration provider (cpu, cuda, coreml)
	Verbose    bool
	TTSThreads int // Number of threads for TTS
}

// AudioOutput type is defined in tts.go.

// NewKokoroSynthesizer creates a [KokoroSynthesizer] that satisfies [Synthesizer].
func NewKokoroSynthesizer(cfg *KokoroConfig) (*KokoroSynthesizer, error) {
	ttsConfig := &sherpa.OfflineTtsConfig{}

	// Configure Kokoro model
	ttsConfig.Model.Kokoro.Model = cfg.Model
	ttsConfig.Model.Kokoro.Voices = cfg.Voices
	ttsConfig.Model.Kokoro.Tokens = cfg.Tokens
	ttsConfig.Model.Kokoro.DataDir = cfg.DataDir
	ttsConfig.Model.Kokoro.Lexicon = cfg.Lexicon
	ttsConfig.Model.Kokoro.Lang = cfg.Language           // Required for multi-lingual Kokoro v1.0+
	ttsConfig.Model.Kokoro.LengthScale = 1.0 / cfg.Speed // Inverse for speed control
	ttsConfig.Model.NumThreads = 2
	ttsConfig.Model.Provider = cfg.Provider // Hardware acceleration (cpu, cuda, coreml)
	ttsConfig.MaxNumSentences = 1           // Kokoro TTS only supports 1
	ttsConfig.Model.Debug = 0
	if cfg.Verbose {
		ttsConfig.Model.Debug = 1
	}

	tts := sherpa.NewOfflineTts(ttsConfig)
	if tts == nil {
		return nil, fmt.Errorf("failed to create Kokoro TTS synthesizer")
	}

	return &KokoroSynthesizer{
		tts:        tts,
		sampleRate: 24000, // Kokoro default sample rate
		speakerID:  cfg.SpeakerID,
		speed:      cfg.Speed,
		verbose:    cfg.Verbose,
	}, nil
}

// Synthesize converts text to audio — satisfies [Synthesizer].
func (s *KokoroSynthesizer) Synthesize(text string) (*AudioOutput, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	text = strings.TrimSpace(text)
	if text == "" {
		return nil, fmt.Errorf("empty text")
	}

	if s.verbose {
		log.Printf("[TTS] Synthesizing: %q", text)
	}

	// Generate audio
	audio := s.tts.Generate(text, s.speakerID, s.speed)
	if audio == nil || len(audio.Samples) == 0 {
		return nil, fmt.Errorf("TTS generation failed")
	}

	log.Printf("🎵 Generated speech (%d samples)", len(audio.Samples))

	return &AudioOutput{
		Samples:    audio.Samples,
		SampleRate: int(audio.SampleRate),
	}, nil
}

// SampleRate returns the output sample rate — satisfies [Synthesizer].
func (s *KokoroSynthesizer) SampleRate() int {
	return s.sampleRate
}

// Close releases all resources — satisfies [Synthesizer].
func (s *KokoroSynthesizer) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.tts != nil {
		sherpa.DeleteOfflineTts(s.tts)
		s.tts = nil
	}
}

// ---------------------------------------------------------------------------
// ModelProvider implementation
// ---------------------------------------------------------------------------

// KokoroModelProvider implements [ModelProvider] for the Kokoro TTS backend.
// It knows how to download the Kokoro multi-lingual model archive (including the
// Silero VAD espeak-ng phonemisation data) so the binary manages its own deps.
type KokoroModelProvider struct{}

// Name returns the human-readable name of this TTS implementation.
func (p *KokoroModelProvider) Name() string {
	return "Kokoro"
}

// EnsureModels downloads missing Kokoro TTS model files.
//
// It fetches:
//  1. kokoro-multi-lang-v1_0.tar.bz2 — main model archive (contains a top-level
//     kokoro-multi-lang-v1_0/ directory that is preserved verbatim)
//  2. espeak-ng-data.tar.bz2         — phonemisation data (skipped if already bundled
//     inside the Kokoro archive)
//
// If force is true, all files are re-downloaded even if they already exist.
func (p *KokoroModelProvider) EnsureModels(modelDir string, force bool) error {
	ttsDir := filepath.Join(modelDir, "tts")
	if err := os.MkdirAll(ttsDir, 0o755); err != nil {
		return err
	}

	kokoroDir := filepath.Join(ttsDir, "kokoro-multi-lang-v1_0")
	modelFile := filepath.Join(kokoroDir, "model.onnx")

	if !force && models.FileExists(modelFile) {
		log.Println("[TTS] Kokoro model already present, skipping")
	} else {
		url := "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
		log.Printf("[TTS] Downloading Kokoro TTS model from %s \u2026", url)
		// The archive contains a top-level "kokoro-multi-lang-v1_0/" directory.
		// ExtractTarBz2Dir strips one level, so pass kokoroDir as destination.
		if err := models.ExtractTarBz2Dir(url, kokoroDir); err != nil {
			return fmt.Errorf("downloading Kokoro TTS: %w", err)
		}
	}

	// espeak-ng-data is usually bundled inside the Kokoro archive; only fetch it
	// separately if it is still absent after the main download.
	espeakDir := filepath.Join(kokoroDir, "espeak-ng-data")
	if _, err := os.Stat(espeakDir); os.IsNotExist(err) {
		url := "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2"
		log.Printf("[TTS] Downloading espeak-ng-data from %s …", url)
		// Archive top-level is "espeak-ng-data/"; extract into kokoroDir so that
		// the result lands at kokoroDir/espeak-ng-data/.
		if err := models.ExtractTarBz2Dir(url, espeakDir); err != nil {
			return fmt.Errorf("downloading espeak-ng-data: %w", err)
		}
	}

	return nil
}

// VerifyModels returns a list of paths to model files that are absent from modelDir.
func (p *KokoroModelProvider) VerifyModels(modelDir string) []string {
	kokoroDir := filepath.Join(modelDir, "tts", "kokoro-multi-lang-v1_0")
	required := []string{
		filepath.Join(kokoroDir, "model.onnx"),
		filepath.Join(kokoroDir, "voices.bin"),
		filepath.Join(kokoroDir, "tokens.txt"),
	}
	var missing []string
	for _, f := range required {
		if !models.FileExists(f) {
			missing = append(missing, f)
		}
	}
	return missing
}
