// Package tts provides text-to-speech functionality using sherpa-onnx.
package tts

import (
	"fmt"
	"log"
	"strings"
	"sync"

	"github.com/agalue/voice-assistant/internal/sherpa"
)

// Synthesizer handles text-to-speech synthesis using Kokoro models.
type Synthesizer struct {
	tts        *sherpa.OfflineTts // Kokoro TTS engine
	sampleRate int                // Output sample rate (24kHz for Kokoro)
	speakerID  int                // Speaker/voice identifier
	speed      float32            // Speech speed multiplier
	verbose    bool               // Enable verbose logging
	mu         sync.Mutex         // Protects TTS engine access
}

// Config holds TTS configuration.
type Config struct {
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

// AudioOutput contains generated audio data.
type AudioOutput struct {
	Samples    []float32 // Generated audio samples (mono)
	SampleRate int       // Sample rate of the audio (24kHz)
}

// NewSynthesizer creates a new TTS synthesizer.
func NewSynthesizer(cfg *Config) (*Synthesizer, error) {
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
		return nil, fmt.Errorf("failed to create TTS synthesizer")
	}

	return &Synthesizer{
		tts:        tts,
		sampleRate: 24000, // Kokoro default sample rate
		speakerID:  cfg.SpeakerID,
		speed:      cfg.Speed,
		verbose:    cfg.Verbose,
	}, nil
}

// Synthesize converts text to audio.
func (s *Synthesizer) Synthesize(text string) (*AudioOutput, error) {
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

// SynthesizeStreaming converts text to audio in chunks for lower latency playback.
// It splits the text into sentences and synthesizes each separately.
func (s *Synthesizer) SynthesizeStreaming(text string) ([]*AudioOutput, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, fmt.Errorf("empty text")
	}

	sentences := SplitSentences(text)
	if len(sentences) == 0 {
		return nil, fmt.Errorf("no sentences to synthesize")
	}

	var results []*AudioOutput
	for _, sentence := range sentences {
		if sentence == "" {
			continue
		}

		s.mu.Lock()
		if s.verbose {
			log.Printf("[TTS] Synthesizing sentence: %q", sentence)
		}

		audio := s.tts.Generate(sentence, s.speakerID, s.speed)
		s.mu.Unlock()

		if audio == nil || len(audio.Samples) == 0 {
			continue // Skip failed sentences
		}

		results = append(results, &AudioOutput{
			Samples:    audio.Samples,
			SampleRate: int(audio.SampleRate),
		})
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("TTS generation failed for all sentences")
	}

	return results, nil
}

// SplitSentences splits text into sentences for streaming synthesis.
// Exported for use by the TTS processor.
func SplitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	for _, c := range text {
		current.WriteRune(c)

		// Check for sentence boundaries
		if c == '.' || c == '!' || c == '?' || c == '\n' {
			trimmed := strings.TrimSpace(current.String())
			if trimmed != "" {
				sentences = append(sentences, trimmed)
			}
			current.Reset()
		}
	}

	// Don't forget remaining text
	trimmed := strings.TrimSpace(current.String())
	if trimmed != "" {
		sentences = append(sentences, trimmed)
	}

	return sentences
}

// SampleRate returns the output sample rate.
func (s *Synthesizer) SampleRate() int {
	return s.sampleRate
}

// Close releases all resources.
func (s *Synthesizer) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.tts != nil {
		sherpa.DeleteOfflineTts(s.tts)
		s.tts = nil
	}
}
