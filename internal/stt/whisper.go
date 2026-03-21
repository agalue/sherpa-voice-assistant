// Package stt provides speech-to-text functionality using sherpa-onnx.
// This file contains the Whisper-based transcription implementation.
package stt

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/agalue/sherpa-voice-assistant/internal/models"
	"github.com/agalue/sherpa-voice-assistant/internal/sherpa"
)

// Compile-time interface compliance check.
var _ Transcriber = (*WhisperRecognizer)(nil)

// WhisperRecognizer implements [Transcriber] using OpenAI Whisper via sherpa-onnx.
//
// Inference is slow (100–500 ms per segment) and must not be called from the audio
// callback thread. Pair with [SileroVAD] to form a complete STT pipeline.
type WhisperRecognizer struct {
	recognizer *sherpa.OfflineRecognizer
	wakeWord   string
	verbose    bool
	sampleRate int
}

// WhisperConfig holds configuration for [WhisperRecognizer].
type WhisperConfig struct {
	ModelDir   string // Base model directory (Whisper files resolved automatically)
	ModelSize  string // Model variant (e.g. "tiny", "base", "small")
	SampleRate int
	WakeWord   string
	Provider   string // Hardware acceleration provider (cpu, cuda, coreml)
	Language   string // Recognition language (e.g. "en", "es", "auto")
	Verbose    bool
	NumThreads int
}

// NewWhisperRecognizer creates a [WhisperRecognizer] that satisfies [Transcriber].
func NewWhisperRecognizer(cfg *WhisperConfig) (*WhisperRecognizer, error) {
	encoder := filepath.Join(cfg.ModelDir, "whisper", fmt.Sprintf("whisper-%s-encoder.int8.onnx", cfg.ModelSize))
	decoder := filepath.Join(cfg.ModelDir, "whisper", fmt.Sprintf("whisper-%s-decoder.int8.onnx", cfg.ModelSize))
	tokens := filepath.Join(cfg.ModelDir, "whisper", fmt.Sprintf("whisper-%s-tokens.txt", cfg.ModelSize))

	recognizerConfig := &sherpa.OfflineRecognizerConfig{}
	recognizerConfig.ModelConfig.Whisper.Encoder = encoder
	recognizerConfig.ModelConfig.Whisper.Decoder = decoder

	// "auto" -> "" triggers Whisper's built-in language detection.
	language := cfg.Language
	if strings.EqualFold(language, "auto") {
		language = ""
	}
	recognizerConfig.ModelConfig.Whisper.Language = language
	recognizerConfig.ModelConfig.Whisper.Task = "transcribe"
	recognizerConfig.ModelConfig.Whisper.TailPaddings = -1
	recognizerConfig.ModelConfig.Tokens = tokens
	recognizerConfig.ModelConfig.NumThreads = cfg.NumThreads
	recognizerConfig.ModelConfig.Provider = cfg.Provider
	recognizerConfig.DecodingMethod = "greedy_search"
	recognizerConfig.ModelConfig.Debug = 0
	if cfg.Verbose {
		recognizerConfig.ModelConfig.Debug = 1
	}

	recognizer := sherpa.NewOfflineRecognizer(recognizerConfig)
	if recognizer == nil {
		return nil, fmt.Errorf("failed to create Whisper recognizer")
	}

	return &WhisperRecognizer{
		recognizer: recognizer,
		wakeWord:   strings.ToLower(cfg.WakeWord),
		verbose:    cfg.Verbose,
		sampleRate: cfg.SampleRate,
	}, nil
}

// TranscribeSegment converts a completed speech segment to text using Whisper.
// Returns an empty string when the segment contains no recognisable speech.
func (r *WhisperRecognizer) TranscribeSegment(samples []float32) string {
	if len(samples) == 0 {
		return ""
	}

	if r.verbose {
		duration := float32(len(samples)) / float32(r.sampleRate)
		log.Printf("[STT] Processing speech segment: %.2fs", duration)
	}

	stream := sherpa.NewOfflineStream(r.recognizer)
	if stream == nil {
		log.Println("[STT] Failed to create offline stream")
		return ""
	}
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(r.sampleRate, samples)
	r.recognizer.Decode(stream)

	text := strings.TrimSpace(stream.GetResult().Text)
	if text == "" {
		return ""
	}

	// Check wake word if configured
	if r.wakeWord != "" {
		lowerText := strings.ToLower(text)
		if !strings.Contains(lowerText, r.wakeWord) {
			if r.verbose {
				log.Printf("[STT] Wake word %q not found in %q, ignoring", r.wakeWord, text)
			}
			return ""
		}
		// Remove wake word from text
		text = removeWakeWord(text, r.wakeWord)
		text = strings.TrimSpace(text)

		// If only wake word was spoken (no query), prompt for hello
		if text == "" {
			log.Printf("🗣️ Wake word %q detected", r.wakeWord)
			text = "Hello"
		} else {
			log.Printf("🗣️ You (wake word detected): %s", text)
		}
		return text
	}

	log.Printf("🗣️ You: %s", text)
	return text
}

// Close releases all resources held by the recognizer.
func (r *WhisperRecognizer) Close() {
	if r.recognizer != nil {
		sherpa.DeleteOfflineRecognizer(r.recognizer)
		r.recognizer = nil
	}
}

// removeWakeWord removes the wake word from text, case-insensitively.
func removeWakeWord(text, wakeWord string) string {
	lowerText := strings.ToLower(text)
	idx := strings.Index(lowerText, strings.ToLower(wakeWord))
	if idx == -1 {
		return text
	}
	result := text[:idx] + text[idx+len(wakeWord):]
	return strings.TrimSpace(strings.TrimLeft(result, " ,.!?;:-'\""))
}

// ---------------------------------------------------------------------------
// WhisperModelProvider
// ---------------------------------------------------------------------------

// WhisperModelProvider implements [ModelProvider] for the Whisper STT backend.
//
// It manages only the Whisper model files (encoder, decoder, tokens).
// The Silero VAD model is managed separately by [SileroModelProvider].
type WhisperModelProvider struct {
	// ModelSize is the Whisper model variant to use (e.g. "tiny", "base", "small").
	ModelSize string
}

// Name returns the human-readable name of this STT implementation.
func (p *WhisperModelProvider) Name() string {
	return "Whisper"
}

// EnsureModels downloads any missing Whisper model files.
//
// It fetches the sherpa-onnx Whisper archive and extracts only the int8 quantised
// encoder, decoder, and tokens files to keep disk usage minimal.
func (p *WhisperModelProvider) EnsureModels(modelDir string, force bool) error {
	if err := p.downloadWhisper(modelDir, force); err != nil {
		return fmt.Errorf("downloading Whisper %s: %w", p.ModelSize, err)
	}
	return nil
}

// VerifyModels returns a list of absent Whisper model file paths.
func (p *WhisperModelProvider) VerifyModels(modelDir string) []string {
	required := []string{
		p.encoderPath(modelDir),
		p.decoderPath(modelDir),
		p.tokensPath(modelDir),
	}
	var missing []string
	for _, f := range required {
		if !models.FileExists(f) {
			missing = append(missing, f)
		}
	}
	return missing
}

func (p *WhisperModelProvider) encoderPath(modelDir string) string {
	return filepath.Join(modelDir, "whisper", fmt.Sprintf("whisper-%s-encoder.int8.onnx", p.ModelSize))
}

func (p *WhisperModelProvider) decoderPath(modelDir string) string {
	return filepath.Join(modelDir, "whisper", fmt.Sprintf("whisper-%s-decoder.int8.onnx", p.ModelSize))
}

func (p *WhisperModelProvider) tokensPath(modelDir string) string {
	return filepath.Join(modelDir, "whisper", fmt.Sprintf("whisper-%s-tokens.txt", p.ModelSize))
}

// downloadWhisper fetches the sherpa-onnx Whisper archive and extracts only the
// int8 quantised files (encoder, decoder, tokens) to keep disk usage minimal.
func (p *WhisperModelProvider) downloadWhisper(modelDir string, force bool) error {
	whisperDir := filepath.Join(modelDir, "whisper")
	if err := os.MkdirAll(whisperDir, 0o755); err != nil {
		return err
	}

	encoder := p.encoderPath(modelDir)
	if !force && models.FileExists(encoder) {
		log.Printf("[STT] Whisper %s already present, skipping", p.ModelSize)
		return nil
	}

	url := fmt.Sprintf(
		"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-%s.tar.bz2",
		p.ModelSize,
	)
	log.Printf("[STT] Downloading Whisper %s from %s …", p.ModelSize, url)

	// Extract only the three int8 files we need.
	wantFiles := map[string]string{
		fmt.Sprintf("sherpa-onnx-whisper-%s/%s-encoder.int8.onnx", p.ModelSize, p.ModelSize): encoder,
		fmt.Sprintf("sherpa-onnx-whisper-%s/%s-decoder.int8.onnx", p.ModelSize, p.ModelSize): p.decoderPath(modelDir),
		fmt.Sprintf("sherpa-onnx-whisper-%s/%s-tokens.txt", p.ModelSize, p.ModelSize):        p.tokensPath(modelDir),
	}

	return models.ExtractTarBz2Selected(url, wantFiles)
}
