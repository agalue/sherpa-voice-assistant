// Package tts provides text-to-speech functionality using sherpa-onnx.
// This file contains the Kokoro-based TTS implementation.
package tts

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/agalue/sherpa-voice-assistant/internal/setup"
	"github.com/agalue/sherpa-voice-assistant/internal/sherpa"
)

// Compile-time interface compliance check.
var _ Synthesizer = (*KokoroSynthesizer)(nil)

// ---------------------------------------------------------------------------
// Kokoro voice catalog (53 voices across 9 languages)
// ---------------------------------------------------------------------------

// kokoroVoice contains essential runtime data for a single Kokoro TTS voice.
type kokoroVoice struct {
	speakerID  int
	espeakCode string // Language code for espeak-ng
	language   string // Human-readable language name
}

// kokoroVoices maps voice names to metadata for all 53 Kokoro v1.0 voices.
var kokoroVoices = map[string]kokoroVoice{
	// American English (20 voices)
	"af_alloy":   {speakerID: 0, espeakCode: "en-us", language: "American English"},
	"af_aoede":   {speakerID: 1, espeakCode: "en-us", language: "American English"},
	"af_bella":   {speakerID: 2, espeakCode: "en-us", language: "American English"},
	"af_heart":   {speakerID: 3, espeakCode: "en-us", language: "American English"},
	"af_jessica": {speakerID: 4, espeakCode: "en-us", language: "American English"},
	"af_kore":    {speakerID: 5, espeakCode: "en-us", language: "American English"},
	"af_nicole":  {speakerID: 6, espeakCode: "en-us", language: "American English"},
	"af_nova":    {speakerID: 7, espeakCode: "en-us", language: "American English"},
	"af_river":   {speakerID: 8, espeakCode: "en-us", language: "American English"},
	"af_sarah":   {speakerID: 9, espeakCode: "en-us", language: "American English"},
	"af_sky":     {speakerID: 10, espeakCode: "en-us", language: "American English"},
	"am_adam":    {speakerID: 11, espeakCode: "en-us", language: "American English"},
	"am_echo":    {speakerID: 12, espeakCode: "en-us", language: "American English"},
	"am_eric":    {speakerID: 13, espeakCode: "en-us", language: "American English"},
	"am_fenrir":  {speakerID: 14, espeakCode: "en-us", language: "American English"},
	"am_liam":    {speakerID: 15, espeakCode: "en-us", language: "American English"},
	"am_michael": {speakerID: 16, espeakCode: "en-us", language: "American English"},
	"am_onyx":    {speakerID: 17, espeakCode: "en-us", language: "American English"},
	"am_puck":    {speakerID: 18, espeakCode: "en-us", language: "American English"},
	"am_santa":   {speakerID: 19, espeakCode: "en-us", language: "American English"},

	// British English (8 voices)
	"bf_alice":    {speakerID: 20, espeakCode: "en-gb", language: "British English"},
	"bf_emma":     {speakerID: 21, espeakCode: "en-gb", language: "British English"},
	"bf_isabella": {speakerID: 22, espeakCode: "en-gb", language: "British English"},
	"bf_lily":     {speakerID: 23, espeakCode: "en-gb", language: "British English"},
	"bm_daniel":   {speakerID: 24, espeakCode: "en-gb", language: "British English"},
	"bm_fable":    {speakerID: 25, espeakCode: "en-gb", language: "British English"},
	"bm_george":   {speakerID: 26, espeakCode: "en-gb", language: "British English"},
	"bm_lewis":    {speakerID: 27, espeakCode: "en-gb", language: "British English"},

	// Spanish (2 voices)
	"ef_dora": {speakerID: 28, espeakCode: "es", language: "Spanish"},
	"em_alex": {speakerID: 29, espeakCode: "es", language: "Spanish"},

	// French (1 voice)
	"ff_siwis": {speakerID: 30, espeakCode: "fr-fr", language: "French"},

	// Hindi (4 voices)
	"hf_alpha": {speakerID: 31, espeakCode: "hi", language: "Hindi"},
	"hf_beta":  {speakerID: 32, espeakCode: "hi", language: "Hindi"},
	"hm_omega": {speakerID: 33, espeakCode: "hi", language: "Hindi"},
	"hm_psi":   {speakerID: 34, espeakCode: "hi", language: "Hindi"},

	// Italian (2 voices)
	"if_sara":   {speakerID: 35, espeakCode: "it", language: "Italian"},
	"im_nicola": {speakerID: 36, espeakCode: "it", language: "Italian"},

	// Japanese (5 voices)
	"jf_alpha":      {speakerID: 37, espeakCode: "ja", language: "Japanese"},
	"jf_gongitsune": {speakerID: 38, espeakCode: "ja", language: "Japanese"},
	"jf_nezumi":     {speakerID: 39, espeakCode: "ja", language: "Japanese"},
	"jf_tebukuro":   {speakerID: 40, espeakCode: "ja", language: "Japanese"},
	"jm_kumo":       {speakerID: 41, espeakCode: "ja", language: "Japanese"},

	// Portuguese BR (3 voices)
	"pf_dora":  {speakerID: 42, espeakCode: "pt-br", language: "Portuguese BR"},
	"pm_alex":  {speakerID: 43, espeakCode: "pt-br", language: "Portuguese BR"},
	"pm_santa": {speakerID: 44, espeakCode: "pt-br", language: "Portuguese BR"},

	// Mandarin Chinese (8 voices)
	"zf_xiaobei":  {speakerID: 45, espeakCode: "cmn", language: "Mandarin Chinese"},
	"zf_xiaoni":   {speakerID: 46, espeakCode: "cmn", language: "Mandarin Chinese"},
	"zf_xiaoxiao": {speakerID: 47, espeakCode: "cmn", language: "Mandarin Chinese"},
	"zf_xiaoyi":   {speakerID: 48, espeakCode: "cmn", language: "Mandarin Chinese"},
	"zm_yunjian":  {speakerID: 49, espeakCode: "cmn", language: "Mandarin Chinese"},
	"zm_yunxi":    {speakerID: 50, espeakCode: "cmn", language: "Mandarin Chinese"},
	"zm_yunxia":   {speakerID: 51, espeakCode: "cmn", language: "Mandarin Chinese"},
	"zm_yunyang":  {speakerID: 52, espeakCode: "cmn", language: "Mandarin Chinese"},
}

// getKokoroVoice returns voice data for a given voice name, or nil if unknown.
func getKokoroVoice(name string) *kokoroVoice {
	if v, ok := kokoroVoices[name]; ok {
		return &v
	}
	return nil
}

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
// Only generic configuration is required — all model-specific paths (model.onnx,
// voices.bin, tokens.txt, espeak-ng-data, lexicon) are derived from ModelDir.
type KokoroConfig struct {
	ModelDir   string // Base model directory (Kokoro files resolved automatically)
	Voice      string // Voice name (e.g. "af_bella"); looked up in the Voices catalog
	SpeakerID  int
	Speed      float32
	Provider   string // Hardware acceleration provider (cpu, cuda, coreml)
	Verbose    bool
	NumThreads int
}

// AudioOutput type is defined in tts.go.

// NewKokoroSynthesizer creates a [KokoroSynthesizer] that satisfies [Synthesizer].
//
// All file paths are derived from cfg.ModelDir. The voice's language and optional
// lexicon are determined automatically from the [Voices] catalog.
func NewKokoroSynthesizer(cfg *KokoroConfig) (*KokoroSynthesizer, error) {
	voice := getKokoroVoice(cfg.Voice)
	if voice == nil {
		return nil, fmt.Errorf("unknown TTS voice %q; run with --list-voices to see available voices", cfg.Voice)
	}

	kokoroDir := filepath.Join(cfg.ModelDir, "tts", "kokoro-multi-lang-v1_0")
	modelPath := filepath.Join(kokoroDir, "model.onnx")
	voicesPath := filepath.Join(kokoroDir, "voices.bin")
	tokensPath := filepath.Join(kokoroDir, "tokens.txt")
	dataDir := filepath.Join(kokoroDir, "espeak-ng-data")
	lexicon := lexiconForVoice(kokoroDir, cfg.Voice)

	ttsConfig := &sherpa.OfflineTtsConfig{}

	// Configure Kokoro model
	ttsConfig.Model.Kokoro.Model = modelPath
	ttsConfig.Model.Kokoro.Voices = voicesPath
	ttsConfig.Model.Kokoro.Tokens = tokensPath
	ttsConfig.Model.Kokoro.DataDir = dataDir
	ttsConfig.Model.Kokoro.Lexicon = lexicon
	ttsConfig.Model.Kokoro.Lang = voice.espeakCode       // Derived from voice catalog
	ttsConfig.Model.Kokoro.LengthScale = 1.0 / cfg.Speed // Inverse for speed control
	ttsConfig.Model.NumThreads = cfg.NumThreads
	ttsConfig.Model.Provider = cfg.Provider // Hardware acceleration (cpu, cuda, coreml)
	ttsConfig.MaxNumSentences = 1           // Kokoro TTS only supports 1
	ttsConfig.Model.Debug = 0
	if cfg.Verbose {
		ttsConfig.Model.Debug = 1
	}

	tts := sherpa.NewOfflineTts(ttsConfig)
	if tts == nil {
		// Check for missing model files to give an actionable error message.
		provider := &KokoroModelProvider{}
		if missing := provider.VerifyModels(cfg.ModelDir); len(missing) > 0 {
			return nil, fmt.Errorf("failed to create Kokoro TTS synthesizer: missing model files %v; run with --setup to download them", missing)
		}
		return nil, fmt.Errorf("failed to create Kokoro TTS synthesizer (model dir: %s); verify model files are valid or re-download with --setup", kokoroDir)
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
	cfg := &sherpa.GenerationConfig{
		Sid:   s.speakerID,
		Speed: s.speed,
	}
	audio := s.tts.GenerateWithConfig(text, cfg, nil)
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
// Kokoro-specific helpers
// ---------------------------------------------------------------------------

// lexiconForVoice returns the path(s) to language-specific lexicon file(s) for the
// given voice, as a comma-separated string suitable for the sherpa-onnx lexicon
// field. Returns an empty string when there is no matching lexicon, which is the
// expected case for most languages.
func lexiconForVoice(kokoroDir, voiceName string) string {
	v := getKokoroVoice(voiceName)
	if v == nil {
		return ""
	}
	switch v.espeakCode {
	case "en-us":
		lexPath := filepath.Join(kokoroDir, "lexicon-us-en.txt")
		if _, err := os.Stat(lexPath); err != nil {
			return ""
		}
		return lexPath
	case "en-gb":
		lexPath := filepath.Join(kokoroDir, "lexicon-gb-en.txt")
		if _, err := os.Stat(lexPath); err != nil {
			return ""
		}
		return lexPath
	case "cmn":
		// Chinese voices use a combined English+Chinese lexicon for best phoneme coverage.
		// Both files must be present; returning partial paths causes sherpa-onnx to error.
		enLex := filepath.Join(kokoroDir, "lexicon-us-en.txt")
		zhLex := filepath.Join(kokoroDir, "lexicon-zh.txt")
		if _, err := os.Stat(enLex); err != nil {
			return ""
		}
		if _, err := os.Stat(zhLex); err != nil {
			return ""
		}
		return enLex + "," + zhLex
	default:
		return ""
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

	if !force && setup.FileExists(modelFile) {
		log.Println("[TTS] Kokoro model already present, skipping")
	} else {
		url := "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
		log.Printf("[TTS] Downloading Kokoro TTS model from %s …", url)
		// The archive contains a top-level "kokoro-multi-lang-v1_0/" directory.
		// ExtractTarBz2Dir strips one level, so pass kokoroDir as destination.
		if err := setup.ExtractTarBz2Dir(url, kokoroDir); err != nil {
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
		if err := setup.ExtractTarBz2Dir(url, espeakDir); err != nil {
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
		if !setup.FileExists(f) {
			missing = append(missing, f)
		}
	}
	return missing
}

// PrintVoices lists all available Kokoro voices — satisfies [ModelProvider].
func (p *KokoroModelProvider) PrintVoices() {
	fmt.Println("═══════════════════════════════════════════════════════════════════")
	fmt.Println("  Kokoro TTS v1.0 - 53 Voices Across 9 Languages")
	fmt.Println("═══════════════════════════════════════════════════════════════════")
	fmt.Println()

	languages := []string{
		"American English", "British English", "Spanish", "French",
		"Hindi", "Italian", "Japanese", "Portuguese BR", "Mandarin Chinese",
	}

	for _, lang := range languages {
		var voiceNames []string
		for name, voice := range kokoroVoices {
			if voice.language == lang {
				voiceNames = append(voiceNames, name)
			}
		}
		sort.Strings(voiceNames)

		fmt.Printf("\n── %s (%d voices) ──\n", lang, len(voiceNames))
		fmt.Printf("%-15s %-4s %s\n", "VOICE", "ID", "ESPEAK")
		fmt.Println(strings.Repeat("─", 50))

		for _, name := range voiceNames {
			voice := kokoroVoices[name]
			fmt.Printf("%-15s %-4d %s\n", name, voice.speakerID, voice.espeakCode)
		}
	}

	fmt.Println()
	fmt.Println(strings.Repeat("─", 70))
	fmt.Println()
	fmt.Println("Default: af_bella (ID 2) - American English")
	fmt.Println("Recommended: af_heart (ID 3) or bf_emma (ID 21)")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  ./voice-assistant --tts-voice af_bella")
	fmt.Println("  ./voice-assistant --tts-voice bf_emma --tts-speaker-id 21")
	fmt.Println()
	fmt.Println("Try different voices to find what sounds best to you!")
	fmt.Println()
}

// PrintVoiceInfo prints detailed information about a specific Kokoro voice — satisfies [ModelProvider].
func (p *KokoroModelProvider) PrintVoiceInfo(name string) error {
	voice := getKokoroVoice(name)
	if voice == nil {
		return fmt.Errorf("voice '%s' not found. Run with --list-voices to see available voices", name)
	}

	fmt.Println()
	fmt.Printf("Voice: %s\n", name)
	fmt.Println(strings.Repeat("─", 40))
	fmt.Printf("Speaker ID:  %d\n", voice.speakerID)
	fmt.Printf("Language:    %s\n", voice.language)
	fmt.Printf("Espeak code: %s\n", voice.espeakCode)
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Printf("  ./voice-assistant --tts-voice %s --tts-speaker-id %d\n", name, voice.speakerID)
	fmt.Println()

	return nil
}
