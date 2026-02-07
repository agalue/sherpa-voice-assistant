// Package config provides configuration and CLI argument parsing for the voice assistant.
package config

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/agalue/voice-assistant/internal/sherpa"
)

// Voice data is now in voices.go - simpler maps with just essential fields.
// Old VoiceInfo struct and AllVoices array removed for simplicity.

// InterruptMode defines how playback interruption is handled.
type InterruptMode int

const (
	// InterruptAlways allows interrupts during playback (best for headsets).
	InterruptAlways InterruptMode = iota
	// InterruptWait pauses microphone during playback (best for open speakers).
	InterruptWait
)

// String returns the string representation of the interrupt mode.
func (m InterruptMode) String() string {
	switch m {
	case InterruptAlways:
		return "always"
	case InterruptWait:
		return "wait"
	default:
		return "unknown"
	}
}

// ParseInterruptMode converts a string to InterruptMode.
func ParseInterruptMode(s string) (InterruptMode, error) {
	switch s {
	case "always":
		return InterruptAlways, nil
	case "wait":
		return InterruptWait, nil
	default:
		return InterruptWait, fmt.Errorf("invalid interrupt mode: %s (must be 'always' or 'wait')", s)
	}
}

// Config holds all configuration for the voice assistant.
// Populated from CLI flags, environment variables, or defaults.
type Config struct {
	// Model paths
	ModelDir string // Base directory containing all model files
	VADModel string // Path to Silero VAD model file

	// Whisper STT model paths
	WhisperEncoder string
	WhisperDecoder string
	WhisperTokens  string

	// TTS model paths (Kokoro)
	TTSModel    string // Path to model.onnx
	TTSVoices   string // Path to voices.bin
	TTSTokens   string // Path to tokens.txt
	TTSData     string // Path to espeak-ng-data
	TTSLexicon  string // Path to lexicon.txt
	TTSLanguage string // Language code for multi-lingual models (e.g., "en-gb")

	// STT settings
	STTLanguage string // Language code for speech recognition (e.g., "en", "es", "auto")

	// LLM settings
	OllamaURL    string
	OllamaModel  string
	SystemPrompt string
	MaxHistory   int     // Maximum conversation history length
	Temperature  float32 // LLM temperature (0.0-2.0, lower=deterministic, higher=creative)

	// Voice assistant settings
	WakeWord     string
	TTSVoice     string // TTS voice name (e.g., "af_bella" for American female Bella)
	TTSSpeakerID int    // Speaker ID for multi-speaker models (af_bella=2 in v1.0)
	TTSSpeed     float32
	SampleRate   int
	VadThreshold float32

	// VAD silence duration in seconds (how long to wait before considering speech ended)
	VADSilenceDuration float32

	// Hardware acceleration provider (cpu, cuda, coreml)
	// Auto-detected based on platform if empty
	Provider string

	// STT-specific provider (overrides Provider for speech recognition)
	STTProvider string

	// TTS-specific provider (overrides Provider for speech synthesis)
	TTSProvider string

	// Interrupt mode: InterruptAlways (headsets) or InterruptWait (open speakers)
	InterruptMode InterruptMode

	// Delay in milliseconds before resuming microphone after playback ends (only for InterruptWait mode)
	PostPlaybackDelayMs int

	// Thread counts for models (0 = auto-detect based on CPU cores)
	NumThreads int // Global default for all models
	VADThreads int // VAD-specific (overrides NumThreads if > 0)
	STTThreads int // STT-specific (overrides NumThreads if > 0)
	TTSThreads int // TTS-specific (overrides NumThreads if > 0)

	// Audio buffer size in milliseconds (0 = default 100ms for Bluetooth)
	// Use 20ms for wired/built-in audio (lower latency)
	// Use 100ms for Bluetooth devices (prevents distortion)
	AudioBufferMs uint32

	// Debug
	Verbose bool
}

// DefaultConfig returns a configuration with sensible defaults.
func DefaultConfig() *Config {
	homeDir, _ := os.UserHomeDir()
	defaultModelDir := filepath.Join(homeDir, ".voice-assistant", "models")

	return &Config{
		ModelDir:           defaultModelDir,
		SampleRate:         16000,
		VadThreshold:       0.5,
		VADSilenceDuration: 0.8, // Allow 800ms pauses in natural speech

		// LLM defaults
		OllamaURL:    "http://localhost:11434",
		OllamaModel:  "gemma3:1b",
		SystemPrompt: "You are a helpful voice assistant. Keep responses brief and concise, maximum 2-3 short sentences. Be conversational and natural for speech output. IMPORTANT: Your responses will be read aloud, so you must NEVER use markdown, asterisks, underscores, backticks, brackets, code blocks, bullet points, numbered lists, special characters, or any formatting. Use only plain text with normal punctuation. Speak naturally as if having a conversation.",
		MaxHistory:   10,
		Temperature:  0.7, // Default creativity level

		// TTS defaults (Kokoro af_bella American female voice - highest quality)
		TTSVoice:     "af_bella", // American female Bella (A- grade, most expressive)
		TTSSpeakerID: 2,          // Speaker ID for af_bella in Kokoro multi-lang v1.0 model
		TTSSpeed:     0.93,

		// STT defaults (Whisper multilingual)
		STTLanguage: "en", // Default to English for STT

		// No wake word by default (always listening)
		WakeWord: "",
		Verbose:  false,
		// Auto-detect provider (empty = auto)
		Provider:    "",
		STTProvider: "",
		TTSProvider: "",

		// Interrupt mode defaults
		InterruptMode:       InterruptWait,
		PostPlaybackDelayMs: 300,

		// Thread count defaults (0 = auto-detect)
		NumThreads: 0,
		VADThreads: 0,
		STTThreads: 0,
		TTSThreads: 0,

		// Audio buffer defaults (0 = 100ms, optimized for Bluetooth)
		AudioBufferMs: 0,
	}
}

// ParseFlags parses command-line flags and returns a Config.
func ParseFlags() (*Config, error) {
	cfg := DefaultConfig()

	// Voice listing flags
	listVoices := flag.Bool("list-voices", false, "List all available TTS voices and exit")
	voiceInfo := flag.String("voice-info", "", "Show detailed information about a specific voice and exit")

	// Model directory
	flag.StringVar(&cfg.ModelDir, "model-dir", cfg.ModelDir, "Directory containing model files (Whisper, VAD, TTS)")

	// Audio settings
	flag.IntVar(&cfg.SampleRate, "sample-rate", cfg.SampleRate, "Audio sample rate for speech recognition")
	vadThreshold := float64(cfg.VadThreshold)
	flag.Float64Var(&vadThreshold, "vad-threshold", vadThreshold, "Voice activity detection threshold (0.0-1.0)")
	vadSilenceDuration := float64(cfg.VADSilenceDuration)
	flag.Float64Var(&vadSilenceDuration, "vad-silence-duration", vadSilenceDuration, "VAD silence duration in seconds (how long to wait before speech is considered ended)")

	// LLM settings
	flag.StringVar(&cfg.OllamaURL, "ollama-url", cfg.OllamaURL, "Ollama API URL")
	flag.StringVar(&cfg.OllamaModel, "ollama-model", cfg.OllamaModel, "Ollama model name")
	flag.StringVar(&cfg.SystemPrompt, "system-prompt", cfg.SystemPrompt, "System prompt for the LLM")
	flag.IntVar(&cfg.MaxHistory, "max-history", cfg.MaxHistory, "Maximum conversation history length")
	temperature := float64(cfg.Temperature)
	flag.Float64Var(&temperature, "temperature", temperature, "LLM temperature (0.0-2.0). Lower values (0.1-0.3) for translation/factual tasks, higher (0.7-1.0) for creative responses")

	// TTS settings
	ttsSpeed := float64(cfg.TTSSpeed)
	flag.Float64Var(&ttsSpeed, "tts-speed", ttsSpeed, "Text-to-speech speed multiplier")
	flag.StringVar(&cfg.TTSVoice, "tts-voice", cfg.TTSVoice, "TTS voice name for Kokoro (e.g., 'bf_emma' British female). See https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models")
	flag.IntVar(&cfg.TTSSpeakerID, "tts-speaker-id", cfg.TTSSpeakerID, "TTS speaker ID for Kokoro model (bf_emma=21, af_bella=2)")

	// STT settings
	flag.StringVar(&cfg.STTLanguage, "stt-language", cfg.STTLanguage, "STT language code (e.g., 'en', 'es', 'fr', 'auto' for detection)")

	// Hardware acceleration
	flag.StringVar(&cfg.Provider, "provider", cfg.Provider, "Hardware acceleration provider (cpu, cuda, coreml). Auto-detected if not specified")
	flag.StringVar(&cfg.STTProvider, "stt-provider", cfg.STTProvider, "Provider for STT (overrides --provider for speech recognition)")
	flag.StringVar(&cfg.TTSProvider, "tts-provider", cfg.TTSProvider, "Provider for TTS (overrides --provider for speech synthesis)")

	// Thread count settings
	flag.IntVar(&cfg.NumThreads, "num-threads", cfg.NumThreads, "Number of threads for all models (0 = auto-detect based on CPU cores)")
	flag.IntVar(&cfg.VADThreads, "vad-threads", cfg.VADThreads, "VAD threads (0 = use num-threads, typically 1)")
	flag.IntVar(&cfg.STTThreads, "stt-threads", cfg.STTThreads, "STT threads (0 = use num-threads, typically cores/2)")
	flag.IntVar(&cfg.TTSThreads, "tts-threads", cfg.TTSThreads, "TTS threads (0 = use num-threads, typically cores/2)")

	// Audio settings
	audioBufferMs := flag.Uint("audio-buffer-ms", uint(cfg.AudioBufferMs), "Audio buffer size in ms (0=auto 100ms for Bluetooth, 20ms for wired/built-in)")

	// Other settings
	flag.StringVar(&cfg.WakeWord, "wake-word", cfg.WakeWord, "Wake word to activate the assistant (optional)")
	flag.BoolVar(&cfg.Verbose, "verbose", cfg.Verbose, "Enable verbose logging")

	// Interrupt mode settings
	var interruptModeStr string
	flag.StringVar(&interruptModeStr, "interrupt-mode", cfg.InterruptMode.String(), "Interrupt mode: 'always' (headsets) or 'wait' (open speakers, pauses mic during playback)")
	flag.IntVar(&cfg.PostPlaybackDelayMs, "post-playback-delay-ms", cfg.PostPlaybackDelayMs, "Delay in milliseconds before resuming mic after playback (only for 'wait' mode)")

	flag.Parse()

	// Handle voice listing commands
	if *listVoices {
		PrintVoices()
		os.Exit(0)
	}

	if *voiceInfo != "" {
		if err := PrintVoiceInfo(*voiceInfo); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		os.Exit(0)
	}

	cfg.TTSSpeed = float32(ttsSpeed)
	cfg.VadThreshold = float32(vadThreshold)
	cfg.VADSilenceDuration = float32(vadSilenceDuration)
	cfg.AudioBufferMs = uint32(*audioBufferMs)
	cfg.Temperature = float32(temperature)

	// Parse interrupt mode
	if mode, err := ParseInterruptMode(interruptModeStr); err != nil {
		return nil, err
	} else {
		cfg.InterruptMode = mode
	}

	// Auto-detect provider if not specified
	if cfg.Provider == "" {
		cfg.Provider = detectProvider()
	}

	// Auto-detect STT provider if not specified (defaults to main provider)
	if cfg.STTProvider == "" {
		cfg.STTProvider = cfg.Provider
	}

	// Auto-detect TTS provider if not specified (defaults to main provider)
	if cfg.TTSProvider == "" {
		cfg.TTSProvider = cfg.Provider
	}

	// Auto-detect and normalize thread counts
	cfg.normalizeThreadCounts()

	// Set derived paths
	cfg.VADModel = filepath.Join(cfg.ModelDir, "silero_vad.onnx")
	cfg.WhisperEncoder = filepath.Join(cfg.ModelDir, "whisper", "whisper-small-encoder.int8.onnx")
	cfg.WhisperDecoder = filepath.Join(cfg.ModelDir, "whisper", "whisper-small-decoder.int8.onnx")
	cfg.WhisperTokens = filepath.Join(cfg.ModelDir, "whisper", "whisper-small-tokens.txt")

	// Kokoro TTS model paths (multi-lang v1.0 - supports CoreML on macOS)
	ttsDir := filepath.Join(cfg.ModelDir, "tts", "kokoro-multi-lang-v1_0")
	cfg.TTSModel = filepath.Join(ttsDir, "model.onnx")
	cfg.TTSVoices = filepath.Join(ttsDir, "voices.bin")
	cfg.TTSTokens = filepath.Join(ttsDir, "tokens.txt")
	cfg.TTSData = filepath.Join(ttsDir, "espeak-ng-data")

	// Set lexicon based on voice language (required for multi-lingual Kokoro v1.0+)
	// The model includes lexicon-us-en.txt, lexicon-gb-en.txt, and lexicon-zh.txt
	cfg.TTSLexicon = getLexiconForVoice(ttsDir, cfg.TTSVoice)
	// For non-English voices without lexicon support, use espeak-ng language code
	cfg.TTSLanguage = getLanguageForVoice(cfg.TTSVoice)

	// Validate paths exist
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return cfg, nil
}

// normalizeThreadCounts auto-detects and sets reasonable thread counts based on CPU cores.
// For edge devices (Jetson Orin Nano: 6 cores), this ensures optimal performance:
// - VAD: 1 thread (lightweight)
// - STT (Whisper): cores/3 (CPU-intensive)
// - TTS (Kokoro): cores/3 (CPU-intensive)
func (c *Config) normalizeThreadCounts() {
	cpuCores := runtime.NumCPU()

	// If global NumThreads is 0, set default based on CPU cores
	if c.NumThreads == 0 {
		// For edge devices: use cores/3 as base (e.g., 6 cores -> 2 threads)
		// This leaves headroom for other tasks and prevents oversubscription
		c.NumThreads = max(1, cpuCores/3)
	}

	// Set VAD threads (typically 1, VAD is lightweight)
	if c.VADThreads == 0 {
		c.VADThreads = 1
	}

	// Set STT threads (Whisper is CPU-intensive)
	if c.STTThreads == 0 {
		c.STTThreads = c.NumThreads
	}

	// Set TTS threads (Kokoro is CPU-intensive)
	if c.TTSThreads == 0 {
		c.TTSThreads = c.NumThreads
	}

	// Log thread configuration in verbose mode
	if c.Verbose {
		fmt.Printf("[Config] CPU cores: %d, Thread counts: VAD=%d, STT=%d, TTS=%d\n",
			cpuCores, c.VADThreads, c.STTThreads, c.TTSThreads)
	}
}

func (c *Config) validate() error {
	// Interrupt mode is validated during parsing via ParseInterruptMode

	requiredFiles := []string{
		c.VADModel,
		c.WhisperEncoder,
		c.WhisperDecoder,
		c.WhisperTokens,
		c.TTSModel,
		c.TTSVoices,
		c.TTSTokens,
	}

	for _, path := range requiredFiles {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return fmt.Errorf("required file not found: %s\nRun scripts/setup.sh to download models", path)
		}
	}

	return nil
}

// detectProvider auto-detects the best hardware acceleration provider for the current platform.
func detectProvider() string {
	switch runtime.GOOS {
	case "darwin":
		// macOS: Use CoreML for Apple Neural Engine acceleration
		return "coreml"
	case "linux":
		// Linux: Check for NVIDIA GPU (discrete or Jetson SOC)
		if sherpa.HasNvidiaGPU() {
			return "cuda"
		}
		return "cpu"
	default:
		return "cpu"
	}
}

// getLexiconForVoice returns the appropriate lexicon file path based on the voice name.
// Kokoro v1.0+ multi-lingual models use lexicon files for phonemization.
// The model includes: lexicon-us-en.txt (American), lexicon-gb-en.txt (British), lexicon-zh.txt (Chinese)
//
// Voice prefixes and their lexicon files:
//   - af_*, am_*: American English -> lexicon-us-en.txt
//   - bf_*, bm_*: British English  -> lexicon-gb-en.txt
//   - zf_*, zm_*: Mandarin Chinese -> lexicon-zh.txt (combined with English)
//   - Other languages (es, fr, hi, it, ja, pt): Use espeak-ng via lang parameter
func getLexiconForVoice(ttsDir, voiceName string) string {
	voice := GetVoice(voiceName)
	if voice == nil {
		return filepath.Join(ttsDir, "lexicon-us-en.txt") // Default to American English
	}

	switch voice.EspeakCode {
	case "en-us": // American English
		return filepath.Join(ttsDir, "lexicon-us-en.txt")
	case "en-gb": // British English
		return filepath.Join(ttsDir, "lexicon-gb-en.txt")
	case "cmn": // Mandarin Chinese (with English fallback)
		return filepath.Join(ttsDir, "lexicon-us-en.txt") + "," + filepath.Join(ttsDir, "lexicon-zh.txt")
	default:
		// For other languages, return empty (will use lang parameter instead)
		return ""
	}
}

// getLanguageForVoice returns the espeak-ng language code for non-English voices.
// This is only used when lexicon files are not available for a language.
// Reference: https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-tts-kokoro-model-config.cc
//
// Voice prefixes and their language codes:
//   - ef_*, em_*: Spanish          -> "es"
//   - ff_*:       French           -> "fr"
//   - hf_*, hm_*: Hindi            -> "hi"
//   - if_*, im_*: Italian          -> "it"
//   - jf_*, jm_*: Japanese         -> "ja"
//   - pf_*, pm_*: Portuguese BR    -> "pt-br"
func getLanguageForVoice(voiceName string) string {
	voice := GetVoice(voiceName)
	if voice == nil {
		return "" // Use lexicon for English
	}

	// English and Chinese use lexicon files, return empty for them
	if voice.EspeakCode == "en-us" || voice.EspeakCode == "en-gb" || voice.EspeakCode == "cmn" {
		return ""
	}

	return voice.EspeakCode
}

// PrintVoices and PrintVoiceInfo are now in voices.go with simpler implementation
