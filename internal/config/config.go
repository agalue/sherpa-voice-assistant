// Package config provides configuration and CLI argument parsing for the voice assistant.
//
// Config holds only generic pipeline parameters. Model-specific configuration
// (paths, voices, validation) lives in the STT and TTS implementations.
package config

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/agalue/sherpa-voice-assistant/internal/sherpa"
)

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
//
// Model-specific paths (Whisper encoder/decoder, Kokoro model/voices, etc.) are
// intentionally absent — each implementation derives them from [Config.ModelDir]
// and [Config.STTModel] internally.
type Config struct {
	// Model directory (base for all model files)
	ModelDir string

	// Backend selection (which STT/TTS implementation to use)
	STTBackend string // STT backend (e.g. "whisper"); selects the Transcriber implementation
	TTSBackend string // TTS backend (e.g. "kokoro"); selects the Synthesizer implementation

	// STT settings (generic — implementations interpret STTModel in their own way)
	STTModel    string // STT model identifier (e.g. "tiny", "base", "small")
	STTLanguage string // Language code for speech recognition (e.g., "en", "es", "auto")

	// LLM settings
	OllamaURL    string
	OllamaModel  string
	SystemPrompt string
	MaxHistory   int     // Maximum conversation history length
	Temperature  float32 // LLM temperature (0.0-2.0, lower=deterministic, higher=creative)
	SearxngURL   string  // Optional SearXNG URL for web search (empty uses DuckDuckGo)

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

	// Setup flags (not persistent at runtime; used during --setup invocation)
	Setup bool // Download model files and exit
	Force bool // Re-download even if model files already exist

	// Informational flags (handled in main, not here)
	ListVoices bool   // List all available TTS voices and exit
	VoiceInfo  string // Show details for a specific voice and exit
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
		OllamaModel:  "qwen2.5:1.5b",
		SystemPrompt: "You are a helpful voice assistant. Keep responses brief and concise, maximum 2-3 short sentences. Be conversational and natural for speech output. IMPORTANT: Your responses will be read aloud, so you must NEVER use markdown, asterisks, underscores, backticks, brackets, code blocks, bullet points, numbered lists, special characters, or any formatting. Use only plain text with normal punctuation. Speak naturally as if having a conversation.",
		MaxHistory:   10,
		Temperature:  0.7, // Default creativity level
		SearxngURL:   "",  // Empty = use DuckDuckGo fallback

		// TTS defaults (voice name and speaker ID are generic TTS concepts)
		TTSVoice:     "af_bella", // Default voice
		TTSSpeakerID: 2,          // Default speaker ID
		TTSSpeed:     0.93,

		// STT defaults
		STTBackend:  "whisper",      // Default STT backend
		TTSBackend:  "kokoro",       // Default TTS backend
		STTModel:    "tiny", // Default STT model name (e.g. "tiny", "base", "small")
		STTLanguage: "en",           // Default to English for STT

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

	// Informational flags (handled by the caller after ParseFlags returns)
	flag.BoolVar(&cfg.ListVoices, "list-voices", false, "List all available TTS voices and exit")
	flag.StringVar(&cfg.VoiceInfo, "voice-info", "", "Show detailed information about a specific voice and exit")

	// Setup flags
	flag.BoolVar(&cfg.Setup, "setup", false, "Download required model files then exit (idempotent, safe to re-run)")
	flag.BoolVar(&cfg.Force, "force", false, "Force re-download of model files even if they already exist (use with --setup)")

	// Model directory
	flag.StringVar(&cfg.ModelDir, "model-dir", cfg.ModelDir, "Base directory for all model files")

	// Audio settings
	flag.IntVar(&cfg.SampleRate, "sample-rate", cfg.SampleRate, "Audio sample rate for speech recognition")
	vadThreshold := float64(cfg.VadThreshold)
	flag.Float64Var(&vadThreshold, "vad-threshold", vadThreshold, "Voice activity detection threshold (0.0-1.0)")
	vadSilenceDuration := float64(cfg.VADSilenceDuration)
	flag.Float64Var(&vadSilenceDuration, "vad-silence-duration", vadSilenceDuration, "VAD silence duration in seconds (how long to wait before speech is considered ended)")

	// LLM settings
	flag.StringVar(&cfg.OllamaURL, "ollama-url", cfg.OllamaURL, "Ollama API URL")
	flag.StringVar(&cfg.OllamaModel, "ollama-model", cfg.OllamaModel, "Ollama model name (must support tool calling, e.g., qwen2.5:1.5b, qwen2.5:3b)")
	flag.StringVar(&cfg.SystemPrompt, "system-prompt", cfg.SystemPrompt, "System prompt for the LLM")
	flag.IntVar(&cfg.MaxHistory, "max-history", cfg.MaxHistory, "Maximum conversation history length")
	temperature := float64(cfg.Temperature)
	flag.Float64Var(&temperature, "temperature", temperature, "LLM temperature (0.0-2.0). Lower values (0.1-0.3) for translation/factual tasks, higher (0.7-1.0) for creative responses")
	flag.StringVar(&cfg.SearxngURL, "searxng-url", cfg.SearxngURL, "Optional SearXNG URL for web search (empty uses DuckDuckGo fallback)")

	// TTS settings
	ttsSpeed := float64(cfg.TTSSpeed)
	flag.Float64Var(&ttsSpeed, "tts-speed", ttsSpeed, "Text-to-speech speed multiplier")
	flag.StringVar(&cfg.TTSVoice, "tts-voice", cfg.TTSVoice, "TTS voice name (e.g., 'bf_emma', 'af_bella')")
	flag.IntVar(&cfg.TTSSpeakerID, "tts-speaker-id", cfg.TTSSpeakerID, "TTS speaker ID (bf_emma=21, af_bella=2)")

	// Backend selection
	flag.StringVar(&cfg.STTBackend, "stt-backend", cfg.STTBackend, "STT backend implementation (e.g. 'whisper')")
	flag.StringVar(&cfg.TTSBackend, "tts-backend", cfg.TTSBackend, "TTS backend implementation (e.g. 'kokoro')")

	// STT settings
	flag.StringVar(&cfg.STTModel, "stt-model", cfg.STTModel, "STT model identifier (e.g. tiny, base, small)")
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

	cfg.TTSSpeed = float32(ttsSpeed)
	cfg.VadThreshold = float32(vadThreshold)
	cfg.VADSilenceDuration = float32(vadSilenceDuration)
	cfg.AudioBufferMs = uint32(*audioBufferMs)
	cfg.Temperature = float32(temperature)

	// Validate numeric ranges
	if cfg.Temperature < 0.0 || cfg.Temperature > 2.0 {
		return nil, fmt.Errorf("temperature must be between 0.0 and 2.0, got %.2f", cfg.Temperature)
	}

	if cfg.VadThreshold < 0.0 || cfg.VadThreshold > 1.0 {
		return nil, fmt.Errorf("vad-threshold must be between 0.0 and 1.0, got %.2f", cfg.VadThreshold)
	}

	if cfg.TTSSpeed <= 0.0 {
		return nil, fmt.Errorf("tts-speed must be positive, got %.2f", cfg.TTSSpeed)
	}

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
