//! Application configuration and CLI argument parsing.
//!
//! [`AppConfig`] holds only generic pipeline parameters. Model-specific
//! configuration (paths, voices, validation) lives in the STT and TTS
//! implementations.

use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use tracing::info;

/// Hardware acceleration provider for ONNX models.
/// Auto-detected based on platform if not specified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// CPU inference (default fallback, always available)
    #[default]
    Cpu,
    /// NVIDIA CUDA acceleration (Linux only, requires CUDA toolkit)
    Cuda,
    /// Apple CoreML acceleration (macOS only, uses Neural Engine)
    #[value(name = "coreml")]
    CoreMl,
}

/// Interrupt mode for handling playback during speech detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InterruptMode {
    /// Always allow interrupts (best for headsets)
    Always,
    /// Pause microphone during playback, resume after (default for open speakers)
    #[default]
    Wait,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Cpu => write!(f, "cpu"),
            Provider::Cuda => write!(f, "cuda"),
            Provider::CoreMl => write!(f, "coreml"),
        }
    }
}

impl Provider {
    /// Convert to sherpa-rs provider string.
    ///
    /// # Returns
    /// Provider string compatible with sherpa-rs library.
    pub fn as_sherpa_provider(&self) -> &'static str {
        match self {
            Provider::Cpu => "cpu",
            Provider::Cuda => "cuda",
            Provider::CoreMl => "coreml",
        }
    }
}

/// Voice assistant application configuration.
#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(name = "voice-assistant")]
#[command(author, version, about = "A real-time voice assistant", long_about = None)]
pub struct AppConfig {
    /// Download required model files then exit (idempotent, safe to re-run)
    #[arg(long)]
    pub setup: bool,

    /// Force re-download of model files even if they already exist (use with --setup)
    #[arg(long)]
    pub force: bool,

    /// List all available TTS voices and exit
    #[arg(long)]
    pub list_voices: bool,

    /// Show detailed information about a specific voice and exit
    #[arg(long)]
    pub voice_info: Option<String>,

    /// Directory containing model files (Whisper, VAD, TTS)
    #[arg(long, short = 'd', env = "MODEL_DIR", default_value_os_t = default_model_dir())]
    pub model_dir: PathBuf,

    /// STT backend implementation (e.g. 'whisper')
    #[arg(long, default_value = "whisper")]
    pub stt_backend: String,

    /// TTS backend implementation (e.g. 'kokoro')
    #[arg(long, default_value = "kokoro")]
    pub tts_backend: String,

    /// Audio sample rate for speech recognition
    #[arg(long, default_value = "16000")]
    pub sample_rate: u32,

    /// Voice activity detection threshold (0.0 - 1.0)
    #[arg(long, default_value = "0.5")]
    pub vad_threshold: f32,

    /// VAD silence duration in seconds (how long to wait before considering speech ended)
    #[arg(long, default_value = "0.8")]
    pub vad_silence_duration: f32,

    /// Ollama API URL
    #[arg(long, short = 'u', env = "OLLAMA_URL", default_value = "http://localhost:11434")]
    pub ollama_url: String,

    /// Ollama model name (qwen2.5:1.5b recommended - multilingual + function calling support)
    #[arg(long, short = 'm', env = "OLLAMA_MODEL", default_value = "qwen2.5:1.5b")]
    pub ollama_model: String,

    /// SearXNG instance URL for web search (optional, uses DuckDuckGo if not provided)
    #[arg(long, env = "SEARXNG_URL")]
    pub searxng_url: Option<String>,

    /// System prompt for the LLM
    #[arg(
        long,
        short = 'p',
        default_value = "You are a helpful voice assistant. Keep responses brief and concise, maximum 2-3 short sentences. Be conversational and natural for speech output. Never use emojis, markdown formatting, bullet points, numbered lists, or special characters. Use plain spoken language only."
    )]
    pub system_prompt: String,

    /// Text-to-speech speed multiplier (0.9-0.95 for more natural, expressive speech)
    #[arg(long, default_value = "0.93")]
    pub tts_speed: f32,

    /// TTS voice name for Kokoro (e.g., af_bella for high-quality American female).
    /// Top voices: af_bella (A-), af_heart, af_nicole, bf_emma
    /// See <https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md>
    #[arg(long, default_value = "af_bella")]
    pub tts_voice: String,

    /// TTS speaker ID for Kokoro model (af_bella=2 in v1.0, bf_emma=21)
    #[arg(long, default_value = "2")]
    pub tts_speaker_id: i32,

    /// STT language code (e.g., en, es, fr, de, it, pt, zh, ja, ko, ru)
    /// Use "auto" for automatic language detection
    #[arg(long, default_value = "en")]
    pub stt_language: String,

    /// STT model identifier (e.g. tiny, base, small)
    #[arg(long, default_value = "tiny")]
    pub stt_model: String,

    /// Hardware acceleration provider (auto-detected if not specified)
    #[arg(long, value_enum)]
    pub provider: Option<Provider>,

    /// Provider for STT (overrides --provider for speech recognition)
    #[arg(long, value_enum)]
    pub stt_provider: Option<Provider>,

    /// Provider for TTS (overrides --provider for speech synthesis)
    #[arg(long, value_enum)]
    pub tts_provider: Option<Provider>,

    /// Enable verbose logging
    #[arg(long, short = 'v')]
    pub verbose: bool,

    /// Wake word to activate the assistant (optional)
    #[arg(long, short = 'w')]
    pub wake_word: Option<String>,

    /// Maximum conversation history length
    #[arg(long, default_value = "10")]
    pub max_history: usize,

    /// LLM temperature (0.0-2.0). Lower for translation/factual (0.1-0.3), higher for creative (0.7-1.0)
    #[arg(long, default_value = "0.7", value_parser = parse_temperature)]
    pub temperature: f32,

    /// Interrupt mode: 'always' allows interrupts (headsets), 'wait' pauses mic during playback (open speakers)
    #[arg(long, value_enum, default_value = "wait")]
    pub interrupt_mode: InterruptMode,

    /// Delay in milliseconds before resuming microphone after playback ends (only for 'wait' mode)
    #[arg(long, default_value = "300")]
    pub post_playback_delay_ms: u64,

    /// Number of threads for all models (0 = auto-detect based on CPU cores)
    #[arg(long, default_value = "0")]
    pub num_threads: usize,

    /// VAD threads (0 = use num_threads, typically 1)
    #[arg(long, default_value = "0")]
    pub vad_threads: usize,

    /// STT threads (0 = use num_threads, typically cores/3)
    #[arg(long, default_value = "0")]
    pub stt_threads: usize,

    /// TTS threads (0 = use num_threads, typically cores/3)
    #[arg(long, default_value = "0")]
    pub tts_threads: usize,
}

impl AppConfig {
    /// Parse configuration from command line arguments.
    ///
    /// Voice listing flags (`--list-voices`, `--voice-info`) are set as fields
    /// but **not** handled here — the caller (main) dispatches them after parsing.
    pub fn from_args() -> Self {
        let mut config = Self::parse();
        config.normalize_thread_counts();
        config
    }

    /// Auto-detect and normalize thread counts based on CPU cores and provider.
    ///
    /// **Important**: When using CUDA/GPU acceleration, fewer threads (typically 1) should be used
    /// because the GPU handles parallelism internally. Multiple CPU threads with GPU inference
    /// can cause resource contention and CUDA allocation failures.
    ///
    /// For edge devices (Jetson Orin Nano: 6 cores), this ensures optimal performance:
    /// - With CUDA: 1 thread for all models (GPU handles parallelism)
    /// - With CPU: cores/3 threads for STT/TTS, 1 for VAD
    fn normalize_thread_counts(&mut self) {
        let cpu_cores = num_cpus::get();
        let using_cuda = self.effective_stt_provider() == Provider::Cuda || self.effective_tts_provider() == Provider::Cuda;

        // If global num_threads is 0, set default based on CPU cores and provider
        if self.num_threads == 0 {
            if using_cuda {
                // When using CUDA, use 1 thread (GPU handles parallelism)
                self.num_threads = 1;
            } else {
                // For CPU: use cores/3 as base (e.g., 6 cores -> 2 threads)
                // This leaves headroom for other tasks and prevents oversubscription
                self.num_threads = (cpu_cores / 3).max(1);
            }
        }

        // Set VAD threads (typically 1, VAD is lightweight)
        if self.vad_threads == 0 {
            self.vad_threads = 1;
        }

        // Set STT threads (Whisper is CPU-intensive on CPU, but use 1 for CUDA)
        if self.stt_threads == 0 {
            self.stt_threads = if self.effective_stt_provider() == Provider::Cuda { 1 } else { self.num_threads };
        }

        // Set TTS threads (Kokoro is CPU-intensive on CPU, but use 1 for CUDA)
        if self.tts_threads == 0 {
            self.tts_threads = if self.effective_tts_provider() == Provider::Cuda { 1 } else { self.num_threads };
        }

        // Log thread configuration
        if self.verbose {
            info!(
                "CPU cores: {}, Provider: STT={}, TTS={}, Thread counts: VAD={}, STT={}, TTS={}",
                cpu_cores,
                self.effective_stt_provider(),
                self.effective_tts_provider(),
                self.vad_threads,
                self.stt_threads,
                self.tts_threads
            );
        }
    }

    /// Get the effective STT provider.
    pub fn effective_stt_provider(&self) -> Provider {
        self.stt_provider.or(self.provider).unwrap_or_else(detect_provider)
    }

    /// Get the effective TTS provider.
    ///
    /// Kokoro TTS supports CoreML on macOS and CUDA on Linux.
    #[allow(clippy::unnecessary_lazy_evaluations)]
    pub fn effective_tts_provider(&self) -> Provider {
        self.tts_provider.or(self.provider).unwrap_or_else(detect_provider)
    }

    /// Log the current configuration.
    pub fn log_config(&self) {
        info!("Configuration:");
        info!("  Model directory: {}", self.model_dir.display());
        info!("  Sample rate: {} Hz", self.sample_rate);
        info!("  VAD threshold: {}", self.vad_threshold);
        info!("  Ollama URL: {}", self.ollama_url);
        info!("  Ollama model: {}", self.ollama_model);
        info!("  System prompt: {}...", &self.system_prompt.chars().take(50).collect::<String>());
        info!("  STT backend: {}", self.stt_backend);
        info!("  TTS backend: {}", self.tts_backend);
        info!("  TTS voice: {}", self.tts_voice);
        info!("  TTS speed: {}", self.tts_speed);
        info!("  STT language: {}", self.stt_language);
        info!("  STT provider: {}", self.effective_stt_provider());
        info!("  TTS provider: {}", self.effective_tts_provider());
        if let Some(ref wake_word) = self.wake_word {
            info!("  Wake word: {}", wake_word);
        }
        info!("  Interrupt mode: {:?}", self.interrupt_mode);
        if matches!(self.interrupt_mode, InterruptMode::Wait) {
            info!("  Post-playback delay: {}ms", self.post_playback_delay_ms);
        }
    }
}

/// Get the default model directory (~/.voice-assistant/models).
fn default_model_dir() -> PathBuf {
    if let Some(home_dir) = dirs::home_dir() {
        home_dir.join(".voice-assistant").join("models")
    } else {
        PathBuf::from("models")
    }
}

/// Auto-detect the best hardware acceleration provider.
fn detect_provider() -> Provider {
    #[cfg(target_os = "macos")]
    {
        info!("Detected macOS, using CoreML provider");
        Provider::CoreMl
    }

    #[cfg(target_os = "linux")]
    {
        if has_nvidia_gpu() {
            info!("Detected NVIDIA GPU, using CUDA provider");
            Provider::Cuda
        } else {
            info!("No GPU detected, using CPU provider");
            Provider::Cpu
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        info!("Using CPU provider");
        Provider::Cpu
    }
}

/// Check if an NVIDIA GPU is available (Linux only).
#[cfg(target_os = "linux")]
fn has_nvidia_gpu() -> bool {
    use std::path::Path;

    // Check for NVIDIA device files
    let nvidia_paths = [
        "/dev/nvidia0",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        // Jetson devices
        "/dev/nvhost-ctrl",
        "/dev/nvhost-ctrl-gpu",
    ];

    for path in &nvidia_paths {
        if Path::new(path).exists() {
            return true;
        }
    }

    // Check for Tegra (Jetson) devices
    if Path::new("/etc/nv_tegra_release").exists() {
        return true;
    }

    false
}

/// Parse and validate temperature value (0.0-2.0).
fn parse_temperature(s: &str) -> Result<f32, String> {
    let value: f32 = s.parse().map_err(|_| format!("'{}' is not a valid float", s))?;
    if (0.0..=2.0).contains(&value) {
        Ok(value)
    } else {
        Err(format!("temperature must be between 0.0 and 2.0, got {}", value))
    }
}
