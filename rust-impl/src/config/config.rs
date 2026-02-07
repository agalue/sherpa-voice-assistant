//! Application configuration and CLI argument parsing.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use tracing::info;

// Voice data is now in voices.rs - simpler struct with just essential fields.
use super::voices;

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
    /// List all available TTS voices and exit
    #[arg(long)]
    pub list_voices: bool,

    /// Show detailed information about a specific voice and exit
    #[arg(long)]
    pub voice_info: Option<String>,

    /// Directory containing model files (Whisper, VAD, TTS)
    #[arg(long, short = 'd', env = "MODEL_DIR", default_value_os_t = default_model_dir())]
    pub model_dir: PathBuf,

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

    /// Ollama model name
    #[arg(long, short = 'm', env = "OLLAMA_MODEL", default_value = "gemma3:1b")]
    pub ollama_model: String,

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
    pub fn from_args() -> Self {
        let mut config = Self::parse();

        // Handle voice listing commands
        if config.list_voices {
            voices::print_voices();
            std::process::exit(0);
        }

        if let Some(ref voice_name) = config.voice_info {
            match voices::print_voice_info(voice_name) {
                Ok(_) => std::process::exit(0),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }

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
        self.tts_provider.or(self.provider).unwrap_or_else(|| {
            // Kokoro TTS supports CoreML acceleration on macOS
            detect_provider()
        })
    }

    /// Get the path to the Whisper encoder model (multilingual).
    pub fn whisper_encoder_path(&self) -> PathBuf {
        self.model_dir.join("whisper").join("whisper-small-encoder.int8.onnx")
    }

    /// Get the path to the Whisper decoder model (multilingual).
    pub fn whisper_decoder_path(&self) -> PathBuf {
        self.model_dir.join("whisper").join("whisper-small-decoder.int8.onnx")
    }

    /// Get the path to the Whisper tokens file (multilingual).
    pub fn whisper_tokens_path(&self) -> PathBuf {
        self.model_dir.join("whisper").join("whisper-small-tokens.txt")
    }

    /// Get the effective STT language code for Whisper.
    /// Returns empty string for auto-detection, otherwise the language code.
    pub fn effective_stt_language(&self) -> &str {
        if self.stt_language.eq_ignore_ascii_case("auto") {
            "" // Empty string triggers auto-detection in Whisper
        } else {
            &self.stt_language
        }
    }

    /// Get the path to the VAD model.
    pub fn vad_model_path(&self) -> PathBuf {
        self.model_dir.join("silero_vad.onnx")
    }

    /// Get the path to the Kokoro TTS model (multi-lang v1.0 - supports CoreML).
    pub fn tts_model_path(&self) -> PathBuf {
        self.model_dir.join("tts").join("kokoro-multi-lang-v1_0").join("model.onnx")
    }

    /// Get the path to the Kokoro TTS voices.bin file.
    pub fn tts_voices_path(&self) -> PathBuf {
        self.model_dir.join("tts").join("kokoro-multi-lang-v1_0").join("voices.bin")
    }

    /// Get the path to the TTS tokens file.
    pub fn tts_tokens_path(&self) -> PathBuf {
        self.model_dir.join("tts").join("kokoro-multi-lang-v1_0").join("tokens.txt")
    }

    /// Get the path to the TTS data directory.
    pub fn tts_data_dir(&self) -> PathBuf {
        self.model_dir.join("tts").join("kokoro-multi-lang-v1_0").join("espeak-ng-data")
    }

    /// Get the path to the TTS dict directory (for Chinese segmentation).
    pub fn tts_dict_dir(&self) -> PathBuf {
        self.model_dir.join("tts").join("kokoro-multi-lang-v1_0").join("dict")
    }

    /// Get the lexicon file path for Kokoro TTS based on voice name.
    /// The model includes lexicon-us-en.txt (American), lexicon-gb-en.txt (British), lexicon-zh.txt (Chinese)
    /// For English/Chinese, use lexicon files. For other languages, return empty (use lang instead).
    pub fn tts_lexicon(&self) -> String {
        let tts_dir = self.model_dir.join("tts").join("kokoro-multi-lang-v1_0");
        if self.tts_voice.len() >= 2 {
            match &self.tts_voice[..2] {
                "af" | "am" => tts_dir.join("lexicon-us-en.txt").to_string_lossy().to_string(),
                "bf" | "bm" => tts_dir.join("lexicon-gb-en.txt").to_string_lossy().to_string(),
                "zf" | "zm" => {
                    // Chinese with English fallback
                    format!("{},{}", tts_dir.join("lexicon-us-en.txt").to_string_lossy(), tts_dir.join("lexicon-zh.txt").to_string_lossy())
                }
                _ => String::new(), // Other languages use lang parameter
            }
        } else {
            tts_dir.join("lexicon-us-en.txt").to_string_lossy().to_string() // Default
        }
    }

    /// Get the language code for non-English voices that need espeak-ng.
    /// For English/Chinese, lexicon files are used instead.
    /// Reference: <https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-tts-kokoro-model-config.cc>
    pub fn tts_language(&self) -> &str {
        if self.tts_voice.len() >= 2 {
            match &self.tts_voice[..2] {
                "ef" | "em" => "es",    // Spanish
                "ff" => "fr",           // French
                "hf" | "hm" => "hi",    // Hindi
                "if" | "im" => "it",    // Italian
                "jf" | "jm" => "ja",    // Japanese
                "pf" | "pm" => "pt-br", // Portuguese BR
                _ => "",                // English/Chinese use lexicon files
            }
        } else {
            "" // Default (use lexicon)
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Check model directory exists
        if !self.model_dir.exists() {
            anyhow::bail!("Model directory does not exist: {}", self.model_dir.display());
        }

        // Check required model files
        let required_files = [
            self.whisper_encoder_path(),
            self.whisper_decoder_path(),
            self.whisper_tokens_path(),
            self.vad_model_path(),
            self.tts_model_path(),
            self.tts_voices_path(),
            self.tts_tokens_path(),
        ];

        for path in &required_files {
            if !path.exists() {
                anyhow::bail!("Required model file not found: {}", path.display());
            }
        }

        // Validate numeric ranges
        if !(0.0..=1.0).contains(&self.vad_threshold) {
            anyhow::bail!("VAD threshold must be between 0.0 and 1.0");
        }

        if self.tts_speed <= 0.0 {
            anyhow::bail!("TTS speed must be positive");
        }

        Ok(())
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
