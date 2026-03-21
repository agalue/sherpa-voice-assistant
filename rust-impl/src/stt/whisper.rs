//! Whisper speech transcriber.
//!
//! [`WhisperRecognizer`] implements [`super::Transcriber`] using the OpenAI Whisper
//! model via sherpa-rs. It is intentionally decoupled from VAD — pair with
//! [`crate::stt::SileroVad`] to form a complete STT pipeline.
//!
//! Model downloads are handled by [`WhisperModelProvider`].

use std::path::PathBuf;

use anyhow::Result;
use parking_lot::Mutex;
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer as SherpaWhisperRecognizer};
use tracing::{debug, info};

use super::Transcriber;
use crate::config::AppConfig;

/// Whisper-based speech transcriber implementing [`Transcriber`].
///
/// Inference is slow (100–500 ms per segment) and **must not** be called from the
/// audio callback thread. The underlying ONNX runtime is not thread-safe, so a
/// mutex serialises access.
pub struct WhisperRecognizer {
    whisper: Mutex<SherpaWhisperRecognizer>,
    pub(crate) sample_rate: u32,
    pub(crate) wake_word: Option<String>,
}

impl WhisperRecognizer {
    /// Create a new Whisper transcriber.
    ///
    /// # Errors
    /// Returns an error if the model files are missing or Whisper fails to initialise.
    pub fn new(config: &AppConfig) -> Result<Self> {
        let provider = config.effective_stt_provider();

        // Derive Whisper model paths from model_dir + stt_model.
        // STT model is fully-qualified (e.g. "whisper-tiny"); strip the prefix.
        let model_size = config.stt_model.strip_prefix("whisper-").unwrap_or(&config.stt_model);
        let whisper_dir = config.model_dir.join("whisper");
        let encoder_path = whisper_dir.join(format!("whisper-{model_size}-encoder.int8.onnx")).to_string_lossy().to_string();
        let decoder_path = whisper_dir.join(format!("whisper-{model_size}-decoder.int8.onnx")).to_string_lossy().to_string();
        let tokens_path = whisper_dir.join(format!("whisper-{model_size}-tokens.txt")).to_string_lossy().to_string();

        // "auto" → "" triggers Whisper's built-in language detection.
        let stt_language = if config.stt_language.eq_ignore_ascii_case("auto") {
            String::new()
        } else {
            config.stt_language.clone()
        };

        let whisper_config = WhisperConfig {
            encoder: encoder_path,
            decoder: decoder_path,
            tokens: tokens_path,
            language: stt_language,
            provider: Some(provider.as_sherpa_provider().to_string()),
            num_threads: Some(config.stt_threads.try_into().unwrap_or(2)),
            debug: config.verbose,
            ..Default::default()
        };

        let whisper = SherpaWhisperRecognizer::new(whisper_config).map_err(|e| anyhow::anyhow!("Failed to initialize Whisper: {}", e))?;

        info!("Whisper recognizer initialized successfully");

        Ok(Self {
            whisper: Mutex::new(whisper),
            sample_rate: config.sample_rate,
            wake_word: config.wake_word.clone(),
        })
    }
}

impl Transcriber for WhisperRecognizer {
    fn transcribe_segment(&self, samples: &[f32]) -> Option<String> {
        if samples.is_empty() {
            debug!("Empty speech segment");
            return None;
        }

        debug!("Transcribing {} samples", samples.len());

        let mut whisper = self.whisper.lock();
        let result = whisper.transcribe(self.sample_rate, samples);
        drop(whisper);

        let text = result.text.trim().to_string();
        if text.is_empty() {
            debug!("Empty transcription result");
            return None;
        }

        if let Some(ref wake_word) = self.wake_word {
            if !text.to_lowercase().contains(&wake_word.to_lowercase()) {
                debug!("Wake word '{}' not detected in '{}', ignoring", wake_word, text);
                return None;
            }
            let cleaned = text
                .to_lowercase()
                .replace(&wake_word.to_lowercase(), "")
                .trim_start_matches(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
                .trim()
                .to_string();

            if cleaned.is_empty() {
                info!("🗣️ Wake word '{}' detected", wake_word);
                return Some("Hello".to_string());
            }

            info!("🗣️ You (wake word detected): {}", cleaned);
            return Some(cleaned);
        }

        info!("🗣️ You: {}", text);
        Some(text)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelProvider
// ─────────────────────────────────────────────────────────────────────────────

/// Manages Whisper model files (encoder, decoder, tokens).
///
/// VAD model files are handled separately by [`crate::stt::SileroModelProvider`].
pub struct WhisperModelProvider {
    /// Whisper model variant (e.g. `"tiny"`, `"base"`, `"small"`).
    pub model_size: String,
}

impl super::ModelProvider for WhisperModelProvider {
    fn name(&self) -> &'static str {
        "Whisper"
    }

    /// Download missing Whisper model files.
    ///
    /// # Errors
    /// Returns an error if any download or extraction fails.
    fn ensure_models(&self, model_dir: &std::path::Path, force: bool) -> anyhow::Result<()> {
        use crate::models;

        let whisper_dir = model_dir.join("whisper");
        std::fs::create_dir_all(&whisper_dir)?;

        let encoder = whisper_dir.join(format!("whisper-{}-encoder.int8.onnx", self.model_size));
        if !force && encoder.exists() {
            info!("[STT] Whisper {} already present, skipping", self.model_size);
            return Ok(());
        }

        let url = format!("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-{}.tar.bz2", self.model_size);
        info!("[STT] Downloading Whisper {} from {} …", self.model_size, url);

        let want_files = vec![
            (
                format!("sherpa-onnx-whisper-{}/{}-encoder.int8.onnx", self.model_size, self.model_size),
                whisper_dir.join(format!("whisper-{}-encoder.int8.onnx", self.model_size)),
            ),
            (
                format!("sherpa-onnx-whisper-{}/{}-decoder.int8.onnx", self.model_size, self.model_size),
                whisper_dir.join(format!("whisper-{}-decoder.int8.onnx", self.model_size)),
            ),
            (
                format!("sherpa-onnx-whisper-{}/{}-tokens.txt", self.model_size, self.model_size),
                whisper_dir.join(format!("whisper-{}-tokens.txt", self.model_size)),
            ),
        ];

        models::extract_tar_bz2_selected(&url, &want_files)?;
        Ok(())
    }

    fn verify_models(&self, model_dir: &std::path::Path) -> Vec<PathBuf> {
        let whisper_dir = model_dir.join("whisper");
        let required = vec![
            whisper_dir.join(format!("whisper-{}-encoder.int8.onnx", self.model_size)),
            whisper_dir.join(format!("whisper-{}-decoder.int8.onnx", self.model_size)),
            whisper_dir.join(format!("whisper-{}-tokens.txt", self.model_size)),
        ];
        required.into_iter().filter(|p| !p.exists()).collect()
    }
}
