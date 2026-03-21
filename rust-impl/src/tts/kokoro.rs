//! Kokoro-based TTS synthesizer.
//!
//! [`KokoroSynthesizer`] implements [`super::SpeechSynthesizer`] using the Kokoro
//! multi-lingual TTS model via sherpa-rs.  [`KokoroModelProvider`] implements
//! [`super::ModelProvider`] so the binary can bootstrap its own model files.

use anyhow::Result;
use sherpa_rs::OnnxConfig;
use sherpa_rs::tts::{CommonTtsConfig, KokoroTts, KokoroTtsConfig};
use tracing::{debug, info};

use super::SpeechSynthesizer;
use crate::config::AppConfig;

/// Kokoro-based TTS synthesizer.
///
/// Implements [`SpeechSynthesizer`]. Kokoro uses a 24 kHz sample rate, supports
/// multiple languages, and requires separate lexicon files for English and Chinese.
pub struct KokoroSynthesizer {
    tts: KokoroTts,   // Kokoro TTS engine
    sample_rate: u32, // Output sample rate (24 kHz for Kokoro)
    speaker_id: i32,  // Speaker/voice identifier
    speed: f32,       // Speech speed multiplier
}

impl KokoroSynthesizer {
    /// Create a new Kokoro TTS synthesizer from application configuration.
    ///
    /// # Arguments
    /// * `config` - Application configuration
    ///
    /// # Returns
    /// A new `KokoroSynthesizer` instance.
    ///
    /// # Errors
    /// Returns an error if TTS initialisation fails (e.g. missing model files).
    pub fn new(config: &AppConfig) -> Result<Self> {
        let provider = config.effective_tts_provider();

        info!("Initializing Kokoro TTS synthesizer with {} provider", provider);
        info!("TTS voice: {} (speaker ID: {})", config.tts_voice, config.tts_speaker_id);

        let tts_config = KokoroTtsConfig {
            model: config.tts_model_path().to_string_lossy().to_string(),
            voices: config.tts_voices_path().to_string_lossy().to_string(),
            tokens: config.tts_tokens_path().to_string_lossy().to_string(),
            data_dir: config.tts_data_dir().to_string_lossy().to_string(),
            dict_dir: config.tts_dict_dir().to_string_lossy().to_string(),
            lexicon: config.tts_lexicon(),           // Lexicon files for English/Chinese voices
            lang: config.tts_language().to_string(), // For non-English voices without lexicon
            length_scale: 1.0 / config.tts_speed,    // length_scale is inverse of speed
            onnx_config: OnnxConfig {
                provider: provider.as_sherpa_provider().to_string(),
                num_threads: config.tts_threads.try_into().unwrap_or(2),
                debug: config.verbose,
            },
            common_config: CommonTtsConfig { max_num_sentences: 1, ..Default::default() }, // Kokoro only supports 1
        };

        let tts = KokoroTts::new(tts_config);

        // Kokoro uses 24000 Hz sample rate
        let sample_rate = 24000_u32;
        info!("TTS sample rate: {} Hz", sample_rate);

        Ok(Self { tts, sample_rate, speaker_id: config.tts_speaker_id, speed: config.tts_speed })
    }
}

impl SpeechSynthesizer for KokoroSynthesizer {
    /// Synthesise a single sentence — implements [`SpeechSynthesizer`].
    ///
    /// # Arguments
    /// * `sentence` - The sentence to synthesise
    ///
    /// # Returns
    /// Audio samples or an error.
    ///
    /// # Errors
    /// Returns an error if TTS generation fails.
    fn synthesize_sentence(&mut self, sentence: &str) -> Result<Vec<f32>> {
        if sentence.trim().is_empty() {
            return Ok(Vec::new());
        }

        debug!("Synthesizing sentence: \"{}\"", sentence);

        let audio = self.tts.create(sentence, self.speaker_id, self.speed).map_err(|e| anyhow::anyhow!("TTS generation failed: {}", e))?;

        info!("Generated speech ({} samples)", audio.samples.len());
        Ok(audio.samples)
    }

    /// Returns the sample rate of the Kokoro synthesizer (24000 Hz).
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelProvider
// ─────────────────────────────────────────────────────────────────────────────

/// Manages model files for the Kokoro TTS backend.
///
/// Embeds the download logic so the binary can bootstrap its own model files
/// without an external shell script.
pub struct KokoroModelProvider;

impl super::ModelProvider for KokoroModelProvider {
    fn name(&self) -> &'static str {
        "Kokoro"
    }

    /// Download missing Kokoro TTS model files.
    ///
    /// Fetches:
    /// 1. `kokoro-multi-lang-v1_0.tar.bz2` — main model archive
    /// 2. `espeak-ng-data.tar.bz2` — phonemisation data (skipped if bundled in archive)
    ///
    /// # Errors
    /// Returns an error if any download or extraction fails.
    fn ensure_models(&self, model_dir: &std::path::Path, force: bool) -> anyhow::Result<()> {
        use crate::models;

        let tts_dir = model_dir.join("tts");
        std::fs::create_dir_all(&tts_dir)?;

        let kokoro_dir = tts_dir.join("kokoro-multi-lang-v1_0");
        let model_file = kokoro_dir.join("model.onnx");

        if !force && model_file.exists() {
            info!("[TTS] Kokoro model already present, skipping");
        } else {
            let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2";
            info!("[TTS] Downloading Kokoro TTS model from {} …", url);
            // Archive contains top-level "kokoro-multi-lang-v1_0/"; strip one level into kokoro_dir.
            models::extract_tar_bz2_dir(url, &kokoro_dir)?;
        }

        // espeak-ng-data is usually bundled inside the Kokoro archive; only fetch
        // it separately if it is still absent after the main download.
        let espeak_dir = kokoro_dir.join("espeak-ng-data");
        if !espeak_dir.exists() {
            let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2";
            info!("[TTS] Downloading espeak-ng-data from {} …", url);
            // Archive top-level is "espeak-ng-data/"; strip into espeak_dir.
            models::extract_tar_bz2_dir(url, &espeak_dir)?;
        }

        Ok(())
    }

    fn verify_models(&self, model_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let kokoro_dir = model_dir.join("tts").join("kokoro-multi-lang-v1_0");
        let required = vec![kokoro_dir.join("model.onnx"), kokoro_dir.join("voices.bin"), kokoro_dir.join("tokens.txt")];
        required.into_iter().filter(|p| !p.exists()).collect()
    }
}
