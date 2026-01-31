//! Text-to-speech synthesizer using Kokoro models.

use anyhow::Result;
use sherpa_rs::OnnxConfig;
use sherpa_rs::tts::{CommonTtsConfig, KokoroTts, KokoroTtsConfig};
use tracing::{debug, info};

use crate::config::AppConfig;

/// Text-to-speech synthesizer using Kokoro models.
pub struct Synthesizer {
    tts: KokoroTts,   // Kokoro TTS engine
    sample_rate: u32, // Output sample rate (24kHz for Kokoro)
    speaker_id: i32,  // Speaker/voice identifier
    speed: f32,       // Speech speed multiplier
}

impl Synthesizer {
    /// Create a new TTS synthesizer.
    ///
    /// # Arguments
    /// * `config` - Application configuration
    ///
    /// # Returns
    /// A new `Synthesizer` instance.
    ///
    /// # Errors
    /// Returns an error if TTS initialization fails (e.g., missing model files).
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

    /// Synthesize a single sentence.
    ///
    /// # Arguments
    /// * `sentence` - The sentence to synthesize
    ///
    /// # Returns
    /// Audio samples or an error.
    ///
    /// # Errors
    /// Returns an error if TTS generation fails.
    pub fn synthesize_sentence(&mut self, sentence: &str) -> Result<Vec<f32>> {
        if sentence.trim().is_empty() {
            return Ok(Vec::new());
        }

        debug!("Synthesizing sentence: \"{}\"", sentence);

        let audio = self.tts.create(sentence, self.speaker_id, self.speed).map_err(|e| anyhow::anyhow!("TTS generation failed: {}", e))?;

        info!("🎵 Generated speech ({} samples)", audio.samples.len());
        Ok(audio.samples)
    }

    /// Get the sample rate of the synthesized audio.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Split text into sentences for streaming synthesis.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);

        // Check for sentence boundaries
        if c == '.' || c == '!' || c == '?' || c == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Don't forget remaining text
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}
