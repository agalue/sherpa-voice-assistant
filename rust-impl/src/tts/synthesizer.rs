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
///
/// Intelligently splits on sentence boundaries (. ! ? \n) while avoiding:
/// - Decimal numbers (e.g., "10.5°C")
/// - Common abbreviations (e.g., "U.S.", "Dr.")
/// - Periods not followed by proper sentence starts
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for i in 0..chars.len() {
        let c = chars[i];
        current.push(c);

        // Check for sentence boundaries
        if c == '.' || c == '!' || c == '?' || c == '\n' {
            // Don't split on periods in decimal numbers
            if c == '.' {
                let prev_is_digit = i > 0 && chars[i - 1].is_ascii_digit();
                let next_is_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();

                if prev_is_digit && next_is_digit {
                    // This is a decimal point (e.g., "10.5"), don't split
                    continue;
                }

                // Don't split on single-letter abbreviations (e.g., "U.S.", "Dr.")
                if i > 0 && chars[i - 1].is_ascii_alphabetic() {
                    // Check if previous is single capital letter
                    let prev_char = chars[i - 1];
                    let before_prev = if i > 1 { Some(chars[i - 2]) } else { None };

                    if prev_char.is_uppercase() && (before_prev.is_none() || before_prev == Some(' ') || before_prev == Some(',')) {
                        // Single capital letter + period (likely abbreviation), don't split
                        continue;
                    }
                }

                // Look ahead: proper sentences have space + uppercase after period
                if i + 1 < chars.len() {
                    let next = chars[i + 1];
                    if next == ' ' && i + 2 < chars.len() {
                        let after_space = chars[i + 2];
                        if !after_space.is_uppercase() && !after_space.is_ascii_digit() {
                            // Not followed by uppercase letter (not a proper sentence start)
                            continue;
                        }
                    } else if next != ' ' && next != '\n' {
                        // No space after period (not a sentence boundary)
                        continue;
                    }
                }
            }

            // Valid sentence boundary
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_with_decimals() {
        let text = "Temperature is 10.5°C, feels like 7.2°C. Humidity is 38%.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "Temperature is 10.5°C, feels like 7.2°C.");
        assert_eq!(sentences[1], "Humidity is 38%.");
    }

    #[test]
    fn test_split_sentences_basic() {
        let text = "Hello world. How are you? I am fine!";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine!");
    }

    #[test]
    fn test_split_sentences_abbreviations() {
        // Note: Multi-letter abbreviations like "Dr." are challenging to detect reliably.
        // For voice assistants, we prioritize decimal numbers (10.5°C) over all abbreviations.
        let text = "I live in the U.S. and it's great. The temperature is 72.5°F!";
        let sentences = split_sentences(text);
        // Should preserve "U.S." in first sentence and "72.5" in second
        assert!(sentences[0].contains("U.S."));
        assert!(sentences.iter().any(|s| s.contains("72.5°F")));
    }

    #[test]
    fn test_split_sentences_no_space_after_period() {
        let text = "Visit example.com for more info. Thank you!";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("example.com"));
    }

    #[test]
    fn test_split_sentences_weather_format() {
        let text = "Weather for Chapel Hill, NC, US: Temperature is 10.5°C, feels like 7.2°C. Humidity is 38%.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("10.5°C"));
        assert!(sentences[0].contains("7.2°C"));
    }
}
