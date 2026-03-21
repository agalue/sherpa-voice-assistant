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

// ─────────────────────────────────────────────────────────────────────────────
// Kokoro voice catalog (53 voices across 9 languages)
// ─────────────────────────────────────────────────────────────────────────────

/// Essential metadata for a single Kokoro TTS voice.
#[derive(Debug, Clone, Copy)]
struct KokoroVoice {
    speaker_id: i32,
    espeak_code: &'static str,
    language: &'static str,
}

/// All 53 Kokoro v1.0 voices as a compile-time constant slice (sorted by name for binary search).
const KOKORO_VOICES: &[(&str, KokoroVoice)] = &[
    ("af_alloy", KokoroVoice { speaker_id: 0, espeak_code: "en-us", language: "American English" }),
    ("af_aoede", KokoroVoice { speaker_id: 1, espeak_code: "en-us", language: "American English" }),
    ("af_bella", KokoroVoice { speaker_id: 2, espeak_code: "en-us", language: "American English" }),
    ("af_heart", KokoroVoice { speaker_id: 3, espeak_code: "en-us", language: "American English" }),
    ("af_jessica", KokoroVoice { speaker_id: 4, espeak_code: "en-us", language: "American English" }),
    ("af_kore", KokoroVoice { speaker_id: 5, espeak_code: "en-us", language: "American English" }),
    ("af_nicole", KokoroVoice { speaker_id: 6, espeak_code: "en-us", language: "American English" }),
    ("af_nova", KokoroVoice { speaker_id: 7, espeak_code: "en-us", language: "American English" }),
    ("af_river", KokoroVoice { speaker_id: 8, espeak_code: "en-us", language: "American English" }),
    ("af_sarah", KokoroVoice { speaker_id: 9, espeak_code: "en-us", language: "American English" }),
    ("af_sky", KokoroVoice { speaker_id: 10, espeak_code: "en-us", language: "American English" }),
    ("am_adam", KokoroVoice { speaker_id: 11, espeak_code: "en-us", language: "American English" }),
    ("am_echo", KokoroVoice { speaker_id: 12, espeak_code: "en-us", language: "American English" }),
    ("am_eric", KokoroVoice { speaker_id: 13, espeak_code: "en-us", language: "American English" }),
    ("am_fenrir", KokoroVoice { speaker_id: 14, espeak_code: "en-us", language: "American English" }),
    ("am_liam", KokoroVoice { speaker_id: 15, espeak_code: "en-us", language: "American English" }),
    ("am_michael", KokoroVoice { speaker_id: 16, espeak_code: "en-us", language: "American English" }),
    ("am_onyx", KokoroVoice { speaker_id: 17, espeak_code: "en-us", language: "American English" }),
    ("am_puck", KokoroVoice { speaker_id: 18, espeak_code: "en-us", language: "American English" }),
    ("am_santa", KokoroVoice { speaker_id: 19, espeak_code: "en-us", language: "American English" }),
    ("bf_alice", KokoroVoice { speaker_id: 20, espeak_code: "en-gb", language: "British English" }),
    ("bf_emma", KokoroVoice { speaker_id: 21, espeak_code: "en-gb", language: "British English" }),
    ("bf_isabella", KokoroVoice { speaker_id: 22, espeak_code: "en-gb", language: "British English" }),
    ("bf_lily", KokoroVoice { speaker_id: 23, espeak_code: "en-gb", language: "British English" }),
    ("bm_daniel", KokoroVoice { speaker_id: 24, espeak_code: "en-gb", language: "British English" }),
    ("bm_fable", KokoroVoice { speaker_id: 25, espeak_code: "en-gb", language: "British English" }),
    ("bm_george", KokoroVoice { speaker_id: 26, espeak_code: "en-gb", language: "British English" }),
    ("bm_lewis", KokoroVoice { speaker_id: 27, espeak_code: "en-gb", language: "British English" }),
    ("ef_dora", KokoroVoice { speaker_id: 28, espeak_code: "es", language: "Spanish" }),
    ("em_alex", KokoroVoice { speaker_id: 29, espeak_code: "es", language: "Spanish" }),
    ("ff_siwis", KokoroVoice { speaker_id: 30, espeak_code: "fr-fr", language: "French" }),
    ("hf_alpha", KokoroVoice { speaker_id: 31, espeak_code: "hi", language: "Hindi" }),
    ("hf_beta", KokoroVoice { speaker_id: 32, espeak_code: "hi", language: "Hindi" }),
    ("hm_omega", KokoroVoice { speaker_id: 33, espeak_code: "hi", language: "Hindi" }),
    ("hm_psi", KokoroVoice { speaker_id: 34, espeak_code: "hi", language: "Hindi" }),
    ("if_sara", KokoroVoice { speaker_id: 35, espeak_code: "it", language: "Italian" }),
    ("im_nicola", KokoroVoice { speaker_id: 36, espeak_code: "it", language: "Italian" }),
    ("jf_alpha", KokoroVoice { speaker_id: 37, espeak_code: "ja", language: "Japanese" }),
    ("jf_gongitsune", KokoroVoice { speaker_id: 38, espeak_code: "ja", language: "Japanese" }),
    ("jf_nezumi", KokoroVoice { speaker_id: 39, espeak_code: "ja", language: "Japanese" }),
    ("jf_tebukuro", KokoroVoice { speaker_id: 40, espeak_code: "ja", language: "Japanese" }),
    ("jm_kumo", KokoroVoice { speaker_id: 41, espeak_code: "ja", language: "Japanese" }),
    ("pf_dora", KokoroVoice { speaker_id: 42, espeak_code: "pt-br", language: "Portuguese BR" }),
    ("pm_alex", KokoroVoice { speaker_id: 43, espeak_code: "pt-br", language: "Portuguese BR" }),
    ("pm_santa", KokoroVoice { speaker_id: 44, espeak_code: "pt-br", language: "Portuguese BR" }),
    ("zf_xiaobei", KokoroVoice { speaker_id: 45, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zf_xiaoni", KokoroVoice { speaker_id: 46, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zf_xiaoxiao", KokoroVoice { speaker_id: 47, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zf_xiaoyi", KokoroVoice { speaker_id: 48, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zm_yunjian", KokoroVoice { speaker_id: 49, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zm_yunxi", KokoroVoice { speaker_id: 50, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zm_yunxia", KokoroVoice { speaker_id: 51, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zm_yunyang", KokoroVoice { speaker_id: 52, espeak_code: "cmn", language: "Mandarin Chinese" }),
];

/// Get voice metadata by name using binary search O(log n).
fn get_kokoro_voice(name: &str) -> Option<&'static KokoroVoice> {
    KOKORO_VOICES.binary_search_by_key(&name, |(n, _)| n).ok().map(|idx| &KOKORO_VOICES[idx].1)
}

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

        // Look up voice in catalog to get espeak code for language/lexicon derivation.
        let voice =
            get_kokoro_voice(&config.tts_voice).ok_or_else(|| anyhow::anyhow!("Unknown TTS voice '{}'; run with --list-voices to see available voices", config.tts_voice))?;

        // Derive all Kokoro paths from model_dir.
        let kokoro_dir = config.model_dir.join("tts").join("kokoro-multi-lang-v1_0");
        let model_path = kokoro_dir.join("model.onnx").to_string_lossy().to_string();
        let voices_path = kokoro_dir.join("voices.bin").to_string_lossy().to_string();
        let tokens_path = kokoro_dir.join("tokens.txt").to_string_lossy().to_string();
        let data_dir = kokoro_dir.join("espeak-ng-data").to_string_lossy().to_string();
        let dict_dir = kokoro_dir.join("dict").to_string_lossy().to_string();
        let lexicon = lexicon_for_voice(&kokoro_dir, &config.tts_voice);
        let lang = voice.espeak_code.to_string();

        info!("Initializing Kokoro TTS synthesizer with {} provider", provider);
        info!("TTS voice: {} (speaker ID: {})", config.tts_voice, config.tts_speaker_id);

        let tts_config = KokoroTtsConfig {
            model: model_path,
            voices: voices_path,
            tokens: tokens_path,
            data_dir,
            dict_dir,
            lexicon,
            lang,
            length_scale: 1.0 / config.tts_speed, // length_scale is inverse of speed
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
// Kokoro-specific helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Derive the lexicon file path(s) for a Kokoro voice.
///
/// The model distribution includes `lexicon-us-en.txt` (American English),
/// `lexicon-gb-en.txt` (British English), and `lexicon-zh.txt` (Chinese).
/// For other languages, an empty string is returned (the model uses the `lang`
/// parameter instead).
fn lexicon_for_voice(kokoro_dir: &std::path::Path, voice_name: &str) -> String {
    if voice_name.len() < 2 {
        return String::new();
    }
    match &voice_name[..2] {
        "af" | "am" => kokoro_dir.join("lexicon-us-en.txt").to_string_lossy().to_string(),
        "bf" | "bm" => kokoro_dir.join("lexicon-gb-en.txt").to_string_lossy().to_string(),
        "zf" | "zm" => {
            // Chinese with English fallback
            format!("{},{}", kokoro_dir.join("lexicon-us-en.txt").to_string_lossy(), kokoro_dir.join("lexicon-zh.txt").to_string_lossy())
        }
        _ => String::new(), // Other languages use lang parameter
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

    fn print_voices(&self) {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("  Kokoro TTS v1.0 - 53 Voices Across 9 Languages");
        println!("═══════════════════════════════════════════════════════════════════");
        println!();

        let languages = [
            "American English",
            "British English",
            "Spanish",
            "French",
            "Hindi",
            "Italian",
            "Japanese",
            "Portuguese BR",
            "Mandarin Chinese",
        ];

        for lang in &languages {
            let count = KOKORO_VOICES.iter().filter(|(_, v)| v.language == *lang).count();

            println!("\n── {} ({} voices) ──", lang, count);
            println!("{:<15} {:<4} ESPEAK", "VOICE", "ID");
            println!("{}", "─".repeat(50));

            let mut lang_voices: Vec<_> = KOKORO_VOICES.iter().filter(|(_, v)| v.language == *lang).collect();
            lang_voices.sort_by_key(|(_, v)| v.speaker_id);

            for (name, voice) in lang_voices {
                println!("{:<15} {:<4} {}", name, voice.speaker_id, voice.espeak_code);
            }
        }

        println!("\n{}\n", "─".repeat(70));
        println!("Default: af_bella (ID 2) - American English");
        println!("Recommended: af_heart (ID 3) or bf_emma (ID 21)");
        println!();
        println!("Usage:");
        println!("  ./voice-assistant --tts-voice af_bella");
        println!("  ./voice-assistant --tts-voice bf_emma --tts-speaker-id 21");
        println!();
        println!("Try different voices to find what sounds best to you!");
    }

    fn print_voice_info(&self, name: &str) -> anyhow::Result<()> {
        let voice = get_kokoro_voice(name).ok_or_else(|| anyhow::anyhow!("Voice '{}' not found. Run with --list-voices to see available voices", name))?;

        println!();
        println!("Voice: {}", name);
        println!("{}", "─".repeat(40));
        println!("Speaker ID:    {}", voice.speaker_id);
        println!("Language:      {}", voice.language);
        println!("espeak code:   {}", voice.espeak_code);
        println!();
        println!("Usage:");
        println!("  ./voice-assistant --tts-voice {} --tts-speaker-id {}", name, voice.speaker_id);
        println!();

        Ok(())
    }
}
