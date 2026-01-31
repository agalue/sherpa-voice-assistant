//! Simplified voice metadata for Kokoro TTS.
//!
//! This module contains essential metadata for all 53 voices. Only runtime-required fields
//! are included (speaker_id, espeak_code, language). Gender, quality ratings, and descriptions
//! were removed to reduce complexity - users test voices themselves.

/// Essential metadata for a TTS voice.
#[derive(Debug, Clone, Copy)]
pub struct Voice {
    pub speaker_id: i32,
    pub espeak_code: &'static str,
    pub language: &'static str,
}

/// All voices as a compile-time constant slice (sorted by name for binary search).
/// Zero runtime cost, no heap allocations, production-grade Rust.
const VOICES: &[(&str, Voice)] = &[
    ("af_alloy", Voice { speaker_id: 0, espeak_code: "en-us", language: "American English" }),
    ("af_aoede", Voice { speaker_id: 1, espeak_code: "en-us", language: "American English" }),
    ("af_bella", Voice { speaker_id: 2, espeak_code: "en-us", language: "American English" }),
    ("af_heart", Voice { speaker_id: 3, espeak_code: "en-us", language: "American English" }),
    ("af_jessica", Voice { speaker_id: 4, espeak_code: "en-us", language: "American English" }),
    ("af_kore", Voice { speaker_id: 5, espeak_code: "en-us", language: "American English" }),
    ("af_nicole", Voice { speaker_id: 6, espeak_code: "en-us", language: "American English" }),
    ("af_nova", Voice { speaker_id: 7, espeak_code: "en-us", language: "American English" }),
    ("af_river", Voice { speaker_id: 8, espeak_code: "en-us", language: "American English" }),
    ("af_sarah", Voice { speaker_id: 9, espeak_code: "en-us", language: "American English" }),
    ("af_sky", Voice { speaker_id: 10, espeak_code: "en-us", language: "American English" }),
    ("am_adam", Voice { speaker_id: 11, espeak_code: "en-us", language: "American English" }),
    ("am_echo", Voice { speaker_id: 12, espeak_code: "en-us", language: "American English" }),
    ("am_eric", Voice { speaker_id: 13, espeak_code: "en-us", language: "American English" }),
    ("am_fenrir", Voice { speaker_id: 14, espeak_code: "en-us", language: "American English" }),
    ("am_liam", Voice { speaker_id: 15, espeak_code: "en-us", language: "American English" }),
    ("am_michael", Voice { speaker_id: 16, espeak_code: "en-us", language: "American English" }),
    ("am_onyx", Voice { speaker_id: 17, espeak_code: "en-us", language: "American English" }),
    ("am_puck", Voice { speaker_id: 18, espeak_code: "en-us", language: "American English" }),
    ("am_santa", Voice { speaker_id: 19, espeak_code: "en-us", language: "American English" }),
    ("bf_alice", Voice { speaker_id: 20, espeak_code: "en-gb", language: "British English" }),
    ("bf_emma", Voice { speaker_id: 21, espeak_code: "en-gb", language: "British English" }),
    ("bf_isabella", Voice { speaker_id: 22, espeak_code: "en-gb", language: "British English" }),
    ("bf_lily", Voice { speaker_id: 23, espeak_code: "en-gb", language: "British English" }),
    ("bm_daniel", Voice { speaker_id: 24, espeak_code: "en-gb", language: "British English" }),
    ("bm_fable", Voice { speaker_id: 25, espeak_code: "en-gb", language: "British English" }),
    ("bm_george", Voice { speaker_id: 26, espeak_code: "en-gb", language: "British English" }),
    ("bm_lewis", Voice { speaker_id: 27, espeak_code: "en-gb", language: "British English" }),
    ("ef_dora", Voice { speaker_id: 28, espeak_code: "es", language: "Spanish" }),
    ("em_alex", Voice { speaker_id: 29, espeak_code: "es", language: "Spanish" }),
    ("ff_siwis", Voice { speaker_id: 30, espeak_code: "fr-fr", language: "French" }),
    ("hf_alpha", Voice { speaker_id: 31, espeak_code: "hi", language: "Hindi" }),
    ("hf_beta", Voice { speaker_id: 32, espeak_code: "hi", language: "Hindi" }),
    ("hm_omega", Voice { speaker_id: 33, espeak_code: "hi", language: "Hindi" }),
    ("hm_psi", Voice { speaker_id: 34, espeak_code: "hi", language: "Hindi" }),
    ("if_sara", Voice { speaker_id: 35, espeak_code: "it", language: "Italian" }),
    ("im_nicola", Voice { speaker_id: 36, espeak_code: "it", language: "Italian" }),
    ("jf_alpha", Voice { speaker_id: 37, espeak_code: "ja", language: "Japanese" }),
    ("jf_gongitsune", Voice { speaker_id: 38, espeak_code: "ja", language: "Japanese" }),
    ("jf_nezumi", Voice { speaker_id: 39, espeak_code: "ja", language: "Japanese" }),
    ("jf_tebukuro", Voice { speaker_id: 40, espeak_code: "ja", language: "Japanese" }),
    ("jm_kumo", Voice { speaker_id: 41, espeak_code: "ja", language: "Japanese" }),
    ("pf_dora", Voice { speaker_id: 42, espeak_code: "pt-br", language: "Portuguese BR" }),
    ("pm_alex", Voice { speaker_id: 43, espeak_code: "pt-br", language: "Portuguese BR" }),
    ("pm_santa", Voice { speaker_id: 44, espeak_code: "pt-br", language: "Portuguese BR" }),
    ("zf_xiaobei", Voice { speaker_id: 45, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zf_xiaoni", Voice { speaker_id: 46, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zf_xiaoxiao", Voice { speaker_id: 47, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zf_xiaoyi", Voice { speaker_id: 48, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zm_yunjian", Voice { speaker_id: 49, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zm_yunxi", Voice { speaker_id: 50, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zm_yunxia", Voice { speaker_id: 51, espeak_code: "cmn", language: "Mandarin Chinese" }),
    ("zm_yunyang", Voice { speaker_id: 52, espeak_code: "cmn", language: "Mandarin Chinese" }),
];

/// Get voice metadata by name using binary search O(log n).
pub fn get_voice(name: &str) -> Option<&'static Voice> {
    VOICES.binary_search_by_key(&name, |(n, _)| n).ok().map(|idx| &VOICES[idx].1)
}

/// Print all available voices (simplified output without gender/quality/description).
pub fn print_voices() {
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
        // Count voices for this language
        let count = VOICES.iter().filter(|(_, v)| v.language == *lang).count();

        println!("\n── {} ({} voices) ──", lang, count);
        println!("{:<15} {:<4} ESPEAK", "VOICE", "ID");
        println!("{}", "─".repeat(50));

        // Collect and sort voices by speaker_id (already sorted by name in VOICES)
        let mut lang_voices: Vec<_> = VOICES.iter().filter(|(_, v)| v.language == *lang).collect();
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

/// Print detailed information about a specific voice.
pub fn print_voice_info(name: &str) -> anyhow::Result<()> {
    let voice = get_voice(name).ok_or_else(|| anyhow::anyhow!("Voice '{}' not found. Run with --list-voices to see available voices", name))?;

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
