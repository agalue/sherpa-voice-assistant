//! Text processing utilities for TTS pipelines.
//!
//! Provides [`split_sentences`] for breaking LLM responses into synthesisable
//! units. Extracted into its own module so any future TTS backend can reuse the
//! same sentence-splitting logic without depending on a specific synthesiser.

/// Split text into sentences for streaming synthesis.
///
/// Intelligently splits on sentence boundaries (. ! ? \n) while avoiding:
/// - Decimal numbers (e.g., "10.5°C")
/// - Single-letter abbreviations (e.g., the letters in "U.S.")
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
