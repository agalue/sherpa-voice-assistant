// Package tts provides text-to-speech functionality using sherpa-onnx.
// This file contains shared text processing utilities for TTS implementations.
package tts

import "strings"

// SplitSentences splits text into sentences for streaming synthesis.
//
// It splits on sentence boundaries (. ! ? \n) while avoiding:
//   - Decimal numbers (e.g., "10.5°C")
//   - Single-letter abbreviations (e.g., the letters in "U.S.")
//   - Periods not followed by a space + uppercase start
func SplitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		c := runes[i]
		current.WriteRune(c)

		if c == '.' || c == '!' || c == '?' || c == '\n' {
			if c == '.' {
				prevIsDigit := i > 0 && isDigit(runes[i-1])
				nextIsDigit := i+1 < len(runes) && isDigit(runes[i+1])

				// Decimal point (e.g. "10.5"), don't split.
				if prevIsDigit && nextIsDigit {
					continue
				}

				// Single capital letter before period — likely abbreviation ("U.S.", "Dr.").
				if i > 0 && isLetter(runes[i-1]) {
					prevChar := runes[i-1]
					var beforePrev rune
					if i > 1 {
						beforePrev = runes[i-2]
					}
					if isUpper(prevChar) && (i == 1 || beforePrev == ' ' || beforePrev == ',') {
						continue
					}
				}

				// Look ahead: proper sentence boundary has space + uppercase after period.
				if i+1 < len(runes) {
					next := runes[i+1]
					if next == ' ' && i+2 < len(runes) {
						afterSpace := runes[i+2]
						if !isUpper(afterSpace) && !isDigit(afterSpace) {
							continue
						}
					} else if next != ' ' && next != '\n' {
						continue
					}
				}
			}

			trimmed := strings.TrimSpace(current.String())
			if trimmed != "" {
				sentences = append(sentences, trimmed)
			}
			current.Reset()
		}
	}

	// Append any trailing text that did not end with punctuation.
	if trimmed := strings.TrimSpace(current.String()); trimmed != "" {
		sentences = append(sentences, trimmed)
	}

	return sentences
}

// isDigit reports whether r is an ASCII decimal digit.
func isDigit(r rune) bool { return r >= '0' && r <= '9' }

// isLetter reports whether r is an ASCII letter.
func isLetter(r rune) bool { return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') }

// isUpper reports whether r is an ASCII uppercase letter.
func isUpper(r rune) bool { return r >= 'A' && r <= 'Z' }
