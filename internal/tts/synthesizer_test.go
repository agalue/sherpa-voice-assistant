// Package tts provides text-to-speech functionality using sherpa-onnx.
package tts

import (
	"reflect"
	"strings"
	"testing"
)

func TestSplitSentencesWithDecimals(t *testing.T) {
	text := "Temperature is 10.5°C, feels like 7.2°C. Humidity is 38%."
	sentences := SplitSentences(text)

	if len(sentences) != 2 {
		t.Errorf("Expected 2 sentences, got %d", len(sentences))
	}

	if sentences[0] != "Temperature is 10.5°C, feels like 7.2°C." {
		t.Errorf("First sentence incorrect: %q", sentences[0])
	}

	if sentences[1] != "Humidity is 38%." {
		t.Errorf("Second sentence incorrect: %q", sentences[1])
	}
}

func TestSplitSentencesBasic(t *testing.T) {
	text := "Hello world. How are you? I am fine!"
	sentences := SplitSentences(text)

	expected := []string{
		"Hello world.",
		"How are you?",
		"I am fine!",
	}

	if !reflect.DeepEqual(sentences, expected) {
		t.Errorf("Expected %v, got %v", expected, sentences)
	}
}

func TestSplitSentencesAbbreviations(t *testing.T) {
	// Note: Multi-letter abbreviations like "Dr." are challenging to detect reliably.
	// For voice assistants, we prioritize decimal numbers (10.5°C) over all abbreviations.
	text := "I live in the U.S. and it's great. The temperature is 72.5°F!"
	sentences := SplitSentences(text)

	// Should preserve "U.S." in first sentence and "72.5" in second
	if !strings.Contains(sentences[0], "U.S.") {
		t.Errorf("Expected first sentence to contain 'U.S.', got: %q", sentences[0])
	}

	hasDecimal := false
	for _, s := range sentences {
		if strings.Contains(s, "72.5°F") {
			hasDecimal = true
			break
		}
	}
	if !hasDecimal {
		t.Errorf("Expected a sentence to contain '72.5°F', got: %v", sentences)
	}
}

func TestSplitSentencesNoSpaceAfterPeriod(t *testing.T) {
	text := "Visit example.com for more info. Thank you!"
	sentences := SplitSentences(text)

	if len(sentences) != 2 {
		t.Errorf("Expected 2 sentences, got %d", len(sentences))
	}

	if !strings.Contains(sentences[0], "example.com") {
		t.Errorf("Expected first sentence to contain 'example.com', got: %q", sentences[0])
	}
}

func TestSplitSentencesWeatherFormat(t *testing.T) {
	text := "Weather for Chapel Hill, NC, US: Temperature is 10.5°C, feels like 7.2°C. Humidity is 38%."
	sentences := SplitSentences(text)

	if len(sentences) != 2 {
		t.Errorf("Expected 2 sentences, got %d", len(sentences))
	}

	if !strings.Contains(sentences[0], "10.5°C") {
		t.Errorf("Expected first sentence to contain '10.5°C', got: %q", sentences[0])
	}

	if !strings.Contains(sentences[0], "7.2°C") {
		t.Errorf("Expected first sentence to contain '7.2°C', got: %q", sentences[0])
	}
}
