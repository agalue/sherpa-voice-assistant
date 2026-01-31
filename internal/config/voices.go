// Package config provides voice metadata for TTS.
package config

import (
	"fmt"
	"sort"
	"strings"
)

// Voice contains essential runtime data for TTS synthesis.
type Voice struct {
	SpeakerID  int
	EspeakCode string // Language code for espeak-ng
	Language   string // Human-readable language name
}

// Voices contains all 53 Kokoro v1.0 voices with essential metadata only.
// Just what's needed at runtime: speaker ID, espeak code, and language name for display.
var Voices = map[string]Voice{
	// American English (20 voices)
	"af_alloy":   {SpeakerID: 0, EspeakCode: "en-us", Language: "American English"},
	"af_aoede":   {SpeakerID: 1, EspeakCode: "en-us", Language: "American English"},
	"af_bella":   {SpeakerID: 2, EspeakCode: "en-us", Language: "American English"},
	"af_heart":   {SpeakerID: 3, EspeakCode: "en-us", Language: "American English"},
	"af_jessica": {SpeakerID: 4, EspeakCode: "en-us", Language: "American English"},
	"af_kore":    {SpeakerID: 5, EspeakCode: "en-us", Language: "American English"},
	"af_nicole":  {SpeakerID: 6, EspeakCode: "en-us", Language: "American English"},
	"af_nova":    {SpeakerID: 7, EspeakCode: "en-us", Language: "American English"},
	"af_river":   {SpeakerID: 8, EspeakCode: "en-us", Language: "American English"},
	"af_sarah":   {SpeakerID: 9, EspeakCode: "en-us", Language: "American English"},
	"af_sky":     {SpeakerID: 10, EspeakCode: "en-us", Language: "American English"},
	"am_adam":    {SpeakerID: 11, EspeakCode: "en-us", Language: "American English"},
	"am_echo":    {SpeakerID: 12, EspeakCode: "en-us", Language: "American English"},
	"am_eric":    {SpeakerID: 13, EspeakCode: "en-us", Language: "American English"},
	"am_fenrir":  {SpeakerID: 14, EspeakCode: "en-us", Language: "American English"},
	"am_liam":    {SpeakerID: 15, EspeakCode: "en-us", Language: "American English"},
	"am_michael": {SpeakerID: 16, EspeakCode: "en-us", Language: "American English"},
	"am_onyx":    {SpeakerID: 17, EspeakCode: "en-us", Language: "American English"},
	"am_puck":    {SpeakerID: 18, EspeakCode: "en-us", Language: "American English"},
	"am_santa":   {SpeakerID: 19, EspeakCode: "en-us", Language: "American English"},

	// British English (8 voices)
	"bf_alice":    {SpeakerID: 20, EspeakCode: "en-gb", Language: "British English"},
	"bf_emma":     {SpeakerID: 21, EspeakCode: "en-gb", Language: "British English"},
	"bf_isabella": {SpeakerID: 22, EspeakCode: "en-gb", Language: "British English"},
	"bf_lily":     {SpeakerID: 23, EspeakCode: "en-gb", Language: "British English"},
	"bm_daniel":   {SpeakerID: 24, EspeakCode: "en-gb", Language: "British English"},
	"bm_fable":    {SpeakerID: 25, EspeakCode: "en-gb", Language: "British English"},
	"bm_george":   {SpeakerID: 26, EspeakCode: "en-gb", Language: "British English"},
	"bm_lewis":    {SpeakerID: 27, EspeakCode: "en-gb", Language: "British English"},

	// Spanish (2 voices)
	"ef_dora": {SpeakerID: 28, EspeakCode: "es", Language: "Spanish"},
	"em_alex": {SpeakerID: 29, EspeakCode: "es", Language: "Spanish"},

	// French (1 voice)
	"ff_siwis": {SpeakerID: 30, EspeakCode: "fr-fr", Language: "French"},

	// Hindi (4 voices)
	"hf_alpha": {SpeakerID: 31, EspeakCode: "hi", Language: "Hindi"},
	"hf_beta":  {SpeakerID: 32, EspeakCode: "hi", Language: "Hindi"},
	"hm_omega": {SpeakerID: 33, EspeakCode: "hi", Language: "Hindi"},
	"hm_psi":   {SpeakerID: 34, EspeakCode: "hi", Language: "Hindi"},

	// Italian (2 voices)
	"if_sara":   {SpeakerID: 35, EspeakCode: "it", Language: "Italian"},
	"im_nicola": {SpeakerID: 36, EspeakCode: "it", Language: "Italian"},

	// Japanese (5 voices)
	"jf_alpha":      {SpeakerID: 37, EspeakCode: "ja", Language: "Japanese"},
	"jf_gongitsune": {SpeakerID: 38, EspeakCode: "ja", Language: "Japanese"},
	"jf_nezumi":     {SpeakerID: 39, EspeakCode: "ja", Language: "Japanese"},
	"jf_tebukuro":   {SpeakerID: 40, EspeakCode: "ja", Language: "Japanese"},
	"jm_kumo":       {SpeakerID: 41, EspeakCode: "ja", Language: "Japanese"},

	// Portuguese BR (3 voices)
	"pf_dora":  {SpeakerID: 42, EspeakCode: "pt-br", Language: "Portuguese BR"},
	"pm_alex":  {SpeakerID: 43, EspeakCode: "pt-br", Language: "Portuguese BR"},
	"pm_santa": {SpeakerID: 44, EspeakCode: "pt-br", Language: "Portuguese BR"},

	// Mandarin Chinese (8 voices)
	"zf_xiaobei":  {SpeakerID: 45, EspeakCode: "cmn", Language: "Mandarin Chinese"},
	"zf_xiaoni":   {SpeakerID: 46, EspeakCode: "cmn", Language: "Mandarin Chinese"},
	"zf_xiaoxiao": {SpeakerID: 47, EspeakCode: "cmn", Language: "Mandarin Chinese"},
	"zf_xiaoyi":   {SpeakerID: 48, EspeakCode: "cmn", Language: "Mandarin Chinese"},
	"zm_yunjian":  {SpeakerID: 49, EspeakCode: "cmn", Language: "Mandarin Chinese"},
	"zm_yunxi":    {SpeakerID: 50, EspeakCode: "cmn", Language: "Mandarin Chinese"},
	"zm_yunxia":   {SpeakerID: 51, EspeakCode: "cmn", Language: "Mandarin Chinese"},
	"zm_yunyang":  {SpeakerID: 52, EspeakCode: "cmn", Language: "Mandarin Chinese"},
}

// GetVoice returns voice data for a given voice name.
// Returns nil if the voice doesn't exist.
func GetVoice(name string) *Voice {
	if voice, ok := Voices[name]; ok {
		return &voice
	}
	return nil
}

// VoiceExists checks if a voice name is valid.
func VoiceExists(name string) bool {
	_, exists := Voices[name]
	return exists
}

// PrintVoices displays all available voices in a simple, organized format.
func PrintVoices() {
	fmt.Println("═══════════════════════════════════════════════════════════════════")
	fmt.Println("  Kokoro TTS v1.0 - 53 Voices Across 9 Languages")
	fmt.Println("═══════════════════════════════════════════════════════════════════")
	fmt.Println()

	// Group voices by language
	languages := []string{
		"American English", "British English", "Spanish", "French",
		"Hindi", "Italian", "Japanese", "Portuguese BR", "Mandarin Chinese",
	}

	for _, lang := range languages {
		// Collect voices for this language
		var voiceNames []string
		for name, voice := range Voices {
			if voice.Language == lang {
				voiceNames = append(voiceNames, name)
			}
		}
		sort.Strings(voiceNames)

		// Print language header with voice count
		fmt.Printf("\n── %s (%d voices) ──\n", lang, len(voiceNames))
		fmt.Printf("%-15s %-4s %s\n", "VOICE", "ID", "ESPEAK")
		fmt.Println(strings.Repeat("─", 50))

		// Print voices
		for _, name := range voiceNames {
			voice := Voices[name]
			fmt.Printf("%-15s %-4d %s\n", name, voice.SpeakerID, voice.EspeakCode)
		}
	}

	fmt.Println()
	fmt.Println(strings.Repeat("─", 70))
	fmt.Println()
	fmt.Println("Default: af_bella (ID 2) - American English")
	fmt.Println("Recommended: af_heart (ID 3) or bf_emma (ID 21)")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  ./voice-assistant --tts-voice af_bella")
	fmt.Println("  ./voice-assistant --tts-voice bf_emma --tts-speaker-id 21")
	fmt.Println()
	fmt.Println("Try different voices to find what sounds best to you!")
	fmt.Println()
}

// PrintVoiceInfo displays information about a specific voice.
func PrintVoiceInfo(name string) error {
	voice := GetVoice(name)
	if voice == nil {
		return fmt.Errorf("voice '%s' not found. Run with --list-voices to see available voices", name)
	}

	fmt.Println()
	fmt.Printf("Voice: %s\n", name)
	fmt.Println(strings.Repeat("─", 40))
	fmt.Printf("Speaker ID:  %d\n", voice.SpeakerID)
	fmt.Printf("Language:    %s\n", voice.Language)
	fmt.Printf("Espeak code: %s\n", voice.EspeakCode)
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Printf("  ./voice-assistant --tts-voice %s --tts-speaker-id %d\n", name, voice.SpeakerID)
	fmt.Println()

	return nil
}
