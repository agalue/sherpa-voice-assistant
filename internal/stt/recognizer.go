// Package stt provides speech-to-text functionality using sherpa-onnx.
package stt

import (
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/agalue/voice-assistant/internal/sherpa"
)

// VAD configuration constants for speech detection.
const (
	// VADMinSpeechDuration is the minimum speech duration (in seconds) to trigger detection.
	// Value of 0.1s allows detection of short utterances like "yes" or "no".
	VADMinSpeechDuration = 0.1

	// VADMaxSpeechDuration is the maximum continuous speech duration (in seconds).
	// Prevents unbounded audio accumulation and forces segmentation of long utterances.
	VADMaxSpeechDuration = 30.0

	// VADWindowSize is the window size in samples for VAD processing.
	// At 16kHz sample rate: 512 samples = 32ms window (standard VAD frame size).
	VADWindowSize = 512

	// VADBufferSize is the maximum audio buffer size in seconds for VAD.
	// Large buffer allows accumulation of speech segments for transcription.
	VADBufferSize = 60.0
)

// Recognizer handles speech recognition with VAD.
// Separates fast VAD operations from slow Whisper transcription using independent locks.
type Recognizer struct {
	vad        *sherpa.VoiceActivityDetector // Voice activity detector (fast, <10ms)
	recognizer *sherpa.OfflineRecognizer     // Whisper transcriber (slow, 100-500ms)
	wakeWord   string                        // Optional wake word for activation
	verbose    bool                          // Enable verbose logging

	mu         sync.Mutex                 // Protects VAD access
	sampleRate int                        // Audio sample rate (16kHz)
	vadConfig  *sherpa.VadModelConfig     // VAD configuration

	// Speech state tracking for immediate feedback.
	// Atomic operations avoid mutex locks in hot path.
	wasSpeaking atomic.Bool   // Previous speaking state
	speechStart atomic.Int64  // Speech start timestamp (Unix nanoseconds)

	// Event-driven segment delivery channel.
	segmentChan chan []float32 // Channel for completed speech segments
}

// Config holds STT configuration.
type Config struct {
	VADModel           string
	VADThreshold       float32
	VADSilenceDuration float32 // Silence duration in seconds before speech is considered ended
	WhisperEncoder     string
	WhisperDecoder     string
	WhisperTokens      string
	SampleRate         int
	WakeWord           string
	Provider           string // Hardware acceleration provider (cpu, cuda, coreml)
	Language           string // Speech recognition language (e.g., "en", "es", "auto")
	Verbose            bool
	VADThreads         int // Number of threads for VAD
	STTThreads         int // Number of threads for STT (Whisper)
}

// NewRecognizer creates a new speech recognizer.
func NewRecognizer(cfg *Config) (*Recognizer, error) {
	// Configure VAD
	vadConfig := &sherpa.VadModelConfig{}
	vadConfig.SileroVad.Model = cfg.VADModel
	vadConfig.SileroVad.Threshold = cfg.VADThreshold
	vadConfig.SileroVad.MinSilenceDuration = cfg.VADSilenceDuration
	vadConfig.SileroVad.MinSpeechDuration = VADMinSpeechDuration
	vadConfig.SileroVad.MaxSpeechDuration = VADMaxSpeechDuration
	vadConfig.SileroVad.WindowSize = VADWindowSize
	vadConfig.SampleRate = cfg.SampleRate
	vadConfig.NumThreads = cfg.VADThreads
	vadConfig.Debug = 0
	if cfg.Verbose {
		vadConfig.Debug = 1
	}

	// Buffer audio for VAD
	vad := sherpa.NewVoiceActivityDetector(vadConfig, VADBufferSize)
	if vad == nil {
		return nil, fmt.Errorf("failed to create VAD")
	}

	// Configure Whisper recognizer
	recognizerConfig := &sherpa.OfflineRecognizerConfig{}
	recognizerConfig.ModelConfig.Whisper.Encoder = cfg.WhisperEncoder
	recognizerConfig.ModelConfig.Whisper.Decoder = cfg.WhisperDecoder
	// Set language: "auto" -> "" (empty triggers auto-detection in Whisper)
	language := cfg.Language
	if strings.EqualFold(language, "auto") {
		language = ""
	}
	recognizerConfig.ModelConfig.Whisper.Language = language
	recognizerConfig.ModelConfig.Whisper.Task = "transcribe"
	recognizerConfig.ModelConfig.Whisper.TailPaddings = -1 // Use default
	recognizerConfig.ModelConfig.Tokens = cfg.WhisperTokens
	recognizerConfig.ModelConfig.NumThreads = cfg.STTThreads
	recognizerConfig.ModelConfig.Provider = cfg.Provider // Hardware acceleration
	recognizerConfig.DecodingMethod = "greedy_search"
	recognizerConfig.ModelConfig.Debug = 0
	if cfg.Verbose {
		recognizerConfig.ModelConfig.Debug = 1
	}

	recognizer := sherpa.NewOfflineRecognizer(recognizerConfig)
	if recognizer == nil {
		sherpa.DeleteVoiceActivityDetector(vad)
		return nil, fmt.Errorf("failed to create offline recognizer")
	}

	return &Recognizer{
		vad:         vad,
		recognizer:  recognizer,
		wakeWord:    strings.ToLower(cfg.WakeWord),
		verbose:     cfg.Verbose,
		sampleRate:  cfg.SampleRate,
		vadConfig:   vadConfig,
		segmentChan: make(chan []float32, 5), // Buffered for bursts
	}, nil
}

// AcceptWaveform feeds audio samples to the VAD and sends completed segments immediately.
// This is event-driven: segments are delivered as soon as VAD completes them,
// eliminating the 50-100ms polling delay of the previous approach.
func (r *Recognizer) AcceptWaveform(samples []float32) {
	// Lock only for VAD access (unavoidable, sherpa-onnx not thread-safe)
	r.mu.Lock()
	r.vad.AcceptWaveform(samples)
	isSpeech := r.vad.IsSpeech()

	// EVENT-DRIVEN: Send completed segments immediately (no polling delay)
	if !r.vad.IsEmpty() {
		segment := r.vad.Front()
		r.vad.Pop()

		if len(segment.Samples) > 0 {
			// Make a copy for the channel (segment.Samples will be reused)
			samplesCopy := make([]float32, len(segment.Samples))
			copy(samplesCopy, segment.Samples)

			// Unlock before sending to avoid holding lock during channel operation
			r.mu.Unlock()

			// Non-blocking send to avoid audio thread blocking
			select {
			case r.segmentChan <- samplesCopy:
				// Sent successfully
			default:
				log.Println("⚠️ Segment channel full, dropping segment")
			}

			// Re-lock for state tracking below
			r.mu.Lock()
		}
	}
	r.mu.Unlock()

	// Speech state tracking using atomic operations (lock-free)
	wasSpk := r.wasSpeaking.Load()

	if isSpeech && !wasSpk {
		// Speech just started
		log.Println("🎤 Speech started")
		r.speechStart.Store(time.Now().UnixNano())
		r.wasSpeaking.Store(true)
	} else if !isSpeech && wasSpk {
		// Speech just ended
		startNano := r.speechStart.Load()
		if startNano > 0 {
			duration := float64(time.Now().UnixNano()-startNano) / 1e9
			log.Printf("🎤 Speech ended (%.1fs)", duration)
		}
		r.wasSpeaking.Store(false)
	}
}

// SegmentChannel returns the channel that receives completed speech segments.
// Use this for event-driven segment processing instead of polling.
func (r *Recognizer) SegmentChannel() <-chan []float32 {
	return r.segmentChan
}

// TranscribeSegment transcribes a completed speech segment.
// This should be called with segments received from SegmentChannel().
func (r *Recognizer) TranscribeSegment(samples []float32) string {
	if len(samples) == 0 {
		return ""
	}

	if r.verbose {
		duration := float32(len(samples)) / float32(r.sampleRate)
		log.Printf("[STT] Processing speech segment: %.2fs", duration)
	}

	// Create a stream and decode
	stream := sherpa.NewOfflineStream(r.recognizer)
	if stream == nil {
		log.Println("[STT] Failed to create offline stream")
		return ""
	}
	defer sherpa.DeleteOfflineStream(stream)

	stream.AcceptWaveform(r.sampleRate, samples)
	r.recognizer.Decode(stream)

	result := stream.GetResult()
	text := strings.TrimSpace(result.Text)

	if text == "" {
		return ""
	}

	// Check wake word if configured
	if r.wakeWord != "" {
		lowerText := strings.ToLower(text)
		if !strings.Contains(lowerText, r.wakeWord) {
			if r.verbose {
				log.Printf("[STT] Wake word %q not found in %q, ignoring", r.wakeWord, text)
			}
			return ""
		}
		// Remove wake word from text
		text = removeWakeWord(text, r.wakeWord)
		text = strings.TrimSpace(text)

		// If only wake word was spoken (no query), prompt for hello
		if text == "" {
			log.Printf("🗣️ Wake word %q detected", r.wakeWord)
			text = "Hello"
		} else {
			log.Printf("🗣️ You (wake word detected): %s", text)
		}
		return text
	}

	log.Printf("🗣️ You: %s", text)
	return text
}

// IsSpeechDetected returns true if VAD currently detects speech.
func (r *Recognizer) IsSpeechDetected() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return !r.vad.IsEmpty() || r.vad.IsSpeech()
}

// HasPendingSegments returns true if there are speech segments to process.
func (r *Recognizer) HasPendingSegments() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return !r.vad.IsEmpty()
}

// Clear resets the VAD state.
func (r *Recognizer) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.vad.Clear()
}

// Close releases all resources.
func (r *Recognizer) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.segmentChan != nil {
		close(r.segmentChan)
		r.segmentChan = nil
	}

	if r.vad != nil {
		sherpa.DeleteVoiceActivityDetector(r.vad)
		r.vad = nil
	}
	if r.recognizer != nil {
		sherpa.DeleteOfflineRecognizer(r.recognizer)
		r.recognizer = nil
	}
}

// removeWakeWord removes the wake word from text, case-insensitively.
func removeWakeWord(text, wakeWord string) string {
	lowerText := strings.ToLower(text)
	lowerWake := strings.ToLower(wakeWord)

	idx := strings.Index(lowerText, lowerWake)
	if idx == -1 {
		return text
	}

	// Remove wake word and clean up
	result := text[:idx] + text[idx+len(wakeWord):]

	// Remove leading punctuation and whitespace
	result = strings.TrimLeft(result, " ,.!?;:-'\"")

	return strings.TrimSpace(result)
}
