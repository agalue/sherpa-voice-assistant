// Package stt provides speech-to-text functionality using sherpa-onnx.
// This file contains the Silero VAD implementation.
package stt

import (
	"fmt"
	"log"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/agalue/sherpa-voice-assistant/internal/models"
	"github.com/agalue/sherpa-voice-assistant/internal/sherpa"
)

// VAD configuration constants.
const (
	// VADMinSpeechDuration is the minimum speech duration (in seconds) required to
	// trigger segment delivery. 0.1 s captures short utterances like "yes" or "no".
	VADMinSpeechDuration = 0.1

	// VADMaxSpeechDuration is the maximum continuous speech duration (in seconds).
	// Prevents unbounded audio accumulation and forces segmentation of long utterances.
	VADMaxSpeechDuration = 30.0

	// VADWindowSize is the VAD processing window in samples.
	// At 16 kHz: 512 samples = 32 ms (standard Silero VAD frame size).
	VADWindowSize = 512

	// VADBufferSize is the VAD audio buffer depth in seconds.
	VADBufferSize = 60.0
)

// Compile-time interface compliance check.
var _ VoiceDetector = (*SileroVAD)(nil)

// SileroVAD implements [VoiceDetector] using the Silero VAD model via sherpa-onnx.
//
// AcceptWaveform is safe to call from the real-time audio callback thread; it never
// blocks. Completed speech segments are delivered on [SileroVAD.SegmentChannel].
type SileroVAD struct {
	vad        *sherpa.VoiceActivityDetector // VAD engine (fast, <10 ms per call)
	mu         sync.Mutex                    // Protects VAD access
	sampleRate int

	// Atomic speech-detection state — lock-free on the hot path.
	wasSpeaking atomic.Bool
	speechStart atomic.Int64 // Unix nanoseconds; 0 if not currently in speech

	// Event-driven segment delivery.
	segmentChan chan []float32
}

// SileroConfig holds configuration for [SileroVAD].
type SileroConfig struct {
	ModelDir        string  // Base model directory (silero_vad.onnx is resolved automatically)
	Threshold       float32 // VAD confidence threshold (0.0–1.0)
	SilenceDuration float32 // Silence duration in seconds before speech is considered ended
	SampleRate      int
	NumThreads      int
	Verbose         bool
}

// NewSileroVAD creates a [SileroVAD] that satisfies [VoiceDetector].
func NewSileroVAD(cfg *SileroConfig) (*SileroVAD, error) {
	modelPath := filepath.Join(cfg.ModelDir, "silero_vad.onnx")

	vadConfig := &sherpa.VadModelConfig{}
	vadConfig.SileroVad.Model = modelPath
	vadConfig.SileroVad.Threshold = cfg.Threshold
	vadConfig.SileroVad.MinSilenceDuration = cfg.SilenceDuration
	vadConfig.SileroVad.MinSpeechDuration = VADMinSpeechDuration
	vadConfig.SileroVad.MaxSpeechDuration = VADMaxSpeechDuration
	vadConfig.SileroVad.WindowSize = VADWindowSize
	vadConfig.SampleRate = cfg.SampleRate
	vadConfig.NumThreads = cfg.NumThreads
	vadConfig.Debug = 0
	if cfg.Verbose {
		vadConfig.Debug = 1
	}

	vad := sherpa.NewVoiceActivityDetector(vadConfig, VADBufferSize)
	if vad == nil {
		return nil, fmt.Errorf("failed to create Silero VAD")
	}

	return &SileroVAD{
		vad:         vad,
		sampleRate:  cfg.SampleRate,
		segmentChan: make(chan []float32, 5),
	}, nil
}

// AcceptWaveform feeds audio samples into the VAD and delivers completed speech
// segments immediately via [SileroVAD.SegmentChannel].
//
// Must never block; called on the real-time audio callback thread.
func (v *SileroVAD) AcceptWaveform(samples []float32) {
	v.mu.Lock()
	v.vad.AcceptWaveform(samples)
	isSpeech := v.vad.IsSpeech()

	// EVENT-DRIVEN: send completed segments without holding the VAD lock.
	if !v.vad.IsEmpty() {
		segment := v.vad.Front()
		v.vad.Pop()

		if len(segment.Samples) > 0 {
			samplesCopy := make([]float32, len(segment.Samples))
			copy(samplesCopy, segment.Samples)
			v.mu.Unlock()

			// Non-blocking send — never block the audio callback thread.
			select {
			case v.segmentChan <- samplesCopy:
			default:
				log.Println("⚠️ Segment channel full, dropping segment")
			}

			v.mu.Lock()
		}
	}
	v.mu.Unlock()

	// Speech-state tracking via atomics (lock-free on this side of the hot path).
	wasSpk := v.wasSpeaking.Load()
	if isSpeech && !wasSpk {
		log.Println("🎤 Speech started")
		v.speechStart.Store(time.Now().UnixNano())
		v.wasSpeaking.Store(true)
	} else if !isSpeech && wasSpk {
		if startNano := v.speechStart.Load(); startNano > 0 {
			duration := float64(time.Now().UnixNano()-startNano) / 1e9
			log.Printf("🎤 Speech ended (%.1fs)", duration)
		}
		v.wasSpeaking.Store(false)
	}
}

// SegmentChannel returns the channel on which completed speech segments are delivered.
func (v *SileroVAD) SegmentChannel() <-chan AudioSegment {
	return v.segmentChan
}

// IsSpeechDetected returns true when the VAD currently considers input to be active speech.
func (v *SileroVAD) IsSpeechDetected() bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	return !v.vad.IsEmpty() || v.vad.IsSpeech()
}

// Clear resets the internal VAD state.
func (v *SileroVAD) Clear() {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.vad.Clear()
}

// Close releases all resources held by the detector.
func (v *SileroVAD) Close() {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.segmentChan != nil {
		close(v.segmentChan)
		v.segmentChan = nil
	}
	if v.vad != nil {
		sherpa.DeleteVoiceActivityDetector(v.vad)
		v.vad = nil
	}
}

// ---------------------------------------------------------------------------
// SileroModelProvider
// ---------------------------------------------------------------------------

// SileroModelProvider implements [ModelProvider] for the Silero VAD model.
type SileroModelProvider struct{}

// Name returns the human-readable name of this VAD implementation.
func (p *SileroModelProvider) Name() string {
	return "Silero VAD"
}

// EnsureModels downloads silero_vad.onnx if it is absent from modelDir.
func (p *SileroModelProvider) EnsureModels(modelDir string, force bool) error {
	dest := filepath.Join(modelDir, "silero_vad.onnx")
	return models.DownloadFile(
		"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
		dest, force,
	)
}

// VerifyModels returns a list of absent model file paths.
func (p *SileroModelProvider) VerifyModels(modelDir string) []string {
	path := filepath.Join(modelDir, "silero_vad.onnx")
	if !models.FileExists(path) {
		return []string{path}
	}
	return nil
}
