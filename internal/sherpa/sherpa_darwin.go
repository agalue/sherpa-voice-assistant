//go:build darwin

// Package sherpa provides platform-specific sherpa-onnx bindings.
// This file contains macOS-specific imports with CoreML support.
package sherpa

import impl "github.com/k2-fsa/sherpa-onnx-go-macos"

// Re-export all sherpa-onnx types and functions for cross-platform use.
// The actual implementation comes from the platform-specific package.

// Type aliases for VAD

type VoiceActivityDetector = impl.VoiceActivityDetector
type VadModelConfig = impl.VadModelConfig
type SpeechSegment = impl.SpeechSegment

// Type aliases for offline recognizer (STT)

type OfflineRecognizer = impl.OfflineRecognizer
type OfflineRecognizerConfig = impl.OfflineRecognizerConfig
type OfflineStream = impl.OfflineStream
type OfflineRecognizerResult = impl.OfflineRecognizerResult

// Type aliases for TTS

type OfflineTts = impl.OfflineTts
type OfflineTtsConfig = impl.OfflineTtsConfig
type GeneratedAudio = impl.GeneratedAudio

// VAD functions

var NewVoiceActivityDetector = impl.NewVoiceActivityDetector
var DeleteVoiceActivityDetector = impl.DeleteVoiceActivityDetector

// Offline recognizer functions

var NewOfflineRecognizer = impl.NewOfflineRecognizer
var DeleteOfflineRecognizer = impl.DeleteOfflineRecognizer
var NewOfflineStream = impl.NewOfflineStream
var DeleteOfflineStream = impl.DeleteOfflineStream

// TTS functions

var NewOfflineTts = impl.NewOfflineTts
var DeleteOfflineTts = impl.DeleteOfflineTts

// DefaultProvider returns the recommended provider for this platform.
// On macOS, CoreML provides hardware acceleration via Apple's Neural Engine.
func DefaultProvider() string {
	return "coreml"
}

// AvailableProviders returns the list of available providers on this platform.
func AvailableProviders() []string {
	return []string{"cpu", "coreml"}
}

// HasNvidiaGPU returns false on macOS as NVIDIA GPUs are not supported.
func HasNvidiaGPU() bool {
	return false
}
