//go:build linux

// Package sherpa provides platform-specific sherpa-onnx bindings.
// This file contains Linux-specific imports.
//
// By default, this uses the pre-built CPU-only sherpa-onnx-go-linux package.
// For CUDA/GPU support on Linux, use the build script with --cuda flag which
// will build sherpa-onnx from source with GPU support enabled.
package sherpa

import (
	"os"

	impl "github.com/k2-fsa/sherpa-onnx-go-linux"
)

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
// On Linux, returns "cuda" if NVIDIA GPU is likely available, otherwise "cpu".
func DefaultProvider() string {
	if HasNvidiaGPU() {
		return "cuda"
	}
	return "cpu"
}

// AvailableProviders returns the list of available providers on this platform.
func AvailableProviders() []string {
	return []string{"cpu", "cuda"}
}

// HasNvidiaGPU checks for NVIDIA GPU availability on Linux.
// Supports both discrete GPUs and Jetson SOC devices (Nano, Orin, etc.).
func HasNvidiaGPU() bool {
	// Check for nvidia-smi (discrete GPUs and some Jetson configurations)
	nvidiaSmiPaths := []string{
		"/usr/bin/nvidia-smi",
		"/usr/local/bin/nvidia-smi",
		"/opt/nvidia/bin/nvidia-smi",
	}
	for _, path := range nvidiaSmiPaths {
		if fileExists(path) {
			return true
		}
	}

	// Check for /dev/nvidia* devices (discrete GPUs)
	if fileExists("/dev/nvidia0") {
		return true
	}

	// Check for NVIDIA Jetson SOC devices (Nano, Orin, AGX, etc.)
	// Jetson devices expose GPU through /dev/nvhost-* and /dev/nvmap
	jetsonIndicators := []string{
		"/dev/nvhost-gpu",             // Jetson GPU device
		"/dev/nvhost-ctrl-gpu",        // Jetson GPU control
		"/dev/nvmap",                  // Jetson memory mapping
		"/etc/nv_tegra_release",       // Jetson L4T release file
		"/sys/devices/gpu.0",          // Jetson GPU sysfs (older)
		"/sys/devices/17000000.ga10b", // Jetson Orin GPU
		"/sys/devices/17000000.gv11b", // Jetson Xavier/Nano GPU
	}
	for _, path := range jetsonIndicators {
		if fileExists(path) {
			return true
		}
	}

	// Check for tegra in /proc/device-tree/compatible (all Jetson devices)
	if data, err := os.ReadFile("/proc/device-tree/compatible"); err == nil {
		compatible := string(data)
		if contains(compatible, "nvidia,tegra") || contains(compatible, "nvidia,jetson") {
			return true
		}
	}

	return false
}

// fileExists checks if a file or directory exists.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// contains checks if substr is in s (simple substring check).
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

// findSubstring returns true if substr is found in s.
func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
