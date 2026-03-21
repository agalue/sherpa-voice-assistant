// Package setup provides model download/extraction helpers and the --setup
// orchestration used by the voice assistant.
//
// The [Run] function accepts a slice of [ModelProvider] values so that it never
// needs to import the stt or tts packages directly (which already import this
// package for the download helpers). Callers — typically main — perform the
// wiring.
package setup

import (
	"fmt"
	"log"
)

// ModelProvider manages the lifecycle of model files for one component (VAD,
// STT, or TTS). The interface is satisfied implicitly by the identically-shaped
// interfaces in the stt and tts packages.
type ModelProvider interface {
	// Name returns a human-readable label (e.g. "Whisper", "Kokoro").
	Name() string

	// EnsureModels downloads any absent model files into modelDir.
	// If force is true, every file is re-downloaded regardless.
	EnsureModels(modelDir string, force bool) error

	// VerifyModels returns paths of missing model files (empty = all present).
	VerifyModels(modelDir string) []string
}

// Run downloads and verifies all model files for the given providers.
// It is the implementation behind the --setup CLI flag.
func Run(modelDir string, force bool, providers []ModelProvider) error {
	log.Println("🔧 Voice Assistant Setup — downloading model files")
	log.Printf("   Model directory: %s", modelDir)
	if force {
		log.Println("   Mode: force re-download")
	} else {
		log.Println("   Mode: skip existing files")
	}

	for _, p := range providers {
		log.Printf("📥 [%s] downloading models…", p.Name())
		if err := p.EnsureModels(modelDir, force); err != nil {
			return fmt.Errorf("%s model download: %w", p.Name(), err)
		}
	}

	// Final verification
	log.Println("🔍 Verifying model files…")
	var allMissing []string
	for _, p := range providers {
		allMissing = append(allMissing, p.VerifyModels(modelDir)...)
	}

	if len(allMissing) > 0 {
		log.Println("❌ Some model files are still missing:")
		for _, f := range allMissing {
			log.Printf("   - %s", f)
		}
		return fmt.Errorf("%d model file(s) missing after setup", len(allMissing))
	}

	log.Println("✅ All model files are present. Run the assistant without --setup to start.")
	return nil
}
