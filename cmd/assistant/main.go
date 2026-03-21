// Voice Assistant - A Go implementation using sherpa-onnx
//
// This program implements a real-time voice assistant with:
// - Voice Activity Detection (Silero-VAD)
// - Speech-to-Text (Whisper, via the stt.Transcriber interface)
// - LLM Integration (Ollama)
// - Text-to-Speech (Kokoro, via the tts.Synthesizer interface)
//
// Run with --setup to download required model files.
// Run with --setup --force to re-download all models.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/agalue/sherpa-voice-assistant/internal/audio"
	"github.com/agalue/sherpa-voice-assistant/internal/config"
	"github.com/agalue/sherpa-voice-assistant/internal/llm"
	"github.com/agalue/sherpa-voice-assistant/internal/stt"
	"github.com/agalue/sherpa-voice-assistant/internal/tts"
)

func main() {
	// Parse configuration
	cfg, err := config.ParseFlags()
	if err != nil {
		log.Fatalf("Configuration error: %v", err)
	}

	// --setup: download all required model files then exit.
	if cfg.Setup {
		if err := runSetup(cfg); err != nil {
			log.Fatalf("Setup failed: %v", err)
		}
		os.Exit(0)
	}

	log.Println("🎤 Voice Assistant starting...")
	log.Printf("⚡ STT acceleration: %s, TTS acceleration: %s", cfg.STTProvider, cfg.TTSProvider)
	log.Printf("🔊 TTS voice: %s (speaker %d)", cfg.TTSVoice, cfg.TTSSpeakerID)

	// Create context for graceful shutdown
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Create LLM client and verify connection
	llmClient, err := llm.NewClient(&llm.Config{
		Host:         cfg.OllamaURL,
		Model:        cfg.OllamaModel,
		SystemPrompt: cfg.SystemPrompt,
		Verbose:      cfg.Verbose,
		MaxHistory:   cfg.MaxHistory,
		Temperature:  cfg.Temperature,
		SearxngURL:   cfg.SearxngURL,
	})
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	log.Printf("🔗 Checking Ollama connection at %s...", cfg.OllamaURL)
	if err := llmClient.HealthCheck(ctx); err != nil {
		log.Fatalf("Ollama connection failed: %v", err)
	}
	log.Printf("✅ Ollama connected (model: %s)", cfg.OllamaModel)

	// Create Silero VAD (voice activity detection)
	log.Println("🧠 Loading speech recognition models...")
	vad, err := stt.NewSileroVAD(&stt.SileroConfig{
		Model:           cfg.VADModel,
		Threshold:       cfg.VadThreshold,
		SilenceDuration: cfg.VADSilenceDuration,
		SampleRate:      cfg.SampleRate,
		NumThreads:      cfg.VADThreads,
		Verbose:         cfg.Verbose,
	})
	if err != nil {
		log.Fatalf("Failed to create VAD: %v", err)
	}
	defer vad.Close()

	// Create Whisper transcriber (speech-to-text)
	recognizer, err := stt.NewWhisperRecognizer(&stt.WhisperConfig{
		Encoder:    cfg.WhisperEncoder,
		Decoder:    cfg.WhisperDecoder,
		Tokens:     cfg.WhisperTokens,
		SampleRate: cfg.SampleRate,
		WakeWord:   cfg.WakeWord,
		Provider:   cfg.STTProvider,
		Language:   cfg.STTLanguage,
		Verbose:    cfg.Verbose,
		NumThreads: cfg.STTThreads,
	})
	if err != nil {
		log.Fatalf("Failed to create Whisper recognizer: %v", err)
	}
	defer recognizer.Close()

	// Expose via interfaces so the rest of the pipeline is implementation-agnostic.
	var detector stt.VoiceDetector = vad
	var transcriber stt.Transcriber = recognizer
	log.Println("✅ Speech recognition ready")

	// Create TTS synthesizer (implements tts.Synthesizer)
	log.Println("🔊 Loading text-to-speech models...")
	synth, err := tts.NewKokoroSynthesizer(&tts.KokoroConfig{
		Model:      cfg.TTSModel,
		Voices:     cfg.TTSVoices,
		Tokens:     cfg.TTSTokens,
		DataDir:    cfg.TTSData,
		Lexicon:    cfg.TTSLexicon,
		Language:   cfg.TTSLanguage,
		SpeakerID:  cfg.TTSSpeakerID,
		Speed:      cfg.TTSSpeed,
		Provider:   cfg.TTSProvider,
		Verbose:    cfg.Verbose,
		TTSThreads: cfg.TTSThreads,
	})
	if err != nil {
		log.Fatalf("Failed to create TTS synthesizer: %v", err)
	}
	defer synth.Close()

	// Expose via interface so the pipeline is implementation-agnostic.
	var synthesizer tts.Synthesizer = synth
	log.Println("✅ Text-to-speech ready")

	// Create interrupt flag for playback
	var playbackInterrupt atomic.Bool

	// Create audio player
	player, err := audio.NewPlayer(synthesizer.SampleRate(), cfg.AudioBufferMs, &playbackInterrupt)
	if err != nil {
		log.Fatalf("Failed to create audio player: %v", err)
	}
	defer player.Close()

	// Channels for pipeline communication
	transcriptions := make(chan string, 5)
	responses := make(chan string, 5)

	// Create audio capturer
	capturer, err := audio.NewCapturer(cfg.SampleRate, func(samples []float32) {
		detector.AcceptWaveform(samples)
	})
	if err != nil {
		log.Fatalf("Failed to create audio capturer: %v", err)
	}
	defer capturer.Close()

	// WaitGroup for goroutines
	var wg sync.WaitGroup

	// Start STT processing goroutine (interface-based, model-agnostic)
	wg.Add(1)
	go func() {
		defer wg.Done()
		stt.RunProcessor(ctx, detector, transcriber, transcriptions, &playbackInterrupt, cfg.Verbose)
	}()

	// Start LLM processing goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		llmClient.RunProcessor(ctx, transcriptions, responses)
	}()

	// Start TTS and playback goroutine (interface-based, model-agnostic)
	wg.Add(1)
	go func() {
		defer wg.Done()
		tts.RunProcessor(ctx, synthesizer, player, responses, &playbackInterrupt, cfg, capturer)
	}()

	// Start audio capture
	if err := capturer.Start(); err != nil {
		log.Fatalf("Failed to start audio capture: %v", err)
	}

	if cfg.WakeWord != "" {
		log.Printf("🎙️ Listening for wake word: %q", cfg.WakeWord)
	} else {
		log.Println("🎙️ Listening... (speak to interact, Ctrl+C to quit)")
	}

	// Wait for shutdown signal
	<-ctx.Done()
	log.Println("🛑 Shutting down...")

	// Stop capture first
	capturer.Stop()

	// Close channels
	close(transcriptions)

	// Wait for goroutines to finish
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		log.Println("✅ Shutdown complete")
	case <-time.After(5 * time.Second):
		log.Println("⚠️ Shutdown timeout, forcing exit")
	}
}

// runSetup downloads all model files required by the configured STT and TTS backends.
func runSetup(cfg *config.Config) error {
	log.Println("🔧 Voice Assistant Setup — downloading model files")
	log.Printf("   Model directory: %s", cfg.ModelDir)
	if cfg.Force {
		log.Println("   Mode: force re-download")
	} else {
		log.Println("   Mode: skip existing files")
	}

	sileroProvider := &stt.SileroModelProvider{}
	whisperProvider := &stt.WhisperModelProvider{ModelSize: cfg.WhisperModel}
	ttsProvider := &tts.KokoroModelProvider{}

	log.Printf("📥 [VAD] %s — downloading models…", sileroProvider.Name())
	if err := sileroProvider.EnsureModels(cfg.ModelDir, cfg.Force); err != nil {
		return fmt.Errorf("VAD model download: %w", err)
	}

	log.Printf("📥 [STT] %s — downloading models…", whisperProvider.Name())
	if err := whisperProvider.EnsureModels(cfg.ModelDir, cfg.Force); err != nil {
		return fmt.Errorf("STT model download: %w", err)
	}

	log.Printf("📥 [TTS] %s — downloading models…", ttsProvider.Name())
	if err := ttsProvider.EnsureModels(cfg.ModelDir, cfg.Force); err != nil {
		return fmt.Errorf("TTS model download: %w", err)
	}

	// Final verification
	log.Println("🔍 Verifying model files…")
	var allMissing []string
	allMissing = append(allMissing, sileroProvider.VerifyModels(cfg.ModelDir)...)
	allMissing = append(allMissing, whisperProvider.VerifyModels(cfg.ModelDir)...)
	allMissing = append(allMissing, ttsProvider.VerifyModels(cfg.ModelDir)...)

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

func init() {
	// Configure logging
	log.SetFlags(log.Ltime)
	log.SetOutput(os.Stdout)
}
