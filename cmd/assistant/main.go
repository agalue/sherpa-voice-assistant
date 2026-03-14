// Voice Assistant - A Go implementation using sherpa-onnx
//
// This program implements a real-time voice assistant with:
// - Voice Activity Detection (Silero-VAD)
// - Speech-to-Text (Whisper)
// - LLM Integration (Ollama)
// - Text-to-Speech (Kokoro)
package main

import (
	"context"
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

	// Create STT recognizer
	log.Println("🧠 Loading speech recognition models...")
	recognizer, err := stt.NewRecognizer(&stt.Config{
		VADModel:           cfg.VADModel,
		VADThreshold:       cfg.VadThreshold,
		VADSilenceDuration: cfg.VADSilenceDuration,
		WhisperEncoder:     cfg.WhisperEncoder,
		WhisperDecoder:     cfg.WhisperDecoder,
		WhisperTokens:      cfg.WhisperTokens,
		SampleRate:         cfg.SampleRate,
		WakeWord:           cfg.WakeWord,
		Provider:           cfg.STTProvider,
		Language:           cfg.STTLanguage,
		Verbose:            cfg.Verbose,
		VADThreads:         cfg.VADThreads,
		STTThreads:         cfg.STTThreads,
	})
	if err != nil {
		log.Fatalf("Failed to create STT recognizer: %v", err)
	}
	defer recognizer.Close()
	log.Println("✅ Speech recognition ready")

	// Create TTS synthesizer
	log.Println("🔊 Loading text-to-speech models...")
	synthesizer, err := tts.NewSynthesizer(&tts.Config{
		Model:      cfg.TTSModel,
		Voices:     cfg.TTSVoices,
		Tokens:     cfg.TTSTokens,
		DataDir:    cfg.TTSData,
		Lexicon:    cfg.TTSLexicon,
		Language:   cfg.TTSLanguage, // Required for multi-lingual Kokoro v1.0+
		SpeakerID:  cfg.TTSSpeakerID,
		Speed:      cfg.TTSSpeed,
		Provider:   cfg.TTSProvider, // CoreML on macOS, CUDA on Linux
		Verbose:    cfg.Verbose,
		TTSThreads: cfg.TTSThreads,
	})
	if err != nil {
		log.Fatalf("Failed to create TTS synthesizer: %v", err)
	}
	defer synthesizer.Close()
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
		recognizer.AcceptWaveform(samples)
	})
	if err != nil {
		log.Fatalf("Failed to create audio capturer: %v", err)
	}
	defer capturer.Close()

	// WaitGroup for goroutines
	var wg sync.WaitGroup

	// Start STT processing goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		recognizer.RunProcessor(ctx, transcriptions, &playbackInterrupt, cfg.Verbose)
	}()

	// Start LLM processing goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		llmClient.RunProcessor(ctx, transcriptions, responses)
	}()

	// Start TTS and playback goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		synthesizer.RunProcessor(ctx, player, responses, &playbackInterrupt, cfg, capturer)
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

func init() {
	// Configure logging
	log.SetFlags(log.Ltime)
	log.SetOutput(os.Stdout)
}
