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

	"github.com/agalue/voice-assistant/internal/audio"
	"github.com/agalue/voice-assistant/internal/config"
	"github.com/agalue/voice-assistant/internal/llm"
	"github.com/agalue/voice-assistant/internal/stt"
	"github.com/agalue/voice-assistant/internal/tts"
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
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup signal handling
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Create LLM client and verify connection
	llmClient, err := llm.NewClient(&llm.Config{
		Host:         cfg.OllamaURL,
		Model:        cfg.OllamaModel,
		SystemPrompt: cfg.SystemPrompt,
		Verbose:      cfg.Verbose,
		MaxHistory:   cfg.MaxHistory,
		Temperature:  cfg.Temperature,
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
		sttProcessor(ctx, recognizer, transcriptions, &playbackInterrupt, cfg.Verbose)
	}()

	// Start LLM processing goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		llmProcessor(ctx, llmClient, transcriptions, responses)
	}()

	// Start TTS and playback goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		ttsProcessor(ctx, synthesizer, player, responses, &playbackInterrupt, cfg, capturer)
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
	<-sigChan
	log.Println("\n🛑 Shutting down...")

	// Stop capture first
	capturer.Stop()

	// Cancel context to stop goroutines
	cancel()

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

// sttProcessor polls the recognizer for transcriptions.
// sttProcessor receives speech segments via channel and transcribes them.
// Event-driven approach eliminates 50-100ms polling delay.
func sttProcessor(ctx context.Context, recognizer *stt.Recognizer, out chan<- string, interrupt *atomic.Bool, verbose bool) {
	for {
		select {
		case <-ctx.Done():
			return
		case samples, ok := <-recognizer.SegmentChannel():
			if !ok {
				// Channel closed
				return
			}

			// Set interrupt flag when processing new speech
			if recognizer.IsSpeechDetected() {
				interrupt.Store(true)
			}

			text := recognizer.TranscribeSegment(samples)
			if text != "" {
				if verbose {
					log.Printf("[STT] Transcription received (%d chars)", len(text))
				}

				select {
				case out <- text:
					// Clear interrupt flag after sending new speech for processing
					// This ensures the next response isn't immediately interrupted
					interrupt.Store(false)
					if verbose {
						log.Println("[STT] Transcription sent to LLM processor")
					}
				case <-ctx.Done():
					return
				}
			}
		}
	}
}

// llmProcessor handles LLM requests with streaming support.
// Streams LLM responses sentence-by-sentence to reduce perceived latency
// while maintaining natural TTS quality (complete sentences preserve prosody).
func llmProcessor(ctx context.Context, client *llm.Client, in <-chan string, out chan<- string) {
	for {
		select {
		case <-ctx.Done():
			return
		case text, ok := <-in:
			if !ok {
				return
			}

			log.Printf("🧠 Processing: %q", text)

			// Get complete response from LLM
			response, err := client.Chat(ctx, text)
			if err != nil {
				log.Printf("❌ LLM error: %v", err)
				// Send error response for TTS
				select {
				case out <- "I'm sorry, I encountered an error.":
				case <-ctx.Done():
					return
				}
				continue
			}

			log.Printf("🤖 Assistant: %s", response)

			// Send complete response for TTS processing
			select {
			case out <- response:
			case <-ctx.Done():
				return
			}
		}
	}
}

// ttsProcessor handles TTS synthesis and playback.
func ttsProcessor(ctx context.Context, synthesizer *tts.Synthesizer, player *audio.Player, in <-chan string, interrupt *atomic.Bool, cfg *config.Config, capturer *audio.Capturer) {
	for {
		select {
		case <-ctx.Done():
			return
		case text, ok := <-in:
			if !ok {
				return
			}

			// In 'always' mode, check if interrupted before synthesis
			if cfg.InterruptMode == config.InterruptAlways && interrupt.Load() {
				discarded := drainChannel(in)
				log.Printf("🗑️  Discarded %d queued LLM response(s) due to interruption", discarded+1)
				continue
			}

			// In 'wait' mode, pause microphone during synthesis/playback
			if cfg.InterruptMode == config.InterruptWait {
				capturer.Pause()
				if cfg.Verbose {
					log.Println("[TTS] Microphone paused for playback")
				}
			}

			// Stream synthesis and playback concurrently for lower latency
			// This allows playback of sentence N while synthesizing sentence N+1
			sentences := tts.SplitSentences(text)
			if len(sentences) == 0 {
				log.Println("⚠️  No sentences to synthesize")
				if cfg.InterruptMode == config.InterruptWait {
					time.Sleep(time.Duration(cfg.PostPlaybackDelayMs) * time.Millisecond)
					capturer.Resume()
				}
				continue
			}

			wasInterrupted := false

			// Process each sentence: synthesize and play immediately
			for i, sentence := range sentences {
				if sentence == "" {
					continue
				}

				// Check interruption before each sentence (in 'always' mode)
				if cfg.InterruptMode == config.InterruptAlways && interrupt.Load() {
					log.Println("⏸️  Synthesis interrupted by speech")
					wasInterrupted = true
					break
				}

				// Synthesize this sentence
				if cfg.Verbose {
					log.Printf("[TTS] Synthesizing sentence %d/%d: %q", i+1, len(sentences), sentence)
				}

				chunk, err := synthesizer.Synthesize(sentence)
				if err != nil {
					log.Printf("❌ TTS error for sentence %d: %v", i+1, err)
					continue // Skip failed sentence, try next
				}

				// Play immediately (don't wait for other sentences)
				log.Printf("🔊 Playing sentence %d/%d (%d samples)", i+1, len(sentences), len(chunk.Samples))

				// Play audio chunk
				if err := player.Play(audio.AudioBuffer{
					Samples:    chunk.Samples,
					SampleRate: chunk.SampleRate,
				}); err != nil {
					log.Printf("❌ Playback error: %v", err)
					wasInterrupted = true
					break
				}

				// Check if interrupted during playback (in 'always' mode)
				if cfg.InterruptMode == config.InterruptAlways && interrupt.Load() {
					log.Println("⏸️  Playback interrupted by speech")
					wasInterrupted = true
					break
				}
			}

			// Resume microphone after playback in 'wait' mode
			if cfg.InterruptMode == config.InterruptWait {
				// Add delay before resuming to avoid capturing playback tail
				time.Sleep(time.Duration(cfg.PostPlaybackDelayMs) * time.Millisecond)
				capturer.Resume()
				if cfg.Verbose {
					log.Println("[TTS] Microphone resumed after playback")
				}
			}

			// If interrupted in 'always' mode, drain remaining queued responses
			if wasInterrupted && cfg.InterruptMode == config.InterruptAlways {
				if discarded := drainChannel(in); discarded > 0 {
					log.Printf("🗑️  Discarded %d queued TTS response(s)", discarded)
				}
			}
		}
	}
}

// drainChannel removes all pending messages from a channel.
// Returns the number of messages drained.
func drainChannel[T any](ch <-chan T) int {
	discarded := 0
	for {
		select {
		case <-ch:
			discarded++
		default:
			return discarded
		}
	}
}

func init() {
	// Configure logging
	log.SetFlags(log.Ltime)
	log.SetOutput(os.Stdout)
}
