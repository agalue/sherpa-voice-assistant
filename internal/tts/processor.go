package tts

import (
	"context"
	"log"
	"sync/atomic"
	"time"

	"github.com/agalue/voice-assistant/internal/audio"
	"github.com/agalue/voice-assistant/internal/config"
)

// RunProcessor handles TTS synthesis and audio playback for incoming LLM responses.
// It reads complete responses from in, splits them into sentences, and runs a pipelined
// synthesis+playback loop where sentence N+1 is synthesized concurrently with playback
// of sentence N to minimise perceived latency.
//
// Microphone pause/resume and playback interruption behaviour are controlled by
// cfg.InterruptMode. This function is intended to be run as a goroutine and returns
// when ctx is cancelled or in is closed.
func (s *Synthesizer) RunProcessor(
	ctx context.Context,
	player *audio.Player,
	in <-chan string,
	interrupt *atomic.Bool,
	cfg *config.Config,
	capturer *audio.Capturer,
) {
	for {
		select {
		case <-ctx.Done():
			return
		case text, ok := <-in:
			if !ok {
				return
			}

			// In 'always' mode, skip the entire response if the user is already speaking.
			if cfg.InterruptMode == config.InterruptAlways && interrupt.Load() {
				discarded := drainChannel(in)
				log.Printf("🗑️  Discarded %d queued LLM response(s) due to interruption", discarded+1)
				continue
			}

			// In 'wait' mode, pause the microphone for the duration of playback.
			if cfg.InterruptMode == config.InterruptWait {
				capturer.Pause()
				if cfg.Verbose {
					log.Println("[TTS] Microphone paused for playback")
				}
			}

			// Pipeline synthesis and playback concurrently for lower latency.
			// Synthesis of sentence N+1 overlaps with playback of sentence N.
			sentences := SplitSentences(text)
			if len(sentences) == 0 {
				log.Println("⚠️  No sentences to synthesize")
				if cfg.InterruptMode == config.InterruptWait {
					time.Sleep(time.Duration(cfg.PostPlaybackDelayMs) * time.Millisecond)
					capturer.Resume()
				}
				continue
			}

			wasInterrupted := false
			// synthExitedEarly is set by the synthesis goroutine when it exits due to
			// an interrupt before sending any audio, so the playback loop's normal
			// channel-close exit can still trigger the response drain.
			var synthExitedEarly atomic.Bool

			synthCtx, synthCancel := context.WithCancel(ctx)
			audioQueue := make(chan *AudioOutput, 1) // 1-slot buffer: prefetch next sentence

			go func() {
				defer close(audioQueue)
				for i, sentence := range sentences {
					if sentence == "" {
						continue
					}

					// Stop if the playback side cancelled (interruption or error).
					select {
					case <-synthCtx.Done():
						return
					default:
					}

					if cfg.InterruptMode == config.InterruptAlways && interrupt.Load() {
						synthExitedEarly.Store(true)
						return
					}

					if cfg.Verbose {
						log.Printf("[TTS] Synthesizing sentence %d/%d: %q", i+1, len(sentences), sentence)
					}

					chunk, err := s.Synthesize(sentence)
					if err != nil {
						log.Printf("❌ TTS error for sentence %d: %v", i+1, err)
						continue
					}

					// Send to playback; abort if cancelled while waiting.
					select {
					case audioQueue <- chunk:
					case <-synthCtx.Done():
						return
					}
				}
			}()

			sentNum := 0
			for chunk := range audioQueue {
				// Pre-play interrupt check: a chunk may have been queued before the
				// user started speaking; avoid playing it over them.
				if cfg.InterruptMode == config.InterruptAlways && interrupt.Load() {
					log.Println("⏸️  Playback interrupted by speech (pre-play)")
					synthCancel()
					wasInterrupted = true
					break
				}

				sentNum++
				log.Printf("🔊 Playing sentence %d/%d (%d samples)", sentNum, len(sentences), len(chunk.Samples))

				if err := player.Play(audio.AudioBuffer{
					Samples:    chunk.Samples,
					SampleRate: chunk.SampleRate,
				}); err != nil {
					log.Printf("❌ Playback error: %v", err)
					synthCancel()
					wasInterrupted = true
					break
				}

				if cfg.InterruptMode == config.InterruptAlways && interrupt.Load() {
					log.Println("⏸️  Playback interrupted by speech")
					synthCancel()
					wasInterrupted = true
					break
				}
			}

			synthCancel() // No-op if already called; ensures goroutine exits.

			// Propagate an interruption that occurred entirely inside the synthesis
			// goroutine (before any audio reached the playback loop), so the drain
			// below still runs when appropriate.
			if !wasInterrupted && synthExitedEarly.Load() {
				wasInterrupted = true
			}

			// Resume microphone after playback in 'wait' mode.
			if cfg.InterruptMode == config.InterruptWait {
				// Delay before resuming to avoid capturing the playback tail.
				time.Sleep(time.Duration(cfg.PostPlaybackDelayMs) * time.Millisecond)
				capturer.Resume()
				if cfg.Verbose {
					log.Println("[TTS] Microphone resumed after playback")
				}
			}

			// If interrupted in 'always' mode, drain any remaining queued responses.
			if wasInterrupted && cfg.InterruptMode == config.InterruptAlways {
				if discarded := drainChannel(in); discarded > 0 {
					log.Printf("🗑️  Discarded %d queued TTS response(s)", discarded)
				}
			}
		}
	}
}

// drainChannel removes all pending messages from ch and returns the count.
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
