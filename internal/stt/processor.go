package stt

import (
	"context"
	"log"
	"sync/atomic"
)

// RunProcessor receives speech segments from the VAD channel and sends transcriptions.
// It accepts the [VoiceDetector] and [Transcriber] interfaces so it is not coupled to
// any specific STT implementation. It is intended to run as a goroutine and returns
// when ctx is cancelled or the segment channel is closed.
//
// interrupt is set to true when speech is detected (to stop any in-progress playback)
// and cleared to false after a transcription is successfully forwarded to out, so the
// next response is not immediately interrupted.
func RunProcessor(ctx context.Context, detector VoiceDetector, transcriber Transcriber, out chan<- string, interrupt *atomic.Bool, verbose bool) {
	for {
		select {
		case <-ctx.Done():
			return
		case samples, ok := <-detector.SegmentChannel():
			if !ok {
				return
			}

			// Set interrupt flag when new speech arrives to stop any active playback.
			if detector.IsSpeechDetected() {
				interrupt.Store(true)
			}

			text := transcriber.TranscribeSegment(samples)
			if text == "" {
				continue
			}

			if verbose {
				log.Printf("[STT] Transcription received (%d chars)", len(text))
			}

			select {
			case out <- text:
				// Clear interrupt after forwarding so the next response is not
				// immediately interrupted before it even starts playing.
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
