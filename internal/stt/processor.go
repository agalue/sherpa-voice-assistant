package stt

import (
	"context"
	"log"
	"sync/atomic"
)

// RunProcessor receives speech segments from the VAD channel and sends transcriptions.
// It is intended to be run as a goroutine and returns when ctx is cancelled or the
// segment channel is closed.
//
// interrupt is set to true when speech is detected (to stop any in-progress playback)
// and cleared to false after a transcription is successfully forwarded to out, so the
// next response is not immediately interrupted.
func (r *Recognizer) RunProcessor(ctx context.Context, out chan<- string, interrupt *atomic.Bool, verbose bool) {
	for {
		select {
		case <-ctx.Done():
			return
		case samples, ok := <-r.SegmentChannel():
			if !ok {
				return
			}

			// Set interrupt flag when new speech arrives to stop any active playback.
			if r.IsSpeechDetected() {
				interrupt.Store(true)
			}

			text := r.TranscribeSegment(samples)
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
