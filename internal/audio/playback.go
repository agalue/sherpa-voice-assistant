// Package audio provides audio playback functionality using malgo.
package audio

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gen2brain/malgo"
)

// Playback ring buffer configuration.
const (
	// playbackRingSize is the number of samples the ring buffer can hold.
	// Size: 524288 samples = ~11 seconds at 48kHz, ~22 seconds at 24kHz.
	// Large enough to buffer long TTS responses without overflow.
	playbackRingSize = 524288
)

// AudioBuffer holds audio samples with metadata.
type AudioBuffer struct {
	Samples    []float32 // Audio sample data (mono, floating point)
	SampleRate int       // Sample rate in Hz (e.g., 24000 for TTS)
}

// playbackRing is a lock-free single-producer single-consumer ring buffer for audio playback.
type playbackRing struct {
	samples [playbackRingSize]float32
	head    atomic.Uint64 // Write position (producer)
	tail    atomic.Uint64 // Read position (consumer)
}

// push adds samples to the ring buffer. Returns number of samples written.
func (rb *playbackRing) push(samples []float32) int {
	head := rb.head.Load()
	tail := rb.tail.Load()

	available := playbackRingSize - int(head-tail)
	toWrite := len(samples)
	if toWrite > available {
		toWrite = available
	}

	for i := 0; i < toWrite; i++ {
		rb.samples[(head+uint64(i))%playbackRingSize] = samples[i]
	}

	rb.head.Add(uint64(toWrite))
	return toWrite
}

// pop retrieves a sample from the ring buffer. Returns 0.0 if empty.
func (rb *playbackRing) pop() (float32, bool) {
	head := rb.head.Load()
	tail := rb.tail.Load()

	if head == tail {
		return 0.0, false // Empty
	}

	sample := rb.samples[tail%playbackRingSize]
	rb.tail.Add(1)
	return sample, true
}

// isEmpty returns true if the ring buffer is empty.
func (rb *playbackRing) isEmpty() bool {
	return rb.head.Load() == rb.tail.Load()
}

// clear resets the ring buffer.
func (rb *playbackRing) clear() {
	rb.tail.Store(rb.head.Load())
}

// Player handles audio playback with a persistent device and lock-free ring buffer.
// Supports interrupt-driven playback for responsive voice interaction.
type Player struct {
	ctx              *malgo.AllocatedContext // Malgo audio context
	device           *malgo.Device           // Audio output device
	sampleRate       uint32                  // Input sample rate (e.g., TTS output rate)
	deviceSampleRate uint32                  // Device's native sample rate
	bufferMs         uint32                  // Buffer size in milliseconds
	interrupt        *atomic.Bool            // Internal interrupt flag
	externalIntr     *atomic.Bool            // External interrupt flag (e.g., when user speaks)
	playing          atomic.Bool             // Flag indicating active playback
	ring             *playbackRing           // Lock-free ring buffer for samples
	mu               sync.Mutex              // Protects ring buffer writes (not callback)
	completeChan     chan struct{}           // Channel to signal playback completion
}

// NewPlayer creates a new audio player with a persistent playback device.
// bufferMs: audio buffer size in milliseconds (20ms for wired, 100ms for Bluetooth, 0 for default 100ms)
func NewPlayer(sampleRate int, bufferMs uint32, externalInterrupt *atomic.Bool) (*Player, error) {
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize audio context: %w", err)
	}

	// Default to 100ms if not specified (Bluetooth-friendly)
	if bufferMs == 0 {
		bufferMs = 100
	}

	// Query device's native sample rate once during initialization
	deviceSampleRate := getDeviceNativeSampleRate()
	log.Printf("🔊 Audio device sample rate: %d Hz (input: %d Hz), buffer: %d ms", deviceSampleRate, sampleRate, bufferMs)

	p := &Player{
		ctx:              ctx,
		sampleRate:       uint32(sampleRate),
		deviceSampleRate: deviceSampleRate,
		bufferMs:         bufferMs,
		externalIntr:     externalInterrupt,
		interrupt:        &atomic.Bool{},
		ring:             &playbackRing{},
		completeChan:     make(chan struct{}, 1), // Buffered to prevent blocking
	}

	// Initialize the persistent playback device
	if err := p.initDevice(); err != nil {
		ctx.Uninit()
		ctx.Free()
		return nil, err
	}

	return p, nil
}

// initDevice initializes and starts the persistent playback device.
func (p *Player) initDevice() error {
	deviceConfig := malgo.DefaultDeviceConfig(malgo.Playback)
	deviceConfig.Playback.Format = malgo.FormatF32
	deviceConfig.Playback.Channels = 1
	deviceConfig.SampleRate = p.deviceSampleRate
	deviceConfig.PeriodSizeInMilliseconds = p.bufferMs

	// Lock-free audio callback
	onSendFrames := func(pOutputSample, pInputSamples []byte, framecount uint32) {
		// Check for interrupts (lock-free)
		interrupted := p.interrupt.Load() || (p.externalIntr != nil && p.externalIntr.Load())

		for i := 0; i < int(framecount); i++ {
			var sample float32
			if !interrupted {
				if s, ok := p.ring.pop(); ok {
					sample = s
				}
			}
			binary.LittleEndian.PutUint32(pOutputSample[i*4:], math.Float32bits(sample))
		}

		// Mark playback as done if buffer is empty or interrupted
		if p.ring.isEmpty() || interrupted {
			p.playing.Store(false)
			// Non-blocking send to completion channel
			select {
			case p.completeChan <- struct{}{}:
			default:
				// Channel already has a signal, no need to send another
			}
		}
	}

	callbacks := malgo.DeviceCallbacks{
		Data: onSendFrames,
	}

	device, err := malgo.InitDevice(p.ctx.Context, deviceConfig, callbacks)
	if err != nil {
		return fmt.Errorf("failed to initialize playback device: %w", err)
	}

	p.device = device

	// Start the device immediately (it will output silence until samples are queued)
	if err := device.Start(); err != nil {
		device.Uninit()
		return fmt.Errorf("failed to start playback device: %w", err)
	}

	log.Printf("🔊 Persistent playback device started (lock-free)")
	return nil
}

// getDeviceNativeSampleRate queries the device's preferred sample rate.
// Falls back to 48000 Hz if unable to determine.
func getDeviceNativeSampleRate() uint32 {
	defaultConfig := malgo.DefaultDeviceConfig(malgo.Playback)
	if defaultConfig.SampleRate > 0 {
		return defaultConfig.SampleRate
	}
	return 48000
}

// Play plays the audio buffer, blocking until complete or interrupted.
func (p *Player) Play(buffer AudioBuffer) error {
	// Resample if device sample rate differs from input
	playbackSamples := buffer.Samples
	if buffer.SampleRate != int(p.deviceSampleRate) {
		log.Printf("🔄 Resampling audio: %d Hz -> %d Hz (%d samples -> %d samples)",
			buffer.SampleRate, p.deviceSampleRate, len(buffer.Samples),
			int(float64(len(buffer.Samples))*float64(p.deviceSampleRate)/float64(buffer.SampleRate)))
		playbackSamples = ResampleInPlace(buffer.Samples, buffer.SampleRate, int(p.deviceSampleRate))
	}

	// Reset interrupt flag
	p.interrupt.Store(false)

	// Queue samples to ring buffer
	p.mu.Lock()
	written := p.ring.push(playbackSamples)
	if written < len(playbackSamples) {
		log.Printf("⚠️  Playback buffer overflow, dropped %d samples", len(playbackSamples)-written)
	}
	p.mu.Unlock()

	// Mark as playing
	p.playing.Store(true)

	// Wait for playback to complete or be interrupted
	timeout := time.Duration(len(playbackSamples)/int(p.deviceSampleRate)+2) * time.Second

	// Use channel-based waiting (more idiomatic in Go than sync.Cond)
	for p.playing.Load() {
		if p.interrupt.Load() || (p.externalIntr != nil && p.externalIntr.Load()) {
			p.ring.clear()
			p.playing.Store(false)
			return nil
		}

		select {
		case <-p.completeChan:
			// Playback completed normally
		case <-time.After(50 * time.Millisecond):
			// Timeout to periodically check interrupt flags
		case <-time.After(timeout):
			log.Println("⚠️  Playback timeout exceeded")
			p.ring.clear()
			p.playing.Store(false)
			return nil
		}
	}

	return nil
}

// Interrupt stops current playback.
func (p *Player) Interrupt() {
	p.interrupt.Store(true)
	p.ring.clear()
	p.playing.Store(false)
	// Non-blocking send to completion channel
	select {
	case p.completeChan <- struct{}{}:
	default:
		// Channel already has a signal
	}
}

// Close releases all resources.
func (p *Player) Close() {
	p.Interrupt()
	if p.device != nil {
		p.device.Stop()
		p.device.Uninit()
		p.device = nil
	}
	if p.ctx != nil {
		_ = p.ctx.Uninit()
		p.ctx.Free()
		p.ctx = nil
	}
}
