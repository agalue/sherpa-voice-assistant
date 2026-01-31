// Package audio provides audio capture functionality using malgo.
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

// Ring buffer configuration constants.
const (
	// ringBufferSize is the number of sample chunks the ring buffer can hold.
	// At 16kHz with 32ms chunks (512 samples), this provides ~4 seconds of buffer.
	// This size balances memory usage with sufficient buffering for VAD processing.
	ringBufferSize = 128

	// maxSamplesPerChunk is the maximum samples per audio callback chunk.
	// This limit prevents excessive memory allocation in the audio callback path.
	maxSamplesPerChunk = 2048
)

// audioChunk represents a chunk of audio samples in the ring buffer.
type audioChunk struct {
	samples []float32 // Pre-allocated buffer for audio samples
	len     int       // Actual number of valid samples in the buffer
}

// ringBuffer is a lock-free single-producer single-consumer ring buffer for audio.
// Uses atomic operations for thread-safe access without mutex locks.
type ringBuffer struct {
	chunks    [ringBufferSize]audioChunk // Fixed-size array of pre-allocated chunks
	head      atomic.Uint64              // Write position (producer increments)
	tail      atomic.Uint64              // Read position (consumer increments)
	dropCount atomic.Uint64              // Number of dropped chunks due to overflow
}

// newRingBuffer creates a new ring buffer with pre-allocated chunks.
func newRingBuffer() *ringBuffer {
	rb := &ringBuffer{}
	for i := range rb.chunks {
		rb.chunks[i].samples = make([]float32, maxSamplesPerChunk)
	}
	return rb
}

// push adds samples to the ring buffer.
// Returns false if buffer is full, causing samples to be dropped.
func (rb *ringBuffer) push(samples []float32) bool {
	head := rb.head.Load()
	tail := rb.tail.Load()

	// Check if buffer is full
	if head-tail >= ringBufferSize {
		count := rb.dropCount.Add(1)
		if count%100 == 0 {
			log.Printf("⚠️  Audio ring buffer full, dropped %d chunks", count)
		}
		return false
	}

	// Copy samples to the next slot
	slot := &rb.chunks[head%ringBufferSize]
	n := copy(slot.samples, samples)
	slot.len = n

	rb.head.Add(1)
	return true
}

// pop retrieves samples from the ring buffer.
// Returns nil if buffer is empty.
func (rb *ringBuffer) pop() []float32 {
	head := rb.head.Load()
	tail := rb.tail.Load()

	if head == tail {
		return nil // Empty
	}

	slot := &rb.chunks[tail%ringBufferSize]
	samples := slot.samples[:slot.len]

	rb.tail.Add(1)
	return samples
}

// Capturer handles microphone audio capture with backpressure support.
// Uses a lock-free ring buffer to prevent audio callback blocking.
type Capturer struct {
	ctx              *malgo.AllocatedContext // Malgo audio context
	device           *malgo.Device           // Audio input device
	sampleRate       uint32                  // Target sample rate (e.g., 16kHz for STT)
	deviceSampleRate uint32                  // Actual device sample rate
	onSamples        func(samples []float32) // Callback for processed samples
	running          atomic.Bool             // Flag for pause/resume (temporary)
	ringBuf          *ringBuffer             // Lock-free buffer for audio callback
	stopChan         chan struct{}           // Channel to signal shutdown
	wg               sync.WaitGroup          // Wait group for goroutine cleanup
	resampler        *PolyphaseResampler     // Resampler for downsampling with anti-aliasing
}

// NewCapturer creates a new audio capturer with ring buffer for backpressure.
func NewCapturer(sampleRate int, onSamples func(samples []float32)) (*Capturer, error) {
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize audio context: %w", err)
	}

	c := &Capturer{
		ctx:        ctx,
		sampleRate: uint32(sampleRate),
		onSamples:  onSamples,
		ringBuf:    newRingBuffer(),
		stopChan:   make(chan struct{}),
	}

	return c, nil
}

// Start begins audio capture from the default microphone.
// Audio is buffered in a ring buffer and processed by a dedicated goroutine
// to avoid blocking the audio callback.
func (c *Capturer) Start() error {
	deviceConfig := malgo.DefaultDeviceConfig(malgo.Capture)
	deviceConfig.Capture.Format = malgo.FormatF32
	deviceConfig.Capture.Channels = 1

	// Try to use the target sample rate, but device may use a different rate
	deviceConfig.SampleRate = c.sampleRate
	deviceConfig.PeriodSizeInMilliseconds = 32 // Low latency: 32ms chunks

	// Query actual device sample rate (may differ from requested)
	tempDevice, err := malgo.InitDevice(c.ctx.Context, deviceConfig, malgo.DeviceCallbacks{})
	if err != nil {
		return fmt.Errorf("failed to query capture device: %w", err)
	}
	c.deviceSampleRate = tempDevice.SampleRate()
	tempDevice.Uninit()

	// Create resampler if device rate differs from target rate
	if c.deviceSampleRate != c.sampleRate {
		if c.deviceSampleRate > c.sampleRate {
			// Downsampling: use polyphase filter to prevent aliasing
			c.resampler = NewPolyphaseResampler(int(c.deviceSampleRate), int(c.sampleRate))
			log.Printf("🔄 Audio resampling: %d Hz -> %d Hz (polyphase anti-aliasing)", c.deviceSampleRate, c.sampleRate)
		} else {
			// Upsampling: will use linear interpolation in processLoop
			log.Printf("🔄 Audio resampling: %d Hz -> %d Hz (linear interpolation)", c.deviceSampleRate, c.sampleRate)
		}
	}

	// Audio callback - runs in audio thread, must be fast and non-blocking
	onRecvFrames := func(pOutputSample, pInputSamples []byte, framecount uint32) {
		if !c.running.Load() {
			return
		}

		// Convert byte buffer to float32 samples (uses pooled buffer)
		pooledSamples := bytesToFloat32(pInputSamples)
		if len(pooledSamples) > 0 {
			// Push to ring buffer (lock-free, never blocks)
			c.ringBuf.push(pooledSamples)
		}
		returnFloat32Buffer(pooledSamples)
	}

	callbacks := malgo.DeviceCallbacks{
		Data: onRecvFrames,
	}

	device, err := malgo.InitDevice(c.ctx.Context, deviceConfig, callbacks)
	if err != nil {
		return fmt.Errorf("failed to initialize capture device: %w", err)
	}

	c.device = device
	c.running.Store(true)

	// Start the consumer goroutine that drains the ring buffer
	c.wg.Add(1)
	go c.processLoop()

	if err := device.Start(); err != nil {
		return fmt.Errorf("failed to start capture device: %w", err)
	}

	return nil
}

// processLoop drains the ring buffer and calls onSamples.
// This runs in a dedicated goroutine, separate from the audio callback.
func (c *Capturer) processLoop() {
	defer c.wg.Done()

	for {
		select {
		case <-c.stopChan:
			return
		default:
			samples := c.ringBuf.pop()
			if samples != nil && c.onSamples != nil && c.running.Load() {
				// Make a copy since the ring buffer slot will be reused
				samplesCopy := make([]float32, len(samples))
				copy(samplesCopy, samples)

				// Apply resampling if needed
				if c.resampler != nil {
					samplesCopy = c.resampler.Resample(samplesCopy)
				} else if c.deviceSampleRate != c.sampleRate {
					// Fallback to linear interpolation for upsampling
					samplesCopy = ResampleInPlace(samplesCopy, int(c.deviceSampleRate), int(c.sampleRate))
				}

				c.onSamples(samplesCopy)
			} else {
				// No samples available, sleep briefly to avoid busy-spinning
				// 100µs maintains low latency while reducing CPU usage significantly
				select {
				case <-c.stopChan:
					return
				case <-time.After(100 * time.Microsecond):
					// Continue checking for samples
				}
			}
		}
	}
}

// Stop halts audio capture.
func (c *Capturer) Stop() {
	c.running.Store(false)

	// Signal the process loop to stop
	select {
	case <-c.stopChan:
		// Already closed
	default:
		close(c.stopChan)
	}

	// Wait for process loop to finish
	c.wg.Wait()

	if c.device != nil {
		c.device.Stop()
		c.device.Uninit()
		c.device = nil
	}
}

// Pause temporarily halts audio capture (for half-duplex mode).
func (c *Capturer) Pause() {
	c.running.Store(false)
}

// Resume restarts audio capture after pause (for half-duplex mode).
func (c *Capturer) Resume() {
	c.running.Store(true)
}

// Close releases all audio resources.
func (c *Capturer) Close() {
	c.Stop()
	if c.ctx != nil {
		_ = c.ctx.Uninit()
		c.ctx.Free()
		c.ctx = nil
	}
}

// float32Pool reduces allocations in the audio callback hot path.
// Buffers are sized for typical 32ms audio chunks at various sample rates.
var float32Pool = sync.Pool{
	New: func() interface{} {
		// Pre-allocate for 32ms at 48kHz (1536 samples) with headroom
		buf := make([]float32, 2048)
		return &buf
	},
}

// bytesToFloat32 converts raw bytes to float32 samples.
// The returned slice is only valid until the next call - caller must copy if needed.
func bytesToFloat32(data []byte) []float32 {
	numSamples := len(data) / 4
	pBuf := float32Pool.Get().(*[]float32)

	// Ensure buffer is large enough
	if cap(*pBuf) < numSamples {
		*pBuf = make([]float32, numSamples)
	}
	samples := (*pBuf)[:numSamples]

	for i := range samples {
		bits := binary.LittleEndian.Uint32(data[i*4:])
		samples[i] = math.Float32frombits(bits)
	}
	return samples
}

// returnFloat32Buffer returns a buffer to the pool.
// Must be called after the samples from bytesToFloat32 are no longer needed.
func returnFloat32Buffer(samples []float32) {
	if samples == nil {
		return
	}
	// Get the underlying array and return to pool
	buf := samples[:cap(samples)]
	float32Pool.Put(&buf)
}
