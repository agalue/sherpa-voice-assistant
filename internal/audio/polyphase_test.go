package audio

import (
	"math"
	"testing"
)

// TestPolyphaseResamplerConstruction validates filter coefficient generation.
func TestPolyphaseResamplerConstruction(t *testing.T) {
	r := NewPolyphaseResampler(48000, 16000)

	// Verify filter coefficients sum to 1.0 (normalized)
	sum := float32(0.0)
	for _, coeff := range r.filter {
		sum += coeff
	}
	if math.Abs(float64(sum-1.0)) > 0.0001 {
		t.Errorf("Filter coefficients sum = %v, want 1.0", sum)
	}

	// Verify filter length is 64 taps
	if r.filterLen != 64 {
		t.Errorf("Filter length = %d, want 64", r.filterLen)
	}

	// Verify ratio is correct (16000/48000 = 0.333...)
	expectedRatio := 16000.0 / 48000.0
	if math.Abs(r.ratio-expectedRatio) > 0.0001 {
		t.Errorf("Ratio = %v, want %v", r.ratio, expectedRatio)
	}
}

// TestPolyphaseDownsampling validates downsampling with a known sine wave.
func TestPolyphaseDownsampling(t *testing.T) {
	// Generate 1kHz sine wave at 48kHz (well below Nyquist)
	duration := 0.1 // 100ms
	fromRate := 48000
	toRate := 16000
	freq := 1000.0 // 1kHz sine wave

	inputLen := int(float64(fromRate) * duration)
	input := make([]float32, inputLen)
	for i := 0; i < inputLen; i++ {
		input[i] = float32(math.Sin(2.0 * math.Pi * freq * float64(i) / float64(fromRate)))
	}

	r := NewPolyphaseResampler(fromRate, toRate)
	output := r.Resample(input)

	// Verify output length is approximately correct (48000 -> 16000 = 1/3)
	expectedLen := int(float64(inputLen) * float64(toRate) / float64(fromRate))
	if math.Abs(float64(len(output)-expectedLen)) > 5 {
		t.Errorf("Output length = %d, expected approximately %d", len(output), expectedLen)
	}

	// Verify output contains valid samples (not all zeros, not clipping)
	nonZero := 0
	for _, sample := range output {
		if sample != 0.0 {
			nonZero++
		}
		if math.Abs(float64(sample)) > 1.5 {
			t.Errorf("Sample clipping detected: %v", sample)
		}
	}
	if nonZero < len(output)/2 {
		t.Errorf("Too many zero samples: %d/%d", nonZero, len(output))
	}
}

// TestPolyphaseUpsampling validates upsampling uses linear interpolation.
func TestPolyphaseUpsampling(t *testing.T) {
	// Upsampling should use linear interpolation (upsample method)
	r := NewPolyphaseResampler(16000, 48000)

	input := []float32{0.0, 1.0, 0.0, -1.0, 0.0}
	output := r.Resample(input)

	// Verify output length is approximately 3x (16kHz -> 48kHz)
	expectedLen := int(float64(len(input)) * 3.0)
	if math.Abs(float64(len(output)-expectedLen)) > 2 {
		t.Errorf("Output length = %d, expected approximately %d", len(output), expectedLen)
	}

	// Verify interpolation creates intermediate values
	if len(output) > 5 {
		hasIntermediate := false
		for _, sample := range output[1:5] {
			if sample != 0.0 && sample != 1.0 && sample != -1.0 {
				hasIntermediate = true
				break
			}
		}
		if !hasIntermediate {
			t.Error("Upsampling should create intermediate values via interpolation")
		}
	}
}

// TestPolyphaseHistoryBuffer validates continuity across multiple chunks.
func TestPolyphaseHistoryBuffer(t *testing.T) {
	r := NewPolyphaseResampler(48000, 16000)

	// Feed same sine wave in two chunks
	freq := 500.0
	rate := 48000.0
	chunk1 := make([]float32, 1000)
	chunk2 := make([]float32, 1000)

	for i := 0; i < 1000; i++ {
		chunk1[i] = float32(math.Sin(2.0 * math.Pi * freq * float64(i) / rate))
		chunk2[i] = float32(math.Sin(2.0 * math.Pi * freq * float64(i+1000) / rate))
	}

	out1 := r.Resample(chunk1)
	out2 := r.Resample(chunk2)

	// Both chunks should produce output
	if len(out1) == 0 || len(out2) == 0 {
		t.Error("History buffer should allow continuous processing")
	}

	// Verify history buffer is populated (64 samples)
	if len(r.history) != 64 {
		t.Errorf("History buffer length = %d, want 64", len(r.history))
	}

	// Verify history contains non-zero values after first chunk
	nonZero := 0
	for _, h := range r.history {
		if h != 0.0 {
			nonZero++
		}
	}
	if nonZero == 0 {
		t.Error("History buffer should contain non-zero samples after first chunk")
	}
}

// TestPolyphaseEdgeCases validates edge case handling.
func TestPolyphaseEdgeCases(t *testing.T) {
	r := NewPolyphaseResampler(48000, 16000)

	// Empty input
	output := r.Resample([]float32{})
	if len(output) != 0 {
		t.Errorf("Empty input: got %d samples, want 0", len(output))
	}

	// Single sample (downsampling ratio < 1.0 means output length rounds to 0)
	output = r.Resample([]float32{0.5})
	// For downsampling with ratio 0.333, output length = int(1 * 0.333) = 0
	// This is expected behavior - need multiple input samples to produce output

	// Exact rate match (ratio = 1.0)
	r2 := NewPolyphaseResampler(16000, 16000)
	input := []float32{1.0, 2.0, 3.0, 4.0}
	output = r2.Resample(input)
	if len(output) != len(input) {
		t.Errorf("Rate match: got %d samples, want %d", len(output), len(input))
	}
	for i := range input {
		if output[i] != input[i] {
			t.Errorf("Rate match should passthrough: output[%d] = %v, want %v", i, output[i], input[i])
		}
	}

	// Input shorter than filter length but enough to produce output
	shortInput := make([]float32, 32) // Less than 64-tap filter
	for i := range shortInput {
		shortInput[i] = float32(i)
	}
	output = r.Resample(shortInput)
	// Should handle gracefully without panic (output = 32 * 0.333 = ~10 samples)
	if len(output) == 0 {
		t.Error("Short input should still produce output when sufficient samples")
	}

	// Test upsampling with single sample (ratio > 1.0)
	r3 := NewPolyphaseResampler(16000, 48000)
	output = r3.Resample([]float32{0.5})
	// Upsampling: output length = 1 * 3 = 3 samples
	if len(output) != 3 {
		t.Errorf("Upsampling single sample: got %d samples, want 3", len(output))
	}
}

// TestResamplePolyphaseConvenience validates the convenience function.
func TestResamplePolyphaseConvenience(t *testing.T) {
	input := make([]float32, 1000)
	for i := range input {
		input[i] = float32(math.Sin(2.0 * math.Pi * 440.0 * float64(i) / 48000.0))
	}

	// Downsampling: should use polyphase
	output := ResamplePolyphase(input, 48000, 16000)
	expectedLen := int(float64(len(input)) * 16000.0 / 48000.0)
	if math.Abs(float64(len(output)-expectedLen)) > 5 {
		t.Errorf("Downsampling: got %d samples, want ~%d", len(output), expectedLen)
	}

	// Rate match: should passthrough
	output = ResamplePolyphase(input, 48000, 48000)
	if len(output) != len(input) {
		t.Errorf("Rate match: got %d samples, want %d", len(output), len(input))
	}

	// Upsampling: should use linear interpolation (from resampler.go)
	output = ResamplePolyphase(input, 16000, 48000)
	expectedLen = int(float64(len(input)) * 48000.0 / 16000.0)
	if math.Abs(float64(len(output)-expectedLen)) > 5 {
		t.Errorf("Upsampling: got %d samples, want ~%d", len(output), expectedLen)
	}
}
