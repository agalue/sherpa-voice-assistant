package audio

import (
	"math"
	"testing"
)

// TestResamplerConstruction validates resampler initialization.
func TestResamplerConstruction(t *testing.T) {
	r := NewResampler(16000, 48000)

	if r.fromRate != 16000.0 {
		t.Errorf("fromRate = %v, want 16000", r.fromRate)
	}
	if r.toRate != 48000.0 {
		t.Errorf("toRate = %v, want 48000", r.toRate)
	}
	if math.Abs(r.ratio-3.0) > 0.0001 {
		t.Errorf("ratio = %v, want 3.0", r.ratio)
	}
	if r.lastSample != 0.0 {
		t.Errorf("lastSample = %v, want 0.0", r.lastSample)
	}
}

// TestResamplerUpsampling validates upsampling to a higher sample rate.
func TestResamplerUpsampling(t *testing.T) {
	r := NewResampler(16000, 48000)

	// Generate simple ramp signal
	input := []float32{0.0, 0.25, 0.5, 0.75, 1.0}
	output := r.Resample(input)

	// Verify output length (16kHz -> 48kHz = 3x)
	expectedLen := len(input) * 3
	if len(output) != expectedLen {
		t.Errorf("Output length = %d, want %d", len(output), expectedLen)
	}

	// Verify interpolation creates intermediate values
	hasIntermediate := false
	for _, sample := range output {
		// Check if any values are between input samples (not exact matches)
		if sample > 0.0 && sample < 0.25 {
			hasIntermediate = true
			break
		}
	}
	if !hasIntermediate && len(output) > 5 {
		t.Error("Upsampling should create intermediate interpolated values")
	}

	// Verify lastSample is stored
	if r.lastSample != input[len(input)-1] {
		t.Errorf("lastSample = %v, want %v", r.lastSample, input[len(input)-1])
	}
}

// TestResamplerDownsampling validates downsampling to a lower sample rate.
func TestResamplerDownsampling(t *testing.T) {
	r := NewResampler(48000, 16000)

	// Generate simple signal at 48kHz
	inputLen := 48
	input := make([]float32, inputLen)
	for i := 0; i < inputLen; i++ {
		input[i] = float32(math.Sin(2.0 * math.Pi * 100.0 * float64(i) / 48000.0))
	}

	output := r.Resample(input)

	// Verify output length (48kHz -> 16kHz = 1/3)
	expectedLen := int(float64(inputLen) / 3.0)
	if len(output) != expectedLen {
		t.Errorf("Output length = %d, want %d", len(output), expectedLen)
	}

	// Verify output contains valid samples
	for i, sample := range output {
		if math.Abs(float64(sample)) > 1.5 {
			t.Errorf("Sample %d clipping: %v", i, sample)
		}
	}

	// Verify lastSample is stored
	if r.lastSample != input[len(input)-1] {
		t.Errorf("lastSample = %v, want %v", r.lastSample, input[len(input)-1])
	}
}

// TestResamplerContinuity validates lastSample state across chunks.
func TestResamplerContinuity(t *testing.T) {
	r := NewResampler(16000, 48000)

	// First chunk
	chunk1 := []float32{0.0, 0.5, 1.0}
	output1 := r.Resample(chunk1)
	if len(output1) == 0 {
		t.Error("First chunk should produce output")
	}

	// Verify lastSample is set
	if r.lastSample != 1.0 {
		t.Errorf("After chunk1: lastSample = %v, want 1.0", r.lastSample)
	}

	// Second chunk starts where first ended
	chunk2 := []float32{0.75, 0.5, 0.25, 0.0}
	output2 := r.Resample(chunk2)
	if len(output2) == 0 {
		t.Error("Second chunk should produce output")
	}

	// Verify continuity: first sample of output2 should use lastSample from chunk1
	// The resampler uses lastSample for interpolation at chunk boundaries
	if r.lastSample != 0.0 {
		t.Errorf("After chunk2: lastSample = %v, want 0.0", r.lastSample)
	}
}

// TestResamplerEdgeCases validates edge case handling.
func TestResamplerEdgeCases(t *testing.T) {
	r := NewResampler(48000, 16000)

	// Empty input
	output := r.Resample([]float32{})
	if len(output) != 0 {
		t.Errorf("Empty input: got %d samples, want 0", len(output))
	}

	// Single sample (downsampling ratio 1/3 means output = int(1 * 1/3) = 0)
	output = r.Resample([]float32{0.5})
	expectedLen := 0 // Downsampling ratio rounds to 0
	if len(output) != expectedLen {
		t.Errorf("Single sample downsampling: got %d samples, want %d", len(output), expectedLen)
	}

	// Single sample with upsampling (should produce output)
	r3 := NewResampler(16000, 48000)
	output = r3.Resample([]float32{0.5})
	expectedLen = 3 // Upsampling 3x
	if len(output) != expectedLen {
		t.Errorf("Single sample upsampling: got %d samples, want %d", len(output), expectedLen)
	}

	// Ratio = 1.0 (rate match)
	r2 := NewResampler(24000, 24000)
	input := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	output = r2.Resample(input)
	if len(output) != len(input) {
		t.Errorf("Rate match: got %d samples, want %d", len(output), len(input))
	}
	for i := range input {
		if output[i] != input[i] {
			t.Errorf("Rate match passthrough failed: output[%d] = %v, want %v", i, output[i], input[i])
		}
	}
}

// TestResamplerFractionalRatio validates non-integer ratio resampling.
func TestResamplerFractionalRatio(t *testing.T) {
	// 1.5x upsampling (e.g., 16kHz -> 24kHz)
	r := NewResampler(16000, 24000)

	input := []float32{0.0, 1.0, 2.0, 3.0, 4.0}
	output := r.Resample(input)

	// Verify fractional ratio calculation
	expectedLen := int(float64(len(input)) * 1.5)
	if len(output) != expectedLen {
		t.Errorf("Fractional ratio: got %d samples, want %d", len(output), expectedLen)
	}

	// Verify output is not all zeros
	nonZero := 0
	for _, sample := range output {
		if sample != 0.0 {
			nonZero++
		}
	}
	if nonZero == 0 {
		t.Error("Fractional ratio should produce non-zero output")
	}
}

// TestResampleInPlace validates the convenience function.
func TestResampleInPlace(t *testing.T) {
	input := []float32{0.0, 1.0, 2.0, 3.0, 4.0}

	// Rate match: should return input unchanged
	output := ResampleInPlace(input, 24000, 24000)
	if len(output) != len(input) {
		t.Errorf("Rate match: got %d samples, want %d", len(output), len(input))
	}

	// Upsampling
	output = ResampleInPlace(input, 16000, 48000)
	expectedLen := len(input) * 3
	if len(output) != expectedLen {
		t.Errorf("Upsampling: got %d samples, want %d", len(output), expectedLen)
	}

	// Downsampling
	output = ResampleInPlace(input, 48000, 16000)
	expectedLen = int(float64(len(input)) / 3.0)
	if len(output) != expectedLen {
		t.Errorf("Downsampling: got %d samples, want %d", len(output), expectedLen)
	}
}

// TestResamplerWithSineWave validates resampling quality with a known frequency.
func TestResamplerWithSineWave(t *testing.T) {
	// Generate 440Hz sine wave at 16kHz
	duration := 0.05 // 50ms
	fromRate := 16000
	toRate := 48000
	freq := 440.0

	inputLen := int(float64(fromRate) * duration)
	input := make([]float32, inputLen)
	for i := 0; i < inputLen; i++ {
		input[i] = float32(math.Sin(2.0 * math.Pi * freq * float64(i) / float64(fromRate)))
	}

	r := NewResampler(fromRate, toRate)
	output := r.Resample(input)

	// Verify output length
	expectedLen := int(float64(inputLen) * float64(toRate) / float64(fromRate))
	if len(output) != expectedLen {
		t.Errorf("Output length = %d, want %d", len(output), expectedLen)
	}

	// Verify output is a valid sine wave (not clipping, has oscillations)
	oscillations := 0
	for i := 1; i < len(output); i++ {
		if math.Abs(float64(output[i])) > 1.5 {
			t.Errorf("Sample %d clipping: %v", i, output[i])
		}
		// Count zero crossings (sign changes)
		if (output[i-1] < 0 && output[i] >= 0) || (output[i-1] >= 0 && output[i] < 0) {
			oscillations++
		}
	}

	// Should have multiple oscillations for a 440Hz wave over 50ms
	if oscillations < 10 {
		t.Errorf("Expected multiple oscillations, got %d", oscillations)
	}
}
