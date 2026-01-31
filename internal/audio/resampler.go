// Package audio provides audio resampling functionality.
package audio

// Resampler provides simple linear interpolation audio resampling.
// This is lightweight and sufficient for voice applications where
// audiophile quality is not required. Uses linear interpolation for
// a good balance between performance and quality.
type Resampler struct {
	fromRate   float64 // Source sample rate
	toRate     float64 // Target sample rate
	ratio      float64 // Conversion ratio (toRate/fromRate)
	lastSample float32 // Last sample for continuity between chunks
}

// NewResampler creates a new audio resampler.
//
// Parameters:
//   - fromRate: Source sample rate in Hz
//   - toRate: Target sample rate in Hz
//
// Returns a Resampler instance configured for the specified conversion.
func NewResampler(fromRate, toRate int) *Resampler {
	ratio := float64(toRate) / float64(fromRate)
	return &Resampler{
		fromRate:   float64(fromRate),
		toRate:     float64(toRate),
		ratio:      ratio,
		lastSample: 0.0,
	}
}

// Resample converts audio samples from one sample rate to another using linear interpolation.
// This is a simple, efficient algorithm suitable for voice applications.
//
// For higher quality resampling, consider using FFT-based methods (like Rust's rubato),
// but linear interpolation provides a good balance of performance and quality for voice.
func (r *Resampler) Resample(input []float32) []float32 {
	if r.ratio == 1.0 {
		return input
	}

	inputLen := len(input)
	if inputLen == 0 {
		return input
	}

	// Calculate output length
	outputLen := int(float64(inputLen) * r.ratio)
	output := make([]float32, outputLen)

	// Linear interpolation resampling
	for i := 0; i < outputLen; i++ {
		// Calculate source position in input array
		srcPos := float64(i) / r.ratio
		srcIdx := int(srcPos)
		frac := float32(srcPos - float64(srcIdx))

		// Get samples for interpolation
		sample1 := r.lastSample
		if srcIdx < inputLen {
			sample1 = input[srcIdx]
		}

		sample2 := sample1
		if srcIdx+1 < inputLen {
			sample2 = input[srcIdx+1]
		} else if srcIdx < inputLen {
			// Use last sample if at the end
			sample2 = input[inputLen-1]
		}

		// Linear interpolation: output = sample1 + (sample2 - sample1) * frac
		output[i] = sample1 + (sample2-sample1)*frac
	}

	// Store last sample for next iteration (helps with continuity)
	if inputLen > 0 {
		r.lastSample = input[inputLen-1]
	}

	return output
}

// ResampleInPlace is a convenience function that creates a resampler and processes samples.
// Use this for one-time resampling. For streaming audio, create a Resampler instance and reuse it.
func ResampleInPlace(input []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate {
		return input
	}
	r := NewResampler(fromRate, toRate)
	return r.Resample(input)
}
