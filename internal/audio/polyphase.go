// Package audio provides polyphase resampling for anti-aliasing.
package audio

import "math"

// PolyphaseResampler implements a polyphase filter for high-quality downsampling.
// Prevents aliasing artifacts when downsampling (e.g., 48kHz -> 16kHz for STT).
// Uses a 64-tap sinc filter with Hamming window for optimal quality/performance.
type PolyphaseResampler struct {
	fromRate   int       // Source sample rate
	toRate     int       // Target sample rate
	ratio      float64   // Conversion ratio
	filterLen  int       // FIR filter length (64 taps)
	filter     []float32 // Low-pass filter coefficients
	history    []float32 // Sample history for filter
	lastSample float32   // Last sample for continuity
}

// NewPolyphaseResampler creates a new polyphase resampler with anti-aliasing filter.
// Use this for downsampling (e.g., 48kHz -> 16kHz). For upsampling, linear interpolation is sufficient.
//
// The resampler uses a 64-tap sinc filter with Hamming window to prevent aliasing.
// Filter cutoff is set to the output Nyquist frequency for downsampling.
func NewPolyphaseResampler(fromRate, toRate int) *PolyphaseResampler {
	ratio := float64(toRate) / float64(fromRate)

	// Filter length: use 64 taps for good quality/performance balance
	filterLen := 64

	// Design a low-pass sinc filter with Hamming window
	// Cutoff frequency is at the lower of the two Nyquist frequencies
	cutoff := 0.5
	if ratio < 1.0 {
		// Downsampling: filter at output Nyquist
		cutoff = ratio * 0.5
	}

	filter := make([]float32, filterLen)
	for i := 0; i < filterLen; i++ {
		n := float64(i) - float64(filterLen-1)/2.0
		if n == 0 {
			filter[i] = float32(2.0 * cutoff)
		} else {
			// Sinc function
			sinc := math.Sin(2.0*math.Pi*cutoff*n) / (math.Pi * n)
			// Hamming window
			window := 0.54 - 0.46*math.Cos(2.0*math.Pi*float64(i)/float64(filterLen-1))
			filter[i] = float32(sinc * window)
		}
	}

	// Normalize filter coefficients
	sum := float32(0.0)
	for _, f := range filter {
		sum += f
	}
	for i := range filter {
		filter[i] /= sum
	}

	return &PolyphaseResampler{
		fromRate:   fromRate,
		toRate:     toRate,
		ratio:      ratio,
		filterLen:  filterLen,
		filter:     filter,
		history:    make([]float32, filterLen),
		lastSample: 0.0,
	}
}

// Resample converts audio samples using polyphase filtering.
func (r *PolyphaseResampler) Resample(input []float32) []float32 {
	if r.ratio == 1.0 {
		return input
	}

	inputLen := len(input)
	if inputLen == 0 {
		return input
	}

	// For upsampling, use simple linear interpolation (good enough for TTS)
	if r.ratio > 1.0 {
		return r.upsample(input)
	}

	// For downsampling, use polyphase filtering to prevent aliasing
	return r.downsample(input)
}

// upsample uses linear interpolation (simple and fast for upsampling)
func (r *PolyphaseResampler) upsample(input []float32) []float32 {
	inputLen := len(input)
	outputLen := int(float64(inputLen) * r.ratio)
	output := make([]float32, outputLen)

	for i := 0; i < outputLen; i++ {
		srcPos := float64(i) / r.ratio
		srcIdx := int(srcPos)
		frac := float32(srcPos - float64(srcIdx))

		sample1 := r.lastSample
		if srcIdx < inputLen {
			sample1 = input[srcIdx]
		}

		sample2 := sample1
		if srcIdx+1 < inputLen {
			sample2 = input[srcIdx+1]
		} else if srcIdx < inputLen {
			sample2 = input[inputLen-1]
		}

		output[i] = sample1 + (sample2-sample1)*frac
	}

	if inputLen > 0 {
		r.lastSample = input[inputLen-1]
	}

	return output
}

// downsample uses polyphase filtering to prevent aliasing
func (r *PolyphaseResampler) downsample(input []float32) []float32 {
	inputLen := len(input)
	outputLen := int(float64(inputLen) * r.ratio)
	output := make([]float32, outputLen)

	// Combine history with new input
	combined := append(r.history, input...)

	for i := 0; i < outputLen; i++ {
		// Calculate source position in combined buffer
		srcPos := float64(i) / r.ratio
		srcIdx := int(srcPos) + len(r.history)

		// Apply FIR filter centered at srcIdx
		sample := float32(0.0)
		for j := 0; j < r.filterLen; j++ {
			idx := srcIdx - r.filterLen/2 + j
			if idx >= 0 && idx < len(combined) {
				sample += combined[idx] * r.filter[j]
			}
		}
		output[i] = sample
	}

	// Update history with last filterLen samples from input
	if inputLen >= r.filterLen {
		copy(r.history, input[inputLen-r.filterLen:])
	} else {
		// Shift existing history and append new samples
		shift := r.filterLen - inputLen
		copy(r.history, r.history[inputLen:])
		copy(r.history[shift:], input)
	}

	return output
}

// ResamplePolyphase is a convenience function for one-time downsampling.
// Use this for STT input (48kHz -> 16kHz). For upsampling, use ResampleInPlace.
func ResamplePolyphase(input []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate {
		return input
	}

	// Only use polyphase for downsampling
	if toRate < fromRate {
		r := NewPolyphaseResampler(fromRate, toRate)
		return r.Resample(input)
	}

	// For upsampling, use simple linear interpolation
	return ResampleInPlace(input, fromRate, toRate)
}
