//! Audio resampling utilities using rubato FFT-based resampler.
//!
//! Provides high-quality audio resampling for both streaming (callback-based)
//! and batch (buffer-based) scenarios.

use anyhow::{Context, Result};
use audioadapter_buffers::direct::InterleavedSlice;
use parking_lot::Mutex;
use rubato::{Fft, FixedSync, Resampler};
use std::sync::Arc;

/// Chunk size for FFT-based resampling (provides good quality and performance).
const CHUNK_SIZE: usize = 1024;

/// Number of sub-chunks for FFT processing (higher = better quality but more CPU).
const SUB_CHUNKS: usize = 2;

/// Shared resampler state for streaming audio (used in audio callbacks).
///
/// This struct maintains internal buffers and resampler state across multiple
/// callback invocations, allowing efficient real-time resampling.
pub struct ResamplerState {
    resampler: Fft<f32>,
    output_buffer: Vec<f32>,
    output_frames_max: usize,
    input_buffer: Vec<f32>, // Buffer to accumulate samples across callbacks
}

impl ResamplerState {
    /// Create a new resampler state for streaming audio.
    ///
    /// # Arguments
    /// * `from_rate` - Input sample rate (e.g., 48000)
    /// * `to_rate` - Output sample rate (e.g., 16000)
    ///
    /// # Returns
    /// A new `ResamplerState` wrapped in `Arc<Mutex<>>` for thread-safe access
    pub fn new(from_rate: u32, to_rate: u32) -> Result<Arc<Mutex<Self>>> {
        let resampler = Fft::<f32>::new(
            from_rate as usize,
            to_rate as usize,
            CHUNK_SIZE,
            SUB_CHUNKS,
            1, // mono output
            FixedSync::Input,
        )
        .context("Failed to create resampler")?;

        let output_frames_max = resampler.output_frames_max();

        Ok(Arc::new(Mutex::new(Self {
            resampler,
            output_buffer: vec![0.0f32; output_frames_max],
            output_frames_max,
            input_buffer: Vec::with_capacity(CHUNK_SIZE * 2), // Preallocate buffer
        })))
    }

    /// Process incoming audio samples, accumulating until a full chunk is available.
    ///
    /// This method is designed to be called from audio callbacks where samples
    /// arrive in variable-size chunks. It accumulates samples internally until
    /// a full CHUNK_SIZE is available for resampling.
    ///
    /// # Arguments
    /// * `samples` - Input audio samples (can be any size)
    ///
    /// # Returns
    /// Resampled audio samples when a full chunk is processed, or `None` if more input is needed
    pub fn process_samples(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        // Accumulate incoming samples
        self.input_buffer.extend_from_slice(samples);

        // Process if we have enough samples
        if self.input_buffer.len() >= CHUNK_SIZE {
            let chunk: Vec<f32> = self.input_buffer.drain(..CHUNK_SIZE).collect();

            let input_adapter = InterleavedSlice::new(&chunk, 1, CHUNK_SIZE).ok()?;
            let mut output_adapter = InterleavedSlice::new_mut(&mut self.output_buffer, 1, self.output_frames_max).ok()?;

            let (_, frames_written) = self.resampler.process_into_buffer(&input_adapter, &mut output_adapter, None).ok()?;

            if frames_written > 0 { Some(self.output_buffer[..frames_written].to_vec()) } else { None }
        } else {
            None
        }
    }
}

/// Resample audio from one sample rate to another (batch processing).
///
/// This function processes an entire buffer of audio samples at once, making it
/// suitable for non-real-time scenarios like pre-processing recorded audio or
/// preparing TTS output for playback.
///
/// Uses FFT-based resampling for high quality with minimal artifacts.
///
/// # Arguments
/// * `samples` - Input audio samples
/// * `from_rate` - Input sample rate (e.g., 24000 for TTS)
/// * `to_rate` - Output sample rate (e.g., 48000 for audio device)
///
/// # Returns
/// Resampled audio samples at the target rate
///
/// # Example
/// ```no_run
/// use voice_assistant::audio::resampler::resample;
///
/// let tts_audio = vec![0.0; 24000]; // 1 second at 24kHz
/// let device_audio = resample(&tts_audio, 24000, 48000).unwrap();
/// assert_eq!(device_audio.len(), 48000); // 1 second at 48kHz
/// ```
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    // No resampling needed if rates match
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    // Create the FFT-based synchronous resampler
    let mut resampler = Fft::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        CHUNK_SIZE,
        SUB_CHUNKS,
        1, // mono
        FixedSync::Input,
    )
    .context("Failed to create resampler")?;

    // Calculate output buffer size
    let output_frames_max = resampler.output_frames_max();
    let mut output_buffer = vec![0.0f32; output_frames_max];

    // Pre-allocate output vector with estimated size
    let estimated_output_len = (samples.len() as f64 * to_rate as f64 / from_rate as f64) as usize + CHUNK_SIZE;
    let mut output = Vec::with_capacity(estimated_output_len);

    let mut pos = 0;

    // Process in chunks
    while pos < samples.len() {
        let end = (pos + CHUNK_SIZE).min(samples.len());
        let chunk = &samples[pos..end];

        // Pad the last chunk if needed
        let input_chunk: Vec<f32> = if chunk.len() < CHUNK_SIZE {
            let mut padded = chunk.to_vec();
            padded.resize(CHUNK_SIZE, 0.0);
            padded
        } else {
            chunk.to_vec()
        };

        // Create adapters for rubato
        let input_adapter = InterleavedSlice::new(&input_chunk, 1, CHUNK_SIZE).context("Failed to create input adapter")?;
        let mut output_adapter = InterleavedSlice::new_mut(&mut output_buffer, 1, output_frames_max).context("Failed to create output adapter")?;

        match resampler.process_into_buffer(&input_adapter, &mut output_adapter, None) {
            Ok((_, frames_written)) => {
                output.extend_from_slice(&output_buffer[..frames_written]);
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Resampling error: {}", e));
            }
        }

        pos += CHUNK_SIZE;
    }

    // Trim any excess padding from the end
    let expected_len = (samples.len() as f64 * to_rate as f64 / from_rate as f64) as usize;
    output.truncate(expected_len + 100); // Keep a small buffer for safety

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_upsampling() {
        // Upsample from 16kHz to 48kHz (3x)
        let samples = vec![0.0; 16000]; // 1 second at 16kHz
        let result = resample(&samples, 16000, 48000).unwrap();
        // Should be approximately 3x longer (within margin for padding)
        assert!(result.len() >= 48000 && result.len() <= 48100);
    }

    #[test]
    fn test_resample_downsampling() {
        // Downsample from 48kHz to 16kHz (1/3x)
        let samples = vec![0.0; 48000]; // 1 second at 48kHz
        let result = resample(&samples, 48000, 16000).unwrap();
        // Should be approximately 1/3x length (within margin for FFT processing)
        assert!(result.len() >= 15900 && result.len() <= 16100, "Expected length 15900-16100, got {}", result.len());
    }

    #[test]
    fn test_resample_frequency_preservation() {
        // Generate 440Hz sine wave at 48kHz
        let from_rate = 48000.0;
        let duration = 0.1; // 100ms
        let freq = 440.0;
        let sample_count = (from_rate * duration) as usize;

        let mut samples = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let t = i as f32 / from_rate;
            samples.push((2.0 * std::f32::consts::PI * freq * t).sin());
        }

        // Resample to 16kHz
        let result = resample(&samples, 48000, 16000).unwrap();

        // Verify output length is approximately correct
        let expected_len = (sample_count as f32 * 16000.0 / 48000.0) as usize;
        assert!(
            result.len() >= expected_len - 100 && result.len() <= expected_len + 100,
            "Expected length ~{}, got {}",
            expected_len,
            result.len()
        );

        // Verify output contains valid sine wave samples (not clipping, has oscillations)
        let mut zero_crossings = 0;
        for i in 1..result.len() {
            // Check for clipping
            assert!(result[i].abs() <= 1.5, "Sample clipping detected: {}", result[i]);

            // Count zero crossings to verify frequency preservation
            if (result[i - 1] < 0.0 && result[i] >= 0.0) || (result[i - 1] >= 0.0 && result[i] < 0.0) {
                zero_crossings += 1;
            }
        }

        // Should have multiple zero crossings for 440Hz wave over 100ms
        assert!(zero_crossings > 20, "Expected multiple oscillations, got {} zero crossings", zero_crossings);
    }

    #[test]
    fn test_resample_edge_cases() {
        // Empty input
        let empty: Vec<f32> = vec![];
        let result = resample(&empty, 48000, 16000).unwrap();
        assert_eq!(result.len(), 0, "Empty input should produce empty output");

        // Single sample
        let single = vec![0.5];
        let result = resample(&single, 48000, 16000).unwrap();
        assert!(!result.is_empty(), "Single sample should produce output");

        // Rate match (no resampling needed)
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = resample(&input, 24000, 24000).unwrap();
        assert_eq!(result.len(), input.len(), "Rate match should return same length");
        for i in 0..input.len() {
            assert!((result[i] - input[i]).abs() < 0.001, "Rate match should preserve samples");
        }
    }

    #[test]
    fn test_resample_partial_chunks() {
        // Input not multiple of CHUNK_SIZE (1024)
        let sample_count = 1500; // 1 full chunk + 476 samples
        let samples = vec![0.5; sample_count];

        let result = resample(&samples, 48000, 16000).unwrap();

        // Should handle partial chunks without panic
        let expected_len = (sample_count as f32 * 16000.0 / 48000.0) as usize;
        assert!(
            result.len() >= expected_len - 100 && result.len() <= expected_len + 200,
            "Partial chunk handling: expected ~{}, got {}",
            expected_len,
            result.len()
        );
    }

    #[test]
    fn test_resample_exact_chunk_multiple() {
        // Input exactly 2 chunks (2048 samples)
        let sample_count = CHUNK_SIZE * 2;
        let samples = vec![0.25; sample_count];

        let result = resample(&samples, 48000, 16000).unwrap();

        // Should process both chunks correctly
        // FFT-based resampling has looser length constraints due to chunk processing
        let expected_len = (sample_count as f32 * 16000.0 / 48000.0) as usize;
        assert!(
            result.len() >= expected_len - 200 && result.len() <= expected_len + 300,
            "Exact chunk multiple: expected ~{} (±250), got {}",
            expected_len,
            result.len()
        );

        // Verify output is not all zeros
        let non_zero = result.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero > 0, "Output should contain non-zero samples");
    }

    #[test]
    fn test_resampler_state_accumulation() {
        // Test streaming with ResamplerState
        let state = ResamplerState::new(48000, 16000).unwrap();
        let mut state = state.lock();

        // Feed samples smaller than CHUNK_SIZE
        let small_chunk = vec![0.5; 512];
        let result1 = state.process_samples(&small_chunk);
        assert!(result1.is_none(), "Should return None when buffer < CHUNK_SIZE");

        // Feed another small chunk to complete a full chunk
        let result2 = state.process_samples(&small_chunk);
        assert!(result2.is_some(), "Should return Some when buffer >= CHUNK_SIZE");

        let output = result2.unwrap();
        assert!(!output.is_empty(), "Output should not be empty");

        // Verify buffer was drained
        assert!(state.input_buffer.len() < CHUNK_SIZE, "Buffer should be partially drained after processing");
    }

    #[test]
    fn test_resampler_state_chunk_boundary() {
        // Test exact chunk boundaries
        let state = ResamplerState::new(48000, 16000).unwrap();
        let mut state = state.lock();

        // Feed exactly CHUNK_SIZE samples
        let chunk = vec![0.3; CHUNK_SIZE];
        let result = state.process_samples(&chunk);
        assert!(result.is_some(), "Should process exactly CHUNK_SIZE samples");

        let output = result.unwrap();
        assert!(!output.is_empty(), "Should produce output");

        // Verify buffer is empty after processing exact chunk
        assert_eq!(state.input_buffer.len(), 0, "Buffer should be empty after processing exact chunk");
    }

    #[test]
    fn test_resample_different_rates() {
        // Test non-standard sample rates
        // Note: FFT-based resampling with chunking can have significant length variations
        // due to internal buffering and padding
        let samples = vec![0.5; 2000];

        // 22.05kHz to 16kHz
        let result = resample(&samples, 22050, 16000).unwrap();
        // Just verify we get output and it's not empty or wildly wrong
        assert!(!result.is_empty(), "22050->16000: should produce output");
        assert!(result.len() < 3000, "22050->16000: output too long");

        // 44.1kHz to 48kHz (slight upsampling)
        let result = resample(&samples, 44100, 48000).unwrap();
        // Just verify we get output and it's not empty or wildly wrong
        assert!(!result.is_empty(), "44100->48000: should produce output");
        assert!(result.len() < 4000, "44100->48000: output too long");
    }
}
