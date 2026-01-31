//! Shared audio utilities for capture and playback.

use anyhow::Result;
use cpal::traits::DeviceTrait;
use cpal::{Device, SampleFormat, SupportedStreamConfig, SupportedStreamConfigRange};

/// Get a human-readable device name.
///
/// # Arguments
/// * `device` - The audio device
///
/// # Returns
/// Device name string, or "Unknown" if the name cannot be retrieved.
pub fn get_device_name(device: &Device) -> String {
    device.description().ok().map(|desc| desc.name().to_string()).unwrap_or_else(|| "Unknown".to_string())
}

/// Find the best matching audio configuration.
///
/// Searches for a configuration that:
/// 1. Supports mono or stereo (max 2 channels)
/// 2. Uses F32 sample format (universally supported on modern hardware)
/// 3. Matches the target sample rate, or uses the closest available rate
///
/// # Arguments
/// * `configs` - Iterator of supported stream configurations
/// * `target_sample_rate` - Desired sample rate (e.g., 16000 for STT, 24000 for TTS)
///
/// # Returns
/// The best matching `SupportedStreamConfig`, or an error if no suitable config found.
pub fn find_best_config(configs: impl Iterator<Item = SupportedStreamConfigRange>, target_sample_rate: u32) -> Result<SupportedStreamConfig> {
    let mut f32_configs: Vec<SupportedStreamConfigRange> = Vec::new();

    for config in configs {
        // Only consider mono or stereo
        if config.channels() > 2 {
            continue;
        }

        // Only accept F32 format (universally supported on Mac, Linux, Jetson)
        if config.sample_format() == SampleFormat::F32 {
            f32_configs.push(config);
        }
    }

    if f32_configs.is_empty() {
        anyhow::bail!("No F32 audio configuration found - this is unexpected on modern hardware");
    }

    // Find config that supports target sample rate, or use first available
    for config in &f32_configs {
        let min_rate = config.min_sample_rate();
        let max_rate = config.max_sample_rate();

        if target_sample_rate >= min_rate && target_sample_rate <= max_rate {
            return Ok((*config).with_sample_rate(target_sample_rate));
        }
    }

    // Use first config with closest sample rate
    let config = &f32_configs[0];
    let rate = if target_sample_rate < config.min_sample_rate() {
        config.min_sample_rate()
    } else {
        config.max_sample_rate()
    };
    Ok((*config).with_sample_rate(rate))
}

/// Convert f32 samples to mono f32 samples.
///
/// Handles both mono and stereo input:
/// - Mono: Returns a copy of the input
/// - Stereo: Mixes channels by averaging
///
/// # Arguments
/// * `data` - Raw f32 samples (interleaved for stereo)
/// * `channels` - Number of channels (1 for mono, 2 for stereo)
///
/// # Returns
/// Vector of mono f32 samples
pub fn convert_to_mono_f32_f32(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        data.to_vec()
    } else {
        // Mix stereo to mono by averaging channels
        data.chunks(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereo_to_mono_f32() {
        let data = vec![0.5f32, 1.0, -0.5, -1.0];
        let result = convert_to_mono_f32_f32(&data, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0.75); // (0.5 + 1.0) / 2
        assert_eq!(result[1], -0.75); // (-0.5 + -1.0) / 2
    }
}
