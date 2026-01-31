//! Audio I/O module for capturing and playing back audio samples.
//!
//! This module provides cross-platform audio capture and playback using cpal,
//! with high-quality resampling support via rubato.

mod capture;
mod playback;
pub mod resampler;
pub mod util;

pub use capture::Capturer;
pub use playback::Player;
