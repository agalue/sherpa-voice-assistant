//! Text-to-speech module using sherpa-rs.
//!
//! Provides speech synthesis using Kokoro models.

mod synthesizer;

pub use synthesizer::{Synthesizer, split_sentences};
