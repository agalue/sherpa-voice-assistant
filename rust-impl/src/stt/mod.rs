//! Speech-to-text module using sherpa-rs.
//!
//! Provides voice activity detection (VAD) and Whisper-based speech recognition.

mod recognizer;

pub use recognizer::Recognizer;
