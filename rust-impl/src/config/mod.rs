//! Configuration module for the voice assistant.
//!
//! Provides CLI argument parsing and configuration management.
//! Model-specific paths, voices, and validation live in the STT/TTS implementations.

#[allow(clippy::module_inception)]
mod config;

pub use config::{AppConfig, InterruptMode};
