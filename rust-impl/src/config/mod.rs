//! Configuration module for the voice assistant.
//!
//! Provides CLI argument parsing and configuration management.

#[allow(clippy::module_inception)]
mod config;
mod voices;

pub use config::{AppConfig, InterruptMode};
