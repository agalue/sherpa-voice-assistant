//! Speech-to-text module.
//!
//! Defines [`VoiceDetector`] and [`Transcriber`] traits that decouple the rest of
//! the voice assistant from any specific STT implementation, and provides:
//! - [`SileroVad`] — voice activity detection via the Silero VAD model
//! - [`WhisperRecognizer`] — speech transcription via OpenAI Whisper
//!
//! To add a new STT backend:
//! 1. Implement [`VoiceDetector`] or [`Transcriber`] in a new file.
//! 2. Implement [`ModelProvider`] so the binary can download/verify model files.
//! 3. Register the new backend in [`new_transcriber`] and [`new_model_provider`].

mod silero;
mod whisper;

pub use silero::{SileroModelProvider, SileroVad};
pub use whisper::{WhisperModelProvider, WhisperRecognizer};

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::debug;

// ─────────────────────────────────────────────────────────────────────────────
// Core traits
// ─────────────────────────────────────────────────────────────────────────────

/// Handles voice activity detection (VAD).
///
/// Implementations must be callable from the real-time audio callback thread and
/// therefore **must never block** (no I/O, no locks that might contend with slow
/// operations).
pub trait VoiceDetector: Send + Sync {
    /// Feed raw PCM samples from the microphone into the VAD.
    ///
    /// **Never block in this method.** It runs on the audio callback thread.
    fn accept_waveform(&self, samples: &[f32]);

    /// Returns the shared flag that is `true` while the user is speaking.
    /// Used by the LLM/TTS pipeline to interrupt playback on the audio thread.
    /// Callers obtain this before the recognizer is wrapped in a trait object.
    #[allow(dead_code)]
    fn speaking_flag(&self) -> Arc<AtomicBool>;
}

/// Converts raw PCM audio segments into text.
///
/// Implementations are expected to be slow (100–500 ms per call) and **must not**
/// be called from the audio callback thread.
pub trait Transcriber: Send + Sync {
    /// Transcribe a completed speech segment.
    ///
    /// Returns `None` when the segment contains no recognisable speech (noise,
    /// silence, or wake-word filter mismatch).
    fn transcribe_segment(&self, samples: &[f32]) -> Option<String>;
}

/// Manages the lifecycle of model files required by an STT backend.
///
/// Every STT implementation must expose a type that implements this trait so that
/// the binary can download, verify, and report on its model files without an
/// external shell script.
pub trait ModelProvider: Send + Sync {
    /// Download any model files that are absent from `model_dir`.
    ///
    /// If `force` is `true`, all files are re-downloaded even if they already exist.
    fn ensure_models(&self, model_dir: &Path, force: bool) -> Result<()>;

    /// Return the paths of model files that are absent from `model_dir`.
    ///
    /// An empty `Vec` means all required files are present.
    fn verify_models(&self, model_dir: &Path) -> Vec<std::path::PathBuf>;

    /// Human-readable name for this STT implementation (e.g. `"Whisper"`).
    fn name(&self) -> &'static str;
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory functions
// ─────────────────────────────────────────────────────────────────────────────

use crate::config::AppConfig;

/// Create the [`Transcriber`] for the configured STT backend.
///
/// Each backend interprets `config.stt_model` in its own way (e.g. Whisper
/// strips the `whisper-` prefix to derive the model size). To add a new
/// backend, add a match arm here and in [`new_model_provider`].
///
/// # Errors
/// Returns an error if the backend name is unknown or initialization fails.
pub fn new_transcriber(config: &AppConfig) -> Result<Arc<dyn Transcriber>> {
    match config.stt_backend.to_lowercase().as_str() {
        "whisper" => Ok(Arc::new(WhisperRecognizer::new(config)?)),
        other => anyhow::bail!("unknown STT backend {:?} (available: whisper)", other),
    }
}

/// Return the [`ModelProvider`] for the configured STT backend.
///
/// # Errors
/// Returns an error if the backend name is unknown.
pub fn new_model_provider(config: &AppConfig) -> Result<Box<dyn ModelProvider>> {
    match config.stt_backend.to_lowercase().as_str() {
        "whisper" => {
            let model_size = config.stt_model.strip_prefix("whisper-").unwrap_or(&config.stt_model).to_string();
            Ok(Box::new(WhisperModelProvider { model_size }))
        }
        other => anyhow::bail!("unknown STT backend {:?} (available: whisper)", other),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline task
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn the transcription task for event-driven STT processing.
///
/// Receives completed speech segments from VAD and transcribes them using the
/// provided [`Transcriber`] implementation. The event-driven approach (receiving
/// from a channel rather than polling) eliminates the 50–100 ms polling delay
/// present in timer-based designs.
///
/// # Arguments
/// * `transcript_tx` - Channel to send completed transcription strings
/// * `segment_rx` - Channel to receive raw speech segments from VAD
/// * `transcriber` - Shared [`Transcriber`] implementation
/// * `shutdown` - Shared shutdown flag; task exits when set to `true`
///
/// # Returns
/// A `JoinHandle` for the spawned async task
pub fn spawn_transcription_task(
    transcript_tx: mpsc::Sender<String>,
    mut segment_rx: mpsc::Receiver<Vec<f32>>,
    transcriber: Arc<dyn Transcriber>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while !shutdown.load(Ordering::Relaxed) {
            // Use a short timeout so the shutdown flag is checked regularly.
            match tokio::time::timeout(tokio::time::Duration::from_millis(100), segment_rx.recv()).await {
                Ok(Some(samples)) => {
                    if let Some(text) = transcriber.transcribe_segment(&samples)
                        && let Err(e) = transcript_tx.send(text).await
                    {
                        debug!("Failed to send transcript: {}", e);
                        break;
                    }
                }
                Ok(None) => {
                    debug!("Segment channel closed");
                    break;
                }
                Err(_) => continue, // timeout – recheck shutdown flag
            }
        }
    })
}
