//! Speech-to-text module using sherpa-rs.
//!
//! Provides voice activity detection (VAD) and Whisper-based speech recognition.

mod recognizer;

pub use recognizer::Recognizer;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::debug;

/// Spawn the transcription task for event-driven STT processing.
///
/// Receives completed speech segments from VAD and transcribes them using Whisper.
/// The event-driven approach (receiving from a channel rather than polling) eliminates
/// the 50–100 ms polling delay present in timer-based designs.
///
/// # Arguments
/// * `transcript_tx` - Channel to send completed transcription strings
/// * `segment_rx` - Channel to receive raw speech segments from VAD
/// * `recognizer` - Shared speech recognizer instance
/// * `shutdown` - Shared shutdown flag; task exits when set to `true`
///
/// # Returns
/// A `JoinHandle` for the spawned async task
pub fn spawn_transcription_task(
    transcript_tx: mpsc::Sender<String>,
    mut segment_rx: mpsc::Receiver<Vec<f32>>,
    recognizer: Arc<Recognizer>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while !shutdown.load(Ordering::Relaxed) {
            // Use a short timeout so the shutdown flag is checked regularly.
            match tokio::time::timeout(
                tokio::time::Duration::from_millis(100),
                segment_rx.recv(),
            )
            .await
            {
                Ok(Some(samples)) => {
                    if let Some(text) = recognizer.transcribe_segment(&samples)
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
