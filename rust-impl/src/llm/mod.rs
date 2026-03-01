//! LLM client module for interacting with language models.
//!
//! Uses RIG with Ollama provider for local LLM inference and supports
//! agentic tool calling for weather and web search capabilities.

mod client;
mod weather;
mod websearch;

pub use client::LlmClient;
pub use weather::WeatherTool;
pub use websearch::WebSearchTool;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info};

use crate::audio::Player;

/// Spawn the LLM processing task for generating responses from user transcriptions.
///
/// Reads transcripts from `transcript_rx`, obtains a response from the LLM via
/// [`LlmClient::chat`], and forwards the response to `response_tx`. Any active
/// playback is interrupted when a new transcript arrives. The task exits when
/// `transcript_rx` is closed or `shutdown` is set to `true`.
///
/// # Arguments
/// * `transcript_rx` - Channel to receive user transcriptions
/// * `response_tx` - Channel to send LLM responses for TTS
/// * `llm_client` - Shared, async-mutex-guarded LLM client
/// * `player` - Audio player; interrupted when a new user utterance arrives
/// * `user_speaking` - Flag that is `true` while the user is speaking
/// * `shutdown` - Shared shutdown flag; task exits when set to `true`
///
/// # Returns
/// A `JoinHandle` for the spawned async task
pub fn spawn_llm_task(
    mut transcript_rx: mpsc::Receiver<String>,
    response_tx: mpsc::Sender<String>,
    llm_client: Arc<tokio::sync::Mutex<LlmClient>>,
    player: Arc<Player>,
    user_speaking: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while !shutdown.load(Ordering::Relaxed) {
            tokio::select! {
                Some(transcript) = transcript_rx.recv() => {
                    // Discard stale transcripts if the user is already speaking again.
                    if user_speaking.load(Ordering::Relaxed) {
                        let mut discarded = 1;
                        while transcript_rx.try_recv().is_ok() {
                            discarded += 1;
                        }
                        info!("\u{1f5d1}\u{fe0f}  Discarded {} queued transcript(s) due to new speech", discarded);
                        continue;
                    }

                    info!("\u{1f9e0} Processing: \"{}\"", transcript);

                    // Interrupt any audio currently playing.
                    player.interrupt();

                    info!("\u{1f914} Analyzing request (may use tools)...");

                    // RIG handles the agentic tool-call loop transparently.
                    let result = {
                        let mut client = llm_client.lock().await;
                        client.chat(&transcript).await
                    };

                    match result {
                        Ok(response) => {
                            info!("\u{1f916} Assistant: {}", response);
                            if let Err(e) = response_tx.send(response).await {
                                debug!("Failed to send response: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("\u{274c} LLM error: {}", e);
                            if response_tx
                                .send("I'm sorry, I encountered an error.".to_string())
                                .await
                                .is_err()
                            {
                                debug!("Response channel closed");
                                break;
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                    // Periodic wakeup to re-check shutdown flag.
                }
            }
        }
    })
}
