//! Text-to-speech module using sherpa-rs.
//!
//! Provides speech synthesis using Kokoro models.

mod synthesizer;

pub use synthesizer::{Synthesizer, split_sentences};

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

use crate::audio::Player;
use crate::config::InterruptMode;

/// Configuration bundle for [`spawn_tts_task`].
pub struct TtsTaskConfig {
    /// TTS synthesizer (CPU-bound; protected by a blocking mutex).
    pub synthesizer: Arc<Mutex<Synthesizer>>,
    /// Audio output device.
    pub player: Arc<Player>,
    /// Microphone active flag; set to `false` to pause capture in `Wait` mode.
    pub capturer_running: Arc<AtomicBool>,
    /// Flag that is `true` while the user is speaking (used for interruption).
    pub user_speaking: Arc<AtomicBool>,
    /// Controls how user speech interrupts TTS playback.
    pub interrupt_mode: InterruptMode,
    /// Milliseconds to wait after playback ends before resuming the microphone.
    pub post_delay_ms: u64,
    /// Shared shutdown flag; task exits when set to `true`.
    pub shutdown: Arc<AtomicBool>,
}

/// Spawn the TTS playback task for speech synthesis and audio output.
///
/// Receives complete LLM responses from `response_rx`, splits them into sentences,
/// and pipelines synthesis with playback: sentence *N+1* is synthesized (via
/// [`tokio::task::spawn_blocking`]) concurrently with playback of sentence *N*, hiding
/// the 100–900 ms synthesis latency behind actual playback time.
///
/// Microphone pause/resume and interruption behaviour are governed by
/// [`TtsTaskConfig::interrupt_mode`]. The task exits when `response_rx` is closed or
/// `config.shutdown` is set to `true`.
///
/// # Arguments
/// * `response_rx` - Channel to receive LLM responses
/// * `config` - TTS task configuration; see [`TtsTaskConfig`]
///
/// # Returns
/// A `JoinHandle` for the spawned async task
pub fn spawn_tts_task(
    mut response_rx: mpsc::Receiver<String>,
    config: TtsTaskConfig,
) -> JoinHandle<()> {
    let TtsTaskConfig {
        synthesizer,
        player,
        capturer_running,
        user_speaking,
        interrupt_mode,
        post_delay_ms,
        shutdown,
    } = config;

    tokio::spawn(async move {
        while !shutdown.load(Ordering::Relaxed) {
            // Wait for the next response with periodic shutdown checks.
            let response = tokio::select! {
                Some(resp) = response_rx.recv() => resp,
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => continue,
            };

            // Skip the whole response if the user is already speaking.
            if interrupt_mode == InterruptMode::Always && user_speaking.load(Ordering::Relaxed) {
                drain_channel(&mut response_rx, "queued LLM response(s) due to interruption");
                continue;
            }

            // In 'wait' mode, pause the microphone for the duration of playback.
            if interrupt_mode == InterruptMode::Wait {
                capturer_running.store(false, Ordering::SeqCst);
                debug!("Microphone paused for playback");
            }

            // Pipeline synthesis and playback concurrently for lower latency.
            // Synthesis of sentence N+1 overlaps with playback of sentence N.
            let sentences = split_sentences(&response);
            if sentences.is_empty() {
                warn!("No sentences to synthesize");
                if interrupt_mode == InterruptMode::Wait {
                    tokio::time::sleep(tokio::time::Duration::from_millis(post_delay_ms)).await;
                    capturer_running.store(true, Ordering::SeqCst);
                }
                continue;
            }

            let mut was_interrupted = false;
            let total_sentences = sentences.len();

            // spawn_blocking is required because ONNX inference is synchronous/CPU-bound
            // and must not block the tokio async runtime.
            let (tx, mut rx) = mpsc::channel::<Vec<f32>>(1); // 1-slot: prefetch next sentence
            // Set by the synthesis task when it exits early due to an interrupt before
            // sending any audio, so the recv loop's None exit still triggers the drain.
            let synth_interrupted = Arc::new(AtomicBool::new(false));

            let synth_task = {
                let synthesizer = Arc::clone(&synthesizer);
                let shutdown = Arc::clone(&shutdown);
                let user_speaking = Arc::clone(&user_speaking);
                let synth_interrupted_flag = Arc::clone(&synth_interrupted);
                tokio::task::spawn_blocking(move || {
                    for (i, sentence) in sentences.iter().enumerate() {
                        if sentence.trim().is_empty() {
                            continue;
                        }
                        if interrupt_mode == InterruptMode::Always
                            && user_speaking.load(Ordering::Relaxed)
                        {
                            synth_interrupted_flag.store(true, Ordering::Relaxed);
                            break;
                        }
                        if shutdown.load(Ordering::Relaxed) {
                            break;
                        }
                        let samples = {
                            let mut synth = synthesizer.lock();
                            match synth.synthesize_sentence(sentence) {
                                Ok(s) => s,
                                Err(e) => {
                                    error!(
                                        "\u{274c} TTS error for sentence {}/{}: {}",
                                        i + 1,
                                        total_sentences,
                                        e
                                    );
                                    continue;
                                }
                            }
                        };
                        // blocking_send returns Err when the receiver is dropped (interrupted).
                        if tx.blocking_send(samples).is_err() {
                            break;
                        }
                    }
                })
            };

            // Play sentences as they arrive from the synthesis task.
            let mut played = 0usize;
            while let Some(samples) = rx.recv().await {
                if samples.is_empty() {
                    continue;
                }
                played += 1;
                info!(
                    "\u{1f50a} Playing sentence {}/{} ({} samples)",
                    played,
                    total_sentences,
                    samples.len()
                );

                if interrupt_mode == InterruptMode::Always
                    && user_speaking.load(Ordering::Relaxed)
                {
                    info!("\u{23f8}\u{fe0f}  Synthesis interrupted by speech");
                    player.interrupt();
                    was_interrupted = true;
                    break;
                }

                if shutdown.load(Ordering::Relaxed) {
                    info!("\u{1f6d1} Shutdown requested during playback, stopping audio");
                    player.interrupt();
                    break;
                }

                if !player.play(&samples) {
                    if interrupt_mode == InterruptMode::Always {
                        info!("\u{23f8}\u{fe0f}  Playback interrupted");
                        was_interrupted = true;
                    }
                    break;
                }

                if interrupt_mode == InterruptMode::Always
                    && user_speaking.load(Ordering::Relaxed)
                {
                    info!("\u{23f8}\u{fe0f}  Playback interrupted by speech");
                    player.interrupt();
                    was_interrupted = true;
                    break;
                }
            }

            // Propagate an interruption that occurred entirely inside the synthesis task
            // (before any audio reached this loop), so the drain below still runs.
            if !was_interrupted && synth_interrupted.load(Ordering::Relaxed) {
                was_interrupted = true;
            }

            // Drop receiver to unblock the synthesis task if it is still running.
            drop(rx);
            if was_interrupted && interrupt_mode == InterruptMode::Always {
                // Abort the join handle so we don't wait for full synthesis latency.
                // Note: this does not forcibly preempt CPU-bound work already in progress.
                synth_task.abort();
            } else {
                let _ = synth_task.await;
            }

            // Resume microphone in 'wait' mode.
            if interrupt_mode == InterruptMode::Wait {
                tokio::time::sleep(tokio::time::Duration::from_millis(post_delay_ms)).await;
                capturer_running.store(true, Ordering::SeqCst);
                debug!("Microphone resumed after playback");
            }

            // Drain any remaining queued responses if interrupted.
            if was_interrupted && interrupt_mode == InterruptMode::Always {
                drain_channel(&mut response_rx, "queued TTS response(s)");
            }
        }
    })
}

/// Drain all pending messages from `rx`, logging how many were discarded.
fn drain_channel(rx: &mut mpsc::Receiver<String>, label: &str) {
    let mut discarded = 1;
    while rx.try_recv().is_ok() {
        discarded += 1;
    }
    if discarded > 0 {
        info!("\u{1f5d1}\u{fe0f}  Discarded {} {}", discarded, label);
    }
}
