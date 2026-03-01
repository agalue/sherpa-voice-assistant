//! Voice Assistant - A real-time voice assistant using local LLMs.
//!
//! This application provides a voice interface to interact with local language models
//! using speech recognition (Whisper), voice activity detection (Silero VAD),
//! text-to-speech (Kokoro), and LLM inference (Ollama via RIG).

mod audio;
mod config;
mod llm;
mod stt;
mod tts;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use parking_lot::Mutex;
use tokio::signal;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::time::LocalTime;

use audio::{Capturer, Player};
use config::{AppConfig, InterruptMode};
use llm::LlmClient;
use stt::Recognizer;
use tts::Synthesizer;

/// Configuration for the TTS playback task.
struct TtsTaskConfig {
    synthesizer: Arc<Mutex<Synthesizer>>, // TTS synthesizer
    player: Arc<Player>,                  // Audio player
    capturer_running: Arc<AtomicBool>,    // Microphone state flag
    user_speaking: Arc<AtomicBool>,       // User speaking flag
    interrupt_mode: InterruptMode,        // Interrupt handling mode
    post_delay_ms: u64,                   // Delay after playback (ms)
    shutdown: Arc<AtomicBool>,            // Shutdown flag
}

/// Spawn the transcription task for event-driven STT processing.
///
/// Receives completed speech segments from VAD and transcribes them using Whisper.
/// This event-driven approach eliminates 50-100ms polling delay.
///
/// # Arguments
/// * `transcript_tx` - Channel to send completed transcriptions
/// * `segment_rx` - Channel to receive speech segments from VAD
/// * `recognizer` - Speech recognizer instance
/// * `shutdown` - Shutdown flag
///
/// # Returns
/// Join handle for the spawned task
fn spawn_transcription_task(
    transcript_tx: mpsc::Sender<String>,
    mut segment_rx: mpsc::Receiver<Vec<f32>>,
    recognizer: Arc<Recognizer>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while !shutdown.load(Ordering::Relaxed) {
            // Use timeout to allow shutdown checks
            match tokio::time::timeout(tokio::time::Duration::from_millis(100), segment_rx.recv()).await {
                Ok(Some(samples)) => {
                    // Transcribe the segment
                    let transcript = recognizer.transcribe_segment(&samples);

                    if let Some(text) = transcript
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
                Err(_) => {
                    // Timeout - continue to check shutdown flag
                    continue;
                }
            }
        }
    })
}

/// Spawn the LLM processing task for generating responses.
///
/// Receives user transcriptions and generates LLM responses.
/// Handles interruption when user starts speaking during LLM processing.
///
/// # Arguments
/// * `transcript_rx` - Channel to receive user transcriptions
/// * `response_tx` - Channel to send LLM responses
/// * `llm_client` - LLM client instance
/// * `player` - Audio player for interrupting playback
/// * `user_speaking` - Flag indicating user is speaking
/// * `shutdown` - Shutdown flag
///
/// # Returns
/// Join handle for the spawned task
fn spawn_llm_task(
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
                    // Check if user started speaking (interrupt pending response)
                    if user_speaking.load(Ordering::Relaxed) {
                        drain_channel(&mut transcript_rx, "queued transcript(s) due to new speech");
                        continue;
                    }

                    info!("🧠 Processing: \"{}\"", transcript);

                    // Interrupt any current playback
                    player.interrupt();

                    // Indicate thinking/tool usage - LLM may invoke tools automatically
                    info!("🤔 Analyzing request (may use tools)...");

                    // Get complete response from LLM (RIG handles tool calling automatically)
                    let result = {
                        let mut client = llm_client.lock().await;
                        client.chat(&transcript).await
                    };

                    match result {
                        Ok(response) => {
                            info!("🤖 Assistant: {}", response);

                            // Send complete response for TTS processing
                            if let Err(e) = response_tx.send(response).await {
                                debug!("Failed to send response: {}", e);
                            }
                        }
                        Err(e) => {
                            error!("❌ LLM error: {}", e);
                            let error_msg = "I'm sorry, I encountered an error.".to_string();
                            if response_tx.send(error_msg).await.is_err() {
                                debug!("Response channel closed");
                                break;
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                    // Check for shutdown
                }
            }
        }
    })
}

/// Spawn the TTS playback task for speech synthesis and audio output.
///
/// Receives LLM responses, synthesizes speech, and plays audio.
/// Handles microphone pause/resume based on interrupt mode.
///
/// # Arguments
/// * `response_rx` - Channel to receive LLM responses
/// * `config` - TTS task configuration
///
/// # Returns
/// Join handle for the spawned task
fn spawn_tts_task(mut response_rx: mpsc::Receiver<String>, config: TtsTaskConfig) -> JoinHandle<()> {
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
            // Wait for next response with periodic shutdown checks
            let response = tokio::select! {
                Some(resp) = response_rx.recv() => resp,
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => continue,
            };

            // Check if user interrupted before synthesis
            if interrupt_mode == InterruptMode::Always && user_speaking.load(Ordering::Relaxed) {
                drain_channel(&mut response_rx, "queued LLM response(s) due to interruption");
                continue;
            }

            // Pause microphone in 'wait' mode
            if interrupt_mode == InterruptMode::Wait {
                capturer_running.store(false, Ordering::SeqCst);
                debug!("Microphone paused for playback");
            }

            // Pipeline synthesis and playback concurrently for lower latency.
            // Synthesis of sentence N+1 overlaps with playback of sentence N.
            let sentences = tts::split_sentences(&response);
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

            // Pipeline: a blocking task synthesizes sentences over a buffered channel
            // while the async loop plays them concurrently, hiding synthesis latency
            // (100–900ms on CPU/Jetson) behind playback time and eliminating the gap
            // between spoken sentences.
            //
            // spawn_blocking is used because ONNX inference is synchronous/CPU-bound
            // and must not block the tokio async runtime.
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<f32>>(1); // 1-slot: prefetch next sentence
            // Set by the synthesis task when it exits early due to an interrupt before
            // sending any audio, so the recv loop's None exit can still trigger the drain.
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
                        if interrupt_mode == InterruptMode::Always && user_speaking.load(Ordering::Relaxed) {
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
                                    error!("❌ TTS error for sentence {}/{}: {}", i + 1, total_sentences, e);
                                    continue;
                                }
                            }
                        };
                        // blocking_send returns Err when receiver is dropped (interrupted)
                        if tx.blocking_send(samples).is_err() {
                            break;
                        }
                    }
                })
            };

            // Play sentences as they arrive from the synthesis task
            let mut played = 0usize;
            while let Some(samples) = rx.recv().await {
                if samples.is_empty() {
                    continue;
                }
                played += 1;
                info!("🔊 Playing sentence {}/{} ({} samples)", played, total_sentences, samples.len());

                // Check interruption before each sentence
                if interrupt_mode == InterruptMode::Always && user_speaking.load(Ordering::Relaxed) {
                    info!("⏸️  Synthesis interrupted by speech");
                    player.interrupt();
                    was_interrupted = true;
                    break;
                }

                if !player.play(&samples) {
                    if interrupt_mode == InterruptMode::Always {
                        info!("⏸️  Playback interrupted");
                        was_interrupted = true;
                    }
                    break;
                }

                if interrupt_mode == InterruptMode::Always && user_speaking.load(Ordering::Relaxed) {
                    info!("⏸️  Playback interrupted by speech");
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

            // Drop receiver to unblock the synthesis task if it is still running
            drop(rx);
            if was_interrupted && interrupt_mode == InterruptMode::Always {
                // Abort synthesis to avoid waiting up to full synthesis latency after interruption
                synth_task.abort();
            } else {
                let _ = synth_task.await;
            }

            // Resume microphone in 'wait' mode
            if interrupt_mode == InterruptMode::Wait {
                tokio::time::sleep(tokio::time::Duration::from_millis(post_delay_ms)).await;
                capturer_running.store(true, Ordering::SeqCst);
                debug!("Microphone resumed after playback");
            }

            // Drain remaining responses if interrupted
            if was_interrupted && interrupt_mode == InterruptMode::Always {
                drain_channel(&mut response_rx, "queued TTS response(s)");
            }
        }
    })
}

/// Drain remaining messages from a channel.
fn drain_channel(rx: &mut mpsc::Receiver<String>, label: &str) {
    let mut discarded = 1;
    while rx.try_recv().is_ok() {
        discarded += 1;
    }
    if discarded > 0 {
        info!("🗑️  Discarded {} {}", discarded, label);
    }
}

/// Wait for shutdown signal (Ctrl+C or SIGTERM).
async fn wait_for_shutdown(shutdown: Arc<AtomicBool>) {
    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("🛑 Received Ctrl+C, shutting down...");
        }
        _ = async {
            #[cfg(unix)]
            {
                let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
                    .expect("Failed to register SIGTERM handler");
                sigterm.recv().await;
            }
            #[cfg(not(unix))]
            {
                std::future::pending::<()>().await;
            }
        } => {
            info!("🛑 Received SIGTERM, shutting down...");
        }
    }

    shutdown.store(true, Ordering::SeqCst);
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let config = AppConfig::from_args();

    // Initialize logging with time-only format (matching Go's log.Ltime)
    // Respect RUST_LOG env var, fallback to verbose flag, default to info
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| if config.verbose { EnvFilter::try_new("debug") } else { EnvFilter::try_new("info") })
        .unwrap();

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_timer(LocalTime::new(time::macros::format_description!("[hour]:[minute]:[second]")))
        .init();

    info!("🎤 Voice Assistant v{}", env!("CARGO_PKG_VERSION"));

    // Validate configuration
    if let Err(e) = config.validate() {
        error!("❌ Configuration error: {}", e);
        error!("Run 'scripts/setup.sh' to download required models.");
        std::process::exit(1);
    }

    // Create and initialize components (event-driven recognizer returns channel)
    let (recognizer, segment_rx) = Recognizer::new(&config)?;
    let user_speaking = recognizer.speaking_flag();

    let synthesizer = Synthesizer::new(&config)?;
    let llm_client = LlmClient::new(&config)?;

    let synth_sample_rate = synthesizer.sample_rate();

    // Wrap components in Arc for shared access
    let recognizer = Arc::new(recognizer);
    let synthesizer = Arc::new(Mutex::new(synthesizer));
    let llm_client = Arc::new(tokio::sync::Mutex::new(llm_client));

    // Create capturer with callback that feeds recognizer directly
    let recognizer_for_audio = recognizer.clone();
    let mut capturer = Capturer::new(config.sample_rate, move |samples: &[f32]| {
        // VAD sends completed segments immediately via channel (event-driven)
        recognizer_for_audio.vad_accept_waveform(samples);
    })?;

    let player = Player::new(synth_sample_rate, Some(user_speaking.clone()))?;
    let player = Arc::new(player);

    let shutdown = Arc::new(AtomicBool::new(false));
    let capturer_running = capturer.running_flag();

    info!("Starting voice assistant...");
    config.log_config();

    // Start audio capture
    capturer.start()?;

    // Create channels for inter-task communication
    let (transcript_tx, transcript_rx) = mpsc::channel::<String>(10);
    let (response_tx, response_rx) = mpsc::channel::<String>(10);

    // Clone senders before moving into tasks (allows proper shutdown via drop)
    let transcript_tx_clone = transcript_tx.clone();
    let response_tx_clone = response_tx.clone();

    // Spawn transcription task (event-driven: receives segments via channel, no polling)
    let transcription_handle = spawn_transcription_task(transcript_tx_clone, segment_rx, recognizer.clone(), shutdown.clone());

    // Spawn LLM task
    let llm_handle = spawn_llm_task(transcript_rx, response_tx_clone, llm_client, player.clone(), user_speaking.clone(), shutdown.clone());

    // Spawn TTS task
    let tts_handle = spawn_tts_task(
        response_rx,
        TtsTaskConfig {
            synthesizer,
            player,
            capturer_running,
            user_speaking,
            interrupt_mode: config.interrupt_mode,
            post_delay_ms: config.post_playback_delay_ms,
            shutdown: shutdown.clone(),
        },
    );

    // Wait for shutdown signal
    wait_for_shutdown(shutdown).await;

    // Stop audio capture first
    capturer.shutdown();

    // Close channels to wake up tasks
    drop(transcript_tx);
    drop(response_tx);

    // Wait for all tasks with timeout
    let graceful_timeout = tokio::time::Duration::from_millis(500);

    tokio::select! {
        _ = transcription_handle => {
            debug!("Transcription task finished gracefully");
        }
        _ = tokio::time::sleep(graceful_timeout) => {
            debug!("Transcription task didn't finish in time");
        }
    }

    // Give tasks a moment to notice the shutdown flag and clean up gracefully
    // before forcefully aborting
    let graceful_timeout = tokio::time::Duration::from_millis(500);

    // Wait for LLM task with timeout, then abort if needed
    tokio::select! {
        _ = llm_handle => {
            debug!("LLM task finished gracefully");
        }
        _ = tokio::time::sleep(graceful_timeout) => {
            debug!("LLM task didn't finish in time, aborting");
        }
    }

    // Wait for TTS task with timeout, then abort if needed
    tokio::select! {
        _ = tts_handle => {
            debug!("TTS task finished gracefully");
        }
        _ = tokio::time::sleep(graceful_timeout) => {
            debug!("TTS task didn't finish in time, aborting");
        }
    }

    info!("✅ Voice assistant stopped");
    Ok(())
}
