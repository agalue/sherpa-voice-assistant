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
use tracing::{debug, error, info};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::time::LocalTime;

use audio::{Capturer, Player};
use config::AppConfig;
use llm::LlmClient;
use stt::Recognizer;
use tts::Synthesizer;

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
    let transcription_handle = stt::spawn_transcription_task(transcript_tx_clone, segment_rx, recognizer.clone(), shutdown.clone());

    // Spawn LLM task
    let llm_handle = llm::spawn_llm_task(transcript_rx, response_tx_clone, llm_client, player.clone(), user_speaking.clone(), shutdown.clone());

    // Spawn TTS task
    let tts_handle = tts::spawn_tts_task(
        response_rx,
        tts::TtsTaskConfig {
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
