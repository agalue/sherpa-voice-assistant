//! Voice Assistant - A real-time voice assistant using local LLMs.
//!
//! This application provides a voice interface to interact with local language models
//! using speech recognition (configurable via `--stt-backend`; default: Whisper),
//! voice activity detection (Silero VAD), text-to-speech (configurable via
//! `--tts-backend`; default: Kokoro), and LLM inference (Ollama via RIG).
//!
//! Run with `--setup` to download required model files.
//! Run with `--setup --force` to re-download all models.

mod audio;
mod config;
mod llm;
mod models;
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
use stt::{ModelProvider as SttModelProvider, SileroModelProvider, SileroVad, VoiceDetector};

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

/// Run the setup mode: download all required model files and exit.
fn run_setup(config: &AppConfig) -> Result<()> {
    info!("🔧 Voice Assistant Setup — downloading model files");
    info!("   Model directory: {}", config.model_dir.display());
    if config.force {
        info!("   Mode: force re-download");
    } else {
        info!("   Mode: skip existing files");
    }

    let silero_provider = SileroModelProvider;
    let stt_provider = stt::new_model_provider(config)?;
    let tts_provider = tts::new_model_provider(config)?;

    info!("📥 [VAD] {} — downloading models…", silero_provider.name());
    silero_provider.ensure_models(&config.model_dir, config.force)?;

    info!("📥 [STT] {} — downloading models…", stt_provider.name());
    stt_provider.ensure_models(&config.model_dir, config.force)?;

    info!("📥 [TTS] {} — downloading models…", tts_provider.name());
    tts_provider.ensure_models(&config.model_dir, config.force)?;

    // Final verification
    info!("🔍 Verifying model files…");
    let mut all_missing: Vec<std::path::PathBuf> = Vec::new();
    all_missing.extend(silero_provider.verify_models(&config.model_dir));
    all_missing.extend(stt_provider.verify_models(&config.model_dir));
    all_missing.extend(tts_provider.verify_models(&config.model_dir));

    if !all_missing.is_empty() {
        error!("❌ Some model files are still missing:");
        for f in &all_missing {
            error!("   - {}", f.display());
        }
        anyhow::bail!("{} model file(s) missing after setup", all_missing.len());
    }

    info!("✅ All model files are present. Run the assistant without --setup to start.");
    Ok(())
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

    // Handle informational flags first (no model loading required).
    let tts_provider = tts::new_model_provider(&config)?;
    if config.list_voices {
        tts_provider.print_voices();
        std::process::exit(0);
    }
    if let Some(ref voice_name) = config.voice_info {
        match tts_provider.print_voice_info(voice_name) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    }

    // --setup: download all required model files then exit.
    if config.setup {
        return run_setup(&config);
    }

    // Create and initialize components.
    // SileroVad handles VAD; WhisperRecognizer handles transcription.
    // They are wired together via the VoiceDetector and Transcriber traits.
    let (silero, segment_rx) = SileroVad::new(&config)?;
    let user_speaking = silero.speaking_flag();

    // Wrap behind trait objects so the pipeline is model-agnostic.
    let voice_detector: Arc<dyn VoiceDetector> = Arc::new(silero);
    let transcriber = stt::new_transcriber(&config)?;

    // Create TTS synthesizer via factory (implements SpeechSynthesizer).
    let synthesizer = tts::new_synthesizer(&config)?;
    let synth_sample_rate = synthesizer.sample_rate();

    let llm_client = LlmClient::new(&config)?;

    // Wrap synthesizer behind trait object inside mutex.
    let synthesizer: Arc<Mutex<Box<dyn tts::SpeechSynthesizer>>> = Arc::new(Mutex::new(synthesizer));
    let llm_client = Arc::new(tokio::sync::Mutex::new(llm_client));

    // Create capturer with callback that feeds VoiceDetector directly.
    // The callback runs on the audio thread — must be non-blocking.
    let voice_detector_for_audio = voice_detector.clone();
    let mut capturer = Capturer::new(config.sample_rate, move |samples: &[f32]| {
        voice_detector_for_audio.accept_waveform(samples);
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

    // Spawn transcription task (event-driven, accepts Arc<dyn Transcriber>)
    let transcription_handle = stt::spawn_transcription_task(transcript_tx_clone, segment_rx, transcriber, shutdown.clone());

    // Spawn LLM task
    let llm_handle = llm::spawn_llm_task(transcript_rx, response_tx_clone, llm_client, player.clone(), user_speaking.clone(), shutdown.clone());

    // Spawn TTS task (accepts Arc<Mutex<dyn SpeechSynthesizer>>)
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

    let graceful_timeout = tokio::time::Duration::from_millis(500);

    tokio::select! {
        _ = llm_handle => {
            debug!("LLM task finished gracefully");
        }
        _ = tokio::time::sleep(graceful_timeout) => {
            debug!("LLM task didn't finish in time, aborting");
        }
    }

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
