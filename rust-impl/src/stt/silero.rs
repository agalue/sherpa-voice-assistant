//! Silero VAD detector — voice activity detection.
//!
//! [`SileroVad`] implements [`super::VoiceDetector`] and detects speech in the audio
//! stream, delivering completed audio segments via an async channel. Pair with
//! [`crate::stt::WhisperRecognizer`] to form a complete STT pipeline.
//!
//! [`SileroModelProvider`] manages the single `silero_vad.onnx` model file.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use parking_lot::Mutex;
// Alias to avoid collision with our public struct name.
use sherpa_rs::silero_vad::{SileroVad as SherpaVad, SileroVadConfig};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use super::VoiceDetector;
use crate::config::AppConfig;

/// Minimum speech duration in seconds to emit a segment.
const MIN_SPEECH_DURATION: f32 = 0.1;

/// Maximum speech duration in seconds (prevents runaway segments).
const MAX_SPEECH_DURATION: f32 = 30.0;

/// VAD window size in samples (512 = 32 ms at 16 kHz).
const VAD_WINDOW_SIZE: i32 = 512;

/// Audio buffer depth in seconds.
const VAD_BUFFER_SIZE_SECONDS: f32 = 60.0;

/// VAD state; separated so the audio-callback lock covers only fast VAD operations.
struct VadState {
    vad: SherpaVad,
    was_speaking: bool,
    speech_start: Option<Instant>,
}

/// Silero VAD implementation of [`VoiceDetector`].
///
/// Feeds audio into the Silero VAD model and delivers completed speech segments
/// via an async channel. Designed for event-driven use — no polling required.
///
/// # Lock discipline
///
/// The inner mutex is held only during VAD calls (<10 ms). The channel send is
/// performed **without** holding the lock to avoid blocking the audio callback thread.
pub struct SileroVad {
    vad_state: Arc<Mutex<VadState>>,
    segment_tx: mpsc::Sender<Vec<f32>>,
    /// Flag that is `true` while the user is speaking (used for playback interruption).
    pub speaking: Arc<AtomicBool>,
}

impl SileroVad {
    /// Create a new Silero VAD detector.
    ///
    /// # Returns
    /// A `(SileroVad, segment_receiver)` tuple. Pass the receiver to
    /// [`crate::stt::spawn_transcription_task`].
    ///
    /// # Errors
    /// Returns an error if the VAD model cannot be initialised.
    pub fn new(config: &AppConfig) -> Result<(Self, mpsc::Receiver<Vec<f32>>)> {
        let provider = config.effective_stt_provider();

        let vad_model = config.model_dir.join("silero_vad.onnx");

        let vad_config = SileroVadConfig {
            model: vad_model.to_string_lossy().to_string(),
            threshold: config.vad_threshold,
            sample_rate: config.sample_rate,
            min_silence_duration: config.vad_silence_duration,
            min_speech_duration: MIN_SPEECH_DURATION,
            max_speech_duration: MAX_SPEECH_DURATION,
            window_size: VAD_WINDOW_SIZE,
            provider: Some(provider.as_sherpa_provider().to_string()),
            num_threads: Some(config.vad_threads.try_into().unwrap_or(1)),
            debug: config.verbose,
        };

        let vad = SherpaVad::new(vad_config, VAD_BUFFER_SIZE_SECONDS).map_err(|e| anyhow::anyhow!("Failed to initialize Silero VAD: {}", e))?;

        info!("VAD initialized successfully");

        let (segment_tx, segment_rx) = mpsc::channel(5);

        let detector = Self {
            vad_state: Arc::new(Mutex::new(VadState { vad, was_speaking: false, speech_start: None })),
            segment_tx,
            speaking: Arc::new(AtomicBool::new(false)),
        };

        Ok((detector, segment_rx))
    }

    /// Get a clone of the speaking flag for external use before wrapping in a trait object.
    pub fn speaking_flag(&self) -> Arc<AtomicBool> {
        self.speaking.clone()
    }
}

impl VoiceDetector for SileroVad {
    /// Feed raw PCM samples to the VAD.
    ///
    /// **Never blocks.** Completed segments are sent via channel after the VAD
    /// lock is released, keeping the audio callback thread free.
    fn accept_waveform(&self, samples: &[f32]) {
        let mut state = self.vad_state.lock();
        state.vad.accept_waveform(samples.to_vec());

        let is_speech = state.vad.is_speech();
        self.speaking.store(is_speech, Ordering::SeqCst);

        // Track speech transitions for logging.
        if is_speech && !state.was_speaking {
            state.speech_start = Some(Instant::now());
            info!("🎤 Speech started");
        } else if !is_speech
            && state.was_speaking
            && let Some(start) = state.speech_start.take()
        {
            info!("🎤 Speech ended ({:.1}s)", start.elapsed().as_secs_f32());
        }
        state.was_speaking = is_speech;

        // EVENT-DRIVEN: send completed segment without holding the VAD lock.
        if !state.vad.is_empty() {
            let segment = state.vad.front();
            state.vad.pop();

            if !segment.samples.is_empty() {
                debug!("Segment completed: {} samples", segment.samples.len());
                let samples_to_send = segment.samples.clone();
                drop(state); // Release lock before channel send.

                if let Err(e) = self.segment_tx.try_send(samples_to_send) {
                    warn!("Failed to send segment (channel full): {}", e);
                }
            }
        }
    }

    #[allow(dead_code)]
    fn speaking_flag(&self) -> Arc<AtomicBool> {
        self.speaking.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelProvider
// ─────────────────────────────────────────────────────────────────────────────

/// Manages the `silero_vad.onnx` model file.
pub struct SileroModelProvider;

impl super::ModelProvider for SileroModelProvider {
    fn name(&self) -> &'static str {
        "Silero VAD"
    }

    /// # Errors
    /// Returns an error if the download fails.
    fn ensure_models(&self, model_dir: &std::path::Path, force: bool) -> Result<()> {
        let vad_dest = model_dir.join("silero_vad.onnx");
        crate::setup::download_file("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx", &vad_dest, force)
    }

    fn verify_models(&self, model_dir: &std::path::Path) -> Vec<PathBuf> {
        let vad_path = model_dir.join("silero_vad.onnx");
        if !vad_path.exists() { vec![vad_path] } else { vec![] }
    }
}
