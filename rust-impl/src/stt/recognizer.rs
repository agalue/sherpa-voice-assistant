//! Speech recognizer combining VAD and Whisper STT.
//!
//! Uses Silero VAD for voice activity detection and Whisper for transcription.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use parking_lot::Mutex;
use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::config::AppConfig;

/// Minimum speech duration in seconds to be considered valid.
const MIN_SPEECH_DURATION: f32 = 0.1;

/// Maximum speech duration in seconds (prevent runaway segments).
const MAX_SPEECH_DURATION: f32 = 30.0;

/// VAD window size in samples (512 samples = 32ms at 16kHz).
const VAD_WINDOW_SIZE: i32 = 512;

/// Buffer size in seconds for VAD (how much audio to accumulate).
const VAD_BUFFER_SIZE_SECONDS: f32 = 60.0;

/// State for VAD that needs to be accessed from audio callback.
/// Separated from Whisper to avoid lock contention (VAD is fast, Whisper is slow).
struct VadState {
    vad: SileroVad,                // Voice activity detector
    was_speaking: bool,            // Previous speaking state for edge detection
    speech_start: Option<Instant>, // Timestamp when speech started
}

/// Speech recognizer combining VAD and Whisper.
/// Uses separate mutexes for VAD (fast) and Whisper (slow) to prevent audio glitches.
pub struct Recognizer {
    vad_state: Arc<Mutex<VadState>>,      // VAD state (fast access, <10ms)
    whisper: Mutex<WhisperRecognizer>,    // Whisper recognizer (slow, 100-500ms)
    segment_tx: mpsc::Sender<Vec<f32>>,   // Channel for completed speech segments
    pub(crate) sample_rate: u32,          // Audio sample rate (16kHz)
    pub(crate) wake_word: Option<String>, // Optional wake word for activation
    /// Flag to signal user is speaking (for playback interruption)
    pub speaking: Arc<AtomicBool>,
}

impl Recognizer {
    /// Create a new speech recognizer with event-driven segment delivery.
    ///
    /// # Arguments
    /// * `config` - Application configuration
    ///
    /// # Returns
    /// A tuple of (Recognizer, segment receiver channel)
    ///
    /// # Errors
    /// Returns an error if:
    /// - Failed to initialize Silero VAD
    /// - Failed to initialize Whisper recognizer
    /// - Model files are missing or invalid
    pub fn new(config: &AppConfig) -> Result<(Self, mpsc::Receiver<Vec<f32>>)> {
        let sample_rate = config.sample_rate;
        let provider = config.effective_stt_provider();

        info!("Initializing speech recognizer with {} provider", provider);

        // Initialize Silero VAD
        let vad_config = SileroVadConfig {
            model: config.vad_model_path().to_string_lossy().to_string(),
            threshold: config.vad_threshold,
            sample_rate,
            min_silence_duration: config.vad_silence_duration,
            min_speech_duration: MIN_SPEECH_DURATION,
            max_speech_duration: MAX_SPEECH_DURATION,
            window_size: VAD_WINDOW_SIZE,
            provider: Some(provider.as_sherpa_provider().to_string()),
            num_threads: Some(config.vad_threads.try_into().unwrap_or(1)),
            debug: config.verbose,
        };

        let vad = SileroVad::new(vad_config, VAD_BUFFER_SIZE_SECONDS).map_err(|e| anyhow::anyhow!("Failed to initialize Silero VAD: {}", e))?;

        info!("VAD initialized successfully");

        // Initialize Whisper recognizer
        let encoder_path = config.whisper_encoder_path().to_string_lossy().to_string();
        let decoder_path = config.whisper_decoder_path().to_string_lossy().to_string();
        let tokens_path = config.whisper_tokens_path().to_string_lossy().to_string();

        info!("Whisper encoder path: {}", encoder_path);
        info!("Whisper decoder path: {}", decoder_path);
        info!("Whisper tokens path: {}", tokens_path);
        info!("Whisper provider: {}", provider.as_sherpa_provider());

        let stt_language = config.effective_stt_language().to_string();
        info!("STT language: {}", if stt_language.is_empty() { "auto" } else { &stt_language });

        let whisper_config = WhisperConfig {
            encoder: encoder_path,
            decoder: decoder_path,
            tokens: tokens_path,
            language: stt_language,
            provider: Some(provider.as_sherpa_provider().to_string()),
            num_threads: Some(config.stt_threads.try_into().unwrap_or(2)),
            debug: config.verbose,
            ..Default::default()
        };

        info!("Creating WhisperRecognizer...");

        let whisper = WhisperRecognizer::new(whisper_config).map_err(|e| anyhow::anyhow!("Failed to initialize Whisper: {}", e))?;

        info!("Whisper recognizer initialized successfully");

        // Create channel for event-driven segment delivery (buffered to handle bursts)
        let (segment_tx, segment_rx) = mpsc::channel(5);

        let recognizer = Self {
            vad_state: Arc::new(Mutex::new(VadState { vad, was_speaking: false, speech_start: None })),
            whisper: Mutex::new(whisper),
            segment_tx,
            sample_rate,
            wake_word: config.wake_word.clone(),
            speaking: Arc::new(AtomicBool::new(false)),
        };

        Ok((recognizer, segment_rx))
    }

    /// Feed audio samples to VAD and send completed segments immediately.
    /// This is event-driven: segments are delivered as soon as VAD completes them,
    /// eliminating the 50-100ms polling delay of the previous approach.
    /// Thread-safe: can be called concurrently with transcription.
    pub fn vad_accept_waveform(&self, samples: &[f32]) {
        let mut state = self.vad_state.lock();
        state.vad.accept_waveform(samples.to_vec());

        // Update speaking state for playback interruption
        let is_speech = state.vad.is_speech();
        self.speaking.store(is_speech, Ordering::SeqCst);

        // Track speech state transitions for logging
        if is_speech && !state.was_speaking {
            state.speech_start = Some(Instant::now());
            info!("🎤 Speech started");
        } else if !is_speech
            && state.was_speaking
            && let Some(start) = state.speech_start.take()
        {
            let duration = start.elapsed().as_secs_f32();
            info!("🎤 Speech ended ({:.1}s)", duration);
        }
        state.was_speaking = is_speech;

        // EVENT-DRIVEN: Send completed segments immediately (no polling delay)
        if !state.vad.is_empty() {
            let segment = state.vad.front();
            state.vad.pop();

            if !segment.samples.is_empty() {
                debug!("Segment completed: {} samples", segment.samples.len());

                // Clone samples before dropping lock
                let samples_to_send = segment.samples.clone();
                drop(state); // Release VAD lock ASAP

                // Non-blocking send (try_send to avoid blocking audio thread)
                if let Err(e) = self.segment_tx.try_send(samples_to_send) {
                    warn!("Failed to send segment (channel full): {}", e);
                }
            }
        }
    }

    /// Transcribe audio samples (call this without holding the recognizer lock).
    pub fn transcribe_segment(&self, samples: &[f32]) -> Option<String> {
        if samples.is_empty() {
            debug!("Empty speech segment");
            return None;
        }

        debug!("Transcribing {} samples", samples.len());

        // Run Whisper transcription
        let mut whisper = self.whisper.lock();
        let transcription_result = whisper.transcribe(self.sample_rate, samples);
        drop(whisper); // Release lock as soon as transcription is done

        let text = transcription_result.text.trim().to_string();

        if text.is_empty() {
            debug!("Empty transcription result");
            return None;
        }

        // Check wake word if configured
        if let Some(ref wake_word) = self.wake_word {
            if !text.to_lowercase().contains(&wake_word.to_lowercase()) {
                debug!("Wake word '{}' not detected in '{}', ignoring", wake_word, text);
                return None;
            }
            // Remove wake word from transcription
            let cleaned = text
                .to_lowercase()
                .replace(&wake_word.to_lowercase(), "")
                .trim_start_matches(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
                .trim()
                .to_string();

            // If only wake word was spoken (no query), prompt for hello
            if cleaned.is_empty() {
                info!("🗣️ Wake word '{}' detected", wake_word);
                return Some("Hello".to_string());
            }

            info!("🗣️ You (wake word detected): {}", cleaned);
            return Some(cleaned);
        }

        info!("🗣️ You: {}", text);
        Some(text)
    }

    /// Get a clone of the speaking flag for external use.
    pub fn speaking_flag(&self) -> Arc<AtomicBool> {
        self.speaking.clone()
    }
}
