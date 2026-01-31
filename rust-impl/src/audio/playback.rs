//! Audio playback module using cpal.
//!
//! Plays audio samples through the default output device with interrupt support.
//! Includes automatic resampling when the device sample rate differs from input.
//! Uses lock-free ring buffer to avoid mutex contention in audio callback.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex as StdMutex};
use std::time::Duration;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use parking_lot::Mutex;
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use tracing::{debug, info, warn};

use super::resampler::resample;
use super::util::{find_best_config, get_device_name};

/// Size of the playback ring buffer in samples (~11 seconds at 48kHz)
/// Increased from 131072 to prevent overflow with long TTS responses
const PLAYBACK_RING_SIZE: usize = 524288;

/// Audio player that outputs samples to the speaker.
/// Uses a lock-free ring buffer for the audio callback to avoid priority inversion.
/// The ring buffer prevents mutex contention in the high-priority audio thread.
pub struct Player {
    /// Kept alive to maintain the audio stream
    _stream: Stream,
    /// Sample rate of the audio device
    device_sample_rate: u32,
    /// Sample rate of the input audio (e.g., TTS output)
    input_sample_rate: u32,
    /// Ring buffer producer for queuing samples (mutex protects multi-threaded queue access)
    producer: Mutex<ringbuf::HeapProd<f32>>,
    /// Number of samples currently queued
    queued_samples: Arc<AtomicUsize>,
    /// Flag to indicate playback should be interrupted
    interrupt: Arc<AtomicBool>,
    /// External interrupt flag (e.g., when user starts speaking)
    external_interrupt: Option<Arc<AtomicBool>>,
    /// Flag to indicate playback is active
    playing: Arc<AtomicBool>,
    /// Mutex and Condvar for efficient waiting on playback completion
    playing_mutex: Arc<StdMutex<()>>,
    playback_complete: Arc<Condvar>,
}

impl Player {
    /// Create a new audio player.
    ///
    /// # Arguments
    /// * `sample_rate` - The sample rate for playback (typically 24000 for TTS)
    /// * `external_interrupt` - Optional external interrupt flag
    ///
    /// # Returns
    /// A new `Player` instance ready for playback.
    ///
    /// # Errors
    /// Returns an error if:
    /// - No output device is available
    /// - Failed to get supported output configurations
    /// - Failed to build output stream
    pub fn new(sample_rate: u32, external_interrupt: Option<Arc<AtomicBool>>) -> Result<Self> {
        let host = cpal::default_host();
        let device = host.default_output_device().context("No output device available")?;

        info!("Using output device: {}", get_device_name(&device));

        // Query device's preferred sample rate for better compatibility
        let device_sample_rate = match device.default_output_config() {
            Ok(default_config) => {
                let rate = default_config.sample_rate();
                info!("Using device's default sample rate: {} Hz", rate);
                rate
            }
            Err(_) => {
                let supported_configs = device.supported_output_configs().context("Failed to get supported output configs")?;
                let config = find_best_config(supported_configs, 48000)?;
                let rate = config.sample_rate();
                info!("Using fallback sample rate: {} Hz", rate);
                rate
            }
        };

        let supported_configs = device.supported_output_configs().context("Failed to get supported output configs")?;
        let config = find_best_config(supported_configs, device_sample_rate)?;

        if device_sample_rate != sample_rate {
            info!("Device sample rate {} Hz differs from input {} Hz - resampling will be applied", device_sample_rate, sample_rate);
        }

        debug!("Audio playback config: {} Hz, {} channels, {:?}", device_sample_rate, config.channels(), config.sample_format());

        // Create lock-free ring buffer for audio callback
        let ring = HeapRb::<f32>::new(PLAYBACK_RING_SIZE);
        let (producer, mut consumer) = ring.split();

        let interrupt = Arc::new(AtomicBool::new(false));
        let playing = Arc::new(AtomicBool::new(false));
        let queued_samples = Arc::new(AtomicUsize::new(0));
        let playing_mutex = Arc::new(StdMutex::new(()));
        let playback_complete = Arc::new(Condvar::new());

        let interrupt_clone = interrupt.clone();
        let external_interrupt_clone = external_interrupt.clone();
        let playing_clone = playing.clone();
        let queued_samples_clone = queued_samples.clone();
        let playing_mutex_clone = playing_mutex.clone();
        let playback_complete_clone = playback_complete.clone();

        let channels = config.channels() as usize;
        let stream_config: StreamConfig = config.config();

        let err_fn = |err| {
            tracing::error!("Audio playback error: {}", err);
        };

        // Build F32 output stream with lock-free callback
        let stream = device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // Check for interrupts (lock-free)
                let interrupted = interrupt_clone.load(Ordering::Relaxed) || external_interrupt_clone.as_ref().is_some_and(|e| e.load(Ordering::Relaxed));

                let mut samples_read = 0;

                for frame in data.chunks_mut(channels) {
                    let sample = if !interrupted {
                        // Lock-free pop from ring buffer
                        consumer.try_pop().unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    if sample != 0.0 {
                        samples_read += 1;
                    }

                    // Duplicate mono sample to all channels
                    for channel in frame.iter_mut() {
                        *channel = sample;
                    }
                }

                // Update queued count
                if samples_read > 0 {
                    queued_samples_clone.fetch_sub(samples_read, Ordering::Relaxed);
                }

                // Check if playback is done
                if consumer.is_empty() || interrupted {
                    playing_clone.store(false, Ordering::SeqCst);
                    // Signal condition variable
                    let _guard = playing_mutex_clone.lock().unwrap();
                    playback_complete_clone.notify_all();
                }
            },
            err_fn,
            None,
        )?;

        stream.play().context("Failed to start playback stream")?;

        info!("Audio playback configured: input {} Hz -> device {} Hz (lock-free)", sample_rate, device_sample_rate);

        Ok(Self {
            _stream: stream,
            device_sample_rate,
            input_sample_rate: sample_rate,
            producer: Mutex::new(producer),
            queued_samples,
            interrupt,
            external_interrupt,
            playing,
            playing_mutex,
            playback_complete,
        })
    }

    /// Play audio samples.
    ///
    /// This method blocks until all samples are played or playback is interrupted.
    ///
    /// # Arguments
    /// * `samples` - The audio samples to play (mono f32 at input_sample_rate)
    ///
    /// # Returns
    /// `true` if playback completed, `false` if interrupted.
    pub fn play(&self, samples: &[f32]) -> bool {
        if samples.is_empty() {
            return true;
        }

        // Reset interrupt flag
        self.interrupt.store(false, Ordering::SeqCst);

        // Resample if needed
        let samples_to_play = if self.device_sample_rate != self.input_sample_rate {
            match resample(samples, self.input_sample_rate, self.device_sample_rate) {
                Ok(resampled) => {
                    debug!(
                        "Resampled {} -> {} samples ({} Hz -> {} Hz)",
                        samples.len(),
                        resampled.len(),
                        self.input_sample_rate,
                        self.device_sample_rate
                    );
                    resampled
                }
                Err(e) => {
                    tracing::error!("Resampling failed: {}, playing without resampling", e);
                    samples.to_vec()
                }
            }
        } else {
            samples.to_vec()
        };

        // Queue samples to ring buffer
        {
            let mut producer = self.producer.lock();
            let written = producer.push_slice(&samples_to_play);
            if written < samples_to_play.len() {
                warn!("Playback buffer overflow, dropped {} samples", samples_to_play.len() - written);
            }
            self.queued_samples.fetch_add(written, Ordering::Relaxed);
        }

        // Mark as playing
        self.playing.store(true, Ordering::SeqCst);

        debug!("Playing {} samples at {} Hz", samples_to_play.len(), self.device_sample_rate);

        // Wait for playback to complete or be interrupted
        let duration_secs = samples_to_play.len() as f64 / self.device_sample_rate as f64;
        let timeout = Duration::from_secs_f64(duration_secs + 1.0);
        let deadline = std::time::Instant::now() + timeout;

        // Use condition variable to wait efficiently instead of busy-polling
        while self.playing.load(Ordering::Relaxed) {
            if self.interrupt.load(Ordering::Relaxed) {
                debug!("Playback interrupted (internal)");
                self.clear();
                return false;
            }

            if let Some(ref ext) = self.external_interrupt
                && ext.load(Ordering::Relaxed)
            {
                debug!("Playback interrupted (external)");
                self.clear();
                return false;
            }

            if std::time::Instant::now() > deadline {
                warn!("Playback timeout exceeded");
                self.clear();
                return false;
            }

            // Wait on condition variable with short timeout to check interrupts
            let guard = self.playing_mutex.lock().unwrap();
            let (_guard, _timeout_result) = self.playback_complete.wait_timeout(guard, Duration::from_millis(50)).unwrap();

            // Check if we should continue waiting
            if !self.playing.load(Ordering::Relaxed) {
                break;
            }
        }

        debug!("Playback completed");
        true
    }

    /// Interrupt current playback.
    pub fn interrupt(&self) {
        self.interrupt.store(true, Ordering::SeqCst);
    }

    /// Stop playback and drain the buffer.
    ///
    /// This uses the interrupt flag to make the audio callback output silence,
    /// effectively draining the buffer without explicit clearing.
    pub fn clear(&self) {
        // Set interrupt to make callback output silence while buffer drains
        let _producer = self.producer.lock();
        self.interrupt.store(true, Ordering::SeqCst);
        self.queued_samples.store(0, Ordering::SeqCst);
        self.playing.store(false, Ordering::SeqCst);
        drop(_producer);

        // Brief sleep to let the callback drain any remaining buffered samples
        std::thread::sleep(Duration::from_millis(20));
        self.interrupt.store(false, Ordering::SeqCst);
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.interrupt.store(true, Ordering::SeqCst);
        self.playing.store(false, Ordering::SeqCst);
    }
}
