//! Audio capture module using cpal.
//!
//! Captures audio from the default input device and sends samples to a channel.
//! Includes automatic resampling when the device sample rate differs from the target.
//! Uses lock-free ring buffer for zero-copy audio transfer from callback.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use tracing::{debug, info, warn};

use super::resampler::ResamplerState;
use super::util::{convert_to_mono_f32_f32, find_best_config, get_device_name};

/// Audio capturer that streams samples from the microphone.
/// Uses a lock-free ring buffer to prevent audio callback blocking.
pub struct Capturer {
    stream: Stream,                                       // cpal audio stream (kept alive)
    running: Arc<AtomicBool>,                             // Pause/resume flag (temporary)
    shutdown: Arc<AtomicBool>,                            // Permanent shutdown flag
    drain_handle: Option<std::thread::JoinHandle<()>>,    // Thread draining ring buffer
    callback_handle: Option<std::thread::JoinHandle<()>>, // Thread calling user callback
    consumer: Option<ringbuf::HeapCons<f32>>,             // Ring buffer consumer
    sender: Option<SyncSender<Vec<f32>>>,                 // Channel sender to callback thread
}

impl Capturer {
    /// Create a new audio capturer.
    ///
    /// # Arguments
    /// * `sample_rate` - The desired sample rate (typically 16000 for STT)
    /// * `callback` - Function to call with captured audio samples
    ///
    /// # Returns
    /// A new `Capturer` instance ready to start capturing.
    ///
    /// # Errors
    /// Returns an error if:
    /// - No input device is available
    /// - Failed to get supported input configurations
    /// - Failed to build input stream
    pub fn new<F>(sample_rate: u32, callback: F) -> Result<Self>
    where
        F: Fn(&[f32]) + Send + 'static,
    {
        // Create bounded channel for backpressure (32 chunks ~= 1 second of audio)
        let (sender, receiver) = mpsc::sync_channel::<Vec<f32>>(32);

        // Spawn thread to consume from channel and call callback
        // This separates channel consumption from VAD lock acquisition
        let callback_handle = std::thread::spawn(move || {
            while let Ok(samples) = receiver.recv() {
                callback(&samples);
            }
            debug!("Audio callback thread exiting");
        });

        let host = cpal::default_host();
        let device = host.default_input_device().context("No input device available")?;

        info!("Using input device: {}", get_device_name(&device));

        // Get supported configs and find best match
        let supported_configs = device.supported_input_configs().context("Failed to get supported input configs")?;

        let config = find_best_config(supported_configs, sample_rate)?;
        let device_sample_rate = config.sample_rate();

        let needs_resampling = device_sample_rate != sample_rate;
        if needs_resampling {
            info!("Device sample rate {} Hz differs from target {} Hz - resampling will be applied", device_sample_rate, sample_rate);
        }

        debug!("Audio capture config: {} Hz, {} channels, {:?}", device_sample_rate, config.channels(), config.sample_format());

        let running = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let running_clone = running.clone();
        let channels = config.channels() as usize;

        let stream_config: StreamConfig = config.config();

        let err_fn = |err| {
            tracing::error!("Audio capture error: {}", err);
        };

        // Create lock-free ring buffer to prevent callback blocking
        // Size: 65536 samples = ~4 seconds at 16kHz
        let ring = HeapRb::<f32>::new(65536);
        let (mut producer, consumer) = ring.split();

        // Create resampler if needed
        let resampler_state = if needs_resampling { Some(ResamplerState::new(device_sample_rate, sample_rate)?) } else { None };

        // Build F32 input stream (guaranteed by find_best_config)
        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if running_clone.load(Ordering::Relaxed) {
                    let samples = convert_to_mono_f32_f32(data, channels);

                    // Process through resampler if needed
                    let final_samples = match &resampler_state {
                        Some(state) => state.lock().process_samples(&samples),
                        None => Some(samples.to_vec()),
                    };

                    // Push to ring buffer (lock-free, non-blocking)
                    if let Some(samples) = final_samples {
                        let written = producer.push_slice(&samples);
                        if written < samples.len() {
                            // Ring buffer full - log warning (safe with relaxed ordering)
                            use std::sync::atomic::{AtomicU64, Ordering};
                            static DROP_COUNT: AtomicU64 = AtomicU64::new(0);
                            let count = DROP_COUNT.fetch_add(1, Ordering::Relaxed);
                            if count.is_multiple_of(100) {
                                tracing::warn!("Ring buffer full, dropped {} audio chunks", count + 1);
                            }
                        }
                    }
                }
            },
            err_fn,
            None,
        )?;

        info!("Audio capture configured: device {} Hz -> output {} Hz", device_sample_rate, sample_rate);

        Ok(Self {
            stream,
            running,
            shutdown,
            drain_handle: None,
            callback_handle: Some(callback_handle),
            consumer: Some(consumer),
            sender: Some(sender),
        })
    }

    /// Start capturing audio.
    pub fn start(&mut self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);
        self.stream.play().context("Failed to start audio stream")?;

        // Only spawn threads if they haven't been spawned yet
        if self.drain_handle.is_none() {
            // Spawn drain thread now that running is true
            let consumer = self.consumer.take().context("Consumer already taken")?;
            let sender = self.sender.take().context("Sender already taken")?;
            let drain_running = self.running.clone();
            let drain_shutdown = self.shutdown.clone();

            let drain_handle = std::thread::spawn(move || {
                let mut consumer = consumer;
                let mut read_buffer = Vec::with_capacity(2048);
                read_buffer.resize(2048, 0.0);

                loop {
                    // Check shutdown flag first (permanent exit)
                    if drain_shutdown.load(Ordering::Relaxed) {
                        debug!("Drain thread shutting down");
                        return;
                    }

                    // Check pause flag (temporary pause)
                    if !drain_running.load(Ordering::Relaxed) {
                        // Paused - sleep and check again
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        continue;
                    }

                    // Read available samples from ring buffer
                    let available = consumer.occupied_len();
                    if available == 0 {
                        // Sleep briefly to avoid busy-waiting
                        // 100µs maintains low latency while reducing CPU usage
                        std::thread::sleep(std::time::Duration::from_micros(100));
                        continue;
                    }

                    // Read samples
                    let to_read = available.min(read_buffer.len());
                    let read = consumer.pop_slice(&mut read_buffer[..to_read]);

                    if read > 0 {
                        // Send to channel (blocks on full for backpressure)
                        let samples_to_send = read_buffer[..read].to_vec();
                        if sender.send(samples_to_send).is_err() {
                            debug!("Audio channel closed, drain thread exiting");
                            return;
                        }
                    }
                }
            });

            self.drain_handle = Some(drain_handle);
            info!("Audio capture started");
        } else {
            debug!("Audio capture resumed (threads already running)");
        }

        Ok(())
    }

    /// Stop capturing audio.
    /// This pauses the stream but keeps threads alive for resume.
    #[allow(dead_code)]
    pub fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        self.stream.pause().context("Failed to stop audio stream")?;
        debug!("Audio capture paused");
        Ok(())
    }

    /// Permanently stop and cleanup.
    /// Call this on program shutdown to ensure threads exit cleanly.
    pub fn shutdown(&mut self) {
        // Signal shutdown to all threads
        self.shutdown.store(true, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
        let _ = self.stream.pause();

        // Drop sender to wake up blocking recv/send
        drop(self.sender.take());

        // Wait for drain thread with timeout
        if let Some(handle) = self.drain_handle.take() {
            // Give it 100ms to exit gracefully
            std::thread::sleep(std::time::Duration::from_millis(100));
            if !handle.is_finished() {
                warn!("Drain thread didn't exit in time");
            }
            if let Err(e) = handle.join() {
                warn!("Failed to join drain thread: {:?}", e);
            }
        }

        // Wait for callback thread with timeout
        if let Some(handle) = self.callback_handle.take() {
            std::thread::sleep(std::time::Duration::from_millis(100));
            if !handle.is_finished() {
                warn!("Callback thread didn't exit in time");
            }
            if let Err(e) = handle.join() {
                warn!("Failed to join callback thread: {:?}", e);
            }
        }

        info!("Audio capture stopped");
    }

    /// Get a clone of the running flag for external control (for half-duplex mode).
    pub fn running_flag(&self) -> Arc<AtomicBool> {
        self.running.clone()
    }
}

impl Drop for Capturer {
    fn drop(&mut self) {
        self.shutdown();
    }
}
