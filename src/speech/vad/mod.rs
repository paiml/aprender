//! Voice Activity Detection (VAD) module.
//!
//! Implements energy-based VAD following WebRTC patterns, suitable for
//! real-time speech processing pipelines.
//!
//! # PMAT Compliance
//!
//! - No `unwrap()` - audio streams can't panic
//! - All public APIs return `Result<T, E>`
//! - `#[must_use]` on all Results

use crate::speech::{SpeechError, SpeechResult};

/// Configuration for Voice Activity Detection.
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig {
    /// Energy threshold for speech detection (0.0 to 1.0).
    /// Higher values require louder audio to trigger detection.
    pub energy_threshold: f32,

    /// Minimum duration of speech segment in milliseconds.
    /// Segments shorter than this are filtered out.
    pub min_speech_duration_ms: u32,

    /// Minimum duration of silence in milliseconds to end a segment.
    pub min_silence_duration_ms: u32,

    /// Frame size in samples for energy calculation.
    pub frame_size: usize,

    /// Hop size between frames in samples.
    pub hop_size: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            energy_threshold: 0.01,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 300,
            frame_size: 512,
            hop_size: 256,
        }
    }
}

impl VadConfig {
    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns `SpeechError::InvalidConfig` if:
    /// - `energy_threshold` is not in range [0.0, 1.0]
    /// - `frame_size` is zero
    /// - `hop_size` is zero or greater than `frame_size`
    pub fn validate(&self) -> SpeechResult<()> {
        if !(0.0..=1.0).contains(&self.energy_threshold) {
            return Err(SpeechError::InvalidConfig(format!(
                "energy_threshold must be in [0.0, 1.0], got {}",
                self.energy_threshold
            )));
        }

        if self.frame_size == 0 {
            return Err(SpeechError::InvalidConfig(
                "frame_size must be greater than 0".to_string(),
            ));
        }

        if self.hop_size == 0 {
            return Err(SpeechError::InvalidConfig(
                "hop_size must be greater than 0".to_string(),
            ));
        }

        if self.hop_size > self.frame_size {
            return Err(SpeechError::InvalidConfig(format!(
                "hop_size ({}) must not exceed frame_size ({})",
                self.hop_size, self.frame_size
            )));
        }

        Ok(())
    }
}

/// A detected voice segment with start and end times.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceSegment {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Average energy level of the segment (0.0 to 1.0).
    pub energy: f32,
}

impl VoiceSegment {
    /// Create a new voice segment.
    #[must_use]
    pub fn new(start: f64, end: f64, energy: f32) -> Self {
        Self { start, end, energy }
    }

    /// Duration of the segment in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Voice Activity Detector using energy-based detection.
#[derive(Debug, Clone)]
pub struct Vad {
    config: VadConfig,
}

impl Vad {
    /// Create a new VAD with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `SpeechError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: VadConfig) -> SpeechResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Create a new VAD with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `SpeechError::InvalidConfig` if default configuration is invalid
    /// (should never happen).
    pub fn with_defaults() -> SpeechResult<Self> {
        Self::new(VadConfig::default())
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Detect voice activity in audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples as f32 in range [-1.0, 1.0]
    /// * `sample_rate` - Sample rate in Hz (e.g., 16000)
    ///
    /// # Returns
    ///
    /// A vector of detected voice segments, sorted by start time.
    ///
    /// # Errors
    ///
    /// Returns `SpeechError::InvalidAudio` if:
    /// - `sample_rate` is zero
    /// - `samples` is empty
    ///
    /// Returns `SpeechError::InsufficientSamples` if samples are fewer than `frame_size`.
    pub fn detect(&self, samples: &[f32], sample_rate: u32) -> SpeechResult<Vec<VoiceSegment>> {
        // Validate inputs
        if sample_rate == 0 {
            return Err(SpeechError::InvalidAudio(
                "sample_rate must be greater than 0".to_string(),
            ));
        }

        if samples.is_empty() {
            return Err(SpeechError::InvalidAudio(
                "samples cannot be empty".to_string(),
            ));
        }

        if samples.len() < self.config.frame_size {
            return Err(SpeechError::InsufficientSamples {
                required: self.config.frame_size,
                provided: samples.len(),
            });
        }

        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut speech_start_sample: usize = 0;
        let mut silence_samples: usize = 0;
        let mut segment_energy_sum: f32 = 0.0;
        let mut segment_frame_count: usize = 0;

        let min_speech_samples = (f64::from(self.config.min_speech_duration_ms)
            * f64::from(sample_rate)
            / 1000.0) as usize;
        let min_silence_samples = (f64::from(self.config.min_silence_duration_ms)
            * f64::from(sample_rate)
            / 1000.0) as usize;

        let mut pos = 0;
        while pos + self.config.frame_size <= samples.len() {
            let frame = &samples[pos..pos + self.config.frame_size];
            let energy = Self::compute_frame_energy(frame);

            let is_speech = energy > self.config.energy_threshold;

            if is_speech {
                if !in_speech {
                    // Start of speech segment
                    in_speech = true;
                    speech_start_sample = pos;
                    segment_energy_sum = 0.0;
                    segment_frame_count = 0;
                }
                silence_samples = 0;
                segment_energy_sum += energy;
                segment_frame_count += 1;
            } else if in_speech {
                // In speech but current frame is silence
                silence_samples += self.config.hop_size;

                if silence_samples >= min_silence_samples {
                    // End of speech segment
                    let speech_end_sample = pos - silence_samples + self.config.hop_size;
                    let speech_duration_samples =
                        speech_end_sample.saturating_sub(speech_start_sample);

                    if speech_duration_samples >= min_speech_samples {
                        let start = speech_start_sample as f64 / f64::from(sample_rate);
                        let end = speech_end_sample as f64 / f64::from(sample_rate);
                        let avg_energy = if segment_frame_count > 0 {
                            segment_energy_sum / segment_frame_count as f32
                        } else {
                            0.0
                        };

                        segments.push(VoiceSegment::new(start, end, avg_energy));
                    }

                    in_speech = false;
                    silence_samples = 0;
                }
            }

            pos += self.config.hop_size;
        }

        // Handle speech segment that extends to end of audio
        if in_speech {
            let speech_duration_samples = samples.len().saturating_sub(speech_start_sample);

            if speech_duration_samples >= min_speech_samples {
                let start = speech_start_sample as f64 / f64::from(sample_rate);
                let end = samples.len() as f64 / f64::from(sample_rate);
                let avg_energy = if segment_frame_count > 0 {
                    segment_energy_sum / segment_frame_count as f32
                } else {
                    0.0
                };

                segments.push(VoiceSegment::new(start, end, avg_energy));
            }
        }

        Ok(segments)
    }

    /// Compute RMS energy of a frame.
    fn compute_frame_energy(frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }

        let sum_sq: f32 = frame.iter().map(|&s| s * s).sum();
        (sum_sq / frame.len() as f32).sqrt()
    }
}


#[cfg(test)]
mod tests;
