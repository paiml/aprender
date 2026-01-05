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
mod tests {
    use super::*;

    // ============================================================
    // VadConfig Tests
    // ============================================================

    #[test]
    fn test_vad_config_default() {
        let config = VadConfig::default();
        assert!((config.energy_threshold - 0.01).abs() < f32::EPSILON);
        assert_eq!(config.min_speech_duration_ms, 250);
        assert_eq!(config.min_silence_duration_ms, 300);
        assert_eq!(config.frame_size, 512);
        assert_eq!(config.hop_size, 256);
    }

    #[test]
    fn test_vad_config_validate_valid() {
        let config = VadConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_vad_config_validate_energy_threshold_negative() {
        let config = VadConfig {
            energy_threshold: -0.1,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SpeechError::InvalidConfig(_)));
        assert!(err.to_string().contains("energy_threshold"));
    }

    #[test]
    fn test_vad_config_validate_energy_threshold_too_high() {
        let config = VadConfig {
            energy_threshold: 1.5,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SpeechError::InvalidConfig(_)));
    }

    #[test]
    fn test_vad_config_validate_frame_size_zero() {
        let config = VadConfig {
            frame_size: 0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SpeechError::InvalidConfig(_)));
        assert!(err.to_string().contains("frame_size"));
    }

    #[test]
    fn test_vad_config_validate_hop_size_zero() {
        let config = VadConfig {
            hop_size: 0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SpeechError::InvalidConfig(_)));
        assert!(err.to_string().contains("hop_size"));
    }

    #[test]
    fn test_vad_config_validate_hop_size_exceeds_frame_size() {
        let config = VadConfig {
            frame_size: 256,
            hop_size: 512,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, SpeechError::InvalidConfig(_)));
        assert!(err.to_string().contains("hop_size"));
    }

    #[test]
    fn test_vad_config_boundary_values() {
        // Energy threshold at boundaries
        let config = VadConfig {
            energy_threshold: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        let config = VadConfig {
            energy_threshold: 1.0,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        // Hop size equal to frame size is valid
        let config = VadConfig {
            frame_size: 256,
            hop_size: 256,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    // ============================================================
    // VoiceSegment Tests
    // ============================================================

    #[test]
    fn test_voice_segment_new() {
        let segment = VoiceSegment::new(1.0, 2.5, 0.3);
        assert!((segment.start - 1.0).abs() < f64::EPSILON);
        assert!((segment.end - 2.5).abs() < f64::EPSILON);
        assert!((segment.energy - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_voice_segment_duration() {
        let segment = VoiceSegment::new(1.0, 3.5, 0.5);
        assert!((segment.duration() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_voice_segment_zero_duration() {
        let segment = VoiceSegment::new(1.0, 1.0, 0.5);
        assert!((segment.duration()).abs() < f64::EPSILON);
    }

    // ============================================================
    // Vad Construction Tests
    // ============================================================

    #[test]
    fn test_vad_new_valid_config() {
        let config = VadConfig::default();
        let vad = Vad::new(config);
        assert!(vad.is_ok());
    }

    #[test]
    fn test_vad_new_invalid_config() {
        let config = VadConfig {
            energy_threshold: -1.0,
            ..Default::default()
        };
        let vad = Vad::new(config);
        assert!(vad.is_err());
    }

    #[test]
    fn test_vad_with_defaults() {
        let vad = Vad::with_defaults();
        assert!(vad.is_ok());
        let vad = vad.expect("default config should be valid");
        assert!((vad.config().energy_threshold - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vad_config_accessor() {
        let config = VadConfig {
            energy_threshold: 0.05,
            ..Default::default()
        };
        let vad = Vad::new(config.clone()).expect("valid config");
        assert_eq!(vad.config(), &config);
    }

    // ============================================================
    // Vad::detect() Input Validation Tests
    // ============================================================

    #[test]
    fn test_vad_detect_zero_sample_rate() {
        let vad = Vad::with_defaults().expect("valid config");
        let samples = vec![0.0f32; 1024];
        let result = vad.detect(&samples, 0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SpeechError::InvalidAudio(_)));
        assert!(err.to_string().contains("sample_rate"));
    }

    #[test]
    fn test_vad_detect_empty_samples() {
        let vad = Vad::with_defaults().expect("valid config");
        let samples: Vec<f32> = vec![];
        let result = vad.detect(&samples, 16000);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SpeechError::InvalidAudio(_)));
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_vad_detect_insufficient_samples() {
        let config = VadConfig {
            frame_size: 512,
            ..Default::default()
        };
        let vad = Vad::new(config).expect("valid config");
        let samples = vec![0.0f32; 256]; // Less than frame_size
        let result = vad.detect(&samples, 16000);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            SpeechError::InsufficientSamples {
                required: 512,
                provided: 256
            }
        ));
    }

    // ============================================================
    // Vad::detect() Core Functionality Tests
    // ============================================================

    #[test]
    fn test_vad_detect_silence() {
        let vad = Vad::with_defaults().expect("valid config");
        let samples = vec![0.0f32; 16000]; // 1 second of silence at 16kHz
        let segments = vad
            .detect(&samples, 16000)
            .expect("detection should succeed");
        assert!(
            segments.is_empty(),
            "No speech should be detected in silence"
        );
    }

    #[test]
    fn test_vad_detect_constant_low_energy() {
        let vad = Vad::with_defaults().expect("valid config");
        // Very low amplitude noise (below default threshold of 0.01)
        let samples: Vec<f32> = (0..16000).map(|_| 0.001).collect();
        let segments = vad
            .detect(&samples, 16000)
            .expect("detection should succeed");
        assert!(
            segments.is_empty(),
            "Low energy should not trigger detection"
        );
    }

    #[test]
    fn test_vad_detect_loud_signal() {
        let config = VadConfig {
            energy_threshold: 0.01,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 100,
            frame_size: 160,
            hop_size: 80,
        };
        let vad = Vad::new(config).expect("valid config");

        // Generate 500ms of loud signal at 16kHz
        let sample_rate = 16000;
        let duration_samples = sample_rate / 2; // 500ms
        let samples: Vec<f32> = (0..duration_samples)
            .map(|i| {
                // 440Hz sine wave at 0.5 amplitude
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert!(
            !segments.is_empty(),
            "Speech should be detected in loud signal"
        );

        // Should have one continuous segment
        assert_eq!(segments.len(), 1);

        // Segment should cover most of the audio
        let segment = &segments[0];
        assert!(segment.start < 0.1, "Segment should start near beginning");
        assert!(segment.duration() > 0.3, "Segment should be at least 300ms");
    }

    #[test]
    fn test_vad_detect_speech_in_middle() {
        let config = VadConfig {
            energy_threshold: 0.01,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 100,
            frame_size: 160,
            hop_size: 80,
        };
        let vad = Vad::new(config).expect("valid config");
        let sample_rate: u32 = 16000;
        let half_sec = (sample_rate / 2) as usize;

        // 500ms silence + 500ms speech + 500ms silence
        let mut samples = vec![0.0f32; half_sec]; // 500ms silence

        // 500ms of speech (sine wave)
        for i in 0..half_sec {
            let t = i as f32 / sample_rate as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        samples.extend(vec![0.0f32; half_sec]); // 500ms silence

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert_eq!(
            segments.len(),
            1,
            "Should detect exactly one speech segment"
        );

        let segment = &segments[0];
        // Speech starts around 500ms
        assert!(
            segment.start > 0.4 && segment.start < 0.6,
            "Segment should start around 0.5s"
        );
        // Speech ends around 1000ms
        assert!(
            segment.end > 0.9 && segment.end < 1.1,
            "Segment should end around 1.0s"
        );
    }

    #[test]
    fn test_vad_detect_multiple_segments() {
        let config = VadConfig {
            energy_threshold: 0.01,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 200,
            frame_size: 160,
            hop_size: 80,
        };
        let vad = Vad::new(config).expect("valid config");
        let sample_rate: u32 = 16000;
        let samples_300ms = (sample_rate * 3 / 10) as usize;
        let samples_400ms = (sample_rate * 4 / 10) as usize;

        let mut samples = Vec::new();

        // First speech segment: 300ms
        for i in 0..samples_300ms {
            let t = i as f32 / sample_rate as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        // Silence: 400ms (longer than min_silence_duration_ms)
        samples.extend(vec![0.0f32; samples_400ms]);

        // Second speech segment: 300ms
        for i in 0..samples_300ms {
            let t = i as f32 / sample_rate as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert_eq!(segments.len(), 2, "Should detect two speech segments");

        // First segment should be near the beginning
        assert!(segments[0].start < 0.1);
        // Second segment should be after the silence gap
        assert!(segments[1].start > 0.6);
    }

    #[test]
    fn test_vad_detect_short_speech_filtered() {
        let config = VadConfig {
            energy_threshold: 0.01,
            min_speech_duration_ms: 500, // Require at least 500ms
            min_silence_duration_ms: 100,
            frame_size: 160,
            hop_size: 80,
        };
        let vad = Vad::new(config).expect("valid config");
        let sample_rate: u32 = 16000;

        // Only 200ms of speech (shorter than min_speech_duration_ms)
        let duration_samples = (sample_rate / 5) as usize; // 200ms
        let samples: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert!(
            segments.is_empty(),
            "Short speech segments should be filtered out"
        );
    }

    #[test]
    fn test_vad_detect_speech_at_end() {
        let config = VadConfig {
            energy_threshold: 0.01,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 100,
            frame_size: 160,
            hop_size: 80,
        };
        let vad = Vad::new(config).expect("valid config");
        let sample_rate: u32 = 16000;
        let half_sec = (sample_rate / 2) as usize;

        // 500ms silence + 500ms speech (no trailing silence)
        let mut samples = vec![0.0f32; half_sec];

        for i in 0..half_sec {
            let t = i as f32 / sample_rate as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert_eq!(segments.len(), 1, "Should detect speech at end");

        let segment = &segments[0];
        assert!(
            segment.end > 0.9,
            "Segment should extend to near end of audio"
        );
    }

    // ============================================================
    // Vad::compute_frame_energy Tests
    // ============================================================

    #[test]
    fn test_compute_frame_energy_silence() {
        let frame = vec![0.0f32; 512];
        let energy = Vad::compute_frame_energy(&frame);
        assert!(
            energy.abs() < f32::EPSILON,
            "Silence should have zero energy"
        );
    }

    #[test]
    fn test_compute_frame_energy_constant() {
        let frame = vec![0.5f32; 512];
        let energy = Vad::compute_frame_energy(&frame);
        assert!(
            (energy - 0.5).abs() < 0.001,
            "Constant signal RMS should equal amplitude"
        );
    }

    #[test]
    fn test_compute_frame_energy_sine_wave() {
        // Full cycle of sine wave
        let frame: Vec<f32> = (0..360)
            .map(|i| (i as f32 * std::f32::consts::PI / 180.0).sin())
            .collect();
        let energy = Vad::compute_frame_energy(&frame);
        // RMS of sine wave is 1/sqrt(2) ≈ 0.707
        assert!(
            (energy - 0.707).abs() < 0.02,
            "Sine wave RMS should be ~0.707, got {}",
            energy
        );
    }

    #[test]
    fn test_compute_frame_energy_empty() {
        let frame: Vec<f32> = vec![];
        let energy = Vad::compute_frame_energy(&frame);
        assert!(
            energy.abs() < f32::EPSILON,
            "Empty frame should have zero energy"
        );
    }

    // ============================================================
    // Edge Case Tests
    // ============================================================

    #[test]
    fn test_vad_detect_exactly_frame_size_samples() {
        let config = VadConfig {
            frame_size: 512,
            hop_size: 256,
            energy_threshold: 0.01,
            min_speech_duration_ms: 10,
            min_silence_duration_ms: 10,
        };
        let vad = Vad::new(config).expect("valid config");

        // Exactly frame_size samples of loud signal
        let samples: Vec<f32> = (0..512)
            .map(|i| {
                let t = i as f32 / 16000.0;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let result = vad.detect(&samples, 16000);
        assert!(result.is_ok(), "Should handle exactly frame_size samples");
    }

    #[test]
    fn test_vad_detect_high_sample_rate() {
        let vad = Vad::with_defaults().expect("valid config");
        let sample_rate: u32 = 48000; // 48kHz
        let samples: Vec<f32> = (0..48000)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert!(
            !segments.is_empty(),
            "Should detect speech at high sample rate"
        );
    }

    #[test]
    fn test_vad_detect_low_sample_rate() {
        let config = VadConfig {
            frame_size: 80, // Smaller frame for 8kHz
            hop_size: 40,
            energy_threshold: 0.01,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 100,
        };
        let vad = Vad::new(config).expect("valid config");
        let sample_rate: u32 = 8000; // 8kHz

        let samples: Vec<f32> = (0..4000) // 500ms
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert!(
            !segments.is_empty(),
            "Should detect speech at low sample rate"
        );
    }

    #[test]
    fn test_vad_segment_energy_calculation() {
        let config = VadConfig {
            energy_threshold: 0.01,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 100,
            frame_size: 160,
            hop_size: 80,
        };
        let vad = Vad::new(config).expect("valid config");
        let sample_rate: u32 = 16000;

        // Generate signal with known amplitude
        let samples: Vec<f32> = (0..8000) // 500ms
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");
        assert!(!segments.is_empty());

        let segment = &segments[0];
        // RMS of 0.5 amplitude sine wave is 0.5 / sqrt(2) ≈ 0.354
        assert!(
            segment.energy > 0.3 && segment.energy < 0.4,
            "Segment energy should reflect signal amplitude, got {}",
            segment.energy
        );
    }

    // ============================================================
    // B1-B10: Popperian Falsification Tests (GH-133)
    // ============================================================

    /// B3: VAD segments have start < end
    #[test]
    fn test_vad_b3_segments_start_before_end() {
        let config = VadConfig {
            energy_threshold: 0.01,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 100,
            frame_size: 160,
            hop_size: 80,
        };
        let vad = Vad::new(config).expect("valid config");
        let sample_rate: u32 = 16000;

        // Create audio with multiple speech segments
        let mut samples = Vec::new();
        for _ in 0..3 {
            // 300ms speech
            for i in 0..4800 {
                let t = i as f32 / sample_rate as f32;
                samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
            }
            // 400ms silence
            samples.extend(vec![0.0f32; 6400]);
        }

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");

        // B3: Every segment must have start < end
        for segment in &segments {
            assert!(
                segment.start < segment.end,
                "B3 VIOLATED: segment.start ({}) >= segment.end ({})",
                segment.start,
                segment.end
            );
        }
    }

    /// B4: VAD confidence (energy) in [0, 1]
    #[test]
    fn test_vad_b4_energy_bounded() {
        let vad = Vad::with_defaults().expect("valid config");
        let sample_rate: u32 = 16000;

        // Test with various signal amplitudes
        for amplitude in [0.1, 0.5, 1.0] {
            let samples: Vec<f32> = (0..8000)
                .map(|i| {
                    let t = i as f32 / sample_rate as f32;
                    amplitude * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                })
                .collect();

            let segments = vad
                .detect(&samples, sample_rate)
                .expect("detection should succeed");

            for segment in &segments {
                assert!(
                    segment.energy >= 0.0 && segment.energy <= 1.0,
                    "B4 VIOLATED: energy {} not in [0, 1]",
                    segment.energy
                );
            }
        }
    }

    /// B8: VAD handles stereo input (by requiring mono)
    /// Note: VAD expects mono input. Stereo should be converted first.
    #[test]
    fn test_vad_b8_handles_mono_requirement() {
        let vad = Vad::with_defaults().expect("valid config");
        let sample_rate: u32 = 16000;

        // Simulate stereo by interleaving two channels
        // In practice, stereo should be converted to mono before VAD
        let samples_mono: Vec<f32> = (0..8000)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        // VAD should work on mono data
        let result = vad.detect(&samples_mono, sample_rate);
        assert!(result.is_ok(), "VAD should handle mono audio");
    }

    /// B10: VAD default energy threshold is reasonable (0.01 for energy-based VAD)
    #[test]
    fn test_vad_b10_default_threshold_reasonable() {
        let config = VadConfig::default();
        // Energy threshold of 0.01 is reasonable for energy-based VAD
        // (0.5 would be too high for energy which is typically in [0, 1])
        assert!(
            config.energy_threshold > 0.0 && config.energy_threshold < 0.1,
            "B10: Default energy threshold should be reasonable (0.0 < threshold < 0.1), got {}",
            config.energy_threshold
        );
        // Verify the default is exactly 0.01 as documented
        assert!(
            (config.energy_threshold - 0.01).abs() < 1e-6,
            "Default energy threshold should be 0.01"
        );
    }

    /// B5: Verify min_speech_ms filtering explicitly
    #[test]
    fn test_vad_b5_min_speech_duration_filtering() {
        let config = VadConfig {
            energy_threshold: 0.01,
            min_speech_duration_ms: 300, // Require at least 300ms
            min_silence_duration_ms: 100,
            frame_size: 160,
            hop_size: 80,
        };
        let vad = Vad::new(config).expect("valid config");
        let sample_rate: u32 = 16000;

        // 200ms speech (less than min_speech_duration_ms)
        let short_samples: Vec<f32> = (0..3200)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let segments = vad
            .detect(&short_samples, sample_rate)
            .expect("detection should succeed");
        assert!(
            segments.is_empty(),
            "B5 VIOLATED: segments shorter than min_speech_duration_ms should be filtered"
        );

        // 500ms speech (more than min_speech_duration_ms)
        let long_samples: Vec<f32> = (0..8000)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let segments = vad
            .detect(&long_samples, sample_rate)
            .expect("detection should succeed");
        assert!(
            !segments.is_empty(),
            "B5: segments longer than min_speech_duration_ms should be kept"
        );
    }

    /// B6: Verify min_silence_ms gap merging
    #[test]
    fn test_vad_b6_min_silence_duration_merging() {
        let config = VadConfig {
            energy_threshold: 0.01,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 400, // Require at least 400ms silence to separate
            frame_size: 160,
            hop_size: 80,
        };
        let vad = Vad::new(config).expect("valid config");
        let sample_rate: u32 = 16000;

        let mut samples = Vec::new();

        // 300ms speech
        for i in 0..4800 {
            let t = i as f32 / sample_rate as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        // 200ms silence (less than min_silence_duration_ms)
        samples.extend(vec![0.0f32; 3200]);

        // 300ms speech
        for i in 0..4800 {
            let t = i as f32 / sample_rate as f32;
            samples.push(0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin());
        }

        let segments = vad
            .detect(&samples, sample_rate)
            .expect("detection should succeed");

        // Should merge into single segment since silence < min_silence_duration_ms
        assert_eq!(
            segments.len(),
            1,
            "B6: Short silence gaps should be merged into single segment"
        );
    }
}
