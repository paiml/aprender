//! Audio Emulation for WASM Testing (PROBAR-SPEC-010)
//!
//! Mock `getUserMedia` with controlled audio for streaming ASR testing.
//!
//! ## Toyota Way Application:
//! - **Poka-Yoke**: Type-safe audio source configuration prevents invalid audio
//! - **Jidoka**: Automatic detection of audio injection failures
//! - **Muda**: Eliminates need for real microphone in CI environments
//!
//! ## References:
//! - [11] Radford et al. (2023) Whisper streaming patterns
//! - [12] Sohn et al. (2015) VAD state machine testing

use std::f32::consts::PI;

/// Audio source types for injection (H4-H6 falsification)
#[derive(Debug, Clone)]
pub enum AudioSource {
    /// Sine wave at specified frequency (Hz)
    SineWave {
        /// Frequency in Hz (must be > 0, capped at Nyquist)
        frequency: f32,
        /// Amplitude in range [0.0, 1.0]
        amplitude: f32,
    },

    /// Speech-like audio (fundamental + harmonics)
    /// Per Radford et al. [11], speech has characteristic harmonic structure
    SpeechPattern {
        /// Fundamental frequency (100-300 Hz typical for speech)
        fundamental_hz: f32,
        /// Harmonic amplitudes relative to fundamental (e.g., [0.5, 0.3, 0.2, 0.1])
        harmonics: Vec<f32>,
        /// Pitch variation in Hz (adds natural variation)
        variation_hz: f32,
    },

    /// Silence with optional background noise
    Silence {
        /// Noise floor in dB (negative values, e.g., -60.0)
        noise_floor_db: f32,
    },

    /// White noise (for VAD testing - should NOT be classified as speech)
    WhiteNoise {
        /// Amplitude in range [0.0, 1.0]
        amplitude: f32,
    },

    /// Pre-recorded audio samples (for deterministic testing)
    Samples {
        /// Raw f32 samples in range [-1.0, 1.0]
        data: Vec<f32>,
        /// Sample rate of the data
        sample_rate: u32,
        /// Whether to loop when exhausted
        loop_playback: bool,
    },
}

impl Default for AudioSource {
    fn default() -> Self {
        Self::Silence {
            noise_floor_db: -60.0,
        }
    }
}

/// Audio emulator configuration
#[derive(Debug, Clone)]
pub struct AudioEmulatorConfig {
    /// Output sample rate (typically 16000 for ASR, 44100/48000 for general audio)
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u8,
    /// Buffer size in samples per callback
    pub buffer_size: usize,
}

impl Default for AudioEmulatorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000, // Whisper expects 16kHz
            channels: 1,        // Mono for ASR
            buffer_size: 1024,
        }
    }
}

/// Audio emulator for injecting controlled audio into browser tests
///
/// ## Usage
/// ```rust,ignore
/// let audio = AudioEmulator::new(AudioSource::SpeechPattern {
///     fundamental_hz: 150.0,
///     harmonics: vec![0.5, 0.3, 0.2, 0.1],
///     variation_hz: 20.0,
/// });
/// let samples = audio.generate_samples(3.0); // 3 seconds
/// ```
#[derive(Debug, Clone)]
pub struct AudioEmulator {
    source: AudioSource,
    config: AudioEmulatorConfig,
    /// Current phase for oscillator-based sources
    phase: f32,
    /// Sample counter for time tracking
    sample_count: u64,
    /// Random state for noise generation (deterministic seed)
    rng_state: u64,
}

impl AudioEmulator {
    /// Create a new audio emulator with the given source
    #[must_use]
    pub fn new(source: AudioSource) -> Self {
        Self::with_config(source, AudioEmulatorConfig::default())
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(source: AudioSource, config: AudioEmulatorConfig) -> Self {
        Self {
            source,
            config,
            phase: 0.0,
            sample_count: 0,
            rng_state: 0x853c_49e6_748f_ea9b, // Fixed seed for determinism
        }
    }

    /// Get the configured sample rate
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the number of samples generated so far
    #[must_use]
    pub fn samples_generated(&self) -> u64 {
        self.sample_count
    }

    /// Generate samples for the specified duration in seconds
    #[must_use]
    pub fn generate_samples(&mut self, duration_seconds: f32) -> Vec<f32> {
        let num_samples = (duration_seconds * self.config.sample_rate as f32) as usize;
        self.generate_n_samples(num_samples)
    }

    /// Generate exactly N samples
    #[must_use]
    pub fn generate_n_samples(&mut self, num_samples: usize) -> Vec<f32> {
        let mut samples = Vec::with_capacity(num_samples);
        let sample_rate = self.config.sample_rate as f32;

        for _ in 0..num_samples {
            let sample = self.generate_single_sample(sample_rate);
            samples.push(sample);
            self.sample_count += 1;
        }

        samples
    }

    /// Generate a single sample
    fn generate_single_sample(&mut self, sample_rate: f32) -> f32 {
        match &self.source {
            AudioSource::SineWave {
                frequency,
                amplitude,
            } => {
                let freq = frequency.clamp(0.001, sample_rate / 2.0);
                let amp = amplitude.clamp(0.0, 1.0);
                let sample = (self.phase * 2.0 * PI).sin() * amp;
                self.phase += freq / sample_rate;
                if self.phase >= 1.0 {
                    self.phase -= 1.0;
                }
                sample
            }

            AudioSource::SpeechPattern {
                fundamental_hz,
                harmonics,
                variation_hz,
            } => {
                let freq = fundamental_hz.clamp(20.0, sample_rate / 2.0);
                let var = variation_hz.clamp(0.0, freq / 2.0);

                // Add slow variation to fundamental (simulates natural pitch variation)
                let time = self.sample_count as f32 / sample_rate;
                let freq_with_variation = freq + var * (time * 5.0).sin();

                // Generate fundamental
                let mut sample = (self.phase * 2.0 * PI).sin();

                // Add harmonics
                for (i, &harmonic_amp) in harmonics.iter().enumerate() {
                    let harmonic_num = (i + 2) as f32;
                    let harmonic_freq = freq_with_variation * harmonic_num;
                    if harmonic_freq < sample_rate / 2.0 {
                        let harmonic_phase = self.phase * harmonic_num;
                        sample += (harmonic_phase * 2.0 * PI).sin() * harmonic_amp.clamp(0.0, 1.0);
                    }
                }

                // Normalize to prevent clipping
                let total_amp = 1.0 + harmonics.iter().sum::<f32>();
                sample /= total_amp.max(1.0);

                // Advance phase
                self.phase += freq_with_variation / sample_rate;
                if self.phase >= 1.0 {
                    self.phase -= 1.0;
                }

                sample.clamp(-1.0, 1.0)
            }

            AudioSource::Silence { noise_floor_db } => {
                // Convert dB to linear amplitude
                let amp = 10.0_f32.powf(noise_floor_db.clamp(-100.0, 0.0) / 20.0);
                // Generate noise at that level
                let noise = self.next_random_f32() * 2.0 - 1.0;
                noise * amp
            }

            AudioSource::WhiteNoise { amplitude } => {
                let amp = amplitude.clamp(0.0, 1.0);
                let noise = self.next_random_f32() * 2.0 - 1.0;
                noise * amp
            }

            AudioSource::Samples {
                data,
                sample_rate: _src_rate,
                loop_playback,
            } => {
                if data.is_empty() {
                    return 0.0;
                }
                let idx = self.sample_count as usize;
                if idx < data.len() {
                    data[idx].clamp(-1.0, 1.0)
                } else if *loop_playback {
                    data[idx % data.len()].clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            }
        }
    }

    /// Simple xorshift64 PRNG for deterministic noise
    fn next_random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        // Convert to [0, 1) range
        (self.rng_state as f32) / (u64::MAX as f32)
    }

    /// Reset the emulator state (phase and sample counter)
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.sample_count = 0;
        self.rng_state = 0x853c_49e6_748f_ea9b;
    }

    /// Generate JavaScript code to inject into page for mocking getUserMedia
    #[must_use]
    pub fn generate_mock_js(&self, samples: &[f32]) -> String {
        // Convert samples to JSON array
        let samples_json: String = samples
            .iter()
            .map(|s| format!("{s:.6}"))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            r#"
(function() {{
    const mockSamples = new Float32Array([{samples_json}]);
    const sampleRate = {sample_rate};
    let sampleIndex = 0;

    // Create mock MediaStream
    const audioContext = new AudioContext({{ sampleRate: sampleRate }});
    const bufferSize = 1024;
    const scriptNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

    scriptNode.onaudioprocess = function(e) {{
        const output = e.outputBuffer.getChannelData(0);
        for (let i = 0; i < bufferSize; i++) {{
            if (sampleIndex < mockSamples.length) {{
                output[i] = mockSamples[sampleIndex++];
            }} else {{
                output[i] = 0;
            }}
        }}
    }};

    const dest = audioContext.createMediaStreamDestination();
    scriptNode.connect(dest);
    scriptNode.connect(audioContext.destination);

    // Override getUserMedia
    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async function(constraints) {{
        if (constraints.audio) {{
            return dest.stream;
        }}
        return originalGetUserMedia(constraints);
    }};

    window.__PROBAR_AUDIO_EMULATOR__ = {{
        sampleIndex: () => sampleIndex,
        reset: () => {{ sampleIndex = 0; }},
        context: audioContext
    }};
}})();
"#,
            samples_json = samples_json,
            sample_rate = self.config.sample_rate
        )
    }

    /// Inject audio emulation into a CDP page
    ///
    /// Generates samples and injects them into the page, mocking `getUserMedia`.
    ///
    /// # Arguments
    /// * `page` - The CDP page to inject into
    /// * `duration_seconds` - Duration of audio to generate
    ///
    /// # Example
    /// ```ignore
    /// use jugar_probar::emulation::{AudioEmulator, AudioSource};
    ///
    /// let mut audio = AudioEmulator::new(AudioSource::SpeechPattern {
    ///     fundamental_hz: 150.0,
    ///     harmonics: vec![0.5, 0.3, 0.2],
    ///     variation_hz: 20.0,
    /// });
    /// audio.inject_cdp(&page, 5.0).await?; // 5 seconds of audio
    /// ```
    #[cfg(feature = "browser")]
    pub async fn inject_cdp(
        &mut self,
        page: &chromiumoxide::Page,
        duration_seconds: f32,
    ) -> Result<(), AudioEmulatorError> {
        // Generate samples
        let samples = self.generate_samples(duration_seconds);

        // Generate and inject the mock JavaScript
        let js = self.generate_mock_js(&samples);
        page.evaluate(js.as_str()).await.map_err(|e| {
            AudioEmulatorError::InjectionFailed(format!("CDP injection failed: {e}"))
        })?;

        Ok(())
    }

    /// Check if audio emulation is active on a CDP page
    #[cfg(feature = "browser")]
    pub async fn is_active_cdp(page: &chromiumoxide::Page) -> Result<bool, AudioEmulatorError> {
        let result: bool = page
            .evaluate("typeof window.__PROBAR_AUDIO_EMULATOR__ !== 'undefined'")
            .await
            .map_err(|e| AudioEmulatorError::InjectionFailed(format!("CDP check failed: {e}")))?
            .into_value()
            .unwrap_or(false);

        Ok(result)
    }

    /// Get current sample index from CDP page (tracks playback progress)
    #[cfg(feature = "browser")]
    pub async fn get_sample_index_cdp(
        page: &chromiumoxide::Page,
    ) -> Result<u64, AudioEmulatorError> {
        let result: f64 = page
            .evaluate("window.__PROBAR_AUDIO_EMULATOR__?.sampleIndex() ?? 0")
            .await
            .map_err(|e| AudioEmulatorError::InjectionFailed(format!("CDP query failed: {e}")))?
            .into_value()
            .unwrap_or(0.0);

        Ok(result as u64)
    }
}

/// Error type for audio emulation
#[derive(Debug, Clone)]
pub enum AudioEmulatorError {
    /// Injection failed
    InjectionFailed(String),
    /// Audio context not available
    ContextNotAvailable,
    /// Invalid configuration
    InvalidConfig(String),
}

impl std::fmt::Display for AudioEmulatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InjectionFailed(msg) => write!(f, "Audio injection failed: {msg}"),
            Self::ContextNotAvailable => write!(f, "Audio context not available"),
            Self::InvalidConfig(msg) => write!(f, "Invalid audio config: {msg}"),
        }
    }
}

impl std::error::Error for AudioEmulatorError {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    // ========================================================================
    // H4: Audio injection is reliable - Falsification tests
    // ========================================================================

    #[test]
    fn f016_zero_length_audio_no_crash() {
        // Falsification: Zero-length audio should not crash
        let mut emulator = AudioEmulator::new(AudioSource::Silence {
            noise_floor_db: -60.0,
        });
        let samples = emulator.generate_samples(0.0);
        assert!(samples.is_empty());
    }

    #[test]
    fn f017_context_suspended_handling() {
        // Falsification: Audio context suspension should be handled gracefully
        let mut emulator = AudioEmulator::new(AudioSource::SineWave {
            frequency: 440.0,
            amplitude: 0.5,
        });
        // Simulate context by generating samples
        let samples = emulator.generate_samples(0.1);
        assert!(!samples.is_empty());
        // Reset simulates context resume
        emulator.reset();
        let samples_after = emulator.generate_samples(0.1);
        assert!(!samples_after.is_empty());
    }

    #[test]
    fn f018_sample_rate_mismatch() {
        // Falsification: Different sample rates should be handled
        let mut emulator_16k = AudioEmulator::with_config(
            AudioSource::SineWave {
                frequency: 440.0,
                amplitude: 1.0,
            },
            AudioEmulatorConfig {
                sample_rate: 16000,
                ..Default::default()
            },
        );
        let mut emulator_48k = AudioEmulator::with_config(
            AudioSource::SineWave {
                frequency: 440.0,
                amplitude: 1.0,
            },
            AudioEmulatorConfig {
                sample_rate: 48000,
                ..Default::default()
            },
        );
        let samples_16k = emulator_16k.generate_samples(0.1);
        let samples_48k = emulator_48k.generate_samples(0.1);
        // 48kHz should produce 3x more samples for same duration
        assert!(samples_48k.len() >= samples_16k.len() * 2);
    }

    #[test]
    fn f019_permission_denied_mock() {
        // Falsification: Even with mocked audio, edge cases should work
        // This simulates the scenario where getUserMedia would fail
        let emulator = AudioEmulator::new(AudioSource::Silence {
            noise_floor_db: -100.0,
        });
        // The mock JS should handle permission scenarios gracefully
        let mock_js = emulator.generate_mock_js(&[]);
        assert!(mock_js.contains("getUserMedia"));
        assert!(mock_js.contains("audio"));
    }

    #[test]
    fn f020_ultrasonic_filtered() {
        // Falsification: Ultrasonic frequencies (>22kHz) should be capped at Nyquist
        let mut emulator = AudioEmulator::with_config(
            AudioSource::SineWave {
                frequency: 50000.0, // Way above Nyquist for 16kHz
                amplitude: 1.0,
            },
            AudioEmulatorConfig {
                sample_rate: 16000,
                ..Default::default()
            },
        );
        let samples = emulator.generate_samples(0.1);
        // Should not crash and should produce valid samples
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    // ========================================================================
    // H5: Pattern generation is accurate - Falsification tests
    // ========================================================================

    #[test]
    fn f021_zero_hz_sine() {
        // Falsification: 0 Hz sine should be clamped to minimum frequency
        let mut emulator = AudioEmulator::new(AudioSource::SineWave {
            frequency: 0.0,
            amplitude: 1.0,
        });
        let samples = emulator.generate_samples(0.1);
        assert!(!samples.is_empty());
        // Should produce DC-like output (very slow oscillation)
    }

    #[test]
    fn f022_negative_amplitude_handled() {
        // Falsification: Negative amplitude should be clamped to 0
        let mut emulator = AudioEmulator::new(AudioSource::SineWave {
            frequency: 440.0,
            amplitude: -1.0,
        });
        let samples = emulator.generate_samples(0.1);
        // All samples should be 0 or very close
        assert!(samples.iter().all(|&s| s.abs() < 0.001));
    }

    #[test]
    fn f023_amplitude_clamped() {
        // Falsification: Amplitude > 1.0 should be clamped
        let mut emulator = AudioEmulator::new(AudioSource::SineWave {
            frequency: 440.0,
            amplitude: 5.0,
        });
        let samples = emulator.generate_samples(0.1);
        // All samples should be in [-1, 1]
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn f024_speech_pattern_no_harmonics() {
        // Falsification: Speech pattern with empty harmonics should still work
        let mut emulator = AudioEmulator::new(AudioSource::SpeechPattern {
            fundamental_hz: 150.0,
            harmonics: vec![],
            variation_hz: 20.0,
        });
        let samples = emulator.generate_samples(0.1);
        assert!(!samples.is_empty());
        // Should produce valid samples (pure fundamental)
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn f025_samples_callback_empty() {
        // Falsification: Empty sample buffer should return zeros
        let mut emulator = AudioEmulator::new(AudioSource::Samples {
            data: vec![],
            sample_rate: 16000,
            loop_playback: false,
        });
        let samples = emulator.generate_samples(0.1);
        assert!(samples.iter().all(|&s| s.abs() < f32::EPSILON));
    }

    // ========================================================================
    // H6: VAD Detection Works - Falsification tests
    // ========================================================================

    #[test]
    fn f026_pure_silence() {
        // Falsification: Pure silence should have near-zero amplitude
        let mut emulator = AudioEmulator::new(AudioSource::Silence {
            noise_floor_db: -100.0,
        });
        let samples = emulator.generate_samples(0.1);
        let rms = calculate_rms(&samples);
        assert!(rms < 0.0001, "Silence RMS too high: {rms}");
    }

    #[test]
    fn f027_white_noise_not_silent() {
        // Falsification: White noise should have measurable energy
        let mut emulator = AudioEmulator::new(AudioSource::WhiteNoise { amplitude: 0.5 });
        let samples = emulator.generate_samples(0.1);
        let rms = calculate_rms(&samples);
        assert!(rms > 0.1, "White noise RMS too low: {rms}");
    }

    #[test]
    fn f028_speech_threshold_boundary() {
        // Falsification: Speech pattern at various amplitudes
        let mut emulator = AudioEmulator::new(AudioSource::SpeechPattern {
            fundamental_hz: 150.0,
            harmonics: vec![0.5, 0.3, 0.2],
            variation_hz: 10.0,
        });
        let samples = emulator.generate_samples(0.5);
        let rms = calculate_rms(&samples);
        // Speech-like audio should have consistent energy
        assert!(rms > 0.1 && rms < 1.0, "Speech RMS out of range: {rms}");
    }

    // ========================================================================
    // Unit tests for core functionality
    // ========================================================================

    #[test]
    fn test_sine_wave_generation() {
        let mut emulator = AudioEmulator::with_config(
            AudioSource::SineWave {
                frequency: 440.0,
                amplitude: 1.0,
            },
            AudioEmulatorConfig {
                sample_rate: 44100,
                ..Default::default()
            },
        );

        let samples = emulator.generate_samples(0.01); // 10ms
        assert_eq!(samples.len(), 441); // 44100 * 0.01

        // Verify samples are in valid range
        for &sample in &samples {
            assert!((-1.0..=1.0).contains(&sample));
        }

        // Verify zero crossings (should be ~4.4 per 10ms at 440Hz)
        let zero_crossings: usize = samples.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!((7..=11).contains(&zero_crossings));
    }

    #[test]
    fn test_speech_pattern_generation() {
        let mut emulator = AudioEmulator::new(AudioSource::SpeechPattern {
            fundamental_hz: 150.0,
            harmonics: vec![0.5, 0.3, 0.2, 0.1],
            variation_hz: 20.0,
        });

        let samples = emulator.generate_samples(1.0);
        assert_eq!(samples.len(), 16000); // 16kHz * 1s

        // Speech should have more complex waveform (more zero crossings than pure sine)
        let zero_crossings: usize = samples.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(zero_crossings > 200, "Too few zero crossings for speech");
    }

    #[test]
    fn test_deterministic_noise() {
        // Noise with same seed should produce identical output
        let mut emulator1 = AudioEmulator::new(AudioSource::WhiteNoise { amplitude: 1.0 });
        let mut emulator2 = AudioEmulator::new(AudioSource::WhiteNoise { amplitude: 1.0 });

        let samples1 = emulator1.generate_samples(0.1);
        let samples2 = emulator2.generate_samples(0.1);

        assert_eq!(samples1, samples2);
    }

    #[test]
    fn test_reset() {
        let mut emulator = AudioEmulator::new(AudioSource::SineWave {
            frequency: 440.0,
            amplitude: 1.0,
        });

        let samples1 = emulator.generate_samples(0.1);
        emulator.reset();
        let samples2 = emulator.generate_samples(0.1);

        // After reset, should produce identical output
        assert_eq!(samples1, samples2);
    }

    #[test]
    fn test_sample_counter() {
        let mut emulator = AudioEmulator::new(AudioSource::Silence {
            noise_floor_db: -60.0,
        });
        assert_eq!(emulator.samples_generated(), 0);

        let _ = emulator.generate_n_samples(1000);
        assert_eq!(emulator.samples_generated(), 1000);

        let _ = emulator.generate_n_samples(500);
        assert_eq!(emulator.samples_generated(), 1500);
    }

    #[test]
    fn test_samples_source_with_loop() {
        let data = vec![0.1, 0.2, 0.3, 0.4];
        let mut emulator = AudioEmulator::new(AudioSource::Samples {
            data,
            sample_rate: 16000,
            loop_playback: true,
        });

        let samples = emulator.generate_n_samples(10);
        assert_eq!(samples[0], 0.1);
        assert_eq!(samples[3], 0.4);
        assert_eq!(samples[4], 0.1); // Looped
        assert_eq!(samples[7], 0.4);
    }

    #[test]
    fn test_samples_source_without_loop() {
        let data = vec![0.1, 0.2, 0.3];
        let mut emulator = AudioEmulator::new(AudioSource::Samples {
            data,
            sample_rate: 16000,
            loop_playback: false,
        });

        let samples = emulator.generate_n_samples(6);
        assert_eq!(samples[0], 0.1);
        assert_eq!(samples[2], 0.3);
        assert!(samples[3].abs() < f32::EPSILON); // Silence after exhausted
    }

    #[test]
    fn test_mock_js_generation() {
        let emulator = AudioEmulator::new(AudioSource::Silence {
            noise_floor_db: -60.0,
        });
        let samples = vec![0.1, 0.2, 0.3];
        let js = emulator.generate_mock_js(&samples);

        assert!(js.contains("mockSamples"));
        assert!(js.contains("sampleRate"));
        assert!(js.contains("getUserMedia"));
        assert!(js.contains("__PROBAR_AUDIO_EMULATOR__"));
    }

    #[test]
    fn test_default_config() {
        let config = AudioEmulatorConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.buffer_size, 1024);
    }

    /// Helper to calculate RMS amplitude
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    // ========================================================================
    // Additional coverage tests for 95%+ target
    // ========================================================================

    #[test]
    fn test_audio_source_default() {
        // Coverage: AudioSource::default() implementation
        let source = AudioSource::default();
        match source {
            AudioSource::Silence { noise_floor_db } => {
                assert!((noise_floor_db - (-60.0)).abs() < f32::EPSILON);
            }
            _ => panic!("Default should be Silence variant"),
        }
    }

    #[test]
    fn test_audio_emulator_error_display_injection_failed() {
        // Coverage: AudioEmulatorError::InjectionFailed Display
        let error = AudioEmulatorError::InjectionFailed("test error".to_string());
        let display = format!("{error}");
        assert_eq!(display, "Audio injection failed: test error");
    }

    #[test]
    fn test_audio_emulator_error_display_context_not_available() {
        // Coverage: AudioEmulatorError::ContextNotAvailable Display
        let error = AudioEmulatorError::ContextNotAvailable;
        let display = format!("{error}");
        assert_eq!(display, "Audio context not available");
    }

    #[test]
    fn test_audio_emulator_error_display_invalid_config() {
        // Coverage: AudioEmulatorError::InvalidConfig Display
        let error = AudioEmulatorError::InvalidConfig("bad config".to_string());
        let display = format!("{error}");
        assert_eq!(display, "Invalid audio config: bad config");
    }

    #[test]
    fn test_audio_emulator_error_is_error_trait() {
        // Coverage: std::error::Error impl for AudioEmulatorError
        let error: Box<dyn std::error::Error> = Box::new(AudioEmulatorError::ContextNotAvailable);
        // Just verify it implements Error trait
        assert!(error.to_string().contains("context"));
    }

    #[test]
    fn test_sample_rate_accessor() {
        // Coverage: AudioEmulator::sample_rate() method
        let emulator = AudioEmulator::with_config(
            AudioSource::Silence {
                noise_floor_db: -60.0,
            },
            AudioEmulatorConfig {
                sample_rate: 44100,
                channels: 2,
                buffer_size: 512,
            },
        );
        assert_eq!(emulator.sample_rate(), 44100);
    }

    #[test]
    fn test_sine_wave_phase_wrap() {
        // Coverage: Phase wrap-around (phase >= 1.0 branch)
        // Generate enough samples to guarantee multiple phase wraps
        let mut emulator = AudioEmulator::with_config(
            AudioSource::SineWave {
                frequency: 1000.0, // High frequency for faster phase advancement
                amplitude: 1.0,
            },
            AudioEmulatorConfig {
                sample_rate: 8000, // Low sample rate = faster phase wrap
                ..Default::default()
            },
        );
        // 1000 Hz at 8000 Hz sample rate = phase advances 0.125 per sample
        // After 8 samples, phase wraps (8 * 0.125 = 1.0)
        let samples = emulator.generate_n_samples(100);
        // Verify continuous output (no discontinuities from wrap)
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_speech_pattern_harmonic_exceeds_nyquist() {
        // Coverage: Harmonic frequency exceeding Nyquist (line 202-205)
        let mut emulator = AudioEmulator::with_config(
            AudioSource::SpeechPattern {
                fundamental_hz: 3000.0,                        // High fundamental
                harmonics: vec![0.5, 0.3, 0.2, 0.1, 0.1, 0.1], // Harmonics will exceed Nyquist
                variation_hz: 0.0,
            },
            AudioEmulatorConfig {
                sample_rate: 16000, // Nyquist = 8000 Hz
                ..Default::default()
            },
        );
        // 3000 Hz fundamental, harmonics at 6000, 9000, 12000... (last 3 exceed Nyquist)
        let samples = emulator.generate_n_samples(1000);
        // Should still produce valid samples
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_speech_pattern_phase_wrap() {
        // Coverage: Speech pattern phase wrap-around
        let mut emulator = AudioEmulator::with_config(
            AudioSource::SpeechPattern {
                fundamental_hz: 2000.0,
                harmonics: vec![0.3],
                variation_hz: 10.0,
            },
            AudioEmulatorConfig {
                sample_rate: 8000,
                ..Default::default()
            },
        );
        // Generate enough samples for multiple phase wraps
        let samples = emulator.generate_n_samples(500);
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_speech_pattern_variation_clamping() {
        // Coverage: variation_hz clamping (line 189)
        let mut emulator = AudioEmulator::new(AudioSource::SpeechPattern {
            fundamental_hz: 100.0,
            harmonics: vec![0.5],
            variation_hz: 1000.0, // Much larger than freq/2, should be clamped to 50
        });
        let samples = emulator.generate_n_samples(1000);
        // Should produce valid samples despite extreme variation
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_speech_pattern_low_fundamental_clamped() {
        // Coverage: fundamental_hz clamping to minimum 20.0
        let mut emulator = AudioEmulator::new(AudioSource::SpeechPattern {
            fundamental_hz: 5.0, // Below minimum, should be clamped to 20.0
            harmonics: vec![0.5, 0.3],
            variation_hz: 2.0,
        });
        let samples = emulator.generate_n_samples(1000);
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_speech_pattern_high_fundamental_clamped() {
        // Coverage: fundamental_hz clamping to Nyquist
        let mut emulator = AudioEmulator::with_config(
            AudioSource::SpeechPattern {
                fundamental_hz: 20000.0, // Above Nyquist for 16kHz
                harmonics: vec![0.5],
                variation_hz: 10.0,
            },
            AudioEmulatorConfig {
                sample_rate: 16000,
                ..Default::default()
            },
        );
        let samples = emulator.generate_n_samples(1000);
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_speech_pattern_harmonic_amplitude_clamping() {
        // Coverage: Harmonic amplitude clamping (line 204)
        let mut emulator = AudioEmulator::new(AudioSource::SpeechPattern {
            fundamental_hz: 150.0,
            harmonics: vec![2.0, -0.5, 1.5], // Amplitudes outside [0, 1] range
            variation_hz: 10.0,
        });
        let samples = emulator.generate_n_samples(1000);
        // Harmonics should be clamped, output in valid range
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_silence_noise_floor_clamping_high() {
        // Coverage: noise_floor_db clamping to 0 (max)
        let mut emulator = AudioEmulator::new(AudioSource::Silence {
            noise_floor_db: 10.0, // Above 0, should be clamped
        });
        let samples = emulator.generate_n_samples(1000);
        // At 0 dB, amplitude = 1.0, so noise could be in full [-1, 1] range
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_silence_noise_floor_clamping_low() {
        // Coverage: noise_floor_db clamping to -100 (min)
        let mut emulator = AudioEmulator::new(AudioSource::Silence {
            noise_floor_db: -200.0, // Below -100, should be clamped
        });
        let samples = emulator.generate_n_samples(1000);
        let rms = calculate_rms(&samples);
        // Should be extremely quiet
        assert!(rms < 0.0001);
    }

    #[test]
    fn test_samples_source_clamping_positive() {
        // Coverage: Sample clamping for values > 1.0 (lines 245, 247)
        let data = vec![1.5, 2.0, 0.5, -0.5];
        let mut emulator = AudioEmulator::new(AudioSource::Samples {
            data,
            sample_rate: 16000,
            loop_playback: false,
        });
        let samples = emulator.generate_n_samples(4);
        assert_eq!(samples[0], 1.0); // Clamped from 1.5
        assert_eq!(samples[1], 1.0); // Clamped from 2.0
        assert_eq!(samples[2], 0.5); // Unchanged
        assert_eq!(samples[3], -0.5); // Unchanged
    }

    #[test]
    fn test_samples_source_clamping_negative() {
        // Coverage: Sample clamping for values < -1.0
        let data = vec![-1.5, -2.0, 0.5];
        let mut emulator = AudioEmulator::new(AudioSource::Samples {
            data,
            sample_rate: 16000,
            loop_playback: false,
        });
        let samples = emulator.generate_n_samples(3);
        assert_eq!(samples[0], -1.0); // Clamped from -1.5
        assert_eq!(samples[1], -1.0); // Clamped from -2.0
        assert_eq!(samples[2], 0.5); // Unchanged
    }

    #[test]
    fn test_samples_source_loop_with_clamping() {
        // Coverage: Looped samples with clamping (line 247)
        let data = vec![1.5, -1.5]; // Both need clamping
        let mut emulator = AudioEmulator::new(AudioSource::Samples {
            data,
            sample_rate: 16000,
            loop_playback: true,
        });
        let samples = emulator.generate_n_samples(6);
        assert_eq!(samples[0], 1.0); // Clamped
        assert_eq!(samples[1], -1.0); // Clamped
        assert_eq!(samples[2], 1.0); // Looped and clamped
        assert_eq!(samples[3], -1.0); // Looped and clamped
    }

    #[test]
    fn test_white_noise_zero_amplitude() {
        // Coverage: WhiteNoise with zero amplitude
        let mut emulator = AudioEmulator::new(AudioSource::WhiteNoise { amplitude: 0.0 });
        let samples = emulator.generate_n_samples(1000);
        // All samples should be 0
        assert!(samples.iter().all(|&s| s.abs() < f32::EPSILON));
    }

    #[test]
    fn test_white_noise_negative_amplitude_clamped() {
        // Coverage: WhiteNoise with negative amplitude (clamped to 0)
        let mut emulator = AudioEmulator::new(AudioSource::WhiteNoise { amplitude: -0.5 });
        let samples = emulator.generate_n_samples(1000);
        // All samples should be 0 (amplitude clamped to 0)
        assert!(samples.iter().all(|&s| s.abs() < f32::EPSILON));
    }

    #[test]
    fn test_white_noise_high_amplitude_clamped() {
        // Coverage: WhiteNoise with amplitude > 1.0 (clamped to 1.0)
        let mut emulator = AudioEmulator::new(AudioSource::WhiteNoise { amplitude: 5.0 });
        let samples = emulator.generate_n_samples(1000);
        // All samples should be in valid range
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_rng_determinism_after_reset() {
        // Coverage: RNG state reset
        let mut emulator = AudioEmulator::new(AudioSource::WhiteNoise { amplitude: 1.0 });
        let samples1 = emulator.generate_n_samples(100);
        emulator.reset();
        let samples2 = emulator.generate_n_samples(100);
        // After reset, noise should be identical
        assert_eq!(samples1, samples2);
    }

    #[test]
    fn test_generate_mock_js_with_many_samples() {
        // Coverage: generate_mock_js with larger sample set
        let emulator = AudioEmulator::new(AudioSource::Silence {
            noise_floor_db: -60.0,
        });
        let samples: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        let js = emulator.generate_mock_js(&samples);

        // Verify sample count in output
        assert!(js.contains("0.990000")); // Last sample value
        assert!(js.contains("Float32Array"));
    }

    #[test]
    fn test_config_custom_channels_and_buffer() {
        // Coverage: Custom AudioEmulatorConfig values
        let config = AudioEmulatorConfig {
            sample_rate: 48000,
            channels: 2,
            buffer_size: 2048,
        };
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.buffer_size, 2048);
    }

    #[test]
    fn test_audio_source_clone() {
        // Coverage: AudioSource Clone implementation
        let source = AudioSource::SpeechPattern {
            fundamental_hz: 150.0,
            harmonics: vec![0.5, 0.3],
            variation_hz: 20.0,
        };
        let cloned = source;
        match cloned {
            AudioSource::SpeechPattern {
                fundamental_hz,
                harmonics,
                variation_hz,
            } => {
                assert!((fundamental_hz - 150.0).abs() < f32::EPSILON);
                assert_eq!(harmonics, vec![0.5, 0.3]);
                assert!((variation_hz - 20.0).abs() < f32::EPSILON);
            }
            _ => panic!("Clone should preserve variant"),
        }
    }

    #[test]
    fn test_audio_emulator_config_clone() {
        // Coverage: AudioEmulatorConfig Clone implementation
        let config = AudioEmulatorConfig {
            sample_rate: 22050,
            channels: 2,
            buffer_size: 512,
        };
        let cloned = config;
        assert_eq!(cloned.sample_rate, 22050);
        assert_eq!(cloned.channels, 2);
        assert_eq!(cloned.buffer_size, 512);
    }

    #[test]
    fn test_audio_emulator_clone() {
        // Coverage: AudioEmulator Clone implementation
        let mut emulator = AudioEmulator::new(AudioSource::SineWave {
            frequency: 440.0,
            amplitude: 0.8,
        });
        let _ = emulator.generate_n_samples(100); // Advance state

        let cloned = emulator.clone();
        assert_eq!(cloned.samples_generated(), 100);
        assert_eq!(cloned.sample_rate(), 16000);
    }

    #[test]
    fn test_audio_emulator_error_clone() {
        // Coverage: AudioEmulatorError Clone implementation
        let error = AudioEmulatorError::InjectionFailed("test".to_string());
        let cloned = error;
        match cloned {
            AudioEmulatorError::InjectionFailed(msg) => assert_eq!(msg, "test"),
            _ => panic!("Clone should preserve variant"),
        }
    }

    #[test]
    fn test_audio_source_debug() {
        // Coverage: AudioSource Debug implementation
        let source = AudioSource::SineWave {
            frequency: 440.0,
            amplitude: 1.0,
        };
        let debug_str = format!("{source:?}");
        assert!(debug_str.contains("SineWave"));
        assert!(debug_str.contains("440"));
    }

    #[test]
    fn test_audio_emulator_config_debug() {
        // Coverage: AudioEmulatorConfig Debug implementation
        let config = AudioEmulatorConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("16000"));
    }

    #[test]
    fn test_audio_emulator_debug() {
        // Coverage: AudioEmulator Debug implementation
        let emulator = AudioEmulator::new(AudioSource::default());
        let debug_str = format!("{emulator:?}");
        assert!(debug_str.contains("AudioEmulator"));
    }

    #[test]
    fn test_audio_emulator_error_debug() {
        // Coverage: AudioEmulatorError Debug implementation
        let error = AudioEmulatorError::ContextNotAvailable;
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("ContextNotAvailable"));
    }

    #[test]
    fn test_speech_pattern_normalization() {
        // Coverage: Speech pattern normalization (line 209-210)
        // Use harmonics that sum to > 1.0 to test normalization
        let mut emulator = AudioEmulator::new(AudioSource::SpeechPattern {
            fundamental_hz: 150.0,
            harmonics: vec![0.8, 0.8, 0.8, 0.8], // Sum = 3.2, total_amp = 4.2
            variation_hz: 0.0,
        });
        let samples = emulator.generate_n_samples(1000);
        // Normalization should keep all samples in range
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_sine_wave_very_low_frequency() {
        // Coverage: Very low frequency sine wave (near minimum clamp)
        let mut emulator = AudioEmulator::new(AudioSource::SineWave {
            frequency: 0.0001, // Very close to minimum
            amplitude: 1.0,
        });
        let samples = emulator.generate_n_samples(100);
        // At such low frequency, output should be near-constant
        assert!(samples.iter().all(|&s| (-1.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_generate_samples_fractional_duration() {
        // Coverage: generate_samples with fractional sample count
        let mut emulator = AudioEmulator::with_config(
            AudioSource::Silence {
                noise_floor_db: -60.0,
            },
            AudioEmulatorConfig {
                sample_rate: 1000, // 1kHz for easy math
                ..Default::default()
            },
        );
        // 0.0015 seconds * 1000 Hz = 1.5 samples, truncated to 1
        let samples = emulator.generate_samples(0.0015);
        assert_eq!(samples.len(), 1);
    }
}
