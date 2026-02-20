
impl MockCaptureSource {
    /// Create a new mock capture source
    ///
    /// # Arguments
    /// * `config` - Audio configuration (sample rate determines signal timing)
    /// * `signal` - Type of signal to generate
    #[must_use]
    pub fn new(config: CaptureConfig, signal: MockSignal) -> Self {
        Self {
            config,
            signal,
            sample_index: 0,
            rng_state: 0x5DEECE66D, // LCG seed
        }
    }

    /// Create with silence signal
    #[must_use]
    pub fn silence(config: CaptureConfig) -> Self {
        Self::new(config, MockSignal::Silence)
    }

    /// Create with 440Hz sine wave (A4 note)
    #[must_use]
    pub fn a440(config: CaptureConfig) -> Self {
        Self::new(
            config,
            MockSignal::Sine {
                frequency: 440.0,
                amplitude: 0.5,
            },
        )
    }

    /// Create with white noise
    #[must_use]
    pub fn white_noise(config: CaptureConfig, amplitude: f32) -> Self {
        Self::new(config, MockSignal::WhiteNoise { amplitude })
    }

    /// Read samples into buffer
    ///
    /// # Arguments
    /// * `buffer` - Output buffer for samples
    ///
    /// # Returns
    /// Number of samples written (always fills the buffer)
    pub fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError> {
        let sample_rate = f64::from(self.config.sample_rate);

        for sample in buffer.iter_mut() {
            *sample = self.generate_sample(sample_rate);
            self.sample_index += 1;
        }

        Ok(buffer.len())
    }

    /// Generate a single sample based on signal type
    fn generate_sample(&mut self, sample_rate: f64) -> f32 {
        let t = self.sample_index as f64 / sample_rate;

        match self.signal {
            MockSignal::Silence => 0.0,

            MockSignal::Sine {
                frequency,
                amplitude,
            } => {
                let phase = 2.0 * std::f64::consts::PI * f64::from(frequency) * t;
                (phase.sin() * f64::from(amplitude)) as f32
            }

            MockSignal::WhiteNoise { amplitude } => {
                // Linear congruential generator for deterministic "random" noise
                self.rng_state = self.rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(11);
                // Take upper 32 bits and normalize to [0, 1]
                let upper_bits = (self.rng_state >> 32) as u32;
                let normalized = f64::from(upper_bits) / f64::from(u32::MAX);
                ((normalized * 2.0 - 1.0) * f64::from(amplitude)) as f32
            }

            MockSignal::Impulse => {
                if self.sample_index == 0 {
                    1.0
                } else {
                    0.0
                }
            }

            MockSignal::Square {
                frequency,
                amplitude,
            } => {
                let phase = 2.0 * std::f64::consts::PI * f64::from(frequency) * t;
                let value = if phase.sin() >= 0.0 { 1.0 } else { -1.0 };
                value * f64::from(amplitude) as f32
            }
        }
    }

    /// Reset the sample counter to start from beginning
    pub fn reset(&mut self) {
        self.sample_index = 0;
        self.rng_state = 0x5DEECE66D; // Reset RNG seed
    }

    /// Get current sample position
    #[must_use]
    pub fn position(&self) -> u64 {
        self.sample_index
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &CaptureConfig {
        &self.config
    }

    /// Get signal type
    #[must_use]
    pub fn signal(&self) -> MockSignal {
        self.signal
    }

    /// Change signal type
    pub fn set_signal(&mut self, signal: MockSignal) {
        self.signal = signal;
    }
}

/// Mock capture source that reads from a pre-recorded buffer
///
/// Useful for testing with specific audio content.
#[derive(Debug)]
pub struct BufferCaptureSource {
    samples: Vec<f32>,
    position: usize,
    loop_playback: bool,
}

impl BufferCaptureSource {
    /// Create from a sample buffer
    #[must_use]
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            position: 0,
            loop_playback: false,
        }
    }

    /// Enable looping (replay from start when buffer exhausted)
    #[must_use]
    pub fn with_loop(mut self, loop_playback: bool) -> Self {
        self.loop_playback = loop_playback;
        self
    }

    /// Read samples into buffer
    ///
    /// # Returns
    /// Number of samples read (may be less than buffer size if not looping)
    pub fn read(&mut self, buffer: &mut [f32]) -> Result<usize, AudioError> {
        if self.samples.is_empty() {
            return Ok(0);
        }

        let mut written = 0;
        for sample in buffer.iter_mut() {
            if self.position >= self.samples.len() {
                if self.loop_playback {
                    self.position = 0;
                } else {
                    break;
                }
            }
            *sample = self.samples[self.position];
            self.position += 1;
            written += 1;
        }

        Ok(written)
    }

    /// Reset position to start
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get current position
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Check if exhausted (for non-looping sources)
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        !self.loop_playback && self.position >= self.samples.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
