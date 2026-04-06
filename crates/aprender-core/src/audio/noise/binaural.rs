//! Binaural beat generation for brainwave entrainment

use std::f32::consts::TAU;

use super::config::{BinauralPreset, NoiseConfig};
use super::generator::NoiseGenerator;
use super::{NoiseError, NoiseResult};

/// Stereo noise generator with binaural beat modulation
#[derive(Debug)]
pub struct BinauralGenerator {
    generator: NoiseGenerator,
    frequency_offset: f32,
    carrier_freq: f32,
    phase_l: f32,
    phase_r: f32,
    sample_rate: f32,
}

impl BinauralGenerator {
    /// Create a new binaural generator with frequency offset in Hz
    pub fn new(config: NoiseConfig, frequency_offset: f32) -> NoiseResult<Self> {
        let sample_rate = config.sample_rate as f32;
        let generator = NoiseGenerator::new(config)?;

        Ok(Self {
            generator,
            frequency_offset: frequency_offset.abs(),
            carrier_freq: 200.0, // Base carrier frequency for binaural modulation
            phase_l: 0.0,
            phase_r: 0.0,
            sample_rate,
        })
    }

    /// Create from a binaural preset
    pub fn from_preset(config: NoiseConfig, preset: BinauralPreset) -> NoiseResult<Self> {
        Self::new(config, preset.frequency())
    }

    /// Generate stereo samples into separate left/right buffers
    pub fn generate_stereo(
        &mut self,
        output_l: &mut [f32],
        output_r: &mut [f32],
    ) -> NoiseResult<()> {
        if output_l.len() != output_r.len() {
            return Err(NoiseError::BufferSizeMismatch {
                expected: output_l.len(),
                actual: output_r.len(),
            });
        }

        let buffer_size = self.generator.config().buffer_size;
        if output_l.len() != buffer_size {
            return Err(NoiseError::BufferSizeMismatch {
                expected: buffer_size,
                actual: output_l.len(),
            });
        }

        // Generate base mono noise
        let mut mono_buffer = vec![0.0; buffer_size];
        self.generator.generate(&mut mono_buffer)?;

        // Apply binaural modulation
        let freq_l = self.carrier_freq;
        let freq_r = self.carrier_freq + self.frequency_offset;
        let phase_inc_l = TAU * freq_l / self.sample_rate;
        let phase_inc_r = TAU * freq_r / self.sample_rate;

        // Modulation depth - subtle to not distort the noise character
        let mod_depth = 0.1;

        for i in 0..buffer_size {
            let noise_sample = mono_buffer[i];

            // Left channel modulation
            let mod_l = 1.0 + mod_depth * self.phase_l.sin();
            output_l[i] = (noise_sample * mod_l).clamp(-1.0, 1.0);

            // Right channel modulation (slightly different frequency)
            let mod_r = 1.0 + mod_depth * self.phase_r.sin();
            output_r[i] = (noise_sample * mod_r).clamp(-1.0, 1.0);

            // Advance phases
            self.phase_l += phase_inc_l;
            self.phase_r += phase_inc_r;

            // Wrap phases to prevent overflow
            if self.phase_l > TAU {
                self.phase_l -= TAU;
            }
            if self.phase_r > TAU {
                self.phase_r -= TAU;
            }
        }

        Ok(())
    }

    /// Generate interleaved stereo samples [L, R, L, R, ...]
    pub fn generate_interleaved(&mut self, output: &mut [f32]) -> NoiseResult<()> {
        let buffer_size = self.generator.config().buffer_size;
        if output.len() != buffer_size * 2 {
            return Err(NoiseError::BufferSizeMismatch {
                expected: buffer_size * 2,
                actual: output.len(),
            });
        }

        let mut left = vec![0.0; buffer_size];
        let mut right = vec![0.0; buffer_size];
        self.generate_stereo(&mut left, &mut right)?;

        // Interleave
        for i in 0..buffer_size {
            output[i * 2] = left[i];
            output[i * 2 + 1] = right[i];
        }

        Ok(())
    }

    /// Update binaural frequency offset in real-time
    pub fn set_frequency_offset(&mut self, hz: f32) {
        self.frequency_offset = hz.abs();
    }

    /// Get current frequency offset
    #[must_use]
    pub fn frequency_offset(&self) -> f32 {
        self.frequency_offset
    }

    /// Update underlying noise configuration
    pub fn update_config(&mut self, config: NoiseConfig) -> NoiseResult<()> {
        self.sample_rate = config.sample_rate as f32;
        self.generator.update_config(config)
    }

    /// Get underlying noise configuration
    #[must_use]
    pub fn config(&self) -> &NoiseConfig {
        self.generator.config()
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        self.generator.reset();
        self.phase_l = 0.0;
        self.phase_r = 0.0;
    }

    /// Get current time
    #[must_use]
    pub fn time(&self) -> f64 {
        self.generator.time()
    }
}

#[cfg(test)]
#[path = "binaural_tests.rs"]
mod tests;
