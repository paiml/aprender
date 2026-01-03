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
mod tests {
    use super::*;

    // ========== NG21: Stereo output has correct frequency difference ==========

    #[test]
    fn test_ng21_frequency_offset_stored() {
        let config = NoiseConfig::brown();
        let gen = BinauralGenerator::new(config, 4.0).unwrap();
        assert!((gen.frequency_offset() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ng21_frequency_offset_from_preset() {
        let config = NoiseConfig::brown();
        let gen = BinauralGenerator::from_preset(config, BinauralPreset::Delta).unwrap();
        assert!((gen.frequency_offset() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ng21_frequency_offset_update() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();
        gen.set_frequency_offset(10.0);
        assert!((gen.frequency_offset() - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ng21_negative_offset_becomes_positive() {
        let config = NoiseConfig::brown();
        let gen = BinauralGenerator::new(config, -4.0).unwrap();
        assert!((gen.frequency_offset() - 4.0).abs() < f32::EPSILON);
    }

    // ========== NG22: Left and right channels are phase-coherent with noise ==========

    #[test]
    fn test_ng22_channels_coherent() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();

        let mut left = vec![0.0; 1024];
        let mut right = vec![0.0; 1024];
        gen.generate_stereo(&mut left, &mut right).unwrap();

        // Both channels should have similar overall characteristics
        let left_energy: f32 = left.iter().map(|x| x * x).sum();
        let right_energy: f32 = right.iter().map(|x| x * x).sum();

        // Energy should be similar (within 50%)
        let ratio = left_energy / right_energy;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Channel energies too different: L={}, R={}, ratio={}",
            left_energy,
            right_energy,
            ratio
        );
    }

    #[test]
    fn test_ng22_channels_correlated() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();

        let mut left = vec![0.0; 1024];
        let mut right = vec![0.0; 1024];
        gen.generate_stereo(&mut left, &mut right).unwrap();

        // Compute correlation
        let mean_l: f32 = left.iter().sum::<f32>() / left.len() as f32;
        let mean_r: f32 = right.iter().sum::<f32>() / right.len() as f32;

        let mut cov = 0.0;
        let mut var_l = 0.0;
        let mut var_r = 0.0;

        for i in 0..left.len() {
            let dl = left[i] - mean_l;
            let dr = right[i] - mean_r;
            cov += dl * dr;
            var_l += dl * dl;
            var_r += dr * dr;
        }

        let correlation = cov / (var_l.sqrt() * var_r.sqrt());

        // Should be highly correlated (same base noise)
        assert!(
            correlation > 0.8,
            "Channels not correlated enough: {}",
            correlation
        );
    }

    // ========== NG23: Frequency offset=0 produces mono-equivalent stereo ==========

    #[test]
    fn test_ng23_zero_offset_similar_channels() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 0.0).unwrap();

        let mut left = vec![0.0; 1024];
        let mut right = vec![0.0; 1024];
        gen.generate_stereo(&mut left, &mut right).unwrap();

        // With zero offset, channels should be nearly identical
        // (small differences due to modulation phase)
        let max_diff: f32 = left
            .iter()
            .zip(right.iter())
            .map(|(l, r)| (l - r).abs())
            .fold(0.0, f32::max);

        // With 0 offset, channels are modulated at the same frequency
        // but may start at different phases, so allow small difference
        assert!(
            max_diff < 0.3,
            "Channels too different with zero offset: max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_ng23_zero_offset_same_energy() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 0.0).unwrap();

        let mut left = vec![0.0; 1024];
        let mut right = vec![0.0; 1024];
        gen.generate_stereo(&mut left, &mut right).unwrap();

        let left_energy: f32 = left.iter().map(|x| x * x).sum();
        let right_energy: f32 = right.iter().map(|x| x * x).sum();

        let diff = (left_energy - right_energy).abs();
        let avg = (left_energy + right_energy) / 2.0;

        assert!(
            diff / avg < 0.1,
            "Energy differs too much: L={}, R={}",
            left_energy,
            right_energy
        );
    }

    // ========== NG24: Preset frequencies match brainwave ranges ==========

    #[test]
    fn test_ng24_delta_preset_in_range() {
        let (min, max) = BinauralPreset::Delta.frequency_range();
        let freq = BinauralPreset::Delta.frequency();
        assert!(
            freq >= min && freq <= max,
            "Delta frequency {} not in range [{}, {}]",
            freq,
            min,
            max
        );
    }

    #[test]
    fn test_ng24_theta_preset_in_range() {
        let (min, max) = BinauralPreset::Theta.frequency_range();
        let freq = BinauralPreset::Theta.frequency();
        assert!(
            freq >= min && freq <= max,
            "Theta frequency {} not in range [{}, {}]",
            freq,
            min,
            max
        );
    }

    #[test]
    fn test_ng24_alpha_preset_in_range() {
        let (min, max) = BinauralPreset::Alpha.frequency_range();
        let freq = BinauralPreset::Alpha.frequency();
        assert!(
            freq >= min && freq <= max,
            "Alpha frequency {} not in range [{}, {}]",
            freq,
            min,
            max
        );
    }

    #[test]
    fn test_ng24_beta_preset_in_range() {
        let (min, max) = BinauralPreset::Beta.frequency_range();
        let freq = BinauralPreset::Beta.frequency();
        assert!(
            freq >= min && freq <= max,
            "Beta frequency {} not in range [{}, {}]",
            freq,
            min,
            max
        );
    }

    #[test]
    fn test_ng24_gamma_preset_in_range() {
        let (min, max) = BinauralPreset::Gamma.frequency_range();
        let freq = BinauralPreset::Gamma.frequency();
        assert!(
            freq >= min && freq <= max,
            "Gamma frequency {} not in range [{}, {}]",
            freq,
            min,
            max
        );
    }

    // ========== NG-F6: Left/right channels cannot be identical when binaural enabled ==========

    #[test]
    fn test_ng_f6_channels_differ_with_offset() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();

        let mut left = vec![0.0; 1024];
        let mut right = vec![0.0; 1024];
        gen.generate_stereo(&mut left, &mut right).unwrap();

        // Channels should NOT be identical
        assert_ne!(left, right, "Binaural channels should differ");
    }

    #[test]
    fn test_ng_f6_channels_differ_measurably() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 10.0).unwrap();

        let mut left = vec![0.0; 1024];
        let mut right = vec![0.0; 1024];
        gen.generate_stereo(&mut left, &mut right).unwrap();

        // Count differences
        let diff_count = left
            .iter()
            .zip(right.iter())
            .filter(|(l, r)| (*l - *r).abs() > 0.001)
            .count();

        assert!(
            diff_count > left.len() / 2,
            "Not enough channel differences: {}/{}",
            diff_count,
            left.len()
        );
    }

    // ========== Additional binaural tests ==========

    #[test]
    fn test_generate_interleaved() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();

        let mut interleaved = vec![0.0; 2048]; // 1024 * 2
        gen.generate_interleaved(&mut interleaved).unwrap();

        // Verify interleaving pattern
        let mut left = vec![0.0; 1024];
        let mut right = vec![0.0; 1024];

        // Reset and generate again
        gen.reset();
        gen.generate_stereo(&mut left, &mut right).unwrap();

        // Reset and compare with interleaved
        gen.reset();
        let mut interleaved2 = vec![0.0; 2048];
        gen.generate_interleaved(&mut interleaved2).unwrap();

        // Interleaved should match
        for i in 0..1024 {
            assert!((interleaved2[i * 2] - left[i]).abs() < f32::EPSILON);
            assert!((interleaved2[i * 2 + 1] - right[i]).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_interleaved_wrong_size() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();

        let mut interleaved = vec![0.0; 1024]; // Wrong size
        let result = gen.generate_interleaved(&mut interleaved);
        assert!(result.is_err());
    }

    #[test]
    fn test_stereo_buffer_mismatch() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();

        let mut left = vec![0.0; 1024];
        let mut right = vec![0.0; 512]; // Wrong size
        let result = gen.generate_stereo(&mut left, &mut right);
        assert!(result.is_err());
    }

    #[test]
    fn test_stereo_wrong_buffer_size() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();

        let mut left = vec![0.0; 512]; // Wrong size
        let mut right = vec![0.0; 512]; // Wrong size
        let result = gen.generate_stereo(&mut left, &mut right);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_config() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();

        let new_config = NoiseConfig::pink();
        gen.update_config(new_config).unwrap();

        assert_eq!(
            gen.config().noise_type,
            super::super::config::NoiseType::Pink
        );
    }

    #[test]
    fn test_reset() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 4.0).unwrap();

        // Advance state
        let mut left = vec![0.0; 1024];
        let mut right = vec![0.0; 1024];
        for _ in 0..10 {
            gen.generate_stereo(&mut left, &mut right).unwrap();
        }

        assert!(gen.time() > 0.0);

        gen.reset();

        assert!((gen.time() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_output_bounded() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 10.0).unwrap();

        for _ in 0..50 {
            let mut left = vec![0.0; 1024];
            let mut right = vec![0.0; 1024];
            gen.generate_stereo(&mut left, &mut right).unwrap();

            for &sample in left.iter().chain(right.iter()) {
                assert!(sample >= -1.0 && sample <= 1.0);
            }
        }
    }

    #[test]
    fn test_no_nan_inf() {
        let config = NoiseConfig::brown();
        let mut gen = BinauralGenerator::new(config, 40.0).unwrap();

        for _ in 0..50 {
            let mut left = vec![0.0; 1024];
            let mut right = vec![0.0; 1024];
            gen.generate_stereo(&mut left, &mut right).unwrap();

            for &sample in left.iter().chain(right.iter()) {
                assert!(!sample.is_nan());
                assert!(!sample.is_infinite());
            }
        }
    }
}
