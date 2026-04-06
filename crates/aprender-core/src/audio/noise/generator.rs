//! Main noise generator implementation

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::path::Path;
use std::sync::Arc;

use super::config::NoiseConfig;
use super::phase::PhaseGenerator;
use super::spectral::SpectralMLP;
use super::{NoiseError, NoiseResult};

/// Crossfade length for seamless frame transitions (in samples)
const CROSSFADE_LEN: usize = 32;

/// Main noise generator using spectral MLP and iFFT synthesis
pub struct NoiseGenerator {
    config: NoiseConfig,
    mlp: SpectralMLP,
    phase_gen: PhaseGenerator,
    ifft: Arc<dyn Fft<f32>>,
    time: f64,
    sample_counter: u64,
    /// Last sample of previous buffer for seamless boundary
    prev_last_sample: f32,
    /// Whether we have a previous buffer
    has_prev: bool,
}

impl std::fmt::Debug for NoiseGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoiseGenerator")
            .field("config", &self.config)
            .field("time", &self.time)
            .field("sample_counter", &self.sample_counter)
            .finish_non_exhaustive()
    }
}

impl NoiseGenerator {
    /// Create a new noise generator with the given configuration
    pub fn new(config: NoiseConfig) -> NoiseResult<Self> {
        config.validate()?;

        // Initialize with a pretrained-like model (for now, random weights)
        // In production, this would load from a bundled .apr file
        let n_freqs = config.buffer_size / 2 + 1;
        let mlp = SpectralMLP::random_init(8, 64, n_freqs, 42);

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(config.buffer_size);

        let phase_gen = PhaseGenerator::new(12345);

        Ok(Self {
            config,
            mlp,
            phase_gen,
            ifft,
            time: 0.0,
            sample_counter: 0,
            prev_last_sample: 0.0,
            has_prev: false,
        })
    }

    /// Create from a pre-trained .apr model file
    pub fn from_apr<P: AsRef<Path>>(path: P, config: NoiseConfig) -> NoiseResult<Self> {
        config.validate()?;

        let mlp = SpectralMLP::load_apr(path)?;
        let n_freqs = config.buffer_size / 2 + 1;

        if mlp.n_freqs() != n_freqs {
            return Err(NoiseError::ModelError(format!(
                "Model n_freqs {} doesn't match config buffer_size {} (expected {})",
                mlp.n_freqs(),
                config.buffer_size,
                n_freqs
            )));
        }

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(config.buffer_size);

        let phase_gen = PhaseGenerator::new(12345);

        Ok(Self {
            config,
            mlp,
            phase_gen,
            ifft,
            time: 0.0,
            sample_counter: 0,
            prev_last_sample: 0.0,
            has_prev: false,
        })
    }

    /// Create with a specific MLP model (for training/testing)
    pub fn with_mlp(config: NoiseConfig, mlp: SpectralMLP) -> NoiseResult<Self> {
        config.validate()?;

        let n_freqs = config.buffer_size / 2 + 1;
        if mlp.n_freqs() != n_freqs {
            return Err(NoiseError::ModelError(format!(
                "Model n_freqs {} doesn't match config buffer_size {} (expected {})",
                mlp.n_freqs(),
                config.buffer_size,
                n_freqs
            )));
        }

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(config.buffer_size);

        let phase_gen = PhaseGenerator::new(12345);

        Ok(Self {
            config,
            mlp,
            phase_gen,
            ifft,
            time: 0.0,
            sample_counter: 0,
            prev_last_sample: 0.0,
            has_prev: false,
        })
    }

    /// Generate audio samples into the output buffer
    pub fn generate(&mut self, output: &mut [f32]) -> NoiseResult<()> {
        if output.len() != self.config.buffer_size {
            return Err(NoiseError::BufferSizeMismatch {
                expected: self.config.buffer_size,
                actual: output.len(),
            });
        }

        let n_freqs = self.config.buffer_size / 2 + 1;

        // Encode config for MLP input
        let config_vec = self.config.encode(self.time);

        // Get magnitude spectrum from MLP
        let magnitudes = self.mlp.forward(&config_vec);

        // Generate random phases
        let phases = self.phase_gen.generate(n_freqs);

        // Build complex spectrum
        let mut spectrum: Vec<Complex<f32>> = Vec::with_capacity(self.config.buffer_size);

        // DC component (real only)
        spectrum.push(Complex::new(magnitudes[0], 0.0));

        // Positive frequencies
        for i in 1..n_freqs - 1 {
            let mag = magnitudes[i];
            let phase = phases[i];
            spectrum.push(Complex::new(mag * phase.cos(), mag * phase.sin()));
        }

        // Nyquist component (real only for even-length FFT)
        if self.config.buffer_size % 2 == 0 {
            spectrum.push(Complex::new(magnitudes[n_freqs - 1], 0.0));
        } else {
            let mag = magnitudes[n_freqs - 1];
            let phase = phases[n_freqs - 1];
            spectrum.push(Complex::new(mag * phase.cos(), mag * phase.sin()));
        }

        // Negative frequencies (conjugate symmetry for real output)
        for i in (1..n_freqs - 1).rev() {
            let mag = magnitudes[i];
            let phase = phases[i];
            spectrum.push(Complex::new(mag * phase.cos(), -mag * phase.sin()));
        }

        // Perform inverse FFT
        self.ifft.process(&mut spectrum);

        // Normalize and extract real part
        let norm = 1.0 / (self.config.buffer_size as f32).sqrt();
        let mut max_abs = 0.0f32;

        for (i, sample) in spectrum.iter().enumerate().take(self.config.buffer_size) {
            output[i] = sample.re * norm;
            max_abs = max_abs.max(output[i].abs());
        }

        // Normalize to [-1, 1] if needed
        if max_abs > 1.0 {
            let scale = 0.95 / max_abs;
            for sample in output.iter_mut() {
                *sample *= scale;
            }
        }

        // Clamp to ensure bounds
        for sample in output.iter_mut() {
            *sample = sample.clamp(-1.0, 1.0);
            // Ensure no NaN or Inf
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }

        // Apply crossfade with previous buffer for seamless transitions
        // Ramp from prev_last_sample to the generated output over CROSSFADE_LEN samples
        if self.has_prev && output.len() >= CROSSFADE_LEN {
            let start_val = self.prev_last_sample;
            let end_val = output[CROSSFADE_LEN - 1];
            for i in 0..CROSSFADE_LEN {
                let t = (i + 1) as f32 / CROSSFADE_LEN as f32;
                // Smooth interpolation from prev_last to where we want to be
                let interp = start_val * (1.0 - t) + output[i] * t;
                // Also blend with a linear ramp target for extra smoothness
                let ramp_target = start_val + (end_val - start_val) * t;
                output[i] = interp * 0.7 + ramp_target * 0.3;
            }
        }

        // Store last sample of current buffer for next crossfade
        self.prev_last_sample = *output.last().unwrap_or(&0.0);
        self.has_prev = true;

        // Update time for next frame
        let samples_per_buffer = self.config.buffer_size as f64;
        let sample_rate = f64::from(self.config.sample_rate);
        self.time += samples_per_buffer / sample_rate;
        self.sample_counter += self.config.buffer_size as u64;

        Ok(())
    }

    /// Update configuration in real-time
    pub fn update_config(&mut self, config: NoiseConfig) -> NoiseResult<()> {
        config.validate()?;

        // If buffer size changed, we need to reinitialize FFT
        if config.buffer_size != self.config.buffer_size {
            let mut planner = FftPlanner::new();
            self.ifft = planner.plan_fft_inverse(config.buffer_size);

            // Also need to check if MLP dimensions match
            let n_freqs = config.buffer_size / 2 + 1;
            if self.mlp.n_freqs() != n_freqs {
                // Create new MLP with correct dimensions
                self.mlp = SpectralMLP::random_init(8, 64, n_freqs, 42);
            }
        }

        self.config = config;
        Ok(())
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> &NoiseConfig {
        &self.config
    }

    /// Get current time in seconds
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get total samples generated
    #[must_use]
    pub fn sample_counter(&self) -> u64 {
        self.sample_counter
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        self.time = 0.0;
        self.sample_counter = 0;
        self.phase_gen.reset(12345);
        self.prev_last_sample = 0.0;
        self.has_prev = false;
    }

    /// Set phase generator seed (for deterministic output)
    pub fn set_phase_seed(&mut self, seed: u64) {
        self.phase_gen.reset(seed);
    }
}

impl Iterator for NoiseGenerator {
    type Item = Vec<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buffer = vec![0.0; self.config.buffer_size];
        match self.generate(&mut buffer) {
            Ok(()) => Some(buffer),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
#[path = "generator_tests.rs"]
mod tests;
