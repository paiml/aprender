//! Noise configuration and type definitions

use serde::{Deserialize, Serialize};

use super::{NoiseError, NoiseResult};

/// Predefined noise color profiles based on spectral slope
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NoiseType {
    /// Flat frequency spectrum (0 dB/octave)
    White,
    /// -3 dB/octave roll-off (1/f noise)
    Pink,
    /// -6 dB/octave roll-off (1/f² noise, Brownian motion)
    Brown,
    /// +3 dB/octave (differentiated pink)
    Blue,
    /// +6 dB/octave (differentiated brown)
    Violet,
    /// User-defined spectral slope in dB/octave
    Custom(f32),
}

impl NoiseType {
    /// Returns the spectral slope in dB/octave
    #[must_use]
    pub fn spectral_slope(&self) -> f32 {
        match self {
            NoiseType::White => 0.0,
            NoiseType::Pink => -3.0,
            NoiseType::Brown => -6.0,
            NoiseType::Blue => 3.0,
            NoiseType::Violet => 6.0,
            NoiseType::Custom(slope) => *slope,
        }
    }

    /// Parse from string name
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "white" => Some(NoiseType::White),
            "pink" => Some(NoiseType::Pink),
            "brown" | "brownian" | "red" => Some(NoiseType::Brown),
            "blue" | "azure" => Some(NoiseType::Blue),
            "violet" | "purple" => Some(NoiseType::Violet),
            _ => None,
        }
    }

    /// Get the name of this noise type
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            NoiseType::White => "white",
            NoiseType::Pink => "pink",
            NoiseType::Brown => "brown",
            NoiseType::Blue => "blue",
            NoiseType::Violet => "violet",
            NoiseType::Custom(_) => "custom",
        }
    }
}

impl Default for NoiseType {
    fn default() -> Self {
        NoiseType::Brown // Best for sleep
    }
}

/// Brainwave entrainment frequency presets for binaural beats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinauralPreset {
    /// Delta waves: 0.5-4 Hz (deep sleep, healing)
    Delta,
    /// Theta waves: 4-8 Hz (meditation, creativity)
    Theta,
    /// Alpha waves: 8-13 Hz (relaxation, calm focus)
    Alpha,
    /// Beta waves: 13-30 Hz (active thinking, focus)
    Beta,
    /// Gamma waves: 30-100 Hz (high cognition, peak awareness)
    Gamma,
}

impl BinauralPreset {
    /// Returns the center frequency for this brainwave band in Hz
    #[must_use]
    pub fn frequency(&self) -> f32 {
        match self {
            BinauralPreset::Delta => 2.0,
            BinauralPreset::Theta => 6.0,
            BinauralPreset::Alpha => 10.0,
            BinauralPreset::Beta => 20.0,
            BinauralPreset::Gamma => 40.0,
        }
    }

    /// Returns the frequency range for this brainwave band
    #[must_use]
    pub fn frequency_range(&self) -> (f32, f32) {
        match self {
            BinauralPreset::Delta => (0.5, 4.0),
            BinauralPreset::Theta => (4.0, 8.0),
            BinauralPreset::Alpha => (8.0, 13.0),
            BinauralPreset::Beta => (13.0, 30.0),
            BinauralPreset::Gamma => (30.0, 100.0),
        }
    }
}

/// Configuration for noise generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Base noise type/color
    pub noise_type: NoiseType,

    /// Spectral slope override in dB/octave (-12.0 to +12.0)
    /// If None, uses noise_type.spectral_slope()
    pub spectral_slope: Option<f32>,

    /// Texture/grain control (0.0 = smooth, 1.0 = grainy)
    pub texture: f32,

    /// LFO modulation depth (0.0 = none, 1.0 = full)
    pub modulation_depth: f32,

    /// LFO modulation rate in Hz (0.1 to 10.0)
    pub modulation_rate: f32,

    /// Sample rate in Hz (typically 44100 or 48000)
    pub sample_rate: u32,

    /// FFT/buffer size (256, 512, 1024, 2048)
    pub buffer_size: usize,
}

impl NoiseConfig {
    /// Create a new configuration with validation
    pub fn new(noise_type: NoiseType) -> Self {
        Self {
            noise_type,
            spectral_slope: None,
            texture: 0.5,
            modulation_depth: 0.0,
            modulation_rate: 1.0,
            sample_rate: 44100,
            buffer_size: 1024,
        }
    }

    /// Preset for brown noise (optimal for sleep)
    #[must_use]
    pub fn brown() -> Self {
        Self::new(NoiseType::Brown)
    }

    /// Preset for white noise
    #[must_use]
    pub fn white() -> Self {
        Self::new(NoiseType::White)
    }

    /// Preset for pink noise
    #[must_use]
    pub fn pink() -> Self {
        Self::new(NoiseType::Pink)
    }

    /// Preset for blue noise
    #[must_use]
    pub fn blue() -> Self {
        Self::new(NoiseType::Blue)
    }

    /// Preset for violet noise
    #[must_use]
    pub fn violet() -> Self {
        Self::new(NoiseType::Violet)
    }

    /// Get the effective spectral slope (override or from type)
    #[must_use]
    pub fn effective_slope(&self) -> f32 {
        self.spectral_slope
            .unwrap_or_else(|| self.noise_type.spectral_slope())
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> NoiseResult<()> {
        // Validate spectral slope if overridden
        if let Some(slope) = self.spectral_slope {
            if !(-12.0..=12.0).contains(&slope) {
                return Err(NoiseError::InvalidConfig(format!(
                    "spectral_slope must be in [-12, 12], got {slope}"
                )));
            }
        }

        // Validate texture
        if !(0.0..=1.0).contains(&self.texture) {
            return Err(NoiseError::InvalidConfig(
                format!("texture must be in [0, 1], got {}", self.texture), // Can't use inline format with self
            ));
        }

        // Validate modulation depth
        if !(0.0..=1.0).contains(&self.modulation_depth) {
            return Err(NoiseError::InvalidConfig(format!(
                "modulation_depth must be in [0, 1], got {}",
                self.modulation_depth
            )));
        }

        // Validate modulation rate
        if !(0.1..=10.0).contains(&self.modulation_rate) {
            return Err(NoiseError::InvalidConfig(format!(
                "modulation_rate must be in [0.1, 10], got {}",
                self.modulation_rate
            )));
        }

        // Validate sample rate
        if self.sample_rate == 0 {
            return Err(NoiseError::InvalidConfig(
                "sample_rate must be > 0".to_string(),
            ));
        }

        // Validate buffer size
        if ![256, 512, 1024, 2048].contains(&self.buffer_size) {
            return Err(NoiseError::InvalidConfig(format!(
                "buffer_size must be 256, 512, 1024, or 2048, got {}",
                self.buffer_size
            )));
        }

        Ok(())
    }

    /// Set spectral slope override with validation
    pub fn with_spectral_slope(mut self, slope: f32) -> NoiseResult<Self> {
        if !(-12.0..=12.0).contains(&slope) {
            return Err(NoiseError::InvalidConfig(format!(
                "spectral_slope must be in [-12, 12], got {}",
                slope
            )));
        }
        self.spectral_slope = Some(slope);
        Ok(self)
    }

    /// Set texture with validation
    pub fn with_texture(mut self, texture: f32) -> NoiseResult<Self> {
        if !(0.0..=1.0).contains(&texture) {
            return Err(NoiseError::InvalidConfig(format!(
                "texture must be in [0, 1], got {}",
                texture
            )));
        }
        self.texture = texture;
        Ok(self)
    }

    /// Set modulation with validation
    pub fn with_modulation(mut self, depth: f32, rate: f32) -> NoiseResult<Self> {
        if !(0.0..=1.0).contains(&depth) {
            return Err(NoiseError::InvalidConfig(format!(
                "modulation_depth must be in [0, 1], got {}",
                depth
            )));
        }
        if !(0.1..=10.0).contains(&rate) {
            return Err(NoiseError::InvalidConfig(format!(
                "modulation_rate must be in [0.1, 10], got {}",
                rate
            )));
        }
        self.modulation_depth = depth;
        self.modulation_rate = rate;
        Ok(self)
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> NoiseResult<Self> {
        if sample_rate == 0 {
            return Err(NoiseError::InvalidConfig(
                "sample_rate must be > 0".to_string(),
            ));
        }
        self.sample_rate = sample_rate;
        Ok(self)
    }

    /// Set buffer size with validation
    pub fn with_buffer_size(mut self, buffer_size: usize) -> NoiseResult<Self> {
        if ![256, 512, 1024, 2048].contains(&buffer_size) {
            return Err(NoiseError::InvalidConfig(format!(
                "buffer_size must be 256, 512, 1024, or 2048, got {}",
                buffer_size
            )));
        }
        self.buffer_size = buffer_size;
        Ok(self)
    }

    /// Encode config as normalized f32 vector for MLP input
    #[must_use]
    pub fn encode(&self, time: f64) -> Vec<f32> {
        let slope = self.effective_slope();
        let lfo_phase = time * f64::from(self.modulation_rate) * std::f64::consts::TAU;

        vec![
            slope / 12.0,                      // Normalized slope [-1, 1]
            self.texture,                      // Already [0, 1]
            self.modulation_depth,             // Already [0, 1]
            self.modulation_rate / 10.0,       // Normalized rate [0, 1]
            lfo_phase.sin() as f32,            // LFO sin component
            lfo_phase.cos() as f32,            // LFO cos component
            self.buffer_size as f32 / 2048.0,  // Normalized buffer size
            self.sample_rate as f32 / 48000.0, // Normalized sample rate
        ]
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self::brown()
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
