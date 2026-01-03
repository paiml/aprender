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
mod tests {
    use super::*;

    // ========== NG1: Config validation rejects invalid ranges ==========

    #[test]
    fn test_ng1_validate_rejects_invalid_spectral_slope_too_low() {
        let mut config = NoiseConfig::brown();
        config.spectral_slope = Some(-15.0);
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("spectral_slope"));
    }

    #[test]
    fn test_ng1_validate_rejects_invalid_spectral_slope_too_high() {
        let mut config = NoiseConfig::white();
        config.spectral_slope = Some(15.0);
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_ng1_validate_rejects_invalid_texture_negative() {
        let mut config = NoiseConfig::pink();
        config.texture = -0.1;
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("texture"));
    }

    #[test]
    fn test_ng1_validate_rejects_invalid_texture_too_high() {
        let mut config = NoiseConfig::pink();
        config.texture = 1.5;
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_ng1_validate_rejects_invalid_modulation_depth() {
        let mut config = NoiseConfig::brown();
        config.modulation_depth = 2.0;
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("modulation_depth"));
    }

    #[test]
    fn test_ng1_validate_rejects_invalid_modulation_rate_too_low() {
        let mut config = NoiseConfig::brown();
        config.modulation_rate = 0.05;
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("modulation_rate"));
    }

    #[test]
    fn test_ng1_validate_rejects_invalid_modulation_rate_too_high() {
        let mut config = NoiseConfig::brown();
        config.modulation_rate = 15.0;
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_ng1_validate_rejects_zero_sample_rate() {
        let mut config = NoiseConfig::brown();
        config.sample_rate = 0;
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("sample_rate"));
    }

    #[test]
    fn test_ng1_validate_rejects_invalid_buffer_size() {
        let mut config = NoiseConfig::brown();
        config.buffer_size = 100;
        let result = config.validate();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("buffer_size"));
    }

    #[test]
    fn test_ng1_validate_accepts_valid_config() {
        let config = NoiseConfig::brown();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ng1_validate_accepts_boundary_values() {
        let config = NoiseConfig {
            noise_type: NoiseType::Custom(-12.0),
            spectral_slope: Some(12.0),
            texture: 0.0,
            modulation_depth: 1.0,
            modulation_rate: 0.1,
            sample_rate: 44100,
            buffer_size: 256,
        };
        assert!(config.validate().is_ok());
    }

    // ========== NG2: Preset configs produce expected spectral slopes ==========

    #[test]
    fn test_ng2_white_noise_slope() {
        let config = NoiseConfig::white();
        assert!((config.effective_slope() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ng2_pink_noise_slope() {
        let config = NoiseConfig::pink();
        assert!((config.effective_slope() - (-3.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ng2_brown_noise_slope() {
        let config = NoiseConfig::brown();
        assert!((config.effective_slope() - (-6.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ng2_blue_noise_slope() {
        let config = NoiseConfig::blue();
        assert!((config.effective_slope() - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ng2_violet_noise_slope() {
        let config = NoiseConfig::violet();
        assert!((config.effective_slope() - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ng2_custom_noise_slope() {
        let config = NoiseConfig::new(NoiseType::Custom(-4.5));
        assert!((config.effective_slope() - (-4.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ng2_slope_override_takes_precedence() {
        let config = NoiseConfig::brown().with_spectral_slope(-2.0).unwrap();
        assert!((config.effective_slope() - (-2.0)).abs() < f32::EPSILON);
    }

    // ========== NG3: NoiseType enum serializes/deserializes correctly ==========

    #[test]
    fn test_ng3_noise_type_serde_white() {
        let original = NoiseType::White;
        let json = serde_json::to_string(&original).unwrap();
        let restored: NoiseType = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_ng3_noise_type_serde_pink() {
        let original = NoiseType::Pink;
        let json = serde_json::to_string(&original).unwrap();
        let restored: NoiseType = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_ng3_noise_type_serde_brown() {
        let original = NoiseType::Brown;
        let json = serde_json::to_string(&original).unwrap();
        let restored: NoiseType = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_ng3_noise_type_serde_custom() {
        let original = NoiseType::Custom(-4.5);
        let json = serde_json::to_string(&original).unwrap();
        let restored: NoiseType = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
        if let NoiseType::Custom(slope) = restored {
            assert!((slope - (-4.5)).abs() < f32::EPSILON);
        } else {
            panic!("Expected Custom variant");
        }
    }

    #[test]
    fn test_ng3_noise_config_serde_roundtrip() {
        let original = NoiseConfig {
            noise_type: NoiseType::Brown,
            spectral_slope: Some(-5.0),
            texture: 0.7,
            modulation_depth: 0.3,
            modulation_rate: 2.0,
            sample_rate: 48000,
            buffer_size: 2048,
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: NoiseConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(original.noise_type, restored.noise_type);
        assert_eq!(original.spectral_slope, restored.spectral_slope);
        assert!((original.texture - restored.texture).abs() < f32::EPSILON);
        assert!((original.modulation_depth - restored.modulation_depth).abs() < f32::EPSILON);
        assert!((original.modulation_rate - restored.modulation_rate).abs() < f32::EPSILON);
        assert_eq!(original.sample_rate, restored.sample_rate);
        assert_eq!(original.buffer_size, restored.buffer_size);
    }

    #[test]
    fn test_ng3_binaural_preset_serde_roundtrip() {
        for preset in [
            BinauralPreset::Delta,
            BinauralPreset::Theta,
            BinauralPreset::Alpha,
            BinauralPreset::Beta,
            BinauralPreset::Gamma,
        ] {
            let json = serde_json::to_string(&preset).unwrap();
            let restored: BinauralPreset = serde_json::from_str(&json).unwrap();
            assert_eq!(preset, restored);
        }
    }

    // ========== Additional config tests ==========

    #[test]
    fn test_noise_type_from_name() {
        assert_eq!(NoiseType::from_name("white"), Some(NoiseType::White));
        assert_eq!(NoiseType::from_name("PINK"), Some(NoiseType::Pink));
        assert_eq!(NoiseType::from_name("Brown"), Some(NoiseType::Brown));
        assert_eq!(NoiseType::from_name("red"), Some(NoiseType::Brown));
        assert_eq!(NoiseType::from_name("blue"), Some(NoiseType::Blue));
        assert_eq!(NoiseType::from_name("violet"), Some(NoiseType::Violet));
        assert_eq!(NoiseType::from_name("purple"), Some(NoiseType::Violet));
        assert_eq!(NoiseType::from_name("unknown"), None);
    }

    #[test]
    fn test_noise_type_name() {
        assert_eq!(NoiseType::White.name(), "white");
        assert_eq!(NoiseType::Pink.name(), "pink");
        assert_eq!(NoiseType::Brown.name(), "brown");
        assert_eq!(NoiseType::Blue.name(), "blue");
        assert_eq!(NoiseType::Violet.name(), "violet");
        assert_eq!(NoiseType::Custom(1.0).name(), "custom");
    }

    #[test]
    fn test_noise_type_default() {
        let default = NoiseType::default();
        assert_eq!(default, NoiseType::Brown);
    }

    #[test]
    fn test_noise_config_default() {
        let default = NoiseConfig::default();
        assert_eq!(default.noise_type, NoiseType::Brown);
        assert!(default.validate().is_ok());
    }

    #[test]
    fn test_binaural_preset_frequencies() {
        assert!((BinauralPreset::Delta.frequency() - 2.0).abs() < f32::EPSILON);
        assert!((BinauralPreset::Theta.frequency() - 6.0).abs() < f32::EPSILON);
        assert!((BinauralPreset::Alpha.frequency() - 10.0).abs() < f32::EPSILON);
        assert!((BinauralPreset::Beta.frequency() - 20.0).abs() < f32::EPSILON);
        assert!((BinauralPreset::Gamma.frequency() - 40.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_binaural_preset_frequency_ranges() {
        assert_eq!(BinauralPreset::Delta.frequency_range(), (0.5, 4.0));
        assert_eq!(BinauralPreset::Theta.frequency_range(), (4.0, 8.0));
        assert_eq!(BinauralPreset::Alpha.frequency_range(), (8.0, 13.0));
        assert_eq!(BinauralPreset::Beta.frequency_range(), (13.0, 30.0));
        assert_eq!(BinauralPreset::Gamma.frequency_range(), (30.0, 100.0));
    }

    #[test]
    fn test_config_encode_produces_correct_length() {
        let config = NoiseConfig::brown();
        let encoded = config.encode(0.0);
        assert_eq!(encoded.len(), 8);
    }

    #[test]
    fn test_config_encode_values_normalized() {
        let config = NoiseConfig::brown();
        let encoded = config.encode(0.0);

        // Slope should be -6/12 = -0.5
        assert!((encoded[0] - (-0.5)).abs() < 0.01);
        // Texture should be 0.5
        assert!((encoded[1] - 0.5).abs() < 0.01);
        // Modulation depth should be 0.0
        assert!((encoded[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = NoiseConfig::white()
            .with_spectral_slope(-2.0)
            .unwrap()
            .with_texture(0.8)
            .unwrap()
            .with_modulation(0.5, 2.0)
            .unwrap()
            .with_sample_rate(48000)
            .unwrap()
            .with_buffer_size(2048)
            .unwrap();

        assert!(config.validate().is_ok());
        assert!((config.effective_slope() - (-2.0)).abs() < f32::EPSILON);
        assert!((config.texture - 0.8).abs() < f32::EPSILON);
        assert!((config.modulation_depth - 0.5).abs() < f32::EPSILON);
        assert!((config.modulation_rate - 2.0).abs() < f32::EPSILON);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.buffer_size, 2048);
    }

    #[test]
    fn test_config_builder_rejects_invalid() {
        assert!(NoiseConfig::white().with_spectral_slope(-20.0).is_err());
        assert!(NoiseConfig::white().with_texture(-0.5).is_err());
        assert!(NoiseConfig::white().with_modulation(2.0, 1.0).is_err());
        assert!(NoiseConfig::white().with_modulation(0.5, 20.0).is_err());
        assert!(NoiseConfig::white().with_sample_rate(0).is_err());
        assert!(NoiseConfig::white().with_buffer_size(100).is_err());
    }
}
