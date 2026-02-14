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
