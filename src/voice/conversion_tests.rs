
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_mode_default() {
        assert_eq!(ConversionMode::default(), ConversionMode::AnyToAny);
    }

    #[test]
    fn test_bottleneck_type_default() {
        assert_eq!(BottleneckType::default(), BottleneckType::AutoEncoder);
    }

    #[test]
    fn test_config_default() {
        let config = VoiceConversionConfig::default();
        assert_eq!(config.mode, ConversionMode::AnyToAny);
        assert_eq!(config.speaker_dim, 256);
        assert_eq!(config.sample_rate, 16000);
        assert!(config.convert_f0);
    }

    #[test]
    fn test_config_autovc() {
        let config = VoiceConversionConfig::autovc();
        assert_eq!(config.mode, ConversionMode::AnyToAny);
        assert_eq!(config.bottleneck, BottleneckType::AutoEncoder);
    }

    #[test]
    fn test_config_stargan_vc() {
        let config = VoiceConversionConfig::stargan_vc();
        assert_eq!(config.mode, ConversionMode::ManyToOne);
        assert_eq!(config.speaker_dim, 64);
    }

    #[test]
    fn test_config_ppg_based() {
        let config = VoiceConversionConfig::ppg_based();
        assert_eq!(config.bottleneck, BottleneckType::Ppg);
        assert_eq!(config.content_dim, 144);
    }

    #[test]
    fn test_config_realtime() {
        let config = VoiceConversionConfig::realtime();
        assert_eq!(config.frame_shift_ms, 5);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = VoiceConversionConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_speaker_dim() {
        let config = VoiceConversionConfig {
            speaker_dim: 0,
            ..VoiceConversionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_content_dim() {
        let config = VoiceConversionConfig {
            content_dim: 0,
            ..VoiceConversionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_pitch_ratio() {
        let config = VoiceConversionConfig {
            pitch_ratio: -1.0,
            ..VoiceConversionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_formant() {
        let config = VoiceConversionConfig {
            formant_preservation: 1.5,
            ..VoiceConversionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_frame_samples() {
        let config = VoiceConversionConfig::default();
        // 16000 Hz * 10 ms / 1000 = 160 samples
        assert_eq!(config.frame_samples(), 160);
    }

    #[test]
    fn test_conversion_result_new() {
        let audio = vec![0.1, 0.2, 0.3];
        let result = ConversionResult::new(audio.clone(), 16000);
        assert_eq!(result.audio, audio);
        assert_eq!(result.sample_rate, 16000);
        assert!(result.duration_secs > 0.0);
    }

    #[test]
    fn test_conversion_result_with_metrics() {
        let result = ConversionResult::new(vec![0.1; 1600], 16000).with_metrics(0.9, 0.1, 0.85);
        assert_eq!(result.confidence, 0.9);
        assert_eq!(result.source_similarity, 0.1);
        assert_eq!(result.target_similarity, 0.85);
    }

    #[test]
    fn test_autovc_converter_new() {
        let converter = AutoVcConverter::default();
        assert_eq!(converter.config().mode, ConversionMode::AnyToAny);
        assert_eq!(converter.downsample_factor(), 32);
    }

    #[test]
    fn test_autovc_extract_content() {
        let converter = AutoVcConverter::default();
        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();
        let content = converter.extract_content(&audio);
        assert!(content.is_ok());
        let features = content.expect("extraction failed");
        assert!(!features.is_empty());
    }

    #[test]
    fn test_autovc_extract_content_empty() {
        let converter = AutoVcConverter::default();
        let result = converter.extract_content(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_autovc_convert() {
        let converter = AutoVcConverter::new(VoiceConversionConfig {
            speaker_dim: 192,
            ..VoiceConversionConfig::default()
        });

        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();
        let target = SpeakerEmbedding::zeros(192);

        let result = converter.convert(&audio, None, &target);
        assert!(result.is_ok());
        let conversion = result.expect("conversion failed");
        assert!(!conversion.audio.is_empty());
    }

    #[test]
    fn test_autovc_convert_empty() {
        let converter = AutoVcConverter::default();
        let target = SpeakerEmbedding::zeros(256);
        let result = converter.convert(&[], None, &target);
        assert!(result.is_err());
    }

    #[test]
    fn test_autovc_convert_dim_mismatch() {
        let converter = AutoVcConverter::default(); // expects 256
        let target = SpeakerEmbedding::zeros(128); // wrong dimension
        let audio = vec![0.1f32; 1600];
        let result = converter.convert(&audio, None, &target);
        assert!(result.is_err());
    }

    #[test]
    fn test_ppg_converter_new() {
        let converter = PpgConverter::default();
        assert_eq!(converter.num_phonemes(), 144);
    }

    #[test]
    fn test_ppg_extract_content() {
        let converter = PpgConverter::default();
        let audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.002).sin()).collect();
        let ppg = converter.extract_content(&audio);
        assert!(ppg.is_ok());
        let features = ppg.expect("extraction failed");
        assert!(!features.is_empty());
        // Each frame should have 144 phoneme posteriors
        assert_eq!(features[0].len(), 144);
    }

    #[test]
    fn test_ppg_convert() {
        let converter = PpgConverter::new(
            VoiceConversionConfig {
                speaker_dim: 192,
                ..VoiceConversionConfig::ppg_based()
            },
            144,
        );

        let audio: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.002).sin()).collect();
        let target = SpeakerEmbedding::zeros(192);

        let result = converter.convert(&audio, None, &target);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convert_f0_basic() {
        let f0 = vec![200.0, 220.0, 0.0, 180.0]; // Hz, with unvoiced
        let converted = convert_f0(&f0, 200.0, 20.0, 150.0, 15.0);

        assert_eq!(converted.len(), f0.len());
        assert_eq!(converted[2], 0.0); // Unvoiced preserved
        assert!(converted[0] < f0[0]); // Pitch lowered
    }

    #[test]
    fn test_convert_f0_empty() {
        let converted = convert_f0(&[], 200.0, 20.0, 150.0, 15.0);
        assert!(converted.is_empty());
    }

    #[test]
    fn test_f0_statistics() {
        let f0 = vec![200.0, 220.0, 0.0, 180.0];
        let (mean, std) = f0_statistics(&f0);

        // Mean of [200, 220, 180] = 200
        assert!((mean - 200.0).abs() < 1.0);
        assert!(std > 0.0);
    }

    #[test]
    fn test_f0_statistics_all_unvoiced() {
        let f0 = vec![0.0, 0.0, 0.0];
        let (mean, std) = f0_statistics(&f0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_ratio_to_semitones() {
        // Octave up = 12 semitones
        assert!((ratio_to_semitones(2.0) - 12.0).abs() < 0.01);
        // No change = 0 semitones
        assert!((ratio_to_semitones(1.0) - 0.0).abs() < 0.01);
        // Octave down = -12 semitones
        assert!((ratio_to_semitones(0.5) - (-12.0)).abs() < 0.01);
    }

    #[test]
    fn test_semitones_to_ratio() {
        // 12 semitones = octave up (2x)
        assert!((semitones_to_ratio(12.0) - 2.0).abs() < 0.01);
        // 0 semitones = no change
        assert!((semitones_to_ratio(0.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_conversion_quality() {
        let source = SpeakerEmbedding::from_vec(vec![1.0, 0.0, 0.0]);
        let target = SpeakerEmbedding::from_vec(vec![0.0, 1.0, 0.0]);
        let converted = SpeakerEmbedding::from_vec(vec![0.1, 0.9, 0.1]);

        let (src_sim, tgt_sim, score) = conversion_quality(&source, &target, &converted);

        // Should be more similar to target than source
        assert!(tgt_sim > src_sim);
        assert!(score > 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
