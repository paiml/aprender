
    #[test]
    fn test_style_config_default() {
        let config = StyleConfig::default();
        assert_eq!(config.prosody_dim, 64);
        assert_eq!(config.timbre_dim, 128);
        assert_eq!(config.rhythm_dim, 32);
        assert_eq!(config.total_dim(), 224);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_style_config_prosody_only() {
        let config = StyleConfig::prosody_only();
        assert!(config.preserve_pitch_contour);
        assert!((config.style_strength - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_style_config_full_conversion() {
        let config = StyleConfig::full_conversion();
        assert!(!config.preserve_pitch_contour);
        assert!((config.style_strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_style_config_validation_prosody() {
        let mut config = StyleConfig::default();
        config.prosody_dim = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_style_config_validation_strength() {
        let mut config = StyleConfig::default();
        config.style_strength = 1.5;
        assert!(config.validate().is_err());

        config.style_strength = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_style_vector_new() {
        let style = StyleVector::new(vec![1.0, 2.0], vec![3.0, 4.0, 5.0], vec![6.0]);
        assert_eq!(style.prosody().len(), 2);
        assert_eq!(style.timbre().len(), 3);
        assert_eq!(style.rhythm().len(), 1);
        assert_eq!(style.dim(), 6);
    }

    #[test]
    fn test_style_vector_zeros() {
        let config = StyleConfig::default();
        let style = StyleVector::zeros(&config);
        assert_eq!(style.prosody().len(), config.prosody_dim);
        assert_eq!(style.timbre().len(), config.timbre_dim);
        assert_eq!(style.rhythm().len(), config.rhythm_dim);
        assert!((style.l2_norm()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_style_vector_from_flat() {
        let config = StyleConfig {
            prosody_dim: 2,
            timbre_dim: 3,
            rhythm_dim: 1,
            ..StyleConfig::default()
        };
        let flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let style = StyleVector::from_flat(&flat, &config).expect("from_flat failed");

        assert_eq!(style.prosody(), &[1.0, 2.0]);
        assert_eq!(style.timbre(), &[3.0, 4.0, 5.0]);
        assert_eq!(style.rhythm(), &[6.0]);
        assert_eq!(style.to_flat(), flat.to_vec());
    }

    #[test]
    fn test_style_vector_from_flat_wrong_size() {
        let config = StyleConfig::default();
        let flat = [1.0, 2.0, 3.0]; // Wrong size
        assert!(StyleVector::from_flat(&flat, &config).is_err());
    }

    #[test]
    fn test_style_vector_interpolate() {
        let style_a = StyleVector::new(vec![0.0, 0.0], vec![0.0, 0.0, 0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![1.0, 1.0], vec![1.0, 1.0, 1.0], vec![1.0]);

        let mid = style_a
            .interpolate(&style_b, 0.5)
            .expect("interpolate failed");
        assert!((mid.prosody()[0] - 0.5).abs() < 1e-6);
        assert!((mid.timbre()[0] - 0.5).abs() < 1e-6);
        assert!((mid.rhythm()[0] - 0.5).abs() < 1e-6);

        // Edge cases
        let start = style_a
            .interpolate(&style_b, 0.0)
            .expect("interpolate 0 failed");
        assert!((start.prosody()[0] - 0.0).abs() < 1e-6);

        let end = style_a
            .interpolate(&style_b, 1.0)
            .expect("interpolate 1 failed");
        assert!((end.prosody()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_vector_interpolate_dimension_mismatch() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![1.0, 1.0], vec![1.0], vec![1.0]);
        assert!(style_a.interpolate(&style_b, 0.5).is_err());
    }

    #[test]
    fn test_style_vector_l2_norm() {
        let style = StyleVector::new(vec![3.0], vec![4.0], vec![0.0]);
        assert!((style.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_vector_normalize() {
        let mut style = StyleVector::new(vec![3.0], vec![4.0], vec![0.0]);
        style.normalize();
        assert!((style.l2_norm() - 1.0).abs() < 1e-6);
        assert!((style.prosody()[0] - 0.6).abs() < 1e-6);
        assert!((style.timbre()[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_prosody_distance() {
        let style_a = StyleVector::new(vec![0.0, 0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![3.0, 4.0], vec![0.0], vec![0.0]);
        let dist = prosody_distance(&style_a, &style_b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_prosody_distance_mismatch() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![0.0, 0.0], vec![0.0], vec![0.0]);
        assert_eq!(prosody_distance(&style_a, &style_b), f32::MAX);
    }

    #[test]
    fn test_timbre_distance() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0, 0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![0.0], vec![3.0, 4.0], vec![0.0]);
        let dist = timbre_distance(&style_a, &style_b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_distance() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![3.0], vec![4.0], vec![0.0]);
        let dist = style_distance(&style_a, &style_b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_from_embedding() {
        let config = StyleConfig {
            prosody_dim: 64,
            timbre_dim: 64,
            rhythm_dim: 64,
            ..StyleConfig::default()
        };
        let embedding = SpeakerEmbedding::from_vec(vec![1.0; 192]);
        let style = style_from_embedding(&embedding, &config);

        assert_eq!(style.prosody().len(), 64);
        assert_eq!(style.timbre().len(), 64);
        assert_eq!(style.rhythm().len(), 64);
        assert!((style.prosody()[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_style_from_embedding_small() {
        let config = StyleConfig::default(); // 64 + 128 + 32 = 224
        let embedding = SpeakerEmbedding::from_vec(vec![1.0; 100]); // Smaller than total
        let style = style_from_embedding(&embedding, &config);

        // Should copy what's available, pad with zeros
        assert_eq!(style.dim(), config.total_dim());
    }

    #[test]
    fn test_average_styles() {
        let style_a = StyleVector::new(vec![0.0, 0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![1.0, 1.0], vec![1.0], vec![1.0]);
        let styles = vec![style_a, style_b];

        let avg = average_styles(&styles).expect("average_styles failed");
        assert!((avg.prosody()[0] - 0.5).abs() < 1e-6);
        assert!((avg.timbre()[0] - 0.5).abs() < 1e-6);
        assert!((avg.rhythm()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_average_styles_empty() {
        let styles: Vec<StyleVector> = vec![];
        assert!(average_styles(&styles).is_err());
    }

    #[test]
    fn test_average_styles_dimension_mismatch() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![1.0, 1.0], vec![1.0], vec![1.0]);
        let styles = vec![style_a, style_b];
        assert!(average_styles(&styles).is_err());
    }

    #[test]
    fn test_gst_encoder_stub() {
        let encoder = GstEncoder::default_config();
        let audio = vec![0.0_f32; 16000];
        assert!(encoder.encode(&audio).is_err());
    }

    #[test]
    fn test_gst_encoder_empty_audio() {
        let encoder = GstEncoder::default_config();
        let result = encoder.encode(&[]);
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    #[test]
    fn test_autovc_transfer_stub() {
        let transfer = AutoVcTransfer::default_config();
        let source = vec![0.0_f32; 16000];
        let style = StyleVector::zeros(&StyleConfig::default());
        assert!(transfer.transfer(&source, &style).is_err());
    }

    #[test]
    fn test_autovc_transfer_empty_source() {
        let transfer = AutoVcTransfer::default_config();
        let style = StyleVector::zeros(&StyleConfig::default());
        let result = transfer.transfer(&[], &style);
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    #[test]
    fn test_autovc_transfer_from_reference() {
        let transfer = AutoVcTransfer::default_config();
        let source = vec![0.0_f32; 16000];
        let reference = vec![0.0_f32; 16000];
        assert!(transfer
            .transfer_from_reference(&source, &reference)
            .is_err());
    }

    #[test]
    fn test_autovc_transfer_from_reference_empty() {
        let transfer = AutoVcTransfer::default_config();
        let source = vec![0.0_f32; 16000];

        let result = transfer.transfer_from_reference(&[], &source);
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));

        let result = transfer.transfer_from_reference(&source, &[]);
        assert!(matches!(result, Err(VoiceError::InvalidAudio(_))));
    }

    // ===== Additional coverage tests =====

    #[test]
    fn test_style_config_validate_timbre_zero() {
        let config = StyleConfig {
            timbre_dim: 0,
            ..StyleConfig::default()
        };
        let err = config.validate().unwrap_err();
        match err {
            VoiceError::InvalidConfig(msg) => assert!(msg.contains("timbre_dim")),
            other => panic!("Expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_style_config_validate_rhythm_zero() {
        let config = StyleConfig {
            rhythm_dim: 0,
            ..StyleConfig::default()
        };
        let err = config.validate().unwrap_err();
        match err {
            VoiceError::InvalidConfig(msg) => assert!(msg.contains("rhythm_dim")),
            other => panic!("Expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_style_config_validate_sample_rate_zero() {
        let config = StyleConfig {
            sample_rate: 0,
            ..StyleConfig::default()
        };
        let err = config.validate().unwrap_err();
        match err {
            VoiceError::InvalidConfig(msg) => assert!(msg.contains("sample_rate")),
            other => panic!("Expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_style_config_debug_clone() {
        let config = StyleConfig::default();
        let cloned = config.clone();
        let debug_str = format!("{config:?}");
        assert!(!debug_str.is_empty());
        assert_eq!(cloned.prosody_dim, config.prosody_dim);
    }

    #[test]
    fn test_style_vector_interpolate_timbre_mismatch() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![0.0], vec![0.0, 0.0], vec![0.0]);
        let result = style_a.interpolate(&style_b, 0.5);
        assert!(result.is_err());
        match result.unwrap_err() {
            VoiceError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("Expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_style_vector_interpolate_rhythm_mismatch() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![0.0], vec![0.0], vec![0.0, 0.0]);
        let result = style_a.interpolate(&style_b, 0.5);
        assert!(result.is_err());
        match result.unwrap_err() {
            VoiceError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("Expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_style_vector_interpolate_clamp_beyond_range() {
        let style_a = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        let style_b = StyleVector::new(vec![1.0], vec![1.0], vec![1.0]);

        // t < 0 should clamp to 0
        let result = style_a.interpolate(&style_b, -0.5).expect("clamp low");
        assert!((result.prosody()[0] - 0.0).abs() < 1e-6);

        // t > 1 should clamp to 1
        let result = style_a.interpolate(&style_b, 1.5).expect("clamp high");
        assert!((result.prosody()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_vector_normalize_zero_vector() {
        let mut style = StyleVector::new(vec![0.0], vec![0.0], vec![0.0]);
        style.normalize();
        // norm <= EPSILON, so vector stays zero
        assert_eq!(style.l2_norm(), 0.0);
        assert!((style.prosody()[0] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_style_vector_debug_clone() {
        let style = StyleVector::new(vec![1.0, 2.0], vec![3.0], vec![4.0]);
        let cloned = style.clone();
        let debug_str = format!("{style:?}");
        assert!(!debug_str.is_empty());
        assert_eq!(cloned.dim(), style.dim());
    }

    #[test]
    fn test_style_vector_to_flat_directly() {
        let style = StyleVector::new(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0]);
        let flat = style.to_flat();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_style_distance_dimension_mismatch() {
        let a = StyleVector::new(vec![1.0], vec![2.0], vec![3.0]);
        let b = StyleVector::new(vec![1.0, 2.0], vec![3.0], vec![4.0]);
        let dist = style_distance(&a, &b);
        assert_eq!(dist, f32::MAX);
    }

    #[test]
    fn test_style_distance_same() {
        let a = StyleVector::new(vec![1.0, 2.0], vec![3.0], vec![4.0]);
        let dist = style_distance(&a, &a);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_timbre_distance_mismatch() {
        let a = StyleVector::new(vec![0.0], vec![1.0], vec![0.0]);
        let b = StyleVector::new(vec![0.0], vec![1.0, 2.0], vec![0.0]);
        let dist = timbre_distance(&a, &b);
        assert_eq!(dist, f32::MAX);
    }

    #[test]
    fn test_timbre_distance_same() {
        let a = StyleVector::new(vec![0.0], vec![3.0, 4.0], vec![0.0]);
        let dist = timbre_distance(&a, &a);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_prosody_distance_rhythm_mismatch() {
        let a = StyleVector::new(vec![0.0], vec![0.0], vec![1.0]);
        let b = StyleVector::new(vec![0.0], vec![0.0], vec![1.0, 2.0]);
        let dist = prosody_distance(&a, &b);
        assert_eq!(dist, f32::MAX);
    }

    #[test]
    fn test_average_styles_timbre_mismatch() {
        let styles = vec![
            StyleVector::new(vec![0.0], vec![0.0], vec![0.0]),
            StyleVector::new(vec![0.0], vec![0.0, 0.0], vec![0.0]),
        ];
        let result = average_styles(&styles);
        assert!(result.is_err());
        match result.unwrap_err() {
            VoiceError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("Expected DimensionMismatch, got {other:?}"),
        }
    }
