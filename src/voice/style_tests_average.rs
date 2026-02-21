
    #[test]
    fn test_average_styles_rhythm_mismatch() {
        let styles = vec![
            StyleVector::new(vec![0.0], vec![0.0], vec![0.0]),
            StyleVector::new(vec![0.0], vec![0.0], vec![0.0, 0.0]),
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

    #[test]
    fn test_average_styles_single() {
        let styles = vec![StyleVector::new(vec![3.0, 4.0], vec![5.0], vec![6.0])];
        let avg = average_styles(&styles).expect("single avg");
        assert!((avg.prosody()[0] - 3.0).abs() < 1e-6);
        assert!((avg.timbre()[0] - 5.0).abs() < 1e-6);
        assert!((avg.rhythm()[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_gst_encoder_new_direct() {
        let config = StyleConfig {
            prosody_dim: 32,
            ..StyleConfig::default()
        };
        let encoder = GstEncoder::new(config);
        assert_eq!(encoder.config().prosody_dim, 32);
    }

    #[test]
    fn test_gst_encoder_debug() {
        let encoder = GstEncoder::default_config();
        let debug_str = format!("{encoder:?}");
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_gst_encoder_config_accessor() {
        let encoder = GstEncoder::default_config();
        let config = encoder.config();
        assert_eq!(config.prosody_dim, 64);
        assert_eq!(config.timbre_dim, 128);
    }

    #[test]
    fn test_autovc_new_direct() {
        let config = StyleConfig {
            timbre_dim: 64,
            ..StyleConfig::default()
        };
        let transfer = AutoVcTransfer::new(config);
        assert_eq!(transfer.config().timbre_dim, 64);
    }

    #[test]
    fn test_autovc_debug() {
        let transfer = AutoVcTransfer::default_config();
        let debug_str = format!("{transfer:?}");
        assert!(!debug_str.is_empty());
    }

    #[test]
    fn test_autovc_config_accessor() {
        let transfer = AutoVcTransfer::default_config();
        let config = transfer.config();
        assert_eq!(config.prosody_dim, 64);
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn test_style_from_embedding_zero_length() {
        let config = StyleConfig::default();
        let embedding = SpeakerEmbedding::from_vec(vec![]);
        let style = style_from_embedding(&embedding, &config);
        // All zeros since embedding is empty
        assert_eq!(style.dim(), config.total_dim());
        for &v in style.prosody() {
            assert!((v - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_style_from_embedding_exact_match() {
        let config = StyleConfig {
            prosody_dim: 2,
            timbre_dim: 2,
            rhythm_dim: 2,
            ..StyleConfig::default()
        };
        let embedding = SpeakerEmbedding::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let style = style_from_embedding(&embedding, &config);
        assert_eq!(style.prosody(), &[1.0, 2.0]);
        assert_eq!(style.timbre(), &[3.0, 4.0]);
        assert_eq!(style.rhythm(), &[5.0, 6.0]);
    }

    #[test]
    fn test_style_from_embedding_larger_than_needed() {
        let config = StyleConfig {
            prosody_dim: 2,
            timbre_dim: 2,
            rhythm_dim: 2,
            ..StyleConfig::default()
        };
        let embedding = SpeakerEmbedding::from_vec(vec![1.0; 100]);
        let style = style_from_embedding(&embedding, &config);
        assert_eq!(style.prosody().len(), 2);
        assert_eq!(style.timbre().len(), 2);
        assert_eq!(style.rhythm().len(), 2);
    }

    #[test]
    fn test_style_vector_from_flat_round_trip() {
        let config = StyleConfig {
            prosody_dim: 3,
            timbre_dim: 4,
            rhythm_dim: 2,
            ..StyleConfig::default()
        };
        let original = StyleVector::new(
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0, 7.0],
            vec![8.0, 9.0],
        );
        let flat = original.to_flat();
        let reconstructed = StyleVector::from_flat(&flat, &config).expect("from_flat");
        assert_eq!(reconstructed.prosody(), original.prosody());
        assert_eq!(reconstructed.timbre(), original.timbre());
        assert_eq!(reconstructed.rhythm(), original.rhythm());
    }

    #[test]
    fn test_prosody_distance_zero() {
        let a = StyleVector::new(vec![1.0, 2.0], vec![0.0], vec![3.0]);
        let dist = prosody_distance(&a, &a);
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_style_config_total_dim() {
        let config = StyleConfig {
            prosody_dim: 10,
            timbre_dim: 20,
            rhythm_dim: 30,
            ..StyleConfig::default()
        };
        assert_eq!(config.total_dim(), 60);
    }

    #[test]
    fn test_style_vector_l2_norm_all_components() {
        // Norm includes all three component vectors
        let style = StyleVector::new(vec![1.0, 0.0], vec![0.0, 2.0], vec![0.0]);
        let expected_norm = (1.0_f32 + 4.0).sqrt(); // sqrt(5)
        assert!((style.l2_norm() - expected_norm).abs() < 1e-6);
    }

    #[test]
    fn test_average_styles_three() {
        let styles = vec![
            StyleVector::new(vec![3.0], vec![6.0], vec![9.0]),
            StyleVector::new(vec![0.0], vec![0.0], vec![0.0]),
            StyleVector::new(vec![0.0], vec![0.0], vec![0.0]),
        ];
        let avg = average_styles(&styles).expect("three avg");
        assert!((avg.prosody()[0] - 1.0).abs() < 1e-6);
        assert!((avg.timbre()[0] - 2.0).abs() < 1e-6);
        assert!((avg.rhythm()[0] - 3.0).abs() < 1e-6);
    }
