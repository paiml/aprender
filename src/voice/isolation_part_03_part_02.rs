
    #[test]
    fn test_isolation_method_default() {
        assert_eq!(
            IsolationMethod::default(),
            IsolationMethod::SpectralSubtraction
        );
    }

    #[test]
    fn test_noise_estimation_default() {
        assert_eq!(NoiseEstimation::default(), NoiseEstimation::InitialSilence);
    }

    #[test]
    fn test_config_default() {
        let config = IsolationConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.fft_size, 512);
        assert_eq!(config.hop_length, 128);
    }

    #[test]
    fn test_config_aggressive() {
        let config = IsolationConfig::aggressive();
        assert_eq!(config.reduction_strength, 0.95);
        assert!(!config.preserve_musical_noise);
    }

    #[test]
    fn test_config_mild() {
        let config = IsolationConfig::mild();
        assert_eq!(config.reduction_strength, 0.5);
        assert!(config.preserve_musical_noise);
    }

    #[test]
    fn test_config_neural() {
        let config = IsolationConfig::neural();
        assert_eq!(config.method, IsolationMethod::NeuralMask);
    }

    #[test]
    fn test_config_realtime() {
        let config = IsolationConfig::realtime();
        assert_eq!(config.fft_size, 256);
        assert_eq!(config.hop_length, 64);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = IsolationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_sample_rate() {
        let config = IsolationConfig {
            sample_rate: 0,
            ..IsolationConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_fft_size() {
        let config = IsolationConfig {
            fft_size: 100, // Not power of 2
            ..IsolationConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_hop_length() {
        let config = IsolationConfig {
            hop_length: 0,
            ..IsolationConfig::default()
        };
        assert!(config.validate().is_err());

        let config2 = IsolationConfig {
            hop_length: 1024, // > fft_size
            fft_size: 512,
            ..IsolationConfig::default()
        };
        assert!(config2.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_strength() {
        let config = IsolationConfig {
            reduction_strength: 1.5,
            ..IsolationConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_freq_bins() {
        let config = IsolationConfig::default();
        assert_eq!(config.freq_bins(), 257); // 512/2 + 1
    }

    #[test]
    fn test_config_frame_duration() {
        let config = IsolationConfig::default();
        let expected = 512.0 / 16000.0;
        assert!((config.frame_duration_secs() - expected).abs() < 0.001);
    }

    #[test]
    fn test_isolation_result_new() {
        let result = IsolationResult::new(vec![0.1; 1600], 16000);
        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.snr_improvement_db, 0.0);
    }

    #[test]
    fn test_isolation_result_with_snr() {
        let result = IsolationResult::new(vec![0.1; 1600], 16000).with_snr(10.0, 20.0);
        assert_eq!(result.input_snr_db, 10.0);
        assert_eq!(result.output_snr_db, 20.0);
        assert_eq!(result.snr_improvement_db, 10.0);
    }

    #[test]
    fn test_noise_profile_from_frames() {
        let frames = vec![vec![1.0, 2.0, 3.0], vec![1.5, 2.5, 3.5]];
        let profile = NoiseProfile::from_frames(&frames, 16000);

        assert_eq!(profile.num_frames, 2);
        assert_eq!(profile.mean_spectrum.len(), 3);
        assert!((profile.mean_spectrum[0] - 1.25).abs() < 0.01);
    }

    #[test]
    fn test_noise_profile_empty() {
        let profile = NoiseProfile::from_frames(&[], 16000);
        assert!(!profile.is_valid());
    }

    #[test]
    fn test_spectral_subtraction_new() {
        let isolator = SpectralSubtractionIsolator::default();
        assert_eq!(isolator.over_subtraction(), 1.5);
    }

    #[test]
    fn test_spectral_subtraction_isolate() {
        let isolator = SpectralSubtractionIsolator::default();
        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();

        let result = isolator.isolate(&audio);
        assert!(result.is_ok());
        let isolated = result.expect("isolation failed");
        assert!(!isolated.audio.is_empty());
    }

    #[test]
    fn test_spectral_subtraction_empty() {
        let isolator = SpectralSubtractionIsolator::default();
        let result = isolator.isolate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectral_subtraction_estimate_noise() {
        let isolator = SpectralSubtractionIsolator::default();
        let noise: Vec<f32> = (0..4000).map(|i| 0.01 * (i as f32 * 0.005).sin()).collect();

        let profile = isolator.estimate_noise(&noise);
        assert!(profile.is_ok());
        let p = profile.expect("estimation failed");
        assert!(p.is_valid());
    }

    #[test]
    fn test_wiener_filter_new() {
        let isolator = WienerFilterIsolator::default();
        assert_eq!(
            isolator.config().method,
            IsolationMethod::SpectralSubtraction
        );
    }

    #[test]
    fn test_wiener_filter_isolate() {
        let isolator = WienerFilterIsolator::default();
        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();

        let result = isolator.isolate(&audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wiener_compute_gain() {
        // High SNR -> high gain
        let gain = WienerFilterIsolator::compute_gain(1.0, 0.1);
        assert!(gain > 0.9);

        // Low SNR -> low gain
        let gain = WienerFilterIsolator::compute_gain(0.1, 1.0);
        assert!(gain < 0.2);
    }

    #[test]
    fn test_estimate_snr() {
        // Pure sine wave should have high SNR
        let sine: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.01).sin()).collect();
        let snr = estimate_snr(&sine);
        assert!(snr > 0.0);

        // Constant value (no variation) - handle edge case
        let constant = vec![0.5f32; 8000];
        let snr_const = estimate_snr(&constant);
        assert!(snr_const.is_finite());
    }

    #[test]
    fn test_estimate_snr_empty() {
        let snr = estimate_snr(&[]);
        assert_eq!(snr, 0.0);
    }

    #[test]
    fn test_spectral_entropy() {
        // Uniform distribution = high entropy
        let uniform = vec![1.0; 100];
        let entropy_uniform = spectral_entropy(&uniform);
        assert!(entropy_uniform > 0.9);

        // Single peak = low entropy
        let mut peaked = vec![0.0; 100];
        peaked[50] = 1.0;
        let entropy_peaked = spectral_entropy(&peaked);
        assert!(entropy_peaked < 0.1);
    }

    #[test]
    fn test_spectral_entropy_empty() {
        let entropy = spectral_entropy(&[]);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_detect_voice_activity() {
        // Create audio with speech and silence
        let mut audio = vec![0.001f32; 8000]; // Silence
        for i in 2000..4000 {
            audio[i] = 0.5 * (i as f32 * 0.01).sin(); // Speech
        }

        let vad = detect_voice_activity(&audio, 1000, 0.1);

        // Should detect speech in middle frames
        assert!(vad.len() >= 4);
        assert!(!vad[0]); // Initial silence
        assert!(vad[2] || vad[3]); // Speech region
    }

    #[test]
    fn test_detect_voice_activity_empty() {
        let vad = detect_voice_activity(&[], 100, 0.1);
        assert!(vad.is_empty());
    }

    // ===== Additional coverage tests =====

    #[test]
    fn test_isolation_method_debug_clone_copy() {
        let method = IsolationMethod::WienerFilter;
        let cloned = method;
        assert_eq!(cloned, IsolationMethod::WienerFilter);

        // Exercise Debug for all variants
        let variants = [
            IsolationMethod::SpectralSubtraction,
            IsolationMethod::WienerFilter,
            IsolationMethod::NeuralMask,
            IsolationMethod::UNet,
            IsolationMethod::ConvTasNet,
        ];
        for v in &variants {
            let debug_str = format!("{v:?}");
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_noise_estimation_debug_clone_all_variants() {
        let variants = [
            NoiseEstimation::InitialSilence,
            NoiseEstimation::MinimumStatistics,
            NoiseEstimation::Adaptive,
            NoiseEstimation::FixedProfile,
        ];
        for v in &variants {
            let cloned = *v;
            assert_eq!(*v, cloned);
            let debug_str = format!("{v:?}");
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_config_validate_zero_fft_size() {
        let config = IsolationConfig {
            fft_size: 0,
            ..IsolationConfig::default()
        };
        let err = config.validate().unwrap_err();
        match err {
            VoiceError::InvalidConfig(msg) => assert!(msg.contains("fft_size")),
            other => panic!("Expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_config_validate_negative_reduction_strength() {
        let config = IsolationConfig {
            reduction_strength: -0.1,
            ..IsolationConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_spectral_floor_out_of_range() {
        let config_high = IsolationConfig {
            spectral_floor: 1.5,
            ..IsolationConfig::default()
        };
        assert!(config_high.validate().is_err());

        let config_neg = IsolationConfig {
            spectral_floor: -0.1,
            ..IsolationConfig::default()
        };
        assert!(config_neg.validate().is_err());
    }

    #[test]
    fn test_isolation_config_debug_clone() {
        let config = IsolationConfig::default();
        let cloned = config.clone();
        let debug_str = format!("{config:?}");
        assert!(!debug_str.is_empty());
        assert_eq!(cloned.sample_rate, config.sample_rate);
    }

    #[test]
    fn test_noise_profile_noise_magnitude_out_of_bounds() {
        let profile = NoiseProfile::from_frames(&[vec![1.0, 2.0, 3.0]], 16000);
        // In-bounds
        assert!((profile.noise_magnitude(0) - 1.0).abs() < f32::EPSILON);
        // Out-of-bounds should return 0.0
        assert_eq!(profile.noise_magnitude(999), 0.0);
    }

    #[test]
    fn test_noise_profile_from_frames_unequal_lengths() {
        // Frames with different lengths trigger the `if i < num_bins` guard
        let frames = vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0]];
        let profile = NoiseProfile::from_frames(&frames, 16000);
        // num_bins = first frame length = 3
        assert_eq!(profile.mean_spectrum.len(), 3);
        // bin 0: (1.0 + 10.0) / 2 = 5.5
        assert!((profile.mean_spectrum[0] - 5.5).abs() < 0.01);
        // bin 2: only first frame contributes: 3.0 / 2 = 1.5
        assert!((profile.mean_spectrum[2] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_noise_profile_debug_clone() {
        let profile = NoiseProfile::from_frames(&[vec![1.0]], 16000);
        let cloned = profile.clone();
        let debug_str = format!("{profile:?}");
        assert!(!debug_str.is_empty());
        assert_eq!(cloned.num_frames, profile.num_frames);
    }

    #[test]
    fn test_noise_profile_is_valid_true() {
        let profile = NoiseProfile::from_frames(&[vec![1.0, 2.0]], 16000);
        assert!(profile.is_valid());
    }

    #[test]
    fn test_isolation_result_with_noise_floor_directly() {
        let result = IsolationResult::new(vec![0.1; 100], 16000).with_noise_floor(vec![0.01, 0.02]);
        assert!(result.noise_floor.is_some());
        let floor = result
            .noise_floor
            .as_ref()
            .expect("should have noise floor");
        assert_eq!(floor.len(), 2);
    }

    #[test]
    fn test_isolation_result_debug_clone() {
        let result = IsolationResult::new(vec![0.5; 10], 16000).with_snr(5.0, 15.0);
        let cloned = result.clone();
        let debug_str = format!("{result:?}");
        assert!(!debug_str.is_empty());
        assert_eq!(cloned.snr_improvement_db, 10.0);
    }

    #[test]
    fn test_spectral_subtraction_with_over_subtraction_clamp() {
        let isolator = SpectralSubtractionIsolator::default();
        // Clamp low
        let low = isolator.with_over_subtraction(0.0);
        assert!((low.over_subtraction() - 0.5).abs() < f32::EPSILON);

        // Clamp high
        let high = SpectralSubtractionIsolator::default().with_over_subtraction(100.0);
        assert!((high.over_subtraction() - 5.0).abs() < f32::EPSILON);

        // Normal value
        let normal = SpectralSubtractionIsolator::default().with_over_subtraction(2.0);
        assert!((normal.over_subtraction() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spectral_subtraction_reconstruct_empty_magnitudes() {
        let isolator = SpectralSubtractionIsolator::default();
        let output = isolator.reconstruct(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_spectral_subtraction_reconstruct_empty_frame_magnitudes() {
        let isolator = SpectralSubtractionIsolator::default();
        // Frame with empty magnitudes vector
        let output = isolator.reconstruct(&[vec![]]);
        // Should produce output with avg_mag = 0.0
        assert!(!output.is_empty());
        // All output values should be zero since avg_mag is 0
        for &v in &output {
            assert!((v - 0.0).abs() < f32::EPSILON);
        }
    }
