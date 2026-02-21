
    #[test]
    fn test_spectral_subtraction_debug_clone() {
        let isolator = SpectralSubtractionIsolator::default();
        let cloned = isolator.clone();
        let debug_str = format!("{isolator:?}");
        assert!(!debug_str.is_empty());
        assert_eq!(cloned.over_subtraction(), isolator.over_subtraction());
    }

    #[test]
    fn test_spectral_subtraction_isolate_short_audio() {
        // Audio shorter than noise_frames * fft_size
        let isolator = SpectralSubtractionIsolator::default();
        let short_audio: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let result = isolator.isolate(&short_audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spectral_subtraction_isolate_with_profile_empty() {
        let isolator = SpectralSubtractionIsolator::default();
        let profile = NoiseProfile::from_frames(&[vec![0.1; 257]], 16000);
        let result = isolator.isolate_with_profile(&[], &profile);
        assert!(result.is_err());
        match result.unwrap_err() {
            VoiceError::InvalidAudio(msg) => assert!(msg.contains("Empty")),
            other => panic!("Expected InvalidAudio, got {other:?}"),
        }
    }

    #[test]
    fn test_spectral_subtraction_estimate_noise_empty() {
        let isolator = SpectralSubtractionIsolator::default();
        let result = isolator.estimate_noise(&[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            VoiceError::InvalidAudio(msg) => assert!(msg.contains("Empty")),
            other => panic!("Expected InvalidAudio, got {other:?}"),
        }
    }

    #[test]
    fn test_wiener_filter_with_smoothing() {
        let isolator = WienerFilterIsolator::default();
        // Normal value
        let filtered = isolator.with_smoothing(0.5);
        assert!((filtered.smoothing - 0.5).abs() < f32::EPSILON);

        // Clamp low
        let low = WienerFilterIsolator::default().with_smoothing(-1.0);
        assert!((low.smoothing - 0.0).abs() < f32::EPSILON);

        // Clamp high
        let high = WienerFilterIsolator::default().with_smoothing(5.0);
        assert!((high.smoothing - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wiener_compute_gain_zero_noise() {
        let gain = WienerFilterIsolator::compute_gain(1.0, 0.0);
        assert!((gain - 1.0).abs() < f32::EPSILON);

        let gain_neg = WienerFilterIsolator::compute_gain(1.0, -1.0);
        assert!((gain_neg - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wiener_compute_gain_equal_power() {
        let gain = WienerFilterIsolator::compute_gain(1.0, 1.0);
        assert!((gain - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wiener_filter_isolate_empty() {
        let isolator = WienerFilterIsolator::default();
        let result = isolator.isolate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_wiener_filter_isolate_with_profile_empty() {
        let isolator = WienerFilterIsolator::default();
        let profile = NoiseProfile::from_frames(&[vec![0.1; 257]], 16000);
        let result = isolator.isolate_with_profile(&[], &profile);
        assert!(result.is_err());
    }

    #[test]
    fn test_wiener_filter_estimate_noise_empty() {
        let isolator = WienerFilterIsolator::default();
        let result = isolator.estimate_noise(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_wiener_filter_isolate_short_audio() {
        // Audio shorter than noise_frames * fft_size
        let isolator = WienerFilterIsolator::default();
        let short_audio: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let result = isolator.isolate(&short_audio);
        assert!(result.is_ok());
        let isolated = result.expect("isolation failed");
        assert!(!isolated.audio.is_empty());
    }

    #[test]
    fn test_wiener_filter_debug_clone() {
        let isolator = WienerFilterIsolator::default();
        let cloned = isolator.clone();
        let debug_str = format!("{isolator:?}");
        assert!(!debug_str.is_empty());
        assert_eq!(cloned.smoothing, isolator.smoothing);
    }

    #[test]
    fn test_estimate_snr_short_audio() {
        // Audio shorter than frame_size (256)
        let audio: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let snr = estimate_snr(&audio);
        // num_frames = 100/100 = 1 (since frame_size is min(256, 100) = 100)
        assert!(snr.is_finite());
    }

    #[test]
    fn test_estimate_snr_zeros() {
        let audio = vec![0.0_f32; 1000];
        let snr = estimate_snr(&audio);
        // With all zeros, signal_rms and noise_rms are both 0
        assert_eq!(snr, 0.0);
    }

    #[test]
    fn test_estimate_snr_single_sample() {
        let audio = vec![1.0_f32];
        let snr = estimate_snr(&audio);
        // frame_size = min(256, 1) = 1, num_frames = 1/1 = 1
        assert!(snr.is_finite());
    }

    #[test]
    fn test_spectral_entropy_all_zeros() {
        let zeros = vec![0.0_f32; 100];
        let entropy = spectral_entropy(&zeros);
        // sum <= 0.0 => return 0.0
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_spectral_entropy_single_element() {
        let single = vec![1.0_f32];
        let entropy = spectral_entropy(&single);
        // max_entropy = ln(1) = 0.0, returns 0.0
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_spectral_entropy_negative_values() {
        // Negative values sum to zero
        let negs = vec![-1.0_f32; 10];
        let entropy = spectral_entropy(&negs);
        // sum = -10.0 which is <= 0.0
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_detect_voice_activity_zero_frame_size() {
        let audio = vec![1.0_f32; 100];
        let vad = detect_voice_activity(&audio, 0, 0.1);
        assert!(vad.is_empty());
    }

    #[test]
    fn test_detect_voice_activity_all_speech() {
        let loud_audio = vec![1.0_f32; 1000];
        let vad = detect_voice_activity(&loud_audio, 100, 0.01);
        assert!(vad.iter().all(|&v| v));
    }

    #[test]
    fn test_detect_voice_activity_all_silence() {
        let quiet_audio = vec![0.0_f32; 1000];
        let vad = detect_voice_activity(&quiet_audio, 100, 0.1);
        assert!(vad.iter().all(|&v| !v));
    }

    #[test]
    fn test_detect_voice_activity_partial_frame() {
        // Audio length not evenly divisible by frame_size
        let audio = vec![1.0_f32; 150];
        let vad = detect_voice_activity(&audio, 100, 0.01);
        // (150 + 100 - 1) / 100 = 2 frames
        assert_eq!(vad.len(), 2);
    }

    #[test]
    fn test_spectral_subtraction_config_accessor() {
        let config = IsolationConfig::aggressive();
        let isolator = SpectralSubtractionIsolator::new(config.clone());
        assert_eq!(
            isolator.config().reduction_strength,
            config.reduction_strength
        );
    }

    #[test]
    fn test_wiener_filter_config_accessor() {
        let config = IsolationConfig::mild();
        let isolator = WienerFilterIsolator::new(config.clone());
        assert_eq!(
            isolator.config().reduction_strength,
            config.reduction_strength
        );
    }

    #[test]
    fn test_wiener_filter_estimate_noise_valid() {
        let isolator = WienerFilterIsolator::default();
        let noise: Vec<f32> = (0..4000).map(|i| 0.01 * (i as f32 * 0.005).sin()).collect();
        let profile = isolator.estimate_noise(&noise);
        assert!(profile.is_ok());
        let p = profile.expect("estimation failed");
        assert!(p.is_valid());
    }

    #[test]
    fn test_wiener_filter_isolate_with_profile_valid() {
        let isolator = WienerFilterIsolator::default();
        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();
        let noise: Vec<f32> = (0..4000).map(|i| 0.01 * (i as f32 * 0.005).sin()).collect();
        let profile = isolator
            .estimate_noise(&noise)
            .expect("noise estimation failed");
        let result = isolator.isolate_with_profile(&audio, &profile);
        assert!(result.is_ok());
        let isolated = result.expect("isolation failed");
        assert_eq!(isolated.audio.len(), audio.len());
    }
