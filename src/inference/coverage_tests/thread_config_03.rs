
// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_thread_config_mass_creation() {
    let configs: Vec<ThreadConfig> = (1..=100).map(|i| ThreadConfig::new(i * 2, i)).collect();

    assert_eq!(configs.len(), 100);

    for (i, config) in configs.iter().enumerate() {
        let expected_batch = (i + 1) * 2;
        let expected_decode = i + 1;
        assert_eq!(config.n_threads_batch, expected_batch);
        assert_eq!(config.n_threads_decode, expected_decode);
    }
}

#[test]
fn test_inference_mode_mass_iteration() {
    let modes: Vec<InferenceMode> = (0..1000)
        .map(|i| {
            if i % 3 == 0 {
                InferenceMode::Prefill
            } else {
                InferenceMode::Decode
            }
        })
        .collect();

    let prefill_count = modes.iter().filter(|m| m.is_prefill()).count();
    let decode_count = modes.iter().filter(|m| m.is_decode()).count();

    // 334 prefill (0, 3, 6, ..., 999) and 666 decode
    assert_eq!(prefill_count, 334);
    assert_eq!(decode_count, 666);
    assert_eq!(prefill_count + decode_count, 1000);
}

// ============================================================================
// Property-like Tests
// ============================================================================

#[test]
fn test_thread_config_clamping_property() {
    // For any input, output should be >= 1
    for batch in [0, 1, 5, 100, usize::MAX] {
        for decode in [0, 1, 5, 100, usize::MAX] {
            let config = ThreadConfig::new(batch, decode);
            assert!(config.n_threads_batch >= 1);
            assert!(config.n_threads_decode >= 1);
        }
    }
}

#[test]
fn test_inference_mode_exclusive_property() {
    // For any mode, exactly one of is_prefill/is_decode is true
    for mode in [InferenceMode::Prefill, InferenceMode::Decode] {
        let prefill = mode.is_prefill();
        let decode = mode.is_decode();

        // XOR: exactly one should be true
        assert!(
            prefill ^ decode,
            "Mode should be exactly one of prefill or decode"
        );
    }
}

#[test]
fn test_threads_for_returns_correct_field() {
    // threads_for(true) should return n_threads_batch
    // threads_for(false) should return n_threads_decode
    for batch in [1, 10, 100] {
        for decode in [1, 5, 50] {
            let config = ThreadConfig::new(batch, decode);
            assert_eq!(config.threads_for(true), config.n_threads_batch);
            assert_eq!(config.threads_for(false), config.n_threads_decode);
        }
    }
}
