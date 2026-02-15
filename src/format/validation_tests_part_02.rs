
// ============================================================================
// SECTION A: Format & Structural Integrity (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_a {
    use super::*;

    // Test 1: Magic bytes valid
    #[test]
    fn test_check_1_magic_valid_aprn() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(check.status.is_pass(), "APR\\0 magic should pass");
    }

    #[test]
    fn test_check_1_magic_valid_apr_unified() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "APR\\0 magic should pass (unified format)"
        );
    }

    #[test]
    fn test_check_1_magic_invalid() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"BAD!");
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(check.status.is_fail(), "Invalid magic should fail");
    }

    // Test 2: Header size fixed
    #[test]
    fn test_check_2_header_complete() {
        let data = vec![0u8; 32];
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 2)
            .unwrap();
        assert!(check.status.is_pass(), "32-byte header should pass");
    }

    #[test]
    fn test_check_2_header_too_small() {
        let data = vec![0u8; 16]; // Only 16 bytes
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 2)
            .unwrap();
        assert!(check.status.is_fail(), "16-byte header should fail");
    }

    // Test 3: Version supported
    #[test]
    fn test_check_3_version_1_0_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 1; // major
        data[5] = 0; // minor
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_pass(), "Version 1.0 should be supported");
    }

    #[test]
    fn test_check_3_version_2_0_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 2; // major
        data[5] = 0; // minor
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_pass(), "Version 2.0 should be supported");
    }

    #[test]
    fn test_check_3_version_unsupported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 3; // major (unsupported)
        data[5] = 0;
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_fail(), "Version 3.0 should fail");
    }

    // Test 11: Flags parsed
    #[test]
    fn test_check_11_known_flags_pass() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 1;
        data[8] = 0x01; // COMPRESSED flag
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 11)
            .unwrap();
        assert!(check.status.is_pass(), "Known flags should pass");
    }

    // GH-178: GGUF format support tests
    #[test]
    fn test_check_1_gguf_magic_valid() {
        // GGUF magic: [71, 71, 85, 70] = "GGUF"
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4] = 3; // GGUF version 3
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "GH-178: GGUF magic [71, 71, 85, 70] should pass"
        );
    }

    #[test]
    fn test_check_3_gguf_version_3_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&3u32.to_le_bytes()); // Version 3
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "GH-178: GGUF version 3 should be supported"
        );
    }

    #[test]
    fn test_check_3_gguf_version_2_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&2u32.to_le_bytes()); // Version 2
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "GH-178: GGUF version 2 should be supported"
        );
    }

    #[test]
    fn test_check_3_gguf_version_1_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&1u32.to_le_bytes()); // Version 1
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(
            check.status.is_pass(),
            "GH-178: GGUF version 1 should be supported"
        );
    }

    #[test]
    fn test_check_3_gguf_version_0_unsupported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&0u32.to_le_bytes()); // Version 0 (invalid)
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_fail(), "GH-178: GGUF version 0 should fail");
    }

    #[test]
    fn test_check_3_gguf_version_4_unsupported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"GGUF");
        data[4..8].copy_from_slice(&4u32.to_le_bytes()); // Version 4 (future)
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(
            check.status.is_fail(),
            "GH-178: GGUF version 4 should fail (future version)"
        );
    }
}

// ============================================================================
// SECTION B: Tensor Physics & Statistics (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_b {
    use super::*;

    // Test 26: No NaNs
    #[test]
    fn test_check_26_no_nan_pass() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let stats = TensorStats::compute("test", &data);
        assert!(stats.has_no_nan(), "Clean data should have no NaN");
    }

    #[test]
    fn test_check_26_nan_detected() {
        let data = vec![1.0f32, f32::NAN, 3.0];
        let stats = TensorStats::compute("test", &data);
        assert!(!stats.has_no_nan(), "Should detect NaN");
        assert_eq!(stats.nan_count, 1);
    }

    // Test 27: No Infs
    #[test]
    fn test_check_27_no_inf_pass() {
        let data = vec![1.0f32, 2.0, 3.0];
        let stats = TensorStats::compute("test", &data);
        assert!(stats.has_no_inf(), "Clean data should have no Inf");
    }

    #[test]
    fn test_check_27_inf_detected() {
        let data = vec![1.0f32, f32::INFINITY, f32::NEG_INFINITY];
        let stats = TensorStats::compute("test", &data);
        assert!(!stats.has_no_inf(), "Should detect Inf");
        assert_eq!(stats.inf_count, 2);
    }

    // Test 28: LayerNorm Mean in [0.5, 3.0]
    #[test]
    fn test_check_28_layernorm_mean_valid() {
        // Mean should be ~1.0 for LayerNorm weights
        let data = vec![1.0f32; 384];
        let stats = TensorStats::compute("encoder.layer_norm.weight", &data);
        assert!(
            stats.is_valid_layernorm_weight(),
            "Mean of 1.0 should be valid"
        );
    }

    #[test]
    fn test_check_28_layernorm_mean_too_high() {
        // Bug case: mean=11.0 (10x too high)
        let data = vec![11.0f32; 384];
        let stats = TensorStats::compute("decoder.layer_norm.weight", &data);
        assert!(
            !stats.is_valid_layernorm_weight(),
            "Mean of 11.0 should FAIL - this is the bug we're catching"
        );
    }

    #[test]
    fn test_check_28_layernorm_mean_too_low() {
        let data = vec![0.1f32; 384];
        let stats = TensorStats::compute("encoder.layer_norm.weight", &data);
        assert!(
            !stats.is_valid_layernorm_weight(),
            "Mean of 0.1 should fail"
        );
    }

    // Test 29: LayerNorm Bias in [-0.5, 0.5]
    #[test]
    fn test_check_29_layernorm_bias_valid() {
        let data = vec![0.0f32; 384];
        let stats = TensorStats::compute("encoder.layer_norm.bias", &data);
        assert!(
            stats.is_valid_layernorm_bias(),
            "Mean of 0.0 should be valid"
        );
    }

    #[test]
    fn test_check_29_layernorm_bias_invalid() {
        let data = vec![5.0f32; 384];
        let stats = TensorStats::compute("decoder.layer_norm.bias", &data);
        assert!(!stats.is_valid_layernorm_bias(), "Mean of 5.0 should fail");
    }

    // Test 31: Zero Tensors
    #[test]
    fn test_check_31_not_all_zeros_pass() {
        let data = vec![0.0f32, 0.0, 1.0, 0.0];
        let stats = TensorStats::compute("test", &data);
        assert!(stats.is_not_all_zeros(), "Should pass with some non-zero");
    }

    #[test]
    fn test_check_31_all_zeros_fail() {
        let data = vec![0.0f32; 100];
        let stats = TensorStats::compute("test", &data);
        assert!(!stats.is_not_all_zeros(), "All zeros should fail");
    }

    // Test 35: Attention/Linear Mean ~0
    #[test]
    fn test_check_35_linear_weight_valid() {
        let data = vec![0.01f32, -0.02, 0.03, -0.01];
        let stats = TensorStats::compute("encoder.layers.0.self_attn.q_proj.weight", &data);
        assert!(stats.is_valid_linear_weight(), "Mean ~0 should be valid");
    }

    #[test]
    fn test_check_35_linear_weight_invalid() {
        let data = vec![1.0f32; 100];
        let stats = TensorStats::compute("encoder.layers.0.fc1.weight", &data);
        assert!(!stats.is_valid_linear_weight(), "Mean of 1.0 should fail");
    }

    // Test statistics computation
    #[test]
    fn test_stats_compute_mean() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::compute("test", &data);
        assert!((stats.mean - 3.0).abs() < 0.001, "Mean should be 3.0");
    }

    #[test]
    fn test_stats_compute_std() {
        let data = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = TensorStats::compute("test", &data);
        // Mean = 5.0, Variance = 32/7 ≈ 4.57, Std ≈ 2.14
        assert!(
            (stats.std - 2.14).abs() < 0.1,
            "Std should be ~2.14, got {}",
            stats.std
        );
    }

    #[test]
    fn test_stats_empty_data() {
        let data: Vec<f32> = vec![];
        let stats = TensorStats::compute("empty", &data);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }
}
