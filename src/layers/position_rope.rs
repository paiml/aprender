
    // ==================== RoPE Tests ====================

    #[test]
    fn test_rope_new_valid_dim() {
        let rope = RoPE::new(64, 10000.0).expect("rope");
        assert_eq!(rope.dim(), 64);
        assert_eq!(rope.base(), 10000.0);
        assert_eq!(rope.inv_freq().len(), 32);
    }

    #[test]
    fn test_rope_new_small_dim() {
        let rope = RoPE::new(2, 10000.0).expect("rope");
        assert_eq!(rope.dim(), 2);
        assert_eq!(rope.inv_freq().len(), 1);
    }

    #[test]
    fn test_rope_new_zero_dim_error() {
        let result = RoPE::new(0, 10000.0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("dim must be > 0"));
    }

    #[test]
    fn test_rope_new_odd_dim_error() {
        let result = RoPE::new(3, 10000.0);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("dim must be even"));
    }

    #[test]
    fn test_rope_with_default_base() {
        let rope = RoPE::with_default_base(64).expect("rope");
        assert_eq!(rope.base(), 10000.0);
        assert_eq!(rope.dim(), 64);
    }

    #[test]
    fn test_rope_with_default_base_error() {
        let result = RoPE::with_default_base(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_inv_freq_values() {
        let rope = RoPE::new(4, 10000.0).expect("rope");
        let inv_freq = rope.inv_freq();
        // inv_freq[0] = 10000^(-2*0/4) = 10000^0 = 1.0
        // inv_freq[1] = 10000^(-2*1/4) = 10000^(-0.5) = 1/sqrt(10000) = 0.01
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);
        assert!((inv_freq[1] - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_rope_forward_identity_at_position_zero() {
        let rope = RoPE::new(4, 10000.0).expect("rope");
        let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).expect("input");
        let output = rope.forward(&input, 0).expect("output");
        // At position 0, all angles are 0, so cos=1, sin=0
        // Output should equal input for x*1 - y*0 = x, x*0 + y*1 = y
        let data = output.data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 1.0).abs() < 1e-6);
        assert!((data[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_forward_rotation() {
        let rope = RoPE::new(2, 1.0).expect("rope");
        // With base=1, inv_freq[0]=1, so angle at pos=1 is 1 radian
        let input = Tensor::from_vec(vec![2], vec![1.0, 0.0]).expect("input");
        let output = rope.forward(&input, 1).expect("output");
        let data = output.data();
        // Expected: [cos(1), sin(1)]
        assert!((data[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((data[1] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn test_rope_forward_multiple_vectors() {
        let rope = RoPE::new(2, 10000.0).expect("rope");
        let input = Tensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).expect("input");
        let output = rope.forward(&input, 0).expect("output");
        assert_eq!(output.shape(), &[2, 2]);
        // At position 0, output == input
        let data = output.data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_forward_empty_input_error() {
        let rope = RoPE::new(4, 10000.0).expect("rope");
        // Tensor::from_vec with empty shape returns error, so we test with 0-size valid tensor
        // Create a 1D tensor with wrong dimension (0 elements in last dim)
        // Actually, empty tensors are rejected by Tensor::from_vec itself
        // So we test shape validation by using a mismatched dimension
        let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("input");
        let result = rope.forward(&input, 0);
        // Expecting error due to dimension mismatch (dim 2 != expected 4)
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_forward_dim_mismatch_error() {
        let rope = RoPE::new(4, 10000.0).expect("rope");
        let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("input");
        let result = rope.forward(&input, 0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Expected last dimension 4"));
    }

    // ==================== RopeScalingType Tests ====================

    #[test]
    fn test_rope_scaling_type_default() {
        let scaling = RopeScalingType::default();
        assert_eq!(scaling, RopeScalingType::None);
    }

    #[test]
    fn test_rope_scaling_type_linear() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        if let RopeScalingType::Linear { scale } = scaling {
            assert!((scale - 2.0).abs() < 1e-6);
        } else {
            panic!("Expected Linear variant");
        }
    }

    #[test]
    fn test_rope_scaling_type_ntk() {
        let scaling = RopeScalingType::Ntk { scale: 4.0 };
        if let RopeScalingType::Ntk { scale } = scaling {
            assert!((scale - 4.0).abs() < 1e-6);
        } else {
            panic!("Expected Ntk variant");
        }
    }

    #[test]
    fn test_rope_scaling_type_dynamic_ntk() {
        let scaling = RopeScalingType::DynamicNtk {
            original_max_len: 2048,
            target_max_len: 8192,
        };
        if let RopeScalingType::DynamicNtk {
            original_max_len,
            target_max_len,
        } = scaling
        {
            assert_eq!(original_max_len, 2048);
            assert_eq!(target_max_len, 8192);
        } else {
            panic!("Expected DynamicNtk variant");
        }
    }

    #[test]
    fn test_rope_scaling_type_yarn() {
        let scaling = RopeScalingType::Yarn {
            original_max_len: 2048,
            target_max_len: 8192,
            attn_factor: 1.5,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        if let RopeScalingType::Yarn {
            original_max_len,
            target_max_len,
            attn_factor,
            ..
        } = scaling
        {
            assert_eq!(original_max_len, 2048);
            assert_eq!(target_max_len, 8192);
            assert!((attn_factor - 1.5).abs() < 1e-6);
        } else {
            panic!("Expected Yarn variant");
        }
    }

    #[test]
    fn test_rope_scaling_type_clone() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        let cloned = scaling;
        assert_eq!(scaling, cloned);
    }

    #[test]
    fn test_rope_scaling_type_copy() {
        let scaling = RopeScalingType::Ntk { scale: 4.0 };
        let copied: RopeScalingType = scaling;
        assert_eq!(copied, RopeScalingType::Ntk { scale: 4.0 });
    }

    // ==================== ScaledRoPE Tests ====================

    #[test]
    fn test_scaled_rope_new_none_scaling() {
        let scaled = ScaledRoPE::new(64, 10000.0, RopeScalingType::None).expect("scaled");
        assert_eq!(scaled.dim(), 64);
        assert_eq!(scaled.original_base(), 10000.0);
        assert_eq!(scaled.scaled_base(), 10000.0);
        assert!((scaled.mscale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_new_zero_dim_error() {
        let result = ScaledRoPE::new(0, 10000.0, RopeScalingType::None);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_rope_new_odd_dim_error() {
        let result = ScaledRoPE::new(7, 10000.0, RopeScalingType::None);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_rope_with_default_base() {
        let scaled = ScaledRoPE::with_default_base(64, RopeScalingType::None).expect("scaled");
        assert_eq!(scaled.original_base(), 10000.0);
    }

    #[test]
    fn test_scaled_rope_linear_scaling() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        // Linear scaling doesn't change base
        assert_eq!(scaled.scaled_base(), 10000.0);
        assert!((scaled.mscale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_ntk_scaling() {
        let scaling = RopeScalingType::Ntk { scale: 4.0 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        // NTK: base' = base * scale^(dim / (dim - 2))
        // 64 / 62 ≈ 1.032, so base' = 10000 * 4^1.032 ≈ 41600
        assert!(scaled.scaled_base() > 10000.0);
        assert!((scaled.mscale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_dynamic_ntk_scaling() {
        let scaling = RopeScalingType::DynamicNtk {
            original_max_len: 2048,
            target_max_len: 8192,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        // scale = 8192/2048 = 4.0, so base should increase like NTK
        assert!(scaled.scaled_base() > 10000.0);
    }

    #[test]
    fn test_scaled_rope_yarn_scaling() {
        let scaling = RopeScalingType::Yarn {
            original_max_len: 2048,
            target_max_len: 8192,
            attn_factor: 1.5,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        // YaRN should have mscale set to attn_factor
        assert!((scaled.mscale() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_yarn_auto_attn_factor() {
        let scaling = RopeScalingType::Yarn {
            original_max_len: 2048,
            target_max_len: 8192,
            attn_factor: 0.0, // auto-compute
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        // Auto-computed mscale should be > 1.0
        assert!(scaled.mscale() > 1.0);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_none() {
        let scaled = ScaledRoPE::new(64, 10000.0, RopeScalingType::None).expect("scaled");
        assert!((scaled.context_length_multiplier() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_linear() {
        let scaling = RopeScalingType::Linear { scale: 2.5 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        assert!((scaled.context_length_multiplier() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_ntk() {
        let scaling = RopeScalingType::Ntk { scale: 3.0 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        assert!((scaled.context_length_multiplier() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_dynamic_ntk() {
        let scaling = RopeScalingType::DynamicNtk {
            original_max_len: 1024,
            target_max_len: 4096,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        assert!((scaled.context_length_multiplier() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_context_length_multiplier_yarn() {
        let scaling = RopeScalingType::Yarn {
            original_max_len: 2048,
            target_max_len: 16384,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        assert!((scaled.context_length_multiplier() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_forward_none_scaling() {
        let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).expect("scaled");
        let input = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).expect("input");
        let output = scaled.forward(&input, 0).expect("output");
        let data = output.data();
        // At position 0, output == input (cos=1, sin=0)
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_forward_linear_scaling() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        let scaled = ScaledRoPE::new(2, 1.0, scaling).expect("scaled");
        let input = Tensor::from_vec(vec![2], vec![1.0, 0.0]).expect("input");
        // With linear scaling, effective_pos = pos / scale = 2 / 2 = 1
        let output = scaled.forward(&input, 2).expect("output");
        let data = output.data();
        // Expected: [cos(1), sin(1)]
        assert!((data[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((data[1] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_rope_forward_empty_input_error() {
        let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).expect("scaled");
        // Empty tensors rejected by Tensor::from_vec, so test dimension mismatch instead
        let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("input");
        let result = scaled.forward(&input, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_rope_forward_dim_mismatch_error() {
        let scaled = ScaledRoPE::new(8, 10000.0, RopeScalingType::None).expect("scaled");
        let input = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).expect("input");
        let result = scaled.forward(&input, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_scaled_rope_inv_freq() {
        let scaled = ScaledRoPE::new(4, 10000.0, RopeScalingType::None).expect("scaled");
        assert_eq!(scaled.inv_freq().len(), 2);
    }

    #[test]
    fn test_scaled_rope_scaling_getter() {
        let scaling = RopeScalingType::Linear { scale: 2.0 };
        let scaled = ScaledRoPE::new(64, 10000.0, scaling).expect("scaled");
        assert_eq!(*scaled.scaling(), RopeScalingType::Linear { scale: 2.0 });
    }

    // ==================== ALiBi Tests ====================

    #[test]
    fn test_alibi_new_power_of_two() {
        let alibi = ALiBi::new(8).expect("alibi");
        assert_eq!(alibi.num_heads(), 8);
        assert_eq!(alibi.slopes().len(), 8);
    }

    #[test]
    fn test_alibi_new_non_power_of_two() {
        let alibi = ALiBi::new(6).expect("alibi");
        assert_eq!(alibi.num_heads(), 6);
        assert_eq!(alibi.slopes().len(), 6);
    }

    #[test]
    fn test_alibi_new_single_head() {
        let alibi = ALiBi::new(1).expect("alibi");
        assert_eq!(alibi.num_heads(), 1);
        assert_eq!(alibi.slopes().len(), 1);
    }

    #[test]
    fn test_alibi_new_zero_heads_error() {
        let result = ALiBi::new(0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("num_heads must be > 0"));
    }

    #[test]
    fn test_alibi_slopes_power_of_two() {
        let alibi = ALiBi::new(4).expect("alibi");
        let slopes = alibi.slopes();
        // For n=4: m[h] = 2^(-8h/4) = 2^(-2h)
        // slopes = [2^0, 2^-2, 2^-4, 2^-6] = [1.0, 0.25, 0.0625, 0.015625]
        assert!((slopes[0] - 1.0).abs() < 1e-6);
        assert!((slopes[1] - 0.25).abs() < 1e-6);
        assert!((slopes[2] - 0.0625).abs() < 1e-6);
        assert!((slopes[3] - 0.015625).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_slopes_8_heads() {
        let alibi = ALiBi::new(8).expect("alibi");
        let slopes = alibi.slopes();
        // For n=8: m[h] = 2^(-8h/8) = 2^(-h)
        // slopes[0] = 2^0 = 1.0
        // slopes[1] = 2^-1 = 0.5
        // slopes[7] = 2^-7 = 0.0078125
        assert!((slopes[0] - 1.0).abs() < 1e-6);
        assert!((slopes[1] - 0.5).abs() < 1e-6);
        assert!((slopes[7] - 0.0078125).abs() < 1e-6);
    }
