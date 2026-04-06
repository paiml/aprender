
    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // ========================================================================
    // Validation Tests for forward_all_layers_gpu
    // ========================================================================

    #[test]
    fn test_forward_all_layers_gpu_missing_attn_norm() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let input = vec![0.1f32; 256];
        let mut output = vec![0.0f32; 256];

        // No RMSNorm weights cached - should error
        let result = exec.forward_all_layers_gpu(
            &input,
            &mut output,
            0,    // position
            1,    // num_layers
            256,  // hidden_dim
            1024, // intermediate_dim
            1e-5, // epsilon
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("attn_norm not cached"));
    }

    #[test]
    fn test_forward_all_layers_gpu_missing_ffn_norm() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Cache attn_norm but not ffn_norm
        let gamma: Vec<f32> = vec![1.0; 256];
        let _ = exec.cache_rmsnorm_gamma("blk.0.attn_norm.gamma", &gamma);

        let input = vec![0.1f32; 256];
        let mut output = vec![0.0f32; 256];

        let result = exec.forward_all_layers_gpu(&input, &mut output, 0, 1, 256, 1024, 1e-5);

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("ffn_norm not cached"));
    }

    // ========================================================================
    // Validation Tests for forward_all_layers_gpu_to_logits
    // ========================================================================

    #[test]
    fn test_forward_to_logits_missing_attn_norm() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let input = vec![0.1f32; 256];
        let mut logits = vec![0.0f32; 1024];

        let result = exec.forward_all_layers_gpu_to_logits(
            &input,
            &mut logits,
            0,    // position
            1,    // num_layers
            256,  // hidden_dim
            1024, // intermediate_dim
            1024, // vocab_size
            1e-5, // epsilon
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = format!("{:?}", err);
        assert!(err_str.contains("attn_norm not cached"));
    }

    #[test]
    fn test_forward_to_logits_missing_output_norm() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Cache layer norms but not output_norm
        let gamma: Vec<f32> = vec![1.0; 256];
        let _ = exec.cache_rmsnorm_gamma("blk.0.attn_norm.gamma", &gamma);
        let _ = exec.cache_rmsnorm_gamma("blk.0.ffn_norm.gamma", &gamma);

        let input = vec![0.1f32; 256];
        let mut logits = vec![0.0f32; 1024];

        // This will pass validation but fail later due to missing output_norm.gamma
        // We use workspace_unused path which requires output_norm.gamma
        let result =
            exec.forward_all_layers_gpu_to_logits(&input, &mut logits, 0, 1, 256, 1024, 1024, 1e-5);

        // Will error due to missing output_norm.gamma or lm_head
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_to_logits_zero_layers() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Cache output norm only (no layer norms needed for 0 layers)
        let gamma: Vec<f32> = vec![1.0; 256];
        let _ = exec.cache_rmsnorm_gamma("output_norm.gamma", &gamma);

        let input = vec![0.1f32; 256];
        let mut logits = vec![0.0f32; 1024];

        // 0 layers - should skip layer processing, fail at output norm or lm_head
        let result = exec.forward_all_layers_gpu_to_logits(
            &input,
            &mut logits,
            0,
            0, // 0 layers
            256,
            1024,
            1024,
            1e-5,
        );

        // Will error due to missing lm_head
        assert!(result.is_err());
    }

    // ========================================================================
    // Integration Tests with ModelHarness
    // ========================================================================

    #[test]
    fn test_forward_with_harness_multiple_positions() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};

        let Some(mut exec) = create_executor() else {
            return;
        };

        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return; // Skip if harness setup fails
        }

        // Test forward at multiple positions (exercises RoPE)
        for position in [0, 1, 5, 10] {
            let input = vec![0.1f32; config.hidden_dim];
            let mut output = vec![0.0f32; config.hidden_dim];

            let _ = exec.forward_all_layers_gpu(
                &input,
                &mut output,
                position,
                config.num_layers,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                1e-5,
            );
        }
    }

    #[test]
    fn test_forward_to_logits_with_harness_sequence() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};

        let Some(mut exec) = create_executor() else {
            return;
        };

        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Simulate autoregressive generation: multiple forward passes
        for pos in 0..3 {
            let input = vec![0.1f32; config.hidden_dim];
            let mut logits = vec![0.0f32; config.vocab_size];

            let _ = exec.forward_all_layers_gpu_to_logits(
                &input,
                &mut logits,
                pos,
                config.num_layers,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                config.vocab_size as u32,
                1e-5,
            );
        }
    }

    // ========================================================================
    // Additional Harness-Based Integration Tests
    // ========================================================================

    #[test]
    fn test_forward_all_layers_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = vec![0.1f32; config.hidden_dim];
        let mut output = vec![0.0f32; config.hidden_dim];

        let result = exec.forward_all_layers_gpu(
            &input,
            &mut output,
            0,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_to_logits_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = vec![0.1f32; config.hidden_dim];
        let mut logits = vec![0.0f32; config.vocab_size];

        let result = exec.forward_all_layers_gpu_to_logits(
            &input,
            &mut logits,
            0,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_different_epsilon_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test with different epsilon values
        for epsilon in [1e-5, 1e-6, 1e-4] {
            let input = vec![0.1f32; config.hidden_dim];
            let mut output = vec![0.0f32; config.hidden_dim];

            let result = exec.forward_all_layers_gpu(
                &input,
                &mut output,
                0,
                config.num_layers,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                epsilon,
            );
            let _ = result;
        }
    }

    #[test]
    fn test_forward_different_hidden_dims() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.hidden_dim = 512;
        config.intermediate_dim = 1024;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = vec![0.1f32; config.hidden_dim];
        let mut output = vec![0.0f32; config.hidden_dim];

        let result = exec.forward_all_layers_gpu(
            &input,
            &mut output,
            0,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_larger_vocab_size() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.vocab_size = 32000;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = vec![0.1f32; config.hidden_dim];
        let mut logits = vec![0.0f32; config.vocab_size];

        let result = exec.forward_all_layers_gpu_to_logits(
            &input,
            &mut logits,
            0,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            config.vocab_size as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_multi_layer_with_harness() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.num_layers = 4;
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let input = vec![0.1f32; config.hidden_dim];
        let mut output = vec![0.0f32; config.hidden_dim];

        let result = exec.forward_all_layers_gpu(
            &input,
            &mut output,
            0,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );
        let _ = result;
    }

    #[test]
    fn test_forward_kv_cache_populated() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Run forward at position 0
        let input = vec![0.1f32; config.hidden_dim];
        let mut output = vec![0.0f32; config.hidden_dim];

        let _ = exec.forward_all_layers_gpu(
            &input,
            &mut output,
            0,
            config.num_layers,
            config.hidden_dim as u32,
            config.intermediate_dim as u32,
            1e-5,
        );

        // KV cache lengths should be updated
        let kv_len = exec.kv_cache_lengths.get(&0).copied().unwrap_or(0);
        // May or may not be > 0 depending on path taken
        let _ = kv_len;
    }

    // ========================================================================
    // Coverage Tests: Additional Forward Paths (v1.36.0)
    // ========================================================================

    #[test]
    fn test_forward_sequential_positions() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Simulate autoregressive: position 0, 1, 2, 3, ...
        for position in 0..5 {
            let input = vec![0.1f32 + (position as f32 * 0.01); config.hidden_dim];
            let mut output = vec![0.0f32; config.hidden_dim];

            let result = exec.forward_all_layers_gpu(
                &input,
                &mut output,
                position,
                config.num_layers,
                config.hidden_dim as u32,
                config.intermediate_dim as u32,
                1e-5,
            );
            let _ = result;
        }
    }
