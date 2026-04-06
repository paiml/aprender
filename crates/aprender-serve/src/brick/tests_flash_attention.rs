//! Additional brick tests - Part 03
//!
//! Focuses on FlashAttention, ActivationQuant, CudaGraph, and reporting tests.
//! This file kept under 400 lines per PMAT file health rules.

#[cfg(test)]
mod tests {
    use crate::brick::*;

    // =========================================================================
    // FlashAttentionBrick Tests
    // =========================================================================

    #[test]
    fn flash_attention_can_run() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        assert!(brick.can_run());
    }

    #[test]
    fn flash_attention_cannot_run_zero_heads() {
        let brick = FlashAttentionBrick::new(0, 2, 64);
        assert!(!brick.can_run());
    }

    #[test]
    fn flash_attention_cannot_run_zero_dim() {
        let brick = FlashAttentionBrick::new(8, 2, 0);
        assert!(!brick.can_run());
    }

    #[test]
    fn flash_attention_forward_invalid_query_len() {
        let brick = FlashAttentionBrick::new(4, 2, 8);
        let query = vec![1.0f32; 16]; // Wrong size (should be 32)
        let keys = vec![0.5f32; 64];
        let values = vec![0.25f32; 64];
        let result = brick.forward(&query, &keys, &values, 4);
        assert!(result.is_err());
    }

    #[test]
    fn flash_attention_forward_invalid_keys_len() {
        let brick = FlashAttentionBrick::new(4, 2, 8);
        let query = vec![1.0f32; 32];
        let keys = vec![0.5f32; 32]; // Wrong size
        let values = vec![0.25f32; 64];
        let result = brick.forward(&query, &keys, &values, 4);
        assert!(result.is_err());
    }

    #[test]
    fn flash_attention_arithmetic_intensity() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        let ai = brick.arithmetic_intensity(512);
        assert!(ai > 0.0);
    }

    #[test]
    fn flash_attention_group_size() {
        let brick = FlashAttentionBrick::new(8, 2, 64);
        assert_eq!(brick.group_size(), 4);
    }

    #[test]
    fn flash_attention_with_budget() {
        let brick = FlashAttentionBrick::new(8, 2, 64).with_budget(TokenBudget::from_latency(10.0));
        assert!((brick.budget().us_per_token - 10.0).abs() < 0.001);
    }

    #[test]
    fn flash_attention_forward_zero_dimension() {
        let brick = FlashAttentionBrick::new(0, 0, 0);
        let result = brick.forward(&[], &[], &[], 0);
        assert!(result.is_err());
    }

    // =========================================================================
    // ActivationQuantBrick Tests
    // =========================================================================

    #[test]
    fn activation_quant_quantize_invalid_input_len() {
        let brick = ActivationQuantBrick::new(32);
        let input = vec![1.0f32; 64]; // Wrong size
        let result = brick.quantize(&input);
        assert!(result.is_err());
    }

    #[test]
    fn activation_quant_dequantize_invalid_len() {
        let brick = ActivationQuantBrick::new(32);
        let quants = vec![0i8; 64]; // Wrong size
        let scales = vec![1.0f32];
        let result = brick.dequantize(&quants, &scales);
        assert!(result.is_err());
    }

    #[test]
    fn activation_quant_zero_dimension() {
        let brick = ActivationQuantBrick::new(0);
        assert!(!brick.can_run());
        let result = brick.quantize(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn activation_quant_measure_error_near_zero() {
        let brick = ActivationQuantBrick::new(32);
        let input = vec![1e-12f32; 32]; // Very small values
        let (quants, scales) = brick.quantize(&input).unwrap();
        let error = brick.measure_error(&input, &quants, &scales).unwrap();
        // Error should be 0 or very small for near-zero input
        assert!(error < 0.1);
    }

    #[test]
    fn activation_quant_with_custom_budget() {
        let brick = ActivationQuantBrick::new(64).with_budget(TokenBudget::from_latency(2.0));
        assert!((brick.budget().us_per_token - 2.0).abs() < 0.001);
    }

    #[test]
    fn activation_quant_per_channel() {
        let brick = ActivationQuantBrick::with_per_channel(64);
        assert!(brick.per_channel);
    }

    // =========================================================================
    // CudaGraphBrick Tests
    // =========================================================================

    #[test]
    fn cuda_graph_replay_not_captured() {
        let brick = CudaGraphBrick::new(28, 1536);
        let result = brick.replay();
        assert!(result.is_err());
        if let Err(BrickError::ComputeError(msg)) = result {
            assert!(msg.contains("not captured"));
        }
    }

    #[test]
    fn cuda_graph_set_captured() {
        let mut brick = CudaGraphBrick::new(28, 1536);
        assert!(!brick.captured);
        brick.set_captured(true);
        assert!(brick.captured);
        assert!(brick.can_replay());
    }

    #[test]
    fn cuda_graph_can_run() {
        let brick = CudaGraphBrick::new(28, 1536);
        assert!(brick.can_run());
    }

    #[test]
    fn cuda_graph_cannot_run_zero_layers() {
        let brick = CudaGraphBrick::new(0, 1536);
        assert!(!brick.can_run());
    }

    #[test]
    fn cuda_graph_cannot_run_zero_hidden() {
        let brick = CudaGraphBrick::new(28, 0);
        assert!(!brick.can_run());
    }

    #[test]
    fn cuda_graph_with_budget() {
        let brick = CudaGraphBrick::new(28, 1536).with_budget(TokenBudget::from_latency(15.0));
        assert!((brick.budget().us_per_token - 15.0).abs() < 0.001);
    }

    // =========================================================================
    // LayerTiming Tests
    // =========================================================================

    #[test]
    fn layer_timing_bottleneck_all_zero() {
        let timing = LayerTiming::default();
        let (name, us) = timing.bottleneck();
        // With all zeros, any brick could be returned
        assert!(!name.is_empty());
        assert!(us == 0.0);
    }

    #[test]
    fn layer_timing_bottleneck_tie() {
        let timing = LayerTiming {
            attn_norm_us: 10.0,
            qkv_us: 10.0, // Tie
            rope_us: 5.0,
            attention_us: 5.0,
            o_proj_us: 5.0,
            ffn_norm_us: 5.0,
            ffn_us: 5.0,
            total_us: 45.0,
        };
        let (name, us) = timing.bottleneck();
        // Either attn_norm or qkv could be returned
        assert!(name == "attn_norm" || name == "qkv");
        assert!((us - 10.0).abs() < 0.001);
    }

    #[test]
    fn layer_timing_bottleneck_ffn() {
        let timing = LayerTiming {
            attn_norm_us: 1.0,
            qkv_us: 2.0,
            rope_us: 1.0,
            attention_us: 5.0,
            o_proj_us: 2.0,
            ffn_norm_us: 1.0,
            ffn_us: 15.0, // Clear bottleneck
            total_us: 27.0,
        };
        let (name, us) = timing.bottleneck();
        assert_eq!(name, "ffn");
        assert!((us - 15.0).abs() < 0.001);
    }

    // =========================================================================
    // TransformerLayerBrick Tests
    // =========================================================================

    #[test]
    fn transformer_layer_from_config() {
        let layer = TransformerLayerBrick::from_config(
            0,    // layer_idx
            896,  // hidden_dim
            14,   // num_heads
            2,    // num_kv_heads
            4864, // intermediate_dim
            1e-5, // eps
            1e6,  // rope_theta
            2,    // rope_type
        );
        assert_eq!(layer.layer_idx, 0);
        assert!(layer.last_timing.is_none());
    }

    #[test]
    fn transformer_layer_verify() {
        let layer = TransformerLayerBrick::from_config(0, 896, 14, 2, 4864, 1e-5, 1e6, 2);
        let v = layer.verify();
        assert!(v.is_valid);
    }

    #[test]
    fn transformer_layer_total_budget() {
        let layer = TransformerLayerBrick::from_config(0, 896, 14, 2, 4864, 1e-5, 1e6, 2);
        let total = layer.total_budget_us();
        // Should be sum of all component budgets
        assert!(total > 0.0);
    }

    // =========================================================================
    // BenchmarkConfig and BenchmarkReport Tests
    // =========================================================================

    #[test]
    fn benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup, 10);
        assert_eq!(config.samples, 100);
        assert!((config.max_cv - 0.05).abs() < 0.001);
    }

    #[test]
    fn benchmark_report_display() {
        let report = BenchmarkReport {
            brick_name: "test_brick".to_string(),
            mean_us: 50.0,
            std_us: 5.0,
            cv: 0.1,
            p50_us: 48.0,
            p99_us: 65.0,
            tokens_per_sec: 20000.0,
            budget_us: 100.0,
            budget_met: true,
            statistically_valid: true,
        };
        let display = format!("{}", report);
        assert!(display.contains("test_brick"));
        assert!(display.contains("PASS"));
    }

    #[test]
    fn benchmark_report_display_fail() {
        let report = BenchmarkReport {
            brick_name: "test_brick".to_string(),
            mean_us: 150.0,
            std_us: 15.0,
            cv: 0.1,
            p50_us: 145.0,
            p99_us: 180.0,
            tokens_per_sec: 6667.0,
            budget_us: 100.0,
            budget_met: false,
            statistically_valid: true,
        };
        let display = format!("{}", report);
        assert!(display.contains("FAIL"));
    }

    // =========================================================================
    // BottleneckReport Tests
    // =========================================================================

    #[test]
    fn bottleneck_report_display() {
        let report = BottleneckReport {
            layer_idx: 5,
            brick_name: "ffn",
            actual_us: 20.0,
            budget_us: 15.0,
            gap_factor: 1.33,
        };
        let display = format!("{}", report);
        assert!(display.contains("ffn"));
        assert!(display.contains("layer 5"));
        assert!(display.contains("20"));
        assert!(display.contains("15"));
        assert!(display.contains("1.33"));
    }

    // =========================================================================
    // RopeBrick Tests
    // =========================================================================

    #[test]
    fn rope_brick_creation() {
        let brick = RopeBrick::new(64, 8, 10000.0, 0);
        assert_eq!(brick.head_dim, 64);
        assert_eq!(brick.num_heads, 8);
        assert!((brick.theta - 10000.0).abs() < 0.1);
        assert_eq!(brick.rope_type, 0);
    }

    #[test]
    fn rope_brick_with_budget() {
        let brick = RopeBrick::new(64, 8, 10000.0, 0).with_budget(TokenBudget::from_latency(5.0));
        assert!((brick.budget().us_per_token - 5.0).abs() < 0.001);
    }

    // =========================================================================
    // FfnBrick Tests
    // =========================================================================

    #[test]
    fn ffn_brick_creation() {
        let brick = FfnBrick::new(1024, 4096);
        assert_eq!(brick.hidden_dim, 1024);
        assert_eq!(brick.intermediate_dim, 4096);
    }

    #[test]
    fn ffn_brick_with_budget() {
        let brick = FfnBrick::new(1024, 4096).with_budget(TokenBudget::from_latency(20.0));
        assert!((brick.budget().us_per_token - 20.0).abs() < 0.001);
    }

    // =========================================================================
    // OProjBrick Tests
    // =========================================================================

    #[test]
    fn oproj_brick_creation() {
        let brick = OProjBrick::new(512, 128);
        assert_eq!(brick.in_dim, 512);
        assert_eq!(brick.out_dim, 128);
    }

    #[test]
    fn oproj_brick_with_budget() {
        let brick = OProjBrick::new(512, 128).with_budget(TokenBudget::from_latency(5.0));
        assert!((brick.budget().us_per_token - 5.0).abs() < 0.001);
    }
}
