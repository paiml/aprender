#[cfg(test)]
mod tests {
    use crate::brick::*;

    // T030: BrickAssertion bounds check - edge values
    #[test]
    fn t030_assertion_bounds_edge() {
        let assertion = BrickAssertion::bounds(-1.0, 1.0);

        // Exact boundary values should pass
        assert!(assertion.check_f32(&[-1.0, 0.0, 1.0], true).is_ok());

        // Just outside should fail (below)
        let result = assertion.check_f32(&[-1.01], true);
        assert!(result.is_err());

        // Just outside should fail (above)
        let result = assertion.check_f32(&[1.01], true);
        assert!(result.is_err());
    }

    // T031: FlashAttentionBrick with single sequence position
    #[test]
    fn t031_flash_attention_single_pos() {
        let brick = FlashAttentionBrick::new(2, 1, 4);
        let seq_len = 1;

        let query = vec![1.0f32; 2 * 4]; // 2 heads * 4 dim
        let keys = vec![1.0f32; 4]; // 1 seq * 1 kv_head * 4 dim
        let values = vec![1.0f32; 4];

        let output = brick
            .forward(&query, &keys, &values, seq_len)
            .expect("forward failed");
        assert_eq!(output.len(), 8);

        // With uniform inputs, output should be uniform
        for &v in &output {
            assert!(!v.is_nan());
            assert!(v.is_finite());
        }
    }

    // T032: CoalescedDp4aBrick FLOPS calculation
    #[test]
    #[cfg(feature = "cuda")]
    fn t032_coalesced_dp4a_flops() {
        let brick = CoalescedDp4aBrick::new(1024, 256);
        let flops = brick.flops();

        // GEMV: 2 * K * N
        assert_eq!(flops, 2 * 1024 * 256);
    }

    // T033: FusedFfnBrick with negative values
    #[test]
    #[cfg(feature = "cuda")]
    fn t033_fused_ffn_negative_values() {
        let brick = FusedFfnBrick::new(2, 4);

        let input = vec![-1.0f32; 2];
        let gate_proj = vec![1.0f32; 8];
        let up_proj = vec![1.0f32; 8];
        let down_proj = vec![1.0f32; 8];

        let output = brick
            .forward(&input, &gate_proj, &up_proj, &down_proj)
            .expect("forward failed");

        // SiLU of negative value should be small but not NaN
        for &v in &output {
            assert!(!v.is_nan());
            assert!(v.is_finite());
        }
    }

    // T034: CoalescedDp4aBrick with all-zero weights
    #[test]
    #[cfg(feature = "cuda")]
    fn t034_coalesced_dp4a_zero_weights() {
        let brick = CoalescedDp4aBrick::new(256, 4);

        let input_q8 = vec![1i8; 256];
        let input_scale = 1.0;
        // All 0x88 = nibbles (8,8) centered to (0,0) - effectively zero weights
        let weights_q4 = vec![0x88u8; 512];
        let weight_scales = vec![0.1f32; 4];

        let output = brick
            .forward(&input_q8, input_scale, &weights_q4, &weight_scales)
            .expect("forward failed");

        // With centered zero weights, output should be zero
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    // T035: ActivationQuantBrick per-channel mode
    #[test]
    fn t035_activation_quant_per_channel() {
        let per_tensor = ActivationQuantBrick::new(1024);
        let per_channel = ActivationQuantBrick::with_per_channel(1024);

        assert!(!per_tensor.per_channel);
        assert!(per_channel.per_channel);

        // Per-channel should have different (tighter) budget
        assert!(per_channel.budget().us_per_token > per_tensor.budget().us_per_token);
    }

    // T036: TransformerLayerBrick total_budget calculation
    #[test]
    fn t036_transformer_layer_total_budget() {
        let layer = TransformerLayerBrick::from_config(0, 64, 4, 2, 256, 1e-5, 10000.0, 0);

        let total = layer.total_budget_us();

        // Total should equal sum of components
        let sum = layer.attn_norm.budget().us_per_token
            + layer.qkv.budget().us_per_token
            + layer.rope.budget().us_per_token
            + layer.attention.budget().us_per_token
            + layer.o_proj.budget().us_per_token
            + layer.ffn_norm.budget().us_per_token
            + layer.ffn.budget().us_per_token;

        assert!((total - sum).abs() < 0.001);
    }

    // T037: BenchmarkReport display format
    #[test]
    fn t037_benchmark_report_display() {
        let report = BenchmarkReport {
            brick_name: "test_brick".to_string(),
            mean_us: 50.5,
            std_us: 5.2,
            cv: 0.103,
            p50_us: 50.0,
            p99_us: 62.0,
            tokens_per_sec: 19802.0,
            budget_us: 100.0,
            budget_met: true,
            statistically_valid: true,
        };

        let display = format!("{}", report);
        assert!(display.contains("test_brick"));
        assert!(display.contains("50.5"));
        assert!(display.contains("PASS"));
    }

    // T038: BenchmarkReport display FAIL case
    #[test]
    fn t038_benchmark_report_display_fail() {
        let report = BenchmarkReport {
            brick_name: "test".to_string(),
            mean_us: 150.0,
            std_us: 10.0,
            cv: 0.067,
            p50_us: 150.0,
            p99_us: 180.0,
            tokens_per_sec: 6667.0,
            budget_us: 100.0,
            budget_met: false,
            statistically_valid: true,
        };

        let display = format!("{}", report);
        assert!(display.contains("FAIL"));
    }

    // T039: FlashAttentionBrick GQA with different group sizes
    #[test]
    fn t039_flash_attention_gqa_groups() {
        // 8 query heads, 2 kv heads = group size 4
        let brick1 = FlashAttentionBrick::new(8, 2, 64);
        assert_eq!(brick1.group_size(), 4);

        // 8 query heads, 8 kv heads = group size 1 (MHA)
        let brick2 = FlashAttentionBrick::new(8, 8, 64);
        assert_eq!(brick2.group_size(), 1);

        // 8 query heads, 1 kv head = group size 8 (MQA)
        let brick3 = FlashAttentionBrick::new(8, 1, 64);
        assert_eq!(brick3.group_size(), 8);
    }

    // T040: ActivationQuantBrick large dimension
    #[test]
    fn t040_activation_quant_large_dim() {
        let brick = ActivationQuantBrick::new(4096);
        let input: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) / 100.0).collect();

        let (quants, scales) = brick.quantize(&input).expect("quantize failed");
        assert_eq!(quants.len(), 4096);
        assert_eq!(scales.len(), 128); // 4096 / 32 = 128 blocks

        let output = brick
            .dequantize(&quants, &scales)
            .expect("dequantize failed");
        assert_eq!(output.len(), 4096);

        // Verify roundtrip error is reasonable
        let error = brick
            .measure_error(&input, &quants, &scales)
            .expect("measure failed");
        assert!(error < 0.05, "Error {} should be < 5%", error);
    }

    // T041: QkvBrick with custom budget
    #[test]
    fn t041_qkv_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(20.0);
        let brick = QkvBrick::new(64, 64, 32, 32).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 20.0).abs() < 0.001);
    }

    // T042: RopeBrick with custom budget
    #[test]
    fn t042_rope_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(5.0);
        let brick = RopeBrick::new(64, 8, 10000.0, 0).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 5.0).abs() < 0.001);
    }

    // T043: AttentionBrick with custom budget
    #[test]
    fn t043_attention_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(25.0);
        let brick = AttentionBrick::new(8, 2, 64).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 25.0).abs() < 0.001);
    }

    // T044: OProjBrick with custom budget
    #[test]
    fn t044_oproj_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(8.0);
        let brick = OProjBrick::new(512, 64).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 8.0).abs() < 0.001);
    }

    // T045: FfnBrick with custom budget
    #[test]
    fn t045_ffn_brick_with_budget() {
        let custom_budget = TokenBudget::from_latency(30.0);
        let brick = FfnBrick::new(64, 256).with_budget(custom_budget);

        assert!((brick.budget().us_per_token - 30.0).abs() < 0.001);
    }

    // T047: TokenBudget edge case - very small latency
    #[test]
    fn t047_token_budget_small_latency() {
        let budget = TokenBudget::from_latency(0.001); // 1 nanosecond
        assert_eq!(budget.tokens_per_sec, 1_000_000_000.0);
        assert!(budget.is_met(0.0005));
        assert!(!budget.is_met(0.002));
    }

    // T048: TokenBudget edge case - very high throughput
    #[test]
    fn t048_token_budget_high_throughput() {
        let budget = TokenBudget::from_throughput(1_000_000.0); // 1M tokens/sec
        assert_eq!(budget.us_per_token, 1.0);
    }

    // T049: BrickVerification accumulate multiple failures
    #[test]
    fn t049_brick_verification_multiple_failures() {
        let mut v = BrickVerification::pass();
        assert!(v.is_valid);

        v.add("check1", false, "failed 1");
        assert!(!v.is_valid);

        v.add("check2", true, "passed");
        assert!(!v.is_valid); // Still invalid

        v.add("check3", false, "failed 3");
        assert!(!v.is_valid);

        assert_eq!(v.results.len(), 3);
    }

    // T050: ComputeBrick verify with zero budget
    #[test]
    fn t050_compute_brick_verify_zero_budget() {
        // Create a brick and manually set zero budget
        let mut brick = RmsNormBrick::new(vec![1.0; 4], 1e-5);
        brick.budget = TokenBudget::from_latency(0.0);

        // Verify should fail due to zero budget
        // Note: This creates a 0/0 situation in throughput
        // but the verify() method checks us_per_token <= 0.0
        let verification = brick.verify();
        assert!(!verification.is_valid);
    }

    // T051: FlashAttentionBrick with varying sequence lengths
    #[test]
    fn t051_flash_attention_varying_seq() {
        let brick = FlashAttentionBrick::new(4, 2, 8);

        for seq_len in [1, 2, 4, 8, 16, 32] {
            let query = vec![1.0f32; 4 * 8];
            let keys = vec![0.5f32; seq_len * 2 * 8];
            let values = vec![0.25f32; seq_len * 2 * 8];

            let output = brick
                .forward(&query, &keys, &values, seq_len)
                .unwrap_or_else(|_| panic!("forward failed for seq_len={}", seq_len));

            assert_eq!(output.len(), 4 * 8);
            assert!(output.iter().all(|&v| !v.is_nan()));
        }
    }
include!("tests_f001_brick_f002.rs");
include!("tests_f056_fused_f057.rs");
include!("tests_r006_fused.rs");
include!("tests_f083_timing_f084.rs");
include!("tests_layer_timing.rs");
}
