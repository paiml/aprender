//! Falsification proptest for ValidatedModelConfig
//!
//! Contract: model-metadata-bounds-v1.yaml
//! Tests: FALSIFY-VMC-001-prop through FALSIFY-VMC-004-prop
//!
//! Popperian falsification: each test attempts to BREAK the validation invariants
//! of `ValidatedModelConfig`. If a test fails, the validation logic has a gap.

use proptest::prelude::*;
use realizar::gguf::{ArchConstraints, GGUFConfig, ValidatedModelConfig};

/// Construct a known-valid base config (LLaMA-7B-like).
fn valid_base_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "llama".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        num_kv_heads: 8,
        vocab_size: 32000,
        intermediate_dim: 11008,
        context_length: 4096,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: Some(128_000),
        eos_token_id: Some(128_001),
    }
}

proptest! {
    // ====================================================================
    // FALSIFY-VMC-001-prop: Non-divisible hidden_dim rejected
    //
    // Prediction: When hidden_dim is NOT divisible by num_heads (32),
    // validation MUST reject the config. Off-by-one hidden_dim values
    // would produce fractional head_dim, corrupting attention computation.
    // ====================================================================
    #[test]
    fn falsify_vmc_001_prop_non_divisible_hidden_dim_rejected(
        // Generate hidden_dim in [1, 65536] that is NOT a multiple of 32
        raw_hidden in 1_usize..=65_536_usize,
    ) {
        // Skip values that happen to be valid (divisible by 32)
        prop_assume!(raw_hidden % 32 != 0);

        let mut cfg = valid_base_config();
        cfg.hidden_dim = raw_hidden;
        // explicit_head_dim = None forces the divisibility check
        cfg.explicit_head_dim = None;

        let result = ValidatedModelConfig::validate(cfg);
        prop_assert!(
            result.is_err(),
            "FALSIFY-VMC-001-prop: hidden_dim={raw_hidden} (not divisible by num_heads=32) \
             should be rejected, but was accepted"
        );
    }

    // ====================================================================
    // FALSIFY-VMC-002-prop: Zero dimensions always rejected
    //
    // Prediction: Setting ANY critical dimension to 0 MUST produce a
    // validation error. Zero dimensions cause division-by-zero panics
    // in attention/FFN kernels.
    //
    // Fields tested: hidden_dim, num_layers, vocab_size, num_heads,
    // num_kv_heads, intermediate_dim.
    // ====================================================================
    #[test]
    fn falsify_vmc_002_prop_zero_dimension_rejected(
        // Select which field to zero out (0..5 maps to 6 fields)
        field_idx in 0_usize..6_usize,
    ) {
        let mut cfg = valid_base_config();

        match field_idx {
            0 => {
                cfg.hidden_dim = 0;
                // Avoid head_dim divisibility hitting first
                cfg.num_heads = 1;
                cfg.num_kv_heads = 1;
            }
            1 => cfg.num_layers = 0,
            2 => cfg.vocab_size = 0,
            3 => {
                cfg.num_heads = 0;
                cfg.num_kv_heads = 0;
            }
            4 => cfg.num_kv_heads = 0,
            5 => cfg.intermediate_dim = 0,
            _ => unreachable!(),
        }

        let result = ValidatedModelConfig::validate(cfg);
        let field_name = ["hidden_dim", "num_layers", "vocab_size", "num_heads", "num_kv_heads", "intermediate_dim"][field_idx];
        prop_assert!(
            result.is_err(),
            "FALSIFY-VMC-002-prop: {field_name}=0 should be rejected, but was accepted"
        );
    }

    // ====================================================================
    // FALSIFY-VMC-003-prop: Off-by-one hidden_dim rejected
    //
    // Prediction: hidden_dim = (num_heads * k) + 1 for any k > 0 MUST be
    // rejected. The "+1" creates a fractional head_dim that would silently
    // produce wrong tensor shapes.
    //
    // This is distinct from FALSIFY-VMC-001 because it specifically targets
    // the off-by-one boundary (the most common real-world misconfiguration).
    // ====================================================================
    #[test]
    fn falsify_vmc_003_prop_off_by_one_hidden_dim_rejected(
        // k: base multiplier for num_heads. hidden_dim = 32*k + 1
        k in 1_usize..=2048_usize,
    ) {
        let hidden_dim = 32 * k + 1;
        // Skip if hidden_dim exceeds the upper bound
        prop_assume!(hidden_dim <= 65_536);

        let mut cfg = valid_base_config();
        cfg.hidden_dim = hidden_dim;
        cfg.explicit_head_dim = None;

        let result = ValidatedModelConfig::validate(cfg);
        prop_assert!(
            result.is_err(),
            "FALSIFY-VMC-003-prop: hidden_dim={hidden_dim} (32*{k}+1) should be rejected"
        );
    }

    // ====================================================================
    // FALSIFY-VMC-004-prop: Valid configs always accepted
    //
    // Prediction: When hidden_dim is a multiple of num_heads, num_heads is
    // a multiple of num_kv_heads, all dimensions are > 0 and within bounds,
    // the config MUST be accepted. This tests the absence of false rejections.
    // ====================================================================
    #[test]
    fn falsify_vmc_004_prop_valid_configs_accepted(
        // head_dim: per-head dimension (powers of 2 are typical)
        head_dim_log2 in 4_u32..=8_u32,      // 16..256
        // num_heads: must be > 0
        num_heads_raw in 1_usize..=128_usize,
        // GQA ratio: num_heads / num_kv_heads
        gqa_ratio_log2 in 0_u32..=3_u32,     // 1, 2, 4, 8
        // num_layers
        num_layers in 1_usize..=128_usize,
        // vocab_size
        vocab_size in 1_usize..=500_000_usize,
        // FFN expansion multiplier (2x..8x)
        ffn_mult in 2_usize..=8_usize,
        // context_length
        context_length in 0_usize..=2_097_152_usize,
    ) {
        let head_dim = 1_usize << head_dim_log2;
        let gqa_ratio = 1_usize << gqa_ratio_log2;

        // num_heads must be a multiple of gqa_ratio
        let num_heads = if num_heads_raw < gqa_ratio {
            gqa_ratio
        } else {
            (num_heads_raw / gqa_ratio) * gqa_ratio
        };
        let num_kv_heads = num_heads / gqa_ratio;

        let hidden_dim = num_heads * head_dim;

        // Skip if hidden_dim exceeds upper bound
        prop_assume!(hidden_dim <= 65_536);
        prop_assume!(num_heads <= 256);
        prop_assume!((1..=256).contains(&num_kv_heads));

        let intermediate_dim = hidden_dim * ffn_mult;
        prop_assume!(intermediate_dim <= 262_144);

        let cfg = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: ArchConstraints::from_architecture("llama"),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
        };

        let result = ValidatedModelConfig::validate(cfg);
        prop_assert!(
            result.is_ok(),
            "FALSIFY-VMC-004-prop: valid config rejected: hidden_dim={hidden_dim}, \
             num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, \
             num_layers={num_layers}, vocab_size={vocab_size}, \
             intermediate_dim={intermediate_dim}: {:?}",
            result.unwrap_err()
        );
    }
}
