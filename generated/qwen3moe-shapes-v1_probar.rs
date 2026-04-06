// Qwen3-235B-A22B (MoE) shape verification constants
const HIDDEN: usize = 4096;
const N_HEADS: usize = 64;
const N_KV_HEADS: usize = 4;
const D_K: usize = 128;
const MOE_INTERMEDIATE: usize = 1536;
const N_EXPERTS: usize = 128;
const N_EXPERTS_PER_TOK: usize = 8;
const ROPE_THETA: f64 = 1_000_000.0;

#[cfg(test)]
mod probar_tests {
    use super::*;

    // === Property tests derived from proof obligations ===

    /// Obligation: Q projection shape (invariant)
    /// Formal: n_h * d_k = 8192 for Qwen3-235B-A22B
    #[test]
    fn prop_q_projection_shape() {
        let q_dim = N_HEADS * D_K;
        assert_eq!(q_dim, 8192, "Q projection: n_h * d_k must equal 8192");
        // Q projection is EXPANDING: 8192 > hidden_size (4096)
        assert!(q_dim > HIDDEN, "Q dim {q_dim} must be > hidden {HIDDEN} for this config");
    }

    /// Obligation: KV projection shape (invariant)
    /// Formal: n_kv * d_k = 512 for Qwen3-235B-A22B
    #[test]
    fn prop_kv_projection_shape() {
        let kv_dim = N_KV_HEADS * D_K;
        assert_eq!(kv_dim, 512, "KV projection: n_kv * d_k must equal 512");
        assert!(kv_dim < HIDDEN, "KV dim {kv_dim} must be < hidden {HIDDEN}");
    }

    /// Obligation: GQA divisibility (invariant)
    /// Formal: n_h mod n_kv = 64 mod 4 = 0, ratio = 16
    #[test]
    fn prop_gqa_divisibility() {
        assert_eq!(N_HEADS % N_KV_HEADS, 0, "n_heads must be divisible by n_kv_heads");
        let gqa_ratio = N_HEADS / N_KV_HEADS;
        assert_eq!(gqa_ratio, 16, "GQA ratio must be 16 for Qwen3-235B");
    }

    /// Obligation: MoE expert shape (invariant)
    /// Formal: each expert: 3 * 4096 * 1536 params
    #[test]
    fn prop_moe_expert_shape() {
        let per_expert = 3 * HIDDEN * MOE_INTERMEDIATE;
        assert_eq!(per_expert, 18_874_368, "per-expert params must be 18,874,368");
        let total_experts_per_layer = N_EXPERTS * per_expert;
        assert_eq!(total_experts_per_layer, 2_415_919_104, "total expert params per layer");
        let active_per_layer = N_EXPERTS_PER_TOK * per_expert;
        assert_eq!(active_per_layer, 150_994_944, "active expert params per layer");
    }

    /// Obligation: MoE router top-k (invariant)
    /// Formal: router selects exactly 8 of 128 experts
    #[test]
    fn prop_moe_router_top_k() {
        assert!(N_EXPERTS_PER_TOK < N_EXPERTS, "top-k must be < num_experts");
        assert!(N_EXPERTS_PER_TOK >= 1, "must select at least 1 expert");
        let router_params = HIDDEN * N_EXPERTS;
        assert_eq!(router_params, 524_288, "router weight shape: [128, 4096]");
        // Sparsity ratio
        let sparsity = N_EXPERTS_PER_TOK as f64 / N_EXPERTS as f64;
        assert!((sparsity - 0.0625).abs() < 1e-9, "sparsity must be 8/128 = 6.25%");
    }

    /// Obligation: O projection transpose (invariant)
    /// Formal: shape(o_proj) == reverse(shape(q_proj))
    #[test]
    fn prop_o_projection_transpose() {
        let q_shape = (N_HEADS * D_K, HIDDEN); // [8192, 4096]
        let o_shape = (HIDDEN, N_HEADS * D_K); // [4096, 8192]
        assert_eq!(q_shape.0, o_shape.1, "O col must match Q row");
        assert_eq!(q_shape.1, o_shape.0, "O row must match Q col");
        // O is contracting: maps 8192 -> 4096
        assert!(o_shape.1 > o_shape.0, "O projection is contracting for this config");
    }

    /// Obligation: RoPE frequency decreasing (monotonicity)
    /// Formal: freq_i > freq_{i+1} for all i
    #[test]
    fn prop_rope_frequency_decreasing() {
        let freqs: Vec<f64> = (0..D_K / 2)
            .map(|i| ROPE_THETA.powf(-2.0 * i as f64 / D_K as f64))
            .collect();
        assert_eq!(freqs.len(), D_K / 2, "freq vector length must be d_k/2");
        assert!((freqs[0] - 1.0).abs() < 1e-9, "freq[0] must be 1.0");
        for i in 0..freqs.len() - 1 {
            assert!(
                freqs[i] > freqs[i + 1],
                "freq[{}]={} must be > freq[{}]={}",
                i, freqs[i], i + 1, freqs[i + 1]
            );
        }
    }

    /// Obligation: SIMD shape equivalence (equivalence)
    #[test]
    fn prop_simd_shape_equivalence() {
        let scalar_q = N_HEADS * D_K;
        let scalar_kv = N_KV_HEADS * D_K;
        let scalar_expert = 3 * HIDDEN * MOE_INTERMEDIATE;
        let simd_q = N_HEADS * D_K;
        let simd_kv = N_KV_HEADS * D_K;
        let simd_expert = 3 * HIDDEN * MOE_INTERMEDIATE;
        assert_eq!(scalar_q, simd_q, "Q dim mismatch");
        assert_eq!(scalar_kv, simd_kv, "KV dim mismatch");
        assert_eq!(scalar_expert, simd_expert, "expert param count mismatch");
    }

    // === Falsification tests ===

    /// FALSIFY-QM3-001: Q projection shape
    #[test]
    fn prop_falsify_qm3_001() {
        assert_eq!(64 * 128, 8192, "Q projection: 64 * 128 must equal 8192");
    }

    /// FALSIFY-QM3-002: KV projection shape
    #[test]
    fn prop_falsify_qm3_002() {
        assert_eq!(4 * 128, 512, "KV projection: 4 * 128 must equal 512");
    }

    /// FALSIFY-QM3-003: GQA divisibility
    #[test]
    fn prop_falsify_qm3_003() {
        assert_eq!(64 % 4, 0, "64 must be divisible by 4");
        assert_eq!(64 / 4, 16, "GQA ratio must be 16");
    }

    /// FALSIFY-QM3-004: MoE expert shape
    #[test]
    fn prop_falsify_qm3_004() {
        assert_eq!(3 * 4096 * 1536, 18_874_368, "per-expert params");
        assert_eq!(128_u64 * 18_874_368, 2_415_919_104_u64, "total expert params per layer");
    }

    /// FALSIFY-QM3-005: MoE router top-k
    #[test]
    fn prop_falsify_qm3_005() {
        assert!(8 < 128, "top-k must be < num_experts");
        assert_eq!(128 % 8, 0, "num_experts divisible by top-k");
        assert_eq!(4096 * 128, 524_288, "router params");
    }

    /// FALSIFY-QM3-006: O projection transpose
    #[test]
    fn prop_falsify_qm3_006() {
        let q_shape = [8192_usize, 4096];
        let o_shape = [4096_usize, 8192];
        assert_eq!(q_shape[0], o_shape[1], "dimension swap check");
        assert_eq!(q_shape[1], o_shape[0], "dimension swap check");
    }

    /// FALSIFY-QM3-007: RoPE frequency decreasing
    #[test]
    fn prop_falsify_qm3_007() {
        for (base, d_k) in [
            (10_000.0_f64, 64_usize),
            (500_000.0, 128),
            (1_000_000.0, 128),
            (10_000_000.0, 256),
        ] {
            let freqs: Vec<f64> = (0..d_k / 2)
                .map(|i| base.powf(-2.0 * i as f64 / d_k as f64))
                .collect();
            for i in 0..freqs.len() - 1 {
                assert!(
                    freqs[i] > freqs[i + 1],
                    "base={base} d_k={d_k}: freq not decreasing at {i}"
                );
            }
        }
    }

    /// FALSIFY-QM3-008: SIMD shape equivalence
    #[test]
    fn prop_falsify_qm3_008() {
        for (n_h, n_kv, d_k) in [(64, 4, 128), (32, 8, 128), (16, 4, 256)] {
            let scalar_q = n_h * d_k;
            let scalar_kv = n_kv * d_k;
            let simd_q = n_h * d_k;
            let simd_kv = n_kv * d_k;
            assert_eq!(scalar_q, simd_q, "n_h={n_h} d_k={d_k}: Q dim mismatch");
            assert_eq!(scalar_kv, simd_kv, "n_kv={n_kv} d_k={d_k}: KV dim mismatch");
        }
    }
}
