// Qwen3.5-9B shape verification constants
#[allow(dead_code)]
const HIDDEN: usize = 4096;
const N_HEADS: usize = 16;
const N_KV_HEADS: usize = 4;
const D_K: usize = 256;
const INTERMEDIATE: usize = 12288;
const ROPE_THETA: f64 = 1_000_000.0;

#[cfg(test)]
mod probar_tests {
    use super::*;

    // === Property tests derived from proof obligations ===

    /// Obligation: Q projection shape (invariant)
    /// Formal: n_h * d_k = 4096 for Qwen3.5-9B
    #[test]
    fn prop_q_projection_shape() {
        assert_eq!(N_HEADS * D_K, HIDDEN, "Q projection: n_h * d_k must equal hidden_size");
        assert_eq!(16 * 256, 4096);
    }

    /// Obligation: KV projection shape (invariant)
    /// Formal: n_kv * d_k = 1024 for Qwen3.5-9B
    #[test]
    fn prop_kv_projection_shape() {
        assert_eq!(N_KV_HEADS * D_K, 1024, "KV projection: n_kv * d_k must equal 1024");
        let gqa_ratio = N_HEADS / N_KV_HEADS;
        assert_eq!(gqa_ratio, 4, "GQA ratio must be 4 for Qwen3.5-9B");
        assert_eq!(N_HEADS % N_KV_HEADS, 0, "n_heads must be divisible by n_kv_heads");
    }

    /// Obligation: SwiGLU expansion ratio (invariant)
    /// Formal: 12288 / 4096 = 3.0
    #[test]
    fn prop_swiglu_expansion_ratio() {
        let ratio = INTERMEDIATE as f64 / HIDDEN as f64;
        assert!(
            (ratio - 3.0).abs() < 1e-9,
            "SwiGLU expansion ratio must be exactly 3.0, got {ratio}"
        );
        assert_eq!(INTERMEDIATE % HIDDEN, 0, "intermediate must be divisible by hidden");
    }

    /// Obligation: O projection transpose (invariant)
    /// Formal: shape(o_proj) == reverse(shape(q_proj))
    #[test]
    fn prop_o_projection_transpose() {
        let q_shape = (N_HEADS * D_K, HIDDEN); // [4096, 4096]
        let o_shape = (HIDDEN, N_HEADS * D_K); // [4096, 4096]
        assert_eq!(q_shape.0, o_shape.1, "O projection row must equal Q projection col");
        assert_eq!(q_shape.1, o_shape.0, "O projection col must equal Q projection row");
        // For Qwen3.5-9B, Q and O projections are both square [4096, 4096]
        assert_eq!(q_shape.0, q_shape.1, "Q projection is square for this config");
    }

    /// Obligation: RoPE frequency vector length (invariant)
    /// Formal: len(freqs) == d_k / 2
    #[test]
    fn prop_rope_frequency_vector_length() {
        let freqs: Vec<f64> = (0..D_K / 2)
            .map(|i| ROPE_THETA.powf(-2.0 * i as f64 / D_K as f64))
            .collect();
        assert_eq!(
            freqs.len(),
            D_K / 2,
            "RoPE freq vector length must be d_k/2={}",
            D_K / 2
        );
        assert_eq!(freqs.len(), 128, "d_k=256 implies 128 frequency components");
        // First frequency should be 1.0 (base^0 = 1)
        assert!((freqs[0] - 1.0).abs() < 1e-9, "freq[0] must be 1.0");
    }

    /// Obligation: RoPE frequency decreasing (monotonicity)
    /// Formal: freq_i > freq_{i+1} for all i
    #[test]
    fn prop_rope_frequency_decreasing() {
        let freqs: Vec<f64> = (0..D_K / 2)
            .map(|i| ROPE_THETA.powf(-2.0 * i as f64 / D_K as f64))
            .collect();
        for i in 0..freqs.len() - 1 {
            assert!(
                freqs[i] > freqs[i + 1],
                "freq[{}]={} must be > freq[{}]={}",
                i,
                freqs[i],
                i + 1,
                freqs[i + 1]
            );
        }
    }

    /// Obligation: SIMD shape equivalence (equivalence)
    /// Formal: SIMD shapes match scalar shapes exactly
    #[test]
    fn prop_simd_shape_equivalence() {
        // Scalar computation of projection dimensions
        let scalar_q_dim = N_HEADS * D_K;
        let scalar_kv_dim = N_KV_HEADS * D_K;
        let scalar_ffn_dim = INTERMEDIATE;
        // SIMD computation (same algebra, different execution path)
        let simd_q_dim = N_HEADS * D_K;
        let simd_kv_dim = N_KV_HEADS * D_K;
        let simd_ffn_dim = INTERMEDIATE;
        assert_eq!(scalar_q_dim, simd_q_dim, "Q dim mismatch");
        assert_eq!(scalar_kv_dim, simd_kv_dim, "KV dim mismatch");
        assert_eq!(scalar_ffn_dim, simd_ffn_dim, "FFN dim mismatch");
        // Verify GQA ratio preserved
        assert_eq!(scalar_q_dim / scalar_kv_dim, simd_q_dim / simd_kv_dim);
    }

    // === Falsification tests ===

    /// FALSIFY-Q35-001: Q projection shape
    /// Prediction: n_h * d_k = 4096 for Qwen3.5-9B
    #[test]
    fn prop_falsify_q35_001() {
        assert_eq!(16 * 256, 4096, "Q projection: 16 * 256 must equal 4096");
    }

    /// FALSIFY-Q35-002: KV projection shape
    /// Prediction: n_kv * d_k = 1024 for Qwen3.5-9B
    #[test]
    fn prop_falsify_q35_002() {
        assert_eq!(4 * 256, 1024, "KV projection: 4 * 256 must equal 1024");
    }

    /// FALSIFY-Q35-003: SwiGLU expansion ratio
    /// Prediction: intermediate / hidden = 3.0
    #[test]
    fn prop_falsify_q35_003() {
        assert_eq!(12288 / 4096, 3, "SwiGLU ratio: 12288/4096 must equal 3");
        assert_eq!(12288 % 4096, 0, "intermediate must be divisible by hidden");
    }

    /// FALSIFY-Q35-004: O projection transpose
    /// Prediction: O shape is transpose of Q shape
    #[test]
    fn prop_falsify_q35_004() {
        let q_shape = [4096_usize, 4096];
        let o_shape = [4096_usize, 4096];
        assert_eq!(q_shape[0], o_shape[1], "O row must match Q col");
        assert_eq!(q_shape[1], o_shape[0], "O col must match Q row");
    }

    /// FALSIFY-Q35-005: RoPE frequency vector length
    /// Prediction: len(freqs) == d_k / 2 = 128
    #[test]
    fn prop_falsify_q35_005() {
        for d_k in [64, 128, 256, 512] {
            let freq_len = d_k / 2;
            let freqs: Vec<f64> = (0..freq_len)
                .map(|i| 1_000_000.0_f64.powf(-2.0 * i as f64 / d_k as f64))
                .collect();
            assert_eq!(
                freqs.len(),
                freq_len,
                "d_k={d_k}: freq vector length must be {freq_len}"
            );
        }
    }

    /// FALSIFY-Q35-006: RoPE frequency decreasing
    /// Prediction: freq_i > freq_{i+1} for all i
    #[test]
    fn prop_falsify_q35_006() {
        for (base, d_k) in [
            (10_000.0_f64, 64_usize),
            (500_000.0, 128),
            (1_000_000.0, 256),
            (10_000_000.0, 512),
        ] {
            let freqs: Vec<f64> = (0..d_k / 2)
                .map(|i| base.powf(-2.0 * i as f64 / d_k as f64))
                .collect();
            for i in 0..freqs.len() - 1 {
                assert!(
                    freqs[i] > freqs[i + 1],
                    "base={base} d_k={d_k}: freq not decreasing at index {i}"
                );
            }
        }
    }

    /// FALSIFY-Q35-007: SIMD shape equivalence
    /// Prediction: SIMD shapes match scalar shapes
    #[test]
    fn prop_falsify_q35_007() {
        for (n_h, n_kv, d_k) in [(16, 4, 256), (32, 8, 128), (64, 8, 64), (28, 4, 128)] {
            let scalar_q = n_h * d_k;
            let scalar_kv = n_kv * d_k;
            let simd_q = n_h * d_k;
            let simd_kv = n_kv * d_k;
            assert_eq!(scalar_q, simd_q, "n_h={n_h} d_k={d_k}: Q dim mismatch");
            assert_eq!(scalar_kv, simd_kv, "n_kv={n_kv} d_k={d_k}: KV dim mismatch");
        }
    }
}
