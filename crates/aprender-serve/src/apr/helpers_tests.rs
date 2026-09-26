
// ============================================================================
// Tests for APR Helpers (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // rms_norm Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rms_norm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;
        let result = rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 4);
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.738
        // Normalized values should sum to approximately 0
        let sum: f32 = result.iter().sum();
        assert!(sum.abs() > 0.0); // Non-zero sum due to weight
    }

    #[test]
    fn test_rms_norm_zeros() {
        let x = vec![0.0, 0.0, 0.0, 0.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;
        let result = rms_norm(&x, &weight, eps);
        // All zeros normalized with small eps
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_rms_norm_seq_len_2() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 sequences of length 4
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;
        let result = rms_norm(&x, &weight, eps);
        assert_eq!(result.len(), 8);
    }

    // -------------------------------------------------------------------------
    // matmul Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_matmul_basic() {
        // 1x2 @ 3x2^T -> 1x3
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 rows of 2 cols
        let result = matmul(&x, &w, 1, 2, 3);
        assert_eq!(result.len(), 3);
        // row 0 of w: [1,0] dot [1,2] = 1
        // row 1 of w: [0,1] dot [1,2] = 2
        // row 2 of w: [1,1] dot [1,2] = 3
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 2.0).abs() < 0.001);
        assert!((result[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_matmul_identity() {
        // Identity matrix
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3x3 identity
        let result = matmul(&x, &w, 1, 3, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 2.0).abs() < 0.001);
        assert!((result[2] - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_matmul_zeros() {
        let x = vec![1.0, 2.0];
        let w = vec![0.0; 6]; // 3x2 zeros
        let result = matmul(&x, &w, 1, 2, 3);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // -------------------------------------------------------------------------
    // simd_dot Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = simd_dot(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((result - 70.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_large() {
        // Large enough to use AVX2 path
        let n = 64;
        let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let b = vec![1.0f32; n];
        let result = simd_dot(&a, &b);
        // Sum of 1 to 64 = 64*65/2 = 2080
        assert!((result - 2080.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let result = simd_dot(&a, &b);
        assert!((result - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_dot_unequal_lengths() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0]; // Shorter
        let result = simd_dot(&a, &b);
        // Uses min(a.len, b.len) = 2
        // 1*4 + 2*5 = 14
        assert!((result - 14.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // apply_rope_norm Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_apply_rope_norm_basic() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        apply_rope_norm(&mut x, 1, 4, 0, 10000.0, 0); // NORM style
        // At position 0, angle = 0, cos=1, sin=0, so values should be unchanged
        assert!((x[0] - 1.0).abs() < 0.001);
        assert!((x[1] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_apply_rope_norm_position_1() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        apply_rope_norm(&mut x, 1, 4, 1, 10000.0, 0); // NORM style
        // At position 1, some rotation should occur
        // Values should be different from original
        let sum: f32 = x.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.5); // Non-zero output
    }

    #[test]
    fn test_apply_rope_norm_multiple_heads() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]; // 2 heads, head_dim=4
        apply_rope_norm(&mut x, 2, 4, 0, 10000.0, 0); // NORM style
        // At position 0, values should be unchanged for both heads
        assert!((x[0] - 1.0).abs() < 0.001);
        assert!((x[4] - 2.0).abs() < 0.001);
    }

    // BUG-2 FIX: Test NEOX style rope (rope_type=2) for Qwen2.5
    #[test]
    fn test_apply_rope_neox_basic() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        apply_rope_norm(&mut x, 1, 4, 0, 10000.0, 2); // NEOX style
        // At position 0, angle = 0, cos=1, sin=0, so values should be unchanged
        assert!((x[0] - 1.0).abs() < 0.001);
        assert!((x[2] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_apply_rope_neox_position_1() {
        let mut x = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, head_dim=4
        apply_rope_norm(&mut x, 1, 4, 1, 10000.0, 2); // NEOX style
        // At position 1, some rotation should occur
        let sum: f32 = x.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.5); // Non-zero output
    }

    // -------------------------------------------------------------------------
    // simple_attention Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simple_attention_basic() {
        // 1 sequence, 1 head, head_dim=2
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![0.5, 0.5];
        let result = simple_attention(&q, &k, &v, 1, 1, 1, 2);
        assert_eq!(result.len(), 2);
        // Single token attending to itself
        assert!((result[0] - 0.5).abs() < 0.001);
        assert!((result[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_simple_attention_seq_len_2() {
        // 2 tokens, 1 head, head_dim=2
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let result = simple_attention(&q, &k, &v, 2, 1, 1, 2);
        assert_eq!(result.len(), 4);
        // Non-trivial attention weights
    }

    // -------------------------------------------------------------------------
    // is_apr_file Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_apr_file_nonexistent() {
        assert!(!is_apr_file("/nonexistent/file.apr"));
    }

    // -------------------------------------------------------------------------
    // detect_format Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_detect_format_by_extension_apr() {
        assert_eq!(detect_format("/some/path/model.apr"), "apr");
    }

    #[test]
    fn test_detect_format_by_extension_gguf() {
        assert_eq!(detect_format("/some/path/model.gguf"), "gguf");
    }

    #[test]
    fn test_detect_format_by_extension_safetensors() {
        assert_eq!(detect_format("/some/path/model.safetensors"), "safetensors");
    }

    #[test]
    fn test_detect_format_nonexistent() {
        // No extension match, file doesn't exist
        assert_eq!(detect_format("/nonexistent/file.bin"), "unknown");
    }
}

/// PP-ARCH-001 §9.14: `simple_attention` moved onto `gguf::ops::attend_row_scalar`.
/// The frozen pre-move body must agree bit for bit, including the zero-fill of
/// short buffers.
#[cfg(test)]
mod simple_attention_equivalence_tests {
    use super::simple_attention;

    #[allow(clippy::too_many_arguments)]
    fn frozen(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let hidden_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0; seq_len * hidden_dim];
        for s in 0..seq_len {
            for h in 0..num_heads {
                let kv_h = h / heads_per_kv;
                let q_base = s * hidden_dim + h * head_dim;
                let k_base = kv_h * head_dim;
                let mut scores = vec![0.0f32; seq_len];
                for t in 0..=s {
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        let q_val = q.get(q_base + d).copied().unwrap_or(0.0);
                        let k_val = k.get(t * kv_dim + k_base + d).copied().unwrap_or(0.0);
                        score += q_val * k_val;
                    }
                    scores[t] = score * scale;
                }
                let row = &mut scores[..=s];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in row.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                for x in row.iter_mut() {
                    *x /= sum;
                }
                for d in 0..head_dim {
                    let mut val = 0.0;
                    for t in 0..=s {
                        let v_val = v.get(kv_dim * t + kv_h * head_dim + d).copied().unwrap_or(0.0);
                        val += scores[t] * v_val;
                    }
                    output[s * hidden_dim + h * head_dim + d] = val;
                }
            }
        }
        output
    }

    fn fill(n: usize, seed: u32, mag: f32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                ((x as f32 / u32::MAX as f32) * 2.0 - 1.0) * mag
            })
            .collect()
    }

    #[test]
    fn matches_frozen_body_bit_for_bit() {
        let mut cases = 0;
        for &(nh, nkv) in &[(1, 1), (4, 4), (4, 2), (8, 1), (6, 3)] {
            for &hd in &[1usize, 7, 8, 16, 33] {
                for &seq in &[0usize, 1, 2, 5, 17] {
                    for &mag in &[0.1f32, 3.0, 40.0] {
                        // short = 0: full buffers; short > 0: truncated k/v/q.
                        for &short in &[0usize, 1, 5] {
                            let seed = (nh * 1000 + hd * 31 + seq * 7) as u32;
                            let qn = (seq * nh * hd).saturating_sub(short);
                            let kn = (seq * nkv * hd).saturating_sub(short);
                            let q = fill(qn, seed, mag);
                            let k = fill(kn, seed + 1, mag);
                            let v = fill(kn, seed + 2, mag);
                            let got = simple_attention(&q, &k, &v, seq, nh, nkv, hd);
                            let want = frozen(&q, &k, &v, seq, nh, nkv, hd);
                            assert_eq!(got.len(), want.len());
                            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                                assert_eq!(
                                    g.to_bits(),
                                    w.to_bits(),
                                    "nh={nh} nkv={nkv} hd={hd} seq={seq} mag={mag} short={short} i={i}: {g} vs {w}"
                                );
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 5 * 5 * 5 * 3 * 3);
    }
}
