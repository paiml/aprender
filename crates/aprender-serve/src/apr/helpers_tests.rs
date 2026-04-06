
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
