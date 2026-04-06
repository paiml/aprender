//! GPU SIMD Operations Module (PMAT-802)
//!
//! Extracted from gpu/mod.rs - SIMD-accelerated compute primitives.
//!
//! ## Contents
//! - `scalar_softmax`, `simd_softmax` - Softmax implementations (IMP-038)
//! - `scalar_rope`, `simd_rope` - RoPE implementations (IMP-041)

// ============================================================================
// SIMD-accelerated operations (M18 - IMP-038)
// ============================================================================

/// Scalar softmax implementation (baseline for comparison)
///
/// Computes softmax using standard scalar operations.
#[must_use]
pub fn scalar_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    // Normalize
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// SIMD-accelerated softmax implementation (M18 - IMP-038)
///
/// Uses Trueno's SIMD operations for vectorized computation.
/// Falls back to scalar for unsupported sizes.
#[must_use]
pub fn simd_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Find max using SIMD via trueno
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) - exp is not SIMD accelerated
    let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();

    // Sum using trueno's SIMD sum (CPU fallback if SIMD fails)
    let exp_vec = trueno::Vector::from_slice(&exp_vals);
    let sum = exp_vec.sum().unwrap_or_else(|e| {
        eprintln!("[WARN] SIMD softmax sum failed ({e}), using scalar fallback");
        exp_vals.iter().sum()
    });

    // Normalize
    exp_vals.iter().map(|&e| e / sum).collect()
}

// ============================================================================
// Scalar and SIMD RoPE implementations (M19 - IMP-041)
// ============================================================================

/// Scalar RoPE (Rotary Position Embedding) implementation
///
/// Standard scalar implementation of rotary position embeddings.
/// Input shape: [seq_len * hidden_dim] flattened
#[must_use]
pub fn scalar_rope(input: &[f32], seq_len: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    if input.is_empty() || seq_len == 0 || head_dim == 0 {
        return Vec::new();
    }

    let hidden_dim = input.len() / seq_len;
    let num_heads = hidden_dim / head_dim;
    let mut output = vec![0.0f32; input.len()];

    // Compute RoPE for each position
    for pos in 0..seq_len {
        for head in 0..num_heads {
            let head_start = pos * hidden_dim + head * head_dim;

            // Apply rotary embedding to pairs of elements
            for i in 0..head_dim / 2 {
                let freq = 1.0 / theta.powf((2.0 * i as f32) / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let idx0 = head_start + i;
                let idx1 = head_start + i + head_dim / 2;

                if idx1 < input.len() {
                    let x0 = input[idx0];
                    let x1 = input[idx1];
                    output[idx0] = x0 * cos_val - x1 * sin_val;
                    output[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    output
}

/// Pre-compute the RoPE frequency table for the given head dimension and theta.
fn rope_frequency_table(half_head: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    contract_pre_rope_frequency!();
    let result: Vec<f32> = (0..half_head)
        .map(|i| 1.0 / theta.powf((2.0 * i as f32) / head_dim as f32))
        .collect();
    contract_post_rope_frequency!(&result);
    result
}

/// Compute cos and sin trueno vectors for a given position and frequency table.
fn rope_trig_vectors(pos: usize, freqs: &[f32]) -> (trueno::Vector<f32>, trueno::Vector<f32>) {
    let angles: Vec<f32> = freqs.iter().map(|&f| pos as f32 * f).collect();
    let cos_vals: Vec<f32> = angles.iter().map(|&a| a.cos()).collect();
    let sin_vals: Vec<f32> = angles.iter().map(|&a| a.sin()).collect();
    (
        trueno::Vector::from_slice(&cos_vals),
        trueno::Vector::from_slice(&sin_vals),
    )
}

/// Apply rotary embedding to one head using SIMD vector operations.
///
/// Returns `Ok(())` on success or a `TruenoError` if any SIMD operation fails.
fn rope_rotate_head(
    input: &[f32],
    output: &mut [f32],
    head_start: usize,
    half_head: usize,
    head_dim: usize,
    cos_vec: &trueno::Vector<f32>,
    sin_vec: &trueno::Vector<f32>,
) -> std::result::Result<(), trueno::TruenoError> {
    let x0_vec = trueno::Vector::from_slice(&input[head_start..head_start + half_head]);
    let x1_vec = trueno::Vector::from_slice(&input[head_start + half_head..head_start + head_dim]);

    // out0 = x0 * cos - x1 * sin
    // out1 = x0 * sin + x1 * cos
    let out0 = x0_vec.mul(cos_vec)?.sub(&x1_vec.mul(sin_vec)?)?;
    let out1 = x0_vec.mul(sin_vec)?.add(&x1_vec.mul(cos_vec)?)?;

    output[head_start..head_start + half_head].copy_from_slice(out0.as_slice());
    output[head_start + half_head..head_start + head_dim].copy_from_slice(out1.as_slice());
    Ok(())
}

/// SIMD-accelerated RoPE implementation (M19 - IMP-041)
///
/// Uses Trueno's SIMD operations for vectorized position encoding.
#[must_use]
pub fn simd_rope(input: &[f32], seq_len: usize, head_dim: usize, theta: f32) -> Vec<f32> {
    if input.is_empty() || seq_len == 0 || head_dim == 0 {
        return Vec::new();
    }

    let hidden_dim = input.len() / seq_len;
    let num_heads = hidden_dim / head_dim;
    let half_head = head_dim / 2;

    let freqs = rope_frequency_table(half_head, head_dim, theta);
    let mut output = vec![0.0f32; input.len()];

    for pos in 0..seq_len {
        let (cos_vec, sin_vec) = rope_trig_vectors(pos, &freqs);

        for head in 0..num_heads {
            let head_start = pos * hidden_dim + head * head_dim;

            // BUG-HUNTER-FIX: If any SIMD op fails, fall back to scalar_rope
            // (returning unrotated input silently corrupts position embeddings)
            if let Err(e) = rope_rotate_head(
                input,
                &mut output,
                head_start,
                half_head,
                head_dim,
                &cos_vec,
                &sin_vec,
            ) {
                eprintln!("[WARN] SIMD RoPE failed ({e}), falling back to scalar");
                return scalar_rope(input, seq_len, head_dim, theta);
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // Softmax tests
    // ============================================================================

    #[test]
    fn test_scalar_softmax_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let result = scalar_softmax(&input);
        assert_eq!(result.len(), 3);
        // Sum should be 1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Larger input should have larger probability
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_scalar_softmax_empty() {
        let result = scalar_softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_scalar_softmax_single() {
        let result = scalar_softmax(&[5.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_softmax_uniform() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let result = scalar_softmax(&input);
        // All values equal => uniform distribution
        for &val in &result {
            assert!((val - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_scalar_softmax_numerical_stability() {
        // Large values that could overflow without max subtraction
        let input = vec![1000.0, 1001.0, 1002.0];
        let result = scalar_softmax(&input);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_softmax_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let result = simd_softmax(&input);
        assert_eq!(result.len(), 3);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_softmax_empty() {
        let result = simd_softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_softmax_matches_scalar() {
        let input = vec![0.5, 1.5, -0.5, 2.0, 0.0];
        let scalar_result = scalar_softmax(&input);
        let simd_result = simd_softmax(&input);

        for (s, d) in scalar_result.iter().zip(simd_result.iter()) {
            assert!((s - d).abs() < 1e-6, "scalar={} simd={}", s, d);
        }
    }

    #[test]
    fn test_simd_softmax_negative_values() {
        let input = vec![-1.0, -2.0, -3.0];
        let result = simd_softmax(&input);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Less negative should have higher probability
        assert!(result[0] > result[1]);
        assert!(result[1] > result[2]);
    }

    // ============================================================================
    // RoPE tests
    // ============================================================================

    #[test]
    fn test_scalar_rope_basic() {
        // 1 position, 1 head, head_dim=4
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let result = scalar_rope(&input, 1, 4, 10000.0);
        assert_eq!(result.len(), 4);
        // At position 0, angle=0, so cos=1, sin=0 => output = input
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_rope_empty() {
        let result = scalar_rope(&[], 0, 4, 10000.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_scalar_rope_zero_seq_len() {
        let result = scalar_rope(&[1.0, 2.0], 0, 2, 10000.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_scalar_rope_zero_head_dim() {
        let result = scalar_rope(&[1.0, 2.0], 1, 0, 10000.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_scalar_rope_multi_position() {
        // 2 positions, 1 head, head_dim=4
        let input = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let result = scalar_rope(&input, 2, 4, 10000.0);
        assert_eq!(result.len(), 8);
        // Position 0 should be unchanged (angle=0)
        // Position 1 should have rotation applied
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_simd_rope_basic() {
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let result = simd_rope(&input, 1, 4, 10000.0);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_simd_rope_empty() {
        let result = simd_rope(&[], 0, 4, 10000.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_rope_matches_scalar() {
        // 2 positions, 2 heads, head_dim=4
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // pos 0
            0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, // pos 1
        ];
        let scalar_result = scalar_rope(&input, 2, 4, 10000.0);
        let simd_result = simd_rope(&input, 2, 4, 10000.0);

        for (i, (s, d)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            assert!((s - d).abs() < 1e-5, "idx={} scalar={} simd={}", i, s, d);
        }
    }

    #[test]
    fn test_simd_rope_different_theta() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result_10k = simd_rope(&input, 1, 4, 10000.0);
        let result_1m = simd_rope(&input, 1, 4, 1_000_000.0);
        // Different theta should give different results at non-zero positions
        // At position 0 they're the same, but frequencies differ
        assert_eq!(result_10k.len(), result_1m.len());
    }

    #[test]
    fn test_scalar_rope_preserves_norm() {
        // RoPE is a rotation, should approximately preserve L2 norm
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = scalar_rope(&input, 1, 4, 10000.0);
        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((input_norm - output_norm).abs() < 1e-4);
    }

    #[test]
    fn test_simd_rope_multi_head() {
        // 1 position, 4 heads, head_dim=2
        let input = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let result = simd_rope(&input, 1, 2, 10000.0);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_softmax_extreme_negative() {
        let input = vec![-100.0, -200.0, -300.0];
        let result = scalar_softmax(&input);
        // First element should dominate
        assert!(result[0] > 0.99);
    }

    #[test]
    fn test_simd_softmax_large_vector() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();
        let result = simd_softmax(&input);
        assert_eq!(result.len(), 256);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
