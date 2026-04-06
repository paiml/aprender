//! Normalization and position encoding operations
//!
//! Provides layer normalization, RMS normalization, and rotary position embeddings
//! used in transformer inference.
//!
//! ## Normalization Functions
//!
//! - [`simd_layer_norm`] - Standard layer normalization with mean and variance
//! - [`simd_rms_norm`] - RMS normalization (faster, used in LLaMA/Mistral)
//!
//! ## Position Encoding
//!
//! - [`apply_rope`] - Rotary Position Embeddings (RoPE)

/// SIMD-accelerated layer normalization
///
/// LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
///
/// # Arguments
///
/// * `input` - Input vector to normalize
/// * `weight` - Scale parameters (gamma)
/// * `bias` - Optional shift parameters (beta)
/// * `eps` - Small constant for numerical stability (typically 1e-5)
///
/// # Example
///
/// ```
/// use realizar::inference::simd_layer_norm;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0];
/// let weight = vec![1.0, 1.0, 1.0, 1.0];
/// let output = simd_layer_norm(&input, &weight, None, 1e-5);
///
/// // Output should have mean ≈ 0 and std ≈ 1
/// let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
/// assert!(mean.abs() < 1e-5);
/// ```
#[must_use]
pub fn simd_layer_norm(input: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute mean
    let mean: f32 = input.iter().sum::<f32>() / n as f32;

    // Compute variance
    let var: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    // Normalize
    let inv_std = 1.0 / (var + eps).sqrt();
    let mut output: Vec<f32> = input.iter().map(|x| (x - mean) * inv_std).collect();

    // Apply affine transformation
    for (i, out) in output.iter_mut().enumerate() {
        *out *= weight[i];
        if let Some(b) = bias {
            *out += b[i];
        }
    }

    output
}

/// SIMD-accelerated RMS normalization
///
/// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
///
/// RMS normalization is faster than LayerNorm as it doesn't require
/// computing the mean. Used in LLaMA, Mistral, and other modern LLMs.
///
/// # Arguments
///
/// * `input` - Input vector to normalize
/// * `weight` - Scale parameters
/// * `eps` - Small constant for numerical stability (typically 1e-5)
///
/// # Example
///
/// ```
/// use realizar::inference::simd_rms_norm;
///
/// let input = vec![1.0, 2.0, 3.0];
/// let weight = vec![1.0, 1.0, 1.0];
/// let output = simd_rms_norm(&input, &weight, 1e-5);
///
/// // RMS of [1,2,3] ≈ 2.16, so normalized ≈ [0.46, 0.93, 1.39]
/// assert!((output[0] - 0.4629).abs() < 0.01);
/// ```
#[must_use]
pub fn simd_rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute RMS
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Normalize and scale
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * inv_rms * w)
        .collect()
}

/// Apply rotary position embeddings (RoPE)
///
/// RoPE encodes position information by rotating pairs of dimensions.
/// This enables relative position encoding that generalizes to longer sequences.
///
/// # Arguments
///
/// * `x` - Mutable slice to apply RoPE to [hidden_dim]
/// * `hidden_dim` - Total hidden dimension (must equal x.len())
/// * `num_heads` - Number of attention heads
/// * `position` - Token position in sequence (0-indexed)
/// * `theta` - Base frequency (typically 10000.0)
///
/// # Algorithm
///
/// For each head and each pair of dimensions (i, i + d/2):
/// ```text
/// freq = 1 / theta^(2i/d)
/// angle = position * freq
/// x[i]     = x[i] * cos(angle) - x[i+d/2] * sin(angle)
/// x[i+d/2] = x[i] * sin(angle) + x[i+d/2] * cos(angle)
/// ```
///
/// # Example
///
/// ```
/// use realizar::inference::apply_rope;
///
/// let mut x = vec![1.0; 64];  // 64 hidden dim
/// apply_rope(&mut x, 64, 4, 0, 10000.0);  // Position 0
///
/// // At position 0, rotations are identity (angle = 0)
/// assert!((x[0] - 1.0).abs() < 1e-5);
/// ```
pub fn apply_rope(x: &mut [f32], hidden_dim: usize, num_heads: usize, position: usize, theta: f32) {
    let head_dim = hidden_dim / num_heads;
    let half_dim = head_dim / 2;

    for h in 0..num_heads {
        let head_offset = h * head_dim;

        for i in 0..half_dim {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let idx0 = head_offset + i;
            let idx1 = head_offset + i + half_dim;

            let x0 = x[idx0];
            let x1 = x[idx1];

            x[idx0] = x0 * cos_val - x1 * sin_val;
            x[idx1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

include!("norm_layer.rs");
