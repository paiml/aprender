//! Batch normalization kernel.
//!
//! Matches `batchnorm-kernel-v1.yaml`.
//!
//! Training mode: computes per-channel mean/variance from the batch, normalizes,
//! applies affine transform, and updates running statistics via EMA.
//!
//! Inference mode: uses running mean/variance directly for normalization.
//!
//! Input layout: N*C flattened, where N = batch size, C = channels.
//! Element (n, c) is at index `n * c_count + c`.

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Scalar reference implementation of BatchNorm.
///
/// # Arguments
///
/// * `input`        - Flattened `[N, C]` tensor (row-major).
/// * `n`            - Batch size (N).
/// * `c`            - Number of channels (C).
/// * `gamma`        - Per-channel scale, length `C`.
/// * `beta`         - Per-channel bias, length `C`.
/// * `eps`          - Small constant for numerical stability.
/// * `running_mean` - Running mean per channel, length `C`. Updated in training.
/// * `running_var`  - Running variance per channel, length `C`. Updated in training.
/// * `output`       - Output buffer, same shape as `input`.
/// * `momentum`     - EMA momentum for running stats update.
/// * `training`     - If true, compute batch stats and update running stats.
///   If false, use running stats for normalization.
///
/// # Panics
///
/// Panics if buffer sizes are inconsistent with `n * c`.
#[allow(clippy::too_many_arguments)]
pub fn batchnorm_scalar(
    input: &[f32],
    n: usize,
    c: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    running_mean: &mut [f32],
    running_var: &mut [f32],
    output: &mut [f32],
    momentum: f32,
    training: bool,
) {
    assert_eq!(input.len(), n * c, "input length must be n * c");
    assert_eq!(output.len(), n * c, "output length must be n * c");
    assert_eq!(gamma.len(), c, "gamma length must be c");
    assert_eq!(beta.len(), c, "beta length must be c");
    assert_eq!(running_mean.len(), c, "running_mean length must be c");
    assert_eq!(running_var.len(), c, "running_var length must be c");
    assert!(n > 0 && c > 0, "batchnorm requires n > 0 and c > 0");

    if training {
        // Compute per-channel batch statistics and normalize
        for ch in 0..c {
            // Compute batch mean for this channel
            let mut sum = 0.0_f32;
            for sample in 0..n {
                sum += input[sample * c + ch];
            }
            let batch_mean = sum / n as f32;

            // Compute batch variance for this channel
            let mut var_sum = 0.0_f32;
            for sample in 0..n {
                let diff = input[sample * c + ch] - batch_mean;
                var_sum += diff * diff;
            }
            let batch_var = var_sum / n as f32;

            // Normalize, scale, and shift
            let inv_std = 1.0 / (batch_var + eps).sqrt();
            for sample in 0..n {
                let idx = sample * c + ch;
                output[idx] = gamma[ch] * (input[idx] - batch_mean) * inv_std + beta[ch];
            }

            // Update running statistics (EMA)
            running_mean[ch] = (1.0 - momentum) * running_mean[ch] + momentum * batch_mean;
            running_var[ch] = (1.0 - momentum) * running_var[ch] + momentum * batch_var;
        }
    } else {
        // Inference: use running stats
        for ch in 0..c {
            let inv_std = 1.0 / (running_var[ch] + eps).sqrt();
            for sample in 0..n {
                let idx = sample * c + ch;
                output[idx] = gamma[ch] * (input[idx] - running_mean[ch]) * inv_std + beta[ch];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 BatchNorm -- delegates to scalar.
///
/// Batch dimension reduction is irregular for SIMD (strided access across
/// samples for each channel), so this delegates to the scalar implementation.
///
/// # Safety
///
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
///
/// Panics if buffer sizes are inconsistent with `n * c`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn batchnorm_avx2(
    input: &[f32],
    n: usize,
    c: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    running_mean: &mut [f32],
    running_var: &mut [f32],
    output: &mut [f32],
    momentum: f32,
    training: bool,
) {
    batchnorm_scalar(
        input,
        n,
        c,
        gamma,
        beta,
        eps,
        running_mean,
        running_var,
        output,
        momentum,
        training,
    );
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

include!("batchnorm_ptx.rs");

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // Scalar + property-based tests
    include!("batchnorm_tests.rs");
    // AVX2 parity + PTX structural tests
    include!("batchnorm_tests2.rs");
}
