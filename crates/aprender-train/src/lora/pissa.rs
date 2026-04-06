//! PiSSA (Principal Singular Values and Singular Vectors Adaptation) — ENT-LoRA-012
//!
//! PiSSA initializes LoRA A and B from the top-r singular components of the base weight,
//! achieving faster convergence (+5% on some benchmarks).
//!
//! Standard LoRA: A ~ N(0, σ²), B = 0 → ΔW = 0 at init
//! PiSSA: SVD(W) = U·S·V^T → A = sqrt(S_r)·V_r^T, B = U_r·sqrt(S_r) → ΔW = U_r·S_r·V_r^T
//! Residual: W_residual = W - U_r·S_r·V_r^T (frozen base becomes the residual)
//!
//! Reference: Meng et al. (2024). "PiSSA: Principal Singular Values and Singular Vectors
//! Adaptation." NeurIPS 2024 Spotlight.

use crate::lora::LoRALayer;
use crate::Tensor;

/// Initialize a LoRA layer using PiSSA (SVD-based initialization)
///
/// Returns a LoRALayer where:
/// - `base_weight` is the residual (W - U_r·S_r·V_r^T)
/// - `lora_a` = sqrt(S_r) · V_r^T [rank × d_in]
/// - `lora_b` = U_r · sqrt(S_r) [d_out × rank]
///
/// The key insight: instead of starting from ΔW=0, PiSSA starts from the principal
/// components, so the residual base weight has lower effective rank and LoRA adapters
/// start from a better initialization.
pub fn pissa_init(
    base_weight: &Tensor,
    d_out: usize,
    d_in: usize,
    rank: usize,
    alpha: f32,
) -> LoRALayer {
    assert_eq!(base_weight.len(), d_out * d_in);
    assert!(rank <= d_out.min(d_in), "Rank must be <= min(d_out, d_in)");

    // Truncated SVD via power iteration
    let (u_r, s_r, v_r) =
        truncated_svd(base_weight.data().as_slice().expect("contiguous"), d_out, d_in, rank);

    // Compute A = sqrt(S_r) · V_r^T [rank × d_in]
    let mut a_data = vec![0.0f32; rank * d_in];
    for r in 0..rank {
        let sqrt_s = s_r[r].sqrt();
        for j in 0..d_in {
            a_data[r * d_in + j] = sqrt_s * v_r[r * d_in + j];
        }
    }

    // Compute B = U_r · sqrt(S_r) [d_out × rank]
    let mut b_data = vec![0.0f32; d_out * rank];
    for i in 0..d_out {
        for r in 0..rank {
            let sqrt_s = s_r[r].sqrt();
            b_data[i * rank + r] = u_r[i * rank + r] * sqrt_s;
        }
    }

    // Compute residual: W_res = W - U_r · S_r · V_r^T
    let scale = alpha / rank as f32;
    let mut residual = base_weight.data().to_vec();
    for i in 0..d_out {
        for j in 0..d_in {
            let mut reconstruction = 0.0f32;
            for r in 0..rank {
                reconstruction += u_r[i * rank + r] * s_r[r] * v_r[r * d_in + j];
            }
            // Adjust: the LoRA contribution will be scale * B @ A = scale * U_r·S_r·V_r^T
            // So we subtract scale * reconstruction from base
            residual[i * d_in + j] -= scale * reconstruction;
        }
    }

    let residual_tensor = Tensor::from_vec(residual, false);
    let mut layer = LoRALayer::new(residual_tensor, d_out, d_in, rank, alpha);

    // Override the default random init with PiSSA init
    *layer.lora_a_mut().data_mut() = ndarray::arr1(&a_data);
    *layer.lora_b_mut().data_mut() = ndarray::arr1(&b_data);

    layer
}

/// Truncated SVD via power iteration method
///
/// Returns (U_r, S_r, V_r) where:
/// - U_r: [d_out × rank] left singular vectors (column-major stored as row-major)
/// - S_r: [rank] singular values (descending)
/// - V_r: [rank × d_in] right singular vectors
fn truncated_svd(
    w: &[f32],
    d_out: usize,
    d_in: usize,
    rank: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let iterations = 20;
    let mut u_r = vec![0.0f32; d_out * rank];
    let mut s_r = vec![0.0f32; rank];
    let mut v_r = vec![0.0f32; rank * d_in];

    // Work on a copy so we can deflate
    let mut w_residual = w.to_vec();

    for r in 0..rank {
        // Initialize random vector v
        let mut v: Vec<f32> =
            (0..d_in).map(|i| ((i as f32 * 0.7 + r as f32 * 1.3).sin())).collect();
        normalize(&mut v);

        let mut u = vec![0.0f32; d_out];
        let mut sigma = 0.0f32;

        for _ in 0..iterations {
            // u = W @ v
            mat_vec_mul(&w_residual, &v, &mut u, d_out, d_in);
            sigma = norm(&u).max(1e-10);
            for val in &mut u {
                *val /= sigma;
            }

            // v = W^T @ u
            mat_t_vec_mul(&w_residual, &u, &mut v, d_out, d_in);
            let v_norm = norm(&v).max(1e-10);
            for val in &mut v {
                *val /= v_norm;
            }
        }

        // Store results
        for i in 0..d_out {
            u_r[i * rank + r] = u[i];
        }
        s_r[r] = sigma;
        for j in 0..d_in {
            v_r[r * d_in + j] = v[j];
        }

        // Deflate: W_residual -= sigma * u * v^T
        for i in 0..d_out {
            for j in 0..d_in {
                w_residual[i * d_in + j] -= sigma * u[i] * v[j];
            }
        }
    }

    (u_r, s_r, v_r)
}

fn mat_vec_mul(w: &[f32], v: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let mut sum = 0.0f32;
        for j in 0..cols {
            sum += w[i * cols + j] * v[j];
        }
        out[i] = sum;
    }
}

fn mat_t_vec_mul(w: &[f32], u: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for j in 0..cols {
        let mut sum = 0.0f32;
        for i in 0..rows {
            sum += w[i * cols + j] * u[i];
        }
        out[j] = sum;
    }
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalize(v: &mut [f32]) {
    let n = norm(v).max(1e-10);
    for val in v.iter_mut() {
        *val /= n;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    #[test]
    fn test_ent_lora_012_pissa_init_dimensions() {
        let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
        let layer = pissa_init(&base, 2, 3, 1, 2.0);
        assert_eq!(layer.d_out(), 2);
        assert_eq!(layer.d_in(), 3);
        assert_eq!(layer.rank(), 1);
        assert_eq!(layer.lora_a().len(), 3);
        assert_eq!(layer.lora_b().len(), 2);
    }

    #[test]
    fn test_ent_lora_012_pissa_nonzero_init() {
        // Unlike standard LoRA where B=0, PiSSA initializes both A and B from SVD
        let base = Tensor::from_vec(vec![1.0, 0.5, 0.5, 1.0], false);
        let layer = pissa_init(&base, 2, 2, 1, 2.0);

        // B should be non-zero (unlike standard LoRA)
        let b_norm: f32 = layer.lora_b().data().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(b_norm > 0.01, "PiSSA B should be non-zero, got norm={b_norm}");

        let a_norm: f32 = layer.lora_a().data().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(a_norm > 0.01, "PiSSA A should be non-zero, got norm={a_norm}");
    }

    #[test]
    fn test_ent_lora_012_pissa_reconstruction_close() {
        // W ≈ residual + scale * B @ A
        let d_out = 4;
        let d_in = 4;
        let base_data: Vec<f32> = (0..d_out * d_in).map(|i| (i as f32 * 0.3).sin()).collect();
        let base = Tensor::from_vec(base_data.clone(), false);
        let layer = pissa_init(&base, d_out, d_in, 2, 2.0);

        // Compute reconstruction: residual + scale * B @ A
        let scale = layer.scale();
        let residual = layer.base_weight().data();
        let a = layer.lora_a().data();
        let b = layer.lora_b().data();
        let rank = layer.rank();

        for i in 0..d_out {
            for j in 0..d_in {
                let mut ba = 0.0f32;
                for r in 0..rank {
                    ba += b[i * rank + r] * a[r * d_in + j];
                }
                let reconstructed = residual[i * d_in + j] + scale * ba;
                assert_abs_diff_eq!(base_data[i * d_in + j], reconstructed, epsilon = 0.3);
            }
        }
    }

    #[test]
    fn test_ent_lora_012_pissa_forward_works() {
        let base = Tensor::from_vec(vec![1.0; 16], false);
        let layer = pissa_init(&base, 4, 4, 2, 4.0);
        let x = Tensor::from_vec(vec![0.5; 4], true);
        let out = layer.forward(&x);
        assert_eq!(out.len(), 4);
        for val in out.data() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_ent_lora_012_truncated_svd_singular_values_descending() {
        let w: Vec<f32> = (0..24).map(|i| (i as f32 * 0.2).sin()).collect();
        let (_, s, _) = truncated_svd(&w, 4, 6, 3);

        for i in 1..s.len() {
            assert!(
                s[i - 1] >= s[i] - 1e-4,
                "Singular values should descend: s[{}]={} < s[{}]={}",
                i - 1,
                s[i - 1],
                i,
                s[i]
            );
        }
    }

    #[test]
    fn test_ent_lora_012_truncated_svd_orthogonal_u() {
        let w: Vec<f32> = (0..24).map(|i| (i as f32 * 0.3).cos()).collect();
        let (u, _, _) = truncated_svd(&w, 4, 6, 2);

        // Check approximate orthogonality of U columns
        // U is stored as [d_out × rank], column r is u[i*rank + r]
        let mut dot = 0.0f32;
        for i in 0..4 {
            dot += u[i * 2] * u[i * 2 + 1];
        }
        assert!(dot.abs() < 0.15, "U columns should be ~orthogonal, dot={dot}");
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(30))]

        #[test]
        fn prop_pissa_forward_finite(
            d_out in 2usize..8,
            d_in in 2usize..8,
        ) {
            let rank = 1.min(d_out.min(d_in));
            let base = Tensor::from_vec(vec![0.5; d_out * d_in], false);
            let layer = pissa_init(&base, d_out, d_in, rank, 4.0);
            let x = Tensor::from_vec(vec![0.1; d_in], true);
            let out = layer.forward(&x);
            prop_assert_eq!(out.len(), d_out);
            for val in out.data() {
                prop_assert!(val.is_finite());
            }
        }
    }
}
