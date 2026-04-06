//! Gradient clipping utilities

use crate::Tensor;

/// Clip gradients by global norm
///
/// Computes the global norm of all gradients and scales them down if the norm
/// exceeds max_norm. This prevents exploding gradients while preserving the
/// relative magnitudes of gradients across parameters.
///
/// Algorithm:
/// 1. global_norm = sqrt(sum of all gradient squared norms)
/// 2. If global_norm > max_norm:
///    - clip_coef = max_norm / global_norm
///    - For each gradient: grad *= clip_coef
///
/// # Arguments
/// * `params` - Mutable slice of parameters with gradients
/// * `max_norm` - Maximum allowed global norm
///
/// # Returns
/// The actual global norm before clipping
pub fn clip_grad_norm(params: &mut [Tensor], max_norm: f32) -> f32 {
    // Compute global norm: sqrt(sum of squared norms)
    let mut total_norm_sq = 0.0;

    for param in params.iter() {
        if let Some(grad) = param.grad() {
            // Compute squared norm of this gradient
            let grad_norm_sq: f32 = grad.iter().map(|&g| g * g).sum();
            total_norm_sq += grad_norm_sq;
        }
    }

    let global_norm = total_norm_sq.sqrt();

    // Only clip if global norm exceeds max_norm
    if global_norm > max_norm {
        let clip_coef = max_norm / global_norm;

        // Scale all gradients
        for param in params.iter_mut() {
            if let Some(grad) = param.grad() {
                let clipped_grad = grad * clip_coef;
                param.set_grad(clipped_grad);
            }
        }
    }

    global_norm
}

/// Clip gradients by global norm on borrowed parameter references.
///
/// Identical to [`clip_grad_norm`] but accepts `&mut [&mut Tensor]` instead of
/// `&mut [Tensor]`. This is useful when parameters are collected as mutable
/// references from a model (e.g., LoRA layers + classification head).
///
/// # Arguments
/// * `params` - Mutable slice of parameter references with gradients
/// * `max_norm` - Maximum allowed global norm
///
/// # Returns
/// The actual global norm before clipping
pub fn clip_grad_norm_refs(params: &mut [&mut Tensor], max_norm: f32) -> f32 {
    // Compute global norm: sqrt(sum of squared norms)
    let mut total_norm_sq = 0.0;

    for param in params.iter() {
        if let Some(grad) = param.grad() {
            let grad_norm_sq: f32 = grad.iter().map(|&g| g * g).sum();
            total_norm_sq += grad_norm_sq;
        }
    }

    let global_norm = total_norm_sq.sqrt();

    // Only clip if global norm exceeds max_norm
    if global_norm > max_norm {
        let clip_coef = max_norm / global_norm;

        for param in params.iter_mut() {
            if let Some(grad) = param.grad() {
                let clipped_grad = grad * clip_coef;
                param.set_grad(clipped_grad);
            }
        }
    }

    global_norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_clip_grad_norm_no_clipping() {
        // Gradients with norm below threshold shouldn't be clipped
        let mut params =
            vec![Tensor::from_vec(vec![1.0, 2.0], true), Tensor::from_vec(vec![3.0], true)];

        // Set small gradients
        params[0].set_grad(ndarray::arr1(&[0.1, 0.2]));
        params[1].set_grad(ndarray::arr1(&[0.1]));

        // Global norm = sqrt(0.1^2 + 0.2^2 + 0.1^2) = sqrt(0.06) ≈ 0.245
        let global_norm = clip_grad_norm(&mut params, 1.0);

        assert_abs_diff_eq!(global_norm, 0.245, epsilon = 1e-3);

        // Gradients should be unchanged
        assert_abs_diff_eq!(
            params[0].grad().expect("gradient should be available")[0],
            0.1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            params[0].grad().expect("gradient should be available")[1],
            0.2,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            params[1].grad().expect("gradient should be available")[0],
            0.1,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_clip_grad_norm_with_clipping() {
        // Gradients with norm above threshold should be scaled
        let mut params =
            vec![Tensor::from_vec(vec![1.0, 2.0], true), Tensor::from_vec(vec![3.0], true)];

        // Set large gradients
        params[0].set_grad(ndarray::arr1(&[3.0, 4.0]));
        params[1].set_grad(ndarray::arr1(&[0.0]));

        // Global norm = sqrt(3^2 + 4^2 + 0^2) = sqrt(25) = 5.0
        let global_norm = clip_grad_norm(&mut params, 1.0);

        assert_abs_diff_eq!(global_norm, 5.0, epsilon = 1e-6);

        // Gradients should be scaled by clip_coef = 1.0 / 5.0 = 0.2
        assert_abs_diff_eq!(
            params[0].grad().expect("gradient should be available")[0],
            0.6,
            epsilon = 1e-6
        ); // 3.0 * 0.2
        assert_abs_diff_eq!(
            params[0].grad().expect("gradient should be available")[1],
            0.8,
            epsilon = 1e-6
        ); // 4.0 * 0.2
        assert_abs_diff_eq!(
            params[1].grad().expect("gradient should be available")[0],
            0.0,
            epsilon = 1e-6
        ); // 0.0 * 0.2
    }

    #[test]
    fn test_clip_grad_norm_exactly_at_threshold() {
        let mut params = vec![Tensor::from_vec(vec![3.0, 4.0], true)];

        // Set gradients with norm exactly equal to max_norm
        params[0].set_grad(ndarray::arr1(&[3.0, 4.0])); // norm = 5.0

        let global_norm = clip_grad_norm(&mut params, 5.0);

        assert_abs_diff_eq!(global_norm, 5.0, epsilon = 1e-6);

        // Should not be clipped (norm == max_norm, not >)
        assert_abs_diff_eq!(
            params[0].grad().expect("gradient should be available")[0],
            3.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            params[0].grad().expect("gradient should be available")[1],
            4.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_clip_grad_norm_preserves_relative_magnitudes() {
        let mut params = vec![Tensor::from_vec(vec![1.0], true), Tensor::from_vec(vec![1.0], true)];

        // Set gradients with different magnitudes
        params[0].set_grad(ndarray::arr1(&[10.0]));
        params[1].set_grad(ndarray::arr1(&[5.0]));

        // Global norm = sqrt(10^2 + 5^2) = sqrt(125) ≈ 11.18
        let _global_norm = clip_grad_norm(&mut params, 1.0);

        let grad0 = params[0].grad().expect("gradient should be available")[0];
        let grad1 = params[1].grad().expect("gradient should be available")[0];

        // Relative magnitude should be preserved: grad0 / grad1 ≈ 10 / 5 = 2
        assert_abs_diff_eq!(grad0 / grad1, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_no_gradients() {
        // Parameters without gradients
        let mut params = vec![
            Tensor::from_vec(vec![1.0, 2.0], false), // requires_grad = false
            Tensor::from_vec(vec![3.0], false),
        ];

        let global_norm = clip_grad_norm(&mut params, 1.0);

        // Global norm should be 0 (no gradients)
        assert_abs_diff_eq!(global_norm, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_mixed_gradients() {
        // Some params have gradients, others don't
        let mut params = vec![Tensor::from_vec(vec![1.0], true), Tensor::from_vec(vec![1.0], true)];

        params[0].set_grad(ndarray::arr1(&[3.0]));
        // params[1] has no gradient set

        // Global norm = sqrt(3^2) = 3.0
        let global_norm = clip_grad_norm(&mut params, 1.0);

        assert_abs_diff_eq!(global_norm, 3.0, epsilon = 1e-6);

        // Only params[0] should be clipped
        assert_abs_diff_eq!(
            params[0].grad().expect("gradient should be available")[0],
            1.0,
            epsilon = 1e-6
        ); // 3.0 * (1.0/3.0)
        assert!(params[1].grad().is_none()); // No gradient
    }

    #[test]
    fn test_clip_grad_norm_zero_max_norm() {
        let mut params = vec![Tensor::from_vec(vec![1.0], true)];
        params[0].set_grad(ndarray::arr1(&[5.0]));

        let global_norm = clip_grad_norm(&mut params, 0.0);

        assert_abs_diff_eq!(global_norm, 5.0, epsilon = 1e-6);

        // With max_norm = 0, gradients should be clipped to 0
        assert_abs_diff_eq!(
            params[0].grad().expect("gradient should be available")[0],
            0.0,
            epsilon = 1e-6
        );
    }

    // ── clip_grad_norm_refs tests (SSC-025) ──────────────────────────

    #[test]
    fn test_clip_grad_norm_refs_no_clipping() {
        let mut p0 = Tensor::from_vec(vec![1.0, 2.0], true);
        let mut p1 = Tensor::from_vec(vec![3.0], true);
        p0.set_grad(ndarray::arr1(&[0.1, 0.2]));
        p1.set_grad(ndarray::arr1(&[0.1]));

        let global_norm = clip_grad_norm_refs(&mut [&mut p0, &mut p1], 1.0);
        assert_abs_diff_eq!(global_norm, 0.245, epsilon = 1e-3);

        assert_abs_diff_eq!(
            p0.grad().expect("gradient should be available")[0],
            0.1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            p0.grad().expect("gradient should be available")[1],
            0.2,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            p1.grad().expect("gradient should be available")[0],
            0.1,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_clip_grad_norm_refs_with_clipping() {
        let mut p0 = Tensor::from_vec(vec![1.0, 2.0], true);
        let mut p1 = Tensor::from_vec(vec![3.0], true);
        p0.set_grad(ndarray::arr1(&[3.0, 4.0]));
        p1.set_grad(ndarray::arr1(&[0.0]));

        let global_norm = clip_grad_norm_refs(&mut [&mut p0, &mut p1], 1.0);
        assert_abs_diff_eq!(global_norm, 5.0, epsilon = 1e-6);

        assert_abs_diff_eq!(
            p0.grad().expect("gradient should be available")[0],
            0.6,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            p0.grad().expect("gradient should be available")[1],
            0.8,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            p1.grad().expect("gradient should be available")[0],
            0.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_clip_grad_norm_refs_preserves_relative_magnitudes() {
        let mut p0 = Tensor::from_vec(vec![1.0], true);
        let mut p1 = Tensor::from_vec(vec![1.0], true);
        p0.set_grad(ndarray::arr1(&[10.0]));
        p1.set_grad(ndarray::arr1(&[5.0]));

        let _global_norm = clip_grad_norm_refs(&mut [&mut p0, &mut p1], 1.0);

        let grad0 = p0.grad().expect("gradient should be available")[0];
        let grad1 = p1.grad().expect("gradient should be available")[0];
        assert_abs_diff_eq!(grad0 / grad1, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_refs_no_gradients() {
        let mut p0 = Tensor::from_vec(vec![1.0, 2.0], false);
        let mut p1 = Tensor::from_vec(vec![3.0], false);

        let global_norm = clip_grad_norm_refs(&mut [&mut p0, &mut p1], 1.0);
        assert_abs_diff_eq!(global_norm, 0.0, epsilon = 1e-6);
    }
}
