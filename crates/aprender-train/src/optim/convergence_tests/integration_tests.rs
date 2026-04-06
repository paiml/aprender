//! Integration tests that compare different optimizers

#[cfg(test)]
mod tests {
    use crate::optim::*;
    use crate::Tensor;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_adam_faster_than_sgd() {
        let mut params_adam = vec![Tensor::from_vec(vec![10.0, -10.0], true)];
        let mut params_sgd = vec![Tensor::from_vec(vec![10.0, -10.0], true)];

        let mut adam = Adam::default_params(0.1);
        let mut sgd = SGD::new(0.1, 0.0);

        for _ in 0..30 {
            // Same gradient for both
            let grad = params_adam[0].data().mapv(|x| 2.0 * x);
            params_adam[0].set_grad(grad.clone());
            params_sgd[0].set_grad(grad);

            adam.step(&mut params_adam);
            sgd.step(&mut params_sgd);
        }

        // Adam typically converges faster on this problem
        let adam_norm: f32 = params_adam[0].data().iter().map(|&x| x * x).sum::<f32>().sqrt();
        let sgd_norm: f32 = params_sgd[0].data().iter().map(|&x| x * x).sum::<f32>().sqrt();

        assert!(adam_norm < sgd_norm);
    }

    #[test]
    fn test_optimizer_with_zero_gradients() {
        let mut params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        params[0].set_grad(ndarray::arr1(&[0.0, 0.0]));

        let mut adam = Adam::default_params(0.1);
        let initial = params[0].data().to_owned();

        adam.step(&mut params);

        // With zero gradients, Adam should still update due to momentum
        // but the change should be minimal after one step
        for i in 0..2 {
            assert_abs_diff_eq!(params[0].data()[i], initial[i], epsilon = 0.1);
        }
    }

    #[test]
    fn test_gradient_clipping_integration() {
        use crate::optim::clip_grad_norm;

        let mut params = vec![Tensor::from_vec(vec![1.0], true)];

        // Set large gradient
        params[0].set_grad(ndarray::arr1(&[100.0]));

        // Clip to max_norm = 1.0
        let global_norm = clip_grad_norm(&mut params, 1.0);

        assert_abs_diff_eq!(global_norm, 100.0, epsilon = 1e-6);
        assert_abs_diff_eq!(
            params[0].grad().expect("gradient should be available")[0],
            1.0,
            epsilon = 1e-6
        );

        // Now optimizer step with clipped gradient
        let mut adam = Adam::default_params(0.1);
        adam.step(&mut params);

        // Should have moved, but not by the full 100.0 gradient
        assert!(params[0].data()[0] < 1.0);
        assert!(params[0].data()[0] > 0.5);
    }

    #[test]
    fn test_learning_rate_scheduler_integration() {
        use crate::optim::{CosineAnnealingLR, LRScheduler};

        let mut params = vec![Tensor::from_vec(vec![5.0], true)];
        let mut optimizer = SGD::new(0.3, 0.0);
        let mut scheduler = CosineAnnealingLR::default_min(0.3, 10);

        let mut losses = Vec::new();

        for _ in 0..10 {
            // Compute loss and gradient
            let x = params[0].data()[0];
            losses.push(x * x);

            let grad = ndarray::arr1(&[2.0 * x]);
            params[0].set_grad(grad);

            // Update with current learning rate
            scheduler.apply(&mut optimizer);
            optimizer.step(&mut params);
            scheduler.step();
        }

        // Loss should decrease over time
        for i in 1..losses.len() {
            assert!(losses[i] < losses[i - 1]);
        }

        // Final loss should be small
        assert!(losses[losses.len() - 1] < 1.0);
    }
}
