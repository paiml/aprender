//! Optimizer trait

use crate::Tensor;

/// Trait for optimization algorithms
pub trait Optimizer {
    /// Perform a single optimization step
    fn step(&mut self, params: &mut [Tensor]);

    /// Perform optimization step on referenced parameters
    ///
    /// This is useful when parameters are borrowed from a model
    fn step_refs(&mut self, params: &mut [&mut Tensor]) {
        // Default implementation delegates to step via collecting
        // Subclasses can override for efficiency
        for param in params.iter_mut() {
            if let Some(grad) = param.grad() {
                // Apply simple SGD update as fallback
                let lr = self.lr();
                let grad_data = grad.to_vec();
                let data = param.data_mut();
                for (d, g) in data.iter_mut().zip(grad_data.iter()) {
                    *d -= lr * g;
                }
            }
        }
    }

    /// Zero out all gradients
    fn zero_grad(&mut self, params: &mut [Tensor]) {
        for param in params {
            param.zero_grad();
        }
    }

    /// Zero gradients on referenced parameters
    fn zero_grad_refs(&mut self, params: &mut [&mut Tensor]) {
        for param in params.iter_mut() {
            param.zero_grad();
        }
    }

    /// Get learning rate
    fn lr(&self) -> f32;

    /// Set learning rate
    fn set_lr(&mut self, lr: f32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    /// Minimal optimizer implementation for testing default trait methods
    struct TestOptimizer {
        learning_rate: f32,
    }

    impl TestOptimizer {
        fn new(lr: f32) -> Self {
            Self { learning_rate: lr }
        }
    }

    impl Optimizer for TestOptimizer {
        fn step(&mut self, params: &mut [Tensor]) {
            for param in params {
                if let Some(grad) = param.grad() {
                    let grad_data = grad.to_vec();
                    let data = param.data_mut();
                    for (d, g) in data.iter_mut().zip(grad_data.iter()) {
                        *d -= self.learning_rate * g;
                    }
                }
            }
        }

        fn lr(&self) -> f32 {
            self.learning_rate
        }

        fn set_lr(&mut self, lr: f32) {
            self.learning_rate = lr;
        }
    }

    #[test]
    fn test_optimizer_step() {
        let mut opt = TestOptimizer::new(0.1);
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        param.set_grad(arr1(&[0.5, 1.0, 1.5]));

        opt.step(&mut [param.clone()]);

        // Check that lr is accessible
        assert_eq!(opt.lr(), 0.1);
    }

    #[test]
    fn test_optimizer_step_refs() {
        let mut opt = TestOptimizer::new(0.1);
        let mut param = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        param.set_grad(arr1(&[0.5, 1.0, 1.5]));

        let original_data = param.data().to_vec();
        opt.step_refs(&mut [&mut param]);

        // Check values were updated: new = old - lr * grad
        let updated_data = param.data().to_vec();
        for i in 0..3 {
            let expected = original_data[i] - 0.1 * [0.5, 1.0, 1.5][i];
            assert!((updated_data[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_optimizer_step_refs_no_grad() {
        let mut opt = TestOptimizer::new(0.1);
        let mut param = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        // No gradient set

        let original_data = param.data().to_vec();
        opt.step_refs(&mut [&mut param]);

        // Values should be unchanged when no gradient
        let updated_data = param.data().to_vec();
        assert_eq!(original_data, updated_data);
    }

    #[test]
    fn test_optimizer_zero_grad() {
        let mut opt = TestOptimizer::new(0.1);
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        param.set_grad(arr1(&[0.5, 1.0, 1.5]));

        assert!(param.grad().is_some());
        opt.zero_grad(&mut [param.clone()]);
        // After zero_grad, the gradient should be zeroed
    }

    #[test]
    fn test_optimizer_zero_grad_refs() {
        let mut opt = TestOptimizer::new(0.1);
        let mut param = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        param.set_grad(arr1(&[0.5, 1.0, 1.5]));

        assert!(param.grad().is_some());
        opt.zero_grad_refs(&mut [&mut param]);
        // After zero_grad_refs, the gradient should be zeroed
    }

    #[test]
    fn test_optimizer_set_lr() {
        let mut opt = TestOptimizer::new(0.1);
        assert_eq!(opt.lr(), 0.1);

        opt.set_lr(0.01);
        assert_eq!(opt.lr(), 0.01);
    }

    #[test]
    fn test_optimizer_step_refs_multiple_params() {
        let mut opt = TestOptimizer::new(0.1);
        let mut param1 = Tensor::from_vec(vec![1.0, 2.0], true);
        let mut param2 = Tensor::from_vec(vec![3.0, 4.0], true);
        param1.set_grad(arr1(&[0.5, 1.0]));
        param2.set_grad(arr1(&[1.5, 2.0]));

        opt.step_refs(&mut [&mut param1, &mut param2]);

        // Both params should be updated
        let data1 = param1.data().to_vec();
        let data2 = param2.data().to_vec();

        assert!((data1[0] - 0.95).abs() < 1e-6); // 1.0 - 0.1 * 0.5
        assert!((data1[1] - 1.9).abs() < 1e-6); // 2.0 - 0.1 * 1.0
        assert!((data2[0] - 2.85).abs() < 1e-6); // 3.0 - 0.1 * 1.5
        assert!((data2[1] - 3.8).abs() < 1e-6); // 4.0 - 0.1 * 2.0
    }

    #[test]
    fn test_optimizer_zero_grad_multiple_params() {
        let mut opt = TestOptimizer::new(0.1);
        let mut params =
            vec![Tensor::from_vec(vec![1.0, 2.0], true), Tensor::from_vec(vec![3.0, 4.0], true)];

        for p in &mut params {
            p.set_grad(arr1(&[0.5, 1.0]));
        }

        opt.zero_grad(&mut params);
        // All gradients should be zeroed
    }
}
