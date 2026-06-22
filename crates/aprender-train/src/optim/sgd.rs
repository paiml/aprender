//! Stochastic Gradient Descent optimizer

use super::Optimizer;
use crate::Tensor;
use ndarray::Array1;

/// SGD optimizer with optional momentum
pub struct SGD {
    lr: f32,
    momentum: f32,
    velocities: Vec<Option<Array1<f32>>>,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self { lr, momentum, velocities: Vec::new() }
    }

    /// Initialize velocities if needed
    fn ensure_velocities(&mut self, params: &[Tensor]) {
        if self.velocities.is_empty() {
            self.velocities = params.iter().map(|_| None).collect();
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [Tensor]) {
        self.ensure_velocities(params);

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                // Use SIMD for large tensors (>= 16 elements for meaningful speedup)
                if grad.len() >= 16 {
                    let grad_slice = grad.as_slice().expect("grad array is contiguous");
                    let param_slice =
                        param.data_mut().as_slice_mut().expect("param array is contiguous");

                    if self.momentum > 0.0 {
                        // Initialize velocity if needed
                        if self.velocities[i].is_none() {
                            self.velocities[i] = Some(Array1::zeros(grad.len()));
                        }

                        let velocity =
                            self.velocities[i].as_mut().expect("velocity buffer initialized above");
                        let velocity_slice =
                            velocity.as_slice_mut().expect("velocity array is contiguous");

                        // PyTorch SGD+momentum (F-SGD-MOMENTUM-LRSCHED-001):
                        // buffer stays UNSCALED so a mid-training lr change
                        // (LR schedule) applies the FRESH lr each step.
                        //   b = momentum * b + grad
                        //   param -= lr * b   (lr read fresh, not baked into b)
                        // First scale the buffer by momentum.
                        for v in velocity_slice.iter_mut() {
                            *v *= self.momentum;
                        }

                        // b += grad (a=1.0) using SIMD axpy
                        super::simd::simd_axpy(1.0, grad_slice, velocity_slice);

                        // param += -lr * b (lr applied fresh at update time)
                        super::simd::simd_axpy(-self.lr, velocity_slice, param_slice);
                    } else {
                        // Simple SGD: param -= lr * grad (using SIMD axpy)
                        super::simd::simd_axpy(-self.lr, grad_slice, param_slice);
                    }
                } else {
                    // Fallback to scalar implementation for small tensors
                    if self.momentum > 0.0 {
                        // PyTorch SGD+momentum (F-SGD-MOMENTUM-LRSCHED-001):
                        // UNSCALED buffer b = momentum * b + grad, then
                        // param -= lr * b with lr read FRESH each step (so an
                        // LR schedule never carries a stale lr in the buffer).
                        let velocity = if let Some(v) = &self.velocities[i] {
                            v * self.momentum + &grad
                        } else {
                            grad.clone()
                        };

                        *param.data_mut() = param.data() - &(&velocity * self.lr);
                        self.velocities[i] = Some(velocity);
                    } else {
                        // Simple SGD: param -= lr * grad
                        *param.data_mut() = param.data() - &(&grad * self.lr);
                    }
                }
            }
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_small_tensor_no_momentum() {
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        param.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));

        let mut opt = SGD::new(0.1, 0.0);
        opt.step(&mut [param.clone()]);
        // Small tensor path, no momentum
    }

    #[test]
    fn test_sgd_small_tensor_with_momentum() {
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        param.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));

        let mut opt = SGD::new(0.1, 0.9);
        // First step initializes velocity from scratch
        opt.step(&mut [param.clone()]);

        // Second step uses existing velocity
        param.set_grad(Array1::from_vec(vec![0.1, 0.2, 0.3]));
        opt.step(&mut [param.clone()]);
    }

    #[test]
    fn test_sgd_large_tensor_with_momentum() {
        // >= 16 elements to trigger SIMD path
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let grad: Vec<f32> = vec![0.1; 20];

        let param = Tensor::from_vec(data, true);
        param.set_grad(Array1::from_vec(grad.clone()));

        let mut opt = SGD::new(0.1, 0.9);
        opt.step(&mut [param.clone()]);

        // Second step with existing velocity
        param.set_grad(Array1::from_vec(grad));
        opt.step(&mut [param.clone()]);
    }

    #[test]
    fn test_sgd_lr_getter_setter() {
        let mut opt = SGD::new(0.1, 0.0);
        assert!((opt.lr() - 0.1).abs() < 1e-6);
        opt.set_lr(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-6);
    }

    /// FALSIFY F-SGD-MOMENTUM-LRSCHED-001 (scalar path):
    /// SGD-with-momentum must match PyTorch under an LR schedule.
    ///
    /// PyTorch SGD+momentum: `b = mu*b + g` (UNSCALED buffer), then
    /// `theta -= lr*b` (lr read FRESH each step). aprender previously baked
    /// `lr` into the velocity buffer, so after `set_lr` the momentum term
    /// carried a STALE lr → divergence.
    ///
    /// Closed-form (g=1.0, mu=0.9, theta0=0.0, lr 0.1 → set_lr(0.01)):
    ///   b1 = 0.9*0 + 1   = 1.0   ; theta1 = 0    - 0.1 *1.0 = -0.1
    ///   b2 = 0.9*1 + 1   = 1.9   ; theta2 = -0.1 - 0.01*1.9 = -0.119
    /// On the buggy (lr-baked) path theta2 = -0.200 (~40% off).
    #[test]
    fn falsify_sgd_momentum_lrsched_scalar() {
        // 1 element → scalar fallback path (< 16 elements).
        let param = Tensor::from_vec(vec![0.0], true);
        let mut opt = SGD::new(0.1, 0.9);

        param.set_grad(Array1::from_vec(vec![1.0]));
        let mut params = [param];
        opt.step(&mut params);
        // theta1 = -0.1
        assert!(
            (params[0].data()[0] - (-0.1)).abs() < 1e-6,
            "FALSIFIED: theta1 = {} != -0.1 (PyTorch step 1)",
            params[0].data()[0]
        );

        opt.set_lr(0.01);
        params[0].set_grad(Array1::from_vec(vec![1.0]));
        opt.step(&mut params);
        // theta2 = -0.119 (PyTorch). Buggy lr-baked path gives -0.200.
        assert!(
            (params[0].data()[0] - (-0.119)).abs() < 1e-6,
            "FALSIFIED F-SGD-MOMENTUM-LRSCHED-001 (scalar): theta2 = {} != -0.119 \
             (PyTorch rule b=mu*b+g, theta-=lr*b). lr baked into velocity buffer?",
            params[0].data()[0]
        );
    }

    /// FALSIFY F-SGD-MOMENTUM-LRSCHED-001 (SIMD path): identical assertion as
    /// the scalar test but with 16 elements (>= 16 triggers the SIMD path).
    /// Every element is independent and shares the same closed-form, so each
    /// must equal -0.119 after the lr-scheduled second step.
    #[test]
    fn falsify_sgd_momentum_lrsched_simd() {
        let n = 16;
        let param = Tensor::from_vec(vec![0.0; n], true);
        let mut opt = SGD::new(0.1, 0.9);

        param.set_grad(Array1::from_vec(vec![1.0; n]));
        let mut params = [param];
        opt.step(&mut params);
        for &x in params[0].data().iter() {
            assert!(
                (x - (-0.1)).abs() < 1e-6,
                "FALSIFIED: theta1 = {x} != -0.1 (SIMD, PyTorch step 1)"
            );
        }

        opt.set_lr(0.01);
        params[0].set_grad(Array1::from_vec(vec![1.0; n]));
        opt.step(&mut params);
        for &x in params[0].data().iter() {
            assert!(
                (x - (-0.119)).abs() < 1e-6,
                "FALSIFIED F-SGD-MOMENTUM-LRSCHED-001 (SIMD): theta2 = {x} != -0.119 \
                 (PyTorch rule b=mu*b+g, theta-=lr*b)."
            );
        }
    }

    /// Control: constant lr must NOT regress. With lr fixed at 0.1, two steps
    /// of (g=1.0, mu=0.9, theta0=0.0):
    ///   b1=1.0; theta1 = -0.1
    ///   b2=1.9; theta2 = -0.1 - 0.1*1.9 = -0.29
    /// This holds identically for the old and new rules (lr never changes).
    #[test]
    fn test_sgd_momentum_constant_lr_no_regression() {
        // Scalar path.
        let param = Tensor::from_vec(vec![0.0], true);
        let mut opt = SGD::new(0.1, 0.9);
        param.set_grad(Array1::from_vec(vec![1.0]));
        let mut params = [param];
        opt.step(&mut params);
        assert!((params[0].data()[0] - (-0.1)).abs() < 1e-6);
        params[0].set_grad(Array1::from_vec(vec![1.0]));
        opt.step(&mut params);
        assert!(
            (params[0].data()[0] - (-0.29)).abs() < 1e-6,
            "constant-lr scalar regression: theta2 = {} != -0.29",
            params[0].data()[0]
        );

        // SIMD path (16 elements), same closed-form.
        let n = 16;
        let param = Tensor::from_vec(vec![0.0; n], true);
        let mut opt = SGD::new(0.1, 0.9);
        param.set_grad(Array1::from_vec(vec![1.0; n]));
        let mut params = [param];
        opt.step(&mut params);
        params[0].set_grad(Array1::from_vec(vec![1.0; n]));
        opt.step(&mut params);
        for &x in params[0].data().iter() {
            assert!(
                (x - (-0.29)).abs() < 1e-6,
                "constant-lr SIMD regression: theta2 = {x} != -0.29"
            );
        }
    }

    #[test]
    fn test_sgd_no_grad_skips() {
        let param = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
        // No gradient set

        let mut opt = SGD::new(0.1, 0.0);
        opt.step(&mut [param.clone()]); // Should not panic
    }
}
