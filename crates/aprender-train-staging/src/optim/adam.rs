//! Adam optimizer

use super::Optimizer;
use crate::Tensor;
use ndarray::Array1;

/// Adam optimizer (Adaptive Moment Estimation)
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: u64,
    m: Vec<Option<Array1<f32>>>, // First moment
    v: Vec<Option<Array1<f32>>>, // Second moment
}

impl Adam {
    /// Create a new Adam optimizer
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self { lr, beta1, beta2, epsilon, t: 0, m: Vec::new(), v: Vec::new() }
    }

    /// Create Adam with default parameters
    pub fn default_params(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8)
    }

    /// Initialize moments if needed
    fn ensure_moments(&mut self, params: &[Tensor]) {
        if self.m.is_empty() {
            self.m = params.iter().map(|_| None).collect();
            self.v = params.iter().map(|_| None).collect();
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [Tensor]) {
        self.ensure_moments(params);
        self.t += 1;

        // Bias adjust factors
        let lr_t = self.lr
            * ((1.0 - self.beta2.powi(self.t as i32)).sqrt()
                / (1.0 - self.beta1.powi(self.t as i32)));

        for (i, param) in params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                // Use SIMD for large tensors (>= 16 elements for meaningful speedup)
                if grad.len() >= 16 {
                    // Initialize moments if needed
                    if self.m[i].is_none() {
                        self.m[i] = Some(Array1::zeros(grad.len()));
                        self.v[i] = Some(Array1::zeros(grad.len()));
                    }

                    let m = self.m[i].as_mut().expect("momentum buffer initialized above");
                    let v = self.v[i].as_mut().expect("velocity buffer initialized above");

                    // Get mutable slices (arrays are always contiguous)
                    let grad_slice = grad.as_slice().expect("grad array is contiguous");
                    let m_slice = m.as_slice_mut().expect("momentum array is contiguous");
                    let v_slice = v.as_slice_mut().expect("velocity array is contiguous");
                    let param_slice =
                        param.data_mut().as_slice_mut().expect("param array is contiguous");

                    // Use SIMD-accelerated update
                    super::simd::simd_adam_update(
                        grad_slice,
                        m_slice,
                        v_slice,
                        param_slice,
                        self.beta1,
                        self.beta2,
                        lr_t,
                        self.epsilon,
                    );
                } else {
                    // Fallback to scalar implementation for small tensors
                    // m_t = β1 * m_{t-1} + (1 - β1) * g
                    let m_t = if let Some(m) = &self.m[i] {
                        m * self.beta1 + &grad * (1.0 - self.beta1)
                    } else {
                        &grad * (1.0 - self.beta1)
                    };

                    // v_t = β2 * v_{t-1} + (1 - β2) * g²
                    let grad_sq = &grad * &grad;
                    let v_t = if let Some(v) = &self.v[i] {
                        v * self.beta2 + &grad_sq * (1.0 - self.beta2)
                    } else {
                        &grad_sq * (1.0 - self.beta2)
                    };

                    // θ_t = θ_{t-1} - lr_t * m_t / (√v_t + ε)
                    let update = &m_t / &(v_t.mapv(f32::sqrt) + self.epsilon) * lr_t;
                    *param.data_mut() = param.data() - &update;

                    self.m[i] = Some(m_t);
                    self.v[i] = Some(v_t);
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
    use crate::autograd::*;

    #[test]
    fn test_adam_quadratic_convergence() {
        // Test convergence on f(x) = x²
        let mut params = vec![Tensor::from_vec(vec![5.0, -3.0, 2.0], true)];
        let mut optimizer = Adam::default_params(0.1);

        for _ in 0..100 {
            // Compute gradient: ∇(x²) = 2x
            let grad = params[0].data().mapv(|x| 2.0 * x);
            params[0].set_grad(grad);

            optimizer.step(&mut params);
        }

        // Should converge close to 0
        for &val in params[0].data() {
            assert!(val.abs() < 0.5, "Value {val} did not converge");
        }
    }
}
