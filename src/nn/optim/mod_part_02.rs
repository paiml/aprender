
impl AdamW {
    /// Create a new `AdamW` optimizer.
    ///
    /// Default: β₁=0.9, β₂=0.999, ε=1e-8, `weight_decay=0.01`
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn new(params: Vec<&mut Tensor>, lr: f32) -> Self {
        let param_ids: Vec<TensorId> = params.iter().map(|p| p.id()).collect();
        Self {
            param_ids,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
            initialized: false,
        }
    }

    #[must_use]
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    #[must_use]
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    #[must_use]
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    fn update_param(&mut self, param: &mut Tensor, idx: usize) {
        let Some(grad) = get_grad(param.id()) else {
            return;
        };

        let grad_data = grad.data();
        let param_data = param.data_mut();

        // Initialize state if needed
        if !self.initialized || idx >= self.m.len() {
            if idx >= self.m.len() {
                self.m.resize(idx + 1, Vec::new());
                self.v.resize(idx + 1, Vec::new());
            }
            self.m[idx] = vec![0.0; param_data.len()];
            self.v[idx] = vec![0.0; param_data.len()];
        }

        let m = &mut self.m[idx];
        let v = &mut self.v[idx];

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..param_data.len() {
            let g = grad_data[i];

            // Update moment estimates (no weight decay in gradient)
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            // Decoupled weight decay: applied directly to parameter
            param_data[i] -= self.lr * self.weight_decay * param_data[i];

            // Adam update
            param_data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    pub fn step_with_params(&mut self, params: &mut [&mut Tensor]) {
        self.t += 1;
        for (idx, param) in params.iter_mut().enumerate() {
            self.update_param(param, idx);
        }
        self.initialized = true;
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.t += 1;
        self.initialized = true;
    }

    fn zero_grad(&mut self) {
        for &id in &self.param_ids {
            crate::autograd::clear_grad(id);
        }
    }

    fn lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// `RMSprop` optimizer.
///
/// Maintains a moving average of squared gradients for adaptive learning rates.
///
/// Update rule:
/// ```text
/// v_t = α * v_{t-1} + (1 - α) * grad²
/// param = param - lr * grad / (√v_t + ε)
/// ```
#[derive(Debug)]
pub struct RMSprop {
    param_ids: Vec<TensorId>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    /// Running average of squared gradients
    v: Vec<Vec<f32>>,
    /// Momentum buffer
    buffer: Vec<Vec<f32>>,
    pub(crate) initialized: bool,
}

impl RMSprop {
    /// Create a new `RMSprop` optimizer.
    ///
    /// Default: α=0.99, ε=1e-8
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn new(params: Vec<&mut Tensor>, lr: f32) -> Self {
        let param_ids: Vec<TensorId> = params.iter().map(|p| p.id()).collect();
        Self {
            param_ids,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            v: Vec::new(),
            buffer: Vec::new(),
            initialized: false,
        }
    }

    #[must_use]
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    #[must_use]
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    #[must_use]
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    #[must_use]
    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    fn update_param(&mut self, param: &mut Tensor, idx: usize) {
        let Some(grad) = get_grad(param.id()) else {
            return;
        };

        let grad_data = grad.data();
        let param_data = param.data_mut();

        // Initialize state if needed
        if !self.initialized || idx >= self.v.len() {
            if idx >= self.v.len() {
                self.v.resize(idx + 1, Vec::new());
                self.buffer.resize(idx + 1, Vec::new());
            }
            self.v[idx] = vec![0.0; param_data.len()];
            self.buffer[idx] = vec![0.0; param_data.len()];
        }

        let v = &mut self.v[idx];
        let buffer = &mut self.buffer[idx];

        for i in 0..param_data.len() {
            let mut g = grad_data[i];

            // Weight decay
            if self.weight_decay != 0.0 {
                g += self.weight_decay * param_data[i];
            }

            // Update running average of squared gradients
            v[i] = self.alpha * v[i] + (1.0 - self.alpha) * g * g;

            // Compute update
            let update = g / (v[i].sqrt() + self.eps);

            if self.momentum > 0.0 {
                buffer[i] = self.momentum * buffer[i] + update;
                param_data[i] -= self.lr * buffer[i];
            } else {
                param_data[i] -= self.lr * update;
            }
        }
    }

    pub fn step_with_params(&mut self, params: &mut [&mut Tensor]) {
        for (idx, param) in params.iter_mut().enumerate() {
            self.update_param(param, idx);
        }
        self.initialized = true;
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        self.initialized = true;
    }

    fn zero_grad(&mut self) {
        for &id in &self.param_ids {
            crate::autograd::clear_grad(id);
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
mod tests;
