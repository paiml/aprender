use crate::primitives::Vector;

use super::Adam;
use crate::optim::Optimizer;

impl Adam {
    /// Creates a new Adam optimizer with the given learning rate and default hyperparameters.
    ///
    /// Default values:
    /// - beta1 = 0.9
    /// - beta2 = 0.999
    /// - epsilon = 1e-8
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Step size (typical values: 0.001, 0.0001)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    ///
    /// let optimizer = Adam::new(0.001);
    /// assert!((optimizer.learning_rate() - 0.001).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: None,
            v: None,
            t: 0,
        }
    }

    /// Sets the beta1 parameter (exponential decay rate for first moment).
    ///
    /// # Arguments
    ///
    /// * `beta1` - Value between 0.0 and 1.0 (typical: 0.9)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    ///
    /// let optimizer = Adam::new(0.001).with_beta1(0.95);
    /// assert!((optimizer.beta1() - 0.95).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets the beta2 parameter (exponential decay rate for second moment).
    ///
    /// # Arguments
    ///
    /// * `beta2` - Value between 0.0 and 1.0 (typical: 0.999)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    ///
    /// let optimizer = Adam::new(0.001).with_beta2(0.9999);
    /// assert!((optimizer.beta2() - 0.9999).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Sets the epsilon parameter (numerical stability constant).
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Small positive value (typical: 1e-8)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    ///
    /// let optimizer = Adam::new(0.001).with_epsilon(1e-7);
    /// assert!((optimizer.epsilon() - 1e-7).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Returns the learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the beta1 parameter.
    #[must_use]
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Returns the beta2 parameter.
    #[must_use]
    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Returns the epsilon parameter.
    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Returns the number of steps taken.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.t
    }

    /// Updates parameters using gradients with adaptive learning rates.
    ///
    /// # Arguments
    ///
    /// * `params` - Mutable reference to parameter vector
    /// * `gradients` - Gradient vector (same length as params)
    ///
    /// # Panics
    ///
    /// Panics if params and gradients have different lengths.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    /// use aprender::primitives::Vector;
    ///
    /// let mut optimizer = Adam::new(0.001);
    /// let mut params = Vector::from_slice(&[1.0, 2.0]);
    /// let gradients = Vector::from_slice(&[0.1, 0.2]);
    ///
    /// optimizer.step(&mut params, &gradients);
    /// ```
    pub fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>) {
        assert_eq!(
            params.len(),
            gradients.len(),
            "Parameters and gradients must have same length"
        );

        let n = params.len();

        // Initialize moment estimates if needed
        if self.m.is_none()
            || self
                .m
                .as_ref()
                .expect("First moment estimate must be initialized")
                .len()
                != n
        {
            self.m = Some(vec![0.0; n]);
            self.v = Some(vec![0.0; n]);
            self.t = 0;
        }

        self.t += 1;
        let t = self.t as f32;

        let m = self.m.as_mut().expect("First moment was just initialized");
        let v = self.v.as_mut().expect("Second moment was just initialized");

        // Compute bias-corrected learning rate
        let lr_t =
            self.learning_rate * (1.0 - self.beta2.powf(t)).sqrt() / (1.0 - self.beta1.powf(t));

        for i in 0..n {
            let g = gradients[i];

            // Update biased first moment estimate
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;

            // Update biased second raw moment estimate
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

            // Update parameters
            params[i] -= lr_t * m[i] / (v[i].sqrt() + self.epsilon);
        }
    }

    /// Resets the optimizer state (moment estimates and step counter).
    ///
    /// Call this when starting training on a new model or after significant
    /// changes to the optimization problem.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::optim::Adam;
    /// use aprender::primitives::Vector;
    ///
    /// let mut optimizer = Adam::new(0.001);
    /// let mut params = Vector::from_slice(&[1.0]);
    /// let gradients = Vector::from_slice(&[1.0]);
    ///
    /// optimizer.step(&mut params, &gradients);
    /// assert_eq!(optimizer.steps(), 1);
    ///
    /// optimizer.reset();
    /// assert_eq!(optimizer.steps(), 0);
    /// ```
    pub fn reset(&mut self) {
        self.m = None;
        self.v = None;
        self.t = 0;
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>) {
        self.step(params, gradients);
    }

    fn reset(&mut self) {
        self.reset();
    }
}
