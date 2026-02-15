
impl IntegratedGradients {
    /// Create new Integrated Gradients explainer.
    ///
    /// # Arguments
    ///
    /// * `n_steps` - Number of steps for Riemann approximation (default: 50)
    #[must_use]
    pub fn new(n_steps: usize) -> Self {
        Self { n_steps }
    }

    /// Compute attributions using numerical gradient approximation.
    ///
    /// # Arguments
    ///
    /// * `model_fn` - Prediction function
    /// * `sample` - Input to explain
    /// * `baseline` - Baseline input (typically zeros)
    ///
    /// # Returns
    ///
    /// Attribution for each input feature
    pub fn attribute<F>(
        &self,
        model_fn: F,
        sample: &Vector<f32>,
        baseline: &Vector<f32>,
    ) -> Vector<f32>
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        assert_eq!(
            sample.len(),
            baseline.len(),
            "Sample and baseline must have same length"
        );

        let n = sample.len();
        let mut attributions = vec![0.0f32; n];

        // Compute path integral using Riemann sum
        for step in 0..self.n_steps {
            let alpha = (step as f32 + 0.5) / self.n_steps as f32;

            // Interpolated point
            let mut x_interp: Vec<f32> = (0..n)
                .map(|i| baseline[i] + alpha * (sample[i] - baseline[i]))
                .collect();

            // Numerical gradient at this point
            let eps = 1e-4;
            for i in 0..n {
                let original = x_interp[i];

                x_interp[i] = original + eps;
                let f_plus = model_fn(&Vector::from_slice(&x_interp));

                x_interp[i] = original - eps;
                let f_minus = model_fn(&Vector::from_slice(&x_interp));

                x_interp[i] = original;

                let grad = (f_plus - f_minus) / (2.0 * eps);
                attributions[i] += grad / self.n_steps as f32;
            }
        }

        // Multiply by (x - x')
        for i in 0..n {
            attributions[i] *= sample[i] - baseline[i];
        }

        Vector::from_slice(&attributions)
    }
}

impl Default for IntegratedGradients {
    fn default() -> Self {
        Self::new(50)
    }
}

/// LIME (Local Interpretable Model-agnostic Explanations).
///
/// Explains predictions by fitting a simple interpretable model locally
/// around the instance being explained.
///
/// # Algorithm
///
/// 1. Generate perturbed samples around the instance
/// 2. Get model predictions for each perturbation
/// 3. Weight samples by proximity to original instance
/// 4. Fit weighted linear model to the perturbations
/// 5. Return linear model coefficients as explanations
///
/// # Reference
///
/// - Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining
///   the Predictions of Any Classifier. KDD.
#[derive(Debug)]
pub struct LIME {
    /// Number of perturbed samples to generate
    n_samples: usize,
    /// Kernel width for proximity weighting
    kernel_width: f32,
}

impl LIME {
    /// Create new LIME explainer.
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of perturbations to generate (default: 1000)
    /// * `kernel_width` - Width of exponential kernel (default: 0.75)
    #[must_use]
    pub fn new(n_samples: usize, kernel_width: f32) -> Self {
        Self {
            n_samples,
            kernel_width,
        }
    }

    /// Explain a prediction by fitting a local linear model.
    ///
    /// # Arguments
    ///
    /// * `model_fn` - Prediction function
    /// * `sample` - Instance to explain
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Coefficients of the local linear model (feature importance)
    pub fn explain<F>(&self, model_fn: F, sample: &Vector<f32>, seed: u64) -> LIMEExplanation
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n = sample.len();
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate perturbed samples
        let mut perturbed_data = Vec::with_capacity(self.n_samples);
        let mut predictions = Vec::with_capacity(self.n_samples);
        let mut weights = Vec::with_capacity(self.n_samples);

        // Original prediction
        let original_pred = model_fn(sample);

        for _ in 0..self.n_samples {
            // Create perturbation (random feature masking)
            let mut perturbed = sample.clone();
            let mut distance_sq = 0.0_f32;

            for i in 0..n {
                if rng.gen::<f32>() < 0.5 {
                    // Perturb this feature by adding noise
                    let noise = rng.gen_range(-0.5..0.5);
                    perturbed[i] += noise * sample[i].abs().max(1.0);
                    distance_sq += noise * noise;
                }
            }

            let pred = model_fn(&perturbed);
            let weight = (-distance_sq / (2.0 * self.kernel_width * self.kernel_width)).exp();

            perturbed_data.push(perturbed);
            predictions.push(pred);
            weights.push(weight);
        }

        // Fit weighted linear regression
        // Using simplified normal equations: (X^T W X)^-1 X^T W y
        let coefficients =
            Self::fit_weighted_linear(&perturbed_data, &predictions, &weights, sample);

        LIMEExplanation {
            coefficients: Vector::from_slice(&coefficients),
            intercept: original_pred
                - coefficients
                    .iter()
                    .zip(sample.as_slice())
                    .map(|(&c, &x)| c * x)
                    .sum::<f32>(),
            original_prediction: original_pred,
        }
    }

    /// Fit weighted linear regression.
    fn fit_weighted_linear(
        x_data: &[Vector<f32>],
        y: &[f32],
        weights: &[f32],
        reference: &Vector<f32>,
    ) -> Vec<f32> {
        let n = reference.len();
        let _m = x_data.len();

        // Compute relative features (difference from reference)
        let x_centered: Vec<Vec<f32>> = x_data
            .iter()
            .map(|xi| {
                xi.as_slice()
                    .iter()
                    .zip(reference.as_slice())
                    .map(|(&x, &r)| x - r)
                    .collect()
            })
            .collect();

        // Y centered around mean
        let y_mean: f32 = y.iter().zip(weights).map(|(&yi, &w)| yi * w).sum::<f32>()
            / weights.iter().sum::<f32>().max(1e-10);

        // Weighted covariance
        let mut xtx = vec![vec![0.0_f32; n]; n];
        let mut xty = vec![0.0_f32; n];

        for (i, (xi, &wi)) in x_centered.iter().zip(weights).enumerate() {
            let yi_centered = y[i] - y_mean;
            for j in 0..n {
                xty[j] += wi * xi[j] * yi_centered;
                for k in 0..n {
                    xtx[j][k] += wi * xi[j] * xi[k];
                }
            }
        }

        // Add regularization
        for (j, row) in xtx.iter_mut().enumerate() {
            row[j] += 0.01;
        }

        // Solve using simple Gaussian elimination
        Self::solve_linear_system(&xtx, &xty)
    }

    /// Solve linear system Ax = b using Gaussian elimination.
    #[allow(clippy::needless_range_loop)]
    fn solve_linear_system(a: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
        let n = b.len();
        let mut aug: Vec<Vec<f32>> = a
            .iter()
            .zip(b)
            .map(|(row, &bi)| {
                let mut r = row.clone();
                r.push(bi);
                r
            })
            .collect();

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[k][i].abs() > aug[max_row][i].abs() {
                    max_row = k;
                }
            }
            aug.swap(i, max_row);

            let pivot = aug[i][i];
            if pivot.abs() < 1e-10 {
                continue;
            }

            // Eliminate below
            for k in (i + 1)..n {
                let factor = aug[k][i] / pivot;
                for j in i..=n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0_f32; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            let pivot = aug[i][i];
            x[i] = if pivot.abs() > 1e-10 {
                sum / pivot
            } else {
                0.0
            };
        }
        x
    }

    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    #[must_use]
    pub fn kernel_width(&self) -> f32 {
        self.kernel_width
    }
}

impl Default for LIME {
    fn default() -> Self {
        Self::new(500, 0.75)
    }
}

/// Saliency Maps for neural network visualization.
///
/// Computes gradients of the output with respect to input features,
/// showing which parts of the input most affect the prediction.
///
/// # Method
///
/// Uses numerical gradient estimation via finite differences:
/// ```text
/// saliency_i = ∂f/∂x_i ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)
/// ```
///
/// # Reference
///
/// - Simonyan, K., et al. (2014). Deep Inside Convolutional Networks:
///   Visualising Image Classification Models and Saliency Maps.
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// Epsilon for numerical gradient
    epsilon: f32,
}

impl SaliencyMap {
    /// Create a new saliency map calculator.
    #[must_use]
    pub fn new() -> Self {
        Self { epsilon: 1e-4 }
    }

    /// Create with custom epsilon.
    #[must_use]
    pub fn with_epsilon(epsilon: f32) -> Self {
        Self { epsilon }
    }

    /// Get epsilon value.
    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Compute saliency (gradient) for each input feature.
    ///
    /// # Arguments
    ///
    /// * `model_fn` - Prediction function
    /// * `sample` - Input to explain
    ///
    /// # Returns
    ///
    /// Gradient values for each input feature.
    pub fn compute<F>(&self, model_fn: F, sample: &Vector<f32>) -> Vector<f32>
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        let n = sample.len();
        let mut gradients = vec![0.0f32; n];

        for i in 0..n {
            // Forward difference
            let mut x_plus = sample.clone();
            x_plus[i] += self.epsilon;
            let f_plus = model_fn(&x_plus);

            // Backward difference
            let mut x_minus = sample.clone();
            x_minus[i] -= self.epsilon;
            let f_minus = model_fn(&x_minus);

            // Central difference gradient
            gradients[i] = (f_plus - f_minus) / (2.0 * self.epsilon);
        }

        Vector::from_slice(&gradients)
    }

    /// Compute absolute saliency (magnitude of gradient).
    ///
    /// Useful for visualizing which features matter, regardless of direction.
    pub fn compute_absolute<F>(&self, model_fn: F, sample: &Vector<f32>) -> Vector<f32>
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        let gradients = self.compute(model_fn, sample);
        let abs_grads: Vec<f32> = gradients.as_slice().iter().map(|&g| g.abs()).collect();
        Vector::from_slice(&abs_grads)
    }

    /// Compute smooth gradient by averaging over noisy samples.
    ///
    /// `SmoothGrad` reduces noise in saliency maps by averaging gradients
    /// over multiple noisy versions of the input.
    ///
    /// # Reference
    ///
    /// - Smilkov, D., et al. (2017). `SmoothGrad`: removing noise by adding noise.
    pub fn smooth_grad<F>(
        &self,
        model_fn: F,
        sample: &Vector<f32>,
        n_samples: usize,
        noise_level: f32,
    ) -> Vector<f32>
    where
        F: Fn(&Vector<f32>) -> f32,
    {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = sample.len();
        let mut sum_grads = vec![0.0f32; n];

        for _ in 0..n_samples {
            // Add noise to sample
            let noisy: Vec<f32> = sample
                .as_slice()
                .iter()
                .map(|&x| x + rng.gen_range(-noise_level..noise_level))
                .collect();
            let noisy_sample = Vector::from_slice(&noisy);

            // Compute gradient for noisy sample
            let grad = self.compute(&model_fn, &noisy_sample);

            for (j, &g) in grad.as_slice().iter().enumerate() {
                sum_grads[j] += g;
            }
        }

        // Average
        for g in &mut sum_grads {
            *g /= n_samples as f32;
        }

        Vector::from_slice(&sum_grads)
    }
}
