
impl Default for SaliencyMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Counterfactual explanation generator.
///
/// Finds the minimal change to an input that changes the model's prediction
/// to a target class. Counterfactuals answer "What would need to change?"
///
/// # Method
///
/// Uses gradient descent to find the closest input in feature space
/// that results in a different classification.
///
/// ```text
/// x_cf = argmin_x' ||x' - x||² s.t. f(x') = target_class
/// ```
///
/// # Reference
///
/// - Wachter, S., et al. (2018). Counterfactual Explanations without Opening
///   the Black Box. Harvard Journal of Law & Technology.
#[derive(Debug, Clone)]
pub struct CounterfactualExplainer {
    /// Maximum optimization iterations
    max_iter: usize,
    /// Step size for gradient descent
    step_size: f32,
    /// Epsilon for numerical gradient
    epsilon: f32,
}

impl CounterfactualExplainer {
    /// Create a new counterfactual explainer.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum iterations for optimization
    /// * `step_size` - Learning rate for gradient descent
    #[must_use]
    pub fn new(max_iter: usize, step_size: f32) -> Self {
        Self {
            max_iter,
            step_size,
            epsilon: 1e-4,
        }
    }

    /// Get max iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Get step size.
    #[must_use]
    pub fn step_size(&self) -> f32 {
        self.step_size
    }

    /// Find a counterfactual explanation.
    ///
    /// # Arguments
    ///
    /// * `original` - Original input
    /// * `target_class` - Desired output class
    /// * `model_fn` - Classification function (returns class index)
    ///
    /// # Returns
    ///
    /// `Some(CounterfactualResult)` if found, `None` if optimization failed.
    pub fn find<F>(
        &self,
        original: &Vector<f32>,
        target_class: usize,
        model_fn: F,
    ) -> Option<CounterfactualResult>
    where
        F: Fn(&Vector<f32>) -> usize,
    {
        let n = original.len();
        let mut current = original.clone();

        for _ in 0..self.max_iter {
            // Check if we've reached target class
            if model_fn(&current) == target_class {
                let distance = Self::euclidean_distance(&current, original);
                return Some(CounterfactualResult {
                    counterfactual: current,
                    original: original.clone(),
                    target_class,
                    distance,
                });
            }

            // Compute gradient toward target class
            // We use the distance to original as a regularizer
            let mut gradient = vec![0.0f32; n];

            for i in 0..n {
                // Numerical gradient of "wrongness" w.r.t. feature i
                let mut x_plus = current.clone();
                x_plus[i] += self.epsilon;

                let mut x_minus = current.clone();
                x_minus[i] -= self.epsilon;

                // Score: 0 if correct class, 1 otherwise
                let score_plus = if model_fn(&x_plus) == target_class {
                    0.0
                } else {
                    1.0
                };
                let score_minus = if model_fn(&x_minus) == target_class {
                    0.0
                } else {
                    1.0
                };

                // Gradient of classification error
                let class_grad = (score_plus - score_minus) / (2.0 * self.epsilon);

                // Gradient of distance regularizer: 2 * (current - original)
                let dist_grad = 2.0 * (current[i] - original[i]);

                // Combined gradient (minimize error + small regularization)
                gradient[i] = class_grad + 0.001 * dist_grad;
            }

            // Update
            for i in 0..n {
                current[i] -= self.step_size * gradient[i];
            }
        }

        // Check if final result achieves target
        if model_fn(&current) == target_class {
            let distance = Self::euclidean_distance(&current, original);
            Some(CounterfactualResult {
                counterfactual: current,
                original: original.clone(),
                target_class,
                distance,
            })
        } else {
            None
        }
    }

    fn euclidean_distance(a: &Vector<f32>, b: &Vector<f32>) -> f32 {
        a.as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Result of counterfactual explanation search.
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// The counterfactual instance
    pub counterfactual: Vector<f32>,
    /// Original instance
    pub original: Vector<f32>,
    /// Target class achieved
    pub target_class: usize,
    /// Distance from original
    pub distance: f32,
}

impl CounterfactualResult {
    /// Get feature changes (counterfactual - original).
    #[must_use]
    pub fn feature_changes(&self) -> Vec<f32> {
        self.counterfactual
            .as_slice()
            .iter()
            .zip(self.original.as_slice())
            .map(|(&cf, &orig)| cf - orig)
            .collect()
    }

    /// Get top k features with largest absolute changes.
    #[must_use]
    pub fn top_changed_features(&self, k: usize) -> Vec<(usize, f32)> {
        let changes = self.feature_changes();
        let mut indexed: Vec<(usize, f32)> = changes.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed
    }
}

/// LIME explanation result.
#[derive(Debug, Clone)]
pub struct LIMEExplanation {
    /// Linear model coefficients (feature importance)
    pub coefficients: Vector<f32>,
    /// Intercept of local model
    pub intercept: f32,
    /// Original model prediction
    pub original_prediction: f32,
}

impl LIMEExplanation {
    /// Get top influential features.
    #[must_use]
    pub fn top_features(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .coefficients
            .as_slice()
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed
    }

    /// Local prediction using the linear model.
    #[must_use]
    pub fn local_prediction(&self, sample: &Vector<f32>) -> f32 {
        self.intercept
            + self
                .coefficients
                .as_slice()
                .iter()
                .zip(sample.as_slice())
                .map(|(&c, &x)| c * x)
                .sum::<f32>()
    }
}

#[cfg(test)]
mod tests;
