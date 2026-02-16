#[allow(clippy::wildcard_imports)]
use super::*;
use crate::error::Result;
use crate::primitives::Matrix;

impl LinearSVM {
    /// Creates a new Linear SVM with default parameters.
    ///
    /// # Default Parameters
    ///
    /// - C: 1.0 (moderate regularization)
    /// - `learning_rate`: 0.01
    /// - `max_iter`: 1000
    /// - tol: 1e-4
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: None,
            bias: 0.0,
            c: 1.0,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Sets the regularization parameter C.
    ///
    /// Larger C means less regularization (fit data more closely).
    /// Smaller C means more regularization (simpler model).
    #[must_use]
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Sets the learning rate for subgradient descent.
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Trains the Linear SVM on the given data.
    ///
    /// # Arguments
    ///
    /// - `x`: Feature matrix (`n_samples` × `n_features`)
    /// - `y`: Binary labels (0 or 1)
    ///
    /// # Returns
    ///
    /// Ok(()) on success, Err with message on failure.
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        if x.n_rows() != y.len() {
            return Err("x and y must have the same number of samples".into());
        }

        if x.n_rows() == 0 {
            return Err("Cannot fit with 0 samples".into());
        }

        // Convert labels to {-1, +1}
        let y_signed: Vec<f32> = y
            .iter()
            .map(|&label| if label == 0 { -1.0 } else { 1.0 })
            .collect();

        let n_samples = x.n_rows();
        let n_features = x.n_cols();

        // Initialize weights and bias
        let mut w = vec![0.0; n_features];
        let mut b = 0.0;

        let lambda = 1.0 / (2.0 * n_samples as f32 * self.c);

        // Subgradient descent with learning rate decay
        for epoch in 0..self.max_iter {
            let eta = self.learning_rate / (1.0 + epoch as f32 * 0.01);
            let prev_w = w.clone();
            let prev_b = b;

            // Iterate over all samples (batch update)
            for (i, &y_i) in y_signed.iter().enumerate() {
                // Compute decision value: w·x + b
                let mut decision = b;
                for (j, &w_j) in w.iter().enumerate() {
                    decision += w_j * x.get(i, j);
                }

                // Compute margin: y * (w·x + b)
                let margin = y_i * decision;

                // Subgradient update
                if margin < 1.0 {
                    // Misclassified or within margin: update with hinge loss gradient
                    for (j, w_j) in w.iter_mut().enumerate() {
                        let gradient = 2.0 * lambda * *w_j - y_i * x.get(i, j);
                        *w_j -= eta * gradient;
                    }
                    b += eta * y_i;
                } else {
                    // Correctly classified outside margin: only regularization gradient
                    for w_j in &mut w {
                        let gradient = 2.0 * lambda * *w_j;
                        *w_j -= eta * gradient;
                    }
                }
            }

            // Check convergence (weight change between iterations)
            let mut weight_change = 0.0;
            for j in 0..n_features {
                weight_change += (w[j] - prev_w[j]).powi(2);
            }
            weight_change += (b - prev_b).powi(2);
            weight_change = weight_change.sqrt();

            if weight_change < self.tol {
                break;
            }
        }

        self.weights = Some(w);
        self.bias = b;

        Ok(())
    }

    /// Computes the decision function for the given samples.
    ///
    /// Returns w·x + b for each sample. Positive values indicate class 1,
    /// negative values indicate class 0.
    ///
    /// # Arguments
    ///
    /// - `x`: Feature matrix (`n_samples` × `n_features`)
    ///
    /// # Returns
    ///
    /// Vector of decision values, one per sample.
    pub fn decision_function(&self, x: &Matrix<f32>) -> Result<Vec<f32>> {
        let weights = self.weights.as_ref().ok_or("Model not trained yet")?;

        if x.n_cols() != weights.len() {
            return Err("Feature dimension mismatch".into());
        }

        let mut decisions = Vec::with_capacity(x.n_rows());

        for i in 0..x.n_rows() {
            let mut decision = self.bias;
            for (j, &w_j) in weights.iter().enumerate() {
                decision += w_j * x.get(i, j);
            }
            decisions.push(decision);
        }

        Ok(decisions)
    }

    /// Predicts class labels for the given samples.
    ///
    /// # Arguments
    ///
    /// - `x`: Feature matrix (`n_samples` × `n_features`)
    ///
    /// # Returns
    ///
    /// Vector of predicted labels (0 or 1).
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let decisions = self.decision_function(x)?;

        Ok(decisions.iter().map(|&d| usize::from(d >= 0.0)).collect())
    }
}

impl Default for LinearSVM {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
