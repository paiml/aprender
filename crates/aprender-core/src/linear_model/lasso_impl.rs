
impl Lasso {
    /// Creates a new `Lasso` regression with the given regularization strength.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Regularization strength. Larger values = more sparsity.
    ///   Must be non-negative.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Sets whether to fit an intercept term.
    #[must_use]
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
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
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Returns the regularization strength (alpha).
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the coefficients (excluding intercept).
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    #[must_use]
    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Returns the intercept term.
    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Returns true if the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Soft-thresholding operator for L1 regularization.
    fn soft_threshold(x: f32, lambda: f32) -> f32 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }

    /// Saves the model to a binary file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {e}"))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {e}"))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {e}"))?;
        Ok(model)
    }

    /// Saves the model to `SafeTensors` format.
    ///
    /// `SafeTensors` format is compatible with:
    /// - `HuggingFace` ecosystem
    /// - Ollama (can convert to GGUF)
    /// - `PyTorch`, TensorFlow
    /// - realizar inference engine
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model is not fitted
    /// - Serialization fails
    /// - File writing fails
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Verify model is fitted
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("Cannot save unfitted model. Call fit() first.")?;

        // Prepare tensors (BTreeMap ensures deterministic ordering)
        let mut tensors = BTreeMap::new();

        // Coefficients tensor
        let coef_data: Vec<f32> = (0..coefficients.len()).map(|i| coefficients[i]).collect();
        let coef_shape = vec![coefficients.len()];
        tensors.insert("coefficients".to_string(), (coef_data, coef_shape));

        // Intercept tensor
        let intercept_data = vec![self.intercept];
        let intercept_shape = vec![1];
        tensors.insert("intercept".to_string(), (intercept_data, intercept_shape));

        // Alpha (regularization strength) as tensor
        let alpha_data = vec![self.alpha];
        let alpha_shape = vec![1];
        tensors.insert("alpha".to_string(), (alpha_data, alpha_shape));

        // Max iterations as tensor (stored as f32 for consistency)
        let max_iter_data = vec![self.max_iter as f32];
        let max_iter_shape = vec![1];
        tensors.insert("max_iter".to_string(), (max_iter_data, max_iter_shape));

        // Tolerance as tensor
        let tol_data = vec![self.tol];
        let tol_shape = vec![1];
        tensors.insert("tol".to_string(), (tol_data, tol_shape));

        // Save to SafeTensors format
        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a model from `SafeTensors` format.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File reading fails
    /// - `SafeTensors` format is invalid
    /// - Required tensors are missing
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract coefficients tensor
        let coef_meta = metadata
            .get("coefficients")
            .ok_or("Missing 'coefficients' tensor in SafeTensors file")?;
        let coef_data = safetensors::extract_tensor(&raw_data, coef_meta)?;

        // Extract intercept tensor
        let intercept_meta = metadata
            .get("intercept")
            .ok_or("Missing 'intercept' tensor in SafeTensors file")?;
        let intercept_data = safetensors::extract_tensor(&raw_data, intercept_meta)?;

        // Extract alpha tensor
        let alpha_meta = metadata
            .get("alpha")
            .ok_or("Missing 'alpha' tensor in SafeTensors file")?;
        let alpha_data = safetensors::extract_tensor(&raw_data, alpha_meta)?;

        // Extract max_iter tensor
        let max_iter_meta = metadata
            .get("max_iter")
            .ok_or("Missing 'max_iter' tensor in SafeTensors file")?;
        let max_iter_data = safetensors::extract_tensor(&raw_data, max_iter_meta)?;

        // Extract tol tensor
        let tol_meta = metadata
            .get("tol")
            .ok_or("Missing 'tol' tensor in SafeTensors file")?;
        let tol_data = safetensors::extract_tensor(&raw_data, tol_meta)?;

        // Validate tensor sizes
        if intercept_data.len() != 1 {
            return Err(format!(
                "Expected intercept tensor to have 1 element, got {}",
                intercept_data.len()
            ));
        }

        if alpha_data.len() != 1 {
            return Err(format!(
                "Expected alpha tensor to have 1 element, got {}",
                alpha_data.len()
            ));
        }

        if max_iter_data.len() != 1 {
            return Err(format!(
                "Expected max_iter tensor to have 1 element, got {}",
                max_iter_data.len()
            ));
        }

        if tol_data.len() != 1 {
            return Err(format!(
                "Expected tol tensor to have 1 element, got {}",
                tol_data.len()
            ));
        }

        // Reconstruct model
        Ok(Self {
            alpha: alpha_data[0],
            coefficients: Some(Vector::from_vec(coef_data)),
            intercept: intercept_data[0],
            fit_intercept: true, // Default to true for loaded models
            max_iter: max_iter_data[0] as usize,
            tol: tol_data[0],
        })
    }
}

impl Estimator for Lasso {
    /// Fits the Lasso regression model using coordinate descent.
    ///
    /// # Errors
    ///
    /// Returns an error if input dimensions don't match.
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples != y.len() {
            return Err("Number of samples must match target length".into());
        }

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        // Center data if fitting intercept
        let (x_centered, y_centered, y_mean) = if self.fit_intercept {
            // Compute means
            let mut x_mean = vec![0.0; n_features];
            let mut y_sum = 0.0;

            for i in 0..n_samples {
                for (j, mean_j) in x_mean.iter_mut().enumerate() {
                    *mean_j += x.get(i, j);
                }
                y_sum += y[i];
            }

            for mean in &mut x_mean {
                *mean /= n_samples as f32;
            }
            let y_mean = y_sum / n_samples as f32;

            // Center data
            let mut x_data = vec![0.0; n_samples * n_features];
            let mut y_data = vec![0.0; n_samples];

            for i in 0..n_samples {
                for j in 0..n_features {
                    x_data[i * n_features + j] = x.get(i, j) - x_mean[j];
                }
                y_data[i] = y[i] - y_mean;
            }

            (
                Matrix::from_vec(n_samples, n_features, x_data)
                    .expect("Valid matrix dimensions for property test"),
                Vector::from_vec(y_data),
                y_mean,
            )
        } else {
            (x.clone(), y.clone(), 0.0)
        };

        // Initialize coefficients to zero
        let mut beta = vec![0.0; n_features];

        // Precompute X^T X diagonal (column norms squared)
        let mut col_norms_sq = vec![0.0; n_features];
        for (j, norm_sq) in col_norms_sq.iter_mut().enumerate() {
            for i in 0..n_samples {
                let val = x_centered.get(i, j);
                *norm_sq += val * val;
            }
        }

        // Coordinate descent
        for _ in 0..self.max_iter {
            let mut max_change = 0.0f32;

            for j in 0..n_features {
                if col_norms_sq[j] < 1e-10 {
                    continue; // Skip zero-variance features
                }

                // Compute residual without current feature
                let mut rho = 0.0;
                for i in 0..n_samples {
                    let mut pred = 0.0;
                    for (k, &beta_k) in beta.iter().enumerate() {
                        if k != j {
                            pred += x_centered.get(i, k) * beta_k;
                        }
                    }
                    let residual = y_centered[i] - pred;
                    rho += x_centered.get(i, j) * residual;
                }

                // Update coefficient with soft-thresholding
                let old_beta = beta[j];
                beta[j] = Self::soft_threshold(rho, self.alpha) / col_norms_sq[j];

                let change = (beta[j] - old_beta).abs();
                if change > max_change {
                    max_change = change;
                }
            }

            // Check convergence
            if max_change < self.tol {
                break;
            }
        }

        // Set intercept
        if self.fit_intercept {
            let mut intercept = y_mean;
            let mut x_mean = vec![0.0; n_features];
            for j in 0..n_features {
                for i in 0..n_samples {
                    x_mean[j] += x.get(i, j);
                }
                x_mean[j] /= n_samples as f32;
                intercept -= beta[j] * x_mean[j];
            }
            self.intercept = intercept;
        } else {
            self.intercept = 0.0;
        }

        self.coefficients = Some(Vector::from_vec(beta));
        Ok(())
    }

    /// Predicts target values for input data.
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        let coefficients = self
            .coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.");

        let result = x
            .matvec(coefficients)
            .expect("Matrix dimensions don't match coefficients");

        result.add_scalar(self.intercept)
    }

    /// Computes the R² score.
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32 {
        let y_pred = self.predict(x);
        r_squared(&y_pred, y)
    }
}
