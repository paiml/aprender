
impl Ridge {
    /// Creates a new `Ridge` regression with the given regularization strength.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Regularization strength. Larger values = more regularization.
    ///   Must be non-negative. Use 0.0 for no regularization (equivalent to OLS).
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
        }
    }

    /// Sets whether to fit an intercept term.
    #[must_use]
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
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

        // Reconstruct model
        Ok(Self {
            alpha: alpha_data[0],
            coefficients: Some(Vector::from_vec(coef_data)),
            intercept: intercept_data[0],
            fit_intercept: true, // Default to true for loaded models
        })
    }
}

impl Estimator for Ridge {
    /// Fits the Ridge regression model using regularized normal equations.
    ///
    /// Solves: β = (X^T X + αI)^-1 X^T y
    ///
    /// # Errors
    ///
    /// Returns an error if input dimensions don't match or matrix is singular.
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples != y.len() {
            return Err("Number of samples must match target length".into());
        }

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        // Create design matrix (with or without intercept)
        let x_design = if self.fit_intercept {
            LinearRegression::add_intercept_column(x)
        } else {
            x.clone()
        };

        let n_params = if self.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Compute X^T X
        let xt = x_design.transpose();
        let mut xtx = xt.matmul(&x_design)?;

        // Add regularization: X^T X + αI
        // Note: We don't regularize the intercept term
        for i in 0..n_params {
            // Skip intercept if fitting intercept (first column)
            if self.fit_intercept && i == 0 {
                continue;
            }
            let current = xtx.get(i, i);
            xtx.set(i, i, current + self.alpha);
        }

        // Compute X^T y
        let xty = xt.matvec(y)?;

        // Solve regularized normal equations via Cholesky decomposition
        let beta = xtx.cholesky_solve(&xty)?;

        // Extract intercept and coefficients
        if self.fit_intercept {
            self.intercept = beta[0];
            self.coefficients = Some(beta.slice(1, n_features + 1));
        } else {
            self.intercept = 0.0;
            self.coefficients = Some(beta);
        }

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

/// Lasso regression with L1 regularization.
///
/// Fits a linear model with L1 penalty on coefficient magnitudes.
/// The optimization objective is:
///
/// ```text
/// minimize ||y - Xβ||² + α||β||₁
/// ```
///
/// where `α` (alpha) controls the regularization strength.
///
/// # Solver
///
/// Uses coordinate descent with soft-thresholding.
///
/// # When to use Lasso
///
/// - For automatic feature selection (produces sparse models)
/// - When you expect only a few features to be relevant
/// - When interpretability through sparsity is desired
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
/// use aprender::linear_model::Lasso;
///
/// // Data with some features
/// let x = Matrix::from_vec(5, 2, vec![
///     1.0, 2.0,
///     2.0, 3.0,
///     3.0, 4.0,
///     4.0, 5.0,
///     5.0, 6.0,
/// ]).expect("Valid matrix dimensions");
/// let y = Vector::from_slice(&[5.0, 8.0, 11.0, 14.0, 17.0]);
///
/// let mut model = Lasso::new(0.1);  // alpha = 0.1
/// model.fit(&x, &y).expect("Fit should succeed with valid data");
///
/// let predictions = model.predict(&x);
/// let r2 = model.score(&x, &y);
/// assert!(r2 > 0.9);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lasso {
    /// Regularization strength (lambda/alpha).
    alpha: f32,
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
    /// Maximum number of iterations for coordinate descent.
    max_iter: usize,
    /// Tolerance for convergence.
    tol: f32,
}
