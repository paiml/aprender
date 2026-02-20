
/// Elastic Net regression with combined L1 and L2 regularization.
///
/// Fits a linear model with both L1 and L2 penalties:
///
/// ```text
/// minimize ||y - Xβ||² + α * l1_ratio * ||β||₁ + α * (1 - l1_ratio) * ||β||²
/// ```
///
/// # Parameters
///
/// - `alpha` - Overall regularization strength
/// - `l1_ratio` - Mix between L1 and L2 (0.0 = Ridge, 1.0 = Lasso)
///
/// # Solver
///
/// Uses coordinate descent with combined soft-thresholding and shrinkage.
///
/// # When to use Elastic Net
///
/// - When you want both sparsity (L1) and grouping effect (L2)
/// - With correlated features where Lasso may be unstable
/// - When you don't know which regularization type to use
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
/// use aprender::linear_model::ElasticNet;
///
/// let x = Matrix::from_vec(5, 2, vec![
///     1.0, 2.0,
///     2.0, 3.0,
///     3.0, 4.0,
///     4.0, 5.0,
///     5.0, 6.0,
/// ]).expect("Valid matrix dimensions");
/// let y = Vector::from_slice(&[5.0, 8.0, 11.0, 14.0, 17.0]);
///
/// // 50% L1, 50% L2
/// let mut model = ElasticNet::new(0.1, 0.5);
/// model.fit(&x, &y).expect("Fit should succeed with valid data");
///
/// let r2 = model.score(&x, &y);
/// assert!(r2 > 0.9);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticNet {
    /// Overall regularization strength.
    alpha: f32,
    /// Mix between L1 and L2 (0.0 = pure L2, 1.0 = pure L1).
    l1_ratio: f32,
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
    /// Maximum number of iterations.
    max_iter: usize,
    /// Tolerance for convergence.
    tol: f32,
}

impl ElasticNet {
    /// Creates a new `ElasticNet` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Overall regularization strength
    /// * `l1_ratio` - Mix between L1 and L2 (0.0 = Ridge, 1.0 = Lasso)
    #[must_use]
    pub fn new(alpha: f32, l1_ratio: f32) -> Self {
        Self {
            alpha,
            l1_ratio: l1_ratio.clamp(0.0, 1.0),
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

    /// Returns the L1/L2 ratio.
    #[must_use]
    pub fn l1_ratio(&self) -> f32 {
        self.l1_ratio
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

    /// Saves the model to a binary file.
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

        // L1 ratio as tensor
        let l1_ratio_data = vec![self.l1_ratio];
        let l1_ratio_shape = vec![1];
        tensors.insert("l1_ratio".to_string(), (l1_ratio_data, l1_ratio_shape));

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

        // Extract l1_ratio tensor
        let l1_ratio_meta = metadata
            .get("l1_ratio")
            .ok_or("Missing 'l1_ratio' tensor in SafeTensors file")?;
        let l1_ratio_data = safetensors::extract_tensor(&raw_data, l1_ratio_meta)?;

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

        if l1_ratio_data.len() != 1 {
            return Err(format!(
                "Expected l1_ratio tensor to have 1 element, got {}",
                l1_ratio_data.len()
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
            l1_ratio: l1_ratio_data[0],
            coefficients: Some(Vector::from_vec(coef_data)),
            intercept: intercept_data[0],
            fit_intercept: true, // Default to true for loaded models
            max_iter: max_iter_data[0] as usize,
            tol: tol_data[0],
        })
    }
}
