
impl RobustScaler {
    /// Creates a new `RobustScaler` with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            median: None,
            iqr: None,
            with_centering: true,
            with_scaling: true,
        }
    }

    /// Sets whether to center the data by subtracting the median.
    #[must_use]
    pub fn with_centering(mut self, centering: bool) -> Self {
        self.with_centering = centering;
        self
    }

    /// Sets whether to scale the data by dividing by IQR.
    #[must_use]
    pub fn with_scaling(mut self, scaling: bool) -> Self {
        self.with_scaling = scaling;
        self
    }

    /// Returns the median of each feature.
    #[must_use]
    pub fn median(&self) -> &[f32] {
        self.median
            .as_ref()
            .expect("Scaler not fitted. Call fit() first.")
    }

    /// Returns the IQR of each feature.
    #[must_use]
    pub fn iqr(&self) -> &[f32] {
        self.iqr
            .as_ref()
            .expect("Scaler not fitted. Call fit() first.")
    }

    /// Returns true if the scaler has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.median.is_some()
    }
}

/// Compute the percentile of a sorted slice using linear interpolation.
fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f32;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

impl Transformer for RobustScaler {
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        let mut medians = Vec::with_capacity(n_features);
        let mut iqrs = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let mut col: Vec<f32> = (0..n_samples).map(|i| x.get(i, j)).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            medians.push(percentile(&col, 0.5));
            let q1 = percentile(&col, 0.25);
            let q3 = percentile(&col, 0.75);
            iqrs.push(q3 - q1);
        }

        self.median = Some(medians);
        self.iqr = Some(iqrs);
        Ok(())
    }

    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let median = self
            .median
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;
        let iqr = self
            .iqr
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;

        let (n_samples, n_features) = x.shape();
        if n_features != median.len() {
            return Err("Feature dimension mismatch".into());
        }

        let mut result = vec![0.0; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let mut val = x.get(i, j);

                if self.with_centering {
                    val -= median[j];
                }

                if self.with_scaling && iqr[j] > 1e-10 {
                    val /= iqr[j];
                }

                result[i * n_features + j] = val;
            }
        }

        Matrix::from_vec(n_samples, n_features, result).map_err(Into::into)
    }
}

include!("pca.rs");
include!("tsne.rs");
