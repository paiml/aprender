
impl Estimator for ElasticNet {
    /// Fits the Elastic Net model using coordinate descent.
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

        // Initialize coefficients
        let mut beta = vec![0.0; n_features];

        // Precompute column norms squared
        let mut col_norms_sq = vec![0.0; n_features];
        for (j, norm_sq) in col_norms_sq.iter_mut().enumerate() {
            for i in 0..n_samples {
                let val = x_centered.get(i, j);
                *norm_sq += val * val;
            }
        }

        // L1 and L2 penalties
        let l1_penalty = self.alpha * self.l1_ratio;
        let l2_penalty = self.alpha * (1.0 - self.l1_ratio);

        // Coordinate descent
        for _ in 0..self.max_iter {
            let mut max_change = 0.0f32;

            for j in 0..n_features {
                if col_norms_sq[j] < 1e-10 {
                    continue;
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

                // Update with soft-thresholding (L1) and shrinkage (L2)
                let old_beta = beta[j];
                let denom = col_norms_sq[j] + l2_penalty;
                beta[j] = Lasso::soft_threshold(rho, l1_penalty) / denom;

                let change = (beta[j] - old_beta).abs();
                if change > max_change {
                    max_change = change;
                }
            }

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

#[cfg(test)]
mod tests;
