
impl Estimator for ElasticNet {
    /// Fits the Elastic Net model using coordinate descent.
    ///
    /// # Errors
    ///
    /// Returns an error if input dimensions don't match.
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (n_samples, n_features) = validate_fit_inputs(x, y)?;
        let (x_centered, y_centered, y_mean) = center_data(x, y, self.fit_intercept);
        let col_norms_sq = column_norms_sq(&x_centered);

        // L1 and L2 penalties.
        //
        // scikit-learn minimizes
        //   (1/(2*n_samples))*||y - Xb||^2
        //     + alpha*l1_ratio*||b||_1
        //     + 0.5*alpha*(1 - l1_ratio)*||b||^2,
        // whose coordinate update thresholds rho at n_samples*alpha*l1_ratio and
        // divides by (col_norms_sq + n_samples*alpha*(1 - l1_ratio)) (PMAT-848).
        // The n_samples factor is required for `alpha`/`l1_ratio` to match the
        // scikit-learn convention; omitting it makes both penalties ~n times too weak.
        let n = n_samples as f32;
        let l1_penalty = n * self.alpha * self.l1_ratio;
        let l2_penalty = n * self.alpha * (1.0 - self.l1_ratio);

        let mut beta = vec![0.0; n_features];
        coordinate_descent(
            &x_centered,
            &y_centered,
            &col_norms_sq,
            &mut beta,
            (l1_penalty, l2_penalty),
            (self.max_iter, self.tol),
        );

        self.intercept = fitted_intercept(x, y_mean, &beta, self.fit_intercept);
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
