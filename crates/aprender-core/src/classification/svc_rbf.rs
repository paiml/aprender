//! RBF-kernel Support Vector Classifier (kernel SVC).
//!
//! Net-new in aprender (the linear `LinearSVM` only handles linearly-separable
//! data). This classifier solves the **dual** SVM problem with a Gaussian
//! (Radial Basis Function) kernel via a self-contained SMO (Sequential Minimal
//! Optimization) solver, mirroring the libsvm/scikit-learn `SVC(kernel='rbf')`
//! formulation so predictions match within tolerance on the same data.
//!
//! # Dual formulation
//!
//! ```text
//! max_α  Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
//! s.t.   0 ≤ αᵢ ≤ C  and  Σᵢ αᵢ yᵢ = 0
//! ```
//!
//! with the RBF kernel
//!
//! ```text
//! K(x, z) = exp(−γ ||x − z||²)
//! ```
//!
//! The decision function is `f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b`, and the predicted
//! label is `sign(f(x))` mapped back to the original `{0, 1}` class labels.
//!
//! Contract: `contracts/svc-rbf-v1.yaml` — `OBLIG-RBF-SVC-SKLEARN-PARITY`.
//!
//! # Example
//!
//! ```
//! use aprender::classification::SVCRbf;
//! use aprender::primitives::Matrix;
//!
//! // XOR — not linearly separable, needs an RBF kernel.
//! let x = Matrix::from_vec(4, 2, vec![
//!     0.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 0.0,
//!     1.0, 1.0,
//! ]).expect("4x2 matrix");
//! let y = vec![0_usize, 1, 1, 0];
//!
//! let mut svc = SVCRbf::new().with_gamma(0.5).with_c(1.0);
//! svc.fit(&x, &y).expect("fit");
//! let preds = svc.predict(&x).expect("predict");
//! assert_eq!(preds, y);
//! ```

use crate::error::Result;
use crate::primitives::{Matrix, Vector};

/// RBF-kernel Support Vector Classifier (binary).
///
/// Solves the dual SVM problem with a Gaussian kernel using SMO. Hyperparameters
/// `gamma` and `c` mirror scikit-learn's `SVC(kernel='rbf', gamma=..., C=...)`.
#[derive(Debug, Clone)]
pub struct SVCRbf {
    /// Kernel coefficient γ for `exp(−γ ||x − z||²)`. Default: 0.5.
    gamma: f32,
    /// Regularization parameter C (upper bound on dual coefficients). Default: 1.0.
    c: f32,
    /// KKT tolerance for SMO. Default: 1e-3 (libsvm default).
    tol: f32,
    /// Maximum SMO passes over the data. Default: 1000.
    max_iter: usize,
    /// Support-vector feature rows (only `α > 0` samples retained).
    support_vectors: Option<Matrix<f32>>,
    /// Per-support-vector dual coefficient `αᵢ · yᵢ` (signed), aligned to `support_vectors` rows.
    dual_coef: Option<Vec<f32>>,
    /// Intercept `b` of the decision function.
    intercept: f32,
    /// The two original class labels in sorted order: `[neg_label, pos_label]`.
    /// `neg_label` maps to internal `−1`, `pos_label` maps to internal `+1`.
    classes: Option<[usize; 2]>,
}

impl SVCRbf {
    /// Creates a new RBF-kernel SVC with default hyperparameters
    /// (`gamma = 0.5`, `c = 1.0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            gamma: 0.5,
            c: 1.0,
            tol: 1e-3,
            max_iter: 1000,
            support_vectors: None,
            dual_coef: None,
            intercept: 0.0,
            classes: None,
        }
    }

    /// Sets the RBF kernel coefficient γ.
    #[must_use]
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Sets the regularization parameter C.
    #[must_use]
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Sets the SMO KKT tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the maximum number of SMO passes.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// RBF kernel `K(a, b) = exp(−γ ||a − b||²)` between two rows.
    fn rbf(gamma: f32, a: &[f32], b: &[f32]) -> f32 {
        let mut sq = 0.0_f32;
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            let d = ai - bi;
            sq += d * d;
        }
        (-gamma * sq).exp()
    }

    /// Fits the SVC to binary training data via SMO.
    ///
    /// # Arguments
    ///
    /// * `x` — feature matrix (`n_samples × n_features`)
    /// * `y` — binary labels (`n_samples`), exactly two distinct values
    ///
    /// # Errors
    ///
    /// Returns an error if X/y lengths mismatch, the data is empty, or there
    /// are not exactly two classes.
    // Contract: svc-rbf-v1, equation = "dual_objective"
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();
        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }
        if n_samples != y.len() {
            return Err("Number of samples in X and y must match".into());
        }

        // Determine the two classes (sorted) and map to internal ±1.
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        if classes.len() != 2 {
            return Err("SVCRbf requires exactly 2 classes".into());
        }
        let neg_label = classes[0];
        let pos_label = classes[1];
        let signs: Vec<f32> = y
            .iter()
            .map(|&l| if l == pos_label { 1.0 } else { -1.0 })
            .collect();

        // Materialize rows once for cache-friendly kernel evaluation.
        let rows: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| (0..n_features).map(|j| x.get(i, j)).collect())
            .collect();

        // Precompute the full kernel matrix (small fixtures — O(n²) is fine).
        let mut kernel = vec![0.0_f32; n_samples * n_samples];
        for i in 0..n_samples {
            for j in i..n_samples {
                let k = Self::rbf(self.gamma, &rows[i], &rows[j]);
                kernel[i * n_samples + j] = k;
                kernel[j * n_samples + i] = k;
            }
        }

        // SMO state.
        let mut alpha = vec![0.0_f32; n_samples];
        let mut bias = 0.0_f32;

        // Decision-function value f(xᵢ) = Σⱼ αⱼ yⱼ K(j,i) + b.
        let decision = |alpha: &[f32], bias: f32, i: usize| -> f32 {
            let mut s = bias;
            for j in 0..n_samples {
                if alpha[j] != 0.0 {
                    s += alpha[j] * signs[j] * kernel[j * n_samples + i];
                }
            }
            s
        };

        // Simplified SMO (Platt 1998 / Andrew Ng's CS229 variant): iterate until
        // no αᵢ violates the KKT conditions for `max_iter` passes.
        let c = self.c;
        let tol = self.tol;
        let mut passes = 0;
        let max_passes = self.max_iter;
        // Deterministic second-index choice keeps fit reproducible.
        let mut iter_count = 0usize;
        let hard_cap = self.max_iter.saturating_mul(n_samples).max(n_samples);

        while passes < max_passes && iter_count < hard_cap {
            let mut num_changed = 0;
            for i in 0..n_samples {
                iter_count += 1;
                let e_i = decision(&alpha, bias, i) - signs[i];
                let yi = signs[i];
                let r_i = e_i * yi;

                // KKT violation check.
                if (r_i < -tol && alpha[i] < c) || (r_i > tol && alpha[i] > 0.0) {
                    // Pick j ≠ i deterministically: maximize |E_i − E_j|.
                    let mut best_j = usize::MAX;
                    let mut best_delta = 0.0_f32;
                    for j in 0..n_samples {
                        if j == i {
                            continue;
                        }
                        let e_j = decision(&alpha, bias, j) - signs[j];
                        let delta = (e_i - e_j).abs();
                        if delta > best_delta {
                            best_delta = delta;
                            best_j = j;
                        }
                    }
                    if best_j == usize::MAX {
                        continue;
                    }
                    let j = best_j;
                    let e_j = decision(&alpha, bias, j) - signs[j];
                    let yj = signs[j];

                    let alpha_i_old = alpha[i];
                    let alpha_j_old = alpha[j];

                    // Compute L and H bounds.
                    let (low, high) = if (yi - yj).abs() > 0.5 {
                        // y_i != y_j
                        (
                            (alpha_j_old - alpha_i_old).max(0.0),
                            c + (alpha_j_old - alpha_i_old).min(0.0),
                        )
                    } else {
                        // y_i == y_j
                        (
                            (alpha_i_old + alpha_j_old - c).max(0.0),
                            (alpha_i_old + alpha_j_old).min(c),
                        )
                    };
                    if (high - low).abs() < 1e-12 {
                        continue;
                    }

                    let k_ii = kernel[i * n_samples + i];
                    let k_jj = kernel[j * n_samples + j];
                    let k_ij = kernel[i * n_samples + j];
                    let eta = 2.0 * k_ij - k_ii - k_jj;
                    if eta >= 0.0 {
                        // Non-positive-definite step; skip (rare with RBF).
                        continue;
                    }

                    let mut alpha_j_new = alpha_j_old - yj * (e_i - e_j) / eta;
                    alpha_j_new = alpha_j_new.clamp(low, high);
                    if (alpha_j_new - alpha_j_old).abs() < 1e-6 {
                        continue;
                    }
                    let alpha_i_new = alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new);

                    // Update bias (Platt's b1/b2).
                    let b1 = bias
                        - e_i
                        - yi * (alpha_i_new - alpha_i_old) * k_ii
                        - yj * (alpha_j_new - alpha_j_old) * k_ij;
                    let b2 = bias
                        - e_j
                        - yi * (alpha_i_new - alpha_i_old) * k_ij
                        - yj * (alpha_j_new - alpha_j_old) * k_jj;

                    alpha[i] = alpha_i_new;
                    alpha[j] = alpha_j_new;

                    bias = if alpha_i_new > 0.0 && alpha_i_new < c {
                        b1
                    } else if alpha_j_new > 0.0 && alpha_j_new < c {
                        b2
                    } else {
                        0.5 * (b1 + b2)
                    };

                    num_changed += 1;
                }
            }
            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }

        // Recompute the intercept as libsvm does: average the bias implied by
        // every FREE support vector (0 < αᵢ < C), for which the KKT conditions
        // give yᵢ·f(xᵢ) = 1 exactly. This is far more accurate than the running
        // Platt b1/b2 estimate and is what drives boundary parity with sklearn.
        // `decision(α, 0, i)` is f(xᵢ) WITHOUT the bias, so b = yᵢ − Σⱼ αⱼyⱼK(j,i).
        {
            let mut b_sum = 0.0_f32;
            let mut b_cnt = 0usize;
            for i in 0..n_samples {
                if alpha[i] > 1e-8 && alpha[i] < c - 1e-8 {
                    b_sum += signs[i] - decision(&alpha, 0.0, i);
                    b_cnt += 1;
                }
            }
            if b_cnt > 0 {
                bias = b_sum / b_cnt as f32;
            }
        }

        // Retain only support vectors (α > 0).
        let mut sv_rows: Vec<f32> = Vec::new();
        let mut dual_coef: Vec<f32> = Vec::new();
        let mut n_sv = 0;
        for i in 0..n_samples {
            if alpha[i] > 1e-8 {
                for j in 0..n_features {
                    sv_rows.push(rows[i][j]);
                }
                dual_coef.push(alpha[i] * signs[i]);
                n_sv += 1;
            }
        }

        if n_sv == 0 {
            // Degenerate (e.g. C too small); fall back to retaining one row
            // with a zero coefficient so prediction is well-defined.
            self.support_vectors = Some(Matrix::from_vec(1, n_features, rows[0].clone())?);
            self.dual_coef = Some(vec![0.0]);
        } else {
            self.support_vectors = Some(Matrix::from_vec(n_sv, n_features, sv_rows)?);
            self.dual_coef = Some(dual_coef);
        }
        self.intercept = bias;
        self.classes = Some([neg_label, pos_label]);
        Ok(())
    }

    /// Computes the raw decision-function value `f(x)` for one feature row.
    fn decision_value(&self, sv: &Matrix<f32>, coef: &[f32], row: &[f32]) -> f32 {
        let (n_sv, _) = sv.shape();
        let n_features = row.len();
        let mut s = self.intercept;
        for i in 0..n_sv {
            let mut sq = 0.0_f32;
            for j in 0..n_features {
                let d = sv.get(i, j) - row[j];
                sq += d * d;
            }
            s += coef[i] * (-self.gamma * sq).exp();
        }
        s
    }

    /// Returns the signed decision-function values `f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b`.
    ///
    /// `sign(f(x)) > 0` ⇒ positive class. Useful for inspecting the decision
    /// boundary margin.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted or feature dims mismatch.
    pub fn decision_function(&self, x: &Matrix<f32>) -> Result<Vector<f32>> {
        let sv = self.support_vectors.as_ref().ok_or("Model not fitted")?;
        let coef = self.dual_coef.as_ref().ok_or("Model not fitted")?;
        let (n_samples, n_features) = x.shape();
        if n_features != sv.shape().1 {
            return Err("Feature dimension mismatch".into());
        }
        let mut out = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let row: Vec<f32> = (0..n_features).map(|j| x.get(i, j)).collect();
            out.push(self.decision_value(sv, coef, &row));
        }
        Ok(Vector::from_vec(out))
    }

    /// Predicts class labels for samples.
    ///
    /// Returns the original class labels (the two values seen during `fit`).
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted or feature dims mismatch.
    // Contract: svc-rbf-v1, equation = "svc_predict"
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let classes = self.classes.as_ref().ok_or("Model not fitted")?;
        let scores = self.decision_function(x)?;
        Ok(scores
            .as_slice()
            .iter()
            .map(|&s| if s > 0.0 { classes[1] } else { classes[0] })
            .collect())
    }

    /// Computes accuracy on test data (fraction of correctly classified samples).
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted or feature dims mismatch.
    pub fn score(&self, x: &Matrix<f32>, y: &[usize]) -> Result<f32> {
        let preds = self.predict(x)?;
        if y.is_empty() {
            return Ok(0.0);
        }
        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        Ok(correct as f32 / y.len() as f32)
    }

    /// Returns the number of support vectors (α > 0), or 0 if not fitted.
    #[must_use]
    pub fn n_support_vectors(&self) -> usize {
        self.support_vectors.as_ref().map_or(0, |sv| sv.shape().0)
    }
}

impl Default for SVCRbf {
    fn default() -> Self {
        Self::new()
    }
}

// Estimator impl so SVCRbf works with generic cross_validate / grid_search
// (Pillar 1). Labels round-trip through f32; inherent &[usize] API unchanged.
// predict returns Result; the post-fit error path falls back to the first class.
impl crate::traits::Estimator for SVCRbf {
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let labels: Vec<usize> = y.as_slice().iter().map(|&v| v.round() as usize).collect();
        SVCRbf::fit(self, x, &labels)
    }
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        let fallback = self.classes.map_or(0, |c| c[0]);
        let labels: Vec<usize> =
            SVCRbf::predict(self, x).unwrap_or_else(|_| vec![fallback; x.shape().0]);
        Vector::from_vec(labels.into_iter().map(|l| l as f32).collect())
    }
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32 {
        let fallback = self.classes.map_or(0, |c| c[0]);
        let preds: Vec<usize> =
            SVCRbf::predict(self, x).unwrap_or_else(|_| vec![fallback; x.shape().0]);
        let n = y.len();
        if n == 0 {
            return 0.0;
        }
        let correct = preds
            .iter()
            .zip(y.as_slice())
            .filter(|(&p, &t)| p == t.round() as usize)
            .count();
        correct as f32 / n as f32
    }
}

#[cfg(test)]
#[path = "tests_svc_rbf.rs"]
mod tests_svc_rbf;
