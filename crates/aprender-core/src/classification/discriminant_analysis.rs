//! Discriminant Analysis classifiers: LDA and QDA.
//!
//! Mirrors scikit-learn's `discriminant_analysis` module, implemented LAPACK-free
//! via the same `Matrix::cholesky_solve` that `LinearRegression` uses (no SVD,
//! no col-major / BLAS path). Both estimators are pure-Rust O(n·d²) per-class-stats
//! + scalar d×d Cholesky — the shape that wins GaussianNB 4.91× / LinReg 1.78×.
//!
//! Contract: `contracts/discriminant-analysis-v1.yaml`
//!
//! # Estimators
//!
//! - [`QuadraticDiscriminantAnalysis`] — one covariance PER class. Predicts via the
//!   per-class multivariate-Gaussian log-likelihood (log-det from the Cholesky
//!   diagonal + Mahalanobis distance from a triangular solve) plus the log-prior.
//!   Parity target: sklearn `QuadraticDiscriminantAnalysis(reg_param=0.0)`, which
//!   uses the unbiased (`/(n_c − 1)`) per-class covariance.
//! - [`LinearDiscriminantAnalysis`] — one POOLED (shared) covariance across classes,
//!   class linear coefficients via `cholesky_solve` on that pooled covariance.
//!   Parity target: sklearn `LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)`,
//!   which uses the biased (`/n`) pooled within-class covariance.
//!
//! Honest speed note: the LAPACK-free speed edge holds vs sklearn `solver='lsqr'`
//! (LDA) and `reg_param`-regularized QDA (sklearn runs a full SVD per class). It does
//! NOT claim to beat sklearn's `svd` default LDA solver. Primary value is sklearn
//! parity + filling the previously-missing LDA/QDA gap.

use crate::error::Result;
use crate::primitives::{Matrix, Vector};

/// Quadratic Discriminant Analysis classifier.
///
/// Fits one Gaussian per class with its own covariance matrix, so decision
/// boundaries are quadratic. Each class covariance is factored with Cholesky;
/// prediction uses the per-class Gaussian log-likelihood + log-prior, and
/// [`predict`](Self::predict) returns the argmax class.
///
/// Matches scikit-learn `QuadraticDiscriminantAnalysis(reg_param=0.0)`:
/// the per-class covariance is **unbiased** (`Σ (x−μ)(x−μ)ᵀ / (n_c − 1)`).
///
/// # Example
///
/// ```
/// use aprender::classification::QuadraticDiscriminantAnalysis;
/// use aprender::primitives::Matrix;
///
/// let x = Matrix::from_vec(6, 2, vec![
///     0.0, 0.0,  0.5, 0.2,  0.2, 0.6,   // class 0
///     5.0, 5.0,  5.4, 4.8,  4.7, 5.3,   // class 1
/// ]).expect("6x2 matrix");
/// let y = vec![0, 0, 0, 1, 1, 1];
///
/// let mut model = QuadraticDiscriminantAnalysis::new();
/// model.fit(&x, &y).expect("valid training data");
/// let preds = model.predict(&x).expect("model is fitted");
/// assert_eq!(preds, vec![0, 0, 0, 1, 1, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct QuadraticDiscriminantAnalysis {
    /// Sorted unique class labels.
    classes: Option<Vec<usize>>,
    /// Per-class mean vectors: `means[class][feature]`.
    means: Option<Vec<Vec<f32>>>,
    /// Per-class log-prior: `log_priors[class]`.
    log_priors: Option<Vec<f32>>,
    /// Per-class lower-triangular Cholesky factor `L` (row-major `d×d`) of the
    /// class covariance: `Σ_c = L Lᵀ`.
    chol: Option<Vec<Vec<f32>>>,
    /// Per-class log-determinant of `Σ_c` = `2·Σ ln(L_ii)`.
    log_det: Option<Vec<f32>>,
    /// Regularization added to the covariance diagonal if the raw covariance is
    /// not positive-definite (a small ridge), retried before erroring.
    reg_param: f32,
}

impl QuadraticDiscriminantAnalysis {
    /// Creates a new QDA classifier (no covariance regularization by default,
    /// matching sklearn `reg_param=0.0`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            classes: None,
            means: None,
            log_priors: None,
            chol: None,
            log_det: None,
            reg_param: 0.0,
        }
    }

    /// Sets the covariance-diagonal regularization (ridge) used when a raw class
    /// covariance is not positive-definite. The ridge is `reg_param · mean(diag)`.
    #[must_use]
    pub fn with_reg_param(mut self, reg_param: f32) -> Self {
        self.reg_param = reg_param;
        self
    }

    /// Returns the sorted unique class labels (after fitting).
    #[must_use]
    pub fn classes(&self) -> Option<&[usize]> {
        self.classes.as_deref()
    }

    /// Fits one Gaussian per class.
    ///
    /// For each class `c`: computes the mean, the unbiased covariance
    /// `Σ_c = Σ_i (x_i − μ_c)(x_i − μ_c)ᵀ / (n_c − 1)`, and its lower Cholesky
    /// factor `L_c`. If `Σ_c` is not positive-definite, a small ridge
    /// (`reg_param · mean(diag)`, or `1e-6 · mean(diag)` when `reg_param == 0`)
    /// is added to the diagonal and the factorization is retried; it errors only
    /// if still non-PSD.
    ///
    /// # Errors
    ///
    /// Returns an error if X/y dimensions disagree, the data is empty, there are
    /// fewer than 2 classes, any class has fewer than 2 samples, or a class
    /// covariance remains non-PSD even after regularization.
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();
        if n_samples == 0 {
            return Err("Cannot fit with empty data".into());
        }
        if y.len() != n_samples {
            return Err("Number of samples in X and y must match".into());
        }

        let classes = unique_sorted(y);
        if classes.len() < 2 {
            return Err("Need at least 2 classes".into());
        }

        let mut means = Vec::with_capacity(classes.len());
        let mut log_priors = Vec::with_capacity(classes.len());
        let mut chol = Vec::with_capacity(classes.len());
        let mut log_det = Vec::with_capacity(classes.len());

        for &class_label in &classes {
            let rows: Vec<usize> = (0..n_samples).filter(|&i| y[i] == class_label).collect();
            let n_c = rows.len();
            if n_c < 2 {
                return Err("Each class needs at least 2 samples for a covariance".into());
            }

            let mean = class_mean(x, &rows, n_features);
            // Unbiased covariance (sklearn QDA reg_param=0 uses /(n_c - 1)).
            let cov = class_covariance(x, &rows, &mean, n_features, (n_c - 1) as f32);

            let l = self.cholesky_with_ridge(&cov, n_features)?;
            let ld = log_det_from_cholesky(&l, n_features);

            means.push(mean);
            log_priors.push((n_c as f32 / n_samples as f32).ln());
            chol.push(l);
            log_det.push(ld);
        }

        self.classes = Some(classes);
        self.means = Some(means);
        self.log_priors = Some(log_priors);
        self.chol = Some(chol);
        self.log_det = Some(log_det);
        Ok(())
    }

    /// Cholesky factor of `cov`, retrying once with a diagonal ridge if non-PSD.
    fn cholesky_with_ridge(&self, cov: &[f32], d: usize) -> Result<Vec<f32>> {
        if let Some(l) = cholesky_lower(cov, d) {
            return Ok(l);
        }
        // Regularize the diagonal with a small ridge and retry.
        let mean_diag = (0..d).map(|i| cov[i * d + i]).sum::<f32>() / d as f32;
        let ridge = if self.reg_param > 0.0 {
            self.reg_param * mean_diag
        } else {
            1e-6 * mean_diag.max(1e-12)
        };
        let mut reg = cov.to_vec();
        for i in 0..d {
            reg[i * d + i] += ridge;
        }
        cholesky_lower(&reg, d).ok_or_else(|| {
            "Class covariance is not positive-definite even after regularization".into()
        })
    }

    /// Per-class log-likelihood + log-prior log-posteriors for one sample.
    fn log_posteriors_row(&self, x: &Matrix<f32>, row: usize) -> Vec<f32> {
        let means = self.means.as_ref().expect("fitted");
        let chol = self.chol.as_ref().expect("fitted");
        let log_det = self.log_det.as_ref().expect("fitted");
        let log_priors = self.log_priors.as_ref().expect("fitted");
        let d = means[0].len();
        let two_pi_term = d as f32 * (2.0 * std::f32::consts::PI).ln();

        let mut out = Vec::with_capacity(means.len());
        for c in 0..means.len() {
            // diff = x - mu_c
            let mut diff = vec![0.0f32; d];
            for (j, dj) in diff.iter_mut().enumerate() {
                *dj = x.get(row, j) - means[c][j];
            }
            // Mahalanobis: solve L z = diff (forward substitution), then ||z||².
            let z = forward_substitute(&chol[c], &diff, d);
            let mahal: f32 = z.iter().map(|&v| v * v).sum();
            // log N(x; mu, Sigma) + log prior
            let lp = -0.5 * (two_pi_term + log_det[c] + mahal) + log_priors[c];
            out.push(lp);
        }
        out
    }

    /// Predicts the class label (argmax log-posterior) for each sample.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted or feature dimensions mismatch.
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let classes = self.classes.as_ref().ok_or("Model not fitted")?;
        let means = self.means.as_ref().ok_or("Model not fitted")?;
        let (n_samples, n_features) = x.shape();
        if n_features != means[0].len() {
            return Err("Feature dimension mismatch".into());
        }
        let mut preds = Vec::with_capacity(n_samples);
        for row in 0..n_samples {
            let lp = self.log_posteriors_row(x, row);
            preds.push(classes[argmax(&lp)]);
        }
        Ok(preds)
    }

    /// Returns the softmax of the per-class log-posteriors for each sample.
    ///
    /// Each inner vector is in the class order returned by [`classes`](Self::classes).
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted or feature dimensions mismatch.
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Result<Vec<Vec<f32>>> {
        let means = self.means.as_ref().ok_or("Model not fitted")?;
        let (n_samples, n_features) = x.shape();
        if n_features != means[0].len() {
            return Err("Feature dimension mismatch".into());
        }
        let mut out = Vec::with_capacity(n_samples);
        for row in 0..n_samples {
            out.push(softmax(&self.log_posteriors_row(x, row)));
        }
        Ok(out)
    }
}

impl Default for QuadraticDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear Discriminant Analysis classifier (`solver='lsqr'`, no shrinkage).
///
/// Fits one POOLED (shared) within-class covariance across all classes, so the
/// decision boundaries are linear. The per-class linear coefficients are obtained
/// by solving `Σ_pooled · wᵀ = μ_cᵀ` with `Matrix::cholesky_solve` (LAPACK-free),
/// and the decision function is `f_c(x) = wᵀ x + b_c` with
/// `b_c = −0.5 · wᵀμ_c + ln P(c)`.
///
/// Matches scikit-learn `LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)`:
/// the pooled within-class covariance is **biased** (`Σ_c Σ_i (x−μ_c)(x−μ_c)ᵀ / n`).
///
/// # Example
///
/// ```
/// use aprender::classification::LinearDiscriminantAnalysis;
/// use aprender::primitives::Matrix;
///
/// let x = Matrix::from_vec(6, 2, vec![
///     0.0, 0.0,  0.5, 0.2,  0.2, 0.6,   // class 0
///     5.0, 5.0,  5.4, 4.8,  4.7, 5.3,   // class 1
/// ]).expect("6x2 matrix");
/// let y = vec![0, 0, 0, 1, 1, 1];
///
/// let mut model = LinearDiscriminantAnalysis::new();
/// model.fit(&x, &y).expect("valid training data");
/// let preds = model.predict(&x).expect("model is fitted");
/// assert_eq!(preds, vec![0, 0, 0, 1, 1, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct LinearDiscriminantAnalysis {
    /// Sorted unique class labels.
    classes: Option<Vec<usize>>,
    /// Per-class linear coefficients: `coef[class][feature]`.
    coef: Option<Vec<Vec<f32>>>,
    /// Per-class intercept: `intercept[class]`.
    intercept: Option<Vec<f32>>,
}

impl LinearDiscriminantAnalysis {
    /// Creates a new LDA classifier (`solver='lsqr'`, no shrinkage).
    #[must_use]
    pub fn new() -> Self {
        Self {
            classes: None,
            coef: None,
            intercept: None,
        }
    }

    /// Returns the sorted unique class labels (after fitting).
    #[must_use]
    pub fn classes(&self) -> Option<&[usize]> {
        self.classes.as_deref()
    }

    /// Returns the per-class linear coefficients (after fitting).
    #[must_use]
    pub fn coef(&self) -> Option<&Vec<Vec<f32>>> {
        self.coef.as_ref()
    }

    /// Returns the per-class intercepts (after fitting).
    #[must_use]
    pub fn intercept(&self) -> Option<&[f32]> {
        self.intercept.as_deref()
    }

    /// Fits the pooled-covariance LDA model.
    ///
    /// Computes the per-class means, the biased pooled within-class covariance
    /// `Σ_pooled = Σ_c Σ_i (x_i − μ_c)(x_i − μ_c)ᵀ / n`, then solves
    /// `Σ_pooled · w_cᵀ = μ_cᵀ` per class via `cholesky_solve`.
    ///
    /// # Errors
    ///
    /// Returns an error if X/y dimensions disagree, the data is empty, there are
    /// fewer than 2 classes, or the pooled covariance is not positive-definite
    /// (singular within-class scatter).
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();
        if n_samples == 0 {
            return Err("Cannot fit with empty data".into());
        }
        if y.len() != n_samples {
            return Err("Number of samples in X and y must match".into());
        }

        let classes = unique_sorted(y);
        if classes.len() < 2 {
            return Err("Need at least 2 classes".into());
        }

        // Per-class means + accumulate the pooled within-class scatter.
        let mut means = Vec::with_capacity(classes.len());
        let mut log_priors = Vec::with_capacity(classes.len());
        let mut scatter = vec![0.0f32; n_features * n_features];

        for &class_label in &classes {
            let rows: Vec<usize> = (0..n_samples).filter(|&i| y[i] == class_label).collect();
            let n_c = rows.len();
            let mean = class_mean(x, &rows, n_features);
            for &i in &rows {
                for a in 0..n_features {
                    let da = x.get(i, a) - mean[a];
                    for b in 0..n_features {
                        let db = x.get(i, b) - mean[b];
                        scatter[a * n_features + b] += da * db;
                    }
                }
            }
            log_priors.push((n_c as f32 / n_samples as f32).ln());
            means.push(mean);
        }

        // Biased pooled within-class covariance: scatter / n (sklearn _class_cov).
        let inv_n = 1.0 / n_samples as f32;
        let cov_data: Vec<f32> = scatter.iter().map(|&v| v * inv_n).collect();
        let cov = Matrix::from_vec(n_features, n_features, cov_data)
            .map_err(Into::<crate::error::AprenderError>::into)?;

        // Per-class coefficients: solve Sigma_pooled * w_c = mu_c (cholesky_solve).
        let mut coef = Vec::with_capacity(classes.len());
        let mut intercept = Vec::with_capacity(classes.len());
        for c in 0..classes.len() {
            let mu = Vector::from_vec(means[c].clone());
            let w = cov
                .cholesky_solve(&mu)
                .map_err(Into::<crate::error::AprenderError>::into)?;
            let w_vec: Vec<f32> = (0..n_features).map(|j| w[j]).collect();
            // intercept = -0.5 * w . mu + ln(prior)
            let dot: f32 = (0..n_features).map(|j| w_vec[j] * means[c][j]).sum();
            intercept.push(-0.5 * dot + log_priors[c]);
            coef.push(w_vec);
        }

        self.classes = Some(classes);
        self.coef = Some(coef);
        self.intercept = Some(intercept);
        Ok(())
    }

    /// Computes the per-class decision function `f_c(x) = w_cᵀ x + b_c`
    /// for one sample row.
    fn decision_row(&self, x: &Matrix<f32>, row: usize) -> Vec<f32> {
        let coef = self.coef.as_ref().expect("fitted");
        let intercept = self.intercept.as_ref().expect("fitted");
        let d = coef[0].len();
        (0..coef.len())
            .map(|c| {
                let mut acc = intercept[c];
                for j in 0..d {
                    acc += coef[c][j] * x.get(row, j);
                }
                acc
            })
            .collect()
    }

    /// Returns the per-class decision function values for each sample.
    ///
    /// Each inner vector is in the class order returned by [`classes`](Self::classes),
    /// matching sklearn `LinearDiscriminantAnalysis.decision_function` for K > 2.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted or feature dimensions mismatch.
    pub fn decision_function(&self, x: &Matrix<f32>) -> Result<Vec<Vec<f32>>> {
        let coef = self.coef.as_ref().ok_or("Model not fitted")?;
        let (n_samples, n_features) = x.shape();
        if n_features != coef[0].len() {
            return Err("Feature dimension mismatch".into());
        }
        Ok((0..n_samples)
            .map(|row| self.decision_row(x, row))
            .collect())
    }

    /// Predicts the class label (argmax decision function) for each sample.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted or feature dimensions mismatch.
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let classes = self.classes.as_ref().ok_or("Model not fitted")?;
        let coef = self.coef.as_ref().ok_or("Model not fitted")?;
        let (n_samples, n_features) = x.shape();
        if n_features != coef[0].len() {
            return Err("Feature dimension mismatch".into());
        }
        let mut preds = Vec::with_capacity(n_samples);
        for row in 0..n_samples {
            let dec = self.decision_row(x, row);
            preds.push(classes[argmax(&dec)]);
        }
        Ok(preds)
    }

    /// Returns the softmax of the per-class decision values for each sample.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not fitted or feature dimensions mismatch.
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Result<Vec<Vec<f32>>> {
        let coef = self.coef.as_ref().ok_or("Model not fitted")?;
        let (n_samples, n_features) = x.shape();
        if n_features != coef[0].len() {
            return Err("Feature dimension mismatch".into());
        }
        let mut out = Vec::with_capacity(n_samples);
        for row in 0..n_samples {
            out.push(softmax(&self.decision_row(x, row)));
        }
        Ok(out)
    }
}

impl Default for LinearDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shared numeric helpers
// ---------------------------------------------------------------------------

/// Sorted unique class labels.
fn unique_sorted(y: &[usize]) -> Vec<usize> {
    let mut c = y.to_vec();
    c.sort_unstable();
    c.dedup();
    c
}

/// Mean feature vector over the given sample rows.
fn class_mean(x: &Matrix<f32>, rows: &[usize], n_features: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; n_features];
    for &i in rows {
        for (j, mj) in mean.iter_mut().enumerate() {
            *mj += x.get(i, j);
        }
    }
    let inv = 1.0 / rows.len() as f32;
    for m in &mut mean {
        *m *= inv;
    }
    mean
}

/// Covariance `Σ (x−μ)(x−μ)ᵀ / denom` (row-major `d×d`), `denom` = `n_c−1`
/// (unbiased) or `n_c` (biased).
fn class_covariance(
    x: &Matrix<f32>,
    rows: &[usize],
    mean: &[f32],
    n_features: usize,
    denom: f32,
) -> Vec<f32> {
    let mut cov = vec![0.0f32; n_features * n_features];
    for &i in rows {
        for a in 0..n_features {
            let da = x.get(i, a) - mean[a];
            for b in 0..n_features {
                let db = x.get(i, b) - mean[b];
                cov[a * n_features + b] += da * db;
            }
        }
    }
    let inv = 1.0 / denom;
    for v in &mut cov {
        *v *= inv;
    }
    cov
}

/// Lower-triangular Cholesky factor `L` (row-major `d×d`) of a symmetric matrix,
/// or `None` if not positive-definite.
fn cholesky_lower(a: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            if i == j {
                for k in 0..j {
                    sum += l[j * n + k] * l[j * n + k];
                }
                let diag = a[j * n + j] - sum;
                if diag <= 0.0 {
                    return None;
                }
                l[j * n + j] = diag.sqrt();
            } else {
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Forward substitution: solve `L z = b` for lower-triangular `L` (row-major `n×n`).
fn forward_substitute(l: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut z = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * n + j] * z[j];
        }
        z[i] = (b[i] - sum) / l[i * n + i];
    }
    z
}

/// Log-determinant of `Σ = L Lᵀ` from the Cholesky diagonal: `2·Σ ln(L_ii)`.
fn log_det_from_cholesky(l: &[f32], n: usize) -> f32 {
    2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f32>()
}

/// Index of the maximum element (first on ties — matches sklearn `argmax`).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// Numerically-stable softmax (subtract max).
fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

#[cfg(test)]
#[path = "tests_discriminant_analysis.rs"]
mod tests_discriminant_analysis;
