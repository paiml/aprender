
impl MinMaxScaler {
    /// Creates a new `MinMaxScaler` with default range [0, 1].
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_min: None,
            data_max: None,
            feature_min: 0.0,
            feature_max: 1.0,
        }
    }

    /// Sets the target range for scaling.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::preprocessing::MinMaxScaler;
    ///
    /// let scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
    /// ```
    #[must_use]
    pub fn with_range(mut self, min: f32, max: f32) -> Self {
        self.feature_min = min;
        self.feature_max = max;
        self
    }

    /// Returns the minimum value of each feature.
    ///
    /// # Panics
    ///
    /// Panics if the scaler is not fitted.
    #[must_use]
    pub fn data_min(&self) -> &[f32] {
        self.data_min
            .as_ref()
            .expect("Scaler not fitted. Call fit() first.")
    }

    /// Returns the maximum value of each feature.
    ///
    /// # Panics
    ///
    /// Panics if the scaler is not fitted.
    #[must_use]
    pub fn data_max(&self) -> &[f32] {
        self.data_max
            .as_ref()
            .expect("Scaler not fitted. Call fit() first.")
    }

    /// Returns true if the scaler has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.data_min.is_some()
    }

    /// Transforms data back to original scale.
    ///
    /// # Errors
    ///
    /// Returns an error if the scaler is not fitted or dimensions mismatch.
    pub fn inverse_transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let data_min = self
            .data_min
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;
        let data_max = self
            .data_max
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;

        let (n_samples, n_features) = x.shape();
        if n_features != data_min.len() {
            return Err("Feature dimension mismatch".into());
        }

        let feature_range = self.feature_max - self.feature_min;
        let mut result = vec![0.0; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = x.get(i, j);
                let data_range = data_max[j] - data_min[j];

                let original = if data_range.abs() > 1e-10 {
                    (val - self.feature_min) / feature_range * data_range + data_min[j]
                } else {
                    data_min[j]
                };

                result[i * n_features + j] = original;
            }
        }

        Matrix::from_vec(n_samples, n_features, result).map_err(Into::into)
    }
}

// Contract: preprocessing-normalization-v1, equation = "minmax_scaler"
impl Transformer for MinMaxScaler {
    /// Computes the min and max of each feature.
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        let mut data_min = vec![f32::INFINITY; n_features];
        let mut data_max = vec![f32::NEG_INFINITY; n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = x.get(i, j);
                if val < data_min[j] {
                    data_min[j] = val;
                }
                if val > data_max[j] {
                    data_max[j] = val;
                }
            }
        }

        self.data_min = Some(data_min);
        self.data_max = Some(data_max);

        Ok(())
    }

    /// Scales the data to the target range.
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let data_min = self
            .data_min
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;
        let data_max = self
            .data_max
            .as_ref()
            .ok_or_else(|| AprenderError::from("Scaler not fitted"))?;

        let (n_samples, n_features) = x.shape();
        if n_features != data_min.len() {
            return Err("Feature dimension mismatch".into());
        }

        let feature_range = self.feature_max - self.feature_min;
        let mut result = vec![0.0; n_samples * n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = x.get(i, j);
                let data_range = data_max[j] - data_min[j];

                let scaled = if data_range.abs() > 1e-10 {
                    (val - data_min[j]) / data_range * feature_range + self.feature_min
                } else {
                    self.feature_min
                };

                result[i * n_features + j] = scaled;
            }
        }

        Matrix::from_vec(n_samples, n_features, result).map_err(Into::into)
    }
}

/// Which SVD routine [`PCA`] and [`TruncatedSVD`] factor the data with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SvdSolver {
    /// Pick by problem shape, following scikit-learn's `svd_solver="auto"`:
    /// exact when `max(n, d) <= 500` or `n_components` is a variance ratio,
    /// randomized when `n_components < 0.8 * min(n, d)`, exact otherwise.
    #[default]
    Auto,
    /// Exact thin SVD (Golub-Reinsch).
    Full,
    /// Randomized top-k SVD (Halko, Martinsson & Tropp), seeded, so still deterministic.
    Randomized,
}

/// How many principal components [`PCA`] keeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NComponents {
    /// Exactly this many components.
    Count(usize),
    /// The fewest components whose explained-variance ratios sum past this value, in `(0, 1)`.
    VarianceRatio(f64),
}

impl From<usize> for NComponents {
    fn from(k: usize) -> Self {
        Self::Count(k)
    }
}

/// Row `r` of a row-major `rows x cols` slice.
fn svd_row(a: &[f64], cols: usize, r: usize) -> &[f64] {
    &a[r * cols..(r + 1) * cols]
}

/// scikit-learn's `svd_flip(u_based_decision=False)`: the first largest-magnitude entry of
/// every component row is made positive, so two fits never disagree on sign.
fn flip_rows_by_max_abs(vt: &mut [f64], cols: usize) {
    for row in vt.chunks_exact_mut(cols) {
        let mut best = 0;
        for (j, v) in row.iter().enumerate() {
            if v.abs() > row[best].abs() {
                best = j;
            }
        }
        if row[best] < 0.0 {
            for v in row.iter_mut() {
                *v = -*v;
            }
        }
    }
}

/// sklearn's `n_iter="auto"` for randomized PCA.
fn auto_power_iters(k: usize, min_dim: usize) -> usize {
    if (k as f64) < 0.1 * min_dim as f64 {
        7
    } else {
        4
    }
}

/// Top-`k` SVD of a row-major `n x d` matrix; `vt` comes back `k x d`, sign-flipped.
fn top_k_svd(
    a: &[f64],
    n: usize,
    d: usize,
    k: usize,
    randomized: Option<(usize, u64)>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let svd = match randomized {
        Some((power_iters, seed)) => {
            let mut cfg = trueno::RandomizedSvdConfig::new(k);
            cfg.power_iters = power_iters;
            cfg.seed = seed;
            trueno::Svd::randomized(a, n, d, cfg)
        }
        None => trueno::Svd::new(a, n, d),
    }
    .map_err(|e| AprenderError::from(format!("SVD failed: {e}")))?;
    let sigma = svd.singular_values().to_vec();
    let mut vt = svd.vt().to_vec();
    flip_rows_by_max_abs(&mut vt, d);
    Ok((sigma, vt))
}

/// Row-major f64 copy of `x`, minus `mean` when one is given.
fn to_f64_centered(x: &[f32], d: usize, mean: Option<&[f64]>) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| f64::from(v) - mean.map_or(0.0, |m| m[i % d]))
        .collect()
}

/// Per-column mean, accumulated in f64.
fn column_means(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut mean = vec![0.0_f64; d];
    for row in x.chunks_exact(d) {
        mean.iter_mut().zip(row).for_each(|(m, v)| *m += v);
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    mean
}

/// `x @ vtᵀ` for row-major `x` (`n x d`) and `vt` (`k x d`).
fn project(x: &[f64], d: usize, vt: &[f64], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len() / d.max(1) * k);
    for row in x.chunks_exact(d) {
        for c in 0..k {
            out.push(row.iter().zip(svd_row(vt, d, c)).map(|(a, b)| a * b).sum());
        }
    }
    out
}

/// Fitted state of a [`PCA`]; every quantity is kept in f64, with f32 views for the
/// long-standing accessors.
#[derive(Debug, Clone)]
struct PcaFit {
    k: usize,
    mean: Vec<f64>,
    components: Vec<f64>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    singular_values: Vec<f64>,
    components_f32: Matrix<f32>,
    explained_variance_f32: Vec<f32>,
    explained_variance_ratio_f32: Vec<f32>,
}

/// Principal Component Analysis (PCA) for dimensionality reduction.
///
/// PCA reduces dimensionality by projecting data onto principal components
/// (directions of maximum variance). The components are the right singular vectors
/// of the mean-centered data, computed by SVD in f64 — the covariance matrix is never
/// formed, so its squared condition number never costs the trailing components.
///
/// # Example
///
/// ```
/// use aprender::preprocessing::PCA;
/// use aprender::traits::Transformer;
/// use aprender::primitives::Matrix;
///
/// let data = Matrix::from_vec(4, 3, vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
///     7.0, 8.0, 9.0,
///     10.0, 11.0, 12.0,
/// ]).expect("valid matrix dimensions");
///
/// let mut pca = PCA::new(2); // Reduce to 2 components
/// let transformed = pca.fit_transform(&data).expect("fit_transform should succeed");
/// assert_eq!(transformed.shape(), (4, 2));
/// ```
#[derive(Debug, Clone)]
pub struct PCA {
    n_components: NComponents,
    svd_solver: SvdSolver,
    seed: u64,
    fit: Option<PcaFit>,
}

impl PCA {
    /// PCA keeping `n_components` components, with [`SvdSolver::Auto`].
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components: NComponents::Count(n_components),
            svd_solver: SvdSolver::Auto,
            seed: 0,
            fit: None,
        }
    }

    /// PCA keeping the fewest components whose explained-variance ratio exceeds `ratio`.
    /// `ratio` must lie in `(0, 1)`; this is checked by [`Transformer::fit`].
    #[must_use]
    pub fn with_variance_ratio(ratio: f64) -> Self {
        Self {
            n_components: NComponents::VarianceRatio(ratio),
            ..Self::new(0)
        }
    }

    /// Choose the SVD routine.
    #[must_use]
    pub fn with_svd_solver(mut self, solver: SvdSolver) -> Self {
        self.svd_solver = solver;
        self
    }

    /// Seed for the randomized solver.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Number of components kept by the last fit.
    #[must_use]
    pub fn n_components_fitted(&self) -> Option<usize> {
        self.fit.as_ref().map(|f| f.k)
    }

    #[must_use]
    pub fn explained_variance(&self) -> Option<&[f32]> {
        self.fit.as_ref().map(|f| f.explained_variance_f32.as_slice())
    }

    #[must_use]
    pub fn explained_variance_ratio(&self) -> Option<&[f32]> {
        self.fit
            .as_ref()
            .map(|f| f.explained_variance_ratio_f32.as_slice())
    }

    #[must_use]
    pub fn components(&self) -> Option<&Matrix<f32>> {
        self.fit.as_ref().map(|f| &f.components_f32)
    }

    /// `σᵢ² / (n - 1)` at full precision.
    #[must_use]
    pub fn explained_variance_f64(&self) -> Option<&[f64]> {
        self.fit.as_ref().map(|f| f.explained_variance.as_slice())
    }

    /// Explained-variance ratios at full precision.
    #[must_use]
    pub fn explained_variance_ratio_f64(&self) -> Option<&[f64]> {
        self.fit.as_ref().map(|f| f.explained_variance_ratio.as_slice())
    }

    /// Components at full precision, row-major `n_components x n_features`.
    #[must_use]
    pub fn components_f64(&self) -> Option<&[f64]> {
        self.fit.as_ref().map(|f| f.components.as_slice())
    }

    /// Singular values of the centered data for the kept components (sklearn `singular_values_`).
    #[must_use]
    pub fn singular_values(&self) -> Option<&[f64]> {
        self.fit.as_ref().map(|f| f.singular_values.as_slice())
    }

    /// Per-feature mean at full precision.
    #[must_use]
    pub fn mean_f64(&self) -> Option<&[f64]> {
        self.fit.as_ref().map(|f| f.mean.as_slice())
    }

    fn fitted(&self) -> Result<&PcaFit> {
        self.fit
            .as_ref()
            .ok_or_else(|| AprenderError::from("PCA not fitted"))
    }

    pub fn inverse_transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let fit = self.fitted()?;
        let (n_samples, n_components) = x.shape();
        let d = fit.mean.len();

        if n_components != fit.k {
            return Err("Input has wrong number of components".into());
        }

        // X_reconstructed = X_pca @ components + mean
        let mut result = Vec::with_capacity(n_samples * d);
        for row in x.as_slice().chunks_exact(n_components.max(1)).take(n_samples) {
            for j in 0..d {
                let mut value = fit.mean[j];
                for (c, &z) in row.iter().enumerate() {
                    value += f64::from(z) * fit.components[c * d + j];
                }
                result.push(value as f32);
            }
        }
        Matrix::from_vec(n_samples, d, result).map_err(Into::into)
    }

    /// Validate the request against the data shape; `None` means "decide from the spectrum".
    fn requested_k(&self, n: usize, d: usize) -> Result<Option<usize>> {
        if n < 2 {
            return Err("PCA requires at least 2 samples".into());
        }
        match self.n_components {
            NComponents::Count(k) if k > d => {
                Err("n_components cannot exceed number of features".into())
            }
            NComponents::Count(k) if k == 0 || k > n => Err(format!(
                "n_components must be in 1..=min(n_samples, n_features) = 1..={}, got {k}",
                n.min(d)
            )
            .into()),
            NComponents::Count(k) => Ok(Some(k)),
            NComponents::VarianceRatio(r) if r > 0.0 && r < 1.0 => Ok(None),
            NComponents::VarianceRatio(r) => {
                Err(format!("variance ratio must lie in (0, 1), got {r}").into())
            }
        }
    }

    /// Resolve [`SvdSolver::Auto`]; `Some(power_iters)` means randomized.
    fn randomized_iters(&self, n: usize, d: usize, k: Option<usize>) -> Result<Option<usize>> {
        let min_dim = n.min(d);
        match (self.svd_solver, k) {
            (SvdSolver::Full, _) => Ok(None),
            (SvdSolver::Randomized, None) => {
                Err("a variance-ratio n_components needs the full spectrum: use SvdSolver::Full or Auto".into())
            }
            (SvdSolver::Randomized, Some(k)) => Ok(Some(auto_power_iters(k, min_dim))),
            (SvdSolver::Auto, None) => Ok(None),
            (SvdSolver::Auto, Some(k)) => {
                let small = n.max(d) <= 500;
                let low_rank = (k as f64) < 0.8 * min_dim as f64;
                Ok((!small && low_rank).then(|| auto_power_iters(k, min_dim)))
            }
        }
    }
}

/// sklearn: `searchsorted(cumsum(ratio), r, side="right") + 1`, capped at the spectrum length.
fn components_for_ratio(ratios: &[f64], r: f64) -> usize {
    let mut acc = 0.0;
    let below = ratios
        .iter()
        .take_while(|&&v| {
            acc += v;
            acc <= r
        })
        .count();
    (below + 1).min(ratios.len())
}

// Contract: pca-v1, equation = "pca_transform"
impl Transformer for PCA {
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n, d) = x.shape();
        let requested = self.requested_k(n, d)?;
        let power_iters = self.randomized_iters(n, d, requested)?;

        let raw = to_f64_centered(x.as_slice(), d, None);
        let mean = column_means(&raw, n, d);
        let centered: Vec<f64> = raw.iter().enumerate().map(|(i, v)| v - mean[i % d]).collect();
        let denom = (n - 1) as f64;

        let (k_svd, randomized) = match (requested, power_iters) {
            (Some(k), Some(q)) => (k, Some((q, self.seed))),
            (Some(k), None) => (k, None),
            (None, _) => (n.min(d), None),
        };
        let (sigma, vt) = top_k_svd(&centered, n, d, k_svd, randomized)?;
        // Exact SVD sees the whole spectrum; a top-k one does not, so its total comes from
        // the Frobenius norm of the centered data (the same sum, computed directly).
        let total = if randomized.is_some() {
            centered.iter().map(|v| v * v).sum::<f64>() / denom
        } else {
            sigma.iter().map(|s| s * s).sum::<f64>() / denom
        };
        let all_ratios: Vec<f64> = sigma.iter().map(|s| s * s / denom / total).collect();
        let k = match self.n_components {
            NComponents::VarianceRatio(r) => components_for_ratio(&all_ratios, r),
            NComponents::Count(k) => k,
        };

        let explained_variance: Vec<f64> = sigma[..k].iter().map(|s| s * s / denom).collect();
        let components = vt[..k * d].to_vec();
        let to_f32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        self.fit = Some(PcaFit {
            k,
            components_f32: Matrix::from_vec(k, d, to_f32(&components))?,
            explained_variance_f32: to_f32(&explained_variance),
            explained_variance_ratio_f32: to_f32(&all_ratios[..k]),
            explained_variance_ratio: all_ratios[..k].to_vec(),
            singular_values: sigma[..k].to_vec(),
            mean,
            components,
            explained_variance,
        });
        Ok(())
    }

    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let fit = self.fitted()?;
        let (n_samples, n_features) = x.shape();
        if n_features != fit.mean.len() {
            return Err("Input has wrong number of features".into());
        }
        // X_pca = (X - mean) @ componentsᵀ, in f64.
        let centered = to_f64_centered(x.as_slice(), n_features, Some(&fit.mean));
        let out = project(&centered, n_features, &fit.components, fit.k);
        Matrix::from_vec(n_samples, fit.k, out.iter().map(|&v| v as f32).collect())
            .map_err(Into::into)
    }
}

/// Fitted state of a [`TruncatedSVD`].
#[derive(Debug, Clone)]
struct TruncatedSvdFit {
    d: usize,
    components: Vec<f64>,
    singular_values: Vec<f64>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
}

/// Truncated SVD (latent semantic analysis): PCA's projection without the centering.
///
/// Centering a TF-IDF or count matrix destroys its sparsity and its meaning (a zero count
/// becomes a negative one), so LSA factors the raw matrix. Works on the dense `f64`
/// matrices [`crate::text::vectorize::TfidfVectorizer`] produces.
///
/// Mirrors `sklearn.decomposition.TruncatedSVD`: randomized by default (`n_iter = 5`,
/// oversample 10), `explained_variance_` is the variance of each transformed column and
/// the ratio divides it by the total per-column variance of the input.
///
/// # Example
///
/// ```
/// use aprender::preprocessing::TruncatedSVD;
/// use aprender::primitives::Matrix;
///
/// let x = Matrix::from_vec(3, 4, vec![
///     1.0, 0.0, 2.0, 0.0,
///     0.0, 3.0, 0.0, 1.0,
///     1.0, 0.0, 2.0, 1.0,
/// ]).expect("valid matrix dimensions");
/// let mut svd = TruncatedSVD::new(2);
/// let z = svd.fit_transform(&x).expect("fit_transform should succeed");
/// assert_eq!(z.shape(), (3, 2));
/// ```
#[derive(Debug, Clone)]
pub struct TruncatedSVD {
    n_components: usize,
    solver: SvdSolver,
    n_iter: usize,
    seed: u64,
    fit: Option<TruncatedSvdFit>,
}

impl TruncatedSVD {
    /// Keep `n_components` components; randomized with `n_iter = 5`, seed 0.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            solver: SvdSolver::Randomized,
            n_iter: 5,
            seed: 0,
            fit: None,
        }
    }

    /// [`SvdSolver::Full`] for an exact decomposition; `Auto` behaves as `Randomized`.
    #[must_use]
    pub fn with_svd_solver(mut self, solver: SvdSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Power iterations for the randomized solver.
    #[must_use]
    pub fn with_n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Seed for the randomized solver.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Components, row-major `n_components x n_features`.
    #[must_use]
    pub fn components(&self) -> Option<&[f64]> {
        self.fit.as_ref().map(|f| f.components.as_slice())
    }

    #[must_use]
    pub fn singular_values(&self) -> Option<&[f64]> {
        self.fit.as_ref().map(|f| f.singular_values.as_slice())
    }

    #[must_use]
    pub fn explained_variance(&self) -> Option<&[f64]> {
        self.fit.as_ref().map(|f| f.explained_variance.as_slice())
    }

    #[must_use]
    pub fn explained_variance_ratio(&self) -> Option<&[f64]> {
        self.fit.as_ref().map(|f| f.explained_variance_ratio.as_slice())
    }

    /// Fit on a dense `n_samples x n_features` matrix.
    pub fn fit(&mut self, x: &Matrix<f64>) -> Result<()> {
        self.fit_transform(x).map(|_| ())
    }

    /// Fit, then return the `n_samples x n_components` projection.
    pub fn fit_transform(&mut self, x: &Matrix<f64>) -> Result<Matrix<f64>> {
        let (n, d) = x.shape();
        let k = self.n_components;
        if k == 0 || k > n.min(d) {
            return Err(format!(
                "n_components must be in 1..=min(n_samples, n_features) = 1..={}, got {k}",
                n.min(d)
            )
            .into());
        }
        let randomized = (self.solver != SvdSolver::Full).then_some((self.n_iter, self.seed));
        let (sigma, vt) = top_k_svd(x.as_slice(), n, d, k, randomized)?;
        let components = vt[..k * d].to_vec();
        let z = project(x.as_slice(), d, &components, k);

        let total: f64 = column_variances(x.as_slice(), n, d).iter().sum();
        let explained_variance = column_variances(&z, n, k);
        let explained_variance_ratio = explained_variance.iter().map(|v| v / total).collect();
        self.fit = Some(TruncatedSvdFit {
            d,
            components,
            singular_values: sigma[..k].to_vec(),
            explained_variance,
            explained_variance_ratio,
        });
        Matrix::from_vec(n, k, z).map_err(Into::into)
    }

    /// Project onto the fitted components (no centering).
    pub fn transform(&self, x: &Matrix<f64>) -> Result<Matrix<f64>> {
        let fit = self
            .fit
            .as_ref()
            .ok_or_else(|| AprenderError::from("TruncatedSVD not fitted"))?;
        let (n, d) = x.shape();
        if d != fit.d {
            return Err("Input has wrong number of features".into());
        }
        let k = fit.singular_values.len();
        Matrix::from_vec(n, k, project(x.as_slice(), d, &fit.components, k)).map_err(Into::into)
    }
}

/// Population (ddof = 0) variance of every column, as `numpy.var(x, axis=0)`.
fn column_variances(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mean = column_means(x, n, d);
    let mut var = vec![0.0_f64; d];
    for row in x.chunks_exact(d) {
        for ((s, v), m) in var.iter_mut().zip(row).zip(&mean) {
            *s += (v - m) * (v - m);
        }
    }
    for s in &mut var {
        *s /= n as f64;
    }
    var
}

#[cfg(test)]
#[path = "tests_pca_contract.rs"]
mod tests_pca_contract;

#[cfg(test)]
#[path = "tests_pca_svd.rs"]
mod tests_pca_svd;
