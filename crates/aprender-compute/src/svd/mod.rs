//! Singular value decomposition for dense `f64` matrices (#3147).
//!
//! The substrate for PCA, TruncatedSVD and least squares: `A = U Σ Vᵀ` for any
//! row-major `m × n` matrix, tall, square or wide.
//!
//! # Algorithms
//!
//! - [`Svd::new`] — thin SVD by Golub-Reinsch: Householder bidiagonalization,
//!   then implicit-shift QR on the bidiagonal. A wide input (`m < n`) is
//!   decomposed as its transpose and the factors are swapped back.
//! - [`Svd::randomized`] — the Halko-Martinsson-Tropp range finder with
//!   oversampling `p` and `q` power iterations, for the top `k` triplets of a
//!   large matrix.
//!
//! # Conventions
//!
//! - Thin shapes, `r = min(m, n)`: `U` is `m × r`, `Σ` has `r` entries, `Vᵀ` is `r × n`.
//!   All are row-major.
//! - Singular values are non-negative and sorted descending.
//! - **Deterministic sign**: the largest-magnitude component of every left
//!   singular vector (column of `U`) is positive, the first such component on a
//!   tie. The matching row of `Vᵀ` flips with it, so `U Σ Vᵀ` is unchanged. Two
//!   runs on the same input return the same bits, and results can be compared
//!   against another library once that library's output gets the same rule.
//! - **Rank deficiency**: singular values past the numerical rank are not
//!   forced to zero. They are bounded by `max(m, n) · ε · σ₀`
//!   ([`Svd::rank_tolerance`]), the backward error of the algorithm.
//!
//! Sparse, complex and GPU SVD are out of scope.
//!
//! # Example
//!
//! ```
//! use trueno::Svd;
//!
//! // 3 × 2, row-major
//! let a = [3.0, 0.0,
//!          0.0, 4.0,
//!          0.0, 0.0];
//! let svd = Svd::new(&a, 3, 2)?;
//! assert!((svd.singular_values()[0] - 4.0).abs() < 1e-12);
//! assert!((svd.singular_values()[1] - 3.0).abs() < 1e-12);
//!
//! let back = svd.reconstruct();
//! assert!(a.iter().zip(&back).all(|(x, y)| (x - y).abs() < 1e-12));
//! # Ok::<(), trueno::TruenoError>(())
//! ```

mod golub_reinsch;
#[cfg(test)]
mod tests;

use crate::TruenoError;

/// Thin singular value decomposition `A = U Σ Vᵀ` of an `m × n` `f64` matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Svd {
    rows: usize,
    cols: usize,
    /// `rows × k`, row-major.
    u: Vec<f64>,
    /// `k` values, descending.
    s: Vec<f64>,
    /// `k × cols`, row-major.
    vt: Vec<f64>,
}

/// Parameters for [`Svd::randomized`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomizedSvdConfig {
    /// Number of singular triplets to return, `1 ≤ k ≤ min(m, n)`.
    pub k: usize,
    /// Extra sample columns `p` beyond `k` (Halko et al. recommend 5–10).
    pub oversample: usize,
    /// Power iterations `q`; each sharpens a slowly decaying spectrum.
    ///
    /// With `p = 10, q = 2` the top-`k` values land within 1% when the spectrum
    /// halves across the oversampling window, `σ_{k+p+1} / σ_k ≤ 0.5`. A flatter
    /// spectrum needs a larger `q` (harmonic `1/i` at `k = 20` measured 1.8% at
    /// `q = 2` and under 1% at `q = 6`).
    pub power_iters: usize,
    /// Seed for the Gaussian test matrix. The same seed gives the same result.
    pub seed: u64,
}

impl RandomizedSvdConfig {
    /// `k` triplets with the defaults the #3147 acceptance measures: `p = 10`, `q = 2`.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k, oversample: 10, power_iters: 2, seed: 0x5eed_5eed }
    }
}

impl Svd {
    /// Thin SVD of the row-major `rows × cols` matrix `a`.
    ///
    /// # Errors
    ///
    /// [`TruenoError::InvalidInput`] if a dimension is zero, `a.len() != rows * cols`,
    /// an entry is not finite, or the QR iteration fails to converge.
    pub fn new(a: &[f64], rows: usize, cols: usize) -> Result<Self, TruenoError> {
        validate(a, rows, cols)?;
        let mut svd = Self::unsigned(a, rows, cols)?;
        svd.canonicalize_signs();
        Ok(svd)
    }

    /// Top-`k` SVD by randomized range finding (Halko, Martinsson & Tropp 2011, Alg. 4.4 + 5.1).
    ///
    /// # Errors
    ///
    /// Everything [`Svd::new`] rejects, plus `k == 0` or `k > min(rows, cols)`.
    pub fn randomized(
        a: &[f64],
        rows: usize,
        cols: usize,
        config: RandomizedSvdConfig,
    ) -> Result<Self, TruenoError> {
        validate(a, rows, cols)?;
        let r = rows.min(cols);
        if config.k == 0 || config.k > r {
            return Err(TruenoError::InvalidInput(format!(
                "randomized SVD needs 1 <= k <= min(rows, cols) = {r}, got k = {}",
                config.k
            )));
        }
        let l = (config.k + config.oversample).min(r);
        let omega = gaussian(cols, l, config.seed);
        let mut q = orthonormal_basis(&matmul(a, &omega, rows, cols, l), rows, l)?;
        let at = transpose(a, rows, cols);
        for _ in 0..config.power_iters {
            let z = orthonormal_basis(&matmul(&at, &q, cols, rows, l), cols, l)?;
            q = orthonormal_basis(&matmul(a, &z, rows, cols, l), rows, l)?;
        }
        // B = Qᵀ A is l × cols with l ≤ cols; its SVD lifts back through Q.
        let b = matmul(&transpose(&q, rows, l), a, l, rows, cols);
        let small = Self::unsigned(&b, l, cols)?;
        let u_full = matmul(&q, &small.u, rows, l, l);
        let k = config.k;
        let mut svd = Self {
            rows,
            cols,
            u: u_full.chunks_exact(l).flat_map(|row| row[..k].iter().copied()).collect(),
            s: small.s[..k].to_vec(),
            vt: small.vt[..k * cols].to_vec(),
        };
        svd.canonicalize_signs();
        Ok(svd)
    }

    /// Rows `m` of the decomposed matrix.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns `n` of the decomposed matrix.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of singular triplets held: `min(m, n)`, or `k` for [`Svd::randomized`].
    #[must_use]
    pub fn len(&self) -> usize {
        self.s.len()
    }

    /// Always false: a decomposition holds at least one triplet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    /// Left singular vectors, `rows × len()`, row-major; column `j` pairs with `σⱼ`.
    #[must_use]
    pub fn u(&self) -> &[f64] {
        &self.u
    }

    /// Singular values, non-negative, descending.
    #[must_use]
    pub fn singular_values(&self) -> &[f64] {
        &self.s
    }

    /// Right singular vectors transposed, `len() × cols`, row-major; row `j` pairs with `σⱼ`.
    #[must_use]
    pub fn vt(&self) -> &[f64] {
        &self.vt
    }

    /// The bound below which a singular value is numerically zero:
    /// `max(m, n) · ε · σ₀`.
    #[must_use]
    pub fn rank_tolerance(&self) -> f64 {
        self.rows.max(self.cols) as f64 * f64::EPSILON * self.s.first().copied().unwrap_or(0.0)
    }

    /// Numerical rank: how many singular values exceed [`Svd::rank_tolerance`].
    #[must_use]
    pub fn rank(&self) -> usize {
        let tol = self.rank_tolerance();
        self.s.iter().filter(|&&x| x > tol).count()
    }

    /// `U Σ Vᵀ`, row-major `rows × cols`.
    #[must_use]
    pub fn reconstruct(&self) -> Vec<f64> {
        let k = self.len();
        let mut out = vec![0.0; self.rows * self.cols];
        for (i, out_row) in out.chunks_exact_mut(self.cols).enumerate() {
            for j in 0..k {
                let w = self.u[i * k + j] * self.s[j];
                for (o, v) in out_row.iter_mut().zip(&self.vt[j * self.cols..(j + 1) * self.cols]) {
                    *o += w * v;
                }
            }
        }
        out
    }

    /// Golub-Reinsch on `a`, transposing a wide input; signs not yet canonical.
    fn unsigned(a: &[f64], rows: usize, cols: usize) -> Result<Self, TruenoError> {
        if rows >= cols {
            let t = golub_reinsch::decompose(a, rows, cols)?;
            Ok(Self { rows, cols, u: t.u, s: t.s, vt: transpose(&t.v, cols, cols) })
        } else {
            // Aᵀ = U' Σ V'ᵀ  ⇒  A = V' Σ U'ᵀ.
            let t = golub_reinsch::decompose(&transpose(a, rows, cols), cols, rows)?;
            Ok(Self { rows, cols, u: t.v, s: t.s, vt: transpose(&t.u, cols, rows) })
        }
    }

    /// Flip each (column of U, row of Vᵀ) pair so U's largest-magnitude entry is positive.
    fn canonicalize_signs(&mut self) {
        let k = self.len();
        for j in 0..k {
            let mut pivot = 0.0_f64;
            for i in 0..self.rows {
                let x = self.u[i * k + j];
                if x.abs() > pivot.abs() {
                    pivot = x;
                }
            }
            if pivot < 0.0 {
                for i in 0..self.rows {
                    self.u[i * k + j] = -self.u[i * k + j];
                }
                for x in &mut self.vt[j * self.cols..(j + 1) * self.cols] {
                    *x = -*x;
                }
            }
        }
    }
}

fn validate(a: &[f64], rows: usize, cols: usize) -> Result<(), TruenoError> {
    if rows == 0 || cols == 0 {
        return Err(TruenoError::InvalidInput(format!("SVD of an empty {rows}x{cols} matrix")));
    }
    if rows.checked_mul(cols) != Some(a.len()) {
        return Err(TruenoError::InvalidInput(format!(
            "SVD input has {} entries, a {rows}x{cols} matrix needs {}",
            a.len(),
            rows.saturating_mul(cols)
        )));
    }
    if let Some(i) = a.iter().position(|x| !x.is_finite()) {
        return Err(TruenoError::InvalidInput(format!(
            "SVD input entry {i} is not finite ({})",
            a[i]
        )));
    }
    Ok(())
}

/// An orthonormal basis (`rows × cols`, `rows ≥ cols`) for the range of `y`,
/// taken from the left factor of its SVD. The columns stay orthonormal when
/// `y` is rank-deficient, which a Gram-Schmidt pass would not guarantee.
fn orthonormal_basis(y: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>, TruenoError> {
    Ok(golub_reinsch::decompose(y, rows, cols)?.u)
}

/// `a (m × k) · b (k × n)`, all row-major.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for (out_row, a_row) in out.chunks_exact_mut(n).zip(a.chunks_exact(k)) {
        for (&x, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
            for (o, y) in out_row.iter_mut().zip(b_row) {
                *o += x * y;
            }
        }
    }
    out
}

/// Transpose of the row-major `rows × cols` matrix `a`.
fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

/// `rows × cols` standard normal matrix from SplitMix64 + Box-Muller.
fn gaussian(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut next_unit = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        // 53 random bits in (0, 1]: never 0, so ln() is finite.
        ((z >> 11) + 1) as f64 / (1u64 << 53) as f64
    };
    (0..rows * cols)
        .map(|_| {
            let (u1, u2) = (next_unit(), next_unit());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect()
}
