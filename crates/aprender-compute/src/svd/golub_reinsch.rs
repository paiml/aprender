//! Golub-Reinsch thin SVD for a row-major `m × n` matrix with `m ≥ n ≥ 1`.
//!
//! Householder bidiagonalization followed by implicit-shift QR on the
//! bidiagonal (Golub & Van Loan §8.6), after the public-domain JAMA
//! formulation. The caller transposes `m < n` inputs, so this file only ever
//! sees tall or square matrices — the case that formulation is correct for.

use crate::TruenoError;

/// `2^-52`, the f64 unit roundoff used by the deflation tests.
const EPS: f64 = f64::EPSILON;
/// `2^-966`: absolute floor so a deflation test never compares against zero.
const TINY: f64 = 1.0e-291;
/// QR sweeps allowed per singular value before reporting non-convergence.
const MAX_SWEEPS_PER_VALUE: usize = 75;

/// Unsorted, unsigned-canonical factors: `A = U diag(s) Vᵀ`.
pub(super) struct Thin {
    /// `m × n`, row-major, orthonormal columns.
    pub u: Vec<f64>,
    /// `n` singular values, non-negative, descending.
    pub s: Vec<f64>,
    /// `n × n`, row-major, orthogonal (V, not Vᵀ).
    pub v: Vec<f64>,
}

struct Work {
    m: usize,
    n: usize,
    a: Vec<f64>,
    s: Vec<f64>,
    e: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
}

/// Decompose `a` (`m × n`, row-major, `m ≥ n ≥ 1`).
pub(super) fn decompose(a: &[f64], m: usize, n: usize) -> Result<Thin, TruenoError> {
    debug_assert!(m >= n && n >= 1 && a.len() == m * n);
    let mut w = Work {
        m,
        n,
        a: a.to_vec(),
        s: vec![0.0; n],
        e: vec![0.0; n],
        u: vec![0.0; m * n],
        v: vec![0.0; n * n],
    };
    let (nct, nrt) = w.bidiagonalize();
    w.generate_u(nct);
    w.generate_v(nrt);
    w.diagonalize()?;
    Ok(Thin { u: w.u, s: w.s, v: w.v })
}

impl Work {
    fn a(&self, i: usize, j: usize) -> f64 {
        self.a[i * self.n + j]
    }

    /// Reduce `A` to upper-bidiagonal form `Uᵀ A V = B`, storing the
    /// Householder vectors in `u` and `v`. Returns `(nct, nrt)`, the number of
    /// left and right reflectors.
    fn bidiagonalize(&mut self) -> (usize, usize) {
        let (m, n) = (self.m, self.n);
        let nct = (m - 1).min(n);
        let nrt = n.saturating_sub(2);
        let mut work = vec![0.0; m];
        for k in 0..nct.max(nrt) {
            if k < nct {
                self.left_reflector(k);
            }
            for j in k + 1..n {
                if k < nct && self.s[k] != 0.0 {
                    self.apply_left_reflector(k, j);
                }
                self.e[j] = self.a(k, j);
            }
            if k < nct {
                for i in k..m {
                    self.u[i * n + k] = self.a(i, k);
                }
            }
            if k < nrt {
                self.right_reflector(k, &mut work);
            }
        }
        if nct < n {
            self.s[nct] = self.a(nct, nct);
        }
        if nrt + 1 < n {
            self.e[nrt] = self.a(nrt, n - 1);
        }
        self.e[n - 1] = 0.0;
        (nct, nrt)
    }

    /// Householder vector annihilating `A[k+1.., k]`; `s[k]` gets the new diagonal.
    fn left_reflector(&mut self, k: usize) {
        let (m, n) = (self.m, self.n);
        let mut norm = 0.0_f64;
        for i in k..m {
            norm = norm.hypot(self.a(i, k));
        }
        if norm != 0.0 {
            if self.a(k, k) < 0.0 {
                norm = -norm;
            }
            for i in k..m {
                self.a[i * n + k] /= norm;
            }
            self.a[k * n + k] += 1.0;
        }
        self.s[k] = -norm;
    }

    /// Apply the `k`-th left reflector to column `j`.
    fn apply_left_reflector(&mut self, k: usize, j: usize) {
        let (m, n) = (self.m, self.n);
        let mut t = 0.0;
        for i in k..m {
            t += self.a(i, k) * self.a(i, j);
        }
        t = -t / self.a(k, k);
        for i in k..m {
            self.a[i * n + j] += t * self.a(i, k);
        }
    }

    /// Householder vector annihilating `A[k, k+2..]` from the right.
    fn right_reflector(&mut self, k: usize, work: &mut [f64]) {
        let (m, n) = (self.m, self.n);
        let mut norm = 0.0_f64;
        for i in k + 1..n {
            norm = norm.hypot(self.e[i]);
        }
        if norm != 0.0 {
            if self.e[k + 1] < 0.0 {
                norm = -norm;
            }
            for i in k + 1..n {
                self.e[i] /= norm;
            }
            self.e[k + 1] += 1.0;
        }
        self.e[k] = -norm;
        if k + 1 < m && self.e[k] != 0.0 {
            work[k + 1..m].fill(0.0);
            for j in k + 1..n {
                for i in k + 1..m {
                    work[i] += self.e[j] * self.a(i, j);
                }
            }
            for j in k + 1..n {
                let t = -self.e[j] / self.e[k + 1];
                for i in k + 1..m {
                    self.a[i * n + j] += t * work[i];
                }
            }
        }
        for i in k + 1..n {
            self.v[i * n + k] = self.e[i];
        }
    }

    /// Accumulate the left reflectors into the explicit `m × n` factor `U`.
    fn generate_u(&mut self, nct: usize) {
        let (m, n) = (self.m, self.n);
        for j in nct..n {
            for i in 0..m {
                self.u[i * n + j] = 0.0;
            }
            self.u[j * n + j] = 1.0;
        }
        for k in (0..nct).rev() {
            if self.s[k] == 0.0 {
                for i in 0..m {
                    self.u[i * n + k] = 0.0;
                }
                self.u[k * n + k] = 1.0;
                continue;
            }
            for j in k + 1..n {
                let mut t = 0.0;
                for i in k..m {
                    t += self.u[i * n + k] * self.u[i * n + j];
                }
                t = -t / self.u[k * n + k];
                for i in k..m {
                    self.u[i * n + j] += t * self.u[i * n + k];
                }
            }
            for i in k..m {
                self.u[i * n + k] = -self.u[i * n + k];
            }
            self.u[k * n + k] += 1.0;
            for i in 0..k {
                self.u[i * n + k] = 0.0;
            }
        }
    }

    /// Accumulate the right reflectors into the explicit `n × n` factor `V`.
    fn generate_v(&mut self, nrt: usize) {
        let n = self.n;
        for k in (0..n).rev() {
            if k < nrt && self.e[k] != 0.0 {
                for j in k + 1..n {
                    let mut t = 0.0;
                    for i in k + 1..n {
                        t += self.v[i * n + k] * self.v[i * n + j];
                    }
                    t = -t / self.v[(k + 1) * n + k];
                    for i in k + 1..n {
                        self.v[i * n + j] += t * self.v[i * n + k];
                    }
                }
            }
            for i in 0..n {
                self.v[i * n + k] = 0.0;
            }
            self.v[k * n + k] = 1.0;
        }
    }

    /// Implicit-shift QR on the bidiagonal `(s, e)` until every `e` deflates.
    fn diagonalize(&mut self) -> Result<(), TruenoError> {
        let mut p = self.n;
        let mut sweeps = 0usize;
        while p > 0 {
            let (kase, k) = self.classify(p);
            match kase {
                Kase::DeflateLast => self.deflate_last(k, p),
                Kase::Split => self.split(k, p),
                Kase::QrStep => {
                    sweeps += 1;
                    if sweeps > MAX_SWEEPS_PER_VALUE * self.n {
                        return Err(TruenoError::InvalidInput(format!(
                            "SVD did not converge after {sweeps} QR sweeps on a {}x{} matrix",
                            self.m, self.n
                        )));
                    }
                    self.qr_step(k, p);
                }
                Kase::Converged => {
                    self.finish_value(k);
                    p -= 1;
                }
            }
        }
        Ok(())
    }

    /// Find the active block `[k, p)` and what to do with it.
    fn classify(&mut self, p: usize) -> (Kase, usize) {
        // `k` is the largest index below p-1 whose off-diagonal e[k] is negligible,
        // or None (the whole leading block is unreduced).
        let mut k: Option<usize> = None;
        for kk in (0..p - 1).rev() {
            let bound = TINY + EPS * (self.s[kk].abs() + self.s[kk + 1].abs());
            if self.e[kk].abs() <= bound {
                self.e[kk] = 0.0;
                k = Some(kk);
                break;
            }
        }
        let lo = k.map_or(0, |kk| kk + 1);
        if lo == p - 1 {
            return (Kase::Converged, lo);
        }
        // Look for a negligible diagonal entry s[ks] with lo ≤ ks < p.
        for ks in (lo..p).rev() {
            let t = if ks != p { self.e[ks].abs() } else { 0.0 }
                + if ks != lo { self.e[ks - 1].abs() } else { 0.0 };
            if self.s[ks].abs() <= TINY + EPS * t {
                self.s[ks] = 0.0;
                return if ks == p - 1 { (Kase::DeflateLast, lo) } else { (Kase::Split, ks + 1) };
            }
        }
        (Kase::QrStep, lo)
    }

    /// `s[p-1]` is negligible: chase `e[p-2]` up the block with right rotations.
    fn deflate_last(&mut self, k: usize, p: usize) {
        let mut f = self.e[p - 2];
        self.e[p - 2] = 0.0;
        for j in (k..p - 1).rev() {
            let t = self.s[j].hypot(f);
            let (cs, sn) = (self.s[j] / t, f / t);
            self.s[j] = t;
            if j != k {
                f = -sn * self.e[j - 1];
                self.e[j - 1] *= cs;
            }
            rotate_cols(&mut self.v, self.n, j, p - 1, cs, sn);
        }
    }

    /// `s[k-1]` is negligible: chase `e[k-1]` down the block with left rotations.
    fn split(&mut self, k: usize, p: usize) {
        let mut f = self.e[k - 1];
        self.e[k - 1] = 0.0;
        for j in k..p {
            let t = self.s[j].hypot(f);
            let (cs, sn) = (self.s[j] / t, f / t);
            self.s[j] = t;
            f = -sn * self.e[j];
            self.e[j] *= cs;
            rotate_cols(&mut self.u, self.n, j, k - 1, cs, sn);
        }
    }

    /// One Golub-Kahan step with a Wilkinson shift on the block `[k, p)`.
    fn qr_step(&mut self, k: usize, p: usize) {
        let (mut f, mut g) = self.shifted_start(k, p);
        for j in k..p - 1 {
            let t = f.hypot(g);
            let (cs, sn) = (f / t, g / t);
            if j != k {
                self.e[j - 1] = t;
            }
            f = cs * self.s[j] + sn * self.e[j];
            self.e[j] = cs * self.e[j] - sn * self.s[j];
            g = sn * self.s[j + 1];
            self.s[j + 1] *= cs;
            rotate_cols(&mut self.v, self.n, j, j + 1, cs, sn);

            let t = f.hypot(g);
            let (cs, sn) = (f / t, g / t);
            self.s[j] = t;
            f = cs * self.e[j] + sn * self.s[j + 1];
            self.s[j + 1] = -sn * self.e[j] + cs * self.s[j + 1];
            g = sn * self.e[j + 1];
            self.e[j + 1] *= cs;
            if j < self.m - 1 {
                rotate_cols(&mut self.u, self.n, j, j + 1, cs, sn);
            }
        }
        self.e[p - 2] = f;
    }

    /// First column of `BᵀB − μI` for the Wilkinson shift μ of the trailing 2×2.
    fn shifted_start(&self, k: usize, p: usize) -> (f64, f64) {
        let scale = [self.s[p - 1], self.s[p - 2], self.e[p - 2], self.s[k], self.e[k]]
            .iter()
            .fold(0.0_f64, |acc, x| acc.max(x.abs()));
        let sp = self.s[p - 1] / scale;
        let spm1 = self.s[p - 2] / scale;
        let epm1 = self.e[p - 2] / scale;
        let sk = self.s[k] / scale;
        let ek = self.e[k] / scale;
        let b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
        let c = (sp * epm1) * (sp * epm1);
        let mut shift = 0.0;
        if b != 0.0 || c != 0.0 {
            shift = (b * b + c).sqrt();
            if b < 0.0 {
                shift = -shift;
            }
            shift = c / (b + shift);
        }
        ((sk + sp) * (sk - sp) + shift, sk * ek)
    }

    /// `s[k]` has converged: make it non-negative and bubble it into
    /// descending position among the already-converged values.
    fn finish_value(&mut self, mut k: usize) {
        let n = self.n;
        if self.s[k] <= 0.0 {
            self.s[k] = if self.s[k] < 0.0 { -self.s[k] } else { 0.0 };
            for i in 0..n {
                self.v[i * n + k] = -self.v[i * n + k];
            }
        }
        while k + 1 < n && self.s[k] < self.s[k + 1] {
            self.s.swap(k, k + 1);
            swap_cols(&mut self.v, n, k, k + 1);
            swap_cols(&mut self.u, n, k, k + 1);
            k += 1;
        }
    }
}

#[derive(Clone, Copy)]
enum Kase {
    DeflateLast,
    Split,
    QrStep,
    Converged,
}

/// Givens rotation of columns `(a, b)` of a row-major matrix with `cols` columns:
/// `a ← cs·a + sn·b`, `b ← −sn·a + cs·b`.
fn rotate_cols(x: &mut [f64], cols: usize, a: usize, b: usize, cs: f64, sn: f64) {
    for row in x.chunks_exact_mut(cols) {
        let t = cs * row[a] + sn * row[b];
        row[b] = -sn * row[a] + cs * row[b];
        row[a] = t;
    }
}

fn swap_cols(x: &mut [f64], cols: usize, a: usize, b: usize) {
    for row in x.chunks_exact_mut(cols) {
        row.swap(a, b);
    }
}
