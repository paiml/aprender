//! R's `lm()` solution for `y ~ x`, step for step (PLOT-GG-001, paiml/aprender#6909): the LINPACK
//! Householder QR `dqrdc2` with limited column pivoting, `dqrsl`, and `predict.lm`'s
//! `qr.solve(qr.R(...))` leverage, all on the UNCENTRED design `[1, x]`.
//!
//! Why not a centred least squares: R declares a column rank-deficient when its norm after the
//! earlier columns are removed falls below `tol * (its original norm)`, `tol = 1e-7`. For x = 1e8,
//! 1e8 + 8, 1e8 + 16 that ratio is 6.5e-8, R drops the slope and draws a flat line, and a centred
//! fit would draw a sloped one. The same arithmetic, in R's order, also reproduces R's own
//! rounding for x far from zero, where QR on the uncentred x loses digits a centred fit keeps.
//!
//! Operation order follows the reference BLAS R is linked against: `ddot` and `dnrm2` are
//! sequential sums (every value in play is far from the over/underflow thresholds of `dnrm2`'s
//! scaled path: data are f32 values widened to f64), `daxpy` and `dscal` are plain loops, and no
//! fused multiply-add is used.
//!
//! A column whose rank test falls within [`KNIFE_EDGE`] of the threshold is refused: R's own verdict
//! there depends on the last digit of a norm and so on the BLAS and platform, which this port cannot
//! promise to share.

use super::path::refuse;
use crate::error::Result;

/// `lm.fit`'s default rank tolerance.
pub(super) const TOL: f64 = 1e-7;
/// Relative distance from the rank threshold inside which the rank verdict is refused.
pub(super) const KNIFE_EDGE: f64 = 1e-6;

fn ddot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |s, (x, y)| s + x * y)
}

fn dnrm2(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |s, v| s + v * v).sqrt()
}

/// A LINPACK QR: column-major `n x p` factor, the Householder scalars, the pivot and the rank.
struct Qr {
    n: usize,
    x: Vec<f64>,
    qraux: Vec<f64>,
    rank: usize,
    /// A rank test came within [`KNIFE_EDGE`] of its threshold.
    edge: bool,
}

/// Is column `l` (0-based) still non-negligible? Records a knife-edge verdict in `edge`.
fn negligible(qraux: f64, orig: f64, edge: &mut bool) -> bool {
    let limit = orig * TOL;
    if (qraux - limit).abs() <= KNIFE_EDGE * limit {
        *edge = true;
    }
    qraux < limit
}

impl Qr {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.x[j * self.n + i]
    }

    /// Cycle negligible columns from `l` to the end, as `dqrdc2` does; returns the updated `k`
    /// (Fortran's 1-based one past the last kept column).
    fn cycle(&mut self, l: usize, mut k: usize, work: &mut [f64], p: usize) -> usize {
        while l + 1 < k && negligible(self.qraux[l], work[l], &mut self.edge) {
            let n = self.n;
            self.x[l * n..p * n].rotate_left(n);
            self.qraux[l..p].rotate_left(1);
            work[l..p].rotate_left(1);
            k -= 1;
        }
        k
    }

    /// The Householder transformation of column `l`, applied to the columns after it.
    fn householder(&mut self, l: usize, p: usize) {
        let n = self.n;
        let (left, right) = self.x.split_at_mut((l + 1) * n);
        let col = &mut left[l * n + l..];
        let mut nrmxl = dnrm2(col);
        if nrmxl == 0.0 {
            return;
        }
        if col[0] != 0.0 {
            nrmxl = nrmxl.copysign(col[0]);
        }
        let inv = 1.0 / nrmxl;
        for v in col.iter_mut() {
            *v *= inv;
        }
        col[0] += 1.0;
        for (jj, cj) in right.chunks_mut(n).take(p - l - 1).enumerate() {
            let target = &mut cj[l..];
            let t = -ddot(col, target) / col[0];
            for (y, u) in target.iter_mut().zip(col.iter()) {
                *y += t * u;
            }
            downdate(&mut self.qraux[l + 1 + jj], target);
        }
        self.qraux[l] = col[0];
        col[0] = -nrmxl;
    }
}

/// `dqrdc2`'s update of a trailing column's norm after a reflection; `col` is the column from the
/// reflected row down.
fn downdate(norm: &mut f64, col: &[f64]) {
    if *norm == 0.0 {
        return;
    }
    let q = col[0].abs() / *norm;
    let t = (1.0 - q * q).max(0.0);
    *norm = if t.abs() < 1e-6 { dnrm2(&col[1..]) } else { *norm * t.sqrt() };
}

/// `dqrdc2(x, n, p, tol = 1e-7)` on a column-major `n x p` matrix.
fn dqrdc2(x: Vec<f64>, n: usize, p: usize) -> Qr {
    let norms: Vec<f64> = x.chunks(n).map(dnrm2).collect();
    let mut work: Vec<f64> = norms.iter().map(|v| if *v == 0.0 { 1.0 } else { *v }).collect();
    let mut qr = Qr { n, x, qraux: norms, rank: 0, edge: false };
    let mut k = p + 1;
    for l in 0..n.min(p) {
        k = qr.cycle(l, k, &mut work, p);
        if l + 1 != n {
            qr.householder(l, p);
        }
    }
    qr.rank = (k - 1).min(n);
    qr
}

impl Qr {
    /// One stored reflection applied to `v` from row `j` down (`x(j, j)` read as `qraux(j)`).
    fn reflect(&self, j: usize, v: &mut [f64]) {
        let n = self.n;
        let col = &self.x[j * n + j..(j + 1) * n];
        let u0 = self.qraux[j];
        let tail = &mut v[j..];
        let dot = col[1..].iter().zip(&tail[1..]).fold(u0 * tail[0], |s, (a, b)| s + a * b);
        let t = -dot / u0;
        tail[0] += t * u0;
        for (y, a) in tail[1..].iter_mut().zip(&col[1..]) {
            *y += t * a;
        }
    }

    /// `Q' y` in place (the reflections in order).
    fn qty(&self, y: &mut [f64]) {
        for j in 0..self.rank.min(self.n - 1) {
            if self.qraux[j] != 0.0 {
                self.reflect(j, y);
            }
        }
    }

    /// The residual vector from `Q' y`: `Q [0; (Q' y)[rank..]]`.
    fn residuals(&self, qty: &[f64]) -> Vec<f64> {
        let mut r: Vec<f64> =
            qty.iter().enumerate().map(|(i, v)| if i < self.rank { 0.0 } else { *v }).collect();
        for j in (0..self.rank.min(self.n - 1)).rev() {
            if self.qraux[j] != 0.0 {
                self.reflect(j, &mut r);
            }
        }
        r
    }

    /// Back-substitution for the first `rank` coefficients from `Q' y`.
    fn coef(&self, qty: &[f64]) -> Result<Vec<f64>> {
        let mut b = qty[..self.rank].to_vec();
        for j in (0..self.rank).rev() {
            let d = self.at(j, j);
            if d == 0.0 {
                return Err(refuse("lm: exact singularity in the QR back-substitution"));
            }
            b[j] /= d;
            let t = -b[j];
            for i in 0..j {
                b[i] += t * self.at(i, j);
            }
        }
        Ok(b)
    }
}

/// `qr.solve(R)` of an upper-triangular `m x m` factor, column-major, as `predict.lm` does: `R` is
/// QR-factored again and solved against the identity.
fn invert_upper(r: Vec<f64>, m: usize) -> Result<Vec<f64>> {
    let qr = dqrdc2(r, m, m);
    if qr.rank != m {
        return Err(refuse("lm: singular matrix in qr.solve(qr.R(...))"));
    }
    let mut inv = Vec::with_capacity(m * m);
    for c in 0..m {
        let mut e = vec![0.0; m];
        e[c] = 1.0;
        qr.qty(&mut e);
        inv.extend(qr.coef(&e)?);
    }
    Ok(inv)
}

/// What `predict.lm(interval = "confidence")` needs from `lm(y ~ x)`.
#[derive(Debug, Clone)]
pub(super) struct LmFit {
    /// R's `object$rank`: 2, or 1 when the slope was dropped.
    pub rank: usize,
    /// Coefficients of the kept columns (`[intercept, slope]` or `[intercept]`).
    beta: Vec<f64>,
    /// `qr.solve(qr.R(qr)[p1, p1])`, column-major `rank x rank`.
    rinv: Vec<f64>,
    /// `rss / df.residual`; `NaN` with no residual degree of freedom and no residual.
    res_var: f64,
    /// `df.residual`.
    pub df: f64,
}

/// `lm(y ~ x)` as R fits it. `Err` when the rank verdict is within [`KNIFE_EDGE`] of its threshold.
pub(super) fn lm_fit(xs: &[f64], ys: &[f64]) -> Result<LmFit> {
    let n = xs.len();
    let design: Vec<f64> = std::iter::repeat(1.0).take(n).chain(xs.iter().copied()).collect();
    let qr = dqrdc2(design, n, 2);
    if qr.edge {
        return Err(refuse(
            "geom_smooth: the slope column is within 1e-6 of R's rank threshold (norm below \
             1e-7 of its original norm), where R's verdict depends on the BLAS; refused",
        ));
    }
    let mut qty = ys.to_vec();
    qr.qty(&mut qty);
    let beta = qr.coef(&qty)?;
    let rss = qr.residuals(&qty).iter().fold(0.0, |s, r| s + r * r);
    let rank = qr.rank;
    let mut r = vec![0.0; rank * rank];
    for j in 0..rank {
        for i in 0..=j {
            r[j * rank + i] = qr.at(i, j);
        }
    }
    let df = (n - rank) as f64;
    Ok(LmFit { rank, beta, rinv: invert_upper(r, rank)?, res_var: rss / df, df })
}

impl LmFit {
    /// `predict.lm(newdata = x0, se.fit = TRUE)`: the fitted value and the standard error.
    pub(super) fn predict(&self, x0: f64) -> (f64, f64) {
        let row = [1.0, x0];
        let rank = self.rank;
        let mut fit = 0.0;
        let mut ip = 0.0;
        for j in 0..rank {
            if self.beta[j] != 0.0 {
                fit += self.beta[j] * row[j];
            }
            let xr = (0..rank).fold(0.0, |s, l| {
                let b = self.rinv[j * rank + l];
                if b == 0.0 {
                    s
                } else {
                    s + b * row[l]
                }
            });
            if self.res_var != 0.0 {
                ip += self.res_var * (xr * xr);
            }
        }
        (fit, ip.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_line_is_recovered() {
        let f = lm_fit(&[1.0, 2.0, 3.0, 4.0], &[3.0, 5.0, 7.0, 9.0]).expect("fits");
        assert_eq!(f.rank, 2);
        assert!((f.beta[0] - 1.0).abs() < 1e-13 && (f.beta[1] - 2.0).abs() < 1e-13);
        let (y, se) = f.predict(2.5);
        assert!((y - 6.0).abs() < 1e-13 && se < 1e-12);
    }

    #[test]
    fn the_rank_rule_is_a_norm_ratio_of_one_in_ten_million() {
        let ys = [1.5, 2.25, 2.0];
        let full = lm_fit(&[1e8, 1e8 + 16.0, 1e8 + 32.0], &ys).expect("full rank");
        assert_eq!(full.rank, 2);
        let flat = lm_fit(&[1e8, 1e8 + 8.0, 1e8 + 16.0], &ys).expect("rank one");
        assert_eq!(flat.rank, 1);
        assert_eq!(flat.df, 2.0);
    }

    #[test]
    fn a_rank_verdict_at_the_threshold_is_refused() {
        let x = [65_319_712.0, 65_319_720.0, 65_319_728.0];
        assert!(lm_fit(&x, &[1.5, 2.25, 2.0]).is_err());
    }
}
