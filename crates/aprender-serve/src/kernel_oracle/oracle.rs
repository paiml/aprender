//! The mathematical definition of each op, evaluated in f64 (KTEST-001 §0.3).
//!
//! Every function here is the plain scalar formula: no SIMD, no blocking, no reassociation trick.
//! Inputs are f32 (the values the kernel under test actually read, widened exactly); outputs are
//! f64. Layouts are row-major, like APR and realizar (LAYOUT-001).
//!
//! The `*_abs` companions compute Σ|aᵢ||bᵢ| per output: the magnitude the error model scales
//! (Higham Thm 3.1), evaluated in f64 from |A| and |B| as §3.2 prescribes.

/// Σ aᵢ·bᵢ.
///
/// # Panics
/// If the lengths differ: a kernel judged against a mis-shaped oracle is a harness bug.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "oracle::dot: length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// Σ |aᵢ|·|bᵢ|: the magnitude `EM-DOT` scales.
///
/// # Panics
/// If the lengths differ.
#[must_use]
pub fn dot_abs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "oracle::dot_abs: length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x).abs() * f64::from(y).abs())
        .sum()
}

/// y = W·x with W row-major `rows × cols`.
///
/// # Panics
/// If `w.len() != rows * cols` or `x.len() != cols`.
#[must_use]
pub fn gemv(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f64> {
    assert_eq!(w.len(), rows * cols, "oracle::gemv: W is not rows × cols");
    assert_eq!(x.len(), cols, "oracle::gemv: x is not cols long");
    w.chunks_exact(cols).map(|row| dot(row, x)).collect()
}

/// (|W|·|x|)ᵣ for every row r of a row-major `rows × cols` W.
///
/// # Panics
/// If `w.len() != rows * cols` or `x.len() != cols`.
#[must_use]
pub fn gemv_abs(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f64> {
    assert_eq!(
        w.len(),
        rows * cols,
        "oracle::gemv_abs: W is not rows × cols"
    );
    assert_eq!(x.len(), cols, "oracle::gemv_abs: x is not cols long");
    w.chunks_exact(cols).map(|row| dot_abs(row, x)).collect()
}

fn gemm_with(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    term: impl Fn(f32, f32) -> f64,
) -> Vec<f64> {
    assert_eq!(a.len(), m * k, "oracle::gemm: A is not m × k");
    assert_eq!(b.len(), k * n, "oracle::gemm: B is not k × n");
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|p| term(a[i * k + p], b[p * n + j])).sum();
        }
    }
    c
}

/// C = A·B with A row-major `m × k`, B row-major `k × n`; C is row-major `m × n`.
///
/// # Panics
/// If a buffer does not match its shape.
#[must_use]
pub fn gemm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f64> {
    gemm_with(a, m, k, b, n, |x, y| f64::from(x) * f64::from(y))
}

/// (|A|·|B|)ᵢⱼ, the `EM-DOT` magnitude of every GEMM output.
///
/// # Panics
/// If a buffer does not match its shape.
#[must_use]
pub fn gemm_abs(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f64> {
    gemm_with(a, m, k, b, n, |x, y| {
        f64::from(x).abs() * f64::from(y).abs()
    })
}

/// Σ xᵢ.
#[must_use]
pub fn sum(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v)).sum()
}

/// Σ |xᵢ|: the magnitude `EM-RED` scales for a sum.
#[must_use]
pub fn sum_abs(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v).abs()).sum()
}

/// max xᵢ, NaN if any xᵢ is NaN (a kernel that drops a NaN hides a fault), −∞ for an empty slice.
#[must_use]
pub fn max(x: &[f32]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in x {
        let v = f64::from(v);
        if v.is_nan() {
            return f64::NAN;
        }
        if v > m {
            m = v;
        }
    }
    m
}

/// softmax(x)ᵢ = exp(xᵢ − max x) / Σⱼ exp(xⱼ − max x), in f64.
#[must_use]
pub fn softmax(x: &[f32]) -> Vec<f64> {
    let m = max(x);
    let e: Vec<f64> = x.iter().map(|&v| (f64::from(v) - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// RMSNorm: yᵢ = xᵢ / sqrt(mean(x²) + eps) · wᵢ.
///
/// # Panics
/// If `x` and `w` differ in length.
#[must_use]
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f64> {
    assert_eq!(x.len(), w.len(), "oracle::rmsnorm: length mismatch");
    #[allow(clippy::cast_precision_loss)] // a row length, far below 2^53
    let n = x.len() as f64;
    let ms = x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (ms + f64::from(eps)).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &g)| f64::from(v) * inv * f64::from(g))
        .collect()
}

/// SiLU: xᵢ / (1 + exp(−xᵢ)).
#[must_use]
pub fn silu(x: &[f32]) -> Vec<f64> {
    x.iter()
        .map(|&v| {
            let v = f64::from(v);
            v / (1.0 + (-v).exp())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemv_is_row_major() {
        // W = [[1, 2, 3], [4, 5, 6]], x = [1, 0, -1] → [-2, -2]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(gemv(&w, 2, 3, &[1.0, 0.0, -1.0]), vec![-2.0, -2.0]);
        assert_eq!(gemv_abs(&w, 2, 3, &[1.0, 0.0, -1.0]), vec![4.0, 10.0]);
    }

    #[test]
    fn gemm_matches_hand_product() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let c = gemm(&[1.0, 2.0, 3.0, 4.0], 2, 2, &[5.0, 6.0, 7.0, 8.0], 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
        let c = gemm_abs(&[1.0, -2.0, 3.0, 4.0], 2, 2, &[5.0, 6.0, -7.0, 8.0], 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_is_gemv_per_column() {
        let a = [0.5, -1.25, 2.0, 3.0, 0.75, -0.5];
        let b = [1.0, -2.0, 0.25, 4.0, 3.0, 1.5];
        let c = gemm(&a, 2, 3, &b, 2);
        for j in 0..2 {
            let col: Vec<f32> = (0..3).map(|p| b[p * 2 + j]).collect();
            let y = gemv(&a, 2, 3, &col);
            assert_eq!([c[j], c[2 + j]], [y[0], y[1]]);
        }
    }

    #[test]
    fn softmax_is_a_distribution_and_shift_invariant() {
        let x = [1.0, 2.0, 3.0, -4.0];
        let p = softmax(&x);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        let shifted: Vec<f32> = x.iter().map(|v| v + 100.0).collect();
        for (a, b) in p.iter().zip(softmax(&shifted)) {
            assert!((a - b).abs() < 1e-15);
        }
        // near-overflow logits stay finite: max-subtraction is part of the definition here
        let big = softmax(&[1000.0, 1000.0]);
        assert_eq!(big, vec![0.5, 0.5]);
    }

    #[test]
    fn max_propagates_nan() {
        assert!(max(&[1.0, f32::NAN, 3.0]).is_nan());
        assert!((max(&[1.0, -3.0, 2.5]) - 2.5).abs() < f64::EPSILON);
        assert!(max(&[]).is_infinite());
    }

    #[test]
    fn rmsnorm_and_silu_known_values() {
        // x = [3, 4]: mean(x²) = 12.5 → y = x / sqrt(12.5)
        let y = rmsnorm(&[3.0, 4.0], &[1.0, 2.0], 0.0);
        let r = 12.5_f64.sqrt();
        assert!((y[0] - 3.0 / r).abs() < 1e-15 && (y[1] - 8.0 / r).abs() < 1e-15);
        let s = silu(&[0.0, 1.0]);
        assert!(s[0].abs() < f64::EPSILON);
        assert!((s[1] - 1.0 / (1.0 + (-1.0_f64).exp())).abs() < 1e-15);
        assert!((sum(&[1.0, -2.0, 4.0]) - 3.0).abs() < f64::EPSILON);
        assert!((sum_abs(&[1.0, -2.0, 4.0]) - 7.0).abs() < f64::EPSILON);
    }
}
