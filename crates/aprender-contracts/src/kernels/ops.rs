//! Shared kernel primitives: dot product, softmax row, score matrix.
//!
//! These building blocks are used across attention, GQA, and flash attention kernels.
//! Centralizing them eliminates duplicated DataTransformation patterns.

/// Dot product of two slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// In-place softmax over a contiguous row.
///
/// Uses the numerically stable formulation: subtract max, exponentiate, normalize.
pub fn softmax_row(row: &mut [f32]) {
    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Apply softmax to each row of a `rows x cols` matrix (in-place).
pub fn softmax_rows(matrix: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(matrix.len(), rows * cols);
    for i in 0..rows {
        softmax_row(&mut matrix[i * cols..(i + 1) * cols]);
    }
}

/// Compute scaled dot-product score matrix: `scores[i,j] = Q[i] . K[j] / sqrt(d)`.
///
/// Q is `m x d`, K is `n x d`, scores is `m x n` (row-major).
pub fn score_matrix(q: &[f32], k: &[f32], m: usize, n: usize, d: usize, scores: &mut [f32]) {
    debug_assert_eq!(q.len(), m * d);
    debug_assert_eq!(k.len(), n * d);
    debug_assert_eq!(scores.len(), m * n);
    let scale = 1.0 / (d as f32).sqrt();

    for i in 0..m {
        for j in 0..n {
            scores[i * n + j] = dot(&q[i * d..(i + 1) * d], &k[j * d..(j + 1) * d]) * scale;
        }
    }
}

/// Matrix multiply: `output = scores * V`, where scores is `rows x cols` and V is `cols x d_v`.
///
/// This is the final step in attention: applying softmax weights to value vectors.
/// `output` must be `rows x d_v`, zeroed or overwritten.
pub fn matmul_sv(
    scores: &[f32],
    v: &[f32],
    rows: usize,
    cols: usize,
    d_v: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(scores.len(), rows * cols);
    debug_assert_eq!(v.len(), cols * d_v);
    debug_assert_eq!(output.len(), rows * d_v);

    for i in 0..rows {
        for j in 0..d_v {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += scores[i * cols + c] * v[c * d_v + j];
            }
            output[i * d_v + j] = sum;
        }
    }
}

/// Weighted sum: `output[i] += weight * v_row[i]` for accumulation in attention.
#[inline]
pub fn weighted_accumulate(output: &mut [f32], weight: f32, v_row: &[f32]) {
    debug_assert_eq!(output.len(), v_row.len());
    for (o, v) in output.iter_mut().zip(v_row.iter()) {
        *o += weight * v;
    }
}

/// Generate a sequential float test vector: `[0*scale, 1*scale, 2*scale, ...]`.
///
/// Used across attention kernel tests to create deterministic Q/K/V test data.
#[cfg(test)]
pub fn sequential_floats(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| (i as f32) * scale).collect()
}

/// Generate a patterned float test vector: `[(i % modulus - offset) * scale, ...]`.
///
/// Used in flash attention tests for varied test data.
#[cfg(test)]
pub fn patterned_floats(len: usize, modulus: usize, offset: f32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % modulus) as f32 - offset) * scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_basic() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn dot_zero() {
        assert_eq!(dot(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn softmax_row_uniform() {
        let mut row = vec![1.0; 4];
        softmax_row(&mut row);
        for v in &row {
            assert!((*v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_row_sums_to_one() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_matrix_basic() {
        // 1x2 Q, 1x2 K => 1x1 scores
        let q = [1.0, 0.0];
        let k = [1.0, 0.0];
        let mut scores = [0.0f32; 1];
        score_matrix(&q, &k, 1, 1, 2, &mut scores);
        // dot = 1.0, scale = 1/sqrt(2) ≈ 0.707
        assert!((scores[0] - 1.0 / 2.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn matmul_sv_basic() {
        // scores = [[0.5, 0.5]], V = [[1.0, 2.0], [3.0, 4.0]]
        // output = [[0.5*1+0.5*3, 0.5*2+0.5*4]] = [[2.0, 3.0]]
        let scores = [0.5, 0.5];
        let v = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 2];
        matmul_sv(&scores, &v, 1, 2, 2, &mut output);
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn matmul_sv_identity_weights() {
        // scores = [[1, 0], [0, 1]], V = [[10, 20], [30, 40]]
        // output = [[10, 20], [30, 40]]
        let scores = [1.0, 0.0, 0.0, 1.0];
        let v = [10.0, 20.0, 30.0, 40.0];
        let mut output = [0.0f32; 4];
        matmul_sv(&scores, &v, 2, 2, 2, &mut output);
        assert!((output[0] - 10.0).abs() < 1e-6);
        assert!((output[1] - 20.0).abs() < 1e-6);
        assert!((output[2] - 30.0).abs() < 1e-6);
        assert!((output[3] - 40.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_accumulate_basic() {
        let mut out = [1.0, 2.0];
        weighted_accumulate(&mut out, 0.5, &[4.0, 6.0]);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }
}
