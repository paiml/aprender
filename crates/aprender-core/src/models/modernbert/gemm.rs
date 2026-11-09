//! Dense products on `trueno::blis::gemm_blis` in the spike-020 layout.
//!
//! `y = x W^T (+ b)` is computed as `C^T = W . X^T`: the checkpoint's `[out, in]`
//! row-major weight IS the GEMM's A operand (no weight transpose), `X^T` is the B
//! operand, and W's output rows are banded across the rayon pool (bands = 2 x
//! threads, band rows a multiple of 8). Numerics and their order are spike 025's.

use super::{check_len, ModernBertError};
use rayon::prelude::*;

/// `[rows, cols]` -> `[cols, rows]` through `trueno::blis::transpose` (values moved,
/// never computed, so the result is bit-exact whatever kernel it picks).
fn transpose(src: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, ModernBertError> {
    let mut dst = vec![0.0f32; rows * cols];
    if rows == 0 || cols == 0 {
        return Ok(dst);
    }
    trueno::blis::transpose::transpose(rows, cols, src, &mut dst)
        .map_err(|e| ModernBertError::Gemm(e.to_string()))?;
    Ok(dst)
}

/// A dense layer: `w` is `[out, inp]` row-major exactly as stored, `b` an optional
/// `[out]` bias.
#[derive(Debug, Clone)]
pub struct Linear {
    /// `[out, inp]` row-major weight.
    pub w: Vec<f32>,
    /// Optional `[out]` bias.
    pub b: Option<Vec<f32>>,
    /// Output features.
    pub out: usize,
    /// Input features.
    pub inp: usize,
}

impl Linear {
    /// `x` is `[m, inp]`; returns `[m, out]`.
    ///
    /// # Errors
    ///
    /// [`ModernBertError::InputShape`] when `x`, `w` or `b` disagree with `m`, `out`,
    /// `inp`; [`ModernBertError::Gemm`] when `gemm_blis` refuses its operands.
    pub fn forward(&self, x: &[f32], m: usize) -> Result<Vec<f32>, ModernBertError> {
        let (n, k) = (self.out, self.inp);
        check_len("linear.x", x.len(), &[m, k])?;
        check_len("linear.w", self.w.len(), &[n, k])?;
        if let Some(b) = &self.b {
            check_len("linear.b", b.len(), &[n])?;
        }
        if m == 0 || n == 0 {
            return Ok(Vec::new());
        }
        let xt = transpose(x, m, k)?;
        let mut ct = vec![0.0f32; n * m];
        let bands = (rayon::current_num_threads() * 2).max(1);
        let band = n.div_ceil(bands).next_multiple_of(8).max(8);
        ct.par_chunks_mut(band * m)
            .enumerate()
            .try_for_each(|(i, c)| {
                let r0 = i * band;
                let rows = c.len() / m;
                trueno::blis::gemm_blis(rows, m, k, &self.w[r0 * k..(r0 + rows) * k], &xt, c, None)
                    .map_err(|e| ModernBertError::Gemm(e.to_string()))
            })?;
        let mut y = transpose(&ct, n, m)?;
        if let Some(b) = &self.b {
            y.par_chunks_mut(n)
                .for_each(|r| r.iter_mut().zip(b).for_each(|(v, bb)| *v += bb));
        }
        Ok(y)
    }
}
