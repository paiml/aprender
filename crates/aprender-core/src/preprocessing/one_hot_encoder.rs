//! `OneHotEncoder` — expand integer-coded categorical features into one-hot
//! binary columns (Pillar 1 — beat scikit-learn). Mirrors
//! `sklearn.preprocessing.OneHotEncoder` (dense): each input column becomes
//! `k` binary columns (`k` = its number of unique categories), concatenated
//! left-to-right in fitted-column order, categories sorted ascending.

use crate::error::{AprenderError, Result};
use crate::primitives::Matrix;
use crate::traits::Transformer;
use core::cmp::Ordering;

/// One-hot encodes categorical (integer-coded) feature columns.
#[derive(Debug, Clone, Default)]
pub struct OneHotEncoder {
    /// Per-column sorted unique categories (learned during fit).
    categories: Option<Vec<Vec<f32>>>,
}

impl OneHotEncoder {
    /// Create a new (unfitted) `OneHotEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self { categories: None }
    }

    /// The fitted per-column categories (sorted), or `None` if unfitted.
    #[must_use]
    pub fn categories(&self) -> Option<&[Vec<f32>]> {
        self.categories.as_deref()
    }

    /// Total number of output columns (sum of per-column category counts).
    #[must_use]
    pub fn output_width(&self) -> usize {
        self.categories
            .as_ref()
            .map_or(0, |c| c.iter().map(Vec::len).sum())
    }
}

impl Transformer for OneHotEncoder {
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();
        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }
        let mut cats = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let mut col: Vec<f32> = (0..n_samples).map(|i| x.get(i, j)).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            col.dedup();
            cats.push(col);
        }
        self.categories = Some(cats);
        Ok(())
    }

    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let cats = self
            .categories
            .as_ref()
            .ok_or_else(|| AprenderError::from("OneHotEncoder not fitted"))?;
        let (n_samples, n_features) = x.shape();
        if n_features != cats.len() {
            return Err("Feature dimension mismatch".into());
        }
        let out_width: usize = cats.iter().map(Vec::len).sum();
        let mut result = vec![0.0f32; n_samples * out_width];
        for i in 0..n_samples {
            let mut offset = 0;
            for (j, col_cats) in cats.iter().enumerate() {
                let v = x.get(i, j);
                // unknown category -> all-zero block (handle_unknown='ignore')
                if let Ok(idx) =
                    col_cats.binary_search_by(|c| c.partial_cmp(&v).unwrap_or(Ordering::Equal))
                {
                    result[i * out_width + offset + idx] = 1.0;
                }
                offset += col_cats.len();
            }
        }
        Matrix::from_vec(n_samples, out_width, result).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FT-PREP-ONEHOT: matches `sklearn.preprocessing.OneHotEncoder` (dense).
    #[test]
    fn one_hot_encoder_matches_sklearn() {
        // col0 categories {0,1}; col1 categories {0,1,2} -> width 5
        let x =
            Matrix::from_vec(4, 2, vec![0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0]).expect("valid");
        let mut oh = OneHotEncoder::new();
        oh.fit(&x).expect("fit");
        assert_eq!(oh.output_width(), 5);
        let out = oh.transform(&x).expect("transform");
        assert_eq!(out.shape(), (4, 5));
        let expect = [
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
        ];
        for (i, row) in expect.iter().enumerate() {
            for (j, e) in row.iter().enumerate() {
                assert!(
                    (out.get(i, j) - *e as f32).abs() < 1e-6,
                    "onehot[{i}][{j}] = {} != {e}",
                    out.get(i, j)
                );
            }
        }
    }
}
