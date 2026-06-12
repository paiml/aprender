//! `OrdinalEncoder` — encode each categorical feature *column* to integer codes
//! `0..n_categories` by sorted-unique order (Pillar 1 — beat scikit-learn).
//! Mirrors `sklearn.preprocessing.OrdinalEncoder`: like a per-column
//! `LabelEncoder`, output has the same shape as the input (one code per cell),
//! whereas `OneHotEncoder` expands to binary columns.

use crate::error::{AprenderError, Result};
use crate::primitives::Matrix;
use crate::traits::Transformer;
use core::cmp::Ordering;

/// Encodes each feature column to ordinal integer codes (as `f32`).
#[derive(Debug, Clone, Default)]
pub struct OrdinalEncoder {
    /// Per-column sorted unique categories (learned during fit).
    categories: Option<Vec<Vec<f32>>>,
}

impl OrdinalEncoder {
    /// Create a new (unfitted) `OrdinalEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self { categories: None }
    }

    /// The fitted per-column categories (sorted), or `None` if unfitted.
    #[must_use]
    pub fn categories(&self) -> Option<&[Vec<f32>]> {
        self.categories.as_deref()
    }
}

impl Transformer for OrdinalEncoder {
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
            .ok_or_else(|| AprenderError::from("OrdinalEncoder not fitted"))?;
        let (n_samples, n_features) = x.shape();
        if n_features != cats.len() {
            return Err("Feature dimension mismatch".into());
        }
        let mut result = vec![0.0f32; n_samples * n_features];
        for i in 0..n_samples {
            for (j, col_cats) in cats.iter().enumerate() {
                let v = x.get(i, j);
                // unknown category -> n_categories (out-of-range sentinel)
                let code = col_cats
                    .binary_search_by(|c| c.partial_cmp(&v).unwrap_or(Ordering::Equal))
                    .unwrap_or(col_cats.len());
                result[i * n_features + j] = code as f32;
            }
        }
        Matrix::from_vec(n_samples, n_features, result).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FT-PREP-ORDINAL: matches `sklearn.preprocessing.OrdinalEncoder`.
    #[test]
    fn ordinal_encoder_matches_sklearn() {
        // col0 {10,20,30} -> 0,1,2 ; col1 {0,1,2} -> 0,1,2
        let x = Matrix::from_vec(4, 2, vec![10.0, 1.0, 30.0, 0.0, 10.0, 2.0, 20.0, 1.0])
            .expect("valid");
        let mut enc = OrdinalEncoder::new();
        enc.fit(&x).expect("fit");
        let out = enc.transform(&x).expect("transform");
        let expect = [[0, 1], [2, 0], [0, 2], [1, 1]];
        for (i, row) in expect.iter().enumerate() {
            for (j, e) in row.iter().enumerate() {
                assert!(
                    (out.get(i, j) - *e as f32).abs() < 1e-6,
                    "ordinal[{i}][{j}]"
                );
            }
        }
    }
}
