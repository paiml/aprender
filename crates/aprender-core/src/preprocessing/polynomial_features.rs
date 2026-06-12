//! `PolynomialFeatures` — generate polynomial and interaction features up to a
//! given degree (Pillar 1 — beat scikit-learn). Mirrors
//! `sklearn.preprocessing.PolynomialFeatures`: columns are all monomials of the
//! inputs from degree 0 (bias) through `degree`, ordered by total degree then by
//! `combinations_with_replacement` of feature indices — e.g. for `[a, b]` and
//! `degree=2` with bias: `[1, a, b, a², a·b, b²]`.

use crate::error::Result;
use crate::primitives::Matrix;
use crate::traits::Transformer;

/// Expands features into polynomial + interaction terms up to `degree`.
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    degree: usize,
    include_bias: bool,
    combos: Vec<Vec<usize>>,
    n_input_features: usize,
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new(2)
    }
}

/// All non-decreasing length-`k` sequences over `0..n` (combinations with
/// replacement), in lexicographic order.
fn combinations_with_replacement(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if n == 0 {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut combo = vec![0usize; k];
    loop {
        result.push(combo.clone());
        let mut i = k;
        loop {
            if i == 0 {
                return result;
            }
            i -= 1;
            if combo[i] < n - 1 {
                let v = combo[i] + 1;
                for slot in combo.iter_mut().skip(i) {
                    *slot = v;
                }
                break;
            }
        }
    }
}

impl PolynomialFeatures {
    /// Create a `PolynomialFeatures` of the given `degree` (`include_bias = true`).
    #[must_use]
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            include_bias: true,
            combos: Vec::new(),
            n_input_features: 0,
        }
    }

    /// Toggle the bias (degree-0 constant) column.
    #[must_use]
    pub fn with_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Number of output features after fit.
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.combos.len()
    }
}

impl Transformer for PolynomialFeatures {
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (_, n_features) = x.shape();
        let start = usize::from(!self.include_bias);
        let mut combos = Vec::new();
        for deg in start..=self.degree {
            combos.extend(combinations_with_replacement(n_features, deg));
        }
        self.combos = combos;
        self.n_input_features = n_features;
        Ok(())
    }

    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let (n_samples, n_features) = x.shape();
        if n_features != self.n_input_features {
            return Err("Feature dimension mismatch".into());
        }
        let out_width = self.combos.len();
        let mut result = vec![0.0f32; n_samples * out_width];
        for i in 0..n_samples {
            for (k, combo) in self.combos.iter().enumerate() {
                let mut v = 1.0f32;
                for &j in combo {
                    v *= x.get(i, j);
                }
                result[i * out_width + k] = v;
            }
        }
        Matrix::from_vec(n_samples, out_width, result).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FT-PREP-POLY: matches `sklearn.preprocessing.PolynomialFeatures`.
    #[test]
    fn polynomial_features_match_sklearn() {
        // [[2,3],[1,4]] degree 2 + bias -> [1, a, b, a^2, ab, b^2]
        let x = Matrix::from_vec(2, 2, vec![2.0, 3.0, 1.0, 4.0]).expect("valid");
        let mut pf = PolynomialFeatures::new(2);
        pf.fit(&x).expect("fit");
        assert_eq!(pf.n_output_features(), 6);
        let out = pf.transform(&x).expect("transform");
        let expect = [
            [1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
            [1.0, 1.0, 4.0, 1.0, 4.0, 16.0],
        ];
        for (i, row) in expect.iter().enumerate() {
            for (j, e) in row.iter().enumerate() {
                assert!((out.get(i, j) - e).abs() < 1e-5, "poly[{i}][{j}]");
            }
        }
        // without bias -> drop the leading 1 (5 columns)
        let mut pf2 = PolynomialFeatures::new(2).with_bias(false);
        pf2.fit(&x).expect("fit");
        assert_eq!(pf2.n_output_features(), 5);
    }
}
