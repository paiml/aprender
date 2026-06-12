//! `LabelEncoder` — encode categorical labels as consecutive integers 0..n_classes
//! (Pillar 1 — beat scikit-learn). Mirrors `sklearn.preprocessing.LabelEncoder`:
//! classes are the sorted unique values, and each label maps to its index.
//!
//! Generic over any `Ord + Clone` label type (`&str`, `i64`, `String`, …), so it
//! handles string or integer categories without a separate API.

/// Encodes labels to `0..n_classes` by sorted-unique order, with inverse mapping.
#[derive(Debug, Clone)]
pub struct LabelEncoder<T> {
    /// Sorted unique classes; the index of a class is its encoded value.
    classes: Vec<T>,
}

impl<T> Default for LabelEncoder<T> {
    fn default() -> Self {
        Self {
            classes: Vec::new(),
        }
    }
}

impl<T: Ord + Clone> LabelEncoder<T> {
    /// Create a new (unfitted) `LabelEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the encoder: record the sorted unique classes. Returns `&mut self`
    /// for chaining.
    pub fn fit(&mut self, y: &[T]) -> &mut Self {
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        self.classes = classes;
        self
    }

    /// Transform labels to their integer codes. Labels unseen during `fit`
    /// encode to `n_classes` (an out-of-range sentinel), mirroring nothing in
    /// sklearn (which raises) but keeping the API total.
    #[must_use]
    pub fn transform(&self, y: &[T]) -> Vec<usize> {
        y.iter()
            .map(|v| self.classes.binary_search(v).unwrap_or(self.classes.len()))
            .collect()
    }

    /// Fit then transform in one call (the common path).
    pub fn fit_transform(&mut self, y: &[T]) -> Vec<usize> {
        self.fit(y);
        self.transform(y)
    }

    /// Invert codes back to labels. Out-of-range codes are skipped.
    #[must_use]
    pub fn inverse_transform(&self, codes: &[usize]) -> Vec<T> {
        codes
            .iter()
            .filter_map(|&c| self.classes.get(c).cloned())
            .collect()
    }

    /// The fitted classes (sorted unique), indexed by encoded value.
    #[must_use]
    pub fn classes(&self) -> &[T] {
        &self.classes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FT-PREP-LABELENC: matches `sklearn.preprocessing.LabelEncoder`.
    #[test]
    fn label_encoder_matches_sklearn() {
        // strings: sorted-unique ["A","B","C"] -> A=0,B=1,C=2
        let mut le = LabelEncoder::new();
        let codes = le.fit_transform(&["A", "B", "C", "A"]);
        assert_eq!(codes, vec![0, 1, 2, 0]);
        assert_eq!(le.classes(), &["A", "B", "C"]);
        assert_eq!(le.inverse_transform(&[2, 0, 1]), vec!["C", "A", "B"]);

        // integers: sorted-unique [5,10,20] -> 5=0,10=1,20=2
        let mut li = LabelEncoder::new();
        assert_eq!(li.fit_transform(&[10i64, 5, 10, 20]), vec![1, 0, 1, 2]);
        assert_eq!(li.classes(), &[5, 10, 20]);
    }
}
