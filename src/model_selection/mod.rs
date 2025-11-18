//! Model selection utilities for cross-validation and train/test splitting.
//!
//! This module provides tools for:
//! - Train/test splitting
//! - K-Fold cross-validation
//! - Cross-validation with multiple metrics

use crate::primitives::{Matrix, Vector};
use crate::traits::Estimator;

/// Results from cross-validation.
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Score for each fold
    pub scores: Vec<f32>,
}

impl CrossValidationResult {
    /// Calculate mean score across folds
    pub fn mean(&self) -> f32 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.iter().sum::<f32>() / self.scores.len() as f32
    }

    /// Calculate standard deviation of scores
    pub fn std(&self) -> f32 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self
            .scores
            .iter()
            .map(|&score| (score - mean).powi(2))
            .sum::<f32>()
            / self.scores.len() as f32;
        variance.sqrt()
    }

    /// Get minimum score
    pub fn min(&self) -> f32 {
        self.scores.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Get maximum score
    pub fn max(&self) -> f32 {
        self.scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

/// Run cross-validation on an estimator.
///
/// Automatically trains and evaluates the model on each fold, returning scores.
///
/// # Arguments
///
/// * `estimator` - The model to cross-validate (must be cloneable)
/// * `x` - Feature matrix
/// * `y` - Target vector
/// * `cv` - Cross-validation splitter (e.g., KFold)
///
/// # Example
///
/// ```rust
/// use aprender::prelude::*;
/// use aprender::model_selection::{cross_validate, KFold};
///
/// let x = Matrix::from_vec(50, 1, (0..50).map(|i| i as f32).collect()).unwrap();
/// let y = Vector::from_slice(&vec![0.0; 50]);
///
/// let model = LinearRegression::new();
/// let kfold = KFold::new(5);
///
/// let results = cross_validate(&model, &x, &y, &kfold).unwrap();
/// println!("Mean R²: {:.3} ± {:.3}", results.mean(), results.std());
/// ```
pub fn cross_validate<E>(
    estimator: &E,
    x: &Matrix<f32>,
    y: &Vector<f32>,
    cv: &KFold,
) -> Result<CrossValidationResult, String>
where
    E: Estimator + Clone,
{
    let n_samples = x.shape().0;
    let splits = cv.split(n_samples);

    let mut scores = Vec::with_capacity(splits.len());

    for (train_idx, test_idx) in splits {
        // Extract fold data
        let (x_train, y_train) = extract_samples(x, y, &train_idx);
        let (x_test, y_test) = extract_samples(x, y, &test_idx);

        // Clone and train model
        let mut fold_model = estimator.clone();
        fold_model
            .fit(&x_train, &y_train)
            .map_err(|e| format!("Training failed: {}", e))?;

        // Score on test fold
        let score = fold_model.score(&x_test, &y_test);
        scores.push(score);
    }

    Ok(CrossValidationResult { scores })
}

/// Helper function to extract samples by indices
fn extract_samples(
    x: &Matrix<f32>,
    y: &Vector<f32>,
    indices: &[usize],
) -> (Matrix<f32>, Vector<f32>) {
    let n_features = x.shape().1;
    let mut x_data = Vec::with_capacity(indices.len() * n_features);
    let mut y_data = Vec::with_capacity(indices.len());

    for &idx in indices {
        for j in 0..n_features {
            x_data.push(x.get(idx, j));
        }
        y_data.push(y.as_slice()[idx]);
    }

    let x_subset =
        Matrix::from_vec(indices.len(), n_features, x_data).expect("Failed to create matrix");
    let y_subset = Vector::from_vec(y_data);

    (x_subset, y_subset)
}

/// K-Fold cross-validator.
///
/// Splits data into K consecutive folds. Each fold is used once as test set
/// while the remaining K-1 folds form the training set.
///
/// # Example
///
/// ```rust
/// use aprender::model_selection::KFold;
/// use aprender::primitives::Matrix;
///
/// let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect()).unwrap();
/// let kfold = KFold::new(5);
///
/// for (train_idx, test_idx) in kfold.split(10) {
///     println!("Train: {:?}, Test: {:?}", train_idx, test_idx);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl KFold {
    /// Create a new K-Fold cross-validator.
    ///
    /// # Arguments
    ///
    /// * `n_splits` - Number of folds. Must be at least 2.
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Enable shuffling before splitting into batches.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random state for reproducible shuffling.
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self.shuffle = true; // Shuffle is implied when random_state is set
        self
    }

    /// Generate train/test indices for each fold.
    ///
    /// Returns a vector of (train_indices, test_indices) tuples.
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        // Create indices
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle if requested
        if self.shuffle {
            if let Some(seed) = self.random_state {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            } else {
                let mut rng = rand::thread_rng();
                indices.shuffle(&mut rng);
            }
        }

        // Calculate fold sizes
        let fold_size = n_samples / self.n_splits;
        let remainder = n_samples % self.n_splits;

        let mut result = Vec::with_capacity(self.n_splits);
        let mut start = 0;

        for i in 0..self.n_splits {
            // Distribute remainder across first folds
            let current_fold_size = if i < remainder {
                fold_size + 1
            } else {
                fold_size
            };

            let end = start + current_fold_size;

            // Test indices for this fold
            let test_indices: Vec<usize> = indices[start..end].to_vec();

            // Train indices are all other indices
            let mut train_indices = Vec::with_capacity(n_samples - current_fold_size);
            train_indices.extend_from_slice(&indices[..start]);
            train_indices.extend_from_slice(&indices[end..]);

            result.push((train_indices, test_indices));

            start = end;
        }

        result
    }
}

/// Split arrays into random train and test subsets.
///
/// # Arguments
///
/// * `x` - Feature matrix
/// * `y` - Target vector (labels or values)
/// * `test_size` - Proportion of dataset to include in test split (0.0 to 1.0)
/// * `random_state` - Optional random seed for reproducibility
///
/// # Returns
///
/// Tuple of (x_train, x_test, y_train, y_test)
///
/// # Example
///
/// ```rust
/// use aprender::model_selection::train_test_split;
/// use aprender::primitives::{Matrix, Vector};
///
/// let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect()).unwrap();
/// let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
///
/// let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42)).unwrap();
/// assert_eq!(x_train.shape().0, 8);  // 80% training
/// assert_eq!(x_test.shape().0, 2);   // 20% test
/// ```
#[allow(clippy::type_complexity)]
pub fn train_test_split(
    x: &Matrix<f32>,
    y: &Vector<f32>,
    test_size: f32,
    random_state: Option<u64>,
) -> Result<(Matrix<f32>, Matrix<f32>, Vector<f32>, Vector<f32>), String> {
    // Validate inputs
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(format!(
            "test_size must be between 0 and 1, got {}",
            test_size
        ));
    }

    let (n_samples, n_features) = x.shape();
    if n_samples != y.len() {
        return Err(format!(
            "X and y must have same number of samples, got {} and {}",
            n_samples,
            y.len()
        ));
    }

    // Calculate split sizes
    let n_test = (n_samples as f32 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    if n_test == 0 || n_train == 0 {
        return Err(format!(
            "Split would result in empty train or test set (n_train={}, n_test={})",
            n_train, n_test
        ));
    }

    // Create shuffled indices
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut indices: Vec<usize> = (0..n_samples).collect();

    if let Some(seed) = random_state {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
    } else {
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
    }

    // Split indices
    let train_indices = &indices[..n_train];
    let test_indices = &indices[n_train..];

    // Build train/test matrices and vectors
    let mut x_train_data = Vec::with_capacity(n_train * n_features);
    let mut y_train_data = Vec::with_capacity(n_train);

    for &idx in train_indices {
        for j in 0..n_features {
            x_train_data.push(x.get(idx, j));
        }
        y_train_data.push(y.as_slice()[idx]);
    }

    let mut x_test_data = Vec::with_capacity(n_test * n_features);
    let mut y_test_data = Vec::with_capacity(n_test);

    for &idx in test_indices {
        for j in 0..n_features {
            x_test_data.push(x.get(idx, j));
        }
        y_test_data.push(y.as_slice()[idx]);
    }

    let x_train = Matrix::from_vec(n_train, n_features, x_train_data)
        .map_err(|e| format!("Failed to create train matrix: {}", e))?;
    let x_test = Matrix::from_vec(n_test, n_features, x_test_data)
        .map_err(|e| format!("Failed to create test matrix: {}", e))?;
    let y_train = Vector::from_vec(y_train_data);
    let y_test = Vector::from_vec(y_test_data);

    Ok((x_train, x_test, y_train, y_test))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split_basic() {
        // Create simple dataset: 10 samples, 2 features
        let x = Matrix::from_vec(
            10,
            2,
            vec![
                1.0, 2.0, // sample 0
                3.0, 4.0, // sample 1
                5.0, 6.0, // sample 2
                7.0, 8.0, // sample 3
                9.0, 10.0, // sample 4
                11.0, 12.0, // sample 5
                13.0, 14.0, // sample 6
                15.0, 16.0, // sample 7
                17.0, 18.0, // sample 8
                19.0, 20.0, // sample 9
            ],
        )
        .unwrap();
        let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        // Split 80/20
        let (x_train, x_test, y_train, y_test) =
            train_test_split(&x, &y, 0.2, Some(42)).expect("Split should succeed");

        // Verify shapes
        assert_eq!(x_train.shape().0, 8, "Training set should have 8 samples");
        assert_eq!(x_test.shape().0, 2, "Test set should have 2 samples");
        assert_eq!(x_train.shape().1, 2, "Training features should be 2");
        assert_eq!(x_test.shape().1, 2, "Test features should be 2");
        assert_eq!(y_train.len(), 8, "Training labels should have 8 samples");
        assert_eq!(y_test.len(), 2, "Test labels should have 2 samples");
    }

    #[test]
    fn test_train_test_split_reproducibility() {
        let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect()).unwrap();
        let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        // Same random state should give same split
        let (x_train1, x_test1, y_train1, y_test1) =
            train_test_split(&x, &y, 0.2, Some(42)).unwrap();
        let (x_train2, x_test2, y_train2, y_test2) =
            train_test_split(&x, &y, 0.2, Some(42)).unwrap();

        // Verify reproducibility
        assert_eq!(x_train1.as_slice(), x_train2.as_slice());
        assert_eq!(x_test1.as_slice(), x_test2.as_slice());
        assert_eq!(y_train1.as_slice(), y_train2.as_slice());
        assert_eq!(y_test1.as_slice(), y_test2.as_slice());
    }

    #[test]
    fn test_train_test_split_different_seeds() {
        let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect()).unwrap();
        let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        // Different random states should give different splits
        let (_, _, y_train1, _) = train_test_split(&x, &y, 0.2, Some(42)).unwrap();
        let (_, _, y_train2, _) = train_test_split(&x, &y, 0.2, Some(123)).unwrap();

        // Very likely to be different with different seeds
        assert_ne!(y_train1.as_slice(), y_train2.as_slice());
    }

    #[test]
    fn test_train_test_split_different_sizes() {
        let x = Matrix::from_vec(100, 3, (0..300).map(|i| i as f32).collect()).unwrap();
        let y = Vector::from_slice(&vec![0.0; 100]);

        // Test 70/30 split
        let (x_train, x_test, _, _) = train_test_split(&x, &y, 0.3, Some(42)).unwrap();
        assert_eq!(x_train.shape().0, 70);
        assert_eq!(x_test.shape().0, 30);

        // Test 50/50 split
        let (x_train, x_test, _, _) = train_test_split(&x, &y, 0.5, Some(42)).unwrap();
        assert_eq!(x_train.shape().0, 50);
        assert_eq!(x_test.shape().0, 50);
    }

    #[test]
    fn test_kfold_basic() {
        let kfold = KFold::new(5);
        let splits = kfold.split(10);

        // Should have 5 folds
        assert_eq!(splits.len(), 5, "Should have 5 folds");

        // Each fold should have 8 train and 2 test samples
        for (i, (train_idx, test_idx)) in splits.iter().enumerate() {
            assert_eq!(
                train_idx.len(),
                8,
                "Fold {} should have 8 training samples",
                i
            );
            assert_eq!(test_idx.len(), 2, "Fold {} should have 2 test samples", i);

            // Verify no overlap between train and test
            for &test_i in test_idx {
                assert!(
                    !train_idx.contains(&test_i),
                    "Test index {} should not be in training set for fold {}",
                    test_i,
                    i
                );
            }
        }

        // All indices should be used exactly once as test
        let mut all_test_indices: Vec<usize> =
            splits.iter().flat_map(|(_, test)| test).copied().collect();
        all_test_indices.sort();
        assert_eq!(all_test_indices, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_kfold_no_shuffle() {
        let kfold = KFold::new(3);
        let splits = kfold.split(9);

        assert_eq!(splits.len(), 3);

        // Without shuffle, folds should be consecutive
        assert_eq!(splits[0].1, vec![0, 1, 2]);
        assert_eq!(splits[1].1, vec![3, 4, 5]);
        assert_eq!(splits[2].1, vec![6, 7, 8]);
    }

    #[test]
    fn test_kfold_shuffle_reproducible() {
        let kfold1 = KFold::new(5).with_random_state(42);
        let kfold2 = KFold::new(5).with_random_state(42);

        let splits1 = kfold1.split(20);
        let splits2 = kfold2.split(20);

        // Same random state should give same splits
        assert_eq!(splits1, splits2);
    }

    #[test]
    fn test_kfold_shuffle_different_states() {
        let kfold1 = KFold::new(5).with_random_state(42);
        let kfold2 = KFold::new(5).with_random_state(123);

        let splits1 = kfold1.split(20);
        let splits2 = kfold2.split(20);

        // Different random states should give different splits
        assert_ne!(splits1, splits2);
    }

    #[test]
    fn test_kfold_uneven_split() {
        let kfold = KFold::new(3);
        let splits = kfold.split(10);

        assert_eq!(splits.len(), 3);

        // With 10 samples and 3 folds: folds should be 3, 3, 4 samples (or similar)
        let test_sizes: Vec<usize> = splits.iter().map(|(_, test)| test.len()).collect();
        let total_test: usize = test_sizes.iter().sum();
        assert_eq!(
            total_test, 10,
            "All samples should be used as test exactly once"
        );
    }

    #[test]
    fn test_cross_validate_basic() {
        use crate::linear_model::LinearRegression;

        // Create simple dataset: y = 2x
        let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x).collect();

        let x = Matrix::from_vec(50, 1, x_data).unwrap();
        let y = Vector::from_vec(y_data);

        let model = LinearRegression::new();
        let kfold = KFold::new(5).with_random_state(42);

        let result =
            cross_validate(&model, &x, &y, &kfold).expect("Cross-validation should succeed");

        // Should have 5 scores (one per fold)
        assert_eq!(result.scores.len(), 5, "Should have 5 fold scores");

        // All scores should be very high (perfect linear relationship)
        for &score in &result.scores {
            assert!(score > 0.99, "Score should be > 0.99, got {}", score);
        }

        // Mean should be close to 1.0
        assert!(result.mean() > 0.99, "Mean R² should be > 0.99");

        // Std should be very low (stable across folds)
        assert!(result.std() < 0.01, "Std should be < 0.01");
    }

    #[test]
    fn test_cross_validate_result_stats() {
        let result = CrossValidationResult {
            scores: vec![0.95, 0.96, 0.94, 0.97, 0.93],
        };

        // Test mean
        let mean = result.mean();
        assert!((mean - 0.95).abs() < 0.001, "Mean should be ~0.95");

        // Test min/max
        assert_eq!(result.min(), 0.93);
        assert_eq!(result.max(), 0.97);

        // Test std
        let std = result.std();
        assert!(std > 0.0, "Std should be > 0");
        assert!(std < 0.02, "Std should be < 0.02");
    }

    #[test]
    fn test_cross_validate_reproducible() {
        use crate::linear_model::LinearRegression;

        let x_data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 3.0 * x + 1.0).collect();

        let x = Matrix::from_vec(30, 1, x_data).unwrap();
        let y = Vector::from_vec(y_data);

        let model = LinearRegression::new();

        // Same random state should give same results
        let kfold1 = KFold::new(5).with_random_state(42);
        let result1 = cross_validate(&model, &x, &y, &kfold1).unwrap();

        let kfold2 = KFold::new(5).with_random_state(42);
        let result2 = cross_validate(&model, &x, &y, &kfold2).unwrap();

        assert_eq!(
            result1.scores, result2.scores,
            "Results should be reproducible"
        );
    }
}
