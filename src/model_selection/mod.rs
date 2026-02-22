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
    #[must_use]
    pub fn mean(&self) -> f32 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.iter().sum::<f32>() / self.scores.len() as f32
    }

    /// Calculate standard deviation of scores
    #[must_use]
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
        self.scores.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Get maximum score
    pub fn max(&self) -> f32 {
        self.scores
            .iter()
            .copied()
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
/// * `cv` - Cross-validation splitter (e.g., `KFold`)
///
/// # Example
///
/// ```rust
/// use aprender::prelude::*;
/// use aprender::model_selection::{cross_validate, KFold};
///
/// let x = Matrix::from_vec(50, 1, (0..50).map(|i| i as f32).collect()).expect("Matrix creation should succeed with valid dimensions and data");
/// let y = Vector::from_slice(&vec![0.0; 50]);
///
/// let model = LinearRegression::new();
/// let kfold = KFold::new(5);
///
/// let results = cross_validate(&model, &x, &y, &kfold).expect("Cross-validation should succeed with valid model and data");
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
            .map_err(|e| format!("Training failed: {e}"))?;

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
/// let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect()).expect("Matrix creation should succeed with valid dimensions and data");
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
    #[must_use]
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Enable shuffling before splitting into batches.
    #[must_use]
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random state for reproducible shuffling.
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self.shuffle = true; // Shuffle is implied when random_state is set
        self
    }

    /// Generate train/test indices for each fold.
    ///
    /// Returns a vector of (`train_indices`, `test_indices`) tuples.
    #[must_use]
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
                let mut rng = rand::rng();
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

/// Stratified K-Fold cross-validator.
///
/// Provides train/test indices to split data into K consecutive folds while
/// maintaining the percentage of samples for each class in each fold.
///
/// This is useful for classification problems with imbalanced class distributions.
///
/// # Example
///
/// ```rust
/// use aprender::model_selection::StratifiedKFold;
/// use aprender::primitives::Vector;
///
/// // Labels with imbalanced classes
/// let y = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]);
///
/// let skfold = StratifiedKFold::new(3);
/// for (train_idx, test_idx) in skfold.split(&y) {
///     // Each fold maintains approximate class distribution
///     println!("Train: {:?}, Test: {:?}", train_idx, test_idx);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new Stratified K-Fold cross-validator.
    ///
    /// # Arguments
    ///
    /// * `n_splits` - Number of folds. Must be at least 2.
    ///
    /// # Example
    ///
    /// ```rust
    /// use aprender::model_selection::StratifiedKFold;
    ///
    /// let skfold = StratifiedKFold::new(5);
    /// ```
    #[must_use]
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Enable shuffling before splitting into batches.
    ///
    /// # Example
    ///
    /// ```rust
    /// use aprender::model_selection::StratifiedKFold;
    ///
    /// let skfold = StratifiedKFold::new(5).with_shuffle(true);
    /// ```
    #[must_use]
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random state for reproducible shuffling.
    ///
    /// # Example
    ///
    /// ```rust
    /// use aprender::model_selection::StratifiedKFold;
    ///
    /// let skfold = StratifiedKFold::new(5).with_random_state(42);
    /// ```
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self.shuffle = true;
        self
    }

    /// Generate stratified train/test indices for each fold.
    ///
    /// Maintains approximate class distribution in each fold by splitting
    /// each class separately and combining the splits.
    ///
    /// # Arguments
    ///
    /// * `y` - Target labels vector
    ///
    /// # Returns
    ///
    /// Vector of (`train_indices`, `test_indices`) tuples
    ///
    /// # Example
    ///
    /// ```rust
    /// use aprender::model_selection::StratifiedKFold;
    /// use aprender::primitives::Vector;
    ///
    /// let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    /// let skfold = StratifiedKFold::new(2);
    ///
    /// let splits = skfold.split(&y);
    /// assert_eq!(splits.len(), 2);
    /// ```
    #[must_use]
    pub fn split(&self, y: &Vector<f32>) -> Vec<(Vec<usize>, Vec<usize>)> {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use std::collections::HashMap;

        let n_samples = y.len();

        // Group indices by class label
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in y.as_slice().iter().enumerate() {
            class_indices.entry(label as i32).or_default().push(i);
        }

        // Shuffle each class's indices if requested
        if self.shuffle {
            for indices in class_indices.values_mut() {
                if let Some(seed) = self.random_state {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                    indices.shuffle(&mut rng);
                } else {
                    let mut rng = rand::rng();
                    indices.shuffle(&mut rng);
                }
            }
        }

        // Initialize folds
        let mut fold_indices: Vec<Vec<usize>> = vec![Vec::new(); self.n_splits];

        // Distribute each class across folds
        for indices in class_indices.values() {
            let class_size = indices.len();
            let fold_size = class_size / self.n_splits;
            let remainder = class_size % self.n_splits;

            let mut start = 0;
            for (i, fold) in fold_indices.iter_mut().enumerate() {
                let current_size = if i < remainder {
                    fold_size + 1
                } else {
                    fold_size
                };
                let end = start + current_size;

                fold.extend_from_slice(&indices[start..end]);
                start = end;
            }
        }

        // Create train/test splits
        let mut result = Vec::with_capacity(self.n_splits);

        for i in 0..self.n_splits {
            let test_indices = fold_indices[i].clone();

            let mut train_indices = Vec::with_capacity(n_samples - test_indices.len());
            for (j, fold) in fold_indices.iter().enumerate() {
                if i != j {
                    train_indices.extend_from_slice(fold);
                }
            }

            result.push((train_indices, test_indices));
        }

        result
    }
}

/// Grid search result containing best parameters and score.
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    /// Best alpha value found
    pub best_alpha: f32,
    /// Best cross-validation score
    pub best_score: f32,
    /// All alpha values tried
    pub alphas: Vec<f32>,
    /// Corresponding scores for each alpha
    pub scores: Vec<f32>,
}

include!("alpha.rs");
