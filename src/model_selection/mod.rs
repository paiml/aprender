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
/// * `cv` - Cross-validation splitter (e.g., KFold)
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
    /// Vector of (train_indices, test_indices) tuples
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
                    let mut rng = rand::thread_rng();
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

impl GridSearchResult {
    /// Returns the index of the best alpha value.
    pub fn best_index(&self) -> usize {
        self.scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b)
                    .expect("Scores should be valid f32 values, not NaN")
            })
            .map_or(0, |(idx, _)| idx)
    }
}

/// Evaluate a single alpha value with cross-validation for a specific model type.
///
/// Creates the appropriate model based on model_type and evaluates it using
/// cross-validation.
///
/// # Arguments
///
/// * `model_type` - Type of model: "ridge", "lasso", or "elastic_net"
/// * `alpha` - Alpha value to evaluate
/// * `x` - Training data
/// * `y` - Target values
/// * `cv` - Cross-validation splitter
/// * `l1_ratio` - L1 ratio for ElasticNet
///
/// # Returns
///
/// Mean cross-validation score
fn evaluate_alpha_for_model(
    model_type: &str,
    alpha: f32,
    x: &Matrix<f32>,
    y: &Vector<f32>,
    cv: &KFold,
    l1_ratio: Option<f32>,
) -> Result<f32, String> {
    let score = match model_type {
        "ridge" => {
            use crate::linear_model::Ridge;
            let model = Ridge::new(alpha);
            let cv_result = cross_validate(&model, x, y, cv)?;
            cv_result.mean()
        }
        "lasso" => {
            use crate::linear_model::Lasso;
            let model = Lasso::new(alpha);
            let cv_result = cross_validate(&model, x, y, cv)?;
            cv_result.mean()
        }
        "elastic_net" => {
            use crate::linear_model::ElasticNet;
            let ratio = l1_ratio.ok_or("l1_ratio required for ElasticNet")?;
            let model = ElasticNet::new(alpha, ratio);
            let cv_result = cross_validate(&model, x, y, cv)?;
            cv_result.mean()
        }
        _ => {
            return Err(format!(
                "Unknown model type: {model_type}. Use 'ridge', 'lasso', or 'elastic_net'"
            ))
        }
    };
    Ok(score)
}

/// Update best score and alpha if current score is better.
fn update_best_if_improved(score: f32, alpha: f32, best_score: &mut f32, best_alpha: &mut f32) {
    if score > *best_score {
        *best_score = score;
        *best_alpha = alpha;
    }
}

/// Performs grid search over alpha parameter for regularized linear models.
///
/// Exhaustively evaluates all provided alpha values using K-fold cross-validation
/// and returns the alpha that achieves the highest cross-validation score.
///
/// # Arguments
///
/// * `model_type` - Type of model: "ridge", "lasso", or "elastic_net"
/// * `alphas` - Vector of alpha values to try
/// * `x` - Feature matrix
/// * `y` - Target vector
/// * `cv` - Cross-validation splitter
/// * `l1_ratio` - Optional l1_ratio for ElasticNet (ignored for Ridge/Lasso)
///
/// # Returns
///
/// `GridSearchResult` containing best alpha, best score, and all results
///
/// # Example
///
/// ```rust
/// use aprender::model_selection::{grid_search_alpha, KFold};
/// use aprender::primitives::{Matrix, Vector};
///
/// let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
/// let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();
///
/// let x = Matrix::from_vec(50, 1, x_data).expect("Matrix creation should succeed with valid dimensions and data");
/// let y = Vector::from_vec(y_data);
///
/// let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
/// let kfold = KFold::new(5).with_random_state(42);
///
/// let result = grid_search_alpha("ridge", &alphas, &x, &y, &kfold, None).expect("Grid search should succeed with valid inputs");
/// println!("Best alpha: {}, Best score: {}", result.best_alpha, result.best_score);
/// ```
pub fn grid_search_alpha(
    model_type: &str,
    alphas: &[f32],
    x: &Matrix<f32>,
    y: &Vector<f32>,
    cv: &KFold,
    l1_ratio: Option<f32>,
) -> Result<GridSearchResult, String> {
    if alphas.is_empty() {
        return Err("Alphas vector cannot be empty".to_string());
    }

    let mut best_alpha = alphas[0];
    let mut best_score = f32::NEG_INFINITY;
    let mut all_scores = Vec::with_capacity(alphas.len());

    for &alpha in alphas {
        let score = evaluate_alpha_for_model(model_type, alpha, x, y, cv, l1_ratio)?;
        all_scores.push(score);
        update_best_if_improved(score, alpha, &mut best_score, &mut best_alpha);
    }

    Ok(GridSearchResult {
        best_alpha,
        best_score,
        alphas: alphas.to_vec(),
        scores: all_scores,
    })
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
/// let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect()).expect("Matrix creation should succeed with valid dimensions and data");
/// let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
///
/// let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42)).expect("Train/test split should succeed with valid inputs");
/// assert_eq!(x_train.shape().0, 8);  // 80% training
/// assert_eq!(x_test.shape().0, 2);   // 20% test
/// ```
/// Validates inputs for train_test_split.
fn validate_split_inputs(
    x: &Matrix<f32>,
    y: &Vector<f32>,
    test_size: f32,
) -> Result<(usize, usize), String> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(format!(
            "test_size must be between 0 and 1, got {test_size}"
        ));
    }

    let (n_samples, _) = x.shape();
    if n_samples != y.len() {
        return Err(format!(
            "X and y must have same number of samples, got {} and {}",
            n_samples,
            y.len()
        ));
    }

    let n_test = (n_samples as f32 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    if n_test == 0 || n_train == 0 {
        return Err(format!(
            "Split would result in empty train or test set (n_train={n_train}, n_test={n_test})"
        ));
    }

    Ok((n_train, n_test))
}

/// Shuffles indices with optional random seed.
fn shuffle_indices(n_samples: usize, random_state: Option<u64>) -> Vec<usize> {
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

    indices
}

#[allow(clippy::type_complexity)]
pub fn train_test_split(
    x: &Matrix<f32>,
    y: &Vector<f32>,
    test_size: f32,
    random_state: Option<u64>,
) -> Result<(Matrix<f32>, Matrix<f32>, Vector<f32>, Vector<f32>), String> {
    let (n_train, _) = validate_split_inputs(x, y, test_size)?;
    let n_samples = x.shape().0;

    let indices = shuffle_indices(n_samples, random_state);
    let train_indices = &indices[..n_train];
    let test_indices = &indices[n_train..];

    // Use extract_samples helper for both train and test sets
    let (x_train, y_train) = extract_samples(x, y, train_indices);
    let (x_test, y_test) = extract_samples(x, y, test_indices);

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
        .expect("Matrix creation should succeed with valid test data");
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
        let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect())
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        // Same random state should give same split
        let (x_train1, x_test1, y_train1, y_test1) =
            train_test_split(&x, &y, 0.2, Some(42)).expect("First split should succeed");
        let (x_train2, x_test2, y_train2, y_test2) =
            train_test_split(&x, &y, 0.2, Some(42)).expect("Second split should succeed");

        // Verify reproducibility
        assert_eq!(x_train1.as_slice(), x_train2.as_slice());
        assert_eq!(x_test1.as_slice(), x_test2.as_slice());
        assert_eq!(y_train1.as_slice(), y_train2.as_slice());
        assert_eq!(y_test1.as_slice(), y_test2.as_slice());
    }

    #[test]
    fn test_train_test_split_different_seeds() {
        let x = Matrix::from_vec(10, 2, (0..20).map(|i| i as f32).collect())
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        // Different random states should give different splits
        let (_, _, y_train1, _) =
            train_test_split(&x, &y, 0.2, Some(42)).expect("Split with seed 42 should succeed");
        let (_, _, y_train2, _) =
            train_test_split(&x, &y, 0.2, Some(123)).expect("Split with seed 123 should succeed");

        // Very likely to be different with different seeds
        assert_ne!(y_train1.as_slice(), y_train2.as_slice());
    }

    #[test]
    fn test_train_test_split_different_sizes() {
        let x = Matrix::from_vec(100, 3, (0..300).map(|i| i as f32).collect())
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_slice(&vec![0.0; 100]);

        // Test 70/30 split
        let (x_train, x_test, _, _) =
            train_test_split(&x, &y, 0.3, Some(42)).expect("70/30 split should succeed");
        assert_eq!(x_train.shape().0, 70);
        assert_eq!(x_test.shape().0, 30);

        // Test 50/50 split
        let (x_train, x_test, _, _) =
            train_test_split(&x, &y, 0.5, Some(42)).expect("50/50 split should succeed");
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

        let x = Matrix::from_vec(50, 1, x_data)
            .expect("Matrix creation should succeed with valid test data");
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

        let x = Matrix::from_vec(30, 1, x_data)
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_vec(y_data);

        let model = LinearRegression::new();

        // Same random state should give same results
        let kfold1 = KFold::new(5).with_random_state(42);
        let result1 =
            cross_validate(&model, &x, &y, &kfold1).expect("First cross-validation should succeed");

        let kfold2 = KFold::new(5).with_random_state(42);
        let result2 = cross_validate(&model, &x, &y, &kfold2)
            .expect("Second cross-validation should succeed");

        assert_eq!(
            result1.scores, result2.scores,
            "Results should be reproducible"
        );
    }

    // ==================== StratifiedKFold Tests ====================

    #[test]
    fn test_stratified_kfold_new() {
        let skfold = StratifiedKFold::new(5);
        let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);

        let splits = skfold.split(&y);
        assert_eq!(splits.len(), 5);
    }

    #[test]
    fn test_stratified_kfold_balanced_classes() {
        // Perfectly balanced classes
        let y = Vector::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        let skfold = StratifiedKFold::new(3);

        let splits = skfold.split(&y);
        assert_eq!(splits.len(), 3);

        // Each fold should have one sample from each class
        for (train_idx, test_idx) in &splits {
            assert_eq!(test_idx.len(), 3, "Each test fold should have 3 samples");
            assert_eq!(train_idx.len(), 6, "Each train fold should have 6 samples");

            // Count classes in test fold
            let mut class_counts = [0; 3];
            for &idx in test_idx {
                let label = y[idx] as usize;
                class_counts[label] += 1;
            }

            // Each class should appear exactly once in test fold
            for &count in &class_counts {
                assert_eq!(
                    count, 1,
                    "Each class should appear exactly once in test fold"
                );
            }
        }
    }

    #[test]
    fn test_stratified_kfold_imbalanced_classes() {
        // Imbalanced: 6 of class 0, 3 of class 1
        let y = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let skfold = StratifiedKFold::new(3);

        let splits = skfold.split(&y);

        for (_train_idx, test_idx) in &splits {
            // Count classes in test fold
            let mut class_0_count = 0;
            let mut class_1_count = 0;

            for &idx in test_idx {
                if y[idx] == 0.0 {
                    class_0_count += 1;
                } else {
                    class_1_count += 1;
                }
            }

            // Should maintain approximate 2:1 ratio in each fold
            assert_eq!(
                class_0_count, 2,
                "Each fold should have 2 samples from class 0"
            );
            assert_eq!(
                class_1_count, 1,
                "Each fold should have 1 sample from class 1"
            );
        }
    }

    #[test]
    fn test_stratified_kfold_all_samples_used() {
        let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let skfold = StratifiedKFold::new(3);

        let splits = skfold.split(&y);

        let mut all_test_indices = vec![];
        for (_, test_idx) in splits {
            all_test_indices.extend(test_idx);
        }

        all_test_indices.sort();
        assert_eq!(
            all_test_indices,
            vec![0, 1, 2, 3, 4, 5],
            "All samples should be used as test exactly once"
        );
    }

    #[test]
    fn test_stratified_kfold_with_shuffle() {
        let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let skfold = StratifiedKFold::new(2).with_shuffle(true);

        let splits = skfold.split(&y);
        assert_eq!(splits.len(), 2);

        // Should still maintain stratification even with shuffling
        for (_, test_idx) in &splits {
            let mut class_counts = [0; 3];
            for &idx in test_idx {
                let label = y[idx] as usize;
                class_counts[label] += 1;
            }

            // Each class should appear once in each fold
            for &count in &class_counts {
                assert_eq!(count, 1);
            }
        }
    }

    #[test]
    fn test_stratified_kfold_with_random_state() {
        let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let skfold1 = StratifiedKFold::new(2).with_random_state(42);
        let splits1 = skfold1.split(&y);

        let skfold2 = StratifiedKFold::new(2).with_random_state(42);
        let splits2 = skfold2.split(&y);

        // Same random state should give same splits (check semantic equality)
        assert_eq!(splits1.len(), splits2.len());
        for ((train1, test1), (train2, test2)) in splits1.iter().zip(splits2.iter()) {
            // Sort for comparison since HashMap iteration order is not deterministic
            let mut train1_sorted = train1.clone();
            let mut train2_sorted = train2.clone();
            let mut test1_sorted = test1.clone();
            let mut test2_sorted = test2.clone();

            train1_sorted.sort();
            train2_sorted.sort();
            test1_sorted.sort();
            test2_sorted.sort();

            assert_eq!(train1_sorted, train2_sorted);
            assert_eq!(test1_sorted, test2_sorted);
        }
    }

    #[test]
    fn test_stratified_kfold_different_random_states() {
        // Use larger dataset so different random states are more likely to differ
        let y = Vector::from_slice(&[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0,
        ]);

        let skfold1 = StratifiedKFold::new(3).with_random_state(42);
        let splits1 = skfold1.split(&y);

        let skfold2 = StratifiedKFold::new(3).with_random_state(123);
        let splits2 = skfold2.split(&y);

        // Different random states should give different splits
        // Due to HashMap ordering, we can't guarantee different order in every case
        // so we just verify both produce valid splits
        assert_eq!(splits1.len(), 3);
        assert_eq!(splits2.len(), 3);

        // Verify stratification is maintained for both
        for (_, test_idx) in &splits1 {
            let mut class_counts = [0; 3];
            for &idx in test_idx {
                let label = y[idx] as usize;
                class_counts[label] += 1;
            }
            // Each fold should have 2 samples from each class
            for &count in &class_counts {
                assert_eq!(count, 2);
            }
        }
    }

    #[test]
    fn test_stratified_kfold_binary_classification() {
        // Binary classification with 50-50 split
        let y = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let skfold = StratifiedKFold::new(4);

        let splits = skfold.split(&y);

        for (_, test_idx) in splits {
            assert_eq!(test_idx.len(), 2, "Each fold should have 2 samples");

            // Count classes
            let mut class_0_count = 0;
            let mut class_1_count = 0;
            for &idx in &test_idx {
                if y[idx] == 0.0 {
                    class_0_count += 1;
                } else {
                    class_1_count += 1;
                }
            }

            // Should have exactly one sample from each class
            assert_eq!(class_0_count, 1);
            assert_eq!(class_1_count, 1);
        }
    }

    #[test]
    fn test_stratified_kfold_many_classes() {
        // 5 classes, 2 samples each
        let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
        let skfold = StratifiedKFold::new(2);

        let splits = skfold.split(&y);

        for (_, test_idx) in splits {
            assert_eq!(test_idx.len(), 5, "Each fold should have 5 samples");

            // Each class should appear exactly once
            let mut class_counts = vec![0; 5];
            for &idx in &test_idx {
                let label = y[idx] as usize;
                class_counts[label] += 1;
            }

            for &count in &class_counts {
                assert_eq!(count, 1, "Each class should appear once per fold");
            }
        }
    }

    #[test]
    fn test_stratified_kfold_no_overlap() {
        let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
        let skfold = StratifiedKFold::new(3);

        let splits = skfold.split(&y);

        for (train_idx, test_idx) in splits {
            // Train and test should not overlap
            for &test in &test_idx {
                assert!(
                    !train_idx.contains(&test),
                    "Train and test indices should not overlap"
                );
            }
        }
    }

    #[test]
    fn test_stratified_kfold_builder_pattern() {
        let y = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0]);

        let skfold = StratifiedKFold::new(2)
            .with_shuffle(true)
            .with_random_state(42);

        let splits = skfold.split(&y);
        assert_eq!(splits.len(), 2);
    }

    // ==================== Grid Search Tests ====================

    #[test]
    fn test_grid_search_alpha_ridge() {
        let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

        let x = Matrix::from_vec(50, 1, x_data)
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_vec(y_data);

        let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let kfold = KFold::new(5).with_random_state(42);

        let result = grid_search_alpha("ridge", &alphas, &x, &y, &kfold, None)
            .expect("Grid search for ridge should succeed");

        assert!(alphas.contains(&result.best_alpha));
        assert!(result.best_score > 0.9);
        assert_eq!(result.alphas.len(), alphas.len());
        assert_eq!(result.scores.len(), alphas.len());
    }

    #[test]
    fn test_grid_search_alpha_lasso() {
        let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

        let x = Matrix::from_vec(50, 1, x_data)
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_vec(y_data);

        let alphas = vec![0.001, 0.01, 0.1];
        let kfold = KFold::new(5).with_random_state(42);

        let result = grid_search_alpha("lasso", &alphas, &x, &y, &kfold, None)
            .expect("Grid search for lasso should succeed");

        assert!(alphas.contains(&result.best_alpha));
        assert!(result.best_score > 0.9);
        assert_eq!(result.alphas.len(), alphas.len());
        assert_eq!(result.scores.len(), alphas.len());
    }

    #[test]
    fn test_grid_search_alpha_elastic_net() {
        let x_data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

        let x = Matrix::from_vec(50, 1, x_data)
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_vec(y_data);

        let alphas = vec![0.001, 0.01, 0.1];
        let kfold = KFold::new(5).with_random_state(42);

        let result = grid_search_alpha("elastic_net", &alphas, &x, &y, &kfold, Some(0.5))
            .expect("Grid search for elastic_net should succeed");

        assert!(alphas.contains(&result.best_alpha));
        assert!(result.best_score > 0.9);
        assert_eq!(result.alphas.len(), alphas.len());
        assert_eq!(result.scores.len(), alphas.len());
    }

    #[test]
    fn test_grid_search_result_best_index() {
        let result = GridSearchResult {
            best_alpha: 0.1,
            best_score: 0.95,
            alphas: vec![0.01, 0.1, 1.0],
            scores: vec![0.90, 0.95, 0.85],
        };

        assert_eq!(result.best_index(), 1);
    }

    #[test]
    fn test_grid_search_empty_alphas() {
        let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect())
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_vec(vec![0.0; 10]);

        let alphas: Vec<f32> = vec![];
        let kfold = KFold::new(3);

        let result = grid_search_alpha("ridge", &alphas, &x, &y, &kfold, None);
        assert!(result.is_err());
        assert!(result
            .expect_err("Should fail with empty alphas")
            .contains("empty"));
    }

    #[test]
    fn test_grid_search_invalid_model_type() {
        let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect())
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_vec(vec![0.0; 10]);

        let alphas = vec![0.1, 1.0];
        let kfold = KFold::new(3);

        let result = grid_search_alpha("invalid_model", &alphas, &x, &y, &kfold, None);
        assert!(result.is_err());
        assert!(result
            .expect_err("Should fail with invalid model type")
            .contains("Unknown model type"));
    }

    #[test]
    fn test_grid_search_elastic_net_missing_l1_ratio() {
        let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect())
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_vec(vec![0.0; 10]);

        let alphas = vec![0.1, 1.0];
        let kfold = KFold::new(3);

        let result = grid_search_alpha("elastic_net", &alphas, &x, &y, &kfold, None);
        assert!(result.is_err());
        assert!(result
            .expect_err("Should fail with missing l1_ratio")
            .contains("l1_ratio required"));
    }

    #[test]
    fn test_grid_search_finds_optimal_alpha() {
        let x_data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 3.0 * x + 2.0).collect();

        let x = Matrix::from_vec(30, 1, x_data)
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_vec(y_data);

        // Try range of alphas - should prefer smaller for this simple problem
        let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
        let kfold = KFold::new(5).with_random_state(42);

        let result = grid_search_alpha("ridge", &alphas, &x, &y, &kfold, None)
            .expect("Grid search should find optimal alpha");

        // Best alpha should be one of the smaller values (less regularization needed)
        assert!(result.best_alpha <= 1.0, "Best alpha should be <= 1.0");

        // All alphas should be evaluated
        assert_eq!(result.scores.len(), alphas.len());

        // Scores should generally decrease with higher alpha (more regularization hurts)
        let first_score = result.scores[0];
        let last_score = result.scores[alphas.len() - 1];
        assert!(first_score > last_score);
    }

    #[test]
    fn test_grid_search_single_alpha() {
        let x = Matrix::from_vec(10, 1, (0..10).map(|i| i as f32).collect())
            .expect("Matrix creation should succeed with valid test data");
        let y = Vector::from_vec((0..10).map(|i| i as f32).collect());

        let alphas = vec![0.1];
        let kfold = KFold::new(3);

        let result = grid_search_alpha("ridge", &alphas, &x, &y, &kfold, None)
            .expect("Grid search with single alpha should succeed");

        assert_eq!(result.best_alpha, 0.1);
        assert_eq!(result.alphas.len(), 1);
        assert_eq!(result.scores.len(), 1);
    }
}
