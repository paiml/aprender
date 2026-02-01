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
    #[must_use]
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
/// Creates the appropriate model based on `model_type` and evaluates it using
/// cross-validation.
///
/// # Arguments
///
/// * `model_type` - Type of model: "ridge", "lasso", or "`elastic_net`"
/// * `alpha` - Alpha value to evaluate
/// * `x` - Training data
/// * `y` - Target values
/// * `cv` - Cross-validation splitter
/// * `l1_ratio` - L1 ratio for `ElasticNet`
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
/// * `model_type` - Type of model: "ridge", "lasso", or "`elastic_net`"
/// * `alphas` - Vector of alpha values to try
/// * `x` - Feature matrix
/// * `y` - Target vector
/// * `cv` - Cross-validation splitter
/// * `l1_ratio` - Optional `l1_ratio` for `ElasticNet` (ignored for Ridge/Lasso)
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
/// Tuple of (`x_train`, `x_test`, `y_train`, `y_test`)
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
/// Validates inputs for `train_test_split`.
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
mod tests;
