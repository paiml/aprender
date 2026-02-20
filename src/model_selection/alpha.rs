
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
