
impl KNearestNeighbors {
    /// Creates a new K-Nearest Neighbors classifier.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of neighbors to use for voting
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::KNearestNeighbors;
    ///
    /// let knn = KNearestNeighbors::new(5);
    /// ```
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            metric: DistanceMetric::Euclidean,
            weights: false,
            x_train: None,
            y_train: None,
        }
    }

    /// Sets the distance metric.
    #[must_use]
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Enables weighted voting (inverse distance weighting).
    #[must_use]
    pub fn with_weights(mut self, weights: bool) -> Self {
        self.weights = weights;
        self
    }

    /// Fits the model by storing the training data.
    ///
    /// kNN is a lazy learner - it simply stores the training data
    /// and defers computation until prediction time.
    ///
    /// # Errors
    ///
    /// Returns error if data dimensions are invalid.
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, _n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        if y.len() != n_samples {
            return Err("Number of samples in X and y must match".into());
        }

        if self.k > n_samples {
            return Err("k cannot be larger than number of training samples".into());
        }

        // Store training data
        self.x_train = Some(x.clone());
        self.y_train = Some(y.to_vec());

        Ok(())
    }

    /// Predicts class labels for samples.
    ///
    /// For each test sample, finds the k nearest training samples
    /// and returns the majority class.
    ///
    /// # Errors
    ///
    /// Returns error if model is not fitted or dimensions mismatch.
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let x_train = self.x_train.as_ref().ok_or("Model not fitted")?;
        let y_train = self.y_train.as_ref().ok_or("Model not fitted")?;

        let (n_samples, n_features) = x.shape();
        let (_n_train, n_train_features) = x_train.shape();

        if n_features != n_train_features {
            return Err("Feature dimension mismatch".into());
        }

        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Compute distances to all training samples
            let mut distances: Vec<(f32, usize)> = Vec::with_capacity(y_train.len());

            for (j, &label) in y_train.iter().enumerate() {
                let dist = self.compute_distance(x, i, x_train, j, n_features);
                distances.push((dist, label));
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("Distance values are valid f32 (not NaN)")
            });
            let k_nearest = &distances[..self.k];

            // Vote for class
            let predicted_class = if self.weights {
                self.weighted_vote(k_nearest)
            } else {
                self.majority_vote(k_nearest)
            };

            predictions.push(predicted_class);
        }

        Ok(predictions)
    }

    /// Returns probability estimates for each class.
    ///
    /// Probabilities are computed as the proportion of neighbors belonging
    /// to each class (optionally weighted by inverse distance).
    ///
    /// # Errors
    ///
    /// Returns error if model is not fitted or dimensions mismatch.
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Result<Vec<Vec<f32>>> {
        let x_train = self.x_train.as_ref().ok_or("Model not fitted")?;
        let y_train = self.y_train.as_ref().ok_or("Model not fitted")?;

        let (n_samples, n_features) = x.shape();
        let (_n_train, n_train_features) = x_train.shape();

        if n_features != n_train_features {
            return Err("Feature dimension mismatch".into());
        }

        // Find number of classes
        let n_classes = *y_train
            .iter()
            .max()
            .expect("Training labels are non-empty (verified in fit())")
            + 1;

        let mut probabilities = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Compute distances to all training samples
            let mut distances: Vec<(f32, usize)> = Vec::with_capacity(y_train.len());

            for (j, &label) in y_train.iter().enumerate() {
                let dist = self.compute_distance(x, i, x_train, j, n_features);
                distances.push((dist, label));
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("Distance values are valid f32 (not NaN)")
            });
            let k_nearest = &distances[..self.k];

            // Compute class probabilities
            let mut class_counts = vec![0.0; n_classes];

            if self.weights {
                // Weighted by inverse distance
                for (dist, label) in k_nearest {
                    let weight = if *dist < 1e-10 { 1.0 } else { 1.0 / dist };
                    class_counts[*label] += weight;
                }
            } else {
                // Uniform weights
                for (_dist, label) in k_nearest {
                    class_counts[*label] += 1.0;
                }
            }

            // Normalize to probabilities
            let total: f32 = class_counts.iter().sum();
            for count in &mut class_counts {
                *count /= total;
            }

            probabilities.push(class_counts);
        }

        Ok(probabilities)
    }

    /// Computes distance between two samples.
    fn compute_distance(
        &self,
        x1: &Matrix<f32>,
        i1: usize,
        x2: &Matrix<f32>,
        i2: usize,
        n_features: usize,
    ) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean => {
                let mut sum = 0.0;
                for k in 0..n_features {
                    let diff = x1.get(i1, k) - x2.get(i2, k);
                    sum += diff * diff;
                }
                sum.sqrt()
            }
            DistanceMetric::Manhattan => {
                let mut sum = 0.0;
                for k in 0..n_features {
                    sum += (x1.get(i1, k) - x2.get(i2, k)).abs();
                }
                sum
            }
            DistanceMetric::Minkowski(p) => {
                let mut sum = 0.0;
                for k in 0..n_features {
                    sum += (x1.get(i1, k) - x2.get(i2, k)).abs().powf(p);
                }
                sum.powf(1.0 / p)
            }
        }
    }

    /// Performs majority voting among k nearest neighbors.
    #[allow(clippy::unused_self)]
    fn majority_vote(&self, neighbors: &[(f32, usize)]) -> usize {
        let mut class_counts = std::collections::HashMap::new();

        for (_dist, label) in neighbors {
            *class_counts.entry(*label).or_insert(0) += 1;
        }

        *class_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label, _)| label)
            .expect("Neighbors slice is non-empty (k >= 1)")
    }

    /// Performs weighted voting (inverse distance weighting).
    #[allow(clippy::unused_self)]
    fn weighted_vote(&self, neighbors: &[(f32, usize)]) -> usize {
        let mut class_weights = std::collections::HashMap::new();

        for (dist, label) in neighbors {
            let weight = if *dist < 1e-10 { 1.0 } else { 1.0 / dist };
            *class_weights.entry(*label).or_insert(0.0) += weight;
        }

        *class_weights
            .iter()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(label, _)| label)
            .expect("Neighbors slice is non-empty (k >= 1)")
    }
}

/// Gaussian Naive Bayes classifier.
///
/// Assumes features follow a Gaussian (normal) distribution within each class.
/// Uses Bayes' theorem with independence assumption between features.
///
/// # Example
///
/// ```
/// use aprender::classification::GaussianNB;
/// use aprender::primitives::Matrix;
///
/// let x = Matrix::from_vec(4, 2, vec![
///     0.0, 0.0,
///     0.0, 1.0,
///     1.0, 0.0,
///     1.0, 1.0,
/// ]).expect("4x2 matrix with 8 values");
/// let y = vec![0, 0, 1, 1];
///
/// let mut model = GaussianNB::new();
/// model.fit(&x, &y).expect("Valid training data");
/// let predictions = model.predict(&x).expect("Model is fitted");
/// ```
#[derive(Debug, Clone)]
pub struct GaussianNB {
    /// Class prior probabilities P(y=c)
    class_priors: Option<Vec<f32>>,
    /// Feature means per class: means[class][feature]
    means: Option<Vec<Vec<f32>>>,
    /// Feature variances per class: variances[class][feature]
    variances: Option<Vec<Vec<f32>>>,
    /// Class labels
    classes: Option<Vec<usize>>,
    /// Laplace smoothing parameter (`var_smoothing`)
    var_smoothing: f32,
}
