
impl KMeans {
    /// Creates a new K-Means with the specified number of clusters.
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            random_state: None,
            n_init: DEFAULT_N_INIT,
            centroids: None,
            labels: None,
            inertia: 0.0,
            n_iter: 0,
        }
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sets the number of seeded restarts (sklearn's `n_init`, default 1).
    ///
    /// Restart 0 starts from the same row a single-init fit uses, and a later
    /// restart replaces it only with a strictly lower inertia, so the fitted
    /// inertia never exceeds the `n_init = 1` inertia. Zero is treated as 1.
    #[must_use]
    pub fn with_n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init.max(1);
        self
    }

    /// Returns the number of seeded restarts.
    #[must_use]
    pub fn n_init(&self) -> usize {
        self.n_init
    }

    /// Returns the cluster centroids.
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    #[must_use]
    pub fn centroids(&self) -> &Matrix<f32> {
        self.centroids
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Returns the inertia (within-cluster sum of squares).
    #[must_use]
    pub fn inertia(&self) -> f32 {
        self.inertia
    }

    /// Returns the number of iterations run.
    #[must_use]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// Returns true if the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.centroids.is_some()
    }

    /// Returns the number of clusters.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Returns the maximum number of iterations.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// Returns the convergence tolerance.
    #[must_use]
    pub fn tol(&self) -> f32 {
        self.tol
    }

    /// Returns the random state.
    #[must_use]
    pub fn random_state(&self) -> Option<u64> {
        self.random_state
    }

    /// Saves the model to a binary file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {e}"))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {e}"))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// Files saved before `n_init` existed still load, with `n_init = 1`.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
        Self::from_bincode(&bytes)
    }

    /// Decodes the current layout, falling back to the pre-`n_init` one.
    ///
    /// `n_init` is the LAST field, so an old file runs out of bytes exactly
    /// where it would start and the current decode fails; a current file is
    /// never tried as legacy.
    pub(crate) fn from_bincode(bytes: &[u8]) -> std::result::Result<Self, String> {
        let current = match bincode::deserialize::<Self>(bytes) {
            Ok(model) => return Ok(model),
            Err(e) => e,
        };
        let old: LegacyKMeans = bincode::deserialize(bytes)
            .map_err(|e| format!("Deserialization failed: {current} (pre-n_init layout: {e})"))?;
        Ok(Self {
            n_clusters: old.n_clusters,
            max_iter: old.max_iter,
            tol: old.tol,
            random_state: old.random_state,
            centroids: old.centroids,
            labels: old.labels,
            inertia: old.inertia,
            n_iter: old.n_iter,
            n_init: 1,
        })
    }

    /// Saves the K-Means model to a `SafeTensors` file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the `SafeTensors` file will be saved
    ///
    /// # Errors
    ///
    /// Returns an error if the model is unfitted or if saving fails.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Check if model is fitted
        let centroids = self
            .centroids
            .as_ref()
            .ok_or("Cannot save unfitted model. Call fit() first.")?;

        let mut tensors = BTreeMap::new();

        // Save centroids matrix as flat array
        let (n_clusters, n_features) = centroids.shape();
        let mut centroids_data = Vec::with_capacity(n_clusters * n_features);
        for i in 0..n_clusters {
            for j in 0..n_features {
                centroids_data.push(centroids.get(i, j));
            }
        }
        tensors.insert(
            "centroids".to_string(),
            (centroids_data, vec![n_clusters, n_features]),
        );

        // Save hyperparameters
        tensors.insert(
            "n_clusters".to_string(),
            (vec![self.n_clusters as f32], vec![1]),
        );
        tensors.insert(
            "max_iter".to_string(),
            (vec![self.max_iter as f32], vec![1]),
        );
        tensors.insert("tol".to_string(), (vec![self.tol], vec![1]));
        tensors.insert(
            "n_init".to_string(),
            (vec![self.n_init as f32], vec![1]),
        );

        let random_state_val = if let Some(state) = self.random_state {
            state as f32
        } else {
            -1.0
        };
        tensors.insert(
            "random_state".to_string(),
            (vec![random_state_val], vec![1]),
        );

        // Save metadata
        tensors.insert("inertia".to_string(), (vec![self.inertia], vec![1]));
        tensors.insert("n_iter".to_string(), (vec![self.n_iter as f32], vec![1]));

        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a K-Means model from a `SafeTensors` file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `SafeTensors` file
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails or if the file format is invalid.
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract centroids tensor
        let centroids_meta = metadata
            .get("centroids")
            .ok_or("Missing 'centroids' tensor in SafeTensors file")?;
        let centroids_data = safetensors::extract_tensor(&raw_data, centroids_meta)?;

        // Get shape from metadata
        let shape = &centroids_meta.shape;
        if shape.len() != 2 {
            return Err("Invalid centroids tensor shape".to_string());
        }
        let n_clusters_from_shape = shape[0];
        let n_features = shape[1];

        // Reconstruct centroids matrix
        let centroids = Matrix::from_vec(n_clusters_from_shape, n_features, centroids_data)
            .map_err(|e| format!("Failed to reconstruct centroids matrix: {e}"))?;

        // Load hyperparameters
        let n_clusters_meta = metadata
            .get("n_clusters")
            .ok_or("Missing 'n_clusters' tensor")?;
        let n_clusters_data = safetensors::extract_tensor(&raw_data, n_clusters_meta)?;
        let n_clusters = n_clusters_data[0] as usize;

        let max_iter_meta = metadata
            .get("max_iter")
            .ok_or("Missing 'max_iter' tensor")?;
        let max_iter_data = safetensors::extract_tensor(&raw_data, max_iter_meta)?;
        let max_iter = max_iter_data[0] as usize;

        let tol_meta = metadata.get("tol").ok_or("Missing 'tol' tensor")?;
        let tol_data = safetensors::extract_tensor(&raw_data, tol_meta)?;
        let tol = tol_data[0];

        // Files written before n_init existed carry no tensor: they were fitted
        // from a single start, so they load as n_init = 1.
        let n_init = match metadata.get("n_init") {
            Some(meta) => (safetensors::extract_tensor(&raw_data, meta)?[0] as usize).max(1),
            None => 1,
        };

        let random_state_meta = metadata
            .get("random_state")
            .ok_or("Missing 'random_state' tensor")?;
        let random_state_data = safetensors::extract_tensor(&raw_data, random_state_meta)?;
        let random_state = if random_state_data[0] < 0.0 {
            None
        } else {
            Some(random_state_data[0] as u64)
        };

        // Load metadata
        let inertia_meta = metadata.get("inertia").ok_or("Missing 'inertia' tensor")?;
        let inertia_data = safetensors::extract_tensor(&raw_data, inertia_meta)?;
        let inertia = inertia_data[0];

        let n_iter_meta = metadata.get("n_iter").ok_or("Missing 'n_iter' tensor")?;
        let n_iter_data = safetensors::extract_tensor(&raw_data, n_iter_meta)?;
        let n_iter = n_iter_data[0] as usize;

        Ok(Self {
            n_clusters,
            max_iter,
            tol,
            random_state,
            n_init,
            centroids: Some(centroids),
            labels: None, // Training labels not serialized
            inertia,
            n_iter,
        })
    }

    /// First centroid row of restart `run`. Restart 0 is `seed % n_samples`,
    /// the row a single-init fit has always used; later restarts draw a row
    /// from the seed through `SplitMix64`, so they are reproducible.
    pub(crate) fn restart_start_row(&self, run: usize, n_samples: usize) -> usize {
        let seed = self.random_state.unwrap_or(42);
        if run == 0 {
            return (seed as usize) % n_samples;
        }
        (splitmix64(seed ^ (run as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)) % n_samples as u64)
            as usize
    }

    /// Greedy k-means++ (D²) seeding for restart `run`, as sklearn and linfa do.
    /// The first centroid is row [`Self::restart_start_row`].
    ///
    /// Each further centroid is the best of `2 + ln k` candidates, each drawn
    /// with probability proportional to its squared distance to the nearest
    /// centroid so far; "best" is the lowest resulting potential (sum of those
    /// distances). The draws come from `SplitMix64` keyed on the seed and
    /// `run`, so a seeded fit is reproducible on every platform, and two
    /// restarts that share a start row still draw differently.
    /// Farthest-point seeding, used before #3146, always took the argmax and
    /// so chased outliers.
    fn kmeans_plusplus_init(&self, x: &Matrix<f32>, run: usize) -> Matrix<f32> {
        let (n_samples, n_features) = x.shape();
        let first_idx = self.restart_start_row(run, n_samples);
        let mut centroids_data = Vec::with_capacity(self.n_clusters * n_features);
        append_row(&mut centroids_data, x, first_idx, n_features);

        let seed = self.random_state.unwrap_or(42);
        let mut rng = SplitMix64(splitmix64(seed ^ 0xA076_1D64_78BD_642F) ^ run as u64);
        let n_trials = 2 + (self.n_clusters as f64).ln() as usize;
        let mut closest = distances_sq_to_sample(x, first_idx);

        for _ in 1..self.n_clusters {
            let cumulative: Vec<f64> = closest
                .iter()
                .scan(0.0_f64, |acc, &d| {
                    *acc += f64::from(d);
                    Some(*acc)
                })
                .collect();
            let total = cumulative[n_samples - 1];

            let mut best: Option<(f64, usize, Vec<f32>)> = None;
            for _ in 0..n_trials {
                let target = rng.next_unit() * total;
                let cand = cumulative
                    .partition_point(|&cum| cum <= target)
                    .min(n_samples - 1);
                let d_cand = distances_sq_to_sample(x, cand);
                let merged: Vec<f32> = closest.iter().zip(&d_cand).map(|(&a, &b)| a.min(b)).collect();
                let potential: f64 = merged.iter().map(|&d| f64::from(d)).sum();
                if best.as_ref().map_or(true, |b| potential < b.0) {
                    best = Some((potential, cand, merged));
                }
            }
            let (_, cand, merged) = best.expect("n_trials >= 2");
            append_row(&mut centroids_data, x, cand, n_features);
            closest = merged;
        }

        Matrix::from_vec(self.n_clusters, n_features, centroids_data)
            .expect("Centroid matrix dimensions match allocated data length")
    }

    /// Assigns each sample to the nearest centroid.
    fn assign_labels(&self, x: &Matrix<f32>, centroids: &Matrix<f32>) -> Vec<usize> {
        let n_samples = x.n_rows();
        let mut labels = vec![0; n_samples];

        for (i, label) in labels.iter_mut().enumerate() {
            let point = x.row(i);
            let mut min_dist = f32::INFINITY;
            let mut min_cluster = 0;

            for k in 0..self.n_clusters {
                let centroid = centroids.row(k);
                let diff = &point - &centroid;
                let dist = diff.norm_squared();

                if dist < min_dist {
                    min_dist = dist;
                    min_cluster = k;
                }
            }

            *label = min_cluster;
        }

        labels
    }

    /// Updates centroids as the mean of assigned samples.
    fn update_centroids(&self, x: &Matrix<f32>, labels: &[usize]) -> Matrix<f32> {
        let (_, n_features) = x.shape();
        let mut new_centroids = vec![0.0; self.n_clusters * n_features];
        let mut counts = vec![0usize; self.n_clusters];

        // Sum points in each cluster
        for (i, &label) in labels.iter().enumerate() {
            counts[label] += 1;
            for j in 0..n_features {
                new_centroids[label * n_features + j] += x.get(i, j);
            }
        }

        // Compute means
        for k in 0..self.n_clusters {
            if counts[k] > 0 {
                for j in 0..n_features {
                    new_centroids[k * n_features + j] /= counts[k] as f32;
                }
            }
        }

        Matrix::from_vec(self.n_clusters, n_features, new_centroids)
            .expect("Updated centroid matrix dimensions match preallocated vector length")
    }

    /// Checks if centroids have converged.
    pub(crate) fn centroids_converged(&self, old: &Matrix<f32>, new: &Matrix<f32>) -> bool {
        let (n_clusters, n_features) = old.shape();

        for k in 0..n_clusters {
            let mut dist_sq = 0.0;
            for j in 0..n_features {
                let diff = old.get(k, j) - new.get(k, j);
                dist_sq += diff * diff;
            }
            if dist_sq > self.tol * self.tol {
                return false;
            }
        }

        true
    }
}

impl UnsupervisedEstimator for KMeans {
    type Labels = Vec<usize>;

    /// Fits the K-Means model to data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data has fewer samples than clusters
    /// - Data is empty
    #[provable_contracts_macros::contract("kmeans-kernel-v1", equation = "update")]
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let n_samples = x.n_rows();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        if n_samples < self.n_clusters {
            return Err("Number of samples must be >= number of clusters".into());
        }

        // Best of n_init seeded restarts; ties keep the earlier run, so restart 0
        // (the single-init result) survives unless a restart strictly beats it.
        let mut best: Option<(Matrix<f32>, Vec<usize>, f32, usize)> = None;
        for run in 0..self.n_init.max(1) {
            let (centroids, labels, run_inertia, n_iter) = self.lloyd(x, run);
            if best.as_ref().map_or(true, |b| run_inertia < b.2) {
                best = Some((centroids, labels, run_inertia, n_iter));
            }
        }
        let (centroids, labels, best_inertia, n_iter) =
            best.expect("n_init >= 1 runs at least one restart");

        self.inertia = best_inertia;
        self.n_iter = n_iter;
        self.labels = Some(labels);
        self.centroids = Some(centroids);

        Ok(())
    }

    /// Predicts cluster labels for new data.
    #[provable_contracts_macros::contract("kmeans-kernel-v1", equation = "assignment")]
    fn predict(&self, x: &Matrix<f32>) -> Vec<usize> {
        let centroids = self
            .centroids
            .as_ref()
            .expect("Model not fitted. Call fit() first.");

        self.assign_labels(x, centroids)
    }
}

impl KMeans {
    /// Restart `run`: D² seeding, then Lloyd iterations.
    fn lloyd(&self, x: &Matrix<f32>, run: usize) -> (Matrix<f32>, Vec<usize>, f32, usize) {
        self.lloyd_from(x, self.kmeans_plusplus_init(x, run))
    }

    /// Lloyd iterations from `centroids`: (centroids, labels, inertia, iterations).
    fn lloyd_from(
        &self,
        x: &Matrix<f32>,
        mut centroids: Matrix<f32>,
    ) -> (Matrix<f32>, Vec<usize>, f32, usize) {
        let mut labels = vec![0; x.n_rows()];
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            // Assign samples to nearest centroid
            labels = self.assign_labels(x, &centroids);

            // Update centroids
            let new_centroids = self.update_centroids(x, &labels);

            // Check convergence
            if self.centroids_converged(&centroids, &new_centroids) {
                n_iter = iter + 1;
                centroids = new_centroids;
                break;
            }

            centroids = new_centroids;
            n_iter = iter + 1;
        }

        let run_inertia = inertia(x, &centroids, &labels);
        (centroids, labels, run_inertia, n_iter)
    }
}

/// `SplitMix64` finalizer: a seeded, platform-independent row draw for restarts.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A seeded `SplitMix64` stream for the D² draws.
struct SplitMix64(u64);

impl SplitMix64 {
    /// Uniform in `[0, 1)` from the top 53 bits.
    fn next_unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (splitmix64(self.0) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Squared distance of every sample to sample `row`.
fn distances_sq_to_sample(x: &Matrix<f32>, row: usize) -> Vec<f32> {
    let n_features = x.n_cols();
    (0..x.n_rows())
        .map(|i| {
            (0..n_features)
                .map(|j| {
                    let diff = x.get(i, j) - x.get(row, j);
                    diff * diff
                })
                .sum()
        })
        .collect()
}

/// Append the `row`-th row of `x` to the flat centroid buffer.
fn append_row(centroids_data: &mut Vec<f32>, x: &Matrix<f32>, row: usize, n_features: usize) {
    for j in 0..n_features {
        centroids_data.push(x.get(row, j));
    }
}
