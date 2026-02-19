//! Agglomerative (bottom-up) hierarchical clustering.
//!
//! Builds a tree of clusters (dendrogram) by iteratively merging
//! closest clusters using various linkage methods.

use crate::error::Result;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;
use serde::{Deserialize, Serialize};

/// Linkage methods for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Linkage {
    /// Minimum distance between clusters (single linkage).
    Single,
    /// Maximum distance between clusters (complete linkage).
    Complete,
    /// Average distance between all pairs (average linkage).
    Average,
    /// Minimize within-cluster variance (Ward's method).
    Ward,
}

/// Dendrogram merge record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Merge {
    /// Clusters being merged.
    pub clusters: (usize, usize),
    /// Distance at which merge occurs.
    pub distance: f32,
    /// New cluster size after merge.
    pub size: usize,
}

/// Agglomerative (bottom-up) hierarchical clustering.
///
/// Builds a tree of clusters (dendrogram) by iteratively merging
/// closest clusters using various linkage methods.
///
/// # Algorithm
///
/// 1. Start with each point as its own cluster
/// 2. Repeat until reaching `n_clusters`:
///    - Find two closest clusters using linkage method
///    - Merge them
///    - Update distance matrix
/// 3. Return cluster labels
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(6, 2, vec![
///     1.0, 1.0, 1.1, 1.0, 1.0, 1.1,
///     5.0, 5.0, 5.1, 5.0, 5.0, 5.1,
/// ]).expect("Valid matrix dimensions and data length");
///
/// let mut hc = AgglomerativeClustering::new(2, Linkage::Average);
/// hc.fit(&data).expect("Fit succeeds with valid data");
///
/// let labels = hc.predict(&data);
/// assert_eq!(labels.len(), 6);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n³) for naive implementation
/// - Space complexity: O(n²) for distance matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgglomerativeClustering {
    /// Target number of clusters.
    n_clusters: usize,
    /// Linkage method for distance calculation.
    linkage: Linkage,
    /// Cluster labels after fitting.
    labels: Option<Vec<usize>>,
    /// Dendrogram merge history.
    dendrogram: Option<Vec<Merge>>,
}

impl AgglomerativeClustering {
    /// Create new `AgglomerativeClustering` with target number of clusters and linkage method.
    #[must_use]
    pub fn new(n_clusters: usize, linkage: Linkage) -> Self {
        Self {
            n_clusters,
            linkage,
            labels: None,
            dendrogram: None,
        }
    }

    /// Get target number of clusters.
    #[must_use]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Get linkage method.
    #[must_use]
    pub fn linkage(&self) -> Linkage {
        self.linkage
    }

    /// Check if model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.labels.is_some()
    }

    /// Get cluster labels (panic if not fitted).
    #[must_use]
    pub fn labels(&self) -> &Vec<usize> {
        self.labels
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Get dendrogram merge history (panic if not fitted).
    #[must_use]
    pub fn dendrogram(&self) -> &Vec<Merge> {
        self.dendrogram
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// ONE PATH: Core computation delegates to `nn::functional::euclidean_distance` (UCBD §4).
    #[allow(clippy::unused_self)]
    fn euclidean_distance(&self, x: &Matrix<f32>, i: usize, j: usize) -> f32 {
        let n_features = x.shape().1;
        let row_i: Vec<f32> = (0..n_features).map(|k| x.get(i, k)).collect();
        let row_j: Vec<f32> = (0..n_features).map(|k| x.get(j, k)).collect();
        crate::nn::functional::euclidean_distance(&row_i, &row_j)
    }

    /// Calculate pairwise distance matrix.
    #[allow(clippy::needless_range_loop)]
    fn pairwise_distances(&self, x: &Matrix<f32>) -> Vec<Vec<f32>> {
        let n_samples = x.shape().0;
        let mut distances = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist = self.euclidean_distance(x, i, j);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }
        distances
    }

    /// Find minimum distance between two active clusters.
    #[allow(clippy::unused_self)]
    fn find_closest_clusters(
        &self,
        distances: &[Vec<f32>],
        active: &[bool],
    ) -> (usize, usize, f32) {
        let n = distances.len();
        let mut min_dist = f32::INFINITY;
        let mut min_i = 0;
        let mut min_j = 1;

        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if distances[i][j] < min_dist {
                    min_dist = distances[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }

        (min_i, min_j, min_dist)
    }

    /// Collect pairwise distances between two clusters.
    fn pairwise_cluster_distances(
        &self,
        x: &Matrix<f32>,
        cluster_a: &[usize],
        cluster_b: &[usize],
    ) -> Vec<f32> {
        let mut dists = Vec::with_capacity(cluster_a.len() * cluster_b.len());
        for &i in cluster_a {
            for &j in cluster_b {
                dists.push(self.euclidean_distance(x, i, j));
            }
        }
        dists
    }

    /// Update distances for a newly merged cluster using specified linkage.
    fn update_distances(
        &self,
        x: &Matrix<f32>,
        distances: &mut [Vec<f32>],
        clusters: &[Vec<usize>],
        merged_idx: usize,
        other_idx: usize,
    ) {
        let merged_cluster = &clusters[merged_idx];
        let other_cluster = &clusters[other_idx];

        let dist = match self.linkage {
            Linkage::Single => {
                let dists = self.pairwise_cluster_distances(x, merged_cluster, other_cluster);
                dists.into_iter().fold(f32::INFINITY, f32::min)
            }
            Linkage::Complete => {
                let dists = self.pairwise_cluster_distances(x, merged_cluster, other_cluster);
                dists.into_iter().fold(0.0_f32, f32::max)
            }
            Linkage::Average => {
                let dists = self.pairwise_cluster_distances(x, merged_cluster, other_cluster);
                if dists.is_empty() {
                    0.0
                } else {
                    dists.iter().sum::<f32>() / dists.len() as f32
                }
            }
            Linkage::Ward => {
                let merged_centroid = self.compute_centroid(x, merged_cluster);
                let other_centroid = self.compute_centroid(x, other_cluster);
                let centroid_dist =
                    crate::nn::functional::euclidean_distance(&merged_centroid, &other_centroid);
                let n1 = merged_cluster.len() as f32;
                let n2 = other_cluster.len() as f32;
                ((n1 * n2) / (n1 + n2)) * centroid_dist
            }
        };

        distances[merged_idx][other_idx] = dist;
        distances[other_idx][merged_idx] = dist;
    }

    /// Compute centroid of a cluster.
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::unused_self)]
    fn compute_centroid(&self, x: &Matrix<f32>, cluster: &[usize]) -> Vec<f32> {
        let n_features = x.shape().1;
        let mut centroid = vec![0.0; n_features];

        for &idx in cluster {
            for k in 0..n_features {
                centroid[k] += x.get(idx, k);
            }
        }

        let size = cluster.len() as f32;
        for val in &mut centroid {
            *val /= size;
        }

        centroid
    }
}

impl UnsupervisedEstimator for AgglomerativeClustering {
    type Labels = Vec<usize>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let n_samples = x.shape().0;

        // Initialize: each point is its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        let mut active = vec![true; n_samples];
        let mut cluster_labels = vec![0; n_samples];
        let mut dendrogram = Vec::new();

        // Calculate initial pairwise distances
        let mut distances = self.pairwise_distances(x);

        // Merge until reaching target number of clusters
        while clusters.iter().filter(|c| !c.is_empty()).count() > self.n_clusters {
            // Find closest pair
            let (i, j, dist) = self.find_closest_clusters(&distances, &active);

            // Merge cluster j into cluster i
            let merged_cluster = clusters[j].clone();
            clusters[i].extend(&merged_cluster);
            clusters[j].clear();
            active[j] = false;

            // Record merge in dendrogram
            dendrogram.push(Merge {
                clusters: (i, j),
                distance: dist,
                size: clusters[i].len(),
            });

            // Update distances for merged cluster
            #[allow(clippy::needless_range_loop)]
            for k in 0..n_samples {
                if k == i || !active[k] {
                    continue;
                }
                self.update_distances(x, &mut distances, &clusters, i, k);
            }
        }

        // Assign labels
        let mut cluster_id = 0;
        for cluster in &clusters {
            if !cluster.is_empty() {
                for &point_idx in cluster {
                    cluster_labels[point_idx] = cluster_id;
                }
                cluster_id += 1;
            }
        }

        self.labels = Some(cluster_labels);
        self.dendrogram = Some(dendrogram);
        Ok(())
    }

    fn predict(&self, _x: &Matrix<f32>) -> Self::Labels {
        // For hierarchical clustering, predict returns fitted labels
        // (new points would require a different strategy)
        self.labels().clone()
    }
}
