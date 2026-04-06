//! Clustering algorithms.
//!
//! Includes K-Means, DBSCAN, Hierarchical, Gaussian Mixture Models, and Isolation Forest.

// Submodules
mod agglomerative;
mod dbscan;
mod gmm;
mod isolation_forest;
mod kmeans;
mod lof;
mod spectral;

// Re-exports
pub use agglomerative::{AgglomerativeClustering, Linkage, Merge};
pub use dbscan::DBSCAN;
pub use gmm::{CovarianceType, GaussianMixture};
pub use isolation_forest::IsolationForest;
pub use kmeans::KMeans;
pub use lof::LocalOutlierFactor;
pub use spectral::{Affinity, SpectralClustering};

#[cfg(test)]
mod tests;
