//! ML-based anomaly detection using Aprender (Sprint 23)
//!
//! Uses `KMeans` clustering to identify anomalous syscall patterns.

use aprender::cluster::KMeans;
use aprender::primitives::Matrix;
use aprender::traits::UnsupervisedEstimator;
use std::collections::HashMap;

/// Result of ML anomaly analysis
#[derive(Debug, Clone)]
pub struct MlAnomalyReport {
    /// Silhouette score for clustering quality (-1 to 1, higher is better)
    pub silhouette_score: f64,
    /// Number of clusters used
    pub num_clusters: usize,
    /// Cluster assignments for each syscall type
    pub cluster_assignments: HashMap<String, usize>,
    /// Cluster centroids (mean time per cluster)
    pub cluster_centers: Vec<f64>,
    /// Anomalous syscalls (in outlier clusters)
    pub anomalies: Vec<MlAnomaly>,
    /// Total samples analyzed
    pub total_samples: usize,
}

/// A detected ML anomaly
#[derive(Debug, Clone)]
pub struct MlAnomaly {
    /// Syscall name
    pub syscall: String,
    /// Average time in microseconds
    pub avg_time_us: f64,
    /// Cluster assignment
    pub cluster: usize,
    /// Distance from cluster center
    pub distance: f64,
}

/// ML Anomaly Analyzer using Aprender `KMeans`
pub struct MlAnomalyAnalyzer {
    num_clusters: usize,
}

impl MlAnomalyAnalyzer {
    /// Create a new ML anomaly analyzer
    pub fn new(num_clusters: usize) -> Self {
        Self { num_clusters }
    }

    /// Analyze syscall data and produce ML anomaly report
    pub fn analyze(&self, syscall_data: &HashMap<String, (u64, u64)>) -> MlAnomalyReport {
        // Convert data to feature vectors: (count, total_time_us)
        let mut syscall_names: Vec<String> = Vec::new();
        let mut features_data: Vec<f32> = Vec::new();

        for (name, (count, total_time_ns)) in syscall_data {
            if *count == 0 {
                continue;
            }
            syscall_names.push(name.clone());
            let total_time_us = *total_time_ns as f64 / 1000.0;
            let avg_time = total_time_us / *count as f64;
            // Features: average time (normalized)
            features_data.push(avg_time as f32);
        }

        // Run KMeans clustering with adjusted k
        let k = self.num_clusters.min(syscall_names.len());

        // Handle insufficient data (need at least 2 for clustering)
        if k < 2 {
            return self.insufficient_data_report(syscall_names.len());
        }

        let mut kmeans = KMeans::new(k);

        // Create matrix from features (n_samples x 1 feature)
        let features = match Matrix::from_vec(syscall_names.len(), 1, features_data.clone()) {
            Ok(m) => m,
            Err(_) => return self.insufficient_data_report(syscall_names.len()),
        };

        // Fit and predict
        if kmeans.fit(&features).is_err() {
            return self.insufficient_data_report(syscall_names.len());
        }
        let labels = kmeans.predict(&features);

        // Calculate cluster centers
        let cluster_centers = self.calculate_centers_from_features(&features_data, &labels, k);

        // Build cluster assignments
        let mut cluster_assignments = HashMap::new();
        for (i, name) in syscall_names.iter().enumerate() {
            cluster_assignments.insert(name.clone(), labels[i]);
        }

        // Calculate silhouette score
        let silhouette_score = self.calculate_silhouette_from_features(&features_data, &labels, k);

        // Identify anomalies (syscalls in smallest cluster or furthest from center)
        let anomalies = self.identify_anomalies_from_features(
            &syscall_names,
            &features_data,
            &labels,
            &cluster_centers,
        );

        MlAnomalyReport {
            silhouette_score,
            num_clusters: k,
            cluster_assignments,
            cluster_centers,
            anomalies,
            total_samples: syscall_names.len(),
        }
    }

    /// Calculate cluster centers from flat features
    fn calculate_centers_from_features(
        &self,
        features: &[f32],
        labels: &[usize],
        k: usize,
    ) -> Vec<f64> {
        let mut centers = vec![0.0; k];
        let mut counts = vec![0usize; k];

        for (i, &feature) in features.iter().enumerate() {
            let cluster = labels[i];
            centers[cluster] += f64::from(feature);
            counts[cluster] += 1;
        }

        for i in 0..k {
            if counts[i] > 0 {
                centers[i] /= counts[i] as f64;
            }
        }

        centers
    }

    /// Calculate simplified silhouette score from flat features
    fn calculate_silhouette_from_features(
        &self,
        features: &[f32],
        labels: &[usize],
        k: usize,
    ) -> f64 {
        if features.len() <= k || k <= 1 {
            return 0.0;
        }

        let mut total_score = 0.0;
        let n = features.len();

        for i in 0..n {
            let cluster_i = labels[i];

            // Calculate a(i): average distance to points in same cluster
            let mut same_cluster_dist = 0.0;
            let mut same_count = 0;

            for j in 0..n {
                if i != j && labels[j] == cluster_i {
                    same_cluster_dist += f64::from((features[i] - features[j]).abs());
                    same_count += 1;
                }
            }

            let a_i = if same_count > 0 { same_cluster_dist / f64::from(same_count) } else { 0.0 };

            // Calculate b(i): minimum average distance to other clusters
            let mut min_other_dist = f64::MAX;

            for c in 0..k {
                if c == cluster_i {
                    continue;
                }

                let mut other_dist = 0.0;
                let mut other_count = 0;

                for j in 0..n {
                    if labels[j] == c {
                        other_dist += f64::from((features[i] - features[j]).abs());
                        other_count += 1;
                    }
                }

                if other_count > 0 {
                    let avg_dist = other_dist / f64::from(other_count);
                    if avg_dist < min_other_dist {
                        min_other_dist = avg_dist;
                    }
                }
            }

            let b_i = if min_other_dist == f64::MAX { 0.0 } else { min_other_dist };

            // Silhouette coefficient for point i
            let max_ab = a_i.max(b_i);
            let s_i = if max_ab > 0.0 { (b_i - a_i) / max_ab } else { 0.0 };

            total_score += s_i;
        }

        total_score / n as f64
    }

    /// Identify anomalous syscalls from flat features
    fn identify_anomalies_from_features(
        &self,
        names: &[String],
        features: &[f32],
        labels: &[usize],
        centers: &[f64],
    ) -> Vec<MlAnomaly> {
        let mut anomalies = Vec::new();

        // Find the cluster with highest average (potential anomalies)
        let max_center = centers.iter().copied().fold(0.0, f64::max);
        let anomaly_threshold = max_center * 0.5; // Syscalls in clusters > 50% of max center

        for (i, name) in names.iter().enumerate() {
            let cluster = labels[i];
            let center = centers[cluster];
            let feature_val = f64::from(features[i]);
            let distance = (feature_val - center).abs();

            // Mark as anomaly if in high-latency cluster
            if center > anomaly_threshold && center == max_center {
                anomalies.push(MlAnomaly {
                    syscall: name.clone(),
                    avg_time_us: feature_val,
                    cluster,
                    distance,
                });
            }
        }

        // Sort by average time descending (handle NaN gracefully)
        anomalies.sort_by(|a, b| {
            b.avg_time_us.partial_cmp(&a.avg_time_us).unwrap_or(std::cmp::Ordering::Equal)
        });

        anomalies
    }

    /// Create report for insufficient data
    fn insufficient_data_report(&self, sample_count: usize) -> MlAnomalyReport {
        MlAnomalyReport {
            silhouette_score: 0.0,
            num_clusters: 0,
            cluster_assignments: HashMap::new(),
            cluster_centers: Vec::new(),
            anomalies: Vec::new(),
            total_samples: sample_count,
        }
    }
}

impl MlAnomalyReport {
    /// Format the report for display
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("\n=== ML Anomaly Detection Report ===\n");

        if self.num_clusters == 0 {
            output.push_str("Insufficient data for ML analysis\n");
            output.push_str(&format!(
                "(Need at least 3 syscall types, found {})\n",
                self.total_samples
            ));
            return output;
        }

        output.push_str(&format!("Clusters: {}\n", self.num_clusters));
        output.push_str(&format!("Samples: {}\n", self.total_samples));
        output.push_str(&format!("Silhouette Score: {:.3}\n", self.silhouette_score));

        // Cluster centers
        output.push_str("\nCluster Centers (avg time in \u{03bc}s):\n");
        for (i, center) in self.cluster_centers.iter().enumerate() {
            output.push_str(&format!("  Cluster {i}: {center:.2} \u{03bc}s\n"));
        }

        // Anomalies
        if self.anomalies.is_empty() {
            output.push_str("\nNo anomalies detected.\n");
        } else {
            output.push_str(&format!("\nAnomalies Detected: {}\n", self.anomalies.len()));
            for anomaly in &self.anomalies {
                output.push_str(&format!(
                    "  - {} (cluster {}): {:.2} \u{03bc}s (distance: {:.2})\n",
                    anomaly.syscall, anomaly.cluster, anomaly.avg_time_us, anomaly.distance
                ));
            }
        }

        output
    }

    /// Format comparison with z-score results
    pub fn format_comparison(&self, zscore_anomalies: &[(String, f64)]) -> String {
        let mut output = self.format();

        output.push_str("\n=== ML vs Z-Score Comparison ===\n");

        let ml_set: std::collections::HashSet<_> =
            self.anomalies.iter().map(|a| &a.syscall).collect();
        let zscore_set: std::collections::HashSet<_> =
            zscore_anomalies.iter().map(|(name, _)| name).collect();

        // Common anomalies
        let common: Vec<_> = ml_set.intersection(&zscore_set).collect();
        output.push_str(&format!("Common anomalies: {}\n", common.len()));

        // ML-only anomalies
        let ml_only: Vec<_> = ml_set.difference(&zscore_set).collect();
        if !ml_only.is_empty() {
            output.push_str(&format!("ML-only anomalies: {ml_only:?}\n"));
        }

        // Z-score-only anomalies
        let zscore_only: Vec<_> = zscore_set.difference(&ml_set).collect();
        if !zscore_only.is_empty() {
            output.push_str(&format!("Z-score-only anomalies: {zscore_only:?}\n"));
        }

        output
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(MlAnomalyReport: Send, Sync);
static_assertions::assert_impl_all!(MlAnomaly: Send, Sync);
static_assertions::assert_impl_all!(MlAnomalyAnalyzer: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_analyzer_creation() {
        let analyzer = MlAnomalyAnalyzer::new(3);
        assert_eq!(analyzer.num_clusters, 3);
    }

    #[test]
    fn test_analyze_empty_data() {
        let analyzer = MlAnomalyAnalyzer::new(3);
        let data = HashMap::new();
        let report = analyzer.analyze(&data);
        assert_eq!(report.total_samples, 0);
        assert_eq!(report.num_clusters, 0);
    }

    #[test]
    fn test_analyze_insufficient_data() {
        let analyzer = MlAnomalyAnalyzer::new(3);
        let mut data = HashMap::new();
        data.insert("write".to_string(), (10, 1000));
        data.insert("read".to_string(), (5, 500));

        let report = analyzer.analyze(&data);
        assert_eq!(report.num_clusters, 2); // k = min(3, 2)
    }

    #[test]
    fn test_analyze_with_sufficient_data() {
        let analyzer = MlAnomalyAnalyzer::new(3);
        let mut data = HashMap::new();
        data.insert("write".to_string(), (100, 100000)); // 1 us avg
        data.insert("read".to_string(), (50, 50000)); // 1 us avg
        data.insert("openat".to_string(), (20, 200000)); // 10 us avg
        data.insert("close".to_string(), (80, 80000)); // 1 us avg
        data.insert("mmap".to_string(), (10, 1000000)); // 100 us avg (anomaly)

        let report = analyzer.analyze(&data);
        assert_eq!(report.num_clusters, 3);
        assert!(report.silhouette_score >= -1.0 && report.silhouette_score <= 1.0);
    }

    #[test]
    fn test_report_format() {
        let report = MlAnomalyReport {
            silhouette_score: 0.75,
            num_clusters: 3,
            cluster_assignments: HashMap::new(),
            cluster_centers: vec![1.0, 10.0, 100.0],
            anomalies: vec![],
            total_samples: 5,
        };

        let formatted = report.format();
        assert!(formatted.contains("ML Anomaly Detection"));
        assert!(formatted.contains("Silhouette Score: 0.750"));
        assert!(formatted.contains("Clusters: 3"));
    }

    #[test]
    fn test_calculate_centers() {
        let analyzer = MlAnomalyAnalyzer::new(2);
        let features = vec![1.0_f32, 2.0, 10.0, 11.0];
        let labels = vec![0, 0, 1, 1];

        let centers = analyzer.calculate_centers_from_features(&features, &labels, 2);
        assert_eq!(centers.len(), 2);
        assert!((centers[0] - 1.5).abs() < 0.01); // (1+2)/2
        assert!((centers[1] - 10.5).abs() < 0.01); // (10+11)/2
    }

    #[test]
    fn test_silhouette_calculation() {
        let analyzer = MlAnomalyAnalyzer::new(2);
        let features = vec![1.0_f32, 2.0, 100.0, 101.0];
        let labels = vec![0, 0, 1, 1];

        let score = analyzer.calculate_silhouette_from_features(&features, &labels, 2);
        // Well-separated clusters should have high silhouette
        assert!(score > 0.8);
    }

    #[test]
    fn test_comparison_format() {
        let report = MlAnomalyReport {
            silhouette_score: 0.75,
            num_clusters: 3,
            cluster_assignments: HashMap::new(),
            cluster_centers: vec![1.0, 10.0, 100.0],
            anomalies: vec![MlAnomaly {
                syscall: "mmap".to_string(),
                avg_time_us: 100.0,
                cluster: 2,
                distance: 0.0,
            }],
            total_samples: 5,
        };

        let zscore_anomalies = vec![("mmap".to_string(), 4.5)];
        let formatted = report.format_comparison(&zscore_anomalies);
        assert!(formatted.contains("Comparison"));
        assert!(formatted.contains("Common anomalies: 1"));
    }
}
