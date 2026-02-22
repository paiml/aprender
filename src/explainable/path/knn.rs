//! KNN Decision Path
//!
//! Decision path for K-Nearest Neighbors classifier.

use serde::{Deserialize, Serialize};

use super::traits::{ByteReader, DecisionPath, PathError};

/// Decision path for KNN
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KNNPath {
    /// Indices of k nearest neighbors
    pub neighbor_indices: Vec<usize>,
    /// Distances to neighbors
    pub distances: Vec<f32>,
    /// Labels of neighbors
    pub neighbor_labels: Vec<usize>,
    /// Vote distribution: (class, count)
    pub votes: Vec<(usize, usize)>,
    /// Weighted vote (if distance-weighted)
    pub weighted_votes: Option<Vec<f32>>,
    /// Final prediction
    pub prediction: f32,
}

impl KNNPath {
    /// Create a new KNN path
    pub fn new(
        neighbor_indices: Vec<usize>,
        distances: Vec<f32>,
        neighbor_labels: Vec<usize>,
        prediction: f32,
    ) -> Self {
        // Compute vote distribution
        let mut vote_map = std::collections::HashMap::new();
        for &label in &neighbor_labels {
            *vote_map.entry(label).or_insert(0usize) += 1;
        }
        let votes: Vec<(usize, usize)> = vote_map.into_iter().collect();

        Self {
            neighbor_indices,
            distances,
            neighbor_labels,
            votes,
            weighted_votes: None,
            prediction,
        }
    }

    /// Set weighted votes
    pub fn with_weighted_votes(mut self, weights: Vec<f32>) -> Self {
        self.weighted_votes = Some(weights);
        self
    }

    /// Number of neighbors
    pub fn k(&self) -> usize {
        self.neighbor_indices.len()
    }
}

impl DecisionPath for KNNPath {
    fn explain(&self) -> String {
        let prediction = self.prediction;
        let k = self.k();
        let mut explanation = format!("KNN Prediction: {prediction:.4} (k={k})\n");

        explanation.push_str("\nNearest neighbors:\n");
        for i in 0..self.k() {
            let rank = i + 1;
            let idx = self.neighbor_indices[i];
            let label = self.neighbor_labels[i];
            let distance = self.distances[i];
            explanation.push_str(&format!(
                "  #{rank}: idx={idx}, label={label}, distance={distance:.4}\n"
            ));
        }

        explanation.push_str("\nVote distribution:\n");
        for (class, count) in &self.votes {
            explanation.push_str(&format!("  class {class}: {count} votes\n"));
        }

        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        &[]
    }

    fn confidence(&self) -> f32 {
        if self.votes.is_empty() {
            return 0.0;
        }

        let max_votes = self.votes.iter().map(|(_, c)| *c).max().unwrap_or(0);
        max_votes as f32 / self.k() as f32
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // K value
        let k = self.neighbor_indices.len() as u32;
        bytes.extend_from_slice(&k.to_le_bytes());

        // Neighbor indices
        for idx in &self.neighbor_indices {
            bytes.extend_from_slice(&(*idx as u32).to_le_bytes());
        }

        // Distances
        for d in &self.distances {
            bytes.extend_from_slice(&d.to_le_bytes());
        }

        // Labels
        for l in &self.neighbor_labels {
            bytes.extend_from_slice(&(*l as u32).to_le_bytes());
        }

        // Votes
        bytes.extend_from_slice(&(self.votes.len() as u32).to_le_bytes());
        for (class, count) in &self.votes {
            bytes.extend_from_slice(&(*class as u32).to_le_bytes());
            bytes.extend_from_slice(&(*count as u32).to_le_bytes());
        }

        // Weighted votes
        let has_weights = self.weighted_votes.is_some();
        bytes.push(u8::from(has_weights));
        if let Some(weights) = &self.weighted_votes {
            bytes.extend_from_slice(&(weights.len() as u32).to_le_bytes());
            for w in weights {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
        }

        // Prediction
        bytes.extend_from_slice(&self.prediction.to_le_bytes());

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.len() < 5 {
            return Err(PathError::InsufficientData {
                expected: 5,
                actual: bytes.len(),
            });
        }

        let mut reader = ByteReader::new(bytes);

        let version = reader.read_u8()?;
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let k = reader.read_u32_as_usize()?;

        let mut neighbor_indices = Vec::with_capacity(k);
        for _ in 0..k {
            neighbor_indices.push(reader.read_u32_as_usize()?);
        }

        let mut distances = Vec::with_capacity(k);
        for _ in 0..k {
            distances.push(reader.read_f32()?);
        }

        let mut neighbor_labels = Vec::with_capacity(k);
        for _ in 0..k {
            neighbor_labels.push(reader.read_u32_as_usize()?);
        }

        let n_votes = reader.read_u32_as_usize()?;
        let mut votes = Vec::with_capacity(n_votes);
        for _ in 0..n_votes {
            let class = reader.read_u32_as_usize()?;
            let count = reader.read_u32_as_usize()?;
            votes.push((class, count));
        }

        let weighted_votes = reader.read_optional(ByteReader::read_f32_vec)?;
        let prediction = reader.read_f32()?;

        Ok(Self {
            neighbor_indices,
            distances,
            neighbor_labels,
            votes,
            weighted_votes,
            prediction,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_path_new() {
        let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0);

        assert_eq!(path.k(), 3);
        assert_eq!(path.neighbor_indices, vec![0, 1, 2]);
        assert_eq!(path.distances, vec![0.1, 0.2, 0.3]);
        assert_eq!(path.neighbor_labels, vec![0, 0, 1]);
        assert_eq!(path.prediction, 0.0);
        assert!(path.weighted_votes.is_none());
        assert!(!path.votes.is_empty());
    }

    #[test]
    fn test_knn_path_with_weighted_votes() {
        let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0)
            .with_weighted_votes(vec![0.5, 0.3, 0.2]);

        assert!(path.weighted_votes.is_some());
        assert_eq!(
            path.weighted_votes.expect("has weights"),
            vec![0.5, 0.3, 0.2]
        );
    }

    #[test]
    fn test_knn_path_k() {
        let path = KNNPath::new(
            vec![0, 1, 2, 3, 4],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![0, 0, 1, 1, 2],
            1.0,
        );

        assert_eq!(path.k(), 5);
    }

    #[test]
    fn test_knn_path_explain() {
        let path = KNNPath::new(vec![10, 20, 30], vec![0.5, 1.0, 1.5], vec![0, 1, 0], 0.0);

        let explanation = path.explain();
        assert!(explanation.contains("KNN Prediction: 0.0000 (k=3)"));
        assert!(explanation.contains("Nearest neighbors:"));
        assert!(explanation.contains("#1: idx=10, label=0, distance=0.5000"));
        assert!(explanation.contains("#2: idx=20, label=1, distance=1.0000"));
        assert!(explanation.contains("#3: idx=30, label=0, distance=1.5000"));
        assert!(explanation.contains("Vote distribution:"));
    }

    #[test]
    fn test_knn_path_feature_contributions() {
        let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0);
        assert!(path.feature_contributions().is_empty());
    }

    #[test]
    fn test_knn_path_confidence() {
        let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 0], 0.0);
        assert!((path.confidence() - 1.0).abs() < 1e-6);

        let path2 = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 1, 2], 0.0);
        assert!((path2.confidence() - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_knn_path_confidence_empty_votes() {
        let mut path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0);
        path.votes.clear();
        assert_eq!(path.confidence(), 0.0);
    }

    #[test]
    fn test_knn_path_serialization_roundtrip() {
        let path = KNNPath::new(vec![5, 10, 15], vec![0.25, 0.5, 0.75], vec![1, 1, 0], 1.0);

        let bytes = path.to_bytes();
        let restored = KNNPath::from_bytes(&bytes).expect("roundtrip");

        assert_eq!(restored.neighbor_indices, path.neighbor_indices);
        assert_eq!(restored.distances, path.distances);
        assert_eq!(restored.neighbor_labels, path.neighbor_labels);
        assert_eq!(restored.prediction, path.prediction);
    }

    #[test]
    fn test_knn_path_serialization_with_weighted_votes() {
        let path = KNNPath::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0)
            .with_weighted_votes(vec![0.6, 0.3, 0.1]);

        let bytes = path.to_bytes();
        let restored = KNNPath::from_bytes(&bytes).expect("roundtrip");

        assert!(restored.weighted_votes.is_some());
        assert_eq!(
            restored.weighted_votes.expect("has weights"),
            vec![0.6, 0.3, 0.1]
        );
    }

    #[test]
    fn test_knn_path_from_bytes_insufficient_data() {
        let bytes = vec![1, 0, 0];

        let result = KNNPath::from_bytes(&bytes);
        assert!(result.is_err());
        match result {
            Err(PathError::InsufficientData { expected, actual }) => {
                assert_eq!(expected, 5);
                assert_eq!(actual, 3);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_knn_path_from_bytes_version_mismatch() {
        let bytes = vec![2, 0, 0, 0, 0];

        let result = KNNPath::from_bytes(&bytes);
        assert!(result.is_err());
        match result {
            Err(PathError::VersionMismatch { expected, actual }) => {
                assert_eq!(expected, 1);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected VersionMismatch error"),
        }
    }

    #[test]
    fn test_knn_path_from_bytes_truncated_neighbor_indices() {
        let bytes = vec![1, 3, 0, 0, 0];

        let result = KNNPath::from_bytes(&bytes);
        assert!(result.is_err());
        match result {
            Err(PathError::InsufficientData { .. }) => {}
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_knn_path_clone() {
        let path = KNNPath::new(vec![0, 1, 2], vec![0.1, 0.2, 0.3], vec![0, 0, 1], 0.0);

        let cloned = path.clone();
        assert_eq!(cloned.neighbor_indices, path.neighbor_indices);
        assert_eq!(cloned.distances, path.distances);
        assert_eq!(cloned.neighbor_labels, path.neighbor_labels);
    }

    #[test]
    fn test_knn_path_debug() {
        let path = KNNPath::new(vec![0], vec![0.5], vec![1], 1.0);

        let debug_str = format!("{:?}", path);
        assert!(debug_str.contains("KNNPath"));
        assert!(debug_str.contains("neighbor_indices"));
    }

    #[test]
    fn test_knn_path_serde_json() {
        let path = KNNPath::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3], vec![0, 1, 0], 0.0);

        let json = serde_json::to_string(&path).expect("serialize");
        let restored: KNNPath = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.neighbor_indices, path.neighbor_indices);
        assert_eq!(restored.prediction, path.prediction);
    }
}
