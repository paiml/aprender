//! LinearPath - Decision path for linear models

use serde::{Deserialize, Serialize};

use super::traits::{DecisionPath, PathError};

/// Decision path for linear regression/logistic regression
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearPath {
    /// Per-feature contributions: coefficient\[i\] * input\[i\]
    pub contributions: Vec<f32>,
    /// Bias term contribution
    pub intercept: f32,
    /// Raw prediction before activation
    pub logit: f32,
    /// Final prediction
    pub prediction: f32,
    /// For classification: probability (sigmoid/softmax output)
    pub probability: Option<f32>,
}

impl LinearPath {
    /// Create a new linear path
    pub fn new(contributions: Vec<f32>, intercept: f32, logit: f32, prediction: f32) -> Self {
        Self {
            contributions,
            intercept,
            logit,
            prediction,
            probability: None,
        }
    }

    /// Set probability for classification
    pub fn with_probability(mut self, prob: f32) -> Self {
        self.probability = Some(prob);
        self
    }

    /// Get top k features by absolute contribution
    pub fn top_features(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .contributions
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();

        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        indexed
    }
}

impl DecisionPath for LinearPath {
    fn explain(&self) -> String {
        let mut explanation = format!("Prediction: {:.4}", self.prediction);

        if let Some(prob) = self.probability {
            explanation.push_str(&format!(" (probability: {:.1}%)", prob * 100.0));
        }

        explanation.push_str("\nTop contributing features:");

        for (idx, contrib) in self.top_features(5) {
            let sign = if contrib >= 0.0 { "+" } else { "" };
            explanation.push_str(&format!("\n  - feature[{idx}]: {sign}{contrib:.4}"));
        }

        explanation.push_str(&format!("\nIntercept: {:.4}", self.intercept));
        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        &self.contributions
    }

    fn confidence(&self) -> f32 {
        self.probability.unwrap_or_else(|| {
            // For regression, use inverse of prediction variance as confidence proxy
            1.0 / (1.0 + self.logit.abs())
        })
    }

    fn to_bytes(&self) -> Vec<u8> {
        let n_features = self.contributions.len() as u32;
        let has_prob = self.probability.is_some();

        let mut bytes = Vec::with_capacity(22 + self.contributions.len() * 4);
        bytes.push(1); // version
        bytes.extend_from_slice(&n_features.to_le_bytes());
        bytes.extend_from_slice(&self.intercept.to_le_bytes());
        bytes.extend_from_slice(&self.logit.to_le_bytes());
        bytes.extend_from_slice(&self.prediction.to_le_bytes());
        bytes.push(u8::from(has_prob));

        if let Some(prob) = self.probability {
            bytes.extend_from_slice(&prob.to_le_bytes());
        } else {
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }

        for c in &self.contributions {
            bytes.extend_from_slice(&c.to_le_bytes());
        }

        bytes
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.len() < 22 {
            return Err(PathError::InsufficientData {
                expected: 22,
                actual: bytes.len(),
            });
        }

        let version = bytes[0];
        if version != 1 {
            return Err(PathError::VersionMismatch {
                expected: 1,
                actual: version,
            });
        }

        let n_features = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]) as usize;
        let expected_len = 22 + n_features * 4;

        if bytes.len() < expected_len {
            return Err(PathError::InsufficientData {
                expected: expected_len,
                actual: bytes.len(),
            });
        }

        let intercept = f32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        let logit = f32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]);
        let prediction = f32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]);
        let has_prob = bytes[17] != 0;
        let prob_value = f32::from_le_bytes([bytes[18], bytes[19], bytes[20], bytes[21]]);

        let probability = if has_prob { Some(prob_value) } else { None };

        let mut contributions = Vec::with_capacity(n_features);
        for i in 0..n_features {
            let offset = 22 + i * 4;
            let c = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            contributions.push(c);
        }

        Ok(Self {
            contributions,
            intercept,
            logit,
            prediction,
            probability,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_path_new() {
        let path = LinearPath::new(vec![0.1, 0.2, 0.3], 0.5, 1.0, 0.8);
        assert_eq!(path.contributions.len(), 3);
        assert!((path.intercept - 0.5).abs() < 1e-6);
        assert!((path.logit - 1.0).abs() < 1e-6);
        assert!((path.prediction - 0.8).abs() < 1e-6);
        assert!(path.probability.is_none());
    }

    #[test]
    fn test_linear_path_with_probability() {
        let path = LinearPath::new(vec![0.1], 0.0, 1.0, 0.73).with_probability(0.73);
        assert_eq!(path.probability, Some(0.73));
    }

    #[test]
    fn test_linear_path_top_features() {
        let path = LinearPath::new(vec![0.1, -0.5, 0.3, -0.2, 0.4], 0.0, 1.0, 0.5);
        let top = path.top_features(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 1); // -0.5 has highest absolute value
        assert_eq!(top[1].0, 4); // 0.4
        assert_eq!(top[2].0, 2); // 0.3
    }

    #[test]
    fn test_linear_path_explain() {
        let path = LinearPath::new(vec![0.1, -0.2], 0.5, 1.0, 0.8).with_probability(0.8);
        let explanation = path.explain();
        assert!(explanation.contains("Prediction: 0.8000"));
        assert!(explanation.contains("probability: 80.0%"));
        assert!(explanation.contains("Intercept: 0.5000"));
    }

    #[test]
    fn test_linear_path_feature_contributions() {
        let path = LinearPath::new(vec![0.1, 0.2, 0.3], 0.0, 1.0, 0.5);
        let contributions = path.feature_contributions();
        assert_eq!(contributions.len(), 3);
        assert!((contributions[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_linear_path_confidence_with_probability() {
        let path = LinearPath::new(vec![0.1], 0.0, 1.0, 0.5).with_probability(0.9);
        assert!((path.confidence() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_linear_path_confidence_without_probability() {
        let path = LinearPath::new(vec![0.1], 0.0, 2.0, 0.5);
        assert!((path.confidence() - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_path_serialization_roundtrip() {
        let original = LinearPath::new(vec![0.1, -0.2, 0.3], 0.5, 1.5, 0.8).with_probability(0.75);

        let bytes = original.to_bytes();
        let restored = LinearPath::from_bytes(&bytes).expect("roundtrip");

        assert_eq!(original.contributions.len(), restored.contributions.len());
        for (a, b) in original
            .contributions
            .iter()
            .zip(restored.contributions.iter())
        {
            assert!((a - b).abs() < 1e-6);
        }
        assert!((original.intercept - restored.intercept).abs() < 1e-6);
        assert!((original.logit - restored.logit).abs() < 1e-6);
        assert!((original.prediction - restored.prediction).abs() < 1e-6);
        assert_eq!(original.probability, restored.probability);
    }

    #[test]
    fn test_linear_path_from_bytes_insufficient_data() {
        let bytes = vec![1u8; 10];
        let result = LinearPath::from_bytes(&bytes);
        assert!(matches!(
            result,
            Err(PathError::InsufficientData { expected: 22, .. })
        ));
    }

    #[test]
    fn test_linear_path_from_bytes_version_mismatch() {
        let mut bytes = vec![0u8; 22];
        bytes[0] = 2;
        let result = LinearPath::from_bytes(&bytes);
        assert!(matches!(
            result,
            Err(PathError::VersionMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_linear_path_clone() {
        let path = LinearPath::new(vec![0.1, 0.2], 0.5, 1.0, 0.8);
        let cloned = path.clone();
        assert_eq!(path.contributions.len(), cloned.contributions.len());
        assert!((path.intercept - cloned.intercept).abs() < 1e-6);
    }
}
