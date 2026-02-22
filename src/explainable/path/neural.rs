//! Neural network decision path (gradient-based)

use serde::{Deserialize, Serialize};

use super::traits::{ByteReader, DecisionPath, PathError};

/// Decision path for neural networks (gradient-based)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuralPath {
    /// Input gradient (saliency map)
    pub input_gradient: Vec<f32>,
    /// Layer activations (optional, feature-gated for memory)
    pub activations: Option<Vec<Vec<f32>>>,
    /// Attention weights (for transformers)
    pub attention_weights: Option<Vec<Vec<f32>>>,
    /// Integrated gradients attribution
    pub integrated_gradients: Option<Vec<f32>>,
    /// Final prediction
    pub prediction: f32,
    /// Confidence (softmax probability)
    pub confidence: f32,
}

impl NeuralPath {
    /// Create a new neural path
    pub fn new(input_gradient: Vec<f32>, prediction: f32, confidence: f32) -> Self {
        Self {
            input_gradient,
            activations: None,
            attention_weights: None,
            integrated_gradients: None,
            prediction,
            confidence,
        }
    }

    /// Set layer activations
    pub fn with_activations(mut self, activations: Vec<Vec<f32>>) -> Self {
        self.activations = Some(activations);
        self
    }

    /// Set attention weights
    pub fn with_attention(mut self, attention: Vec<Vec<f32>>) -> Self {
        self.attention_weights = Some(attention);
        self
    }

    /// Set integrated gradients
    pub fn with_integrated_gradients(mut self, ig: Vec<f32>) -> Self {
        self.integrated_gradients = Some(ig);
        self
    }

    /// Get top salient features by absolute gradient
    pub fn top_salient_features(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .input_gradient
            .iter()
            .enumerate()
            .map(|(i, &g)| (i, g))
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

impl DecisionPath for NeuralPath {
    fn explain(&self) -> String {
        let mut explanation = format!(
            "Neural Network Prediction: {:.4} (confidence: {:.1}%)\n",
            self.prediction,
            self.confidence * 100.0
        );

        explanation.push_str("\nTop salient input features (by gradient):\n");
        for (idx, grad) in self.top_salient_features(5) {
            let sign = if grad >= 0.0 { "+" } else { "" };
            explanation.push_str(&format!("  input[{idx}]: {sign}{grad:.6}\n"));
        }

        if let Some(ig) = &self.integrated_gradients {
            explanation.push_str("\nIntegrated gradients available (");
            let len = ig.len();
            explanation.push_str(&format!("{len} features)\n"));
        }

        if self.attention_weights.is_some() {
            explanation.push_str("\nAttention weights available\n");
        }

        explanation
    }

    fn feature_contributions(&self) -> &[f32] {
        self.integrated_gradients
            .as_deref()
            .unwrap_or(&self.input_gradient)
    }

    fn confidence(&self) -> f32 {
        self.confidence
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1); // version

        // Input gradient
        bytes.extend_from_slice(&(self.input_gradient.len() as u32).to_le_bytes());
        for g in &self.input_gradient {
            bytes.extend_from_slice(&g.to_le_bytes());
        }

        // Prediction and confidence
        bytes.extend_from_slice(&self.prediction.to_le_bytes());
        bytes.extend_from_slice(&self.confidence.to_le_bytes());

        // Activations
        let has_activations = self.activations.is_some();
        bytes.push(u8::from(has_activations));
        if let Some(activations) = &self.activations {
            bytes.extend_from_slice(&(activations.len() as u32).to_le_bytes());
            for layer in activations {
                bytes.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                for a in layer {
                    bytes.extend_from_slice(&a.to_le_bytes());
                }
            }
        }

        // Attention weights
        let has_attention = self.attention_weights.is_some();
        bytes.push(u8::from(has_attention));
        if let Some(attention) = &self.attention_weights {
            bytes.extend_from_slice(&(attention.len() as u32).to_le_bytes());
            for layer in attention {
                bytes.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                for a in layer {
                    bytes.extend_from_slice(&a.to_le_bytes());
                }
            }
        }

        // Integrated gradients
        let has_ig = self.integrated_gradients.is_some();
        bytes.push(u8::from(has_ig));
        if let Some(ig) = &self.integrated_gradients {
            bytes.extend_from_slice(&(ig.len() as u32).to_le_bytes());
            for g in ig {
                bytes.extend_from_slice(&g.to_le_bytes());
            }
        }

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

        let input_gradient = reader.read_f32_vec()?;
        let prediction = reader.read_f32()?;
        let confidence = reader.read_f32()?;
        let activations = reader.read_optional(ByteReader::read_nested_f32_vecs)?;
        let attention_weights = reader.read_optional(ByteReader::read_nested_f32_vecs)?;
        let integrated_gradients = reader.read_optional(ByteReader::read_f32_vec)?;

        Ok(Self {
            input_gradient,
            activations,
            attention_weights,
            integrated_gradients,
            prediction,
            confidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_path_new() {
        let path = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.87, 0.92);
        assert_eq!(path.input_gradient.len(), 3);
        assert_eq!(path.prediction, 0.87);
        assert_eq!(path.confidence, 0.92);
    }

    #[test]
    fn test_neural_path_top_salient() {
        let path = NeuralPath::new(vec![0.1, -0.5, 0.3], 0.0, 0.0);
        let top = path.top_salient_features(2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 2);
    }

    #[test]
    fn test_neural_path_serialization_roundtrip() {
        let path = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.87, 0.92)
            .with_activations(vec![vec![0.5, 0.6], vec![0.7, 0.8]])
            .with_attention(vec![vec![0.1, 0.9]])
            .with_integrated_gradients(vec![0.15, -0.25, 0.35]);

        let bytes = path.to_bytes();
        let restored = NeuralPath::from_bytes(&bytes).expect("Failed to deserialize");

        assert_eq!(path.input_gradient.len(), restored.input_gradient.len());
        assert!((path.prediction - restored.prediction).abs() < 1e-6);
        assert!((path.confidence - restored.confidence).abs() < 1e-6);
        assert!(restored.activations.is_some());
        assert!(restored.attention_weights.is_some());
        assert!(restored.integrated_gradients.is_some());
    }

    #[test]
    fn test_neural_path_feature_contributions() {
        let path = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.0, 0.0);
        assert_eq!(path.feature_contributions(), &[0.1, -0.2, 0.3]);

        let path_with_ig = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.0, 0.0)
            .with_integrated_gradients(vec![0.5, 0.5]);
        assert_eq!(path_with_ig.feature_contributions(), &[0.5, 0.5]);
    }

    #[test]
    fn test_neural_path_invalid_version() {
        let result = NeuralPath::from_bytes(&[2u8, 0, 0, 0, 0]);
        assert!(matches!(result, Err(PathError::VersionMismatch { .. })));
    }

    #[test]
    fn test_neural_path_insufficient_data() {
        let result = NeuralPath::from_bytes(&[1u8, 0, 0]);
        assert!(matches!(result, Err(PathError::InsufficientData { .. })));
    }

    #[test]
    fn test_neural_path_explain_with_ig() {
        let path =
            NeuralPath::new(vec![0.1], 0.5, 0.9).with_integrated_gradients(vec![0.2, 0.3, 0.5]);
        let explanation = path.explain();
        assert!(explanation.contains("Integrated gradients"));
        assert!(explanation.contains("3 features"));
    }

    #[test]
    fn test_neural_path_explain_with_attention() {
        let path = NeuralPath::new(vec![0.1], 0.5, 0.9).with_attention(vec![vec![0.5, 0.5]]);
        let explanation = path.explain();
        assert!(explanation.contains("Attention weights"));
    }

    #[test]
    fn test_neural_path_serialization_minimal() {
        let path = NeuralPath::new(vec![0.1, 0.2], 0.5, 0.9);
        let bytes = path.to_bytes();
        let restored = NeuralPath::from_bytes(&bytes).expect("Failed to deserialize");
        assert!(restored.activations.is_none());
        assert!(restored.attention_weights.is_none());
        assert!(restored.integrated_gradients.is_none());
    }

    #[test]
    fn test_neural_path_with_activations() {
        let path = NeuralPath::new(vec![0.1], 0.5, 0.9)
            .with_activations(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(path.activations.is_some());
        let activations = path.activations.expect("has activations");
        assert_eq!(activations.len(), 2);
        assert_eq!(activations[0], vec![1.0, 2.0]);
        assert_eq!(activations[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_neural_path_confidence_method() {
        let path = NeuralPath::new(vec![0.1], 0.5, 0.85);
        assert_eq!(DecisionPath::confidence(&path), 0.85);
    }

    #[test]
    fn test_neural_path_explain_basic() {
        let path = NeuralPath::new(vec![0.1, -0.2, 0.3], 0.75, 0.90);
        let explanation = path.explain();
        assert!(explanation.contains("Neural Network Prediction"));
        assert!(explanation.contains("0.75"));
        assert!(explanation.contains("90.0%"));
        assert!(explanation.contains("Top salient input features"));
    }

    #[test]
    fn test_neural_path_top_salient_features_empty() {
        let path = NeuralPath::new(vec![], 0.5, 0.9);
        let top = path.top_salient_features(5);
        assert!(top.is_empty());
    }

    #[test]
    fn test_neural_path_top_salient_features_more_than_available() {
        let path = NeuralPath::new(vec![0.1, 0.2], 0.5, 0.9);
        let top = path.top_salient_features(10);
        assert_eq!(top.len(), 2);
    }
}
