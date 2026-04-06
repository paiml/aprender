#![allow(missing_docs)]
//! Kernel classifier model for ML tuner.

use serde::{Deserialize, Serialize};

#[cfg(feature = "ml-tuner")]
use aprender::{tree::RandomForestClassifier, Matrix};

#[cfg(feature = "ml-tuner")]
use super::super::error::TunerError;
use super::super::features::TunerFeatures;
use super::super::types::KernelType;
use super::KernelRecommendation;

/// Kernel classifier using simple rule-based logic.
///
/// With `ml-tuner` feature: uses aprender::RandomForestClassifier (SHOWCASE-BRICK-001)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KernelClassifier {
    /// Kernel accuracy on validation (for confidence)
    accuracy: f32,
    /// RandomForest classifier when ml-tuner feature is enabled
    #[cfg(feature = "ml-tuner")]
    #[serde(skip)]
    rf_classifier: Option<RandomForestClassifier>,
}

impl KernelClassifier {
    pub fn new() -> Self {
        Self {
            accuracy: 0.85,
            #[cfg(feature = "ml-tuner")]
            rf_classifier: None,
        }
    }

    /// Create a classifier with aprender RandomForest (ml-tuner feature)
    #[cfg(feature = "ml-tuner")]
    pub fn with_random_forest(n_estimators: usize) -> Self {
        Self { accuracy: 0.85, rf_classifier: Some(RandomForestClassifier::new(n_estimators)) }
    }

    /// Train the classifier using aprender RandomForest (ml-tuner feature)
    ///
    /// Labels should be kernel type indices (0=TiledQ4K, 1=CoalescedQ4K, etc.)
    #[cfg(feature = "ml-tuner")]
    pub fn train(&mut self, data: &[(TunerFeatures, u32)]) -> Result<(), TunerError> {
        if data.len() < 10 {
            return Err(TunerError::InsufficientData(data.len()));
        }

        // Convert to aprender format (Matrix<f32> for features, &[usize] for labels)
        let n_samples = data.len();
        let n_features = TunerFeatures::DIM;
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data: Vec<usize> = Vec::with_capacity(n_samples);

        for (features, label) in data {
            x_data.extend(features.to_vector());
            y_data.push(*label as usize);
        }

        let x_matrix = Matrix::from_vec(n_samples, n_features, x_data)
            .map_err(|e| TunerError::TrainingFailed(e.to_string()))?;

        let rf = self.rf_classifier.get_or_insert_with(|| RandomForestClassifier::new(50));
        rf.fit(&x_matrix, &y_data).map_err(|e| TunerError::TrainingFailed(e.to_string()))?;

        // Calculate accuracy on training data
        let predictions = rf.predict(&x_matrix);
        let mut correct = 0;
        for (i, (_, label)) in data.iter().enumerate() {
            if predictions[i] as u32 == *label {
                correct += 1;
            }
        }
        self.accuracy = correct as f32 / data.len().max(1) as f32;

        Ok(())
    }

    /// Predict best kernel based on features
    pub fn predict(&self, features: &TunerFeatures) -> KernelRecommendation {
        // Rule-based kernel selection from SHOWCASE-BRICK-001 learnings
        let batch_size = (features.batch_size_norm * 64.0).round() as u32;
        let seq_len = (2.0_f32.powf(features.seq_len_log * 15.0)).round() as u32;

        // Determine best Q4K variant based on batch size
        let (top_kernel, confidence) = if batch_size >= 4 {
            // M >= 4: Use batched kernels
            (KernelType::BatchedQ4K, 0.90)
        } else if batch_size >= 2 {
            // M = 2-3: Vectorized is good
            (KernelType::VectorizedQ4K, 0.85)
        } else {
            // M = 1: Coalesced or Vectorized
            if features.cuda_graphs > 0.5 {
                (KernelType::VectorizedQ4K, 0.88)
            } else {
                (KernelType::CoalescedQ4K, 0.82)
            }
        };

        // Check for attention-bound cases
        let attention_kernel = if seq_len > 128 {
            KernelType::MultiWarpAttention
        } else {
            KernelType::IncrementalAttention
        };

        // Build alternatives
        let alternatives = vec![
            (KernelType::VectorizedQ4K, 0.85),
            (KernelType::CoalescedQ4K, 0.75),
            (attention_kernel, 0.70),
        ]
        .into_iter()
        .filter(|(k, _)| *k != top_kernel)
        .take(2)
        .collect();

        KernelRecommendation { top_kernel, confidence, alternatives }
    }
}
