#![allow(missing_docs)]
//! Kernel classifier model for ML tuner.

use serde::{Deserialize, Serialize};

use super::super::features::TunerFeatures;
use super::super::types::KernelType;
use super::KernelRecommendation;

/// Kernel classifier using simple rule-based logic.
///
/// The `ml-tuner` showcase (SHOWCASE-BRICK-001) trains an
/// `aprender::RandomForestClassifier` on these features in
/// `examples/ml_tuner_demo.rs` — aprender is a dev-dependency so the SIMD foundation
/// stays free of the ML layer (no compute→core layer inversion).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KernelClassifier {
    /// Kernel accuracy on validation (for confidence)
    accuracy: f32,
}

impl KernelClassifier {
    pub fn new() -> Self {
        Self { accuracy: 0.85 }
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
