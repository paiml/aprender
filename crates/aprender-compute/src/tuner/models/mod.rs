#![allow(missing_docs)]
//! ML Models for Tuner
//!
//! Throughput regressor, kernel classifier, and bottleneck classifier implementations.

mod kernel;
mod throughput;

pub use kernel::KernelClassifier;
pub use throughput::ThroughputRegressor;

use serde::{Deserialize, Serialize};

use super::features::TunerFeatures;
use super::types::{BottleneckClass, KernelType};

// ============================================================================
// Prediction Results
// ============================================================================

/// Throughput prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputPrediction {
    /// Predicted tokens per second
    pub predicted_tps: f32,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Top contributing features
    pub top_features: Vec<(String, f32)>,
}

/// Kernel recommendation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelRecommendation {
    /// Top recommended kernel
    pub top_kernel: KernelType,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Alternative kernels with probabilities
    pub alternatives: Vec<(KernelType, f32)>,
}

/// Bottleneck prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckPrediction {
    /// Predicted bottleneck class
    pub class: BottleneckClass,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Human-readable explanation
    pub explanation: String,
    /// Recommended action
    pub recommended_action: String,
}

// ============================================================================
// BottleneckClassifier
// ============================================================================

/// Bottleneck classifier using heuristics from profiler data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BottleneckClassifier {
    /// Classification accuracy
    accuracy: f32,
}

impl BottleneckClassifier {
    pub fn new() -> Self {
        Self { accuracy: 0.90 }
    }

    /// Predict bottleneck from features
    pub fn predict(&self, features: &TunerFeatures) -> BottleneckPrediction {
        // Use already-computed bottleneck if available
        if let Some(class) = features.bottleneck_class {
            return BottleneckPrediction {
                class,
                confidence: 0.95,
                explanation: format!("Bottleneck classified from profiler data: {}", class),
                recommended_action: class.recommended_action().to_string(),
            };
        }

        // Heuristic classification based on features
        let batch_size = (features.batch_size_norm * 64.0).round() as u32;
        let seq_len = (2.0_f32.powf(features.seq_len_log * 15.0)).round() as u32;

        let (class, confidence, explanation) = if batch_size == 1 && features.cuda_graphs < 0.5 {
            (
                BottleneckClass::LaunchBound,
                0.75,
                "Single sequence without CUDA graphs: kernel launch overhead may dominate".into(),
            )
        } else if seq_len > 512 {
            (
                BottleneckClass::AttentionBound,
                0.80,
                format!("Long sequence (len={}) likely makes attention the bottleneck", seq_len),
            )
        } else {
            (
                BottleneckClass::MemoryBound,
                0.85,
                "Q4K GEMV is typically memory-bound for LLM inference".into(),
            )
        };

        BottleneckPrediction {
            class,
            confidence,
            explanation,
            recommended_action: class.recommended_action().to_string(),
        }
    }
}
