#![allow(missing_docs)]
//! BrickTuner - ML-based ComputeBrick Tuner Ensemble
//!
//! Combines throughput regression, kernel classification, and bottleneck analysis.

mod persistence;
mod render;

use serde::{Deserialize, Serialize};

use super::error::TunerError;
use super::features::TunerFeatures;
use super::helpers::chrono_lite_now;
use super::models::{
    BottleneckClassifier, BottleneckPrediction, KernelClassifier, KernelRecommendation,
    ThroughputPrediction, ThroughputRegressor,
};
use super::types::{BottleneckClass, KernelType};

// ============================================================================
// TunerRecommendation
// ============================================================================

/// Combined tuner recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerRecommendation {
    /// Throughput prediction
    pub throughput: ThroughputPrediction,
    /// Kernel recommendation
    pub kernel: KernelRecommendation,
    /// Bottleneck analysis
    pub bottleneck: BottleneckPrediction,
    /// Model version
    pub model_version: String,
    /// Overall confidence
    pub confidence_overall: f32,
    /// Suggested experiments to try
    pub suggested_experiments: Vec<ExperimentSuggestion>,
}

/// Suggested experiment to improve performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentSuggestion {
    /// Increase batch size
    IncreaseBatchSize { from: u32, to: u32 },
    /// Enable CUDA graphs
    EnableCudaGraphs,
    /// Try a specific kernel
    TryKernel { kernel: KernelType },
    /// Reduce sequence length
    ReduceSequenceLength { factor: f32 },
    /// Enable multi-KV cache
    EnableMultiKvCache { count: u32 },
}

impl std::fmt::Display for ExperimentSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExperimentSuggestion::IncreaseBatchSize { from, to } => {
                write!(f, "Increase batch size: M={} → M={}", from, to)
            }
            ExperimentSuggestion::EnableCudaGraphs => {
                write!(f, "Enable CUDA graphs for kernel launch amortization")
            }
            ExperimentSuggestion::TryKernel { kernel } => {
                write!(f, "Try kernel: {:?}", kernel)
            }
            ExperimentSuggestion::ReduceSequenceLength { factor } => {
                write!(f, "Reduce sequence length by {:.0}%", (1.0 - factor) * 100.0)
            }
            ExperimentSuggestion::EnableMultiKvCache { count } => {
                write!(f, "Enable {} separate KV caches for batched attention", count)
            }
        }
    }
}

// ============================================================================
// BrickTuner
// ============================================================================

/// ML-based ComputeBrick tuner ensemble.
///
/// Combines three models for comprehensive recommendations:
/// - ThroughputRegressor: Predicts tok/s
/// - KernelClassifier: Selects best kernel
/// - BottleneckClassifier: Identifies performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrickTuner {
    /// Throughput regression model
    pub(crate) throughput: ThroughputRegressor,
    /// Kernel classification model
    pub(crate) kernel: KernelClassifier,
    /// Bottleneck classification model
    pub(crate) bottleneck: BottleneckClassifier,
    /// Model version
    pub(crate) version: String,
    /// Training timestamp
    pub(crate) trained_at: String,
    /// Number of training samples
    pub(crate) sample_count: usize,
}

impl Default for BrickTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickTuner {
    /// Model version
    pub const VERSION: &'static str = "1.0.0";

    /// Create a new tuner with default models
    pub fn new() -> Self {
        Self {
            throughput: ThroughputRegressor::new(),
            kernel: KernelClassifier::new(),
            bottleneck: BottleneckClassifier::new(),
            version: Self::VERSION.to_string(),
            trained_at: chrono_lite_now(),
            sample_count: 0,
        }
    }

    /// Get the model version string
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Get the throughput regressor's MAPE (Mean Absolute Percentage Error)
    pub fn throughput_mape(&self) -> f32 {
        self.throughput.mape
    }

    /// Get the number of training samples used
    pub fn throughput_sample_count(&self) -> usize {
        self.throughput.sample_count
    }

    /// Get comprehensive tuning recommendation
    pub fn recommend(&self, features: &TunerFeatures) -> TunerRecommendation {
        let throughput = self.throughput.predict(features);
        let kernel = self.kernel.predict(features);
        let bottleneck = self.bottleneck.predict(features);

        // Calculate overall confidence
        let confidence_overall =
            (throughput.confidence + kernel.confidence + bottleneck.confidence) / 3.0;

        // Generate experiment suggestions based on bottleneck
        let suggested_experiments = self.suggest_experiments(features, &bottleneck);

        TunerRecommendation {
            throughput,
            kernel,
            bottleneck,
            model_version: self.version.clone(),
            confidence_overall,
            suggested_experiments,
        }
    }

    /// Suggest experiments based on current bottleneck
    pub fn suggest_experiments(
        &self,
        features: &TunerFeatures,
        bottleneck: &BottleneckPrediction,
    ) -> Vec<ExperimentSuggestion> {
        let mut suggestions = Vec::new();
        let batch_size = (features.batch_size_norm * 64.0).round() as u32;

        match bottleneck.class {
            BottleneckClass::MemoryBound => {
                if batch_size < 8 {
                    suggestions.push(ExperimentSuggestion::IncreaseBatchSize {
                        from: batch_size,
                        to: (batch_size * 2).min(8),
                    });
                }
                suggestions
                    .push(ExperimentSuggestion::TryKernel { kernel: KernelType::BatchedQ4K });
                if batch_size > 1 {
                    suggestions
                        .push(ExperimentSuggestion::EnableMultiKvCache { count: batch_size });
                }
            }
            BottleneckClass::LaunchBound => {
                if features.cuda_graphs < 0.5 {
                    suggestions.push(ExperimentSuggestion::EnableCudaGraphs);
                }
                suggestions
                    .push(ExperimentSuggestion::TryKernel { kernel: KernelType::FusedRmsNormQ4K });
            }
            BottleneckClass::AttentionBound => {
                suggestions
                    .push(ExperimentSuggestion::TryKernel { kernel: KernelType::BatchedAttention });
                suggestions.push(ExperimentSuggestion::ReduceSequenceLength { factor: 0.5 });
            }
            _ => {
                // Default suggestions
                if batch_size < 4 {
                    suggestions
                        .push(ExperimentSuggestion::IncreaseBatchSize { from: batch_size, to: 4 });
                }
            }
        }

        suggestions
    }

    /// Train all models on labeled data
    pub fn train(&mut self, data: &[(TunerFeatures, f32)]) -> Result<(), TunerError> {
        self.throughput.train(data)?;
        self.sample_count = data.len();
        self.trained_at = chrono_lite_now();
        Ok(())
    }
}
