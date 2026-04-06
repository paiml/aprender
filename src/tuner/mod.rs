//! ML-Based ComputeBrick Tuner
//!
//! Implements learned cost models for kernel selection and throughput prediction.
//! See: `docs/specifications/ml-tuner-bricks.md`
//!
//! # Architecture
//!
//! ```text
//! BrickProfiler → FeatureExtractor → TunerModel → Recommendations
//! ```
//!
//! # Scientific Foundations
//!
//! - Chen et al. (2018). "TVM: An Automated End-to-End Optimizing Compiler." OSDI '18.
//! - Williams et al. (2009). "Roofline: An Insightful Visual Performance Model." CACM.
//! - Friedman (2001). "Greedy Function Approximation: A Gradient Boosting Machine."
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::tuner::{BrickTuner, TunerFeatures};
//!
//! let features = TunerFeatures::builder()
//!     .model_params_b(1.5)
//!     .hidden_dim(1536)
//!     .batch_size(4)
//!     .quant_type(QuantType::Q4K)
//!     .build();
//!
//! let tuner = BrickTuner::load_or_default();
//! let recommendation = tuner.recommend(&features);
//! println!("Predicted: {} tok/s", recommendation.throughput.predicted_tps);
//! ```

// Submodules
mod brick_tuner;
mod data_collector;
pub mod error;
mod evolution;
mod features;
pub(crate) mod helpers;
mod models;
pub mod pretrained;
mod types;

// Re-export all public types
pub use brick_tuner::{BrickTuner, ExperimentSuggestion, TunerRecommendation};
pub use data_collector::{
    ConceptDriftStatus, TrainingSample, TrainingStats, TunerDataCollector, UserFeedback,
};
pub use error::TunerError;
pub use evolution::{CalibrationResult, KernelArm, KernelBandit, OnlineLearner};
pub use features::{FeatureExtractor, RunConfig, TunerFeatures, TunerFeaturesBuilder};
pub use models::{
    BottleneckClassifier, BottleneckPrediction, KernelClassifier, KernelRecommendation,
    ThroughputPrediction, ThroughputRegressor,
};
pub use types::{BottleneckClass, KernelType, QuantType};

// Re-export helpers for tests (crate-internal)
#[cfg(test)]
pub(crate) use helpers::{chrono_lite_now, pad_right};

// ============================================================================
// BrickProfiler Integration
// ============================================================================

use crate::brick::BrickProfiler;

impl BrickProfiler {
    /// Get ML-based tuning recommendations.
    ///
    /// Extracts features from current profile and returns recommendations.
    pub fn get_tuner_recommendations(&self, config: &RunConfig) -> Option<TunerRecommendation> {
        if !self.is_enabled() {
            return None;
        }

        // Create feature extractor
        let extractor = FeatureExtractor::new();

        // Extract features
        let features = extractor.extract(self, config);

        // Get recommendation from global tuner
        let tuner = BrickTuner::new();
        Some(tuner.recommend(&features))
    }

    /// Print tuner recommendations to console.
    pub fn print_tuner_recommendations(&self, config: &RunConfig) {
        if let Some(rec) = self.get_tuner_recommendations(config) {
            let tuner = BrickTuner::new();
            tuner.print_recommendation(&rec);
        } else {
            println!("Tuner recommendations not available (profiler disabled)");
        }
    }

    /// Get tokens per second from profiler.
    pub fn tokens_per_sec(&self) -> Option<f32> {
        let total_ns = self.total_ns();
        let total_tokens = self.total_tokens();
        if total_ns == 0 || total_tokens == 0 {
            return None;
        }
        Some(total_tokens as f32 * 1e9 / total_ns as f32)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
