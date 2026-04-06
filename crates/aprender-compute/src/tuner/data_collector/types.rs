//! Data types for training data collection.
//!
//! Contains `TrainingSample`, `UserFeedback`, `ConceptDriftStatus`, and `TrainingStats`.

use serde::{Deserialize, Serialize};

use crate::tuner::features::TunerFeatures;
use crate::tuner::types::{BottleneckClass, KernelType};

// ============================================================================
// TrainingSample
// ============================================================================

/// Training sample for the tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Features
    pub features: TunerFeatures,
    /// Measured throughput (label)
    pub throughput_tps: f32,
    /// Best kernel (label)
    pub best_kernel: KernelType,
    /// Bottleneck class (label)
    pub bottleneck: BottleneckClass,
    /// Timestamp
    pub timestamp: String,
    /// Hardware ID
    pub hardware_id: String,
}

// ============================================================================
// UserFeedback
// ============================================================================

/// User feedback on a recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum UserFeedback {
    /// User accepted the recommendation
    Accepted,
    /// User rejected the recommendation
    Rejected,
    /// User provided alternative (overrode recommendation)
    Alternative,
    /// No feedback (default)
    #[default]
    None,
}

// ============================================================================
// ConceptDriftStatus
// ============================================================================

/// Concept drift detection result
#[derive(Debug, Clone)]
pub struct ConceptDriftStatus {
    /// Whether drift has been detected
    pub drift_detected: bool,
    /// Estimated model staleness (0.0 = fresh, 1.0 = very stale)
    pub staleness_score: f32,
    /// Number of samples since last training
    pub samples_since_training: usize,
    /// Recommendation: should retrain?
    pub recommend_retrain: bool,
    /// Explanation of drift status
    pub explanation: String,
}

// ============================================================================
// TrainingStats
// ============================================================================

/// Training statistics summary
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Total samples collected
    pub total_samples: usize,
    /// Samples since last training
    pub samples_since_training: usize,
    /// Accepted recommendations count
    pub accepted_count: usize,
    /// Rejected recommendations count
    pub rejected_count: usize,
    /// Alternative provided count
    pub alternative_count: usize,
    /// Staleness score (0.0 = fresh, 1.0 = stale)
    pub staleness_score: f32,
    /// Whether concept drift was detected
    pub drift_detected: bool,
    /// Whether online learning is enabled
    pub online_learning_enabled: bool,
}
