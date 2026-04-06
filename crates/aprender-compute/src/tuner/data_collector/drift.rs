//! Online learning and concept drift detection (T-TUNER-005, GitHub #82).
//!
//! Tracks prediction errors via a sliding window to detect model staleness
//! and trigger auto-retraining when concept drift is observed.

use crate::tuner::brick_tuner::BrickTuner;
use crate::tuner::features::TunerFeatures;

use super::types::{ConceptDriftStatus, TrainingStats, UserFeedback};
use super::TunerDataCollector;

impl TunerDataCollector {
    // ========================================================================
    // T-TUNER-005: Online Learning (GitHub #82)
    // ========================================================================

    /// Record user feedback on a recommendation
    pub fn record_feedback(&mut self, sample_index: usize, feedback: UserFeedback) {
        // Extend feedback vector if needed
        while self.feedback.len() <= sample_index {
            self.feedback.push(UserFeedback::None);
        }
        self.feedback[sample_index] = feedback;
    }

    /// Get feedback for a sample
    pub fn get_feedback(&self, sample_index: usize) -> UserFeedback {
        self.feedback.get(sample_index).copied().unwrap_or(UserFeedback::None)
    }

    /// Record prediction error for concept drift detection
    pub fn record_prediction_error(&mut self, predicted: f32, actual: f32) {
        if !self.online_learning_enabled {
            return;
        }

        // Compute relative error (0.0 = perfect, 1.0 = 100% error)
        let error = if actual > 0.0 { ((predicted - actual) / actual).abs().min(1.0) } else { 1.0 };

        // Add to sliding window
        self.error_window.push(error);

        // Trim window to max size
        if self.error_window.len() > self.error_window_size {
            self.error_window.remove(0);
        }
    }

    /// Detect concept drift based on prediction error trends
    pub fn detect_concept_drift(&self) -> ConceptDriftStatus {
        let samples_since_training = self.samples.len().saturating_sub(self.samples_at_last_train);

        // Not enough data for drift detection
        if self.error_window.len() < 10 {
            return ConceptDriftStatus {
                drift_detected: false,
                staleness_score: 0.0,
                samples_since_training,
                recommend_retrain: false,
                explanation: "Insufficient data for drift detection".to_string(),
            };
        }

        // Compute mean error
        let mean_error: f32 =
            self.error_window.iter().sum::<f32>() / self.error_window.len().max(1) as f32;

        // Compute staleness score (0.0 = fresh, 1.0 = stale)
        let staleness_score =
            (samples_since_training as f32 / Self::STALENESS_THRESHOLD as f32).min(1.0);

        // Detect drift
        let drift_detected = mean_error > Self::DRIFT_ERROR_THRESHOLD;

        // Recommend retrain if drift detected OR stale
        let recommend_retrain = drift_detected || staleness_score > 0.8;

        let explanation = if drift_detected {
            format!(
                "Concept drift detected: mean error {:.1}% exceeds threshold {:.1}%",
                mean_error * 100.0,
                Self::DRIFT_ERROR_THRESHOLD * 100.0
            )
        } else if staleness_score > 0.8 {
            format!(
                "Model stale: {} samples since last training (threshold: {})",
                samples_since_training,
                Self::STALENESS_THRESHOLD
            )
        } else {
            format!(
                "Model fresh: mean error {:.1}%, {} samples since training",
                mean_error * 100.0,
                samples_since_training
            )
        };

        ConceptDriftStatus {
            drift_detected,
            staleness_score,
            samples_since_training,
            recommend_retrain,
            explanation,
        }
    }

    /// Check if auto-retrain should trigger
    pub fn should_retrain(&self) -> bool {
        if !self.online_learning_enabled {
            return false;
        }

        let samples_since = self.samples.len().saturating_sub(self.samples_at_last_train);

        // Retrain if we have enough new samples
        if samples_since >= self.retrain_threshold {
            return true;
        }

        // Or if concept drift is detected
        let drift = self.detect_concept_drift();
        drift.recommend_retrain && samples_since >= 10
    }

    /// Mark that training occurred (resets drift counters)
    pub fn mark_trained(&mut self) {
        self.samples_at_last_train = self.samples.len();
        self.error_window.clear();
    }

    /// Get training statistics
    pub fn training_stats(&self) -> TrainingStats {
        let drift = self.detect_concept_drift();

        // Count feedback types
        let accepted_count = self.feedback.iter().filter(|f| **f == UserFeedback::Accepted).count();
        let rejected_count = self.feedback.iter().filter(|f| **f == UserFeedback::Rejected).count();
        let alternative_count =
            self.feedback.iter().filter(|f| **f == UserFeedback::Alternative).count();

        TrainingStats {
            total_samples: self.samples.len(),
            samples_since_training: drift.samples_since_training,
            accepted_count,
            rejected_count,
            alternative_count,
            staleness_score: drift.staleness_score,
            drift_detected: drift.drift_detected,
            online_learning_enabled: self.online_learning_enabled,
        }
    }

    /// Auto-retrain and update BrickTuner if conditions are met
    pub fn auto_retrain(&mut self, tuner: &mut BrickTuner) -> bool {
        if !self.should_retrain() {
            return false;
        }

        // Weight samples by feedback
        let training_data = self.prepare_weighted_training_data();

        if training_data.len() < 10 {
            return false;
        }

        // Train and update
        match tuner.train(&training_data) {
            Ok(()) => {
                self.mark_trained();
                true
            }
            Err(_) => false,
        }
    }

    /// Prepare training data with feedback weighting
    pub(super) fn prepare_weighted_training_data(&self) -> Vec<(TunerFeatures, f32)> {
        self.samples
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let feedback = self.get_feedback(i);

                // Skip rejected samples (they had bad throughput measurements)
                if feedback == UserFeedback::Rejected {
                    return None;
                }

                // Weight accepted samples higher (duplicate them)
                let weight = match feedback {
                    UserFeedback::Accepted => 2,
                    UserFeedback::Alternative | UserFeedback::Rejected | UserFeedback::None => 1,
                };

                Some((0..weight).map(|_| (s.features.clone(), s.throughput_tps)))
            })
            .flatten()
            .collect()
    }
}
