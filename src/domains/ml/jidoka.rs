use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{RollingStats, TrainingAnomaly};

// ============================================================================
// Jidoka ML Feedback Loop
// ============================================================================

/// Rule patch types for Kaizen improvements.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RuleType {
    /// Clip gradients to max norm.
    GradientClipping,
    /// Reduce learning rate.
    LearningRateDecay,
    /// Add warmup steps.
    LearningRateWarmup,
    /// Increase batch size.
    BatchSizeIncrease,
    /// Add regularization.
    Regularization,
    /// Manual review required.
    ManualReview,
}

/// Improvement patch generated from anomaly pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePatch {
    /// Type of rule patch.
    pub rule_type: RuleType,
    /// Parameters for the patch.
    pub parameters: HashMap<String, String>,
}

impl Default for RulePatch {
    fn default() -> Self {
        Self {
            rule_type: RuleType::ManualReview,
            parameters: HashMap::new(),
        }
    }
}

/// Anomaly pattern classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Loss became NaN/Inf.
    NonFiniteLoss,
    /// Gradient norm exceeded threshold.
    GradientExplosion,
    /// Gradient vanished below threshold.
    GradientVanishing,
    /// Loss spike (statistical outlier).
    LossSpike,
    /// Prediction confidence below threshold.
    LowConfidence,
    /// Model output inconsistent with oracle.
    OracleMismatch,
}

/// Pattern detected during training/inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyPattern {
    /// Pattern type.
    pub pattern_type: AnomalyType,
    /// Frequency of occurrence.
    pub frequency: u64,
    /// Context information.
    pub context: HashMap<String, String>,
    /// Suggested fix description.
    pub suggested_fix: Option<String>,
}

/// Jidoka feedback loop for ML simulation.
///
/// Each detected anomaly generates improvement patches (Kaizen).
pub struct JidokaMLFeedback {
    /// Anomaly patterns detected.
    patterns: Vec<AnomalyPattern>,
    /// Generated fixes (rule patches).
    patches: Vec<RulePatch>,
    /// Rolling stats for anomaly rate tracking.
    anomaly_rate: RollingStats,
    /// Threshold for auto-patch generation.
    auto_patch_threshold: u64,
}

impl Default for JidokaMLFeedback {
    fn default() -> Self {
        Self::new()
    }
}

impl JidokaMLFeedback {
    /// Create new Jidoka feedback loop.
    #[must_use]
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            patches: Vec::new(),
            anomaly_rate: RollingStats::new(100),
            auto_patch_threshold: 3,
        }
    }

    /// Set threshold for automatic patch generation.
    #[must_use]
    pub fn with_auto_patch_threshold(mut self, threshold: u64) -> Self {
        self.auto_patch_threshold = threshold;
        self
    }

    /// Record anomaly and potentially generate improvement patch.
    pub fn record_anomaly(&mut self, anomaly: TrainingAnomaly) -> Option<RulePatch> {
        let pattern_type = self.classify_type(&anomaly);
        self.anomaly_rate.update(1.0);

        // Check if we've seen this pattern before
        let existing_idx = self
            .patterns
            .iter()
            .position(|p| p.pattern_type == pattern_type);

        if let Some(idx) = existing_idx {
            self.patterns[idx].frequency += 1;

            // After threshold occurrences, generate automated fix
            if self.patterns[idx].frequency >= self.auto_patch_threshold {
                let pattern_clone = self.patterns[idx].clone();
                let patch = self.generate_patch(&pattern_clone);
                self.patches.push(patch.clone());
                return Some(patch);
            }
        } else {
            let suggested_fix = Some(self.suggest_fix(pattern_type));
            let new_pattern = AnomalyPattern {
                pattern_type,
                frequency: 1,
                context: HashMap::new(),
                suggested_fix,
            };

            // Check if threshold is 1 (generate patch on first occurrence)
            if self.auto_patch_threshold <= 1 {
                let patch = self.generate_patch(&new_pattern);
                self.patterns.push(new_pattern);
                self.patches.push(patch.clone());
                return Some(patch);
            }

            self.patterns.push(new_pattern);
        }

        None
    }

    /// Classify anomaly into pattern type.
    #[allow(clippy::unused_self)]
    fn classify_type(&self, anomaly: &TrainingAnomaly) -> AnomalyType {
        match anomaly {
            TrainingAnomaly::NonFiniteLoss => AnomalyType::NonFiniteLoss,
            TrainingAnomaly::GradientExplosion { .. } => AnomalyType::GradientExplosion,
            TrainingAnomaly::GradientVanishing { .. } => AnomalyType::GradientVanishing,
            TrainingAnomaly::LossSpike { .. } => AnomalyType::LossSpike,
            TrainingAnomaly::LowConfidence { .. } => AnomalyType::LowConfidence,
        }
    }

    /// Suggest fix for pattern type.
    #[allow(clippy::unused_self)]
    fn suggest_fix(&self, pattern_type: AnomalyType) -> String {
        match pattern_type {
            AnomalyType::GradientExplosion => "Apply gradient clipping with max_norm=1.0",
            AnomalyType::GradientVanishing => "Use skip connections or residual architecture",
            AnomalyType::LossSpike => "Reduce learning rate or add warmup",
            AnomalyType::NonFiniteLoss => "Check for numerical stability issues",
            AnomalyType::LowConfidence => "Increase model capacity or training data",
            AnomalyType::OracleMismatch => "Review training data distribution",
        }
        .to_string()
    }

    /// Generate rule patch from pattern (Kaizen improvement).
    #[allow(clippy::unused_self)]
    fn generate_patch(&self, pattern: &AnomalyPattern) -> RulePatch {
        let mut params = HashMap::new();

        match pattern.pattern_type {
            AnomalyType::GradientExplosion => {
                params.insert("max_norm".to_string(), "1.0".to_string());
                RulePatch {
                    rule_type: RuleType::GradientClipping,
                    parameters: params,
                }
            }
            AnomalyType::LossSpike => {
                params.insert("warmup_steps".to_string(), "1000".to_string());
                RulePatch {
                    rule_type: RuleType::LearningRateWarmup,
                    parameters: params,
                }
            }
            AnomalyType::GradientVanishing => {
                params.insert("factor".to_string(), "10.0".to_string());
                RulePatch {
                    rule_type: RuleType::LearningRateDecay,
                    parameters: params,
                }
            }
            _ => RulePatch::default(),
        }
    }

    /// Get all detected patterns.
    #[must_use]
    pub fn patterns(&self) -> &[AnomalyPattern] {
        &self.patterns
    }

    /// Get all generated patches.
    #[must_use]
    pub fn patches(&self) -> &[RulePatch] {
        &self.patches
    }

    /// Get current anomaly rate.
    #[must_use]
    pub fn anomaly_rate(&self) -> f64 {
        self.anomaly_rate.mean()
    }

    /// Reset feedback loop state.
    pub fn reset(&mut self) {
        self.patterns.clear();
        self.patches.clear();
        self.anomaly_rate.reset();
    }
}
