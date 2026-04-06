//! Adaptive curriculum scheduler for error-specific training

use std::collections::HashMap;

use super::CurriculumScheduler;

/// Adaptive curriculum that adjusts based on error class performance
///
/// Tracks accuracy per error class and increases difficulty for
/// well-learned classes while maintaining focus on struggling classes.
///
/// Supports the CITL adaptive tier selection pattern.
#[derive(Debug, Clone)]
pub struct AdaptiveCurriculum {
    /// Accuracy per error class
    pub(crate) class_accuracy: HashMap<String, f32>,
    /// Attempts per error class
    pub(crate) class_attempts: HashMap<String, usize>,
    /// Default tier for unknown errors
    default_tier: usize,
    /// Overall difficulty based on mean accuracy
    overall_difficulty: f32,
}

impl AdaptiveCurriculum {
    /// Create new adaptive curriculum
    pub fn new() -> Self {
        Self {
            class_accuracy: HashMap::new(),
            class_attempts: HashMap::new(),
            default_tier: 1,
            overall_difficulty: 0.0,
        }
    }

    /// Get recommended tier for an error class
    ///
    /// Based on the CITL `select_tier()` pattern
    pub fn tier_for_error(&self, error_code: &str, attempt: usize) -> usize {
        // Special cases
        if error_code.starts_with("ICE") {
            return 4; // ICEs always need full debug
        }

        // Type/trait errors benefit from traces
        if matches!(error_code, "E0308" | "E0277" | "E0382") && attempt >= 1 {
            return 3;
        }

        // Name resolution needs verbose
        if matches!(error_code, "E0425" | "E0433") && attempt >= 2 {
            return 3;
        }

        // Default escalation pattern
        match attempt {
            0 => self.default_tier,
            1 => 2,
            2.. => 3,
        }
    }

    /// Update accuracy for an error class
    pub fn update_class(&mut self, error_code: &str, correct: bool) {
        let attempts = self.class_attempts.entry(error_code.to_string()).or_insert(0);
        *attempts += 1;

        let acc = self.class_accuracy.entry(error_code.to_string()).or_insert(0.0);
        // Exponential moving average
        let alpha = 0.1;
        *acc = *acc * (1.0 - alpha) + if correct { alpha } else { 0.0 };

        // Update overall difficulty
        if !self.class_accuracy.is_empty() {
            self.overall_difficulty =
                self.class_accuracy.values().sum::<f32>() / self.class_accuracy.len() as f32;
        }
    }

    /// Get sample weight based on class rarity and accuracy
    ///
    /// Long-tail (rare) errors get higher weights per Feldman (2020)
    pub fn weight_for_class(&self, error_code: &str) -> f32 {
        let attempts = *self.class_attempts.get(error_code).unwrap_or(&0);
        let accuracy = *self.class_accuracy.get(error_code).unwrap_or(&0.0);

        // Rare classes get higher weight
        let rarity_weight = 1.0 / (attempts as f32 + 1.0).sqrt();

        // Low accuracy classes get higher weight
        let difficulty_weight = 1.0 - accuracy;

        // Combine weights (normalize to reasonable range)
        (1.0 + rarity_weight + difficulty_weight).min(3.0)
    }
}

impl Default for AdaptiveCurriculum {
    fn default() -> Self {
        Self::new()
    }
}

impl CurriculumScheduler for AdaptiveCurriculum {
    fn difficulty(&self) -> f32 {
        self.overall_difficulty
    }

    fn tier(&self) -> usize {
        if self.overall_difficulty < 0.25 {
            1
        } else if self.overall_difficulty < 0.5 {
            2
        } else if self.overall_difficulty < 0.75 {
            3
        } else {
            4
        }
    }

    fn step(&mut self, _epoch: usize, accuracy: f32) {
        // Update overall difficulty based on recent accuracy
        let alpha = 0.1;
        self.overall_difficulty = self.overall_difficulty * (1.0 - alpha) + accuracy * alpha;
    }

    fn reset(&mut self) {
        self.class_accuracy.clear();
        self.class_attempts.clear();
        self.overall_difficulty = 0.0;
    }

    fn name(&self) -> &'static str {
        "AdaptiveCurriculum"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_curriculum_new() {
        let curriculum = AdaptiveCurriculum::new();
        assert!(curriculum.class_accuracy.is_empty());
        assert!(curriculum.class_attempts.is_empty());
        assert_eq!(curriculum.overall_difficulty, 0.0);
    }

    #[test]
    fn test_adaptive_curriculum_default() {
        let curriculum = AdaptiveCurriculum::default();
        assert_eq!(curriculum.difficulty(), 0.0);
    }

    #[test]
    fn test_tier_for_error_ice() {
        let curriculum = AdaptiveCurriculum::new();
        assert_eq!(curriculum.tier_for_error("ICE001", 0), 4);
        assert_eq!(curriculum.tier_for_error("ICE-crash", 5), 4);
    }

    #[test]
    fn test_tier_for_error_type_errors() {
        let curriculum = AdaptiveCurriculum::new();
        // E0308 on attempt 0 uses default
        assert_eq!(curriculum.tier_for_error("E0308", 0), 1);
        // E0308 on attempt 1+ gets tier 3
        assert_eq!(curriculum.tier_for_error("E0308", 1), 3);
        assert_eq!(curriculum.tier_for_error("E0277", 2), 3);
        assert_eq!(curriculum.tier_for_error("E0382", 1), 3);
    }

    #[test]
    fn test_tier_for_error_name_resolution() {
        let curriculum = AdaptiveCurriculum::new();
        // E0425 needs attempt >= 2 for tier 3
        assert_eq!(curriculum.tier_for_error("E0425", 0), 1);
        assert_eq!(curriculum.tier_for_error("E0425", 1), 2);
        assert_eq!(curriculum.tier_for_error("E0425", 2), 3);
        assert_eq!(curriculum.tier_for_error("E0433", 3), 3);
    }

    #[test]
    fn test_tier_for_error_default_escalation() {
        let curriculum = AdaptiveCurriculum::new();
        // Generic error escalation
        assert_eq!(curriculum.tier_for_error("E0001", 0), 1);
        assert_eq!(curriculum.tier_for_error("E0001", 1), 2);
        assert_eq!(curriculum.tier_for_error("E0001", 2), 3);
        assert_eq!(curriculum.tier_for_error("E0001", 5), 3);
    }

    #[test]
    fn test_update_class() {
        let mut curriculum = AdaptiveCurriculum::new();

        curriculum.update_class("E0308", true);
        assert_eq!(curriculum.class_attempts.get("E0308"), Some(&1));
        // First correct: 0.0 * 0.9 + 0.1 = 0.1
        assert!(
            (curriculum.class_accuracy.get("E0308").expect("key should exist") - 0.1).abs() < 0.001
        );

        curriculum.update_class("E0308", false);
        assert_eq!(curriculum.class_attempts.get("E0308"), Some(&2));
        // 0.1 * 0.9 + 0 = 0.09
        assert!(
            (curriculum.class_accuracy.get("E0308").expect("key should exist") - 0.09).abs()
                < 0.001
        );
    }

    #[test]
    fn test_weight_for_class_unknown() {
        let curriculum = AdaptiveCurriculum::new();
        let weight = curriculum.weight_for_class("unknown");
        // rarity = 1/sqrt(1) = 1.0, difficulty = 1.0 - 0 = 1.0
        // weight = 1.0 + 1.0 + 1.0 = 3.0
        assert!((weight - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_for_class_known() {
        let mut curriculum = AdaptiveCurriculum::new();

        // Add some attempts
        for _ in 0..10 {
            curriculum.update_class("E0308", true);
        }

        let weight = curriculum.weight_for_class("E0308");
        // rarity = 1/sqrt(11) ≈ 0.3, difficulty = 1 - accuracy
        // Should be lower than unknown
        assert!(weight < 3.0);
        assert!(weight >= 1.0);
    }

    #[test]
    fn test_curriculum_scheduler_difficulty() {
        let mut curriculum = AdaptiveCurriculum::new();
        assert_eq!(curriculum.difficulty(), 0.0);

        curriculum.step(0, 0.5);
        assert!(curriculum.difficulty() > 0.0);
    }

    #[test]
    fn test_curriculum_scheduler_tier() {
        let mut curriculum = AdaptiveCurriculum::new();

        // Tier 1 for low difficulty
        assert_eq!(curriculum.tier(), 1);

        // Tier 2 for 0.25-0.5
        curriculum.overall_difficulty = 0.3;
        assert_eq!(curriculum.tier(), 2);

        // Tier 3 for 0.5-0.75
        curriculum.overall_difficulty = 0.6;
        assert_eq!(curriculum.tier(), 3);

        // Tier 4 for >= 0.75
        curriculum.overall_difficulty = 0.8;
        assert_eq!(curriculum.tier(), 4);
    }

    #[test]
    fn test_curriculum_scheduler_step() {
        let mut curriculum = AdaptiveCurriculum::new();

        curriculum.step(0, 1.0);
        assert!((curriculum.difficulty() - 0.1).abs() < 0.001);

        curriculum.step(1, 1.0);
        // 0.1 * 0.9 + 1.0 * 0.1 = 0.19
        assert!((curriculum.difficulty() - 0.19).abs() < 0.001);
    }

    #[test]
    fn test_curriculum_scheduler_reset() {
        let mut curriculum = AdaptiveCurriculum::new();
        curriculum.update_class("E0308", true);
        curriculum.step(0, 0.5);

        assert!(!curriculum.class_accuracy.is_empty());
        assert!(curriculum.difficulty() > 0.0);

        curriculum.reset();

        assert!(curriculum.class_accuracy.is_empty());
        assert!(curriculum.class_attempts.is_empty());
        assert_eq!(curriculum.difficulty(), 0.0);
    }

    #[test]
    fn test_curriculum_scheduler_name() {
        let curriculum = AdaptiveCurriculum::new();
        assert_eq!(curriculum.name(), "AdaptiveCurriculum");
    }

    #[test]
    fn test_adaptive_curriculum_clone() {
        let mut curriculum = AdaptiveCurriculum::new();
        curriculum.update_class("E0308", true);

        let cloned = curriculum.clone();
        assert_eq!(curriculum.class_attempts, cloned.class_attempts);
        assert_eq!(curriculum.class_accuracy, cloned.class_accuracy);
    }

    #[test]
    fn test_adaptive_curriculum_debug() {
        let curriculum = AdaptiveCurriculum::new();
        let debug = format!("{curriculum:?}");
        assert!(debug.contains("AdaptiveCurriculum"));
    }
}
