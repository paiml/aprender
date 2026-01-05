//! Validation for synthetic data samples.

use std::fmt;

/// Result of validating a synthetic sample.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    /// Sample is valid and can be included in training.
    Accepted,
    /// Sample was rejected with a reason.
    Rejected(String),
}

impl ValidationResult {
    /// Check if the result is accepted.
    #[must_use]
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }

    /// Check if the result is rejected.
    #[must_use]
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected(_))
    }

    /// Get rejection reason if rejected, None if accepted.
    #[must_use]
    pub fn rejection_reason(&self) -> Option<&str> {
        match self {
            Self::Accepted => None,
            Self::Rejected(reason) => Some(reason),
        }
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accepted => write!(f, "Accepted"),
            Self::Rejected(reason) => write!(f, "Rejected: {reason}"),
        }
    }
}

/// Validates generated synthetic samples before inclusion.
///
/// Checks semantic similarity bounds and runs domain-specific validators
/// to ensure generated samples are useful for training.
///
/// # Example
///
/// ```
/// use aprender::synthetic::{SyntheticValidator, ValidationResult};
///
/// let validator = SyntheticValidator::new()
///     .with_min_similarity(0.3)
///     .with_max_overlap(0.95);
///
/// // Validate based on similarity score
/// let result = validator.validate_similarity(0.7);
/// assert!(result.is_accepted());
///
/// let result = validator.validate_similarity(0.1);
/// assert!(result.is_rejected());
/// ```
#[derive(Debug, Clone)]
pub struct SyntheticValidator {
    /// Minimum semantic similarity to seed sample.
    min_similarity: f32,
    /// Maximum allowed overlap with existing data (near-duplicate detection).
    max_overlap: f32,
    /// Minimum novelty score (distance from nearest existing sample).
    min_novelty: f32,
}

impl Default for SyntheticValidator {
    fn default() -> Self {
        Self {
            min_similarity: 0.3,
            max_overlap: 0.95,
            min_novelty: 0.05,
        }
    }
}

impl SyntheticValidator {
    /// Create a new validator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum similarity threshold.
    ///
    /// Samples too dissimilar from their seed are rejected as potentially
    /// semantically corrupted.
    #[must_use]
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set maximum overlap threshold.
    ///
    /// Samples too similar to existing data are rejected as near-duplicates
    /// that don't add new information.
    #[must_use]
    pub fn with_max_overlap(mut self, threshold: f32) -> Self {
        self.max_overlap = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set minimum novelty threshold.
    ///
    /// Samples must be at least this different from the nearest existing sample.
    #[must_use]
    pub fn with_min_novelty(mut self, threshold: f32) -> Self {
        self.min_novelty = threshold.clamp(0.0, 1.0);
        self
    }

    /// Get the minimum similarity threshold.
    #[must_use]
    pub fn min_similarity(&self) -> f32 {
        self.min_similarity
    }

    /// Get the maximum overlap threshold.
    #[must_use]
    pub fn max_overlap(&self) -> f32 {
        self.max_overlap
    }

    /// Get the minimum novelty threshold.
    #[must_use]
    pub fn min_novelty(&self) -> f32 {
        self.min_novelty
    }

    /// Validate a sample based on its similarity to the seed.
    ///
    /// # Arguments
    ///
    /// * `similarity` - Similarity score between generated sample and seed [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// `Accepted` if within bounds, `Rejected` otherwise.
    #[must_use]
    pub fn validate_similarity(&self, similarity: f32) -> ValidationResult {
        if similarity < self.min_similarity {
            return ValidationResult::Rejected(format!(
                "Too dissimilar from seed: {similarity:.3} < {:.3}",
                self.min_similarity
            ));
        }
        if similarity > self.max_overlap {
            return ValidationResult::Rejected(format!(
                "Too similar (near-duplicate): {similarity:.3} > {:.3}",
                self.max_overlap
            ));
        }
        ValidationResult::Accepted
    }

    /// Validate a sample based on its novelty score.
    ///
    /// # Arguments
    ///
    /// * `novelty` - Distance from nearest existing sample [0.0, 1.0]
    ///
    /// # Returns
    ///
    /// `Accepted` if above threshold, `Rejected` otherwise.
    #[must_use]
    pub fn validate_novelty(&self, novelty: f32) -> ValidationResult {
        if novelty < self.min_novelty {
            return ValidationResult::Rejected(format!(
                "Not novel enough: {novelty:.3} < {:.3}",
                self.min_novelty
            ));
        }
        ValidationResult::Accepted
    }

    /// Validate both similarity and novelty.
    ///
    /// # Arguments
    ///
    /// * `similarity` - Similarity score between generated sample and seed
    /// * `novelty` - Distance from nearest existing sample
    ///
    /// # Returns
    ///
    /// `Accepted` if both pass, first rejection reason otherwise.
    #[must_use]
    pub fn validate(&self, similarity: f32, novelty: f32) -> ValidationResult {
        let sim_result = self.validate_similarity(similarity);
        if sim_result.is_rejected() {
            return sim_result;
        }

        self.validate_novelty(novelty)
    }

    /// Validate a batch of samples, returning acceptance counts.
    ///
    /// # Arguments
    ///
    /// * `similarities` - Similarity scores for each sample
    /// * `novelties` - Novelty scores for each sample
    ///
    /// # Returns
    ///
    /// Tuple of (`accepted_count`, `rejected_count`, `rejection_reasons`).
    #[must_use]
    pub fn validate_batch(
        &self,
        similarities: &[f32],
        novelties: &[f32],
    ) -> (usize, usize, Vec<String>) {
        let mut accepted = 0;
        let mut rejected = 0;
        let mut reasons = Vec::new();

        for (sim, nov) in similarities.iter().zip(novelties.iter()) {
            match self.validate(*sim, *nov) {
                ValidationResult::Accepted => accepted += 1,
                ValidationResult::Rejected(reason) => {
                    rejected += 1;
                    reasons.push(reason);
                }
            }
        }

        (accepted, rejected, reasons)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_accepted() {
        let result = ValidationResult::Accepted;
        assert!(result.is_accepted());
        assert!(!result.is_rejected());
        assert_eq!(result.rejection_reason(), None);
        assert_eq!(format!("{result}"), "Accepted");
    }

    #[test]
    fn test_validation_result_rejected() {
        let result = ValidationResult::Rejected("too similar".to_string());
        assert!(!result.is_accepted());
        assert!(result.is_rejected());
        assert_eq!(result.rejection_reason(), Some("too similar"));
        assert!(format!("{result}").contains("too similar"));
    }

    #[test]
    fn test_default_validator() {
        let v = SyntheticValidator::default();
        assert!((v.min_similarity() - 0.3).abs() < f32::EPSILON);
        assert!((v.max_overlap() - 0.95).abs() < f32::EPSILON);
        assert!((v.min_novelty() - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_builder_pattern() {
        let v = SyntheticValidator::new()
            .with_min_similarity(0.4)
            .with_max_overlap(0.9)
            .with_min_novelty(0.1);

        assert!((v.min_similarity() - 0.4).abs() < f32::EPSILON);
        assert!((v.max_overlap() - 0.9).abs() < f32::EPSILON);
        assert!((v.min_novelty() - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_threshold_clamping() {
        let v = SyntheticValidator::new()
            .with_min_similarity(-0.5)
            .with_max_overlap(1.5)
            .with_min_novelty(-1.0);

        assert!((v.min_similarity() - 0.0).abs() < f32::EPSILON);
        assert!((v.max_overlap() - 1.0).abs() < f32::EPSILON);
        assert!((v.min_novelty() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_validate_similarity_accepted() {
        let v = SyntheticValidator::new()
            .with_min_similarity(0.3)
            .with_max_overlap(0.95);

        assert!(v.validate_similarity(0.5).is_accepted());
        assert!(v.validate_similarity(0.3).is_accepted());
        assert!(v.validate_similarity(0.95).is_accepted());
    }

    #[test]
    fn test_validate_similarity_too_low() {
        let v = SyntheticValidator::new().with_min_similarity(0.3);

        let result = v.validate_similarity(0.1);
        assert!(result.is_rejected());
        assert!(result
            .rejection_reason()
            .expect("should have rejection reason")
            .contains("dissimilar"));
    }

    #[test]
    fn test_validate_similarity_too_high() {
        let v = SyntheticValidator::new().with_max_overlap(0.95);

        let result = v.validate_similarity(0.99);
        assert!(result.is_rejected());
        assert!(result
            .rejection_reason()
            .expect("should have rejection reason")
            .contains("duplicate"));
    }

    #[test]
    fn test_validate_novelty_accepted() {
        let v = SyntheticValidator::new().with_min_novelty(0.1);

        assert!(v.validate_novelty(0.2).is_accepted());
        assert!(v.validate_novelty(0.1).is_accepted());
        assert!(v.validate_novelty(1.0).is_accepted());
    }

    #[test]
    fn test_validate_novelty_rejected() {
        let v = SyntheticValidator::new().with_min_novelty(0.1);

        let result = v.validate_novelty(0.05);
        assert!(result.is_rejected());
        assert!(result
            .rejection_reason()
            .expect("should have rejection reason")
            .contains("novel"));
    }

    #[test]
    fn test_validate_combined() {
        let v = SyntheticValidator::new()
            .with_min_similarity(0.3)
            .with_max_overlap(0.95)
            .with_min_novelty(0.1);

        // Both pass
        assert!(v.validate(0.5, 0.2).is_accepted());

        // Similarity fails first
        let result = v.validate(0.1, 0.2);
        assert!(result.is_rejected());
        assert!(result
            .rejection_reason()
            .expect("should have rejection reason")
            .contains("dissimilar"));

        // Novelty fails
        let result = v.validate(0.5, 0.01);
        assert!(result.is_rejected());
        assert!(result
            .rejection_reason()
            .expect("should have rejection reason")
            .contains("novel"));
    }

    #[test]
    fn test_validate_batch() {
        let v = SyntheticValidator::new()
            .with_min_similarity(0.3)
            .with_max_overlap(0.95)
            .with_min_novelty(0.1);

        let similarities = vec![0.5, 0.1, 0.8, 0.99];
        let novelties = vec![0.2, 0.2, 0.05, 0.3];

        let (accepted, rejected, reasons) = v.validate_batch(&similarities, &novelties);

        assert_eq!(accepted, 1); // Only first passes
        assert_eq!(rejected, 3);
        assert_eq!(reasons.len(), 3);
    }

    #[test]
    fn test_validate_batch_empty() {
        let v = SyntheticValidator::new();
        let (accepted, rejected, reasons) = v.validate_batch(&[], &[]);

        assert_eq!(accepted, 0);
        assert_eq!(rejected, 0);
        assert!(reasons.is_empty());
    }

    #[test]
    fn test_validate_batch_all_pass() {
        let v = SyntheticValidator::new()
            .with_min_similarity(0.3)
            .with_max_overlap(0.95)
            .with_min_novelty(0.05);

        let similarities = vec![0.5, 0.6, 0.7];
        let novelties = vec![0.2, 0.3, 0.4];

        let (accepted, rejected, _) = v.validate_batch(&similarities, &novelties);

        assert_eq!(accepted, 3);
        assert_eq!(rejected, 0);
    }

    #[test]
    fn test_validation_result_clone() {
        let r1 = ValidationResult::Rejected("test".to_string());
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_validator_clone() {
        let v1 = SyntheticValidator::new().with_min_similarity(0.5);
        let v2 = v1.clone();
        assert!((v1.min_similarity() - v2.min_similarity()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_validator_debug() {
        let v = SyntheticValidator::new();
        let debug = format!("{v:?}");
        assert!(debug.contains("SyntheticValidator"));
        assert!(debug.contains("min_similarity"));
    }
}
