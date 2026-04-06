//! Integrity assessment categories
//!
//! Provides human-readable assessment levels for behavioral integrity.

/// Overall integrity assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityAssessment {
    /// Score >= 0.9, no critical violations
    Excellent,
    /// Score >= 0.7
    Good,
    /// Score >= 0.5
    Fair,
    /// Score < 0.5
    Poor,
    /// Has critical violations
    Critical,
}

impl std::fmt::Display for IntegrityAssessment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "Excellent"),
            Self::Good => write!(f, "Good"),
            Self::Fair => write!(f, "Fair"),
            Self::Poor => write!(f, "Poor"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrity_assessment_display_excellent() {
        assert_eq!(IntegrityAssessment::Excellent.to_string(), "Excellent");
    }

    #[test]
    fn test_integrity_assessment_display_good() {
        assert_eq!(IntegrityAssessment::Good.to_string(), "Good");
    }

    #[test]
    fn test_integrity_assessment_display_fair() {
        assert_eq!(IntegrityAssessment::Fair.to_string(), "Fair");
    }

    #[test]
    fn test_integrity_assessment_display_poor() {
        assert_eq!(IntegrityAssessment::Poor.to_string(), "Poor");
    }

    #[test]
    fn test_integrity_assessment_display_critical() {
        assert_eq!(IntegrityAssessment::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_integrity_assessment_clone() {
        let assessment = IntegrityAssessment::Excellent;
        let cloned = assessment;
        assert_eq!(assessment, cloned);
    }

    #[test]
    fn test_integrity_assessment_eq() {
        assert_eq!(IntegrityAssessment::Excellent, IntegrityAssessment::Excellent);
        assert_ne!(IntegrityAssessment::Excellent, IntegrityAssessment::Good);
        assert_ne!(IntegrityAssessment::Good, IntegrityAssessment::Fair);
        assert_ne!(IntegrityAssessment::Fair, IntegrityAssessment::Poor);
        assert_ne!(IntegrityAssessment::Poor, IntegrityAssessment::Critical);
    }

    #[test]
    fn test_integrity_assessment_debug() {
        assert_eq!(format!("{:?}", IntegrityAssessment::Excellent), "Excellent");
        assert_eq!(format!("{:?}", IntegrityAssessment::Good), "Good");
        assert_eq!(format!("{:?}", IntegrityAssessment::Fair), "Fair");
        assert_eq!(format!("{:?}", IntegrityAssessment::Poor), "Poor");
        assert_eq!(format!("{:?}", IntegrityAssessment::Critical), "Critical");
    }
}
