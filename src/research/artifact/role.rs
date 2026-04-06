//! Contributor roles following the CRediT taxonomy.

use serde::{Deserialize, Serialize};

/// Contributor roles following the CRediT (Contributor Roles Taxonomy)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContributorRole {
    /// Ideas; formulation or evolution of overarching research goals and aims
    Conceptualization,
    /// Management activities to annotate, scrub data and maintain research data
    DataCuration,
    /// Application of statistical, mathematical, computational techniques
    FormalAnalysis,
    /// Acquisition of financial support for the project
    FundingAcquisition,
    /// Conducting research and investigation process
    Investigation,
    /// Development or design of methodology
    Methodology,
    /// Management and coordination responsibility
    ProjectAdministration,
    /// Provision of study materials, reagents, materials, laboratory samples
    Resources,
    /// Programming, software development; designing computer programs
    Software,
    /// Oversight and leadership responsibility
    Supervision,
    /// Verification of results/experiments
    Validation,
    /// Preparation, creation and/or presentation of data visualization
    Visualization,
    /// Preparation and creation of the published work (original draft)
    WritingOriginal,
    /// Critical review, commentary or revision
    WritingReview,
}

impl std::fmt::Display for ContributorRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conceptualization => write!(f, "Conceptualization"),
            Self::DataCuration => write!(f, "Data curation"),
            Self::FormalAnalysis => write!(f, "Formal analysis"),
            Self::FundingAcquisition => write!(f, "Funding acquisition"),
            Self::Investigation => write!(f, "Investigation"),
            Self::Methodology => write!(f, "Methodology"),
            Self::ProjectAdministration => write!(f, "Project administration"),
            Self::Resources => write!(f, "Resources"),
            Self::Software => write!(f, "Software"),
            Self::Supervision => write!(f, "Supervision"),
            Self::Validation => write!(f, "Validation"),
            Self::Visualization => write!(f, "Visualization"),
            Self::WritingOriginal => write!(f, "Writing – original draft"),
            Self::WritingReview => write!(f, "Writing – review & editing"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contributor_role_display() {
        assert_eq!(ContributorRole::Conceptualization.to_string(), "Conceptualization");
        assert_eq!(ContributorRole::DataCuration.to_string(), "Data curation");
        assert_eq!(ContributorRole::FormalAnalysis.to_string(), "Formal analysis");
        assert_eq!(ContributorRole::FundingAcquisition.to_string(), "Funding acquisition");
        assert_eq!(ContributorRole::Investigation.to_string(), "Investigation");
        assert_eq!(ContributorRole::Methodology.to_string(), "Methodology");
        assert_eq!(ContributorRole::ProjectAdministration.to_string(), "Project administration");
        assert_eq!(ContributorRole::Resources.to_string(), "Resources");
        assert_eq!(ContributorRole::Software.to_string(), "Software");
        assert_eq!(ContributorRole::Supervision.to_string(), "Supervision");
        assert_eq!(ContributorRole::Validation.to_string(), "Validation");
        assert_eq!(ContributorRole::Visualization.to_string(), "Visualization");
        assert_eq!(ContributorRole::WritingOriginal.to_string(), "Writing – original draft");
        assert_eq!(ContributorRole::WritingReview.to_string(), "Writing – review & editing");
    }

    #[test]
    fn test_contributor_role_clone() {
        let role = ContributorRole::Software;
        let cloned = role;
        assert_eq!(role, cloned);
    }

    #[test]
    fn test_contributor_role_debug() {
        let role = ContributorRole::Software;
        assert_eq!(format!("{role:?}"), "Software");
    }

    #[test]
    fn test_contributor_role_eq() {
        assert_eq!(ContributorRole::Software, ContributorRole::Software);
        assert_ne!(ContributorRole::Software, ContributorRole::Validation);
    }

    #[test]
    fn test_contributor_role_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ContributorRole::Software);
        set.insert(ContributorRole::Software);
        assert_eq!(set.len(), 1);
        set.insert(ContributorRole::Validation);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_contributor_role_serde() {
        let role = ContributorRole::Software;
        let json = serde_json::to_string(&role).expect("JSON serialization should succeed");
        let deserialized: ContributorRole =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(role, deserialized);
    }
}
