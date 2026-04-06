//! Model lifecycle stages (Kanban workflow)

use serde::{Deserialize, Serialize};

/// Model lifecycle stages (Kanban workflow)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelStage {
    /// Not assigned to any stage
    None,
    /// In active development
    Development,
    /// Being tested/validated
    Staging,
    /// Deployed and serving traffic
    Production,
    /// Retired from active use
    Archived,
}

impl ModelStage {
    /// Check if transition to target stage is valid
    pub fn can_transition_to(&self, target: ModelStage) -> bool {
        match (self, target) {
            // Any stage can go to Archived
            (
                ModelStage::None
                | ModelStage::Development
                | ModelStage::Staging
                | ModelStage::Production
                | ModelStage::Archived,
                ModelStage::Archived,
            ) => true,
            // None can go to Development
            (ModelStage::None, ModelStage::Development) => true,
            // Development can go to Staging
            (ModelStage::Development, ModelStage::Staging) => true,
            // Staging can go to Production
            (ModelStage::Staging, ModelStage::Production) => true,
            // Production can go back to Staging (rollback)
            (ModelStage::Production, ModelStage::Staging) => true,
            // Staging can go back to Development (rejected)
            (ModelStage::Staging, ModelStage::Development) => true,
            // Archived can be restored to Development
            (ModelStage::Archived, ModelStage::Development) => true,
            // Same stage is a no-op
            (ModelStage::None, ModelStage::None)
            | (ModelStage::Development, ModelStage::Development)
            | (ModelStage::Staging, ModelStage::Staging)
            | (ModelStage::Production, ModelStage::Production) => true,
            // Invalid transitions
            (
                ModelStage::Development
                | ModelStage::Staging
                | ModelStage::Production
                | ModelStage::Archived,
                ModelStage::None,
            )
            | (ModelStage::Production, ModelStage::Development)
            | (ModelStage::None | ModelStage::Archived, ModelStage::Staging)
            | (
                ModelStage::None | ModelStage::Development | ModelStage::Archived,
                ModelStage::Production,
            ) => false,
        }
    }

    /// Get display name
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelStage::None => "None",
            ModelStage::Development => "Development",
            ModelStage::Staging => "Staging",
            ModelStage::Production => "Production",
            ModelStage::Archived => "Archived",
        }
    }
}

impl std::fmt::Display for ModelStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_none_to_development() {
        assert!(ModelStage::None.can_transition_to(ModelStage::Development));
    }

    #[test]
    fn test_stage_development_to_staging() {
        assert!(ModelStage::Development.can_transition_to(ModelStage::Staging));
    }

    #[test]
    fn test_stage_staging_to_production() {
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Production));
    }

    #[test]
    fn test_stage_production_rollback_to_staging() {
        assert!(ModelStage::Production.can_transition_to(ModelStage::Staging));
    }

    #[test]
    fn test_stage_any_to_archived() {
        assert!(ModelStage::None.can_transition_to(ModelStage::Archived));
        assert!(ModelStage::Development.can_transition_to(ModelStage::Archived));
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Archived));
        assert!(ModelStage::Production.can_transition_to(ModelStage::Archived));
    }

    #[test]
    fn test_stage_invalid_transitions() {
        assert!(!ModelStage::None.can_transition_to(ModelStage::Production));
        assert!(!ModelStage::Development.can_transition_to(ModelStage::Production));
    }

    #[test]
    fn test_stage_display() {
        assert_eq!(ModelStage::Production.to_string(), "Production");
        assert_eq!(ModelStage::Development.as_str(), "Development");
    }

    #[test]
    fn test_stage_staging_to_development_rejected() {
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Development));
    }

    #[test]
    fn test_stage_archived_to_development_restore() {
        assert!(ModelStage::Archived.can_transition_to(ModelStage::Development));
    }

    #[test]
    fn test_stage_same_stage_noop() {
        assert!(ModelStage::Development.can_transition_to(ModelStage::Development));
        assert!(ModelStage::Staging.can_transition_to(ModelStage::Staging));
        assert!(ModelStage::Production.can_transition_to(ModelStage::Production));
        assert!(ModelStage::Archived.can_transition_to(ModelStage::Archived));
        assert!(ModelStage::None.can_transition_to(ModelStage::None));
    }

    #[test]
    fn test_stage_invalid_none_to_staging() {
        assert!(!ModelStage::None.can_transition_to(ModelStage::Staging));
    }

    #[test]
    fn test_stage_invalid_archived_to_staging() {
        assert!(!ModelStage::Archived.can_transition_to(ModelStage::Staging));
    }

    #[test]
    fn test_stage_invalid_archived_to_production() {
        assert!(!ModelStage::Archived.can_transition_to(ModelStage::Production));
    }

    #[test]
    fn test_as_str_all_stages() {
        assert_eq!(ModelStage::None.as_str(), "None");
        assert_eq!(ModelStage::Staging.as_str(), "Staging");
        assert_eq!(ModelStage::Archived.as_str(), "Archived");
    }

    #[test]
    fn test_display_all_stages() {
        assert_eq!(format!("{}", ModelStage::None), "None");
        assert_eq!(format!("{}", ModelStage::Development), "Development");
        assert_eq!(format!("{}", ModelStage::Staging), "Staging");
        assert_eq!(format!("{}", ModelStage::Archived), "Archived");
    }

    #[test]
    fn test_stage_serialization() {
        let stage = ModelStage::Production;
        let json = serde_json::to_string(&stage).expect("JSON serialization should succeed");
        assert!(json.contains("Production"));
    }

    #[test]
    fn test_stage_deserialization() {
        let json = "\"Staging\"";
        let stage: ModelStage =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert_eq!(stage, ModelStage::Staging);
    }

    #[test]
    fn test_stage_roundtrip() {
        let stages = [
            ModelStage::None,
            ModelStage::Development,
            ModelStage::Staging,
            ModelStage::Production,
            ModelStage::Archived,
        ];
        for stage in stages {
            let json = serde_json::to_string(&stage).expect("JSON serialization should succeed");
            let deserialized: ModelStage =
                serde_json::from_str(&json).expect("JSON deserialization should succeed");
            assert_eq!(stage, deserialized);
        }
    }

    #[test]
    fn test_stage_clone() {
        let stage = ModelStage::Development;
        let cloned = stage;
        assert_eq!(stage, cloned);
    }

    #[test]
    fn test_stage_copy() {
        let stage = ModelStage::Production;
        let copied = stage;
        assert_eq!(stage, copied);
    }

    #[test]
    fn test_stage_debug() {
        let stage = ModelStage::Staging;
        let debug = format!("{stage:?}");
        assert!(debug.contains("Staging"));
    }

    #[test]
    fn test_stage_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ModelStage::Development);
        set.insert(ModelStage::Production);
        assert_eq!(set.len(), 2);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_stage_self_transition(stage in any::<u8>().prop_map(|n| match n % 5 {
            0 => ModelStage::None,
            1 => ModelStage::Development,
            2 => ModelStage::Staging,
            3 => ModelStage::Production,
            _ => ModelStage::Archived,
        })) {
            // Self-transition is always valid
            prop_assert!(stage.can_transition_to(stage));
        }

        #[test]
        fn prop_all_stages_can_archive(stage in any::<u8>().prop_map(|n| match n % 5 {
            0 => ModelStage::None,
            1 => ModelStage::Development,
            2 => ModelStage::Staging,
            3 => ModelStage::Production,
            _ => ModelStage::Archived,
        })) {
            // All stages can transition to Archived
            prop_assert!(stage.can_transition_to(ModelStage::Archived));
        }
    }
}
