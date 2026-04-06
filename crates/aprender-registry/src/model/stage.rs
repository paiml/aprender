//! Model lifecycle stages.

use crate::error::{PachaError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Lifecycle stage of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelStage {
    /// Model is in active development.
    #[default]
    Development,
    /// Model is in staging for testing.
    Staging,
    /// Model is in production.
    Production,
    /// Model is archived (no longer in use).
    Archived,
}

impl ModelStage {
    /// Get all valid stages.
    #[must_use]
    pub fn all() -> &'static [ModelStage] {
        &[
            ModelStage::Development,
            ModelStage::Staging,
            ModelStage::Production,
            ModelStage::Archived,
        ]
    }

    /// Check if transition to another stage is valid.
    ///
    /// Valid transitions:
    /// - Development -> Staging, Archived
    /// - Staging -> Development, Production, Archived
    /// - Production -> Staging, Archived
    /// - Archived -> Development (for resurrection)
    #[must_use]
    pub fn can_transition_to(&self, target: ModelStage) -> bool {
        if *self == target {
            return true; // Same stage is always valid
        }

        match self {
            Self::Development => matches!(target, Self::Staging | Self::Archived),
            Self::Staging => {
                matches!(target, Self::Development | Self::Production | Self::Archived)
            }
            Self::Production => matches!(target, Self::Staging | Self::Archived),
            Self::Archived => matches!(target, Self::Development),
        }
    }

    /// Attempt to transition to another stage.
    ///
    /// # Errors
    ///
    /// Returns an error if the transition is invalid.
    pub fn transition_to(&self, target: ModelStage) -> Result<ModelStage> {
        if self.can_transition_to(target) {
            Ok(target)
        } else {
            Err(PachaError::InvalidStageTransition {
                from: self.to_string(),
                to: target.to_string(),
            })
        }
    }

    /// Check if this stage allows modification.
    #[must_use]
    pub fn is_mutable(&self) -> bool {
        matches!(self, Self::Development)
    }

    /// Check if this stage is active (not archived).
    #[must_use]
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Archived)
    }
}

impl fmt::Display for ModelStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Development => "development",
            Self::Staging => "staging",
            Self::Production => "production",
            Self::Archived => "archived",
        };
        write!(f, "{s}")
    }
}

impl FromStr for ModelStage {
    type Err = PachaError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "development" | "dev" => Ok(Self::Development),
            "staging" | "stage" => Ok(Self::Staging),
            "production" | "prod" => Ok(Self::Production),
            "archived" | "archive" => Ok(Self::Archived),
            _ => Err(PachaError::Validation(format!("unknown stage: {s}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_display() {
        assert_eq!(ModelStage::Development.to_string(), "development");
        assert_eq!(ModelStage::Staging.to_string(), "staging");
        assert_eq!(ModelStage::Production.to_string(), "production");
        assert_eq!(ModelStage::Archived.to_string(), "archived");
    }

    #[test]
    fn test_stage_parse() {
        assert_eq!("development".parse::<ModelStage>().unwrap(), ModelStage::Development);
        assert_eq!("dev".parse::<ModelStage>().unwrap(), ModelStage::Development);
        assert_eq!("staging".parse::<ModelStage>().unwrap(), ModelStage::Staging);
        assert_eq!("stage".parse::<ModelStage>().unwrap(), ModelStage::Staging);
        assert_eq!("production".parse::<ModelStage>().unwrap(), ModelStage::Production);
        assert_eq!("prod".parse::<ModelStage>().unwrap(), ModelStage::Production);
        assert_eq!("archived".parse::<ModelStage>().unwrap(), ModelStage::Archived);
    }

    #[test]
    fn test_stage_parse_error() {
        assert!("invalid".parse::<ModelStage>().is_err());
        assert!("".parse::<ModelStage>().is_err());
    }

    #[test]
    fn test_valid_transitions_from_development() {
        let dev = ModelStage::Development;
        assert!(dev.can_transition_to(ModelStage::Development));
        assert!(dev.can_transition_to(ModelStage::Staging));
        assert!(!dev.can_transition_to(ModelStage::Production)); // Must go through staging
        assert!(dev.can_transition_to(ModelStage::Archived));
    }

    #[test]
    fn test_valid_transitions_from_staging() {
        let staging = ModelStage::Staging;
        assert!(staging.can_transition_to(ModelStage::Development)); // Rollback
        assert!(staging.can_transition_to(ModelStage::Staging));
        assert!(staging.can_transition_to(ModelStage::Production));
        assert!(staging.can_transition_to(ModelStage::Archived));
    }

    #[test]
    fn test_valid_transitions_from_production() {
        let prod = ModelStage::Production;
        assert!(!prod.can_transition_to(ModelStage::Development)); // Can't go back to dev
        assert!(prod.can_transition_to(ModelStage::Staging)); // Rollback to staging
        assert!(prod.can_transition_to(ModelStage::Production));
        assert!(prod.can_transition_to(ModelStage::Archived));
    }

    #[test]
    fn test_valid_transitions_from_archived() {
        let archived = ModelStage::Archived;
        assert!(archived.can_transition_to(ModelStage::Development)); // Resurrection
        assert!(!archived.can_transition_to(ModelStage::Staging));
        assert!(!archived.can_transition_to(ModelStage::Production));
        assert!(archived.can_transition_to(ModelStage::Archived));
    }

    #[test]
    fn test_transition_to_success() {
        let dev = ModelStage::Development;
        let result = dev.transition_to(ModelStage::Staging);
        assert_eq!(result.unwrap(), ModelStage::Staging);
    }

    #[test]
    fn test_transition_to_error() {
        let dev = ModelStage::Development;
        let result = dev.transition_to(ModelStage::Production);
        assert!(matches!(result, Err(PachaError::InvalidStageTransition { .. })));
    }

    #[test]
    fn test_is_mutable() {
        assert!(ModelStage::Development.is_mutable());
        assert!(!ModelStage::Staging.is_mutable());
        assert!(!ModelStage::Production.is_mutable());
        assert!(!ModelStage::Archived.is_mutable());
    }

    #[test]
    fn test_is_active() {
        assert!(ModelStage::Development.is_active());
        assert!(ModelStage::Staging.is_active());
        assert!(ModelStage::Production.is_active());
        assert!(!ModelStage::Archived.is_active());
    }

    #[test]
    fn test_serialization() {
        let stage = ModelStage::Production;
        let json = serde_json::to_string(&stage).unwrap();
        assert_eq!(json, "\"production\"");

        let deserialized: ModelStage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, ModelStage::Production);
    }

    #[test]
    fn test_all_stages() {
        let all = ModelStage::all();
        assert_eq!(all.len(), 4);
        assert!(all.contains(&ModelStage::Development));
        assert!(all.contains(&ModelStage::Staging));
        assert!(all.contains(&ModelStage::Production));
        assert!(all.contains(&ModelStage::Archived));
    }
}
