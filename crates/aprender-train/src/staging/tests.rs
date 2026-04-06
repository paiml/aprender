//! Tests for staging module (GH-70)

use super::*;

// ---------------------------------------------------------------------------
// Stage enum tests
// ---------------------------------------------------------------------------

#[test]
fn test_stage_ordinal_ordering() {
    assert!(Stage::Dev.ordinal() < Stage::Staging.ordinal());
    assert!(Stage::Staging.ordinal() < Stage::Production.ordinal());
}

#[test]
fn test_stage_display() {
    assert_eq!(Stage::Dev.to_string(), "Dev");
    assert_eq!(Stage::Staging.to_string(), "Staging");
    assert_eq!(Stage::Production.to_string(), "Production");
}

#[test]
fn test_stage_as_str() {
    assert_eq!(Stage::Dev.as_str(), "Dev");
    assert_eq!(Stage::Staging.as_str(), "Staging");
    assert_eq!(Stage::Production.as_str(), "Production");
}

#[test]
fn test_stage_serialization_roundtrip() {
    for stage in [Stage::Dev, Stage::Staging, Stage::Production] {
        let json = serde_json::to_string(&stage).expect("JSON serialization should succeed");
        let deserialized: Stage =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(stage, deserialized);
    }
}

#[test]
fn test_stage_equality() {
    assert_eq!(Stage::Dev, Stage::Dev);
    assert_ne!(Stage::Dev, Stage::Staging);
    assert_ne!(Stage::Staging, Stage::Production);
}

// ---------------------------------------------------------------------------
// Registration tests
// ---------------------------------------------------------------------------

#[test]
fn test_register_model_creates_at_dev() {
    let mut registry = StagingRegistry::new();
    let mv = registry.register_model("my-model", "1.0.0", "/models/v1");
    assert_eq!(mv.name, "my-model");
    assert_eq!(mv.version, "1.0.0");
    assert_eq!(mv.stage, Stage::Dev);
    assert_eq!(mv.path, "/models/v1");
    assert!(mv.promoted_at.is_none());
    assert!(mv.metadata.is_empty());
}

#[test]
fn test_register_model_idempotent() {
    let mut registry = StagingRegistry::new();
    let mv1 = registry.register_model("m", "1.0.0", "/path/a");
    let mv2 = registry.register_model("m", "1.0.0", "/path/b");
    // Second registration returns the original, not a new one
    assert_eq!(mv1.created_at, mv2.created_at);
    assert_eq!(mv2.path, "/path/a");
}

#[test]
fn test_register_multiple_versions() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/v1");
    registry.register_model("m", "2.0.0", "/v2");
    let versions = registry.list_versions("m");
    assert_eq!(versions.len(), 2);
}

// ---------------------------------------------------------------------------
// Promote tests
// ---------------------------------------------------------------------------

#[test]
fn test_promote_dev_to_staging() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    let mv = registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    assert_eq!(mv.stage, Stage::Staging);
    assert!(mv.promoted_at.is_some());
}

#[test]
fn test_promote_staging_to_production() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    let mv = registry.promote("m", "1.0.0", Stage::Production).expect("operation should succeed");
    assert_eq!(mv.stage, Stage::Production);
    assert!(mv.promoted_at.is_some());
}

#[test]
fn test_promote_full_lifecycle() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");

    let mv = registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    assert_eq!(mv.stage, Stage::Staging);

    let mv = registry.promote("m", "1.0.0", Stage::Production).expect("operation should succeed");
    assert_eq!(mv.stage, Stage::Production);
}

#[test]
fn test_promote_skip_stage_rejected() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    let err = registry.promote("m", "1.0.0", Stage::Production).unwrap_err();
    match err {
        StagingError::InvalidTransition { from, to, .. } => {
            assert_eq!(from, Stage::Dev);
            assert_eq!(to, Stage::Production);
        }
        _ => panic!("expected InvalidTransition error"),
    }
}

#[test]
fn test_promote_same_stage_rejected() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    let err = registry.promote("m", "1.0.0", Stage::Dev).unwrap_err();
    match err {
        StagingError::InvalidTransition { from, to, .. } => {
            assert_eq!(from, Stage::Dev);
            assert_eq!(to, Stage::Dev);
        }
        _ => panic!("expected InvalidTransition error"),
    }
}

#[test]
fn test_promote_beyond_production_rejected() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    registry.promote("m", "1.0.0", Stage::Production).expect("operation should succeed");
    // Already at Production, promoting to Staging should fail (not +1)
    let err = registry.promote("m", "1.0.0", Stage::Staging).unwrap_err();
    assert!(matches!(err, StagingError::InvalidTransition { .. }));
}

#[test]
fn test_promote_nonexistent_model() {
    let mut registry = StagingRegistry::new();
    let err = registry.promote("ghost", "1.0.0", Stage::Staging).unwrap_err();
    match err {
        StagingError::NotFound { name, version } => {
            assert_eq!(name, "ghost");
            assert_eq!(version, "1.0.0");
        }
        _ => panic!("expected NotFound error"),
    }
}

// ---------------------------------------------------------------------------
// Demote tests
// ---------------------------------------------------------------------------

#[test]
fn test_demote_production_to_staging() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    registry.promote("m", "1.0.0", Stage::Production).expect("operation should succeed");

    let mv = registry.demote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    assert_eq!(mv.stage, Stage::Staging);
    assert!(mv.promoted_at.is_some());
}

#[test]
fn test_demote_staging_to_dev() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");

    let mv = registry.demote("m", "1.0.0", Stage::Dev).expect("operation should succeed");
    assert_eq!(mv.stage, Stage::Dev);
}

#[test]
fn test_demote_skip_stage_rejected() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    registry.promote("m", "1.0.0", Stage::Production).expect("operation should succeed");

    let err = registry.demote("m", "1.0.0", Stage::Dev).unwrap_err();
    match err {
        StagingError::InvalidTransition { from, to, .. } => {
            assert_eq!(from, Stage::Production);
            assert_eq!(to, Stage::Dev);
        }
        _ => panic!("expected InvalidTransition error"),
    }
}

#[test]
fn test_demote_from_dev_rejected() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    // Cannot demote below Dev
    let err = registry.demote("m", "1.0.0", Stage::Dev).unwrap_err();
    assert!(matches!(err, StagingError::InvalidTransition { .. }));
}

#[test]
fn test_demote_nonexistent_model() {
    let mut registry = StagingRegistry::new();
    let err = registry.demote("ghost", "1.0.0", Stage::Dev).unwrap_err();
    assert!(matches!(err, StagingError::NotFound { .. }));
}

#[test]
fn test_demote_same_stage_rejected() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");
    registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    let err = registry.demote("m", "1.0.0", Stage::Staging).unwrap_err();
    assert!(matches!(err, StagingError::InvalidTransition { .. }));
}

// ---------------------------------------------------------------------------
// get_latest tests
// ---------------------------------------------------------------------------

#[test]
fn test_get_latest_returns_correct_version() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/v1");
    registry.register_model("m", "2.0.0", "/v2");

    // Both at Dev, latest by creation time should be v2
    let latest = registry.get_latest("m", Stage::Dev).expect("operation should succeed");
    assert_eq!(latest.version, "2.0.0");
}

#[test]
fn test_get_latest_filters_by_stage() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/v1");
    registry.register_model("m", "2.0.0", "/v2");
    registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");

    // Only v1 is in Staging
    let latest_staging =
        registry.get_latest("m", Stage::Staging).expect("operation should succeed");
    assert_eq!(latest_staging.version, "1.0.0");

    // v2 is still in Dev
    let latest_dev = registry.get_latest("m", Stage::Dev).expect("operation should succeed");
    assert_eq!(latest_dev.version, "2.0.0");
}

#[test]
fn test_get_latest_returns_none_when_no_match() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/v1");
    assert!(registry.get_latest("m", Stage::Production).is_none());
}

#[test]
fn test_get_latest_returns_none_for_unknown_model() {
    let registry = StagingRegistry::new();
    assert!(registry.get_latest("nonexistent", Stage::Dev).is_none());
}

// ---------------------------------------------------------------------------
// list_versions tests
// ---------------------------------------------------------------------------

#[test]
fn test_list_versions_returns_all() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/v1");
    registry.register_model("m", "2.0.0", "/v2");
    registry.register_model("m", "3.0.0", "/v3");

    let versions = registry.list_versions("m");
    assert_eq!(versions.len(), 3);
}

#[test]
fn test_list_versions_sorted_by_creation() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/v1");
    registry.register_model("m", "2.0.0", "/v2");

    let versions = registry.list_versions("m");
    assert!(versions[0].created_at <= versions[1].created_at);
}

#[test]
fn test_list_versions_empty_for_unknown() {
    let registry = StagingRegistry::new();
    let versions = registry.list_versions("nonexistent");
    assert!(versions.is_empty());
}

#[test]
fn test_list_versions_only_for_named_model() {
    let mut registry = StagingRegistry::new();
    registry.register_model("alpha", "1.0.0", "/a");
    registry.register_model("beta", "1.0.0", "/b");

    let alpha_versions = registry.list_versions("alpha");
    assert_eq!(alpha_versions.len(), 1);
    assert_eq!(alpha_versions[0].name, "alpha");
}

// ---------------------------------------------------------------------------
// Error display tests
// ---------------------------------------------------------------------------

#[test]
fn test_error_display_not_found() {
    let err = StagingError::NotFound { name: "m".to_string(), version: "1.0.0".to_string() };
    assert!(err.to_string().contains('m'));
    assert!(err.to_string().contains("1.0.0"));
}

#[test]
fn test_error_display_invalid_transition() {
    let err = StagingError::InvalidTransition {
        name: "m".to_string(),
        version: "1.0.0".to_string(),
        from: Stage::Dev,
        to: Stage::Production,
    };
    let msg = err.to_string();
    assert!(msg.contains("Dev"));
    assert!(msg.contains("Production"));
}

#[test]
fn test_error_display_already_exists() {
    let err = StagingError::AlreadyExists { name: "m".to_string(), version: "1.0.0".to_string() };
    assert!(err.to_string().contains("already exists"));
}

// ---------------------------------------------------------------------------
// ModelVersion struct tests
// ---------------------------------------------------------------------------

#[test]
fn test_model_version_serialization_roundtrip() {
    let mut registry = StagingRegistry::new();
    let mv = registry.register_model("m", "1.0.0", "/path");

    let json = serde_json::to_string(&mv).expect("JSON serialization should succeed");
    let deserialized: ModelVersion =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(deserialized.name, mv.name);
    assert_eq!(deserialized.version, mv.version);
    assert_eq!(deserialized.stage, mv.stage);
    assert_eq!(deserialized.path, mv.path);
}

// ---------------------------------------------------------------------------
// Integration / lifecycle tests
// ---------------------------------------------------------------------------

#[test]
fn test_promote_then_demote_roundtrip() {
    let mut registry = StagingRegistry::new();
    registry.register_model("m", "1.0.0", "/path");

    registry.promote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    registry.promote("m", "1.0.0", Stage::Production).expect("operation should succeed");
    registry.demote("m", "1.0.0", Stage::Staging).expect("operation should succeed");
    registry.demote("m", "1.0.0", Stage::Dev).expect("operation should succeed");

    let mv = registry.get_latest("m", Stage::Dev).expect("operation should succeed");
    assert_eq!(mv.stage, Stage::Dev);
    assert_eq!(mv.version, "1.0.0");
}

#[test]
fn test_multiple_models_independent_stages() {
    let mut registry = StagingRegistry::new();
    registry.register_model("alpha", "1.0.0", "/a");
    registry.register_model("beta", "1.0.0", "/b");

    registry.promote("alpha", "1.0.0", Stage::Staging).expect("operation should succeed");

    // alpha is Staging, beta is still Dev
    assert_eq!(
        registry.get_latest("alpha", Stage::Staging).expect("operation should succeed").stage,
        Stage::Staging
    );
    assert_eq!(
        registry.get_latest("beta", Stage::Dev).expect("operation should succeed").stage,
        Stage::Dev
    );
    assert!(registry.get_latest("beta", Stage::Staging).is_none());
}
