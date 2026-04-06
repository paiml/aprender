//! Tests for InMemoryRegistry

use std::collections::HashMap;

use super::super::comparison::Comparison;
use super::super::error::RegistryError;
use super::super::policy::PromotionPolicy;
use super::super::stage::ModelStage;
use super::super::traits::ModelRegistry;
use super::InMemoryRegistry;

#[test]
fn test_registry_register_model() {
    let mut registry = InMemoryRegistry::new();
    let model =
        registry.register_model("test-model", "/path/v1").expect("operation should succeed");

    assert_eq!(model.name, "test-model");
    assert_eq!(model.version, 1);
}

#[test]
fn test_registry_register_multiple_versions() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");
    let model2 =
        registry.register_model("test-model", "/path/v2").expect("operation should succeed");

    assert_eq!(model2.version, 2);
}

#[test]
fn test_registry_get_model() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");

    let model = registry.get_model("test-model", 1).expect("operation should succeed");
    assert_eq!(model.artifact_uri, "/path/v1");
}

#[test]
fn test_registry_get_model_not_found() {
    let registry = InMemoryRegistry::new();
    let result = registry.get_model("nonexistent", 1);
    assert!(matches!(result, Err(RegistryError::VersionNotFound(_, _))));
}

#[test]
fn test_registry_get_latest() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");
    registry.register_model("test-model", "/path/v2").expect("operation should succeed");

    let latest = registry.get_latest("test-model").expect("operation should succeed");
    assert_eq!(latest.version, 2);
}

#[test]
fn test_registry_get_latest_by_stage() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");
    registry.register_model("test-model", "/path/v2").expect("operation should succeed");

    registry
        .transition_stage("test-model", 1, ModelStage::Development, None)
        .expect("operation should succeed");
    registry
        .transition_stage("test-model", 2, ModelStage::Development, None)
        .expect("operation should succeed");
    registry
        .transition_stage("test-model", 2, ModelStage::Staging, None)
        .expect("operation should succeed");

    let latest_dev = registry.get_latest_by_stage("test-model", ModelStage::Development);
    let latest_staging = registry.get_latest_by_stage("test-model", ModelStage::Staging);

    assert_eq!(latest_dev.map(|m| m.version), Some(1));
    assert_eq!(latest_staging.map(|m| m.version), Some(2));
}

#[test]
fn test_registry_list_versions() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");
    registry.register_model("test-model", "/path/v2").expect("operation should succeed");

    let versions = registry.list_versions("test-model").expect("operation should succeed");
    assert_eq!(versions.len(), 2);
    assert_eq!(versions[0].version, 1);
    assert_eq!(versions[1].version, 2);
}

#[test]
fn test_registry_transition_stage() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");

    registry
        .transition_stage("test-model", 1, ModelStage::Development, Some("alice"))
        .expect("operation should succeed");

    let model = registry.get_model("test-model", 1).expect("operation should succeed");
    assert_eq!(model.stage, ModelStage::Development);
    assert_eq!(model.promoted_by, Some("alice".to_string()));
}

#[test]
fn test_registry_transition_invalid() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");

    // Try to go directly to Production from None
    let result = registry.transition_stage("test-model", 1, ModelStage::Production, None);
    assert!(matches!(result, Err(RegistryError::InvalidTransition(_, _))));
}

#[test]
fn test_registry_compare_versions() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");
    registry.register_model("test-model", "/path/v2").expect("operation should succeed");

    let mut metrics1 = HashMap::new();
    metrics1.insert("accuracy".to_string(), 0.90);
    registry.log_metrics("test-model", 1, metrics1).expect("operation should succeed");

    let mut metrics2 = HashMap::new();
    metrics2.insert("accuracy".to_string(), 0.95);
    registry.log_metrics("test-model", 2, metrics2).expect("operation should succeed");

    let comparison =
        registry.compare_versions("test-model", 1, 2).expect("operation should succeed");
    assert!(comparison.v2_is_better);
    assert!(comparison.metric_diffs.get("accuracy").expect("key should exist") > &0.0);
}

#[test]
fn test_registry_log_metrics() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");

    let mut metrics = HashMap::new();
    metrics.insert("accuracy".to_string(), 0.95);
    metrics.insert("f1".to_string(), 0.92);
    registry.log_metrics("test-model", 1, metrics).expect("operation should succeed");

    let model = registry.get_model("test-model", 1).expect("operation should succeed");
    assert_eq!(model.metrics.get("accuracy"), Some(&0.95));
    assert_eq!(model.metrics.get("f1"), Some(&0.92));
}

#[test]
fn test_registry_get_transition_history() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");

    registry
        .transition_stage("test-model", 1, ModelStage::Development, None)
        .expect("operation should succeed");
    registry
        .transition_stage("test-model", 1, ModelStage::Staging, None)
        .expect("operation should succeed");

    let history = registry.get_transition_history("test-model").expect("operation should succeed");
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].to_stage, ModelStage::Development);
    assert_eq!(history[1].to_stage, ModelStage::Staging);
}

#[test]
fn test_registry_set_and_get_policy() {
    let mut registry = InMemoryRegistry::new();

    let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
        "accuracy",
        Comparison::Gte,
        0.95,
    );

    registry.set_policy(policy);

    let retrieved = registry.get_policy(ModelStage::Production);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.expect("operation should succeed").required_metrics.len(), 1);
}

#[test]
fn test_registry_can_promote_with_policy() {
    let mut registry = InMemoryRegistry::new();
    registry.register_model("test-model", "/path/v1").expect("operation should succeed");

    registry
        .transition_stage("test-model", 1, ModelStage::Development, None)
        .expect("operation should succeed");
    registry
        .transition_stage("test-model", 1, ModelStage::Staging, None)
        .expect("operation should succeed");

    let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
        "accuracy",
        Comparison::Gte,
        0.95,
    );
    registry.set_policy(policy);

    // Without required metrics
    let result = registry
        .can_promote("test-model", 1, ModelStage::Production, 0)
        .expect("operation should succeed");
    assert!(!result.passed);

    // Add metrics
    let mut metrics = HashMap::new();
    metrics.insert("accuracy".to_string(), 0.96);
    registry.log_metrics("test-model", 1, metrics).expect("operation should succeed");

    // With required metrics
    let result = registry
        .can_promote("test-model", 1, ModelStage::Production, 0)
        .expect("operation should succeed");
    assert!(result.passed);
}

#[test]
fn test_registry_auto_rollback() {
    let mut registry = InMemoryRegistry::new();
    registry.enable_auto_rollback("test-model", "accuracy", 0.90);

    // Should rollback
    assert!(registry.check_rollback("test-model", 0.85));
    // Should not rollback
    assert!(!registry.check_rollback("test-model", 0.95));
    // No rollback config
    assert!(!registry.check_rollback("other-model", 0.50));
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_version_numbers_increase(count in 1usize..20) {
            let mut registry = InMemoryRegistry::new();
            let mut last_version = 0u32;

            for _ in 0..count {
                let model = registry.register_model("test", "/path").expect("operation should succeed");
                prop_assert!(model.version > last_version);
                last_version = model.version;
            }
        }

        #[test]
        fn prop_metrics_preserved(
            metrics in prop::collection::hash_map(
                "[a-z]{1,10}",
                0.0f64..1.0,
                1..10
            )
        ) {
            let mut registry = InMemoryRegistry::new();
            registry.register_model("test", "/path").expect("operation should succeed");
            registry.log_metrics("test", 1, metrics.clone()).expect("operation should succeed");

            let model = registry.get_model("test", 1).expect("operation should succeed");
            for (key, value) in &metrics {
                prop_assert_eq!(model.metrics.get(key), Some(value));
            }
        }
    }
}
