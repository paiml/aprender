use super::*;

#[test]
fn test_derivation_prune_details() {
    let parent = [0xDD; 32];
    let deriv = DerivationType::Prune {
        parent_hash: parent,
        sparsity: 0.9,
    };
    if let DerivationType::Prune {
        sparsity,
        parent_hash,
    } = deriv
    {
        assert!((sparsity - 0.9).abs() < 1e-6);
        assert_eq!(parent_hash, parent);
    } else {
        panic!("Expected Prune");
    }
}

#[test]
fn test_stack_component_copy() {
    let comp = StackComponent::Realizar;
    let copy = comp;
    // Both should be usable (Copy trait)
    assert_eq!(comp.name(), "realizar");
    assert_eq!(copy.name(), "realizar");
}

#[test]
fn test_quantization_type_copy() {
    let qt = QuantizationType::BFloat16;
    let copy = qt;
    assert_eq!(qt.bits(), 16);
    assert_eq!(copy.bits(), 16);
}

#[test]
fn test_model_stage_copy() {
    let stage = ModelStage::Archived;
    let copy = stage;
    assert_eq!(stage.name(), "archived");
    assert_eq!(copy.name(), "archived");
}

#[test]
fn test_health_status_copy() {
    let status = HealthStatus::Unknown;
    let copy = status;
    assert!(!status.is_operational());
    assert!(!copy.is_operational());
}

#[test]
fn test_format_compatibility_copy() {
    let compat = FormatCompatibility::current();
    let copy = compat;
    assert_eq!(compat.apr_version, copy.apr_version);
    assert_eq!(compat.ald_version, copy.ald_version);
}

#[test]
fn test_stack_health_multiple_updates() {
    let mut health = StackHealth::new();

    // First: all healthy
    health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0"));
    health.set_component(StackComponent::Pacha, ComponentHealth::healthy("1.0"));
    assert_eq!(health.overall, HealthStatus::Healthy);

    // Second: one becomes unhealthy
    health.set_component(StackComponent::Aprender, ComponentHealth::unhealthy("down"));
    assert_eq!(health.overall, HealthStatus::Unhealthy);

    // Third: fix it, now degraded
    health.set_component(
        StackComponent::Aprender,
        ComponentHealth::degraded("1.0", "slow"),
    );
    assert_eq!(health.overall, HealthStatus::Degraded);

    // Fourth: all healthy again
    health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0"));
    assert_eq!(health.overall, HealthStatus::Healthy);
}

#[test]
fn test_component_health_unhealthy_no_version() {
    let health = ComponentHealth::unhealthy("service unavailable");
    assert!(health.version.is_none());
    assert!(health.response_time_ms.is_none());
}

#[test]
fn test_inference_config_port_range() {
    let config = InferenceConfig::new("m.apr").with_port(0);
    assert_eq!(config.port, 0);

    let config = InferenceConfig::new("m.apr").with_port(65535);
    assert_eq!(config.port, 65535);
}

#[test]
fn test_inference_config_large_batch_size() {
    let config = InferenceConfig::new("m.apr").with_batch_size(1_000_000);
    assert_eq!(config.max_batch_size, 1_000_000);
}

#[test]
fn test_inference_config_zero_timeout() {
    let config = InferenceConfig::new("m.apr").with_timeout_ms(0);
    assert_eq!(config.timeout_ms, 0);
}

#[test]
fn test_model_version_large_size() {
    let version = ModelVersion::new("1.0.0", [0u8; 32]).with_size(100_000_000_000); // 100GB
    assert_eq!(version.size_bytes, 100_000_000_000);
}

#[test]
fn test_model_version_quality_score_max() {
    let version = ModelVersion::new("1.0.0", [0u8; 32])
        .with_stage(ModelStage::Production)
        .with_quality_score(100.0);
    assert!(version.is_production_ready());
}
