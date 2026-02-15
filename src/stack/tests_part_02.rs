
#[test]
fn test_model_stage_debug() {
    let stage = ModelStage::Staging;
    let debug_str = format!("{:?}", stage);
    assert!(debug_str.contains("Staging"));
}

#[test]
fn test_model_version_debug() {
    let version = ModelVersion::new("1.0.0", [0u8; 32]);
    let debug_str = format!("{:?}", version);
    assert!(debug_str.contains("ModelVersion"));
}

#[test]
fn test_inference_config_debug() {
    let config = InferenceConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("InferenceConfig"));
}

#[test]
fn test_stack_health_debug() {
    let health = StackHealth::new();
    let debug_str = format!("{:?}", health);
    assert!(debug_str.contains("StackHealth"));
}

#[test]
fn test_component_health_debug() {
    let health = ComponentHealth::healthy("1.0.0");
    let debug_str = format!("{:?}", health);
    assert!(debug_str.contains("ComponentHealth"));
}

#[test]
fn test_health_status_debug() {
    let status = HealthStatus::Degraded;
    let debug_str = format!("{:?}", status);
    assert!(debug_str.contains("Degraded"));
}

#[test]
fn test_format_compatibility_debug() {
    let compat = FormatCompatibility::current();
    let debug_str = format!("{:?}", compat);
    assert!(debug_str.contains("FormatCompatibility"));
}

#[test]
fn test_stack_component_clone() {
    let comp = StackComponent::Pacha;
    let cloned = comp;
    assert_eq!(comp, cloned);
}

#[test]
fn test_stack_component_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(StackComponent::Aprender);
    set.insert(StackComponent::Pacha);
    assert!(set.contains(&StackComponent::Aprender));
    assert!(!set.contains(&StackComponent::Batuta));
}

#[test]
fn test_derivation_type_clone() {
    let deriv = DerivationType::Original;
    let cloned = deriv.clone();
    assert_eq!(deriv, cloned);
}

#[test]
fn test_quantization_type_clone() {
    let qt = QuantizationType::Float16;
    let cloned = qt;
    assert_eq!(qt, cloned);
}

#[test]
fn test_quantization_type_eq() {
    assert_eq!(QuantizationType::Int8, QuantizationType::Int8);
    assert_ne!(QuantizationType::Int8, QuantizationType::Int4);
}

#[test]
fn test_model_stage_clone() {
    let stage = ModelStage::Production;
    let cloned = stage;
    assert_eq!(stage, cloned);
}

#[test]
fn test_model_version_clone() {
    let version = ModelVersion::new("1.0.0", [0u8; 32]).with_tag("test");
    let cloned = version.clone();
    assert_eq!(cloned.version, "1.0.0");
    assert!(cloned.tags.contains(&"test".to_string()));
}

#[test]
fn test_inference_config_clone() {
    let config = InferenceConfig::new("model.apr").with_port(9000);
    let cloned = config.clone();
    assert_eq!(cloned.port, 9000);
}

#[test]
fn test_stack_health_clone() {
    let mut health = StackHealth::new();
    health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0.0"));
    let cloned = health.clone();
    assert!(cloned.components.contains_key(&StackComponent::Aprender));
}

#[test]
fn test_component_health_clone() {
    let health = ComponentHealth::degraded("1.0.0", "slow").with_response_time(100);
    let cloned = health.clone();
    assert_eq!(cloned.status, HealthStatus::Degraded);
    assert_eq!(cloned.response_time_ms, Some(100));
}

#[test]
fn test_health_status_clone() {
    let status = HealthStatus::Healthy;
    let cloned = status;
    assert_eq!(status, cloned);
}

#[test]
fn test_format_compatibility_clone() {
    let compat = FormatCompatibility::current();
    let cloned = compat;
    assert_eq!(compat.apr_version, cloned.apr_version);
}

// =========================================================================
// Additional coverage tests for all branches
// =========================================================================

#[test]
fn test_stack_component_display_all() {
    for comp in StackComponent::all() {
        let display = format!("{}", comp);
        assert!(display.contains(comp.name()));
        assert!(display.contains(comp.english()));
    }
}

#[test]
fn test_model_stage_display_all() {
    let stages = [
        ModelStage::Development,
        ModelStage::Staging,
        ModelStage::Production,
        ModelStage::Archived,
    ];
    for stage in stages {
        let display = format!("{}", stage);
        assert_eq!(display, stage.name());
    }
}

#[test]
fn test_health_status_display_all() {
    let statuses = [
        HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Unhealthy,
        HealthStatus::Unknown,
    ];
    for status in statuses {
        let display = format!("{}", status);
        assert_eq!(display, status.name());
    }
}

#[test]
fn test_stack_health_only_degraded_components() {
    let mut health = StackHealth::new();
    health.set_component(
        StackComponent::Aprender,
        ComponentHealth::degraded("1.0.0", "high latency"),
    );
    health.set_component(
        StackComponent::Pacha,
        ComponentHealth::degraded("1.0.0", "disk full"),
    );
    // All degraded, none unhealthy -> overall should be Degraded
    assert_eq!(health.overall, HealthStatus::Degraded);
    assert!(!health.is_healthy());
}

#[test]
fn test_stack_health_mixed_healthy_degraded() {
    let mut health = StackHealth::new();
    health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0.0"));
    health.set_component(
        StackComponent::Pacha,
        ComponentHealth::degraded("1.0.0", "slow"),
    );
    // Some healthy, some degraded -> overall should be Degraded
    assert_eq!(health.overall, HealthStatus::Degraded);
}

#[test]
fn test_stack_health_clear_and_recheck() {
    let mut health = StackHealth::new();
    health.set_component(StackComponent::Aprender, ComponentHealth::healthy("1.0.0"));
    assert!(health.is_healthy());

    // Replace with unhealthy
    health.set_component(
        StackComponent::Aprender,
        ComponentHealth::unhealthy("crashed"),
    );
    assert_eq!(health.overall, HealthStatus::Unhealthy);
}

#[test]
fn test_model_version_multiple_tags() {
    let version = ModelVersion::new("1.0.0", [0u8; 32])
        .with_tag("classification")
        .with_tag("nlp")
        .with_tag("transformer");

    assert_eq!(version.tags.len(), 3);
    assert!(version.tags.contains(&"nlp".to_string()));
}

#[test]
fn test_model_version_empty_metadata() {
    let version = ModelVersion::new("1.0.0", [0u8; 32]);
    assert!(version.metadata.is_empty());
}

#[test]
fn test_model_version_quality_boundary() {
    // Exactly at 85.0 boundary
    let version = ModelVersion::new("1.0.0", [0u8; 32])
        .with_stage(ModelStage::Production)
        .with_quality_score(85.0);
    assert!(version.is_production_ready());

    // Just below boundary
    let version = ModelVersion::new("1.0.0", [0u8; 32])
        .with_stage(ModelStage::Production)
        .with_quality_score(84.99);
    assert!(!version.is_production_ready());
}

#[test]
fn test_inference_config_with_no_metrics_or_health() {
    let mut config = InferenceConfig::new("model.apr");
    config.metrics_path = None;
    config.health_path = None;
    assert!(config.metrics_path.is_none());
    assert!(config.health_path.is_none());
}

#[test]
fn test_format_compatibility_apr_major_mismatch() {
    let compat = FormatCompatibility::current();
    assert!(!compat.is_apr_compatible(0, 1)); // Major 0 != 1
    assert!(!compat.is_apr_compatible(2, 0)); // Major 2 != 1
}

#[test]
fn test_format_compatibility_ald_major_mismatch() {
    let compat = FormatCompatibility::current();
    assert!(!compat.is_ald_compatible(0, 1)); // Major 0 != 1
    assert!(!compat.is_ald_compatible(2, 0)); // Major 2 != 1
}

#[test]
fn test_derivation_merge_empty_parents() {
    let deriv = DerivationType::Merge {
        parent_hashes: vec![],
        method: "empty".into(),
    };
    assert!(deriv.parent_hashes().is_empty());
    assert!(deriv.is_derived());
}

#[test]
fn test_derivation_merge_many_parents() {
    let parents = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
    let deriv = DerivationType::Merge {
        parent_hashes: parents.clone(),
        method: "ensemble".into(),
    };
    assert_eq!(deriv.parent_hashes().len(), 4);
}

#[test]
fn test_component_health_degraded_has_version_and_error() {
    let health = ComponentHealth::degraded("2.0.0", "high memory");
    assert_eq!(health.version, Some("2.0.0".into()));
    assert_eq!(health.error, Some("high memory".into()));
    assert_eq!(health.status, HealthStatus::Degraded);
}

#[test]
fn test_model_version_hash_all_ff() {
    let hash = [0xFF; 32];
    let version = ModelVersion::new("1.0.0", hash);
    let hex = version.hash_hex();
    assert!(hex.chars().all(|c| c == 'f'));
    assert_eq!(hex.len(), 64);
}

#[test]
fn test_model_version_hash_mixed() {
    let mut hash = [0u8; 32];
    hash[0] = 0x12;
    hash[1] = 0x34;
    hash[31] = 0xAB;
    let version = ModelVersion::new("1.0.0", hash);
    let hex = version.hash_hex();
    assert!(hex.starts_with("1234"));
    assert!(hex.ends_with("ab"));
}

#[test]
fn test_stack_component_eq() {
    assert_eq!(StackComponent::Aprender, StackComponent::Aprender);
    assert_ne!(StackComponent::Aprender, StackComponent::Pacha);
}

#[test]
fn test_health_status_eq() {
    assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
    assert_ne!(HealthStatus::Healthy, HealthStatus::Degraded);
}

#[test]
fn test_model_stage_eq() {
    assert_eq!(ModelStage::Production, ModelStage::Production);
    assert_ne!(ModelStage::Production, ModelStage::Staging);
}

#[test]
fn test_derivation_type_eq() {
    assert_eq!(DerivationType::Original, DerivationType::Original);
    let d1 = DerivationType::FineTune {
        parent_hash: [0; 32],
        epochs: 10,
    };
    let d2 = DerivationType::FineTune {
        parent_hash: [0; 32],
        epochs: 10,
    };
    let d3 = DerivationType::FineTune {
        parent_hash: [1; 32],
        epochs: 10,
    };
    assert_eq!(d1, d2);
    assert_ne!(d1, d3);
}

#[test]
fn test_inference_config_model_path() {
    let config = InferenceConfig::new("/path/to/model.apr");
    assert_eq!(config.model_path, PathBuf::from("/path/to/model.apr"));
}

#[test]
fn test_stack_health_timestamp() {
    let health = StackHealth::new();
    assert!(!health.checked_at.is_empty());
}

#[test]
fn test_model_version_created_at_default() {
    let version = ModelVersion::new("1.0.0", [0u8; 32]);
    assert!(!version.created_at.is_empty());
}

#[test]
fn test_model_version_size_default() {
    let version = ModelVersion::new("1.0.0", [0u8; 32]);
    assert_eq!(version.size_bytes, 0);
}

#[test]
fn test_derivation_finetune_details() {
    let parent = [0xAA; 32];
    let deriv = DerivationType::FineTune {
        parent_hash: parent,
        epochs: 100,
    };
    if let DerivationType::FineTune {
        epochs,
        parent_hash,
    } = deriv
    {
        assert_eq!(epochs, 100);
        assert_eq!(parent_hash, parent);
    } else {
        panic!("Expected FineTune");
    }
}

#[test]
fn test_derivation_distillation_details() {
    let teacher = [0xBB; 32];
    let deriv = DerivationType::Distillation {
        teacher_hash: teacher,
        temperature: 1.5,
    };
    if let DerivationType::Distillation {
        temperature,
        teacher_hash,
    } = deriv
    {
        assert!((temperature - 1.5).abs() < 1e-6);
        assert_eq!(teacher_hash, teacher);
    } else {
        panic!("Expected Distillation");
    }
}

#[test]
fn test_derivation_quantize_details() {
    let parent = [0xCC; 32];
    let deriv = DerivationType::Quantize {
        parent_hash: parent,
        quant_type: QuantizationType::Int4,
    };
    if let DerivationType::Quantize {
        quant_type,
        parent_hash,
    } = deriv
    {
        assert_eq!(quant_type, QuantizationType::Int4);
        assert_eq!(parent_hash, parent);
    } else {
        panic!("Expected Quantize");
    }
}
