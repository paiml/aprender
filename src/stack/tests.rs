pub(crate) use super::*;
#[test]
fn test_stack_component() {
    assert_eq!(StackComponent::Aprender.name(), "aprender");
    assert_eq!(StackComponent::Aprender.english(), "to learn");
    assert_eq!(StackComponent::Aprender.format(), Some(".apr"));
    assert_eq!(
        StackComponent::Aprender.magic(),
        Some([0x41, 0x50, 0x52, 0x4E])
    );
}

#[test]
fn test_stack_component_all() {
    let all = StackComponent::all();
    assert_eq!(all.len(), 6);
    assert_eq!(all[0], StackComponent::Alimentar);
    assert_eq!(all[5], StackComponent::Batuta);
}

#[test]
fn test_stack_component_display() {
    let comp = StackComponent::Realizar;
    assert_eq!(format!("{comp}"), "realizar (to accomplish)");
}

#[test]
fn test_derivation_type_original() {
    let deriv = DerivationType::Original;
    assert_eq!(deriv.type_name(), "original");
    assert!(!deriv.is_derived());
    assert!(deriv.parent_hashes().is_empty());
}

#[test]
fn test_derivation_type_fine_tune() {
    let parent = [1u8; 32];
    let deriv = DerivationType::FineTune {
        parent_hash: parent,
        epochs: 10,
    };

    assert_eq!(deriv.type_name(), "fine-tune");
    assert!(deriv.is_derived());
    assert_eq!(deriv.parent_hashes(), vec![parent]);
}

#[test]
fn test_derivation_type_merge() {
    let parents = vec![[1u8; 32], [2u8; 32]];
    let deriv = DerivationType::Merge {
        parent_hashes: parents.clone(),
        method: "TIES".into(),
    };

    assert_eq!(deriv.type_name(), "merge");
    assert_eq!(deriv.parent_hashes(), parents);
}

#[test]
fn test_quantization_type() {
    assert_eq!(QuantizationType::Int8.bits(), 8);
    assert_eq!(QuantizationType::Int4.bits(), 4);
    assert_eq!(QuantizationType::Float16.bits(), 16);
    assert_eq!(QuantizationType::Int8.name(), "int8");
}

#[test]
fn test_model_stage_transitions() {
    assert!(ModelStage::Development.can_transition_to(ModelStage::Staging));
    assert!(ModelStage::Staging.can_transition_to(ModelStage::Production));
    assert!(ModelStage::Production.can_transition_to(ModelStage::Archived));
    assert!(!ModelStage::Archived.can_transition_to(ModelStage::Development));
    assert!(ModelStage::Development.can_transition_to(ModelStage::Development));
}

#[test]
fn test_model_version() {
    let version = ModelVersion::new("1.0.0", [0u8; 32])
        .with_stage(ModelStage::Production)
        .with_size(1_000_000)
        .with_quality_score(95.0)
        .with_tag("classification");

    assert_eq!(version.version, "1.0.0");
    assert_eq!(version.stage, ModelStage::Production);
    assert_eq!(version.size_bytes, 1_000_000);
    assert_eq!(version.quality_score, Some(95.0));
    assert!(version.tags.contains(&"classification".to_string()));
    assert!(version.is_production_ready());
}

#[test]
fn test_model_version_not_production_ready() {
    let version = ModelVersion::new("1.0.0", [0u8; 32])
        .with_stage(ModelStage::Development)
        .with_quality_score(95.0);
    assert!(!version.is_production_ready()); // wrong stage

    let version = ModelVersion::new("1.0.0", [0u8; 32])
        .with_stage(ModelStage::Production)
        .with_quality_score(70.0);
    assert!(!version.is_production_ready()); // low quality
}

#[test]
fn test_model_version_hash_hex() {
    let hash = [
        0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00,
    ];
    let version = ModelVersion::new("1.0.0", hash);
    assert!(version.hash_hex().starts_with("abcdef01234567890000"));
}

#[test]
fn test_inference_config() {
    let config = InferenceConfig::new("model.apr")
        .with_port(9000)
        .with_batch_size(64)
        .with_timeout_ms(200)
        .without_cors();

    assert_eq!(config.port, 9000);
    assert_eq!(config.max_batch_size, 64);
    assert_eq!(config.timeout_ms, 200);
    assert!(!config.enable_cors);
    assert_eq!(config.predict_url(), "http://localhost:9000/predict");
}

#[test]
fn test_inference_config_default() {
    let config = InferenceConfig::default();
    assert_eq!(config.port, 8080);
    assert!(config.enable_cors);
}

#[test]
fn test_stack_health() {
    let mut health = StackHealth::new();
    assert_eq!(health.overall, HealthStatus::Unknown);

    health.set_component(StackComponent::Aprender, ComponentHealth::healthy("0.15.0"));
    health.set_component(StackComponent::Pacha, ComponentHealth::healthy("1.0.0"));

    assert!(health.is_healthy());
    assert_eq!(health.overall, HealthStatus::Healthy);
}

#[test]
fn test_stack_health_degraded() {
    let mut health = StackHealth::new();

    health.set_component(StackComponent::Aprender, ComponentHealth::healthy("0.15.0"));
    health.set_component(
        StackComponent::Pacha,
        ComponentHealth::degraded("1.0.0", "high latency"),
    );

    assert!(!health.is_healthy());
    assert_eq!(health.overall, HealthStatus::Degraded);
}

#[test]
fn test_stack_health_unhealthy() {
    let mut health = StackHealth::new();

    health.set_component(StackComponent::Aprender, ComponentHealth::healthy("0.15.0"));
    health.set_component(
        StackComponent::Pacha,
        ComponentHealth::unhealthy("connection refused"),
    );

    assert!(!health.is_healthy());
    assert_eq!(health.overall, HealthStatus::Unhealthy);
}

#[test]
fn test_component_health() {
    let healthy = ComponentHealth::healthy("1.0.0").with_response_time(50);
    assert_eq!(healthy.status, HealthStatus::Healthy);
    assert_eq!(healthy.response_time_ms, Some(50));

    let unhealthy = ComponentHealth::unhealthy("timeout");
    assert_eq!(unhealthy.status, HealthStatus::Unhealthy);
    assert_eq!(unhealthy.error, Some("timeout".into()));
}

#[test]
fn test_health_status() {
    assert!(HealthStatus::Healthy.is_operational());
    assert!(HealthStatus::Degraded.is_operational());
    assert!(!HealthStatus::Unhealthy.is_operational());
    assert!(!HealthStatus::Unknown.is_operational());

    assert_eq!(HealthStatus::Healthy.name(), "healthy");
    assert_eq!(format!("{}", HealthStatus::Degraded), "degraded");
}

#[test]
fn test_format_compatibility() {
    let compat = FormatCompatibility::current();

    assert!(compat.is_apr_compatible(1, 0));
    assert!(!compat.is_apr_compatible(2, 0));

    assert!(compat.is_ald_compatible(1, 0));
    assert!(compat.is_ald_compatible(1, 2));
    assert!(!compat.is_ald_compatible(1, 3));
}

#[test]
fn test_model_stage_display() {
    assert_eq!(format!("{}", ModelStage::Production), "production");
}

#[test]
fn test_derivation_distillation() {
    let teacher = [0xAA; 32];
    let deriv = DerivationType::Distillation {
        teacher_hash: teacher,
        temperature: 2.0,
    };

    assert_eq!(deriv.type_name(), "distillation");
    assert_eq!(deriv.parent_hashes(), vec![teacher]);
}

#[test]
fn test_derivation_quantize() {
    let parent = [0xBB; 32];
    let deriv = DerivationType::Quantize {
        parent_hash: parent,
        quant_type: QuantizationType::Int8,
    };

    assert_eq!(deriv.type_name(), "quantize");
    assert_eq!(deriv.parent_hashes(), vec![parent]);
}

#[test]
fn test_derivation_prune() {
    let parent = [0xCC; 32];
    let deriv = DerivationType::Prune {
        parent_hash: parent,
        sparsity: 0.5,
    };

    assert_eq!(deriv.type_name(), "prune");
    assert_eq!(deriv.parent_hashes(), vec![parent]);
}

// =========================================================================
// Extended coverage tests
// =========================================================================

#[test]
fn test_stack_component_all_names() {
    assert_eq!(StackComponent::Alimentar.name(), "alimentar");
    assert_eq!(StackComponent::Pacha.name(), "pacha");
    assert_eq!(StackComponent::Realizar.name(), "realizar");
    assert_eq!(StackComponent::Presentar.name(), "presentar");
    assert_eq!(StackComponent::Batuta.name(), "batuta");
}

#[test]
fn test_stack_component_all_english() {
    assert_eq!(StackComponent::Alimentar.english(), "to feed");
    assert_eq!(StackComponent::Pacha.english(), "earth/universe");
    assert_eq!(StackComponent::Presentar.english(), "to present");
    assert_eq!(StackComponent::Batuta.english(), "baton");
}

#[test]
fn test_stack_component_descriptions() {
    assert!(StackComponent::Alimentar
        .description()
        .contains("Data loading"));
    assert!(StackComponent::Aprender
        .description()
        .contains("Machine learning"));
    assert!(StackComponent::Pacha.description().contains("registry"));
    assert!(StackComponent::Realizar.description().contains("inference"));
    assert!(StackComponent::Presentar
        .description()
        .contains("visualization"));
    assert!(StackComponent::Batuta
        .description()
        .contains("Orchestration"));
}

#[test]
fn test_stack_component_formats_none() {
    assert_eq!(StackComponent::Pacha.format(), None);
    assert_eq!(StackComponent::Realizar.format(), None);
    assert_eq!(StackComponent::Presentar.format(), None);
}

#[test]
fn test_stack_component_formats_some() {
    assert_eq!(StackComponent::Alimentar.format(), Some(".ald"));
    assert_eq!(StackComponent::Batuta.format(), Some(".bat"));
}

#[test]
fn test_stack_component_magic_none() {
    assert_eq!(StackComponent::Pacha.magic(), None);
    assert_eq!(StackComponent::Realizar.magic(), None);
    assert_eq!(StackComponent::Presentar.magic(), None);
}

#[test]
fn test_stack_component_magic_some() {
    assert_eq!(
        StackComponent::Alimentar.magic(),
        Some([0x41, 0x4C, 0x44, 0x46])
    ); // "ALDF"
    assert_eq!(
        StackComponent::Batuta.magic(),
        Some([0x42, 0x41, 0x54, 0x41])
    ); // "BATA"
}

#[test]
fn test_quantization_type_all_bits() {
    assert_eq!(QuantizationType::BFloat16.bits(), 16);
    assert_eq!(QuantizationType::Dynamic.bits(), 8);
    assert_eq!(QuantizationType::QAT.bits(), 8);
}

#[test]
fn test_quantization_type_all_names() {
    assert_eq!(QuantizationType::Int4.name(), "int4");
    assert_eq!(QuantizationType::Float16.name(), "fp16");
    assert_eq!(QuantizationType::BFloat16.name(), "bf16");
    assert_eq!(QuantizationType::Dynamic.name(), "dynamic");
    assert_eq!(QuantizationType::QAT.name(), "qat");
}

#[test]
fn test_model_stage_default() {
    let stage: ModelStage = Default::default();
    assert_eq!(stage, ModelStage::Development);
}

#[test]
fn test_model_stage_all_transitions() {
    // Development transitions
    assert!(ModelStage::Development.can_transition_to(ModelStage::Archived));

    // Staging transitions
    assert!(ModelStage::Staging.can_transition_to(ModelStage::Staging));
    assert!(ModelStage::Staging.can_transition_to(ModelStage::Development));

    // Production transitions
    assert!(ModelStage::Production.can_transition_to(ModelStage::Production));
    assert!(!ModelStage::Production.can_transition_to(ModelStage::Development));
    assert!(!ModelStage::Production.can_transition_to(ModelStage::Staging));

    // Archived - cannot transition anywhere
    assert!(!ModelStage::Archived.can_transition_to(ModelStage::Archived));
    assert!(!ModelStage::Archived.can_transition_to(ModelStage::Production));
    assert!(!ModelStage::Archived.can_transition_to(ModelStage::Staging));
}

#[test]
fn test_model_stage_all_names() {
    assert_eq!(ModelStage::Development.name(), "development");
    assert_eq!(ModelStage::Staging.name(), "staging");
    assert_eq!(ModelStage::Archived.name(), "archived");
}

#[test]
fn test_model_version_with_derivation() {
    let parent = [0xDD; 32];
    let deriv = DerivationType::FineTune {
        parent_hash: parent,
        epochs: 5,
    };
    let version = ModelVersion::new("2.0.0", [0u8; 32]).with_derivation(deriv);
    assert!(version.derivation.is_derived());
}

#[test]
fn test_model_version_no_quality_score_not_production_ready() {
    let version = ModelVersion::new("1.0.0", [0u8; 32]).with_stage(ModelStage::Production);
    // No quality score set
    assert!(!version.is_production_ready());
}

#[test]
fn test_inference_config_batch_predict_url() {
    let config = InferenceConfig::new("model.apr").with_port(3000);
    assert_eq!(
        config.batch_predict_url(),
        "http://localhost:3000/batch_predict"
    );
}

#[test]
fn test_stack_health_default() {
    let health: StackHealth = Default::default();
    assert_eq!(health.overall, HealthStatus::Unknown);
    assert!(health.components.is_empty());
}

#[test]
fn test_format_compatibility_default() {
    let compat: FormatCompatibility = Default::default();
    assert_eq!(compat.apr_version, (1, 0));
    assert!(compat.compatible);
}

#[test]
fn test_health_status_default() {
    let status: HealthStatus = Default::default();
    assert_eq!(status, HealthStatus::Unknown);
}

#[test]
fn test_health_status_all_names() {
    assert_eq!(HealthStatus::Unhealthy.name(), "unhealthy");
    assert_eq!(HealthStatus::Unknown.name(), "unknown");
}

#[test]
fn test_stack_component_debug() {
    let comp = StackComponent::Aprender;
    let debug_str = format!("{:?}", comp);
    assert!(debug_str.contains("Aprender"));
}

#[test]
fn test_derivation_type_debug() {
    let deriv = DerivationType::Original;
    let debug_str = format!("{:?}", deriv);
    assert!(debug_str.contains("Original"));
}

#[test]
fn test_quantization_type_debug() {
    let qt = QuantizationType::Int8;
    let debug_str = format!("{:?}", qt);
    assert!(debug_str.contains("Int8"));
}

#[path = "tests_part_02.rs"]

mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
