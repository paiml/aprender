use super::*;

#[test]
fn test_model_id_display() {
    let id = ModelId::new("Qwen", "Qwen2.5-Coder-1.5B");
    assert_eq!(id.to_string(), "Qwen/Qwen2.5-Coder-1.5B");

    let id_with_variant = ModelId::with_variant("Qwen", "Qwen2.5-Coder-1.5B", "Instruct");
    assert_eq!(
        id_with_variant.to_string(),
        "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    );
}

#[test]
fn test_size_category_memory() {
    assert_eq!(SizeCategory::Tiny.memory_f32_gb(), 2);
    assert_eq!(SizeCategory::Small.memory_f32_gb(), 6);
    assert_eq!(SizeCategory::Large.memory_f32_gb(), 30);
}

#[test]
fn test_registry_with_defaults() {
    let registry = ModelRegistry::with_defaults();
    assert!(!registry.is_empty());
    assert!(registry.len() >= 90);
}

#[test]
fn test_registry_by_size() {
    let registry = ModelRegistry::with_defaults();
    let small_models = registry.by_size(SizeCategory::Small);
    assert!(!small_models.is_empty());
    for model in small_models {
        assert_eq!(model.size, SizeCategory::Small);
    }
}

#[test]
fn test_size_category_approx_params() {
    assert_eq!(SizeCategory::Tiny.approx_params(), 500_000_000);
    assert_eq!(SizeCategory::Small.approx_params(), 1_500_000_000);
    assert_eq!(SizeCategory::Medium.approx_params(), 3_500_000_000);
    assert_eq!(SizeCategory::Large.approx_params(), 7_500_000_000);
    assert_eq!(SizeCategory::XLarge.approx_params(), 13_500_000_000);
    assert_eq!(SizeCategory::Huge.approx_params(), 70_000_000_000);
}

#[test]
fn test_size_category_memory_q4k() {
    assert_eq!(SizeCategory::Tiny.memory_q4k_gb(), 0);
    assert_eq!(SizeCategory::Small.memory_q4k_gb(), 0);
    assert_eq!(SizeCategory::Medium.memory_q4k_gb(), 1);
    assert_eq!(SizeCategory::Large.memory_q4k_gb(), 3);
    assert_eq!(SizeCategory::XLarge.memory_q4k_gb(), 6);
    assert_eq!(SizeCategory::Huge.memory_q4k_gb(), 35);
}

#[test]
fn test_model_id_hf_repo() {
    let id = ModelId::new("org", "model");
    assert_eq!(id.hf_repo(), "org/model");
}

#[test]
fn test_model_id_clone() {
    let id = ModelId::new("org", "model");
    let cloned = id.clone();
    assert_eq!(cloned.org, id.org);
    assert_eq!(cloned.name, id.name);
}

#[test]
fn test_model_id_eq() {
    let id1 = ModelId::new("org", "model");
    let id2 = ModelId::new("org", "model");
    assert_eq!(id1, id2);
}

#[test]
fn test_model_capabilities_default() {
    let caps = ModelCapabilities::default();
    assert!(caps.arithmetic);
    assert!(caps.instruction_following);
    assert!(!caps.code_completion);
    assert!(caps.multi_turn);
}

#[test]
fn test_registry_new() {
    let registry = ModelRegistry::new();
    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
}

#[test]
fn test_registry_add_and_get() {
    let mut registry = ModelRegistry::new();
    let metadata = ModelMetadata {
        id: ModelId::new("test", "model"),
        size: SizeCategory::Small,
        architecture: "test".to_string(),
        quantizations: vec!["q4_k_m".to_string()],
        has_chat_template: true,
        supports_system_prompt: true,
        capabilities: ModelCapabilities::default(),
    };

    registry.add(metadata);
    assert_eq!(registry.len(), 1);

    let retrieved = registry.get("test/model");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().architecture, "test");
}

#[test]
fn test_registry_get_nonexistent() {
    let registry = ModelRegistry::new();
    assert!(registry.get("nonexistent/model").is_none());
}

#[test]
fn test_registry_all() {
    let registry = ModelRegistry::with_defaults();
    let all = registry.all();
    assert_eq!(all.len(), registry.len());
}

#[test]
fn test_model_metadata_clone() {
    let metadata = ModelMetadata {
        id: ModelId::new("test", "model"),
        size: SizeCategory::Small,
        architecture: "test".to_string(),
        quantizations: vec!["q4_k_m".to_string()],
        has_chat_template: true,
        supports_system_prompt: true,
        capabilities: ModelCapabilities::default(),
    };
    let cloned = metadata.clone();
    assert_eq!(cloned.id, metadata.id);
    assert_eq!(cloned.architecture, metadata.architecture);
}

#[test]
fn test_size_category_debug() {
    let size = SizeCategory::Large;
    let debug_str = format!("{size:?}");
    assert!(debug_str.contains("Large"));
}

#[test]
fn test_model_id_serialize() {
    let id = ModelId::new("test", "model");
    let json = serde_json::to_string(&id).expect("serialize");
    assert!(json.contains("test"));
    assert!(json.contains("model"));
}

#[test]
fn test_size_category_eq_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(SizeCategory::Small);
    set.insert(SizeCategory::Medium);
    set.insert(SizeCategory::Small); // duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn test_registry_default() {
    let registry = ModelRegistry::default();
    assert!(registry.is_empty());
}

#[test]
fn test_model_id_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ModelId::new("org1", "model1"));
    set.insert(ModelId::new("org2", "model2"));
    set.insert(ModelId::new("org1", "model1")); // duplicate
    assert_eq!(set.len(), 2);
}

// --- Mutation-killing tests for hf_repo ---
#[test]
fn test_hf_repo_format_with_slash() {
    let id = ModelId::new("MyOrg", "MyModel");
    let repo = id.hf_repo();
    assert!(!repo.is_empty());
    assert!(repo.contains('/'));
    assert_eq!(repo, "MyOrg/MyModel");
    assert!(repo.starts_with("MyOrg"));
    assert!(repo.ends_with("MyModel"));
}

#[test]
fn test_hf_repo_with_variant_format() {
    let id = ModelId::with_variant("Org", "Model", "Variant");
    let repo = id.hf_repo();
    assert!(!repo.is_empty());
    assert!(repo.contains('/'));
    assert!(repo.contains('-'));
    assert_eq!(repo, "Org/Model-Variant");
}

#[test]
fn test_hf_repo_variant_none_uses_name_only() {
    let id = ModelId::new("X", "Y");
    assert!(id.variant.is_none());
    assert_eq!(id.hf_repo(), "X/Y");
    assert!(!id.hf_repo().contains('-'));
}

#[test]
fn test_hf_repo_variant_some_appends_dash() {
    let id = ModelId::with_variant("X", "Y", "Z");
    assert!(id.variant.is_some());
    assert_eq!(id.hf_repo(), "X/Y-Z");
    assert!(id.hf_repo().contains('-'));
}

// --- Mutation-killing tests for SizeCategory memory calculations ---
#[test]
fn test_size_memory_f32_nonzero() {
    // Verify calculations don't return zero for all
    assert!(SizeCategory::Tiny.memory_f32_gb() >= 1);
    assert!(SizeCategory::Small.memory_f32_gb() > SizeCategory::Tiny.memory_f32_gb());
    assert!(SizeCategory::Large.memory_f32_gb() > SizeCategory::Medium.memory_f32_gb());
}

#[test]
fn test_size_memory_q4k_ordering() {
    // Larger models should have more memory
    assert!(SizeCategory::Huge.memory_q4k_gb() > SizeCategory::XLarge.memory_q4k_gb());
    assert!(SizeCategory::XLarge.memory_q4k_gb() > SizeCategory::Large.memory_q4k_gb());
}

#[test]
fn test_approx_params_ordering() {
    // Strict ordering from tiny to huge
    let tiny = SizeCategory::Tiny.approx_params();
    let small = SizeCategory::Small.approx_params();
    let medium = SizeCategory::Medium.approx_params();
    let large = SizeCategory::Large.approx_params();
    let xlarge = SizeCategory::XLarge.approx_params();
    let huge = SizeCategory::Huge.approx_params();

    assert!(tiny < small);
    assert!(small < medium);
    assert!(medium < large);
    assert!(large < xlarge);
    assert!(xlarge < huge);
}

// --- Mutation-killing tests for ModelCapabilities ---
#[test]
fn test_capabilities_default_values_explicit() {
    let caps = ModelCapabilities::default();
    // Explicitly test all boolean values
    assert!(caps.arithmetic, "arithmetic should be true");
    assert!(
        caps.instruction_following,
        "instruction_following should be true"
    );
    assert!(!caps.code_completion, "code_completion should be false");
    assert!(caps.multi_turn, "multi_turn should be true");
}
