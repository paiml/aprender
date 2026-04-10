use super::*;

/// Sample YAML family contract used across all test cases
const SAMPLE_YAML: &str = r#"
family: qwen2
display_name: "Qwen2 / Qwen2.5-Coder"
vendor: Alibaba
architectures:
  - Qwen2ForCausalLM
hf_pattern: "Qwen/Qwen2*"

size_variants:
  0.5b:
    parameters: "0.5B"
    hidden_dim: 896
    num_layers: 24
    num_heads: 14
    num_kv_heads: 2
    intermediate_dim: 4864
    vocab_size: 151936
  1.5b:
    parameters: "1.5B"
    hidden_dim: 1536
    num_layers: 28
    num_heads: 12
    num_kv_heads: 2

constraints:
  attention_type: gqa
  activation: silu
  norm_type: rmsnorm
  has_bias: true
  tied_embeddings: false
  positional_encoding: rope
  mlp_type: swiglu

tensor_template:
  embedding: "model.embed_tokens.weight"
  lm_head: "lm_head.weight"
  final_norm: "model.norm.weight"
  per_layer:
    q_proj: "model.layers.{n}.self_attn.q_proj.weight"
    k_proj: "model.layers.{n}.self_attn.k_proj.weight"
    input_layernorm: "model.layers.{n}.input_layernorm.weight"

quantizations:
  - q4_k_m
  - q5_k_m
  - q6_k

certification:
  playbook_path: "../apr-model-qa-playbook/playbooks/models/qwen2.5-coder-{size}.playbook.yaml"
  csv_family_key: "qwen-coder"
  size_categories:
    0.5b: tiny
    1.5b: small
    3b: small
    7b: medium
"#;

#[test]
fn test_parse_family_contract() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    assert_eq!(contract.family, "qwen2");
    assert_eq!(
        contract.display_name,
        Some("Qwen2 / Qwen2.5-Coder".to_string())
    );
    assert_eq!(contract.vendor, Some("Alibaba".to_string()));
}

#[test]
fn test_size_variants() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    assert_eq!(contract.size_variants.len(), 2);

    let v05b = contract.get_size_variant("0.5b").expect("0.5b");
    assert_eq!(v05b.hidden_dim, 896);
    assert_eq!(v05b.num_layers, 24);
    assert_eq!(v05b.num_heads, Some(14));
    assert_eq!(v05b.num_kv_heads, Some(2));

    let v15b = contract.get_size_variant("1.5b").expect("1.5b");
    assert_eq!(v15b.hidden_dim, 1536);
    assert_eq!(v15b.num_layers, 28);
}

#[test]
fn test_constraints() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    let constraints = contract.constraints.expect("constraints");

    assert_eq!(constraints.attention_type, Some("gqa".to_string()));
    assert_eq!(constraints.activation, Some("silu".to_string()));
    assert_eq!(constraints.has_bias, Some(true));
}

#[test]
fn test_tensor_template() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    let template = contract.tensor_template.expect("template");

    assert_eq!(
        template.embedding,
        Some("model.embed_tokens.weight".to_string())
    );
    assert_eq!(template.lm_head, Some("lm_head.weight".to_string()));
    assert!(template.per_layer.contains_key("q_proj"));
}

#[test]
fn test_required_tensors() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    let template = contract.tensor_template.expect("template");

    // With 2 layers
    let tensors = template.required_tensors(2);

    // Should have embedding, lm_head, final_norm (3)
    // Plus per_layer tensors for 2 layers (3 per layer * 2 = 6)
    // Total: 9
    assert_eq!(tensors.len(), 9);
    assert!(tensors.contains(&"model.embed_tokens.weight".to_string()));
    assert!(tensors.contains(&"lm_head.weight".to_string()));
    assert!(tensors.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
    assert!(tensors.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
}

#[test]
fn test_certification_config() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    let cert = contract.certification.expect("certification");

    assert_eq!(cert.csv_family_key, Some("qwen-coder".to_string()));
    assert_eq!(cert.size_categories.get("0.5b"), Some(&"tiny".to_string()));
    assert_eq!(cert.size_categories.get("1.5b"), Some(&"small".to_string()));
}

#[test]
fn test_get_size_category() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");

    assert_eq!(contract.get_size_category("0.5b"), Some("tiny"));
    assert_eq!(contract.get_size_category("1.5b"), Some("small"));
    assert_eq!(contract.get_size_category("7b"), Some("medium"));
    assert_eq!(contract.get_size_category("100b"), None);
}

#[test]
fn test_resolve_playbook_path() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");

    let path = contract.resolve_playbook_path("0.5b", "mvp");
    assert_eq!(
        path,
        Some(
            "../apr-model-qa-playbook/playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml"
                .to_string()
        )
    );
}

#[test]
fn test_required_tensors_for_size() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");

    // 0.5b has 24 layers
    let tensors = contract.required_tensors_for_size("0.5b");
    // 3 global + 3 per layer * 24 layers = 75
    assert_eq!(tensors.len(), 75);
}

#[test]
fn test_family_registry_new() {
    let registry = FamilyRegistry::new();
    assert!(registry.families().is_empty());
}

#[test]
fn test_family_registry_with_path() {
    let registry = FamilyRegistry::with_path("/custom/path");
    assert!(!registry.aprender_available()); // path doesn't exist
}

#[test]
fn test_family_registry_load_all() {
    let mut registry = FamilyRegistry::new();

    // May or may not have aprender available
    let result = registry.load_all();
    assert!(result.is_ok());
}

#[test]
fn test_architectures() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    assert_eq!(contract.architectures.len(), 1);
    assert_eq!(contract.architectures[0], "Qwen2ForCausalLM");
}

#[test]
fn test_hf_pattern() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    assert_eq!(contract.hf_pattern, Some("Qwen/Qwen2*".to_string()));
}

#[test]
fn test_quantizations() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    assert_eq!(contract.quantizations.len(), 3);
    assert!(contract.quantizations.contains(&"q4_k_m".to_string()));
}

#[test]
fn test_missing_optional_fields() {
    let minimal_yaml = r#"
family: minimal
size_variants:
  1b:
    parameters: "1B"
    hidden_dim: 1024
    num_layers: 12
    num_heads: 16
"#;
    let contract = FamilyContract::from_yaml(minimal_yaml).expect("parse");
    assert_eq!(contract.family, "minimal");
    assert!(contract.display_name.is_none());
    assert!(contract.vendor.is_none());
    assert!(contract.constraints.is_none());
    assert!(contract.tensor_template.is_none());
    assert!(contract.certification.is_none());
}

// FALSIFY-FAM-001: Size category alignment
#[test]
fn test_falsify_fam_001_size_category_alignment() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");

    // Verify all size variants have a size category
    for size in contract.size_variants.keys() {
        if let Some(cat) = contract.get_size_category(size) {
            // Category must be one of the valid values
            assert!(
                ["tiny", "small", "medium", "large", "xlarge", "huge"].contains(&cat),
                "Invalid size category '{cat}' for size '{size}'"
            );
        }
    }
}

#[test]
fn test_registry_load_all_with_yaml_dir() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("qwen2.yaml"), SAMPLE_YAML).unwrap();
    std::fs::write(
        dir.path().join("minimal.yaml"),
        "family: minimal\nsize_variants:\n  1b:\n    parameters: \"1B\"\n    hidden_dim: 1024\n    num_layers: 12\n    num_heads: 16\n",
    ).unwrap();
    // Non-YAML files should be skipped
    std::fs::write(dir.path().join("readme.txt"), "not yaml").unwrap();
    // Underscore-prefixed files should be skipped
    std::fs::write(dir.path().join("_schema.yaml"), "not a contract").unwrap();

    let mut registry = FamilyRegistry::with_path(dir.path());
    let count = registry.load_all().unwrap();
    assert_eq!(count, 2);
    assert!(registry.has_family("qwen2"));
    assert!(registry.has_family("minimal"));
    assert!(!registry.has_family("readme"));
    assert_eq!(registry.families().len(), 2);
}

#[test]
fn test_registry_load_family() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("qwen2.yaml"), SAMPLE_YAML).unwrap();

    let mut registry = FamilyRegistry::with_path(dir.path());
    let contract = registry.load_family("qwen2").unwrap();
    assert_eq!(contract.family, "qwen2");

    // Second call should return cached version
    let contract2 = registry.load_family("qwen2").unwrap();
    assert_eq!(contract2.family, "qwen2");
}

#[test]
fn test_registry_load_family_missing() {
    let dir = tempfile::tempdir().unwrap();
    let mut registry = FamilyRegistry::with_path(dir.path());
    let result = registry.load_family("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_registry_get() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("qwen2.yaml"), SAMPLE_YAML).unwrap();

    let mut registry = FamilyRegistry::with_path(dir.path());
    assert!(registry.get("qwen2").is_none());
    registry.load_all().unwrap();
    assert!(registry.get("qwen2").is_some());
    assert_eq!(registry.get("qwen2").unwrap().family, "qwen2");
}

#[test]
fn test_registry_load_all_nonexistent_dir() {
    let mut registry = FamilyRegistry::with_path("/nonexistent/path");
    let count = registry.load_all().unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_registry_load_all_with_invalid_yaml() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("good.yaml"), SAMPLE_YAML).unwrap();
    std::fs::write(dir.path().join("bad.yaml"), "invalid: [[[").unwrap();

    let mut registry = FamilyRegistry::with_path(dir.path());
    let count = registry.load_all().unwrap();
    // Only valid YAML should be loaded
    assert_eq!(count, 1);
    assert!(registry.has_family("qwen2"));
}

#[test]
fn test_constraints_to_arch_constraints() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    let constraints = contract.constraints.expect("constraints");
    let arch = constraints.to_arch_constraints();

    assert_eq!(arch.attention_type, Some("gqa".to_string()));
    assert_eq!(arch.activation, Some("silu".to_string()));
    assert_eq!(arch.norm_type, Some("rmsnorm".to_string()));
    assert_eq!(arch.has_bias, Some(true));
    assert_eq!(arch.tied_embeddings, Some(false));
    assert_eq!(arch.positional_encoding, Some("rope".to_string()));
    assert_eq!(arch.mlp_type, Some("swiglu".to_string()));
}

#[test]
fn test_size_variant_to_arch_size_variant() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    let variant = contract.get_size_variant("1.5b").expect("1.5b");
    let arch = variant.to_arch_size_variant();

    assert_eq!(arch.parameters, "1.5B");
    assert_eq!(arch.hidden_dim, 1536);
    assert_eq!(arch.num_layers, 28);
    assert_eq!(arch.num_heads, Some(12));
    assert_eq!(arch.num_kv_heads, Some(2));
}

#[test]
fn test_arch_constraints_kernel_profile() {
    let contract = FamilyContract::from_yaml(SAMPLE_YAML).expect("parse");
    let variant = contract.get_size_variant("1.5b").expect("1.5b");
    let arch_variant = variant.to_arch_size_variant();
    let constraints = contract.constraints.expect("constraints");
    let arch = constraints.to_arch_constraints();

    let profile = aprender_qa_gen::profile_from_constraints(
        &contract.family,
        &arch,
        arch_variant.max_position_embeddings,
    );

    assert_eq!(profile.family, "qwen2");
    assert!(profile
        .kernel_ops
        .contains(&aprender_qa_gen::KernelOp::GroupedQueryAttention));
    assert!(profile
        .kernel_ops
        .contains(&aprender_qa_gen::KernelOp::RmsNorm));
    assert!(!profile.all_prompts().is_empty());
}
