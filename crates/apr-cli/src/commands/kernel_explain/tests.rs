//! Tests for kernel explainability.

use super::config::{
    detect_constraint_mismatches, enrich_rationale, extract_architecture_display,
    extract_config_mapping, extract_json_string,
};
use super::family::{
    derive_kernel_class, extract_constraints, load_families, yaml_bool, yaml_list, yaml_str,
};
use super::kernel_ops::{kernel_ops_for_class, kernel_ops_for_family};
use super::output::build_json_output;
use super::proof::{proof_status_for_class, proof_status_for_contract, ProofLevel};
use super::resolve::{
    compact_input, family_aliases, normalize_input, resolve_family, resolve_from_config_json,
    strip_arch_suffix,
};
use super::{
    ConfigField, Constraints, FamilyInfo, KernelClass, FAMILY_ALIASES, FAMILY_YAMLS,
    KERNEL_CONTRACTS,
};
use std::collections::BTreeMap;
use std::path::Path;

// ── yaml_str ────────────────────────────────────────────────────────

#[test]
fn yaml_str_plain_value() {
    let yaml = "family: qwen2\ndisplay_name: \"Qwen2 / Qwen2.5-Coder\"\n";
    assert_eq!(yaml_str(yaml, "family"), Some("qwen2".to_string()));
    assert_eq!(
        yaml_str(yaml, "display_name"),
        Some("Qwen2 / Qwen2.5-Coder".to_string())
    );
}

#[test]
fn yaml_str_missing_key() {
    assert_eq!(yaml_str("family: bert\n", "missing"), None);
}

#[test]
fn yaml_str_inline_comment_stripped() {
    let yaml = "attention_type: gqa  # MLA dispatches as GQA\n";
    assert_eq!(yaml_str(yaml, "attention_type"), Some("gqa".to_string()));
}

#[test]
fn yaml_str_quoted_values() {
    assert_eq!(
        yaml_str("name: 'single'\n", "name"),
        Some("single".to_string())
    );
    assert_eq!(
        yaml_str("name: \"double\"\n", "name"),
        Some("double".to_string())
    );
}

#[test]
fn yaml_str_null_returns_none() {
    assert_eq!(yaml_str("val: null\n", "val"), None);
}

#[test]
fn yaml_str_empty_value_returns_none() {
    assert_eq!(yaml_str("val:\n", "val"), None);
}

#[test]
fn yaml_str_value_with_only_comment() {
    // "val: # just a comment" → after stripping comment, value is empty → None
    assert_eq!(yaml_str("val: # just a comment\n", "val"), None);
}

// ── yaml_bool ───────────────────────────────────────────────────────

#[test]
fn yaml_bool_true() {
    assert!(yaml_bool("has_bias: true\n", "has_bias"));
}

#[test]
fn yaml_bool_false() {
    assert!(!yaml_bool("has_bias: false\n", "has_bias"));
}

#[test]
fn yaml_bool_missing_is_false() {
    assert!(!yaml_bool("other: true\n", "has_bias"));
}

// ── yaml_list ───────────────────────────────────────────────────────

#[test]
fn yaml_list_basic() {
    let yaml =
        "architectures:\n  - Qwen2ForCausalLM\n  - Qwen2ForTokenClassification\nother: val\n";
    let items = yaml_list(yaml, "architectures");
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], "Qwen2ForCausalLM");
    assert_eq!(items[1], "Qwen2ForTokenClassification");
}

#[test]
fn yaml_list_stops_at_next_key() {
    let yaml = "items:\n  - a\n  - b\nnext_key: val\n";
    let items = yaml_list(yaml, "items");
    assert_eq!(items.len(), 2);
}

#[test]
fn yaml_list_empty() {
    let yaml = "items:\nother: val\n";
    let items = yaml_list(yaml, "items");
    assert!(items.is_empty());
}

#[test]
fn yaml_list_missing_key() {
    let yaml = "other:\n  - a\n";
    let items = yaml_list(yaml, "items");
    assert!(items.is_empty());
}

#[test]
fn yaml_list_strips_quotes() {
    let yaml = "items:\n  - \"quoted\"\n  - 'single'\n";
    let items = yaml_list(yaml, "items");
    assert_eq!(items[0], "quoted");
    assert_eq!(items[1], "single");
}

// ── extract_constraints ─────────────────────────────────────────────

#[test]
fn extract_constraints_basic() {
    let yaml = "\nconstraints:\n  attention_type: gqa\n  activation: silu\n  norm_type: rmsnorm\n  mlp_type: swiglu\n  positional_encoding: rope\n  has_bias: false\n  tied_embeddings: true\n";
    let c = extract_constraints(yaml);
    assert_eq!(c.attention_type, "gqa");
    assert_eq!(c.activation, "silu");
    assert_eq!(c.norm_type, "rmsnorm");
    assert_eq!(c.mlp_type, "swiglu");
    assert_eq!(c.positional_encoding, "rope");
    assert!(!c.has_bias);
    assert!(c.tied_embeddings);
}

#[test]
fn extract_constraints_missing_section_defaults() {
    let yaml = "family: test\n";
    let c = extract_constraints(yaml);
    assert_eq!(c.attention_type, "");
    assert!(!c.has_bias);
}

#[test]
fn extract_constraints_inline_comments() {
    let yaml = "\nconstraints:\n  attention_type: ssm  # state space model\n  activation: silu\n";
    let c = extract_constraints(yaml);
    assert_eq!(c.attention_type, "ssm");
}

// ── normalize_input ─────────────────────────────────────────────────

#[test]
fn normalize_input_hyphens() {
    assert_eq!(normalize_input("falcon-h1"), "falcon_h1");
}

#[test]
fn normalize_input_dots() {
    assert_eq!(normalize_input("qwen3.5"), "qwen3_5");
}

#[test]
fn normalize_input_uppercase() {
    assert_eq!(normalize_input("Qwen2ForCausalLM"), "qwen2forcausallm");
}

#[test]
fn normalize_input_mixed() {
    assert_eq!(normalize_input("Phi-3.5-mini"), "phi_3_5_mini");
}

#[test]
fn normalize_input_already_normal() {
    assert_eq!(normalize_input("llama"), "llama");
}

// ── compact_input ───────────────────────────────────────────────────

#[test]
fn compact_input_removes_underscores() {
    assert_eq!(compact_input("phi_3"), Some("phi3".to_string()));
    assert_eq!(compact_input("gpt_2"), Some("gpt2".to_string()));
    assert_eq!(compact_input("rwkv_7"), Some("rwkv7".to_string()));
}

#[test]
fn compact_input_no_underscores() {
    assert_eq!(compact_input("llama"), None);
    assert_eq!(compact_input("bert"), None);
}

#[test]
fn compact_input_multiple_underscores() {
    assert_eq!(
        compact_input("qwen3_5_mini"),
        Some("qwen35mini".to_string())
    );
}

// ── strip_arch_suffix ───────────────────────────────────────────────

#[test]
fn strip_forcausallm() {
    assert_eq!(strip_arch_suffix("graniteforcausallm"), "granite");
}

#[test]
fn strip_forconditionalgeneration() {
    assert_eq!(
        strip_arch_suffix("whisperforconditionalgeneration"),
        "whisper"
    );
}

#[test]
fn strip_model_suffix() {
    assert_eq!(strip_arch_suffix("distilbertmodel"), "distilbert");
}

#[test]
fn strip_no_match() {
    assert_eq!(strip_arch_suffix("llama"), "llama");
}

#[test]
fn strip_empty_prefix_skipped() {
    // "forcausallm" alone should NOT strip to empty string
    assert_eq!(strip_arch_suffix("forcausallm"), "forcausallm");
}

// ── derive_kernel_class (all 9 classes) ─────────────────────────────

#[test]
fn derive_class_a_gqa() {
    let c = Constraints {
        attention_type: "gqa".into(),
        activation: "silu".into(),
        norm_type: "rmsnorm".into(),
        mlp_type: "swiglu".into(),
        positional_encoding: "rope".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::A);
}

#[test]
fn derive_class_a_mha_degenerate() {
    // MHA is degenerate GQA — same kernel dispatch
    let c = Constraints {
        attention_type: "mha".into(),
        activation: "silu".into(),
        norm_type: "rmsnorm".into(),
        mlp_type: "swiglu".into(),
        positional_encoding: "rope".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::A);
}

#[test]
fn derive_class_b() {
    let c = Constraints {
        attention_type: "mha".into(),
        activation: "gelu".into(),
        norm_type: "layernorm".into(),
        mlp_type: "gelu_mlp".into(),
        positional_encoding: "absolute".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::B);
}

#[test]
fn derive_class_c() {
    let c = Constraints {
        attention_type: "mqa".into(),
        activation: "gelu".into(),
        norm_type: "layernorm".into(),
        mlp_type: "gelu_mlp".into(),
        positional_encoding: "alibi".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::C);
}

#[test]
fn derive_class_d_gqa_layernorm() {
    let c = Constraints {
        attention_type: "gqa".into(),
        activation: "gelu".into(),
        norm_type: "layernorm".into(),
        mlp_type: "gelu_mlp".into(),
        positional_encoding: "rope".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::D);
}

#[test]
fn derive_class_d_silu_layernorm() {
    let c = Constraints {
        attention_type: "mha".into(),
        activation: "silu".into(),
        norm_type: "layernorm".into(),
        mlp_type: "swiglu".into(),
        positional_encoding: "rope".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::D);
}

#[test]
fn derive_class_f() {
    let c = Constraints {
        attention_type: "gqa".into(),
        activation: "gelu".into(),
        norm_type: "rmsnorm".into(),
        mlp_type: "gated_mlp".into(),
        positional_encoding: "rope".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::F);
}

#[test]
fn derive_class_f_checked_before_b() {
    // F requires RMSNorm+GELU+GatedMlp+RoPE. If MHA+RMSNorm+GELU+GatedMlp+RoPE,
    // should be F (not B, because B requires LayerNorm)
    let c = Constraints {
        attention_type: "mha".into(),
        activation: "gelu".into(),
        norm_type: "rmsnorm".into(),
        mlp_type: "gated_mlp".into(),
        positional_encoding: "rope".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::F);
}

#[test]
fn derive_class_ssm() {
    let c = Constraints {
        attention_type: "ssm".into(),
        activation: "silu".into(),
        norm_type: "rmsnorm".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::Ssm);
}

#[test]
fn derive_class_linear() {
    let c = Constraints {
        attention_type: "linear".into(),
        activation: "gelu".into(),
        norm_type: "layernorm".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::Linear);
}

#[test]
fn derive_class_unknown_empty() {
    let c = Constraints::default();
    assert_eq!(derive_kernel_class(&c), KernelClass::Unknown);
}

#[test]
fn derive_class_unknown_partial_match() {
    // Partial constraints that don't fit any class
    let c = Constraints {
        attention_type: "gqa".into(),
        activation: "relu".into(),
        norm_type: "rmsnorm".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), KernelClass::Unknown);
}

#[test]
fn derive_deterministic() {
    let c = Constraints {
        attention_type: "gqa".into(),
        activation: "silu".into(),
        norm_type: "rmsnorm".into(),
        mlp_type: "swiglu".into(),
        positional_encoding: "rope".into(),
        ..Default::default()
    };
    assert_eq!(derive_kernel_class(&c), derive_kernel_class(&c));
}

// ── KernelClass label/letter ────────────────────────────────────────

#[test]
fn kernel_class_label_all_variants() {
    let all = [
        KernelClass::A,
        KernelClass::B,
        KernelClass::C,
        KernelClass::D,
        KernelClass::E,
        KernelClass::F,
        KernelClass::Ssm,
        KernelClass::Linear,
        KernelClass::Unknown,
    ];
    for class in all {
        assert!(!class.label().is_empty());
        assert!(!class.letter().is_empty());
    }
}

#[test]
fn kernel_class_letters_correct() {
    assert_eq!(KernelClass::A.letter(), "A");
    assert_eq!(KernelClass::Ssm.letter(), "SSM");
    assert_eq!(KernelClass::Linear.letter(), "Linear");
    assert_eq!(KernelClass::Unknown.letter(), "Unknown");
}

#[test]
fn kernel_class_label_contains_letter() {
    assert!(KernelClass::A.label().starts_with("A "));
    assert!(KernelClass::Ssm.label().starts_with("SSM "));
}

// ── kernel_ops_for_class ────────────────────────────────────────────

#[test]
fn ops_class_a_has_gqa_rms_silu_swiglu_rope() {
    let ops = kernel_ops_for_class(KernelClass::A);
    assert!(ops.iter().any(|o| o.kernel == "gqa_forward"));
    assert!(ops.iter().any(|o| o.kernel == "rms_norm"));
    assert!(ops.iter().any(|o| o.kernel == "silu"));
    assert!(ops.iter().any(|o| o.kernel == "swiglu"));
    assert!(ops.iter().any(|o| o.kernel == "rope_forward"));
    assert!(ops.iter().any(|o| o.kernel == "softmax"));
}

#[test]
fn ops_class_b_has_mha_layernorm_gelu() {
    let ops = kernel_ops_for_class(KernelClass::B);
    assert!(ops.iter().any(|o| o.kernel == "mha_forward"));
    assert!(ops.iter().any(|o| o.kernel == "layer_norm"));
    assert!(ops.iter().any(|o| o.kernel == "gelu"));
    assert!(ops.iter().any(|o| o.kernel == "gelu_mlp"));
    assert!(!ops.iter().any(|o| o.kernel == "rope_forward"));
}

#[test]
fn ops_class_c_has_mqa_alibi() {
    let ops = kernel_ops_for_class(KernelClass::C);
    assert!(ops.iter().any(|o| o.kernel == "mqa_forward"));
    assert!(ops.iter().any(|o| o.kernel == "alibi"));
}

#[test]
fn ops_class_d_has_gated_mlp() {
    let ops = kernel_ops_for_class(KernelClass::D);
    assert!(ops.iter().any(|o| o.kernel == "gated_mlp"));
    assert!(ops.iter().any(|o| o.kernel == "rope_forward"));
}

#[test]
fn ops_class_e_has_moe_router() {
    let ops = kernel_ops_for_class(KernelClass::E);
    assert!(ops.iter().any(|o| o.kernel == "moe_routing"));
    assert!(ops.iter().any(|o| o.kernel == "swiglu"));
}

#[test]
fn ops_class_f_has_gelu_gated_mlp() {
    let ops = kernel_ops_for_class(KernelClass::F);
    assert!(ops.iter().any(|o| o.kernel == "gelu"));
    assert!(ops.iter().any(|o| o.kernel == "gated_mlp"));
    assert!(ops.iter().any(|o| o.kernel == "rms_norm"));
}

#[test]
fn ops_ssm_no_softmax() {
    let ops = kernel_ops_for_class(KernelClass::Ssm);
    assert!(!ops.iter().any(|o| o.kernel == "softmax"));
    assert!(ops.iter().any(|o| o.kernel == "selective_scan"));
    assert!(ops.iter().any(|o| o.kernel == "depthwise_conv1d"));
}

#[test]
fn ops_linear_no_softmax() {
    let ops = kernel_ops_for_class(KernelClass::Linear);
    assert!(!ops.iter().any(|o| o.kernel == "softmax"));
    assert!(ops.iter().any(|o| o.kernel == "wkv_forward"));
    assert!(ops.iter().any(|o| o.kernel == "token_shift"));
    assert!(ops.iter().any(|o| o.kernel == "channel_mix"));
}

#[test]
fn ops_unknown_minimal() {
    let ops = kernel_ops_for_class(KernelClass::Unknown);
    // Should have base ops only (MatVec Q4K, Q6K, Softmax, Fusion)
    assert_eq!(ops.len(), 4);
}

#[test]
fn ops_all_classes_have_matvec() {
    let all = [
        KernelClass::A,
        KernelClass::B,
        KernelClass::C,
        KernelClass::D,
        KernelClass::E,
        KernelClass::F,
        KernelClass::Ssm,
        KernelClass::Linear,
        KernelClass::Unknown,
    ];
    for class in all {
        let ops = kernel_ops_for_class(class);
        assert!(
            ops.iter().any(|o| o.kernel == "fused_q4k_parallel_matvec"),
            "Class {:?} missing Q4K matvec",
            class
        );
    }
}

// ── kernel_ops_for_family (RoPE enrichment) ─────────────────────────

#[test]
fn family_ops_phi_gets_rope_enrichment() {
    // Phi is Class B (no RoPE in base ops) but uses RoPE positional encoding
    let c = Constraints {
        attention_type: "mha".into(),
        activation: "gelu".into(),
        norm_type: "layernorm".into(),
        mlp_type: "gelu_mlp".into(),
        positional_encoding: "rope".into(),
        ..Default::default()
    };
    let ops = kernel_ops_for_family(KernelClass::B, &c);
    assert!(
        ops.iter().any(|o| o.kernel == "rope_forward"),
        "Phi (class B + rope) should get rope_forward enrichment"
    );
}

#[test]
fn family_ops_no_double_rope() {
    // Class A already has RoPE — enrichment should not duplicate it
    let c = Constraints {
        positional_encoding: "rope".into(),
        ..Default::default()
    };
    let ops = kernel_ops_for_family(KernelClass::A, &c);
    let rope_count = ops.iter().filter(|o| o.kernel == "rope_forward").count();
    assert_eq!(rope_count, 1, "Should not duplicate rope_forward");
}

#[test]
fn family_ops_absolute_no_rope() {
    let c = Constraints {
        positional_encoding: "absolute".into(),
        ..Default::default()
    };
    let ops = kernel_ops_for_family(KernelClass::B, &c);
    assert!(!ops.iter().any(|o| o.kernel == "rope_forward"));
}

// ── load_families ───────────────────────────────────────────────────

#[test]
fn load_families_count() {
    let families = load_families();
    assert_eq!(families.len(), FAMILY_YAMLS.len());
}

#[test]
fn load_families_expected_classes() {
    let families = load_families();
    let find = |name: &str| families.iter().find(|f| f.family == name).expect("family == name");

    assert_eq!(find("llama").kernel_class, KernelClass::A);
    assert_eq!(find("qwen2").kernel_class, KernelClass::A);
    assert_eq!(find("qwen3").kernel_class, KernelClass::A);
    assert_eq!(find("mistral").kernel_class, KernelClass::A);
    assert_eq!(find("deepseek").kernel_class, KernelClass::A);
    assert_eq!(find("falcon_h1").kernel_class, KernelClass::A);
    assert_eq!(find("openelm").kernel_class, KernelClass::A);

    assert_eq!(find("bert").kernel_class, KernelClass::B);
    assert_eq!(find("gpt2").kernel_class, KernelClass::B);
    assert_eq!(find("whisper").kernel_class, KernelClass::B);

    assert_eq!(find("gemma").kernel_class, KernelClass::F);

    assert_eq!(find("mamba").kernel_class, KernelClass::Ssm);
    assert_eq!(find("rwkv7").kernel_class, KernelClass::Linear);
}

#[test]
fn load_families_all_have_display_name() {
    for f in &load_families() {
        assert!(
            !f.display_name.is_empty(),
            "Family {} missing display_name",
            f.family
        );
    }
}

// ── resolve_family ──────────────────────────────────────────────────

// Direct matches
#[test]
fn resolve_direct_llama() {
    let f = resolve_family("llama").expect("resolve_family('llama')");
    assert_eq!(f.family, "llama");
    assert_eq!(f.kernel_class, KernelClass::A);
}

#[test]
fn resolve_direct_bert() {
    let f = resolve_family("bert").expect("resolve_family('bert')");
    assert_eq!(f.family, "bert");
    assert_eq!(f.kernel_class, KernelClass::B);
}

#[test]
fn resolve_direct_mamba() {
    let f = resolve_family("mamba").expect("resolve_family('mamba')");
    assert_eq!(f.family, "mamba");
    assert_eq!(f.kernel_class, KernelClass::Ssm);
}

#[test]
fn resolve_direct_rwkv7() {
    let f = resolve_family("rwkv7").expect("resolve_family('rwkv7')");
    assert_eq!(f.family, "rwkv7");
    assert_eq!(f.kernel_class, KernelClass::Linear);
}

// Normalized matches (hyphens, dots, case)
#[test]
fn resolve_normalized_hyphens() {
    let f = resolve_family("falcon-h1").expect("resolve_family('falcon-h1')");
    assert_eq!(f.family, "falcon_h1");
}

#[test]
fn resolve_normalized_dots() {
    let f = resolve_family("qwen3.5").expect("5'");
    assert_eq!(f.family, "qwen3_5");
}

#[test]
fn resolve_normalized_uppercase() {
    let f = resolve_family("LLAMA").expect("resolve_family('LLAMA')");
    assert_eq!(f.family, "llama");
}

// Cross-compact matches
#[test]
fn resolve_cross_compact_qwen_3_5() {
    // "qwen-3-5" → normalized "qwen_3_5" → compact "qwen35"
    // family "qwen3_5" → compact "qwen35" → match
    let f = resolve_family("qwen-3-5").expect("resolve_family('qwen-3-5')");
    assert_eq!(f.family, "qwen3_5");
}

// Alias matches
#[test]
fn resolve_alias_mixtral() {
    let f = resolve_family("mixtral").expect("resolve_family('mixtral')");
    assert_eq!(f.family, "mistral");
    assert!(f.display_name.contains("via"));
}

#[test]
fn resolve_alias_phi3() {
    let f = resolve_family("phi3").expect("resolve_family('phi3')");
    assert_eq!(f.family, "llama");
    assert!(f.display_name.contains("phi3"));
}

#[test]
fn resolve_alias_phi4() {
    let f = resolve_family("phi4").expect("resolve_family('phi4')");
    assert_eq!(f.family, "llama");
}

#[test]
fn resolve_alias_bloom() {
    let f = resolve_family("bloom").expect("resolve_family('bloom')");
    assert_eq!(f.family, "bert");
}

#[test]
fn resolve_alias_falcon() {
    let f = resolve_family("falcon").expect("resolve_family('falcon')");
    assert_eq!(f.family, "bert");
}

#[test]
fn resolve_alias_smollm() {
    let f = resolve_family("smollm").expect("resolve_family('smollm')");
    assert_eq!(f.family, "llama");
}

#[test]
fn resolve_alias_smollm2() {
    let f = resolve_family("smollm2").expect("resolve_family('smollm2')");
    assert_eq!(f.family, "llama");
}

#[test]
fn resolve_alias_codegemma() {
    let f = resolve_family("codegemma").expect("resolve_family('codegemma')");
    assert_eq!(f.family, "gemma");
}

#[test]
fn resolve_alias_gpt_neo() {
    let f = resolve_family("gpt_neo").expect("resolve_family('gpt_neo')");
    assert_eq!(f.family, "bert");
}

#[test]
fn resolve_alias_gptneo_compact() {
    let f = resolve_family("gptneo").expect("resolve_family('gptneo')");
    assert_eq!(f.family, "bert");
}

#[test]
fn resolve_alias_starcoder2() {
    let f = resolve_family("starcoder2").expect("resolve_family('starcoder2')");
    assert_eq!(f.family, "qwen2");
}

#[test]
fn resolve_alias_vicuna() {
    let f = resolve_family("vicuna").expect("resolve_family('vicuna')");
    assert_eq!(f.family, "llama");
}

#[test]
fn resolve_alias_qwq() {
    let f = resolve_family("qwq").expect("resolve_family('qwq')");
    assert_eq!(f.family, "qwen2");
}

#[test]
fn resolve_alias_qwen2_moe() {
    let f = resolve_family("qwen2_moe").expect("resolve_family('qwen2_moe')");
    assert_eq!(f.family, "qwen2");
}

#[test]
fn resolve_alias_normalized_gpt_j_hyphen() {
    let f = resolve_family("gpt-j").expect("resolve_family('gpt-j')");
    assert_eq!(f.family, "bert");
}

// Architecture string match
#[test]
fn resolve_arch_qwen2forcausallm() {
    let f = resolve_family("Qwen2ForCausalLM").expect("resolve_family('Qwen2ForCausal");
    assert_eq!(f.family, "qwen2");
}

#[test]
fn resolve_arch_bertmodel() {
    let f = resolve_family("BertModel").expect("resolve_family('BertModel')");
    assert_eq!(f.family, "bert");
}

// Architecture string → stripped → alias re-check
#[test]
fn resolve_stripped_graniteforcausallm() {
    let f = resolve_family("GraniteForCausalLM").expect("resolve_family('GraniteForCaus");
    assert_eq!(f.family, "llama");
}

// Partial match (3-pass)
#[test]
fn resolve_partial_qwen_matches_qwen2() {
    // "qwen" is prefix of family "qwen2" → Pass 2
    let f = resolve_family("qwen").expect("resolve_family('qwen')");
    assert_eq!(f.family, "qwen2");
}

#[test]
fn resolve_partial_phi3mini_via_alias() {
    // "phi3mini" starts_with alias "phi3" → Pass 1
    let f = resolve_family("phi-3-mini").expect("resolve_family('phi-3-mini')");
    assert_eq!(f.family, "llama");
    assert!(f.display_name.contains("phi3"));
}

#[test]
fn resolve_partial_gpt_matches_gpt2() {
    let f = resolve_family("gpt").expect("resolve_family('gpt')");
    assert_eq!(f.family, "gpt2");
}

// Edge cases
#[test]
fn resolve_empty_string() {
    assert!(resolve_family("").is_none());
}

#[test]
fn resolve_whitespace_only() {
    assert!(resolve_family("   ").is_none());
}

#[test]
fn resolve_emoji_stripped() {
    // Non-ASCII chars stripped, leaving empty → None
    assert!(resolve_family("\u{1f999}").is_none());
}

#[test]
fn resolve_emoji_with_text() {
    // "🦙llama" → strip non-ASCII → "llama" → match
    let f = resolve_family("\u{1f999}llama").expect("resolve_family('\u{1f999}llama");
    assert_eq!(f.family, "llama");
}

#[test]
fn resolve_unknown_returns_none() {
    assert!(resolve_family("nonexistent-xyz-123").is_none());
}

#[test]
fn resolve_short_input_no_partial() {
    // "ab" is < 3 chars, partial matching skipped
    assert!(resolve_family("ab").is_none());
}

// ── extract_json_string ─────────────────────────────────────────────

#[test]
fn json_string_value() {
    let json = r#"{"model_type": "qwen2"}"#;
    assert_eq!(
        extract_json_string(json, "model_type"),
        Some("qwen2".to_string())
    );
}

#[test]
fn json_numeric_value() {
    let json = r#"{"hidden_size": 4096, "other": 1}"#;
    assert_eq!(
        extract_json_string(json, "hidden_size"),
        Some("4096".to_string())
    );
}

#[test]
fn json_float_value() {
    let json = r#"{"rms_norm_eps": 1e-06, "x": 1}"#;
    assert_eq!(
        extract_json_string(json, "rms_norm_eps"),
        Some("1e-06".to_string())
    );
}

#[test]
fn json_boolean_value() {
    let json = r#"{"tie_word_embeddings": true, "x": 1}"#;
    assert_eq!(
        extract_json_string(json, "tie_word_embeddings"),
        Some("true".to_string())
    );
}

#[test]
fn json_null_returns_none() {
    let json = r#"{"model_type": null, "x": 1}"#;
    assert_eq!(extract_json_string(json, "model_type"), None);
}

#[test]
fn json_array_returns_none() {
    let json = r#"{"architectures": ["QwenForCausalLM"], "x": 1}"#;
    assert_eq!(extract_json_string(json, "architectures"), None);
}

#[test]
fn json_object_returns_none() {
    let json = r#"{"rope_scaling": {"type": "yarn"}, "x": 1}"#;
    assert_eq!(extract_json_string(json, "rope_scaling"), None);
}

#[test]
fn json_missing_key() {
    let json = r#"{"model_type": "bert"}"#;
    assert_eq!(extract_json_string(json, "hidden_act"), None);
}

#[test]
fn json_whitespace_around_colon() {
    let json = "{\n  \"hidden_size\" : 4096,\n  \"x\": 1\n}";
    assert_eq!(
        extract_json_string(json, "hidden_size"),
        Some("4096".to_string())
    );
}

// ── enrich_rationale ────────────────────────────────────────────────

#[test]
fn enrich_hidden_act_silu() {
    let r = enrich_rationale("hidden_act", "silu", "{}").expect("'{}')");
    assert!(r.contains("SiLU"));
}

#[test]
fn enrich_hidden_act_gelu() {
    let r = enrich_rationale("hidden_act", "gelu", "{}").expect("'{}')");
    assert!(r.contains("GELU"));
}

#[test]
fn enrich_hidden_act_gelu_new() {
    let r = enrich_rationale("hidden_act", "gelu_new", "{}").expect("'{}')");
    assert!(r.contains("GELU"));
}

#[test]
fn enrich_hidden_act_unknown() {
    let r = enrich_rationale("hidden_act", "relu", "{}").expect("'{}')");
    assert!(r.contains("relu"));
}

#[test]
fn enrich_rms_norm_eps() {
    let r = enrich_rationale("rms_norm_eps", "1e-06", "{}").expect("'{}')");
    assert!(r.contains("RMSNorm"));
}

#[test]
fn enrich_num_kv_heads_gqa() {
    let json = r#"{"num_attention_heads": 32, "num_key_value_heads": 8}"#;
    let r = enrich_rationale("num_key_value_heads", "8", json).expect("json)");
    assert!(r.contains("GQA"));
    assert!(r.contains("8"));
    assert!(r.contains("32"));
}

#[test]
fn enrich_num_kv_heads_mqa() {
    let json = r#"{"num_attention_heads": 32, "num_key_value_heads": 1}"#;
    let r = enrich_rationale("num_key_value_heads", "1", json).expect("json)");
    assert!(r.contains("MQA"));
}

#[test]
fn enrich_num_kv_heads_mha() {
    let json = r#"{"num_attention_heads": 32, "num_key_value_heads": 32}"#;
    let r = enrich_rationale("num_key_value_heads", "32", json).expect("json)");
    assert!(r.contains("MHA"));
}

#[test]
fn enrich_rope_theta() {
    let r = enrich_rationale("rope_theta", "10000.0", "{}").expect("0', '{}'");
    assert!(r.contains("RoPE"));
}

#[test]
fn enrich_intermediate_size_swiglu() {
    let json = r#"{"hidden_size": 4096, "intermediate_size": 11008, "hidden_act": "silu"}"#;
    let r = enrich_rationale("intermediate_size", "11008", json).expect("json)");
    assert!(r.contains("SwiGLU"));
}

#[test]
fn enrich_intermediate_size_gelu() {
    let json = r#"{"hidden_size": 768, "intermediate_size": 3072, "hidden_act": "gelu"}"#;
    let r = enrich_rationale("intermediate_size", "3072", json).expect("json)");
    assert!(r.contains("GELU"));
}

#[test]
fn enrich_num_local_experts_moe() {
    let r = enrich_rationale("num_local_experts", "8", "{}").expect("'{}')");
    assert!(r.contains("MoE"));
    assert!(r.contains("8"));
}

#[test]
fn enrich_num_experts_single() {
    let r = enrich_rationale("num_local_experts", "1", "{}").expect("'{}')");
    assert!(r.contains("dense"));
}

#[test]
fn enrich_num_experts_negative() {
    let r = enrich_rationale("num_local_experts", "-1", "{}").expect("'{}')");
    assert!(r.contains("negative"));
}

#[test]
fn enrich_tie_word_embeddings_true() {
    let r = enrich_rationale("tie_word_embeddings", "true", "{}").expect("'{}')");
    assert!(r.contains("Shared"));
}

#[test]
fn enrich_tie_word_embeddings_false() {
    let r = enrich_rationale("tie_word_embeddings", "false", "{}").expect("'{}')");
    assert!(r.contains("Separate"));
}

#[test]
fn enrich_num_attention_heads_gqa() {
    let json = r#"{"num_attention_heads": 32, "num_key_value_heads": 4}"#;
    let r = enrich_rationale("num_attention_heads", "32", json).expect("json)");
    assert!(r.contains("GQA"));
}

#[test]
fn enrich_num_attention_heads_mha() {
    let json = r#"{"num_attention_heads": 12, "num_key_value_heads": 12}"#;
    let r = enrich_rationale("num_attention_heads", "12", json).expect("json)");
    assert!(r.contains("MHA"));
}

#[test]
fn enrich_hidden_size_with_params() {
    let json = r#"{"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008, "hidden_act": "silu", "vocab_size": 32000, "num_attention_heads": 32, "num_key_value_heads": 8}"#;
    let r = enrich_rationale("hidden_size", "4096", json).expect("json)");
    assert!(r.contains("Hidden dim"));
    assert!(r.contains("params"));
}

#[test]
fn enrich_hidden_size_gelu_model() {
    let json = r#"{"hidden_size": 768, "num_hidden_layers": 12, "intermediate_size": 3072, "hidden_act": "gelu", "vocab_size": 30522, "num_attention_heads": 12, "num_key_value_heads": 12}"#;
    let r = enrich_rationale("hidden_size", "768", json).expect("json)");
    assert!(r.contains("Hidden dim"));
    assert!(r.contains("params"));
}

#[test]
fn enrich_num_hidden_layers() {
    let r = enrich_rationale("num_hidden_layers", "32", "{}").expect("'{}')");
    assert!(r.contains("32"));
    assert!(r.contains("layers"));
}

#[test]
fn enrich_vocab_size_with_hidden() {
    let json = r#"{"vocab_size": 32000, "hidden_size": 4096}"#;
    let r = enrich_rationale("vocab_size", "32000", json).expect("json)");
    assert!(r.contains("32000"));
    assert!(r.contains("MB"));
}

#[test]
fn enrich_max_position_1m() {
    let r = enrich_rationale("max_position_embeddings", "1048576", "{}").expect("'{}')");
    assert!(r.contains("1M+"));
}

#[test]
fn enrich_max_position_128k() {
    let r = enrich_rationale("max_position_embeddings", "131072", "{}").expect("'{}')");
    assert!(r.contains("128K+"));
}

#[test]
fn enrich_max_position_8k() {
    let r = enrich_rationale("max_position_embeddings", "8192", "{}").expect("'{}')");
    assert!(r.contains("8K+"));
}

#[test]
fn enrich_max_position_small() {
    let r = enrich_rationale("max_position_embeddings", "512", "{}").expect("'{}')");
    assert!(r.contains("512"));
    assert!(!r.contains("K+"));
}

#[test]
fn enrich_unknown_key() {
    assert!(enrich_rationale("unknown_key", "val", "{}").is_none());
}

#[test]
fn enrich_num_experts_per_tok() {
    let r = enrich_rationale("num_experts_per_tok", "2", "{}").expect("'{}')");
    assert!(r.contains("2"));
    assert!(r.contains("experts"));
}

#[test]
fn enrich_num_experts_per_tok_one() {
    let r = enrich_rationale("num_experts_per_tok", "1", "{}").expect("'{}')");
    assert!(r.contains("1"));
    assert!(r.contains("expert"));
    // singular
    assert!(!r.contains("experts"));
}

// ── detect_constraint_mismatches ────────────────────────────────────

fn make_family(name: &str, constraints: Constraints) -> FamilyInfo {
    let kernel_class = derive_kernel_class(&constraints);
    FamilyInfo {
        family: name.to_string(),
        display_name: name.to_string(),
        architectures: vec![],
        constraints,
        kernel_class,
    }
}

fn make_config(entries: &[(&str, &str)]) -> BTreeMap<String, ConfigField> {
    entries
        .iter()
        .map(|(k, v)| {
            (
                (*k).to_string(),
                ConfigField {
                    value: (*v).to_string(),
                    rationale: String::new(),
                },
            )
        })
        .collect()
}

#[test]
fn mismatch_activation_silu_vs_gelu() {
    let family = make_family(
        "test",
        Constraints {
            activation: "silu".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[("hidden_act", "gelu")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(
        warnings.iter().any(|w| w.contains("Activation mismatch")),
        "Expected activation mismatch warning"
    );
}

#[test]
fn mismatch_activation_gelu_vs_silu() {
    let family = make_family(
        "test",
        Constraints {
            activation: "gelu".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[("hidden_act", "silu")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("Activation mismatch")));
}

#[test]
fn mismatch_activation_gegelu() {
    let family = make_family(
        "test",
        Constraints {
            activation: "gelu".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[("hidden_act", "gegelu")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("gegelu")));
}

#[test]
fn mismatch_norm_rms_vs_layernorm() {
    let family = make_family(
        "test",
        Constraints {
            norm_type: "layernorm".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[("rms_norm_eps", "1e-06")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("Norm mismatch")));
}

#[test]
fn mismatch_norm_layernorm_vs_rms() {
    let family = make_family(
        "test",
        Constraints {
            norm_type: "rmsnorm".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[("layer_norm_epsilon", "1e-05")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("Norm mismatch")));
}

#[test]
fn mismatch_conflicting_norm_fields() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("rms_norm_eps", "1e-06"), ("layer_norm_epsilon", "1e-05")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("Conflicting norm")));
}

#[test]
fn mismatch_attention_gqa_vs_mha() {
    let family = make_family(
        "test",
        Constraints {
            attention_type: "mha".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[("num_key_value_heads", "4"), ("num_attention_heads", "32")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("Attention mismatch")));
}

#[test]
fn mismatch_mha_degenerate_gqa_suppressed() {
    // MHA (kv==q) is degenerate GQA — no warning
    let family = make_family(
        "test",
        Constraints {
            attention_type: "gqa".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[("num_key_value_heads", "32"), ("num_attention_heads", "32")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(
        !warnings.iter().any(|w| w.contains("Attention mismatch")),
        "MHA-as-GQA should not warn"
    );
}

#[test]
fn mismatch_kv_greater_than_q() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("num_key_value_heads", "64"), ("num_attention_heads", "32")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("cannot exceed")));
}

#[test]
fn mismatch_invalid_gqa_grouping() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("num_key_value_heads", "5"), ("num_attention_heads", "32")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("not divisible")));
}

#[test]
fn mismatch_multi_query_flag() {
    let family = make_family(
        "test",
        Constraints {
            attention_type: "mha".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[("multi_query", "true")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("multi_query")));
}

#[test]
fn mismatch_moe_non_e_class() {
    let family = make_family(
        "test",
        Constraints {
            attention_type: "gqa".into(),
            activation: "silu".into(),
            norm_type: "rmsnorm".into(),
            mlp_type: "swiglu".into(),
            positional_encoding: "rope".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[("num_local_experts", "8")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("MoE")));
}

#[test]
fn mismatch_moe_negative_experts() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("num_local_experts", "-1")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("negative")));
}

#[test]
fn mismatch_negative_hidden_size() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("hidden_size", "-1")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("negative")));
}

#[test]
fn mismatch_zero_hidden_size() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("hidden_size", "0")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("zero")));
}

#[test]
fn mismatch_implausible_hidden_size() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("hidden_size", "999999")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("Implausible")));
}

#[test]
fn mismatch_hidden_not_divisible_by_heads() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("hidden_size", "5120"), ("num_attention_heads", "24")]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("not divisible")));
}

#[test]
fn mismatch_hidden_divisibility_skipped_with_explicit_head_dim() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[
        ("hidden_size", "5120"),
        ("num_attention_heads", "24"),
        ("head_dim", "256"),
    ]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(
        !warnings.iter().any(|w| w.contains("not divisible")),
        "Should skip divisibility check when head_dim is explicit"
    );
}

#[test]
fn mismatch_model_type_arch_conflict() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[
        ("model_type", "llama"),
        ("_architectures", "MistralForCausalLM"),
    ]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("conflicts")));
}

#[test]
fn mismatch_model_type_arch_no_conflict_deepseek() {
    // deepseek_v2 vs DeepseekV2ForCausalLM: compact forms match
    let family = make_family("test", Constraints::default());
    let config = make_config(&[
        ("model_type", "deepseek_v2"),
        ("_architectures", "DeepseekV2ForCausalLM"),
    ]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(
        !warnings.iter().any(|w| w.contains("conflicts")),
        "deepseek_v2 vs DeepseekV2 should not conflict"
    );
}

#[test]
fn mismatch_moe_from_alias_name() {
    let family = FamilyInfo {
        family: "mistral".to_string(),
        display_name: "mixtral (via mistral kernel pipeline)".to_string(),
        architectures: vec![],
        constraints: Constraints::default(),
        kernel_class: KernelClass::A,
    };
    let config = make_config(&[]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(warnings.iter().any(|w| w.contains("MoE")));
}

#[test]
fn no_mismatch_clean_config() {
    let family = make_family(
        "test",
        Constraints {
            attention_type: "gqa".into(),
            activation: "silu".into(),
            norm_type: "rmsnorm".into(),
            ..Default::default()
        },
    );
    let config = make_config(&[
        ("hidden_act", "silu"),
        ("rms_norm_eps", "1e-06"),
        ("num_key_value_heads", "8"),
        ("num_attention_heads", "32"),
        ("hidden_size", "4096"),
    ]);
    let warnings = detect_constraint_mismatches(&family, &config);
    assert!(
        warnings.is_empty(),
        "Expected no warnings, got: {warnings:?}"
    );
}

// ── extract_architecture_display ────────────────────────────────────

#[test]
fn arch_display_from_config_architectures() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("_architectures", "LlamaForCausalLM")]);
    assert_eq!(
        extract_architecture_display(&family, &config),
        "LlamaForCausalLM"
    );
}

#[test]
fn arch_display_from_model_type() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[("model_type", "llama")]);
    assert_eq!(extract_architecture_display(&family, &config), "llama");
}

#[test]
fn arch_display_alias_uses_alias_arch_table() {
    let family = FamilyInfo {
        family: "llama".to_string(),
        display_name: "vicuna (via llama kernel pipeline)".to_string(),
        architectures: vec!["LlamaForCausalLM".to_string()],
        constraints: Constraints::default(),
        kernel_class: KernelClass::A,
    };
    let config = make_config(&[]);
    assert_eq!(
        extract_architecture_display(&family, &config),
        "LlamaForCausalLM"
    );
}

#[test]
fn arch_display_fallback_to_family_arch() {
    let family = FamilyInfo {
        family: "llama".to_string(),
        display_name: "LLaMA".to_string(),
        architectures: vec!["LlamaForCausalLM".to_string()],
        constraints: Constraints::default(),
        kernel_class: KernelClass::A,
    };
    let config = make_config(&[]);
    assert_eq!(
        extract_architecture_display(&family, &config),
        "LlamaForCausalLM"
    );
}

#[test]
fn arch_display_no_archs() {
    let family = make_family("test", Constraints::default());
    let config = make_config(&[]);
    assert_eq!(extract_architecture_display(&family, &config), "Unknown");
}

// ── proof_status ────────────────────────────────────────────────────

#[test]
fn proof_all_known_contracts() {
    for (name, _) in KERNEL_CONTRACTS {
        let proof = proof_status_for_contract(name);
        assert_ne!(
            proof.level,
            ProofLevel::Unknown,
            "Contract {name} should be known"
        );
    }
}

#[test]
fn proof_unknown_contract() {
    let proof = proof_status_for_contract("does-not-exist-v1");
    assert_eq!(proof.level, ProofLevel::Unknown);
    assert!(proof.evidence.contains("No contract"));
}

#[test]
fn proof_level_symbols() {
    assert_eq!(ProofLevel::Proven.symbol(), "✓");
    assert_eq!(ProofLevel::Tested.symbol(), "◉");
    assert_eq!(ProofLevel::Documented.symbol(), "○");
    assert_eq!(ProofLevel::Unknown.symbol(), "?");
}

#[test]
fn proof_level_labels() {
    assert_eq!(ProofLevel::Proven.label(), "Proven");
    assert_eq!(ProofLevel::Tested.label(), "Tested");
    assert_eq!(ProofLevel::Documented.label(), "Documented");
    assert_eq!(ProofLevel::Unknown.label(), "Unknown");
}

// ── proof_status_for_class ──────────────────────────────────────────

#[test]
fn proof_class_deduplicates_contracts() {
    let proofs = proof_status_for_class(KernelClass::A);
    let names: Vec<&str> = proofs.iter().map(|p| p.contract.as_str()).collect();
    // Verify no duplicates
    let mut seen = Vec::new();
    for name in &names {
        assert!(
            !seen.contains(name),
            "Duplicate contract in proof list: {name}"
        );
        seen.push(*name);
    }
}

#[test]
fn proof_class_unknown_has_some_proofs() {
    // Unknown still has base ops (MatVec, Softmax, Fusion)
    let proofs = proof_status_for_class(KernelClass::Unknown);
    assert!(!proofs.is_empty());
}

// ── build_json_output ───────────────────────────────────────────────

#[test]
fn json_output_basic() {
    let family = resolve_family("llama").expect("resolve_family('llama')");
    let config = make_config(&[]);
    let json = build_json_output(&family, config, false);
    assert_eq!(json.family, "llama");
    assert_eq!(json.kernel_class, "A");
    assert_eq!(json.layout, "row_major");
    assert!(json.proof_summary.is_none());
}

#[test]
fn json_output_with_proof() {
    let family = resolve_family("llama").expect("resolve_family('llama')");
    let config = make_config(&[]);
    let json = build_json_output(&family, config, true);
    assert!(json.proof_summary.is_some());
    let ps = json.proof_summary.expect("proof_summary");
    assert!(ps.total > 0);
}

#[test]
fn json_output_internal_fields_removed() {
    let family = resolve_family("llama").expect("resolve_family('llama')");
    let mut config = make_config(&[("_architectures", "LlamaForCausalLM")]);
    config.insert(
        "model_type".to_string(),
        ConfigField {
            value: "llama".to_string(),
            rationale: "test".to_string(),
        },
    );
    let json = build_json_output(&family, config, false);
    assert!(
        !json.config_mapping.contains_key("_architectures"),
        "Internal fields should be stripped"
    );
}

#[test]
fn json_output_equivalence_class() {
    let family = resolve_family("qwen2").expect("resolve_family('qwen2')");
    let json = build_json_output(&family, make_config(&[]), false);
    // Class A has multiple families
    assert!(json.equivalence_class_families.len() > 1);
    assert!(json
        .equivalence_class_families
        .contains(&"llama".to_string()));
}

// ── FAMILY_ALIASES coverage ─────────────────────────────────────────

#[test]
fn all_aliases_resolve_to_valid_family() {
    let families = load_families();
    for (alias, target) in FAMILY_ALIASES {
        assert!(
            families.iter().any(|f| f.family == *target),
            "Alias {alias} → {target}: target family not found in loaded families"
        );
    }
}

#[test]
fn all_aliases_resolvable() {
    for (alias, target) in FAMILY_ALIASES {
        let resolved = resolve_family(alias);
        assert!(
            resolved.is_some(),
            "Alias {alias} should resolve (expected target: {target})"
        );
        assert_eq!(
            resolved.expect("resolved").family,
            *target,
            "Alias {alias} resolved to wrong family"
        );
    }
}

// ── ALIAS_ARCHITECTURES coverage ────────────────────────────────────

#[test]
fn all_alias_architectures_have_matching_alias() {
    use super::resolve::ALIAS_ARCHITECTURES;
    for (alias_name, _hf_arch) in ALIAS_ARCHITECTURES {
        assert!(
            FAMILY_ALIASES.iter().any(|(a, _)| a == alias_name),
            "ALIAS_ARCHITECTURES has {alias_name} but no matching FAMILY_ALIASES entry"
        );
    }
}

// ── Regression tests ────────────────────────────────────────────────

#[test]
fn regression_phi3_mini_resolves_to_class_a() {
    // phi-3-mini was resolving to phi (Class B/D) instead of phi3 alias (Class A)
    let f = resolve_family("phi-3-mini").expect("resolve_family('phi-3-mini')");
    assert_eq!(f.family, "llama");
    assert_eq!(f.kernel_class, KernelClass::A);
}

#[test]
fn regression_qwen_resolves_to_qwen2_not_moe() {
    // "qwen" was resolving to qwen2_moe alias before family match
    let f = resolve_family("qwen").expect("resolve_family('qwen')");
    assert_eq!(f.family, "qwen2");
}

#[test]
fn regression_phi2_resolves_to_phi() {
    // phi-2 is the phi family directly (Class B/D, NOT phi3→llama alias)
    let f = resolve_family("phi").expect("resolve_family('phi')");
    assert_eq!(f.family, "phi");
}

#[test]
fn regression_gemma2_alias() {
    let f = resolve_family("gemma2").expect("resolve_family('gemma2')");
    assert_eq!(f.family, "gemma");
    assert_eq!(f.kernel_class, KernelClass::F);
}

#[test]
fn regression_mamba_is_ssm() {
    let f = resolve_family("mamba").expect("resolve_family('mamba')");
    assert_eq!(f.kernel_class, KernelClass::Ssm);
    let ops = kernel_ops_for_class(KernelClass::Ssm);
    assert!(!ops.iter().any(|o| o.kernel == "softmax"));
}

#[test]
fn regression_rwkv7_is_linear() {
    let f = resolve_family("rwkv7").expect("resolve_family('rwkv7')");
    assert_eq!(f.kernel_class, KernelClass::Linear);
    let ops = kernel_ops_for_class(KernelClass::Linear);
    assert!(!ops.iter().any(|o| o.kernel == "softmax"));
}

// ── Parameter estimate tests ────────────────────────────────────────

#[test]
fn param_estimate_gqa_model() {
    // Qwen2.5-7B: hidden=3584, layers=28, inter=18944, heads=28, kv=4, vocab=152064
    let json = r#"{"hidden_size": 3584, "num_hidden_layers": 28, "intermediate_size": 18944, "hidden_act": "silu", "vocab_size": 152064, "num_attention_heads": 28, "num_key_value_heads": 4}"#;
    let r = enrich_rationale("hidden_size", "3584", json).expect("json)");
    // Should estimate ~7-8B
    assert!(r.contains("B params"), "Expected B params in: {r}");
}

#[test]
fn param_estimate_gelu_model() {
    // BERT-base: hidden=768, layers=12, inter=3072, heads=12, kv=12, vocab=30522
    let json = r#"{"hidden_size": 768, "num_hidden_layers": 12, "intermediate_size": 3072, "hidden_act": "gelu", "vocab_size": 30522, "num_attention_heads": 12, "num_key_value_heads": 12, "tie_word_embeddings": "true"}"#;
    let r = enrich_rationale("hidden_size", "768", json).expect("json)");
    // BERT-base is ~110M
    assert!(r.contains("M params"), "Expected M params in: {r}");
}

#[test]
fn param_estimate_moe_model() {
    // Model with experts should include expert weights
    let json = r#"{"hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008, "hidden_act": "silu", "vocab_size": 32000, "num_attention_heads": 32, "num_key_value_heads": 8, "num_local_experts": 8, "moe_intermediate_size": 4096}"#;
    let r = enrich_rationale("hidden_size", "4096", json).expect("json)");
    assert!(r.contains("B params"), "MoE model should be large: {r}");
}

// ── ConfigField type ────────────────────────────────────────────────

#[test]
fn config_field_construction() {
    let field = ConfigField {
        value: "silu".to_string(),
        rationale: "SiLU activation".to_string(),
    };
    assert_eq!(field.value, "silu");
    assert_eq!(field.rationale, "SiLU activation");
}

// ── family_aliases() accessor ───────────────────────────────────────

#[test]
fn family_aliases_not_empty() {
    assert!(!family_aliases().is_empty());
    assert!(family_aliases().len() >= 50, "Expected at least 50 aliases");
}

// ── Constraints struct ──────────────────────────────────────────────

#[test]
fn constraints_default_all_empty() {
    let c = Constraints::default();
    assert_eq!(c.attention_type, "");
    assert_eq!(c.activation, "");
    assert_eq!(c.norm_type, "");
    assert_eq!(c.mlp_type, "");
    assert_eq!(c.positional_encoding, "");
    assert!(!c.has_bias);
    assert!(!c.tied_embeddings);
}

// ── FamilyInfo struct ───────────────────────────────────────────────

#[test]
fn family_info_clone_eq() {
    let f = resolve_family("llama").expect("resolve_family('llama')");
    let f2 = f.clone();
    assert_eq!(f.family, f2.family);
    assert_eq!(f.kernel_class, f2.kernel_class);
}

// ── Integration: round-trip family → class → ops → proof ────────────

#[test]
fn integration_llama_full_pipeline() {
    let family = resolve_family("llama").expect("resolve_family('llama')");
    assert_eq!(family.kernel_class, KernelClass::A);
    let ops = kernel_ops_for_family(family.kernel_class, &family.constraints);
    assert!(ops.len() > 4);
    let proofs = proof_status_for_class(family.kernel_class);
    assert!(!proofs.is_empty());
    let json = build_json_output(&family, BTreeMap::new(), true);
    assert_eq!(json.kernel_class, "A");
    assert!(json.proof_summary.is_some());
}

#[test]
fn integration_bert_full_pipeline() {
    let family = resolve_family("bert").expect("resolve_family('bert')");
    assert_eq!(family.kernel_class, KernelClass::B);
    let ops = kernel_ops_for_class(KernelClass::B);
    assert!(!ops.iter().any(|o| o.kernel == "rope_forward"));
}

#[test]
fn integration_gemma_full_pipeline() {
    let family = resolve_family("gemma").expect("resolve_family('gemma')");
    assert_eq!(family.kernel_class, KernelClass::F);
    let ops = kernel_ops_for_class(KernelClass::F);
    assert!(ops.iter().any(|o| o.kernel == "gated_mlp"));
    assert!(ops.iter().any(|o| o.kernel == "gelu"));
}

#[test]
fn integration_mamba_full_pipeline() {
    let family = resolve_family("mamba").expect("resolve_family('mamba')");
    assert_eq!(family.kernel_class, KernelClass::Ssm);
    let ops = kernel_ops_for_class(KernelClass::Ssm);
    assert!(ops.iter().any(|o| o.kernel == "selective_scan"));
}

#[test]
fn integration_alias_full_pipeline() {
    let family = resolve_family("mixtral").expect("resolve_family('mixtral')");
    assert!(family.display_name.contains("via"));
    let json = build_json_output(&family, BTreeMap::new(), false);
    assert!(json.display_name.contains("via"));
}

// ── resolve_from_config_json (filesystem tests) ─────────────────────

#[test]
fn resolve_config_json_valid() {
    let dir = tempfile::tempdir().expect("tempfile::tempdir()");
    let path = dir.path().join("config.json");
    std::fs::write(&path, r#"{"model_type": "qwen2", "hidden_size": 4096}"#).expect("4096}'#)");
    let f = resolve_from_config_json(&path).expect("unwrap();     let f = resolve_");
    assert_eq!(f.family, "qwen2");
}

#[test]
fn resolve_config_json_no_model_type_uses_architectures() {
    let dir = tempfile::tempdir().expect("tempfile::tempdir()");
    let path = dir.path().join("config.json");
    std::fs::write(
        &path,
        r#"{"architectures": ["LlamaForCausalLM"], "hidden_size": 4096}"#,
    )
    .expect("expected value");
    let f = resolve_from_config_json(&path).expect("unwrap();     let f = resolve_");
    assert_eq!(f.family, "llama");
}

#[test]
fn resolve_config_json_missing_file() {
    assert!(resolve_from_config_json(Path::new("/nonexistent/config.json")).is_none());
}

#[test]
fn resolve_config_json_array_rejected() {
    let dir = tempfile::tempdir().expect("tempfile::tempdir()");
    let path = dir.path().join("config.json");
    std::fs::write(&path, r#"[{"model_type": "bert"}]"#).expect("'bert'}]'#)");
    assert!(resolve_from_config_json(&path).is_none());
}

#[test]
fn resolve_config_json_unknown_model_type() {
    let dir = tempfile::tempdir().expect("tempfile::tempdir()");
    let path = dir.path().join("config.json");
    std::fs::write(&path, r#"{"model_type": "totally_unknown_xyz"}"#).expect("'totally_unknown_xyz'}'#)");
    assert!(resolve_from_config_json(&path).is_none());
}

// ── extract_config_mapping (filesystem tests) ───────────────────────

#[test]
fn config_mapping_extracts_fields() {
    let dir = tempfile::tempdir().expect("tempfile::tempdir()");
    let path = dir.path().join("config.json");
    std::fs::write(
        &path,
        r#"{
                "model_type": "qwen2",
                "hidden_act": "silu",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "rms_norm_eps": 1e-06,
                "rope_theta": 10000.0,
                "architectures": ["Qwen2ForCausalLM"]
            }"#,
    )
    .expect("expected value");
    let map = extract_config_mapping(&path);
    assert!(map.contains_key("model_type"));
    assert!(map.contains_key("hidden_act"));
    assert!(map.contains_key("hidden_size"));
    assert!(map.contains_key("num_attention_heads"));
    assert!(map.contains_key("num_key_value_heads"));
    assert!(map.contains_key("rms_norm_eps"));
    assert!(map.contains_key("_architectures"));
    // Enriched rationale
    assert!(map["hidden_act"].rationale.contains("SiLU"));
}

#[test]
fn config_mapping_missing_file() {
    let map = extract_config_mapping(Path::new("/nonexistent/config.json"));
    assert!(map.is_empty());
}

#[test]
fn config_mapping_enriches_gqa() {
    let dir = tempfile::tempdir().expect("tempfile::tempdir()");
    let path = dir.path().join("config.json");
    std::fs::write(
        &path,
        r#"{"num_attention_heads": 32, "num_key_value_heads": 8}"#,
    )
    .expect("expected value");
    let map = extract_config_mapping(&path);
    assert!(map["num_key_value_heads"].rationale.contains("GQA"));
}

// ── print_human_output (smoke test) ─────────────────────────────────

#[test]
fn print_human_output_no_panic() {
    use super::output::print_human_output;
    let family = resolve_family("llama").expect("resolve_family('llama')");
    let config = BTreeMap::new();
    // Just verify it doesn't panic — output goes to stdout
    print_human_output(&family, &config, false, false);
    print_human_output(&family, &config, true, true);
}
