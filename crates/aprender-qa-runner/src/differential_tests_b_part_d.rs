#[test]
fn test_model_preparation_result_clone() {
    use crate::provenance::{Provenance, SourceProvenance};

    let result = ModelPreparationResult {
        provenance: Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-01-01T00:00:00Z".to_string(),
            },
            derived: vec![],
        },
        safetensors_path: std::path::PathBuf::from("/test"),
        gguf_path: None,
        apr_path: None,
        conversions: vec![],
    };
    let cloned = result.clone();
    assert_eq!(cloned.provenance.source.hf_repo, "test/model");
}

#[test]
fn test_model_preparation_result_debug() {
    use crate::provenance::{Provenance, SourceProvenance};

    let result = ModelPreparationResult {
        provenance: Provenance {
            source: SourceProvenance {
                format: "safetensors".to_string(),
                path: "model.safetensors".to_string(),
                sha256: "abc".to_string(),
                hf_repo: "test/model".to_string(),
                downloaded_at: "2026-01-01T00:00:00Z".to_string(),
            },
            derived: vec![],
        },
        safetensors_path: std::path::PathBuf::from("/test"),
        gguf_path: None,
        apr_path: None,
        conversions: vec![],
    };
    let debug = format!("{result:?}");
    assert!(debug.contains("ModelPreparationResult"));
}

#[test]
fn test_bench_result_clone_debug() {
    let result = BenchResult {
        throughput_tps: 10.0,
        passed: true,
        backend: "cpu".to_string(),
        format: "apr".to_string(),
    };
    let cloned = result.clone();
    assert_eq!(cloned.backend, "cpu");
    let debug = format!("{result:?}");
    assert!(debug.contains("BenchResult"));
}

// =========================================================================
// InspectResult tests (T-GH192-01, MR-CARD)
// =========================================================================

#[test]
fn test_inspect_result_from_json() {
    let json = r#"{
            "tensor_count": 338,
            "tensor_names": ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"],
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "hidden_size": 896,
            "architecture": "Qwen2ForCausalLM"
        }"#;
    let result: InspectResult = serde_json::from_str(json).unwrap();
    assert_eq!(result.tensor_count, 338);
    assert_eq!(result.tensor_names.len(), 2);
    assert_eq!(result.num_attention_heads, Some(14));
    assert_eq!(result.num_key_value_heads, Some(2));
    assert_eq!(result.hidden_size, Some(896));
    assert_eq!(result.architecture.as_deref(), Some("Qwen2ForCausalLM"));
}

#[test]
fn test_inspect_result_minimal_json() {
    let json = r#"{"tensor_count": 100}"#;
    let result: InspectResult = serde_json::from_str(json).unwrap();
    assert_eq!(result.tensor_count, 100);
    assert!(result.tensor_names.is_empty());
    assert!(result.num_attention_heads.is_none());
    assert!(result.architecture.is_none());
}

#[test]
fn test_inspect_result_serialization_round_trip() {
    let result = InspectResult {
        tensor_count: 227,
        tensor_names: vec![
            "model.embed_tokens.weight".to_string(),
            "lm_head.weight".to_string(),
        ],
        num_attention_heads: Some(32),
        num_key_value_heads: Some(8),
        hidden_size: Some(4096),
        architecture: Some("LlamaForCausalLM".to_string()),
    };
    let json = serde_json::to_string(&result).unwrap();
    let parsed: InspectResult = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.tensor_count, 227);
    assert_eq!(parsed.tensor_names.len(), 2);
    assert_eq!(parsed.hidden_size, Some(4096));
}

#[test]
fn test_inspect_result_clone() {
    let result = InspectResult {
        tensor_count: 50,
        tensor_names: vec!["test.weight".to_string()],
        num_attention_heads: Some(12),
        num_key_value_heads: None,
        hidden_size: Some(768),
        architecture: None,
    };
    let cloned = result.clone();
    assert_eq!(cloned.tensor_count, result.tensor_count);
    assert_eq!(cloned.tensor_names, result.tensor_names);
}

#[test]
fn test_inspect_result_debug() {
    let result = InspectResult {
        tensor_count: 100,
        tensor_names: vec![],
        num_attention_heads: None,
        num_key_value_heads: None,
        hidden_size: None,
        architecture: None,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("InspectResult"));
}

#[test]
fn test_parse_inspect_text_with_tensor_count() {
    let output = "Tensors: 338\nmodel.embed_tokens.weight [151936, 896]\nmodel.layers.0.self_attn.q_proj.weight [896, 896]";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 338);
    assert_eq!(result.tensor_names.len(), 2);
    assert!(
        result
            .tensor_names
            .contains(&"model.embed_tokens.weight".to_string())
    );
}

#[test]
fn test_parse_inspect_text_with_metadata() {
    let output = "Tensors: 100\narchitecture: Qwen2ForCausalLM\nnum_attention_heads: 14\nnum_key_value_heads: 2\nhidden_size: 896";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 100);
    assert_eq!(result.architecture.as_deref(), Some("Qwen2ForCausalLM"));
    assert_eq!(result.num_attention_heads, Some(14));
    assert_eq!(result.num_key_value_heads, Some(2));
    assert_eq!(result.hidden_size, Some(896));
}

#[test]
fn test_parse_inspect_text_empty() {
    let output = "";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 0);
    assert!(result.tensor_names.is_empty());
}

#[test]
fn test_parse_inspect_text_tensor_count_from_names() {
    let output = "model.layers.0.weight [768, 768]\nmodel.layers.1.weight [768, 768]";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 2);
    assert_eq!(result.tensor_names.len(), 2);
}

#[test]
fn test_parse_inspect_text_alternate_prefix() {
    let output = "tensor_count: 42";
    let result = parse_inspect_text(output).unwrap();
    assert_eq!(result.tensor_count, 42);
}

#[test]
fn test_run_inspect_nonexistent_binary() {
    let path = std::path::PathBuf::from("model.gguf");
    let result = run_inspect(&path, "/nonexistent/apr/binary");
    assert!(result.is_err());
}
