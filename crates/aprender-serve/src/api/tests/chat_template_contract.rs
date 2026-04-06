
//! Contract tests: chat-template-v1.yaml FALSIFY-CT-002 (PMAT-187)
//! AppState MUST cache GGUF architecture at construction time.

use crate::api::AppState;
use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{ArchConstraints, GGUFConfig};

fn make_config(architecture: &str) -> GGUFConfig {
    GGUFConfig {
        architecture: architecture.to_string(),
        constraints: ArchConstraints::from_architecture(architecture),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        vocab_size: 100,
        rope_theta: 10000.0,
        context_length: 512,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    }
}

/// FALSIFY-CT-002: with_quantized_model_and_vocab MUST cache architecture.
#[test]
fn falsify_ct_002_architecture_cached_for_qwen3() {
    let model = create_test_model_with_config(&make_config("qwen3"));
    let mut vocab: Vec<String> = (0..100).map(|i| format!("tok_{i}")).collect();
    vocab.push("<unk>".to_string());
    let state = AppState::with_quantized_model_and_vocab(model, vocab).unwrap();
    assert_eq!(
        state.model_architecture(),
        Some("qwen3".to_string()),
        "FALSIFY-CT-002: AppState MUST cache 'qwen3' architecture from GGUF model"
    );
}

#[test]
fn falsify_ct_002_architecture_cached_for_llama() {
    let model = create_test_model_with_config(&make_config("llama"));
    let mut vocab: Vec<String> = (0..100).map(|i| format!("tok_{i}")).collect();
    vocab.push("<unk>".to_string());
    let state = AppState::with_quantized_model_and_vocab(model, vocab).unwrap();
    assert_eq!(
        state.model_architecture(),
        Some("llama".to_string()),
        "FALSIFY-CT-002: AppState MUST cache 'llama' architecture from GGUF model"
    );
}

#[test]
fn falsify_ct_002_architecture_never_none_for_gguf() {
    for arch in &["qwen3", "llama", "phi2", "mistral", "gpt2"] {
        let model = create_test_model_with_config(&make_config(arch));
        let mut vocab: Vec<String> = (0..100).map(|i| format!("tok_{i}")).collect();
    vocab.push("<unk>".to_string());
        let state = AppState::with_quantized_model_and_vocab(model, vocab).unwrap();
        assert!(
            state.model_architecture().is_some(),
            "FALSIFY-CT-002: cached_architecture MUST be Some for GGUF with architecture '{arch}'"
        );
    }
}
