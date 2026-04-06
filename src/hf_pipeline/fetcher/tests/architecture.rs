//! Tests for Architecture enum.

use crate::hf_pipeline::fetcher::Architecture;

#[test]
fn test_bert_param_count() {
    let arch = Architecture::BERT { num_layers: 12, hidden_size: 768, num_attention_heads: 12 };
    let params = arch.param_count();
    // 12 layers * (4 * 768^2 + 4 * 768^2) = 12 * 8 * 589824 = 56,621,568
    assert!(params > 50_000_000);
    assert!(params < 200_000_000);
}

#[test]
fn test_llama_param_count() {
    let arch = Architecture::Llama {
        num_layers: 32,
        hidden_size: 4096,
        num_attention_heads: 32,
        intermediate_size: 11008,
    };
    let params = arch.param_count();
    // Should be in billions range for 7B model
    assert!(params > 1_000_000_000);
}

#[test]
fn test_custom_param_count_is_zero() {
    let arch = Architecture::Custom { config: serde_json::json!({}) };
    assert_eq!(arch.param_count(), 0);
}

#[test]
fn test_gpt2_param_count() {
    let arch = Architecture::GPT2 { num_layers: 12, hidden_size: 768, num_attention_heads: 12 };
    let params = arch.param_count();
    assert!(params > 50_000_000);
}

#[test]
fn test_t5_param_count() {
    let arch = Architecture::T5 { encoder_layers: 12, decoder_layers: 12, hidden_size: 768 };
    let params = arch.param_count();
    assert!(params > 100_000_000);
}

#[test]
fn test_architecture_serde() {
    let arch = Architecture::Llama {
        num_layers: 32,
        hidden_size: 4096,
        num_attention_heads: 32,
        intermediate_size: 11008,
    };
    let serialized = serde_json::to_string(&arch).expect("JSON serialization should succeed");
    let deserialized: Architecture =
        serde_json::from_str(&serialized).expect("JSON deserialization should succeed");
    assert_eq!(arch.param_count(), deserialized.param_count());
}

#[test]
fn test_bert_architecture_serde() {
    let arch = Architecture::BERT { num_layers: 12, hidden_size: 768, num_attention_heads: 12 };
    let serialized = serde_json::to_string(&arch).expect("JSON serialization should succeed");
    let deserialized: Architecture =
        serde_json::from_str(&serialized).expect("JSON deserialization should succeed");
    assert_eq!(arch.param_count(), deserialized.param_count());
}

#[test]
fn test_gpt2_architecture_serde() {
    let arch = Architecture::GPT2 { num_layers: 12, hidden_size: 768, num_attention_heads: 12 };
    let serialized = serde_json::to_string(&arch).expect("JSON serialization should succeed");
    let deserialized: Architecture =
        serde_json::from_str(&serialized).expect("JSON deserialization should succeed");
    assert_eq!(arch.param_count(), deserialized.param_count());
}

#[test]
fn test_t5_architecture_serde() {
    let arch = Architecture::T5 { encoder_layers: 12, decoder_layers: 12, hidden_size: 768 };
    let serialized = serde_json::to_string(&arch).expect("JSON serialization should succeed");
    let deserialized: Architecture =
        serde_json::from_str(&serialized).expect("JSON deserialization should succeed");
    assert_eq!(arch.param_count(), deserialized.param_count());
}

#[test]
fn test_custom_architecture_serde() {
    let arch =
        Architecture::Custom { config: serde_json::json!({"model_type": "custom", "layers": 10}) };
    let serialized = serde_json::to_string(&arch).expect("JSON serialization should succeed");
    let deserialized: Architecture =
        serde_json::from_str(&serialized).expect("JSON deserialization should succeed");
    assert_eq!(arch.param_count(), deserialized.param_count());
    assert_eq!(arch.param_count(), 0); // Custom always returns 0
}

#[test]
fn test_architecture_debug() {
    let bert = Architecture::BERT { num_layers: 6, hidden_size: 384, num_attention_heads: 6 };
    let debug = format!("{bert:?}");
    assert!(debug.contains("BERT"));
    assert!(debug.contains("384"));
}

#[test]
fn test_architecture_clone() {
    let original = Architecture::Llama {
        num_layers: 16,
        hidden_size: 2048,
        num_attention_heads: 16,
        intermediate_size: 5504,
    };
    let cloned = original.clone();
    assert_eq!(original.param_count(), cloned.param_count());
}

#[test]
fn test_bert_small_config() {
    let bert = Architecture::BERT { num_layers: 6, hidden_size: 256, num_attention_heads: 4 };
    let params = bert.param_count();
    // Should be smaller than standard BERT
    assert!(params > 0);
    assert!(params < 50_000_000);
}

#[test]
fn test_llama_small_config() {
    let llama = Architecture::Llama {
        num_layers: 8,
        hidden_size: 512,
        num_attention_heads: 8,
        intermediate_size: 1024,
    };
    let params = llama.param_count();
    assert!(params > 0);
}

#[test]
fn test_gpt2_small_config() {
    let gpt2 = Architecture::GPT2 { num_layers: 6, hidden_size: 384, num_attention_heads: 6 };
    let params = gpt2.param_count();
    assert!(params > 0);
}

#[test]
fn test_t5_small_config() {
    let t5 = Architecture::T5 { encoder_layers: 4, decoder_layers: 4, hidden_size: 256 };
    let params = t5.param_count();
    assert!(params > 0);
}
