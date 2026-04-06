//! Bench Runtime Tests Part 03: Non-HTTP Coverage Tests
//!
//! Protocol T-COV-95 Directive 2 (Popper): Tests for bench/runtime.rs that
//! don't require the bench-http feature, allowing coverage without external services.
//!
//! These tests exercise:
//! - RuntimeType parsing and display
//! - InferenceRequest builder pattern
//! - InferenceResponse calculations
//! - MockBackend behavior
//! - BackendRegistry operations
//! - LlamaCppConfig and LlamaCppBackend (no actual execution)
//! - Error handling paths

use crate::bench::runtime::{
    BackendInfo, BackendRegistry, InferenceRequest, InferenceResponse, LlamaCppBackend,
    LlamaCppConfig, MockBackend, RuntimeBackend, RuntimeType,
};

// =============================================================================
// RuntimeType Tests
// =============================================================================

#[test]
fn test_runtime_type_as_str_all_variants() {
    assert_eq!(RuntimeType::Realizar.as_str(), "realizar");
    assert_eq!(RuntimeType::LlamaCpp.as_str(), "llama-cpp");
    assert_eq!(RuntimeType::Vllm.as_str(), "vllm");
    assert_eq!(RuntimeType::Ollama.as_str(), "ollama");
}

#[test]
fn test_runtime_type_parse_case_insensitive() {
    // Standard cases
    assert_eq!(RuntimeType::parse("realizar"), Some(RuntimeType::Realizar));
    assert_eq!(RuntimeType::parse("REALIZAR"), Some(RuntimeType::Realizar));
    assert_eq!(RuntimeType::parse("Realizar"), Some(RuntimeType::Realizar));

    // LlamaCpp variations
    assert_eq!(RuntimeType::parse("llama-cpp"), Some(RuntimeType::LlamaCpp));
    assert_eq!(RuntimeType::parse("LLAMA-CPP"), Some(RuntimeType::LlamaCpp));
    assert_eq!(RuntimeType::parse("llama.cpp"), Some(RuntimeType::LlamaCpp));
    assert_eq!(RuntimeType::parse("LLAMA.CPP"), Some(RuntimeType::LlamaCpp));
    assert_eq!(RuntimeType::parse("llamacpp"), Some(RuntimeType::LlamaCpp));
    assert_eq!(RuntimeType::parse("LLAMACPP"), Some(RuntimeType::LlamaCpp));

    // Others
    assert_eq!(RuntimeType::parse("vllm"), Some(RuntimeType::Vllm));
    assert_eq!(RuntimeType::parse("VLLM"), Some(RuntimeType::Vllm));
    assert_eq!(RuntimeType::parse("ollama"), Some(RuntimeType::Ollama));
    assert_eq!(RuntimeType::parse("OLLAMA"), Some(RuntimeType::Ollama));
}

#[test]
fn test_runtime_type_parse_invalid() {
    assert_eq!(RuntimeType::parse("unknown"), None);
    assert_eq!(RuntimeType::parse(""), None);
    assert_eq!(RuntimeType::parse("pytorch"), None);
    assert_eq!(RuntimeType::parse("tensorflow"), None);
    assert_eq!(RuntimeType::parse("onnx"), None);
}

#[test]
fn test_runtime_type_equality() {
    assert_eq!(RuntimeType::Realizar, RuntimeType::Realizar);
    assert_ne!(RuntimeType::Realizar, RuntimeType::LlamaCpp);
    assert_ne!(RuntimeType::Vllm, RuntimeType::Ollama);
}

#[test]
fn test_runtime_type_clone() {
    let rt = RuntimeType::LlamaCpp;
    let cloned = rt;
    assert_eq!(rt, cloned);
}

#[test]
fn test_runtime_type_debug() {
    let debug_str = format!("{:?}", RuntimeType::Realizar);
    assert!(debug_str.contains("Realizar"));
}

// =============================================================================
// InferenceRequest Tests
// =============================================================================

#[test]
fn test_inference_request_default() {
    let req = InferenceRequest::default();
    assert_eq!(req.prompt, "");
    assert_eq!(req.max_tokens, 100);
    assert!((req.temperature - 0.7).abs() < 0.001);
    assert!(req.stop.is_empty());
}

#[test]
fn test_inference_request_new() {
    let req = InferenceRequest::new("Hello, world!");
    assert_eq!(req.prompt, "Hello, world!");
    assert_eq!(req.max_tokens, 100); // default
}

#[test]
fn test_inference_request_builder_chain() {
    let req = InferenceRequest::new("Test prompt")
        .with_max_tokens(50)
        .with_temperature(0.5)
        .with_stop(vec!["END".to_string()]);

    assert_eq!(req.prompt, "Test prompt");
    assert_eq!(req.max_tokens, 50);
    assert!((req.temperature - 0.5).abs() < 0.001);
    assert_eq!(req.stop, vec!["END"]);
}

#[test]
fn test_inference_request_builder_multiple_stops() {
    let req = InferenceRequest::new("test").with_stop(vec![
        "<|end|>".to_string(),
        "###".to_string(),
        "\n\n".to_string(),
    ]);

    assert_eq!(req.stop.len(), 3);
    assert!(req.stop.contains(&"<|end|>".to_string()));
}

#[test]
fn test_inference_request_builder_zero_tokens() {
    let req = InferenceRequest::new("test").with_max_tokens(0);
    assert_eq!(req.max_tokens, 0);
}

#[test]
fn test_inference_request_builder_large_tokens() {
    let req = InferenceRequest::new("test").with_max_tokens(100_000);
    assert_eq!(req.max_tokens, 100_000);
}

#[test]
fn test_inference_request_builder_extreme_temperature() {
    let req_hot = InferenceRequest::new("test").with_temperature(2.0);
    assert!((req_hot.temperature - 2.0).abs() < 0.001);

    let req_cold = InferenceRequest::new("test").with_temperature(0.0);
    assert!((req_cold.temperature - 0.0).abs() < 0.001);
}

#[test]
fn test_inference_request_clone() {
    let req = InferenceRequest::new("test")
        .with_max_tokens(42)
        .with_temperature(0.8);
    let cloned = req.clone();

    assert_eq!(cloned.prompt, req.prompt);
    assert_eq!(cloned.max_tokens, req.max_tokens);
}

// =============================================================================
// InferenceResponse Tests
// =============================================================================

#[test]
fn test_inference_response_tokens_per_second_normal() {
    let response = InferenceResponse {
        text: "Hello world".to_string(),
        tokens_generated: 100,
        ttft_ms: 50.0,
        total_time_ms: 1000.0,
        itl_ms: vec![10.0; 100],
    };
    // 100 tokens / 1.0 seconds = 100 tok/s
    assert!((response.tokens_per_second() - 100.0).abs() < 0.1);
}

#[test]
fn test_inference_response_tokens_per_second_zero_time() {
    let response = InferenceResponse {
        text: String::new(),
        tokens_generated: 100,
        ttft_ms: 0.0,
        total_time_ms: 0.0,
        itl_ms: vec![],
    };
    assert_eq!(response.tokens_per_second(), 0.0);
}

#[test]
fn test_inference_response_tokens_per_second_small_time() {
    let response = InferenceResponse {
        text: "test".to_string(),
        tokens_generated: 10,
        ttft_ms: 5.0,
        total_time_ms: 10.0, // 10ms = 0.01s
        itl_ms: vec![1.0; 10],
    };
    // 10 tokens / 0.01 seconds = 1000 tok/s
    assert!((response.tokens_per_second() - 1000.0).abs() < 1.0);
}

#[test]
fn test_inference_response_tokens_per_second_single_token() {
    let response = InferenceResponse {
        text: "a".to_string(),
        tokens_generated: 1,
        ttft_ms: 100.0,
        total_time_ms: 100.0,
        itl_ms: vec![100.0],
    };
    // 1 token / 0.1 seconds = 10 tok/s
    assert!((response.tokens_per_second() - 10.0).abs() < 0.1);
}

#[test]
fn test_inference_response_itl_average() {
    let response = InferenceResponse {
        text: "test".to_string(),
        tokens_generated: 5,
        ttft_ms: 50.0,
        total_time_ms: 250.0,
        itl_ms: vec![10.0, 20.0, 30.0, 40.0, 50.0],
    };

    // Average ITL = (10+20+30+40+50)/5 = 30ms
    let avg: f64 = response.itl_ms.iter().sum::<f64>() / response.itl_ms.len() as f64;
    assert!((avg - 30.0).abs() < 0.001);
}

#[test]
fn test_inference_response_clone() {
    let response = InferenceResponse {
        text: "test".to_string(),
        tokens_generated: 10,
        ttft_ms: 25.0,
        total_time_ms: 100.0,
        itl_ms: vec![5.0; 10],
    };
    let cloned = response.clone();

    assert_eq!(cloned.text, response.text);
    assert_eq!(cloned.tokens_generated, response.tokens_generated);
}

// =============================================================================
// MockBackend Tests
// =============================================================================

#[test]
fn test_mock_backend_creation() {
    let backend = MockBackend::new(42.0, 150.0);
    let info = backend.info();
    assert_eq!(info.runtime_type, RuntimeType::Realizar);
}

#[test]
fn test_mock_backend_info() {
    let backend = MockBackend::new(30.0, 140.0);
    let info = backend.info();

    assert_eq!(info.runtime_type, RuntimeType::Realizar);
    assert!(!info.version.is_empty());
    assert!(info.supports_streaming);
    assert!(info.loaded_model.is_none());
}

#[test]
fn test_mock_backend_inference() {
    let backend = MockBackend::new(42.0, 150.0);
    let req = InferenceRequest::new("test prompt").with_max_tokens(10);
    let response = backend.inference(&req);

    assert!(response.is_ok());
    let resp = response.unwrap();
    assert!((resp.ttft_ms - 42.0).abs() < 0.001);
    assert!(resp.tokens_generated > 0);
    assert!(resp.tokens_generated <= 10);
}

#[test]
fn test_mock_backend_inference_max_100() {
    let backend = MockBackend::new(10.0, 100.0);
    let req = InferenceRequest::new("test").with_max_tokens(200);
    let response = backend.inference(&req).unwrap();

    // MockBackend caps at 100 tokens
    assert!(response.tokens_generated <= 100);
}

#[test]
fn test_mock_backend_inference_zero_tokens() {
    let backend = MockBackend::new(10.0, 100.0);
    let req = InferenceRequest::new("test").with_max_tokens(0);
    let response = backend.inference(&req).unwrap();

    assert_eq!(response.tokens_generated, 0);
}

#[test]
fn test_mock_backend_inference_ttft_preserved() {
    let backend = MockBackend::new(55.5, 200.0);
    let req = InferenceRequest::new("test").with_max_tokens(1);
    let response = backend.inference(&req).unwrap();

    assert!((response.ttft_ms - 55.5).abs() < 0.001);
}

#[test]
fn test_mock_backend_load_model_noop() {
    let mut backend = MockBackend::new(10.0, 100.0);
    let result = backend.load_model("/path/to/model");
    assert!(result.is_ok());
}

// =============================================================================
// BackendRegistry Tests
// =============================================================================

#[test]
fn test_backend_registry_new() {
    let registry = BackendRegistry::new();
    assert!(registry.get(RuntimeType::Realizar).is_none());
    assert!(registry.list().is_empty());
}

#[test]
fn test_backend_registry_default() {
    let registry = BackendRegistry::default();
    assert!(registry.list().is_empty());
}

#[test]
fn test_backend_registry_register_single() {
    let mut registry = BackendRegistry::new();
    let backend = Box::new(MockBackend::new(30.0, 140.0));
    registry.register(RuntimeType::Realizar, backend);

    assert!(registry.get(RuntimeType::Realizar).is_some());
    assert!(registry.get(RuntimeType::LlamaCpp).is_none());
}

#[test]
fn test_backend_registry_register_multiple() {
    let mut registry = BackendRegistry::new();
    registry.register(
        RuntimeType::Realizar,
        Box::new(MockBackend::new(30.0, 140.0)),
    );
    registry.register(
        RuntimeType::LlamaCpp,
        Box::new(MockBackend::new(35.0, 130.0)),
    );

    let list = registry.list();
    assert_eq!(list.len(), 2);
    assert!(list.contains(&RuntimeType::Realizar));
    assert!(list.contains(&RuntimeType::LlamaCpp));
}

#[test]
fn test_backend_registry_overwrite() {
    let mut registry = BackendRegistry::new();

    registry.register(
        RuntimeType::Realizar,
        Box::new(MockBackend::new(10.0, 100.0)),
    );
    registry.register(
        RuntimeType::Realizar,
        Box::new(MockBackend::new(20.0, 200.0)),
    );

    // Should only have one entry (overwritten)
    assert_eq!(registry.list().len(), 1);

    // The second one should be active
    let backend = registry.get(RuntimeType::Realizar).unwrap();
    let info = backend.info();
    assert_eq!(info.runtime_type, RuntimeType::Realizar);
}

#[test]
fn test_backend_registry_get_inference() {
    let mut registry = BackendRegistry::new();
    registry.register(
        RuntimeType::Realizar,
        Box::new(MockBackend::new(25.0, 100.0)),
    );

    let backend = registry.get(RuntimeType::Realizar).unwrap();
    let req = InferenceRequest::new("test").with_max_tokens(5);
    let response = backend.inference(&req);

    assert!(response.is_ok());
}

// =============================================================================
// BackendInfo Tests
// =============================================================================

#[test]
fn test_backend_info_clone() {
    let info = BackendInfo {
        runtime_type: RuntimeType::LlamaCpp,
        version: "v1.0".to_string(),
        supports_streaming: true,
        loaded_model: Some("/path/to/model".to_string()),
    };
    let cloned = info.clone();

    assert_eq!(cloned.runtime_type, info.runtime_type);
    assert_eq!(cloned.version, info.version);
    assert_eq!(cloned.supports_streaming, info.supports_streaming);
    assert_eq!(cloned.loaded_model, info.loaded_model);
}

#[test]
fn test_backend_info_debug() {
    let info = BackendInfo {
        runtime_type: RuntimeType::Vllm,
        version: "0.4.0".to_string(),
        supports_streaming: false,
        loaded_model: None,
    };
    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("Vllm"));
}

// =============================================================================
// LlamaCppConfig Tests
// =============================================================================

#[test]
fn test_llama_cpp_config_default() {
    let config = LlamaCppConfig::default();
    assert_eq!(config.binary_path, "llama-cli");
    assert!(config.model_path.is_none());
    assert_eq!(config.n_gpu_layers, 0);
    assert_eq!(config.ctx_size, 2048);
    assert_eq!(config.threads, 4);
}

#[test]
fn test_llama_cpp_config_new() {
    let config = LlamaCppConfig::new("/usr/local/bin/llama-cli");
    assert_eq!(config.binary_path, "/usr/local/bin/llama-cli");
}

include!("tests_llama_cpp.rs");
