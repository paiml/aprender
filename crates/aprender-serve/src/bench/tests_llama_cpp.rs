
#[test]
fn test_llama_cpp_config_builder() {
    let config = LlamaCppConfig::new("llama-cli")
        .with_model("/models/test.gguf")
        .with_gpu_layers(32)
        .with_ctx_size(4096)
        .with_threads(8);

    assert_eq!(config.model_path, Some("/models/test.gguf".to_string()));
    assert_eq!(config.n_gpu_layers, 32);
    assert_eq!(config.ctx_size, 4096);
    assert_eq!(config.threads, 8);
}

#[test]
fn test_llama_cpp_config_zero_gpu_layers() {
    let config = LlamaCppConfig::new("llama-cli").with_gpu_layers(0);
    assert_eq!(config.n_gpu_layers, 0);
}

#[test]
fn test_llama_cpp_config_max_gpu_layers() {
    let config = LlamaCppConfig::new("llama-cli").with_gpu_layers(100);
    assert_eq!(config.n_gpu_layers, 100);
}

#[test]
fn test_llama_cpp_config_clone() {
    let config = LlamaCppConfig::new("llama-cli")
        .with_model("/models/test.gguf")
        .with_gpu_layers(16);
    let cloned = config.clone();

    assert_eq!(cloned.binary_path, config.binary_path);
    assert_eq!(cloned.model_path, config.model_path);
    assert_eq!(cloned.n_gpu_layers, config.n_gpu_layers);
}

// =============================================================================
// LlamaCppBackend Tests
// =============================================================================

#[test]
fn test_llama_cpp_backend_creation() {
    let config = LlamaCppConfig::new("llama-cli");
    let backend = LlamaCppBackend::new(config);
    let info = backend.info();

    assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
    assert!(!info.version.is_empty());
}

#[test]
fn test_llama_cpp_backend_info_with_model() {
    let config = LlamaCppConfig::new("llama-cli").with_model("test.gguf");
    let backend = LlamaCppBackend::new(config);
    let info = backend.info();

    assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
    assert!(!info.supports_streaming);
    assert_eq!(info.loaded_model, Some("test.gguf".to_string()));
}

#[test]
fn test_llama_cpp_backend_inference_missing_model() {
    let config = LlamaCppConfig::new("llama-cli"); // No model set
    let backend = LlamaCppBackend::new(config);
    let req = InferenceRequest::new("test");

    let result = backend.inference(&req);
    assert!(result.is_err());
}

#[test]
fn test_llama_cpp_backend_inference_missing_binary() {
    let config = LlamaCppConfig::new("/nonexistent/path/to/llama-cli").with_model("test.gguf");
    let backend = LlamaCppBackend::new(config);
    let req = InferenceRequest::new("test");

    let result = backend.inference(&req);
    assert!(result.is_err());
}

#[test]
fn test_llama_cpp_backend_load_model() {
    let config = LlamaCppConfig::new("llama-cli");
    let mut backend = LlamaCppBackend::new(config);

    // load_model is a no-op by default
    let result = backend.load_model("/path/to/model.gguf");
    assert!(result.is_ok());
}
