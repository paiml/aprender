
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // RuntimeType Tests
    // =========================================================================

    #[test]
    fn test_runtime_type_as_str() {
        assert_eq!(RuntimeType::Realizar.as_str(), "realizar");
        assert_eq!(RuntimeType::LlamaCpp.as_str(), "llama-cpp");
        assert_eq!(RuntimeType::Vllm.as_str(), "vllm");
        assert_eq!(RuntimeType::Ollama.as_str(), "ollama");
    }

    #[test]
    fn test_runtime_type_parse() {
        assert_eq!(RuntimeType::parse("realizar"), Some(RuntimeType::Realizar));
        assert_eq!(RuntimeType::parse("llama-cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("llama.cpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("llamacpp"), Some(RuntimeType::LlamaCpp));
        assert_eq!(RuntimeType::parse("vllm"), Some(RuntimeType::Vllm));
        assert_eq!(RuntimeType::parse("ollama"), Some(RuntimeType::Ollama));
        assert_eq!(RuntimeType::parse("REALIZAR"), Some(RuntimeType::Realizar)); // case-insensitive
        assert_eq!(RuntimeType::parse("unknown"), None);
    }

    #[test]
    fn test_runtime_type_clone_eq() {
        let rt = RuntimeType::Realizar;
        assert_eq!(rt, rt.clone());
    }

    #[test]
    fn test_runtime_type_debug() {
        let debug = format!("{:?}", RuntimeType::Vllm);
        assert!(debug.contains("Vllm"));
    }

    #[test]
    fn test_runtime_type_serialize() {
        let json = serde_json::to_string(&RuntimeType::LlamaCpp).expect("json");
        assert!(json.contains("LlamaCpp"));
    }

    // =========================================================================
    // InferenceRequest Tests
    // =========================================================================

    #[test]
    fn test_inference_request_default() {
        let req = InferenceRequest::default();
        assert!(req.prompt.is_empty());
        assert_eq!(req.max_tokens, 100);
        assert!((req.temperature - 0.7).abs() < 0.01);
        assert!(req.stop.is_empty());
    }

    #[test]
    fn test_inference_request_new() {
        let req = InferenceRequest::new("Hello world");
        assert_eq!(req.prompt, "Hello world");
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn test_inference_request_builder() {
        let req = InferenceRequest::new("test")
            .with_max_tokens(50)
            .with_temperature(0.5)
            .with_stop(vec!["END".to_string()]);

        assert_eq!(req.prompt, "test");
        assert_eq!(req.max_tokens, 50);
        assert!((req.temperature - 0.5).abs() < 0.01);
        assert_eq!(req.stop, vec!["END".to_string()]);
    }

    #[test]
    fn test_inference_request_serialize() {
        let req = InferenceRequest::new("prompt");
        let json = serde_json::to_string(&req).expect("json");
        assert!(json.contains("prompt"));
        assert!(json.contains("100")); // max_tokens
    }

    // =========================================================================
    // InferenceResponse Tests
    // =========================================================================

    #[test]
    fn test_inference_response_tokens_per_second() {
        let resp = InferenceResponse {
            text: "Hello".to_string(),
            tokens_generated: 100,
            ttft_ms: 10.0,
            total_time_ms: 1000.0, // 1 second
            itl_ms: vec![],
        };
        assert!((resp.tokens_per_second() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_inference_response_tokens_per_second_zero_time() {
        let resp = InferenceResponse {
            text: "Hello".to_string(),
            tokens_generated: 100,
            ttft_ms: 0.0,
            total_time_ms: 0.0,
            itl_ms: vec![],
        };
        assert_eq!(resp.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_inference_response_tokens_per_second_negative_time() {
        let resp = InferenceResponse {
            text: "Hello".to_string(),
            tokens_generated: 100,
            ttft_ms: 0.0,
            total_time_ms: -1.0,
            itl_ms: vec![],
        };
        assert_eq!(resp.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_inference_response_serialize() {
        let resp = InferenceResponse {
            text: "Generated text".to_string(),
            tokens_generated: 42,
            ttft_ms: 15.5,
            total_time_ms: 100.0,
            itl_ms: vec![2.0, 3.0],
        };
        let json = serde_json::to_string(&resp).expect("json");
        assert!(json.contains("Generated text"));
        assert!(json.contains("42"));
    }

    // =========================================================================
    // BackendInfo Tests
    // =========================================================================

    #[test]
    fn test_backend_info_serialize() {
        let info = BackendInfo {
            runtime_type: RuntimeType::Realizar,
            version: "1.0.0".to_string(),
            supports_streaming: true,
            loaded_model: Some("llama-7b".to_string()),
        };
        let json = serde_json::to_string(&info).expect("json");
        assert!(json.contains("1.0.0"));
        assert!(json.contains("llama-7b"));
    }

    #[test]
    fn test_backend_info_clone() {
        let info = BackendInfo {
            runtime_type: RuntimeType::Vllm,
            version: "0.4.0".to_string(),
            supports_streaming: true,
            loaded_model: None,
        };
        let cloned = info.clone();
        assert_eq!(info.version, cloned.version);
    }

    // =========================================================================
    // MockBackend Tests
    // =========================================================================

    #[test]
    fn test_mock_backend_new() {
        let backend = MockBackend::new(50.0, 10.0);
        assert!((backend.ttft_ms - 50.0).abs() < 0.01);
        assert!((backend.tokens_per_second - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_mock_backend_info() {
        let backend = MockBackend::new(10.0, 100.0);
        let info = backend.info();
        assert_eq!(info.runtime_type, RuntimeType::Realizar);
        assert!(info.supports_streaming);
        assert!(info.loaded_model.is_none());
    }

    #[test]
    fn test_mock_backend_inference() {
        let backend = MockBackend::new(20.0, 50.0);
        let request = InferenceRequest::new("Hello").with_max_tokens(25);
        let response = backend.inference(&request).expect("response");

        assert_eq!(response.text, "Mock response");
        assert_eq!(response.tokens_generated, 25);
        assert!((response.ttft_ms - 20.0).abs() < 0.01);
        // gen_time = 25 / 50 * 1000 = 500ms
        // total = 20 + 500 = 520ms
        assert!((response.total_time_ms - 520.0).abs() < 0.1);
        assert_eq!(response.itl_ms.len(), 25);
    }

    #[test]
    fn test_mock_backend_inference_max_100() {
        let backend = MockBackend::new(10.0, 100.0);
        let request = InferenceRequest::new("Hello").with_max_tokens(200);
        let response = backend.inference(&request).expect("response");
        // Should be capped at 100
        assert_eq!(response.tokens_generated, 100);
    }

    // =========================================================================
    // BackendRegistry Tests
    // =========================================================================

    #[test]
    fn test_backend_registry_new() {
        let registry = BackendRegistry::new();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_backend_registry_default() {
        let registry = BackendRegistry::default();
        assert!(registry.list().is_empty());
    }

    #[test]
    fn test_backend_registry_register_and_get() {
        let mut registry = BackendRegistry::new();
        let backend = MockBackend::new(10.0, 100.0);
        registry.register(RuntimeType::Realizar, Box::new(backend));

        let retrieved = registry.get(RuntimeType::Realizar);
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.expect("retrieved").info().runtime_type,
            RuntimeType::Realizar
        );
    }

    #[test]
    fn test_backend_registry_get_missing() {
        let registry = BackendRegistry::new();
        assert!(registry.get(RuntimeType::Vllm).is_none());
    }

    #[test]
    fn test_backend_registry_list() {
        let mut registry = BackendRegistry::new();
        registry.register(
            RuntimeType::Realizar,
            Box::new(MockBackend::new(10.0, 100.0)),
        );
        registry.register(
            RuntimeType::LlamaCpp,
            Box::new(MockBackend::new(20.0, 50.0)),
        );

        let list = registry.list();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&RuntimeType::Realizar));
        assert!(list.contains(&RuntimeType::LlamaCpp));
    }

    // =========================================================================
    // LlamaCppConfig Tests
    // =========================================================================

    #[test]
    fn test_llamacpp_config_default() {
        let config = LlamaCppConfig::default();
        assert_eq!(config.binary_path, "llama-cli");
        assert!(config.model_path.is_none());
        assert_eq!(config.n_gpu_layers, 0);
        assert_eq!(config.ctx_size, 2048);
        assert_eq!(config.threads, 4);
    }

    #[test]
    fn test_llamacpp_config_new() {
        let config = LlamaCppConfig::new("/usr/local/bin/llama-cli");
        assert_eq!(config.binary_path, "/usr/local/bin/llama-cli");
    }

    #[test]
    fn test_llamacpp_config_builder() {
        let config = LlamaCppConfig::new("llama-cli")
            .with_model("/models/llama.gguf")
            .with_gpu_layers(32)
            .with_ctx_size(4096)
            .with_threads(8);

        assert_eq!(config.model_path, Some("/models/llama.gguf".to_string()));
        assert_eq!(config.n_gpu_layers, 32);
        assert_eq!(config.ctx_size, 4096);
        assert_eq!(config.threads, 8);
    }

    #[test]
    fn test_llamacpp_config_serialize() {
        let config = LlamaCppConfig::new("llama-cli").with_model("model.gguf");
        let json = serde_json::to_string(&config).expect("json");
        assert!(json.contains("llama-cli"));
        assert!(json.contains("model.gguf"));
    }

    // =========================================================================
    // VllmConfig Tests
    // =========================================================================

    #[test]
    fn test_vllm_config_default() {
        let config = VllmConfig::default();
        assert_eq!(config.base_url, "http://localhost:8000");
        assert_eq!(config.api_version, "v1");
        assert!(config.model.is_none());
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_vllm_config_new() {
        let config = VllmConfig::new("http://myserver:8080");
        assert_eq!(config.base_url, "http://myserver:8080");
    }

    #[test]
    fn test_vllm_config_builder() {
        let config = VllmConfig::new("http://localhost:8000")
            .with_model("mistral-7b")
            .with_api_key("sk-secret");

        assert_eq!(config.model, Some("mistral-7b".to_string()));
        assert_eq!(config.api_key, Some("sk-secret".to_string()));
    }

    #[test]
    fn test_vllm_config_serialize() {
        let config = VllmConfig::new("http://test").with_model("phi-2");
        let json = serde_json::to_string(&config).expect("json");
        assert!(json.contains("http://test"));
        assert!(json.contains("phi-2"));
    }

    // =========================================================================
    // LlamaCppBackend Tests
    // =========================================================================

    #[test]
    fn test_llamacpp_backend_build_cli_args() {
        let config = LlamaCppConfig::new("llama-cli")
            .with_model("/models/llama.gguf")
            .with_gpu_layers(10)
            .with_ctx_size(2048)
            .with_threads(4);
        let backend = LlamaCppBackend::new(config);

        let request = InferenceRequest::new("Hello").with_max_tokens(50);
        let args = backend.build_cli_args(&request);

        assert!(args.contains(&"-m".to_string()));
        assert!(args.contains(&"/models/llama.gguf".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"Hello".to_string()));
        assert!(args.contains(&"-n".to_string()));
        assert!(args.contains(&"50".to_string()));
        assert!(args.contains(&"-ngl".to_string()));
        assert!(args.contains(&"10".to_string()));
    }

    #[test]
    fn test_llamacpp_backend_build_cli_args_custom_temp() {
        let config = LlamaCppConfig::new("llama-cli").with_model("model.gguf");
        let backend = LlamaCppBackend::new(config);

        let request = InferenceRequest::new("test").with_temperature(0.5);
        let args = backend.build_cli_args(&request);

        assert!(args.contains(&"--temp".to_string()));
        assert!(args.iter().any(|a| a.contains("0.50")));
    }

    #[test]
    fn test_llamacpp_backend_build_cli_args_default_temp() {
        let config = LlamaCppConfig::new("llama-cli").with_model("model.gguf");
        let backend = LlamaCppBackend::new(config);

        let request = InferenceRequest::new("test").with_temperature(0.8);
        let args = backend.build_cli_args(&request);

        // Default temp (0.8) should not add --temp flag
        assert!(!args.contains(&"--temp".to_string()));
    }

    #[test]
    fn test_llamacpp_backend_parse_timing_line() {
        let output = r"
llama_perf_context_print: prompt eval time =      12.34 ms /    10 tokens
llama_perf_context_print: eval time =     123.45 ms /   100 tokens
llama_perf_context_print: total time =     135.79 ms /   110 runs
        ";

        let (prompt_time, prompt_tokens) =
            LlamaCppBackend::parse_timing_line(output, "prompt eval time").expect("expected value");
        assert!((prompt_time - 12.34).abs() < 0.01);
        assert_eq!(prompt_tokens, 10);

        let (eval_time, eval_tokens) =
            LlamaCppBackend::parse_timing_line(output, "eval time").expect("expected value");
        assert!((eval_time - 123.45).abs() < 0.01);
        assert_eq!(eval_tokens, 100);

        let (total_time, total_runs) =
            LlamaCppBackend::parse_timing_line(output, "total time").expect("expected value");
        assert!((total_time - 135.79).abs() < 0.01);
        assert_eq!(total_runs, 110);
    }

    #[test]
    fn test_llamacpp_backend_parse_timing_line_not_found() {
        let output = "No timing info here";
        assert!(LlamaCppBackend::parse_timing_line(output, "eval time").is_none());
    }

    #[test]
    fn test_llamacpp_backend_extract_generated_text() {
        let output = r"Hello world!
This is generated text.
llama_perf_context_print: eval time = 100 ms
sampler stats follow...";

        let text = LlamaCppBackend::extract_generated_text(output);
        assert_eq!(text, "Hello world!\nThis is generated text.");
    }

    #[test]
    fn test_llamacpp_backend_extract_generated_text_empty() {
        let output = "llama_perf_context_print: eval time = 100 ms";
        let text = LlamaCppBackend::extract_generated_text(output);
        assert!(text.is_empty());
    }

    #[test]
    fn test_llamacpp_backend_parse_cli_output() {
        let output = r"Generated response text
llama_perf_context_print: prompt eval time =      50.00 ms /     5 tokens
llama_perf_context_print: eval time =     200.00 ms /    20 tokens
llama_perf_context_print: total time =     250.00 ms /    25 runs";

        let response = LlamaCppBackend::parse_cli_output(output).expect("response");
        assert_eq!(response.text, "Generated response text");
        assert!((response.ttft_ms - 50.0).abs() < 0.01);
        assert_eq!(response.tokens_generated, 20);
        assert!((response.total_time_ms - 250.0).abs() < 0.01);
        // ITL should be estimated from eval time / tokens
        assert_eq!(response.itl_ms.len(), 19); // tokens - 1
    }

    #[test]
    fn test_llamacpp_backend_parse_cli_output_minimal() {
        let output = "Just text, no timing";
        let response = LlamaCppBackend::parse_cli_output(output).expect("response");
        assert_eq!(response.text, "Just text, no timing");
        assert_eq!(response.ttft_ms, 0.0);
        assert_eq!(response.tokens_generated, 0);
    }

    #[test]
    fn test_llamacpp_backend_info() {
        let config = LlamaCppConfig::new("llama-cli").with_model("model.gguf");
        let backend = LlamaCppBackend::new(config);
        let info = backend.info();

        assert_eq!(info.runtime_type, RuntimeType::LlamaCpp);
        assert!(!info.supports_streaming);
        assert_eq!(info.loaded_model, Some("model.gguf".to_string()));
    }

    // =========================================================================
    // Runtime Backend Trait Tests
    // =========================================================================

    #[test]
    fn test_runtime_backend_load_model_default() {
        let mut backend = MockBackend::new(10.0, 100.0);
        // Default load_model should be a no-op
        let result = backend.load_model("any/path.gguf");
        assert!(result.is_ok());
    }
}
