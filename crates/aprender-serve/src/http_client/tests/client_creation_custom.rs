use crate::http_client::*;

// =========================================================================
// Unit Tests (No network required)
// =========================================================================

#[test]
fn test_client_creation() {
    let client = ModelHttpClient::new();
    assert_eq!(client.timeout_secs(), 60);
}

#[test]
fn test_client_custom_timeout() {
    let client = ModelHttpClient::with_timeout(120);
    assert_eq!(client.timeout_secs(), 120);
}

#[test]
fn test_completion_request_serialization() {
    let request = CompletionRequest {
        model: "llama2".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: 100,
        temperature: Some(0.7),
        stream: false,
    };

    let json = serde_json::to_string(&request).expect("serialization failed");
    assert!(json.contains("\"model\":\"llama2\""));
    assert!(json.contains("\"prompt\":\"Hello\""));
    assert!(json.contains("\"max_tokens\":100"));
}

#[test]
fn test_ollama_request_serialization() {
    let request = OllamaRequest {
        model: "llama2".to_string(),
        prompt: "Hello".to_string(),
        stream: false,
        options: Some(OllamaOptions {
            num_predict: Some(100),
            temperature: Some(0.7),
        }),
    };

    let json = serde_json::to_string(&request).expect("serialization failed");
    assert!(json.contains("\"model\":\"llama2\""));
    assert!(json.contains("\"prompt\":\"Hello\""));
}

#[test]
fn test_completion_response_deserialization() {
    let json = r#"{
        "id": "cmpl-123",
        "choices": [{"text": "World!", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    }"#;

    let response: CompletionResponse = serde_json::from_str(json).expect("deserialization failed");

    assert_eq!(response.id, "cmpl-123");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].text, "World!");
}

#[test]
fn test_ollama_response_deserialization() {
    let json = r#"{
        "model": "llama2",
        "response": "Hello back!",
        "done": true,
        "total_duration": 5000000000,
        "load_duration": 1000000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 2000000000,
        "eval_count": 5,
        "eval_duration": 2000000000
    }"#;

    let response: OllamaResponse = serde_json::from_str(json).expect("deserialization failed");

    assert_eq!(response.model, "llama2");
    assert_eq!(response.response, "Hello back!");
    assert!(response.done);
    assert_eq!(response.eval_count, 5);
}

// =========================================================================
// Integration Tests (Require running servers)
// Mark with #[ignore] for CI - run manually with: cargo test -- --ignored
// =========================================================================

#[test]
#[ignore = "Requires vLLM server at localhost:8000"]
fn test_vllm_real_inference() {
    let client = ModelHttpClient::new();

    let request = CompletionRequest {
        model: "meta-llama/Llama-2-7b-hf".to_string(),
        prompt: "The capital of France is".to_string(),
        max_tokens: 20,
        temperature: Some(0.1),
        stream: false,
    };

    let result = client.openai_completion("http://localhost:8000", &request, None);

    // This MUST succeed with a real server
    let timing = result.expect("vLLM inference failed - is server running?");

    // Verify we got REAL data, not mock data
    assert!(timing.ttft_ms > 0.0, "TTFT must be positive (real latency)");
    assert!(
        timing.total_time_ms > 0.0,
        "Total time must be positive (real latency)"
    );
    assert!(!timing.text.is_empty(), "Must get actual generated text");

    println!("vLLM Real Inference:");
    println!("  TTFT: {:.2}ms", timing.ttft_ms);
    println!("  Total: {:.2}ms", timing.total_time_ms);
    println!("  Tokens: {}", timing.tokens_generated);
    println!("  Text: {}", timing.text);
}

#[test]
#[ignore = "Requires Ollama server at localhost:11434"]
fn test_ollama_real_inference() {
    let client = ModelHttpClient::new();

    let request = OllamaRequest {
        model: "phi2:2.7b".to_string(),
        prompt: "The capital of France is".to_string(),
        stream: false,
        options: Some(OllamaOptions {
            num_predict: Some(20),
            temperature: Some(0.1),
        }),
    };

    let result = client.ollama_generate("http://localhost:11434", &request);

    // This MUST succeed with a real server
    let timing = result.expect("Ollama inference failed - is server running?");

    // Verify we got REAL data, not mock data
    assert!(timing.ttft_ms > 0.0, "TTFT must be positive (real latency)");
    assert!(
        timing.total_time_ms > 0.0,
        "Total time must be positive (real latency)"
    );
    assert!(!timing.text.is_empty(), "Must get actual generated text");

    println!("Ollama Real Inference:");
    println!("  TTFT: {:.2}ms", timing.ttft_ms);
    println!("  Total: {:.2}ms", timing.total_time_ms);
    println!("  Tokens: {}", timing.tokens_generated);
    println!("  Text: {}", timing.text);
}

#[test]
#[ignore = "Requires llama.cpp server at localhost:8080"]
fn test_llamacpp_real_inference() {
    let client = ModelHttpClient::new();

    let request = CompletionRequest {
        model: "default".to_string(), // llama.cpp uses loaded model
        prompt: "The capital of France is".to_string(),
        max_tokens: 20,
        temperature: Some(0.1),
        stream: false,
    };

    let result = client.openai_completion("http://localhost:8080", &request, None);

    let timing = result.expect("llama.cpp inference failed - is server running?");

    assert!(timing.ttft_ms > 0.0, "TTFT must be positive (real latency)");
    assert!(
        timing.total_time_ms > 0.0,
        "Total time must be positive (real latency)"
    );

    println!("llama.cpp Real Inference:");
    println!("  TTFT: {:.2}ms", timing.ttft_ms);
    println!("  Total: {:.2}ms", timing.total_time_ms);
    println!("  Tokens: {}", timing.tokens_generated);
    println!("  Text: {}", timing.text);
}

#[test]
fn test_connection_error_handling() {
    let client = ModelHttpClient::with_timeout(1); // 1 second timeout

    let request = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: None,
        stream: false,
    };

    // This should fail because no server is running on this port
    let result = client.openai_completion("http://localhost:59999", &request, None);

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        RealizarError::ConnectionError(msg) => {
            assert!(msg.contains("HTTP request failed"));
        },
        other => panic!("Expected ConnectionError, got: {:?}", other),
    }
}

#[test]
fn test_ollama_connection_error() {
    let client = ModelHttpClient::with_timeout(1);

    let request = OllamaRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        stream: false,
        options: None,
    };

    let result = client.ollama_generate("http://localhost:59998", &request);

    assert!(result.is_err());
}

#[test]
fn test_llamacpp_response_deserialization() {
    let json = r#"{
        "content": "Machine learning is a subset of AI.",
        "model": "/path/to/model.gguf",
        "tokens_predicted": 8,
        "tokens_evaluated": 5,
        "stop": true,
        "timings": {
            "prompt_n": 5,
            "prompt_ms": 10.5,
            "predicted_n": 8,
            "predicted_ms": 25.3,
            "predicted_per_second": 316.2
        }
    }"#;

    let response: LlamaCppResponse = serde_json::from_str(json).expect("deserialization failed");

    assert_eq!(response.content, "Machine learning is a subset of AI.");
    assert_eq!(response.tokens_predicted, 8);
    assert_eq!(response.tokens_evaluated, 5);
    assert!(response.stop);

    let timings = response.timings.expect("timings should be present");
    assert_eq!(timings.prompt_n, 5);
    assert_eq!(timings.predicted_n, 8);
    assert!((timings.predicted_per_second - 316.2).abs() < 0.1);
}

#[test]
fn test_llamacpp_response_minimal() {
    // llama.cpp response with only required field
    let json = r#"{"content": "Hello world"}"#;

    let response: LlamaCppResponse = serde_json::from_str(json).expect("deserialization failed");

    assert_eq!(response.content, "Hello world");
    assert_eq!(response.tokens_predicted, 0); // default
    assert!(response.timings.is_none());
}

#[test]
fn test_llamacpp_connection_error() {
    let client = ModelHttpClient::with_timeout(1);

    let request = CompletionRequest {
        model: "test".to_string(),
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: None,
        stream: false,
    };

    let result = client.llamacpp_completion("http://localhost:59997", &request);

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        RealizarError::ConnectionError(msg) => {
            assert!(msg.contains("HTTP request failed"));
        },
        other => panic!("Expected ConnectionError, got: {:?}", other),
    }
}

// =========================================================================
// HTTP Benchmark Runner Tests
// =========================================================================

#[test]
fn test_benchmark_config_default() {
    let config = HttpBenchmarkConfig::default();
    assert_eq!(config.min_samples(), 5);
    assert_eq!(config.max_samples(), 30);
    assert!((config.cv_threshold() - 0.05).abs() < 0.001); // Default is now 5% per spec
    assert_eq!(config.warmup_iterations, 2);
    assert!(config.run_preflight);
    assert!(config.filter_outliers);
}

#[test]
fn test_benchmark_config_relaxed() {
    let config = HttpBenchmarkConfig::relaxed();
    assert_eq!(config.min_samples(), 3);
    assert_eq!(config.max_samples(), 10);
    assert!((config.cv_threshold() - 0.20).abs() < 0.001);
    assert!(!config.run_preflight);
    assert!(!config.filter_outliers);
}

#[test]
fn test_benchmark_config_reproducible() {
    let config = HttpBenchmarkConfig::reproducible();
    assert_eq!(config.min_samples(), 10);
    assert_eq!(config.max_samples(), 50);
    assert!((config.cv_threshold() - 0.03).abs() < 0.001);
    assert!(config.run_preflight);
    assert!(config.filter_outliers);
}

#[test]
fn test_benchmark_runner_creation() {
    let runner = HttpBenchmarkRunner::with_defaults();
    assert_eq!(runner.config.min_samples(), 5);
}

#[test]
fn test_benchmark_runner_relaxed() {
    let runner = HttpBenchmarkRunner::with_relaxed();
    assert_eq!(runner.config.min_samples(), 3);
}

#[test]
fn test_benchmark_runner_reproducible() {
    let runner = HttpBenchmarkRunner::with_reproducible();
    assert_eq!(runner.config.min_samples(), 10);
}

#[test]
fn test_cv_calculation_identical_values() {
    // Identical values should have CV = 0
    let samples = vec![100.0, 100.0, 100.0, 100.0, 100.0];
    let cv = HttpBenchmarkRunner::calculate_cv(&samples);
    assert!(
        cv < 0.001,
        "CV of identical values should be ~0, got {}",
        cv
    );
}

#[test]
fn test_cv_calculation_varied_values() {
    // Known CV case: mean=100, std=10, CV=0.1
    let samples = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let cv = HttpBenchmarkRunner::calculate_cv(&samples);
    // CV should be around 0.079 (std ~7.9, mean 100)
    assert!(
        cv > 0.05 && cv < 0.15,
        "CV should be reasonable, got {}",
        cv
    );
}

#[test]
fn test_cv_calculation_single_value() {
    let samples = vec![100.0];
    let cv = HttpBenchmarkRunner::calculate_cv(&samples);
    assert_eq!(cv, f64::MAX, "Single value should return MAX CV");
}

#[test]
fn test_cv_calculation_empty() {
    let samples: Vec<f64> = vec![];
    let cv = HttpBenchmarkRunner::calculate_cv(&samples);
    assert_eq!(cv, f64::MAX, "Empty samples should return MAX CV");
}

#[test]
fn test_compute_results_basic() {
    let latencies = vec![100.0, 110.0, 90.0, 105.0, 95.0];
    let throughputs = vec![50.0, 45.0, 55.0, 48.0, 52.0];
    let cold_start = 120.0;
    let cv_threshold = 0.10;

    let result =
        HttpBenchmarkRunner::compute_results(&latencies, &throughputs, cold_start, cv_threshold);

    assert_eq!(result.sample_count, 5);
    assert!((result.mean_latency_ms - 100.0).abs() < 0.01);
    assert!(result.p50_latency_ms > 0.0);
    assert!(result.p99_latency_ms >= result.p50_latency_ms);
    assert!(result.throughput_tps > 0.0);
    assert_eq!(result.cold_start_ms, 120.0);
}

#[test]
fn test_compute_results_percentiles() {
    // Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let latencies: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let throughputs = vec![];
    let cold_start = 1.0;
    let cv_threshold = 0.10;

    let result =
        HttpBenchmarkRunner::compute_results(&latencies, &throughputs, cold_start, cv_threshold);

    // p50 at index 5 = 6.0
    assert!((result.p50_latency_ms - 6.0).abs() < 0.1);
    // p99 at index 9 = 10.0
    assert!((result.p99_latency_ms - 10.0).abs() < 0.1);
}

include!("benchmark_runner.rs");
include!("imp_146b.rs");
include!("e2e_comparison.rs");
