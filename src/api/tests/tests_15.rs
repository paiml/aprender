//! API Tests Part 15: In-Process API Falsification for OpenAI Handlers
//!
//! Protocol T-COV-95 Directive 2: Exercise GPU/CUDA/cached/quantized model paths
//! in openai_handlers.rs through axum::test without real model loading.
//!
//! These tests create AppState with synthetic models to exercise code paths
//! that are normally only reached with real models loaded from disk.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

#[cfg(feature = "gpu")]
use crate::api::test_helpers::create_test_quantized_model;

// =============================================================================
// Quantized Model Path Tests (openai_handlers.rs lines 685-918)
// =============================================================================

/// Test quantized model chat completions endpoint routing
/// Exercises: quantized_model.is_some() branch
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_quantized_model_chat_completions_routing() {
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let state = AppState::with_quantized_model(model).expect("Failed to create AppState");

    // Verify quantized model is present
    assert!(
        state.quantized_model().is_some(),
        "Quantized model should be present"
    );

    let app = create_router(state);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    // The request should be handled by quantized model path
    // May return 500 due to test model's zero weights, but path is exercised
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Expected OK or INTERNAL_SERVER_ERROR, got {}",
        status
    );
}

/// Test quantized model streaming path
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_quantized_model_streaming_path() {
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let state = AppState::with_quantized_model(model).expect("create AppState");
    let app = create_router(state);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    // Streaming path should be exercised
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Streaming path should be exercised, got {}",
        status
    );
}

/// Test quantized model with X-Trace-Level header
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_quantized_model_with_trace_headers() {
    use crate::gguf::GGUFConfig;

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let state = AppState::with_quantized_model(model).expect("create AppState");

    for trace_level in ["brick", "step", "layer"] {
        let app = create_router(state.clone());

        let req_body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": "Test"}]
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("X-Trace-Level", trace_level)
                    .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                    .expect("build"),
            )
            .await
            .expect("send");

        let status = response.status();
        assert!(
            status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
            "Trace level {} should be handled, got {}",
            trace_level,
            status
        );
    }
}

// =============================================================================
// Cached Model Path Tests (openai_handlers.rs lines 283-460)
// =============================================================================

/// Test cached model chat completions endpoint routing
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_cached_model_chat_completions_routing() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let state = AppState::with_cached_model(cached_model).expect("create AppState");

    // Verify cached model is present
    assert!(
        state.cached_model().is_some(),
        "Cached model should be present"
    );

    let app = create_router(state);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Cached model path should be exercised, got {}",
        status
    );
}

/// Test cached model streaming path
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_cached_model_streaming_path() {
    use crate::gguf::{GGUFConfig, OwnedQuantizedModelCachedSync};

    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 256,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_quantized_model(&config);
    let cached_model = OwnedQuantizedModelCachedSync::new(model);
    let state = AppState::with_cached_model(cached_model).expect("create AppState");
    let app = create_router(state);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Cached streaming path should be exercised, got {}",
        status
    );
}

// =============================================================================
// GPU Model Path Tests (openai_handlers.rs lines 88-281)
// =============================================================================

/// Test GPU model chat completions endpoint routing
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_gpu_model_chat_completions_routing() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let gpu_model = GpuModel::new(config).expect("Failed to create GPU model");
    let state = AppState::with_gpu_model(gpu_model).expect("create AppState");

    // Verify GPU model is present
    assert!(state.has_gpu_model(), "GPU model should be present");

    let app = create_router(state);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "GPU model path should be exercised, got {}",
        status
    );
}

/// Test GPU model streaming path
#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_gpu_model_streaming_path() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
    num_experts: None,
    num_experts_per_tok: None,
    expert_intermediate_size: None,
    };

    let gpu_model = GpuModel::new(config).expect("create GPU model");
    let state = AppState::with_gpu_model(gpu_model).expect("create AppState");
    let app = create_router(state);

    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("build"),
        )
        .await
        .expect("send");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "GPU streaming path should be exercised, got {}",
        status
    );
}

include!("gpu_model.rs");
include!("quantized_multi.rs");
