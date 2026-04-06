//! Integration tests for multi-model API
//!
//! Tests the ModelRegistry integration with the HTTP API,
//! including model selection, concurrent access, and error handling.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use realizar::{
    api::{
        create_router, AppState, GenerateRequest, GenerateResponse, ModelsResponse,
        TokenizeRequest, TokenizeResponse,
    },
    layers::{Model, ModelConfig},
    registry::ModelRegistry,
    tokenizer::BPETokenizer,
};
use tower::ServiceExt;

/// Create a test model with specific vocab size
fn create_test_model(vocab_size: usize) -> (Model, BPETokenizer) {
    let config = ModelConfig {
        vocab_size,
        hidden_dim: 32,
        num_heads: 1,
        num_layers: 1,
        intermediate_dim: 64,
        eps: 1e-5,
    };
    let model = Model::new(config).expect("test");

    let vocab: Vec<String> = (0..vocab_size)
        .map(|i| {
            if i == 0 {
                "<unk>".to_string()
            } else {
                format!("token{i}")
            }
        })
        .collect();
    let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>").expect("test");

    (model, tokenizer)
}

#[tokio::test]
async fn test_multi_model_registry_list() {
    // Create registry with multiple models
    let registry = ModelRegistry::new(5);

    let (model1, tokenizer1) = create_test_model(100);
    let (model2, tokenizer2) = create_test_model(200);

    registry
        .register("model-1", model1, tokenizer1)
        .expect("test");
    registry
        .register("model-2", model2, tokenizer2)
        .expect("test");

    // Create app state with registry
    let state = AppState::with_registry(registry, "model-1").expect("test");
    let app = create_router(state);

    // Request models list
    let request = Request::builder()
        .uri("/models")
        .body(Body::empty())
        .expect("test");

    let response = app.oneshot(request).await.expect("test");
    assert_eq!(response.status(), StatusCode::OK);

    // Parse response
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let models_response: ModelsResponse = serde_json::from_slice(&body).expect("test");

    assert_eq!(models_response.models.len(), 2);

    let model_ids: Vec<String> = models_response
        .models
        .iter()
        .map(|m| m.id.clone())
        .collect();
    assert!(model_ids.contains(&"model-1".to_string()));
    assert!(model_ids.contains(&"model-2".to_string()));
}

#[tokio::test]
async fn test_multi_model_tokenize_with_model_id() {
    // Create registry with multiple models
    let registry = ModelRegistry::new(5);

    let (model1, tokenizer1) = create_test_model(100);
    let (model2, tokenizer2) = create_test_model(200);

    registry
        .register("small-model", model1, tokenizer1)
        .expect("test");
    registry
        .register("large-model", model2, tokenizer2)
        .expect("test");

    let state = AppState::with_registry(registry, "small-model").expect("test");
    let app = create_router(state);

    // Tokenize with specific model
    let tokenize_request = TokenizeRequest {
        text: "token1 token2".to_string(),
        model_id: Some("large-model".to_string()),
    };

    let request = Request::builder()
        .uri("/tokenize")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&tokenize_request).expect("test"),
        ))
        .expect("test");

    let response = app.oneshot(request).await.expect("test");
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let tokenize_response: TokenizeResponse = serde_json::from_slice(&body).expect("test");

    // Verify tokens were generated
    assert!(tokenize_response.num_tokens > 0);
}

#[tokio::test]
async fn test_multi_model_tokenize_default_model() {
    // Create registry with default model
    let registry = ModelRegistry::new(5);

    let (model1, tokenizer1) = create_test_model(100);
    let (model2, tokenizer2) = create_test_model(200);

    registry
        .register("small-model", model1, tokenizer1)
        .expect("test");
    registry
        .register("large-model", model2, tokenizer2)
        .expect("test");

    let state = AppState::with_registry(registry, "small-model").expect("test");
    let app = create_router(state);

    // Tokenize without specifying model_id (should use default)
    let tokenize_request = TokenizeRequest {
        text: "token1 token2".to_string(),
        model_id: None,
    };

    let request = Request::builder()
        .uri("/tokenize")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&tokenize_request).expect("test"),
        ))
        .expect("test");

    let response = app.oneshot(request).await.expect("test");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_multi_model_generate_with_model_id() {
    // Create registry with multiple models
    let registry = ModelRegistry::new(5);

    let (model1, tokenizer1) = create_test_model(100);
    let (model2, tokenizer2) = create_test_model(200);

    registry
        .register("model-a", model1, tokenizer1)
        .expect("test");
    registry
        .register("model-b", model2, tokenizer2)
        .expect("test");

    let state = AppState::with_registry(registry, "model-a").expect("test");
    let app = create_router(state);

    // Generate with specific model
    let generate_request = GenerateRequest {
        prompt: "token1".to_string(),
        max_tokens: 5,
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        strategy: "greedy".to_string(),
        seed: Some(42),
        model_id: Some("model-b".to_string()),
    };

    let request = Request::builder()
        .uri("/generate")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&generate_request).expect("test"),
        ))
        .expect("test");

    let response = app.oneshot(request).await.expect("test");
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let generate_response: GenerateResponse = serde_json::from_slice(&body).expect("test");

    // Verify generation occurred
    assert!(!generate_response.text.is_empty());
    assert!(generate_response.num_generated > 0);
}

#[tokio::test]
async fn test_multi_model_generate_default_model() {
    // Create registry with default model
    let registry = ModelRegistry::new(5);

    let (model1, tokenizer1) = create_test_model(100);
    let (model2, tokenizer2) = create_test_model(200);

    registry
        .register("default", model1, tokenizer1)
        .expect("test");
    registry
        .register("alternative", model2, tokenizer2)
        .expect("test");

    let state = AppState::with_registry(registry, "default").expect("test");
    let app = create_router(state);

    // Generate without specifying model_id (should use default)
    let generate_request = GenerateRequest {
        prompt: "token1".to_string(),
        max_tokens: 5,
        temperature: 1.0,
        top_k: 50,
        top_p: 0.9,
        strategy: "greedy".to_string(),
        seed: Some(42),
        model_id: None,
    };

    let request = Request::builder()
        .uri("/generate")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&generate_request).expect("test"),
        ))
        .expect("test");

    let response = app.oneshot(request).await.expect("test");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_multi_model_not_found_error() {
    // Create registry with one model
    let registry = ModelRegistry::new(5);

    let (model, tokenizer) = create_test_model(100);
    registry.register("exists", model, tokenizer).expect("test");

    let state = AppState::with_registry(registry, "exists").expect("test");
    let app = create_router(state);

    // Try to use non-existent model
    let generate_request = GenerateRequest {
        prompt: "token1".to_string(),
        max_tokens: 5,
        temperature: 1.0,
        top_k: 50,
        top_p: 0.9,
        strategy: "greedy".to_string(),
        seed: None,
        model_id: Some("does-not-exist".to_string()),
    };

    let request = Request::builder()
        .uri("/generate")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&generate_request).expect("test"),
        ))
        .expect("test");

    let response = app.oneshot(request).await.expect("test");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    // Verify error response contains expected message
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let body_str = String::from_utf8_lossy(&body);
    assert!(body_str.contains("not found"));
}

#[tokio::test]
async fn test_multi_model_backward_compatibility() {
    // Test that single-model mode still works
    let (model, tokenizer) = create_test_model(100);

    let state = AppState::new(model, tokenizer);
    let app = create_router(state);

    // Old-style request without model_id
    let generate_request = GenerateRequest {
        prompt: "token1".to_string(),
        max_tokens: 3,
        temperature: 1.0,
        top_k: 50,
        top_p: 0.9,
        strategy: "greedy".to_string(),
        seed: Some(42),
        model_id: None,
    };

    let request = Request::builder()
        .uri("/generate")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&generate_request).expect("test"),
        ))
        .expect("test");

    let response = app.oneshot(request).await.expect("test");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_multi_model_single_model_mode_default() {
    // Test that /models in single-model mode returns default model
    let (model, tokenizer) = create_test_model(100);

    let state = AppState::new(model, tokenizer);
    let app = create_router(state);

    let request = Request::builder()
        .uri("/models")
        .body(Body::empty())
        .expect("test");

    let response = app.oneshot(request).await.expect("test");
    // Should return OK with default model
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let models_response: ModelsResponse = serde_json::from_slice(&body).expect("test");

    // Single model mode returns a default model entry
    assert_eq!(models_response.models.len(), 1);
    assert_eq!(models_response.models[0].id, "default");
    assert_eq!(models_response.models[0].name, "Default Model");
    assert!(models_response.models[0].loaded);
}

#[tokio::test]
async fn test_concurrent_model_access() {
    use tokio::task::JoinSet;

    // Create registry with multiple models
    let registry = Arc::new(ModelRegistry::new(10));

    let (model1, tokenizer1) = create_test_model(100);
    let (model2, tokenizer2) = create_test_model(100);
    let (model3, tokenizer3) = create_test_model(100);

    registry
        .register("model-1", model1, tokenizer1)
        .expect("test");
    registry
        .register("model-2", model2, tokenizer2)
        .expect("test");
    registry
        .register("model-3", model3, tokenizer3)
        .expect("test");

    // Spawn concurrent tasks accessing different models
    let mut set = JoinSet::new();

    for i in 0..30 {
        let registry_clone = Arc::clone(&registry);
        set.spawn(async move {
            let model_id = format!("model-{}", (i % 3) + 1);
            let result = registry_clone.get(&model_id);
            assert!(result.is_ok(), "Failed to get model {}", model_id);
        });
    }

    // Wait for all tasks to complete
    while let Some(result) = set.join_next().await {
        result.expect("test");
    }
}

#[tokio::test]
async fn test_model_info_metadata() {
    use realizar::registry::ModelInfo;

    // Create registry with detailed model info
    let registry = ModelRegistry::new(5);

    let (model, tokenizer) = create_test_model(100);

    let info = ModelInfo {
        id: "llama-7b".to_string(),
        name: "Llama 7B".to_string(),
        description: "7B parameter model".to_string(),
        format: "GGUF".to_string(),
        loaded: false,
    };

    registry
        .register_with_info(info, model, tokenizer)
        .expect("test");

    let state = AppState::with_registry(registry, "llama-7b").expect("test");
    let app = create_router(state);

    // Request models list
    let request = Request::builder()
        .uri("/models")
        .body(Body::empty())
        .expect("test");

    let response = app.oneshot(request).await.expect("test");
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("test");
    let models_response: ModelsResponse = serde_json::from_slice(&body).expect("test");

    assert_eq!(models_response.models.len(), 1);
    let model_info = &models_response.models[0];
    assert_eq!(model_info.id, "llama-7b");
    assert_eq!(model_info.name, "Llama 7B");
    assert_eq!(model_info.description, "7B parameter model");
    assert_eq!(model_info.format, "GGUF");
    assert!(model_info.loaded);
}
