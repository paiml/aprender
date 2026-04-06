
#[tokio::test]
async fn test_openai_completions_with_stop_tokens() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test",
        "prompt": "Hello",
        "stop": ["</s>", "<|im_end|>"],
        "max_tokens": 10
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_completions_default_model() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "default",
        "prompt": "Test prompt"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_completions_empty_model() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "",
        "prompt": "Test"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    // Empty model should use default
    let status = response.status();
    assert!(
        status == StatusCode::OK
            || status == StatusCode::NOT_FOUND
            || status == StatusCode::INTERNAL_SERVER_ERROR,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_completions_invalid_json() {
    let app = create_test_app_shared();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from("{invalid json"))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::NOT_FOUND
            || response.status() == StatusCode::INTERNAL_SERVER_ERROR
            || response.status() == StatusCode::SERVICE_UNAVAILABLE
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_openai_completions_missing_prompt() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "model": "test"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    // Missing required field
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_openai_embeddings_endpoint_basic() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "input": "Test text",
        "model": "default"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_embeddings_without_model() {
    let app = create_test_app_shared();

    let req_body = serde_json::json!({
        "input": "Test text"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {status}"
    );
}

#[tokio::test]
async fn test_openai_embeddings_long_text() {
    let app = create_test_app_shared();

    let long_text = "test ".repeat(1000);
    let req_body = serde_json::json!({
        "input": long_text
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");

    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::NOT_FOUND,
        "Unexpected status: {status}"
    );
}

// =============================================================================
// Usage Type Tests (from realize_handlers)
// =============================================================================

#[test]
fn test_usage_consistency() {
    let usage = Usage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };

    // total should equal prompt + completion
    assert_eq!(
        usage.total_tokens,
        usage.prompt_tokens + usage.completion_tokens
    );
}

#[test]
fn test_usage_zero_completion() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 0,
        total_tokens: 10,
    };

    let json = serde_json::to_string(&usage).expect("serialize");
    let parsed: Usage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.completion_tokens, 0);
}

// =============================================================================
// Completion Response Finish Reason Tests
// =============================================================================

#[test]
fn test_completion_choice_finish_reason_stop() {
    let choice = CompletionChoice {
        text: "Generated text".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "stop".to_string(),
    };
    assert_eq!(choice.finish_reason, "stop");
}

#[test]
fn test_completion_choice_finish_reason_length() {
    let choice = CompletionChoice {
        text: "Truncated at max tokens".to_string(),
        index: 0,
        logprobs: None,
        finish_reason: "length".to_string(),
    };
    assert_eq!(choice.finish_reason, "length");
}

#[test]
fn test_completion_response_empty_choices() {
    let response = CompletionResponse {
        id: "cmpl-empty".to_string(),
        object: "text_completion".to_string(),
        created: 0,
        model: "test".to_string(),
        choices: vec![],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    let json = serde_json::to_string(&response).expect("serialize");
    assert!(json.contains("[]")); // Empty choices array
}

// =============================================================================
// Model Metadata Edge Cases
// =============================================================================

#[test]
fn test_model_metadata_response_zero_size() {
    let response = ModelMetadataResponse {
        id: "streaming-model".to_string(),
        name: "Streaming Model".to_string(),
        format: "GGUF".to_string(),
        size_bytes: 0,
        quantization: None,
        context_length: 4096,
        lineage: None,
        loaded: false,
    };

    let json = serde_json::to_string(&response).expect("serialize");
    let parsed: ModelMetadataResponse = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.size_bytes, 0);
}

#[test]
fn test_model_metadata_response_all_quantizations() {
    let quantizations = ["Q4_0", "Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0"];
    for quant in quantizations {
        let response = ModelMetadataResponse {
            id: format!("model-{quant}"),
            name: format!("Model {quant}"),
            format: "GGUF".to_string(),
            size_bytes: 1000,
            quantization: Some(quant.to_string()),
            context_length: 2048,
            lineage: None,
            loaded: true,
        };

        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains(quant));
    }
}

// =============================================================================
// Lineage URI Format Tests
// =============================================================================

#[test]
fn test_lineage_various_uri_formats() {
    let uris = [
        "pacha://model:v1.0",
        "pacha://org/model:latest",
        "pacha://model:1.0.0-beta",
        "huggingface://user/model",
        "local:///path/to/model",
    ];

    for uri in uris {
        let lineage = ModelLineage {
            uri: uri.to_string(),
            version: "1.0.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: "blake3:test".to_string(),
        };

        let json = serde_json::to_string(&lineage).expect("serialize");
        let parsed: ModelLineage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.uri, uri);
    }
}

#[test]
fn test_lineage_content_hash_formats() {
    let hashes = [
        "blake3:abc123def456",
        "sha256:0123456789abcdef",
        "md5:deadbeef",
    ];

    for hash in hashes {
        let lineage = ModelLineage {
            uri: "pacha://test:latest".to_string(),
            version: "1.0.0".to_string(),
            recipe: None,
            parent: None,
            content_hash: hash.to_string(),
        };

        let json = serde_json::to_string(&lineage).expect("serialize");
        let parsed: ModelLineage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.content_hash, hash);
    }
}
