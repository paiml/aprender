
#[tokio::test]
async fn test_realize_reload_endpoint() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"path": "/nonexistent/model.gguf"});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/reload")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_openai_completions_endpoint() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"model": "default", "prompt": "Hello", "max_tokens": 10});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_openai_embeddings_endpoint() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"model": "default", "input": "Hello world"});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_gpu_warmup_endpoint() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/gpu/warmup")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_gpu_status_endpoint() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/gpu/status")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_batch_completions_endpoint() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"prompts": ["Hello", "World"], "max_tokens": 10});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_tokenize_endpoint() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"text": "Hello world"});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_generate_endpoint() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"prompt": "Hello", "max_tokens": 5});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_batch_tokenize_endpoint() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"texts": ["Hello", "World"]});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_batch_generate_endpoint() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"prompts": ["Hello", "World"], "max_tokens": 5});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/batch/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_openai_models_endpoint() {
    let app = create_test_app_shared();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_openai_chat_completions_empty_messages() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"model": "default", "messages": []});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_openai_chat_completions_with_stream_false() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"model": "default", "messages": [{"role": "user", "content": "Hi"}], "stream": false});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_openai_chat_completions_with_temperature() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"model": "default", "messages": [{"role": "user", "content": "Test"}], "temperature": 0.5});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}

#[tokio::test]
async fn test_openai_chat_completions_with_max_tokens() {
    let app = create_test_app_shared();
    let body = serde_json::json!({"model": "default", "messages": [{"role": "user", "content": "Count"}], "max_tokens": 20});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).expect("JSON serialization failed")))
                .expect("test value should be present"),
        )
        .await
        .expect("test value should be present");
    assert!(is_acceptable_status(response.status()));
}
