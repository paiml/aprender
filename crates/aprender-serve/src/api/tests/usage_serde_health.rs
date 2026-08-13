
#[test]
fn test_usage_serde() {
    let usage = crate::api::Usage {
        prompt_tokens: 10,
        completion_tokens: 5,
        total_tokens: 15,
    };
    let json = serde_json::to_string(&usage).expect("JSON serialization failed");
    let deserialized: crate::api::Usage = serde_json::from_str(&json).expect("JSON deserialization failed");
    assert_eq!(deserialized.total_tokens, 15);
}

// ============================================================================
// B3: Additional endpoint error paths
// ============================================================================

#[tokio::test]
async fn test_health_endpoint() {
    // CRUX-C-34: /health returns 200 only when model is loaded.
    // Use create_test_app() (demo state with model), not shared mock state.
    let app = create_test_app();
    let request = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_nonexistent_endpoint_404() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("GET")
        .uri("/v1/nonexistent")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_wrong_method_for_endpoint() {
    let app = create_test_app_shared();
    // GET on a POST-only endpoint
    let request = Request::builder()
        .method("GET")
        .uri("/v1/generate")
        .body(Body::empty())
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::METHOD_NOT_ALLOWED
            || response.status() == StatusCode::NOT_FOUND,
    );
}

#[tokio::test]
async fn test_chat_completions_missing_messages() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"model":"test"}"#))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY,
    );
}

#[tokio::test]
async fn test_chat_completions_with_trace_header() {
    let app = create_test_app_shared();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-trace-level", "brick")
        .body(Body::from(
            r#"{"model":"test","messages":[{"role":"user","content":"Hi"}]}"#,
        ))
        .expect("test value should be present");

    let response = app.oneshot(request).await.expect("test value should be present");
    // aprender#2375(4): this fixture has NO model loaded, so exactly one answer
    // is correct - 503, the status `model_resolution_status` assigns to a
    // server-side condition. The disjunction this replaced admitted 200, 404
    // and 500 simultaneously and therefore could not fail whatever the route
    // did (the 0.63.0 audit's root cause).
    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "a model-less server reports the condition, it does not 404 a mounted route"
    );

}
