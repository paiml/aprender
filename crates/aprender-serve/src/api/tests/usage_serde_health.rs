
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
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}
