
/// Registry: multiple non-existent models in sequence (no state leak)
#[tokio::test]
async fn test_registry_multiple_failures_no_state_leak() {
    let app = create_test_app_shared();

    // First request with non-existent model
    let req1 = serde_json::json!({
        "model": "fake-model-1",
        "messages": [{"role": "user", "content": "Hello"}]
    });

    let request1 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req1).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response1 = app.clone().oneshot(request1).await.expect("test value should be present");
    let status1 = response1.status();

    // Second request with different non-existent model
    let req2 = serde_json::json!({
        "model": "fake-model-2",
        "messages": [{"role": "user", "content": "World"}]
    });

    let request2 = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req2).expect("JSON serialization failed")))
        .expect("test value should be present");

    let response2 = app.clone().oneshot(request2).await.expect("test value should be present");
    let status2 = response2.status();

    // Both should fail gracefully with the same behaviour (no state
    // corruption). aprender#2609: each status is pinned at 503 rather than
    // admitted from a disjunction that excluded nothing.
    crate::api::test_helpers::assert_no_model_status(status1);
    crate::api::test_helpers::assert_no_model_status(status2);
    // aprender#2375(4): the equality is UNCONDITIONAL. Guarding it on
    // "both non-OK" let the pair pass with the two requests behaving
    // differently — the very state leak this test names.
    assert_eq!(
        status2, status1,
        "consecutive failures must be identical; a difference is leaked state"
    );
}

// =============================================================================
// Infinite Stream Falsification (T-COV-95 Final Corroboration)
// =============================================================================

/// Test streaming completion with bounded resource usage
/// (Popper: "Resource Boundedness" hypothesis test)
#[tokio::test]
async fn test_stream_resource_boundedness() {
    use std::time::Duration;
    use tokio::time::timeout;

    let app = create_test_app_shared();

    // Request with very large max_tokens to test resource limits
    let req_body = serde_json::json!({
        "model": "default",
        "messages": [{"role": "user", "content": "Generate a very long response"}],
        "stream": true,
        "max_tokens": 1000  // Large but bounded
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
        .expect("test value should be present");

    // The request MUST complete within a reasonable timeout
    // This falsifies the hypothesis of "Zombified Connections"
    let result = timeout(Duration::from_secs(30), app.oneshot(request)).await;

    assert!(
        result.is_ok(),
        "Stream request must complete within timeout (no zombified connection)"
    );

    let response = result.expect("test value should be present").expect("test value should be present");
    // Must return a response, not hang
    // aprender#2609: this was a disjunction over four or five statuses (several
    // listing NOT_FOUND twice), so it excluded nothing and passed against the
    // very behaviour #2609 reports. The shared test app is `demo_mock()` — a
    // server with no model of any kind — so the one correct answer for a
    // MOUNTED route is 503, and that is now what is asserted.
    crate::api::test_helpers::assert_no_model_status(response.status());
}

/// Test that stream handler doesn't consume unbounded memory
#[tokio::test]
async fn test_stream_memory_boundedness() {
    let app = create_test_app_shared();

    // Multiple concurrent requests should not cause memory issues
    let mut handles = vec![];

    for i in 0..3 {
        let app_clone = app.clone();
        let handle = tokio::spawn(async move {
            let req_body = serde_json::json!({
                "model": "default",
                "messages": [{"role": "user", "content": format!("Request {i}")}],
                "stream": true,
                "max_tokens": 50
            });

            let request = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&req_body).expect("JSON serialization failed")))
                .expect("test value should be present");

            app_clone.oneshot(request).await
        });
        handles.push(handle);
    }

    // All requests must complete (no deadlock, no OOM)
    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok(), "Concurrent stream request must complete");
        let response = result.expect("test value should be present");
        assert!(response.is_ok(), "Concurrent stream must not error");
    }
}
