# API Testing

This chapter covers the canonical approach to testing axum web APIs in Rust using `tower::ServiceExt`.

## Why tower::ServiceExt?

The idiomatic way to test axum applications is using `tower::ServiceExt::oneshot()`:

| Approach | Pros | Cons |
|----------|------|------|
| **`tower::oneshot()`** | Fast, no server startup, same code path | Requires understanding tower |
| External HTTP client | Tests full stack | Slow, requires running server |
| Mock handlers | Isolated | Doesn't test routing |

`tower::oneshot()` is the **recommended approach** because:
- No actual server startup (fast)
- Direct router testing (no network overhead)
- Full request/response cycle
- Same code path as production

## Dependencies

```toml
[dev-dependencies]
# Canonical tower/axum testing
http-body-util = "0.1"
hyper = { version = "1.4", features = ["full"] }
mime = "0.3"
```

## Basic Test Structure

```rust
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use tower::ServiceExt;  // Provides oneshot()

#[tokio::test]
async fn test_health_endpoint() {
    // Create router (same as production)
    let app = create_serve_router(create_test_state());

    // Send request via oneshot (no server needed)
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Assert response
    assert_eq!(response.status(), StatusCode::OK);
}
```

## Testing POST Endpoints

```rust
#[tokio::test]
async fn test_predict_endpoint() {
    let app = create_serve_router(create_test_state());

    let request = PredictRequest {
        model_id: None,
        features: vec![0.9, 0.9],
        options: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
```

## Extracting Response Bodies

```rust
/// Helper to extract body as string
async fn body_to_string(body: Body) -> String {
    let bytes = body.collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// Helper to extract body as JSON
async fn body_to_json<T: serde::de::DeserializeOwned>(body: Body) -> T {
    let bytes = body.collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn test_response_body() {
    let app = create_serve_router(create_test_state());

    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    let body = body_to_string(response.into_body()).await;
    assert!(body.contains("healthy"));
}
```

## Testing Error Cases

```rust
#[tokio::test]
async fn test_404_for_unknown_route() {
    let app = create_serve_router(create_test_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/unknown/route")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_invalid_json() {
    let app = create_serve_router(create_test_state());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/predict")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
```

## Testing with Different States

```rust
/// Create ServeState with a loaded model
fn create_test_state() -> ServeState {
    let model = create_trained_model();
    ServeState::with_logistic_regression(model, "v1".to_string(), 2)
}

/// Create ServeState without a model (for error testing)
fn create_empty_state() -> ServeState {
    ServeState::new("empty".to_string(), "v0".to_string())
}

#[tokio::test]
async fn test_predict_without_model() {
    let app = create_serve_router(create_empty_state());

    let response = app
        .oneshot(/* POST /predict */)
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}
```

## Performance Testing

```rust
#[tokio::test]
async fn test_predict_latency_is_submillisecond() {
    let app = create_serve_router(create_test_state());

    let start = std::time::Instant::now();

    let response = app
        .oneshot(/* request */)
        .await
        .unwrap();

    let elapsed = start.elapsed();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        elapsed.as_millis() < 10,
        "Prediction took {}ms, expected <10ms",
        elapsed.as_millis()
    );
}
```

## Running Tests

```bash
# Run all HTTP tests
cargo test --features aprender-serve --test serve_http_tests

# Run with output
cargo test --features aprender-serve --test serve_http_tests -- --nocapture

# Run specific test
cargo test --features aprender-serve test_predict_endpoint
```

## Test Organization

```
tests/
├── serve_http_tests.rs      # HTTP integration tests (tower::oneshot)
├── load_test.rs             # Load/stress tests
└── property_*.rs            # Property-based tests
```

## Best Practices

### 1. Use oneshot() for Unit-Style Tests

```rust
// Good: Fast, isolated
let response = app.oneshot(request).await.unwrap();

// Avoid: Starts actual server (slow)
let server = axum::serve(listener, app).await;
```

### 2. Create Helper Functions

```rust
fn create_test_state() -> ServeState { /* ... */ }
fn create_predict_request(features: Vec<f32>) -> Request<Body> { /* ... */ }
async fn body_to_json<T>(body: Body) -> T { /* ... */ }
```

### 3. Test Error Cases

Every error path should have a test:
- Invalid JSON
- Missing required fields
- Wrong HTTP method
- Model not loaded
- Invalid input dimensions

### 4. Validate Response Bodies

Don't just check status codes:

```rust
assert_eq!(response.status(), StatusCode::OK);

// Also check body
let body = body_to_string(response.into_body()).await;
assert!(body.contains("prediction"));
```

## Complete Test File Example

See `tests/serve_http_tests.rs` for a complete example with 18 tests covering:
- Health endpoint
- Ready endpoint
- Predict endpoint (success, errors, probabilities)
- Batch predict endpoint
- Models endpoint
- Metrics endpoint
- Error handling (404, invalid JSON, wrong dimensions)
- Performance validation

## References

- [axum Testing Documentation](https://docs.rs/axum/latest/axum/#testing)
- [tower::ServiceExt](https://docs.rs/tower/latest/tower/trait.ServiceExt.html)
- [http-body-util](https://docs.rs/http-body-util)
