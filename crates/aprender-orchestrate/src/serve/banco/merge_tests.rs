//! Model merge endpoint tests.

use super::handlers_merge::{MergeRequest, MergeStrategy};

// ============================================================================
// Merge type unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_MERGE_001_strategy_serde() {
    let json = serde_json::json!("weighted_average");
    let strategy: MergeStrategy = serde_json::from_value(json).expect("parse");
    assert_eq!(strategy, MergeStrategy::WeightedAverage);

    let json = serde_json::json!("ties");
    let strategy: MergeStrategy = serde_json::from_value(json).expect("parse");
    assert_eq!(strategy, MergeStrategy::Ties);

    let json = serde_json::json!("slerp");
    let strategy: MergeStrategy = serde_json::from_value(json).expect("parse");
    assert_eq!(strategy, MergeStrategy::Slerp);
}

#[test]
#[allow(non_snake_case)]
fn test_MERGE_002_request_deserialize() {
    let json = serde_json::json!({
        "models": ["model-a.gguf", "model-b.gguf"],
        "strategy": "slerp",
        "interpolation_t": 0.3
    });
    let req: MergeRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.models.len(), 2);
    assert_eq!(req.strategy, MergeStrategy::Slerp);
    assert!((req.interpolation_t.unwrap() - 0.3).abs() < f32::EPSILON);
}

#[test]
#[allow(non_snake_case)]
fn test_MERGE_003_request_with_weights() {
    let json = serde_json::json!({
        "models": ["a", "b", "c"],
        "strategy": "weighted_average",
        "weights": [0.5, 0.3, 0.2]
    });
    let req: MergeRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.weights.as_ref().unwrap().len(), 3);
}

#[test]
#[allow(non_snake_case)]
fn test_MERGE_004_request_ties_config() {
    let json = serde_json::json!({
        "models": ["a", "b"],
        "strategy": "ties",
        "density": 0.3
    });
    let req: MergeRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.strategy, MergeStrategy::Ties);
    assert!((req.density.unwrap() - 0.3).abs() < f32::EPSILON);
}

#[test]
#[allow(non_snake_case)]
fn test_MERGE_005_request_dare_config() {
    let json = serde_json::json!({
        "models": ["a", "b"],
        "strategy": "dare",
        "drop_prob": 0.7,
        "seed": 42
    });
    let req: MergeRequest = serde_json::from_value(json).expect("parse");
    assert_eq!(req.strategy, MergeStrategy::Dare);
    assert!((req.drop_prob.unwrap() - 0.7).abs() < f32::EPSILON);
    assert_eq!(req.seed, Some(42));
}

// ============================================================================
// Merge endpoint tests
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MERGE_HDL_001_weighted_average() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "models": ["model-a.gguf", "model-b.gguf"],
        "strategy": "weighted_average",
        "weights": [0.7, 0.3]
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/models/merge")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert!(json["merge_id"].as_str().expect("id").starts_with("merge-"));
    assert_eq!(json["strategy"], "weighted_average");
    assert_eq!(json["models"].as_array().expect("models").len(), 2);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MERGE_HDL_002_ties_merge() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "models": ["a", "b", "c"],
        "strategy": "ties",
        "density": 0.2
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/models/merge")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["strategy"], "ties");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MERGE_HDL_003_slerp_merge() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "models": ["model-a", "model-b"],
        "strategy": "slerp",
        "interpolation_t": 0.5
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/models/merge")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MERGE_HDL_004_slerp_rejects_three_models() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "models": ["a", "b", "c"],
        "strategy": "slerp"
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/models/merge")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MERGE_HDL_005_rejects_single_model() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "models": ["only-one"],
        "strategy": "weighted_average"
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/models/merge")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MERGE_HDL_006_list_strategies() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let response = app
        .oneshot(Request::get("/api/v1/models/merge/strategies").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    let strategies = json["strategies"].as_array().expect("strategies");
    assert_eq!(strategies.len(), 4);
}

#[tokio::test]
#[allow(non_snake_case)]
async fn test_MERGE_HDL_007_dare_with_seed() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    let body = serde_json::json!({
        "models": ["a", "b"],
        "strategy": "dare",
        "drop_prob": 0.5,
        "seed": 42
    });
    let response = app
        .oneshot(
            Request::post("/api/v1/models/merge")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("json")))
                .expect("req"),
        )
        .await
        .expect("resp");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1_048_576).await.expect("body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(json["strategy"], "dare");
}
