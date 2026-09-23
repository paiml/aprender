//! #3979: every `apr serve` router advertises exactly what it mounts, and its streams
//! end with their wire's terminal event. Driven through the REAL router builders.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

async fn call(
    router: &axum::Router,
    method: Method,
    path: &str,
    body: &str,
) -> (StatusCode, String) {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// `GET /` as the GGUF router serves it, or panic naming what was served instead.
async fn index(router: &axum::Router) -> Vec<String> {
    let (st, body) = call(router, Method::GET, "/", "").await;
    assert_eq!(st, StatusCode::OK, "GET / must answer: {body}");
    let v: serde_json::Value = serde_json::from_str(&body)
        .unwrap_or_else(|_| panic!("GET / must be the JSON route index, got {body:?}"));
    assert_eq!(v["service"], "apr serve", "{v}");
    v["routes"]
        .as_array()
        .expect("routes array")
        .iter()
        .map(|r| r.as_str().unwrap().to_string())
        .collect()
}

/// Both directions: every listed route is MOUNTED (not the 404 fallback, not a 405 for
/// the method the index names), and an unlisted path 404s carrying the SAME list.
async fn index_is_the_mounted_surface(router: &axum::Router) -> Vec<String> {
    let routes = index(router).await;
    assert!(
        routes.first().map(String::as_str) == Some("GET /"),
        "{routes:?}"
    );
    for r in routes.iter().skip(1) {
        let (m, p) = r.split_once(' ').expect("METHOD /path");
        let method = Method::from_bytes(m.as_bytes()).unwrap();
        let (st, body) = call(router, method, p, "{}").await;
        assert!(
            st != StatusCode::NOT_FOUND && st != StatusCode::METHOD_NOT_ALLOWED,
            "the index lists `{r}` but the router does not serve it: {st} {body}"
        );
    }
    let (st, body) = call(router, Method::GET, "/not-a-route-3979", "").await;
    assert_eq!(st, StatusCode::NOT_FOUND, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let listed: Vec<String> = v["routes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_str().unwrap().to_string())
        .collect();
    assert_eq!(listed, routes, "the 404 must carry the same index as GET /");
    routes
}

fn safetensors_state() -> super::safetensors::SafeTensorsState {
    super::safetensors::SafeTensorsState {
        transformer: None,
        tokenizer_info: None,
        model_path: "m.safetensors".into(),
    }
}

#[tokio::test]
async fn the_apr_cpu_router_serves_an_index_of_exactly_what_it_mounts() {
    let r = super::handlers::build_demo_apr_cpu_router_for_test();
    let routes = index_is_the_mounted_surface(&r).await;
    for must in [
        "GET /health",
        "POST /v1/chat/completions",
        "POST /api/chat",
        "POST /api/show",
        "DELETE /api/delete",
    ] {
        assert!(
            routes.iter().any(|x| x == must),
            "APR-CPU index must list `{must}` (it is mounted): {routes:?}"
        );
    }
}

#[tokio::test]
async fn the_safetensors_inspection_router_serves_an_index_of_exactly_what_it_mounts() {
    let r = super::safetensors::safetensors_app(
        serde_json::json!({"count": 0}),
        false,
        safetensors_state(),
        "m".into(),
    );
    let routes = index_is_the_mounted_surface(&r).await;
    assert_eq!(
        routes,
        vec!["GET /", "GET /health", "GET /tensors"],
        "inspection-only surface"
    );
}

#[tokio::test]
async fn the_safetensors_inference_router_serves_an_index_of_exactly_what_it_mounts() {
    let r = super::safetensors::safetensors_app(
        serde_json::json!({"count": 0}),
        true,
        safetensors_state(),
        "m".into(),
    );
    let routes = index_is_the_mounted_surface(&r).await;
    for must in [
        "POST /v1/chat/completions",
        "POST /api/chat",
        "GET /api/tags",
        "POST /api/show",
    ] {
        assert!(
            routes.iter().any(|x| x == must),
            "SafeTensors index must list `{must}`: {routes:?}"
        );
    }
}

/// The Ollama wire's terminal is `done: true` on the LAST NDJSON line.
fn last_ndjson_is_done(body: &str) -> bool {
    body.lines()
        .filter(|l| !l.trim().is_empty())
        .last()
        .and_then(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .is_some_and(|v| v["done"] == true)
}

#[tokio::test]
async fn the_apr_cpu_ollama_stream_ends_with_done_true() {
    let r = super::handlers::build_demo_streaming_apr_cpu_router_for_test();
    let (st, body) = call(
        &r,
        Method::POST,
        "/api/chat",
        r#"{"model":"apr","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
    )
    .await;
    assert_eq!(st, StatusCode::OK, "{body}");
    assert!(
        last_ndjson_is_done(&body),
        "APR-CPU /api/chat stream must end with done:true: {body}"
    );
}

#[tokio::test]
async fn the_safetensors_ollama_stream_ends_with_done_true() {
    let r = super::safetensors::safetensors_app(
        serde_json::json!({"count": 0}),
        true,
        safetensors_state(),
        "m".into(),
    );
    let (st, body) = call(
        &r,
        Method::POST,
        "/api/chat",
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
    )
    .await;
    assert!(
        last_ndjson_is_done(&body),
        "SafeTensors /api/chat stream must end with done:true ({st}): {body}"
    );
}
