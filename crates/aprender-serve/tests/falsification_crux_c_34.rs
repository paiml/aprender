//! CRUX-C-34 falsification tests — `/health`, `/health/live`, `/health/ready`.
//!
//! Contract: `contracts/crux-C-34-v1.yaml` (competitor parity: vLLM + llama.cpp
//! server + k8s probe idioms).
//!
//! These tests exercise the axum router via `tower::ServiceExt::oneshot` so
//! they run entirely in-process (no TCP, no network). Every gate below is
//! enforced on the same router the `apr serve` CLI builds.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use realizar::api::{create_router, AppState};
use serial_test::serial;
use tower::ServiceExt;

// Serialization of tests that read or write `APR_TEST_FORCE_LOADING` is handled
// via `#[serial(env_force_loading)]` on each test below. This serializes at the
// test-harness level (one test at a time) rather than holding a `std::sync::Mutex`
// guard across the `.await` points inside each test — which would risk a deadlock
// (await_holding_lock). FALSIFY-001/004 tests expect the env var *unset*;
// FALSIFY-005 tests set it. Sharing one serial key keeps both sides mutually
// exclusive while leaving no guard live across any await.

fn router_with_model() -> axum::Router {
    let state = AppState::demo().expect("demo state should build");
    create_router(state)
}

fn router_without_model() -> axum::Router {
    let state = AppState::demo_mock().expect("mock state should build");
    create_router(state)
}

fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .expect("request build")
}

async fn json_body(resp: axum::http::Response<Body>) -> serde_json::Value {
    let bytes = resp
        .into_body()
        .collect()
        .await
        .expect("collect")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("json decode")
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-34-001: GET /health returns 200 with contract-shaped body.
// ---------------------------------------------------------------------------

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_001_health_schema_200() {
    let app = router_with_model();
    let resp = app.oneshot(get("/health")).await.expect("oneshot");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "FALSIFY-CRUX-C-34-001: ready server must return 200 on /health"
    );
    let json = json_body(resp).await;

    let status = json["status"].as_str().expect("status is string");
    assert_eq!(
        status, "ok",
        "FALSIFY-CRUX-C-34-001: ready server body.status must be \"ok\""
    );
    assert_eq!(
        json["model_loaded"].as_bool(),
        Some(true),
        "FALSIFY-CRUX-C-34-001: demo() state has a model loaded"
    );
    let uptime = json["uptime_sec"].as_f64().expect("uptime_sec is number");
    assert!(
        uptime > 0.0,
        "FALSIFY-CRUX-C-34-001: uptime_sec must be strictly positive, got {uptime}"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-34-002: body.status ∈ {ok, loading, degraded} — enum check.
// ---------------------------------------------------------------------------

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_002_status_enum_ready() {
    let app = router_with_model();
    let resp = app.oneshot(get("/health")).await.expect("oneshot");
    let json = json_body(resp).await;
    let status = json["status"].as_str().expect("string");
    assert!(
        matches!(status, "ok" | "loading" | "degraded"),
        "FALSIFY-CRUX-C-34-002: status \"{status}\" ∉ {{ok, loading, degraded}}"
    );
}

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_002_status_enum_loading() {
    // No model resident ⇒ status must be the "loading" enum variant, not a
    // free-form string like the legacy "healthy".
    let app = router_without_model();
    let resp = app.oneshot(get("/health")).await.expect("oneshot");
    let json = json_body(resp).await;
    let status = json["status"].as_str().expect("string");
    assert_eq!(
        status, "loading",
        "FALSIFY-CRUX-C-34-002: unloaded server must report status=\"loading\""
    );
    assert!(
        matches!(status, "ok" | "loading" | "degraded"),
        "FALSIFY-CRUX-C-34-002: status \"{status}\" ∉ {{ok, loading, degraded}}"
    );
}

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_002_loading_returns_503() {
    let app = router_without_model();
    let resp = app.oneshot(get("/health")).await.expect("oneshot");
    // Contract §health_response_schema: status==503 ⇔ body.status ∈ {loading, degraded}.
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "FALSIFY-CRUX-C-34-002: loading state must map to HTTP 503"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-34-003: uptime_sec is strictly monotonic across GETs.
// ---------------------------------------------------------------------------

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_003_uptime_monotonic() {
    // Two sequential oneshots on the SAME process must see uptime_sec advance.
    // Router is rebuilt each call but the OnceLock<Instant> is process-wide,
    // so start time is shared.
    let a = router_with_model()
        .oneshot(get("/health"))
        .await
        .expect("oneshot 1");
    let json_a = json_body(a).await;
    let u1 = json_a["uptime_sec"].as_f64().expect("u1");

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let b = router_with_model()
        .oneshot(get("/health"))
        .await
        .expect("oneshot 2");
    let json_b = json_body(b).await;
    let u2 = json_b["uptime_sec"].as_f64().expect("u2");

    assert!(
        u2 > u1,
        "FALSIFY-CRUX-C-34-003: uptime must strictly increase, got u1={u1} u2={u2}"
    );
    let delta = u2 - u1;
    assert!(
        (0.01..=5.0).contains(&delta),
        "FALSIFY-CRUX-C-34-003: delta {delta}s outside plausible [0.01, 5.0] window"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-34-004: /health/live (liveness) + /health/ready (readiness).
// k8s idioms: /health/live is always-200 once bound; /health/ready gates on
// model_loaded == true.
// ---------------------------------------------------------------------------

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_004_live_always_200_ready_server() {
    // Ready server: /health/live MUST return 200 (process is alive).
    let app = router_with_model();
    let resp = app.oneshot(get("/health/live")).await.expect("oneshot");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "FALSIFY-CRUX-C-34-004: /health/live must return 200 on a ready server"
    );
}

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_004_live_always_200_loading_server() {
    // Even when model is NOT loaded, /health/live MUST still return 200 —
    // this is what makes it a valid k8s liveness probe (process is alive
    // regardless of model readiness).
    let app = router_without_model();
    let resp = app.oneshot(get("/health/live")).await.expect("oneshot");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "FALSIFY-CRUX-C-34-004: /health/live must return 200 even when model not loaded"
    );
}

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_004_ready_200_when_model_loaded() {
    let app = router_with_model();
    let resp = app.oneshot(get("/health/ready")).await.expect("oneshot");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "FALSIFY-CRUX-C-34-004: /health/ready must return 200 iff model is loaded"
    );
}

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_004_ready_503_when_no_model() {
    let app = router_without_model();
    let resp = app.oneshot(get("/health/ready")).await.expect("oneshot");
    assert_eq!(
        resp.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "FALSIFY-CRUX-C-34-004: /health/ready must return 503 when no model (k8s readiness)"
    );
}

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_004_main_health_agrees_with_ready() {
    // If /health==200 (status=ok) then /health/ready MUST also be 200.
    let app_ready = router_with_model();
    let main = app_ready.oneshot(get("/health")).await.expect("oneshot");
    assert_eq!(main.status(), StatusCode::OK);
    let main_json = json_body(main).await;
    assert_eq!(main_json["status"].as_str(), Some("ok"));

    let app_ready2 = router_with_model();
    let ready = app_ready2
        .oneshot(get("/health/ready"))
        .await
        .expect("oneshot");
    assert_eq!(
        ready.status(),
        StatusCode::OK,
        "FALSIFY-CRUX-C-34-004: /health=ok implies /health/ready=200"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-C-34-005: startup loading state observable via test-only env.
// APR_TEST_FORCE_LOADING=1 forces status=loading + HTTP 503, independent of
// model presence. This gives k8s operators and our own integration harness a
// way to exercise the loading branch without racing against real model init.
// ---------------------------------------------------------------------------

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_005_force_loading_env_flips_status() {
    let prior = std::env::var("APR_TEST_FORCE_LOADING").ok();
    // SAFETY: std::env::{set_var, remove_var} are marked unsafe as of the
    // 2024 edition due to threading concerns. We serialize access through
    // ENV_LOCK above, and the env var is scoped to this single test.
    unsafe {
        std::env::set_var("APR_TEST_FORCE_LOADING", "1");
    }

    // Even with a model loaded, the force-loading hook MUST dominate.
    let app = router_with_model();
    let resp = app.oneshot(get("/health")).await.expect("oneshot");
    let status = resp.status();
    let json = json_body(resp).await;

    // Restore env before asserting (so a panic still cleans up).
    unsafe {
        match prior {
            Some(v) => std::env::set_var("APR_TEST_FORCE_LOADING", v),
            None => std::env::remove_var("APR_TEST_FORCE_LOADING"),
        }
    }

    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "FALSIFY-CRUX-C-34-005: APR_TEST_FORCE_LOADING=1 must force HTTP 503"
    );
    assert_eq!(
        json["status"].as_str(),
        Some("loading"),
        "FALSIFY-CRUX-C-34-005: APR_TEST_FORCE_LOADING=1 must force body.status=loading"
    );
}

#[tokio::test]
#[serial(env_force_loading)]
async fn falsify_crux_c_34_005_force_loading_ready_is_503() {
    // The readiness probe MUST also flip to 503 under the force-loading hook.

    let prior = std::env::var("APR_TEST_FORCE_LOADING").ok();
    unsafe {
        std::env::set_var("APR_TEST_FORCE_LOADING", "1");
    }

    let app = router_with_model();
    let resp = app.oneshot(get("/health/ready")).await.expect("oneshot");
    let status = resp.status();

    unsafe {
        match prior {
            Some(v) => std::env::set_var("APR_TEST_FORCE_LOADING", v),
            None => std::env::remove_var("APR_TEST_FORCE_LOADING"),
        }
    }

    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "FALSIFY-CRUX-C-34-005: force-loading must also flip /health/ready to 503"
    );
}
