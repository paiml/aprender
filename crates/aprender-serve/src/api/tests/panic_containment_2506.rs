//! FALSIFY-SURF-7 / R14 (#2506): a handler panic must become a JSON 500, not a
//! dropped connection.
//!
//! `catch_unwind` and `CatchPanicLayer` appear nowhere in `aprender-serve/src`,
//! `apr-cli/src` or `aprender-mcp/src` outside a test helper and
//! `commands/qualify.rs`, and `tower_http`'s `catch-panic` feature was not even
//! enabled. So a panic in any axum handler unwound out of the service and the
//! client got a transport error — no status, no body, nothing to act on.
//!
//! That also broke an invariant this crate already asserts. `route_surface_2376`
//! establishes that **no error leaves this server as anything but actionable
//! JSON**; a dropped connection is the one error shape that escapes it entirely,
//! because it never becomes a response at all.
//!
//! These are black-box: a client with curl can observe every assertion.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    routing::get,
    Router,
};
use tower::util::ServiceExt;

use crate::api::{create_router_with_config, AppState, RouterConfig};

/// The real router, plus one route that panics.
///
/// Mounted onto the production router rather than a bare `Router::new()` on
/// purpose: the claim is that *this server* contains panics, which depends on
/// the layer stack `create_router_with_config` installs. A bare router would
/// prove something about axum instead.
async fn panic_probe() -> &'static str {
    // A named fn with a concrete return type: an `async {}` block whose only
    // expression is `panic!` has type `!`, which trips never-type-fallback.
    panic!("deliberate probe panic: a handler bug")
}

fn router_with_a_panicking_route() -> Router {
    // The probe route is added AFTER `create_router_with_config`, and axum's
    // `Router::layer` only wraps routes present when it is called -- so the
    // production layer does not cover it and the layer must be re-applied here.
    // That is a property of the test scaffold, not a hole in the server: every
    // route the server actually mounts is added to the table BEFORE the layer.
    //
    // Production coverage is therefore established by mutation rather than by
    // this scaffold. Making the REAL `/health` handler panic:
    //
    //     async fn health_handler(..) { panic!("MUTATION"); ... }
    //
    // yields `left: 500, right: 200` from the test below -- a response, with a
    // status. Before the layer existed the same mutation unwound out of the
    // service and `oneshot(..).expect(..)` fired instead. 500-instead-of-200 is
    // the whole finding: the panic became an answer.
    create_router_with_config(AppState::with_cache(10), RouterConfig::default())
        .route("/__panic_probe", get(panic_probe))
        .layer(tower_http::catch_panic::CatchPanicLayer::custom(
            crate::api::panic_to_json_500,
        ))
}

async fn body_string(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    String::from_utf8_lossy(&bytes).into_owned()
}

#[tokio::test]
async fn a_panicking_handler_returns_json_500_instead_of_dropping_the_connection() {
    let response = router_with_a_panicking_route()
        .oneshot(
            Request::builder()
                .uri("/__panic_probe")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("the service must RESPOND to a panicking handler, not unwind into the caller");

    assert_eq!(
        response.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "a handler panic must surface as 500"
    );

    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.contains("application/json"),
        "panic response is {content_type:?}, not JSON — route_surface_2376 \
         establishes that every error leaves this server as actionable JSON"
    );

    let body = body_string(response).await;
    let parsed: serde_json::Value =
        serde_json::from_str(&body).unwrap_or_else(|e| panic!("panic body is not JSON: {e}\n{body}"));
    assert!(
        parsed.get("error").is_some(),
        "panic body has no `error` field, so a client cannot act on it: {body}"
    );

    // The panic message must NOT reach the client: it is a Rust-internals
    // detail, and #2376 finding 7 already bans naming things a client cannot
    // act on. Its absence is also what distinguishes a contained panic from a
    // handler that merely formatted the panic itself.
    assert!(
        !body.contains("deliberate probe panic"),
        "the panic message leaked to the client: {body}"
    );
}

#[tokio::test]
async fn the_server_still_answers_after_a_handler_panics() {
    // Containment is worth nothing if the panic poisons the service. Same
    // router instance, panic first, then a real route.
    let app = router_with_a_panicking_route();

    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/__panic_probe")
                .body(Body::empty())
                .expect("build request"),
        )
        .await;

    let after = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("the server must survive a handler panic");

    assert_eq!(
        after.status(),
        StatusCode::OK,
        "/health stopped answering after another handler panicked"
    );
}

/// Non-vacuity companion. Both tests above are about a route that panics; if
/// the probe route were somehow not reached — a typo, a 404 — they would be
/// asserting things about a missing route rather than a contained panic.
#[tokio::test]
async fn the_panic_probe_route_is_actually_reached() {
    let response = router_with_a_panicking_route()
        .oneshot(
            Request::builder()
                .uri("/__panic_probe_typo")
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("404 path must respond");

    assert_eq!(
        response.status(),
        StatusCode::NOT_FOUND,
        "a route that does not exist must 404 — if this were also 500, the \
         tests above would pass without ever reaching a panicking handler"
    );
}
