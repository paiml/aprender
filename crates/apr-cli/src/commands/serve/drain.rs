//! Drain-on-signal for `apr serve` (#4449).
//!
//! Today `shutdown_signal()` awaits only `ctrl_c()`, and every `axum::serve`
//! site that DOES wire it up hands it straight to `with_graceful_shutdown`,
//! which stops the listener the instant the signal fires — a request that
//! arrives during shutdown gets a reset, not an answer. The WGPU site
//! (`handlers.rs` run_wgpu_server) has no shutdown handling at all.
//!
//! This module is the one shared implementation every `axum::serve` call
//! site (CPU, CUDA, WGPU) wires in the same two ways:
//! 1. `router.layer(drain::layer(drain.clone()))` — so a new request made
//!    while draining gets `503` + `Retry-After` instead of reaching a handler.
//! 2. `.with_graceful_shutdown(drain::shutdown_after_drain(drain))` — so the
//!    listener keeps accepting connections (answered by the layer above)
//!    until every in-flight request finishes or `--drain-timeout` elapses,
//!    at which point axum stops the accept loop.
//!
//! On SIGTERM/SIGINT the process therefore never closes the socket first —
//! it starts answering `503` first, and only closes once nothing is left in
//! flight (or the timeout forces it).

use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;

struct Inner {
    draining: AtomicBool,
    in_flight: AtomicUsize,
    idle: Notify,
    timeout_secs: u64,
}

/// Shared drain state: one per server, cloned into the middleware layer and
/// into the shutdown-signal task.
#[derive(Clone)]
pub struct DrainHandle(Arc<Inner>);

impl DrainHandle {
    pub fn new(timeout_secs: u64) -> Self {
        Self(Arc::new(Inner {
            draining: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
            idle: Notify::new(),
            timeout_secs,
        }))
    }

    fn is_draining(&self) -> bool {
        self.0.draining.load(Ordering::Acquire)
    }

    /// Enter drain mode. From this point every NEW request the middleware
    /// sees is answered `503` without reaching a handler; requests already
    /// past the check keep running.
    pub fn begin_drain(&self) {
        self.0.draining.store(true, Ordering::Release);
        // A request that finished between the last decrement and this call
        // already saw in_flight == 0 and skipped the notify (draining was
        // still false then). Catch that race here.
        if self.0.in_flight.load(Ordering::Acquire) == 0 {
            self.0.idle.notify_waiters();
        }
    }

    /// Wait until in-flight requests reach zero, or `timeout_secs` elapses.
    /// Returns `true` on a clean drain, `false` on timeout.
    async fn wait_for_idle(&self) -> bool {
        if self.0.in_flight.load(Ordering::Acquire) == 0 {
            return true;
        }
        let wait = self.0.idle.notified();
        tokio::time::timeout(Duration::from_secs(self.0.timeout_secs), wait)
            .await
            .is_ok()
    }
}

fn drain_rejection(retry_after_secs: u64) -> Response {
    let mut resp = (
        StatusCode::SERVICE_UNAVAILABLE,
        "server is draining, retry shortly",
    )
        .into_response();
    if let Ok(v) = HeaderValue::from_str(&retry_after_secs.to_string()) {
        resp.headers_mut().insert(header::RETRY_AFTER, v);
    }
    resp
}

/// The middleware body: 503 while draining, else count the request in-flight
/// for the duration of the handler (including a streamed body).
async fn drain_guard(State(drain): State<DrainHandle>, req: Request, next: Next) -> Response {
    if drain.is_draining() {
        return drain_rejection(drain.0.timeout_secs);
    }
    drain.0.in_flight.fetch_add(1, Ordering::AcqRel);
    let resp = next.run(req).await;
    let remaining = drain.0.in_flight.fetch_sub(1, Ordering::AcqRel) - 1;
    if remaining == 0 && drain.is_draining() {
        drain.0.idle.notify_waiters();
    }
    resp
}

/// Layer the drain guard onto a router, independent of the router's own
/// state type — every serve site's `axum::Router<AppState>` wires this in
/// the same call: `app = drain::layer(drain.clone(), app);`.
#[must_use]
pub fn layer<S>(drain: DrainHandle, router: axum::Router<S>) -> axum::Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router.layer(from_fn_with_state(drain, drain_guard))
}

/// Await SIGTERM (unix only) or Ctrl+C, whichever comes first.
#[cfg(unix)]
async fn wait_for_stop_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let sigterm = signal(SignalKind::terminate());
    match sigterm {
        Ok(mut term) => {
            tokio::select! {
                _ = term.recv() => {}
                _ = ctrl_c_ignoring_error() => {}
            }
        }
        Err(_) => ctrl_c_ignoring_error().await,
    }
}

#[cfg(not(unix))]
async fn wait_for_stop_signal() {
    ctrl_c_ignoring_error().await;
}

async fn ctrl_c_ignoring_error() {
    // A failure to install the handler leaves this future pending forever,
    // which is exactly the fallback we want: fall through to whichever
    // signal DID install (SIGTERM on unix) rather than exit immediately.
    if tokio::signal::ctrl_c().await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// The future to hand `axum::serve(..).with_graceful_shutdown(..)`.
///
/// Resolves only once drain has completed (in-flight reached zero, or the
/// timeout elapsed) — never on the raw signal — so the listener keeps
/// running, answered by [`drain_guard`], for the whole drain window.
pub async fn shutdown_after_drain(drain: DrainHandle) {
    wait_for_stop_signal().await;
    drain.begin_drain();
    drain.wait_for_idle().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    fn test_router(drain: DrainHandle) -> Router {
        let app = Router::new().route("/health", get(|| async { (StatusCode::OK, "ok") }));
        layer(drain, app)
    }

    /// Falsifier 1: a request already in flight when drain begins still
    /// completes 200 — the guard must not cancel it mid-handler.
    #[tokio::test]
    async fn in_flight_request_completes_after_drain_begins() {
        let drain = DrainHandle::new(5);
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let (started_h, release_h) = (started.clone(), release.clone());

        let app = Router::new().route(
            "/slow",
            get(move || {
                let started = started_h.clone();
                let release = release_h.clone();
                async move {
                    started.notify_one();
                    release.notified().await;
                    (StatusCode::OK, "done")
                }
            }),
        );
        let app = layer(drain.clone(), app);

        let req = axum::http::Request::builder()
            .uri("/slow")
            .body(Body::empty())
            .expect("request builds");
        let call = tokio::spawn(app.oneshot(req));

        started.notified().await;
        // The signal fires while the request is still inside the handler.
        drain.begin_drain();
        release.notify_one();

        let resp = call
            .await
            .expect("task joins")
            .expect("service is infallible");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Falsifier 2: a request made after drain begins gets 503 with
    /// Retry-After, never reaching the handler.
    #[tokio::test]
    async fn new_request_after_drain_gets_503_with_retry_after() {
        let drain = DrainHandle::new(7);
        drain.begin_drain();
        let app = test_router(drain);

        let req = axum::http::Request::builder()
            .uri("/health")
            .body(Body::empty())
            .expect("request builds");
        let resp = app.oneshot(req).await.expect("service is infallible");

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let retry_after = resp
            .headers()
            .get(header::RETRY_AFTER)
            .expect("Retry-After header present")
            .to_str()
            .expect("header is ascii");
        assert_eq!(retry_after, "7");
    }

    #[tokio::test]
    async fn wait_for_idle_returns_true_once_in_flight_hits_zero() {
        let drain = DrainHandle::new(5);
        drain.0.in_flight.fetch_add(1, Ordering::AcqRel);

        let waiter = drain.clone();
        let wait_task = tokio::spawn(async move { waiter.wait_for_idle().await });

        // Give the waiter a moment to register, then finish the "request".
        tokio::task::yield_now().await;
        drain.begin_drain();
        let remaining = drain.0.in_flight.fetch_sub(1, Ordering::AcqRel) - 1;
        assert_eq!(remaining, 0);
        drain.0.idle.notify_waiters();

        assert!(wait_task.await.expect("task joins"));
    }

    #[tokio::test]
    async fn wait_for_idle_times_out_when_a_request_never_finishes() {
        let drain = DrainHandle::new(0);
        drain.0.in_flight.fetch_add(1, Ordering::AcqRel);
        drain.begin_drain();
        assert!(!drain.wait_for_idle().await);
    }
}
