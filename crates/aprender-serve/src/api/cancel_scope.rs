//! The handler half of aprender#2376(3): turn an axum client-disconnect into a
//! cancellation signal the decode loops can actually observe.
//!
//! # Why "axum drops the future" was not already enough
//!
//! Every generate handler in this crate calls its backend *synchronously* from
//! inside its `async fn`:
//!
//! ```ignore
//! pub async fn generate_handler(...) -> ... {
//!     let generated = model.generate_with_cache(&prompt_ids, &q_config)?; // no .await
//!     ...
//! }
//! ```
//!
//! Axum cancels an abandoned request by **dropping the response future**, and a
//! future can only be dropped while it is suspended at an `.await`. There is no
//! `.await` anywhere inside a synchronous decode loop, so the task never yielded,
//! the drop never landed, and generation ran all the way to `max_tokens` for a
//! client that had already hung up. That is the ~250%-CPU-with-zero-open-
//! connections symptom in aprender#2376(3).
//!
//! # The shape that works
//!
//! [`cancel_on_disconnect`] is a `tower` layer over the whole router. Per request
//! it:
//!
//! 1. mints a [`CancelToken`] and puts a clone in the request extensions, so any
//!    handler can pull it out with `Extension<CancelToken>` and install it on its
//!    generation config;
//! 2. holds a [`CancelOnDrop`](crate::generate::CancelOnDrop) guard for the whole
//!    middleware future — this is the piece axum drops on disconnect; and
//! 3. runs the inner handler in a **separate task** ([`tokio::spawn`]) and awaits
//!    its `JoinHandle`.
//!
//! Step 3 is not decoration. Awaiting is what gives this future a suspension point
//! to be dropped at, and running the handler in its own task is what lets the
//! decode loop still be alive — and therefore still able to observe the flag —
//! *after* the drop. Dropping a `JoinHandle` detaches its task rather than killing
//! it, and nothing can preempt a synchronous loop anyway, which is exactly why the
//! loop has to cooperate by polling.
//!
//! Take any one of the three away and the defect returns:
//!
//! | Missing | Result |
//! |---------|--------|
//! | 1 | The loop polls a token nobody shares — always false. |
//! | 2 | Nothing ever sets the flag. |
//! | 3 | No await point; the drop never happens mid-generation. |
//!
//! # Panics are preserved
//!
//! A panicking handler is re-raised with [`std::panic::resume_unwind`] so hyper
//! still sees a panic rather than this layer converting it into a 500. Interposing
//! a task must not change what a completed — or a failing — request returns.
//!
//! # Contract
//!
//! `contracts/apr-serve-cancellation-v1.yaml`.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};

use crate::generate::CancelToken;

use super::ErrorResponse;

/// Per-request cancellation layer. See the module docs for why all three steps
/// are load-bearing.
pub(crate) async fn cancel_on_disconnect(mut request: Request<Body>, next: Next) -> Response {
    let token = CancelToken::new();
    request.extensions_mut().insert(token.clone());

    // (2) Lives in THIS future. Dropped when the response is finished *or* when
    // axum abandons the request because the client went away.
    let _disconnect_guard = token.cancel_on_drop();

    // (3) The handler outlives this future's drop, so its decode loop is still
    // running to see the flag the guard just set.
    let handle = tokio::spawn(async move { next.run(request).await });

    match handle.await {
        Ok(response) => response,
        Err(join_err) if join_err.is_panic() => std::panic::resume_unwind(join_err.into_panic()),
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("Request task did not complete: {join_err}"),
            }),
        )
            .into_response(),
    }
}

/// The token a handler should install on its generation config.
///
/// Handlers take `Extension<CancelToken>`; this exists for the paths that hold a
/// `Request` rather than running as an extractor-based handler, and for tests that
/// invoke a handler without going through [`cancel_on_disconnect`].
#[must_use]
pub fn request_cancel_token(request: &Request<Body>) -> CancelToken {
    request
        .extensions()
        .get::<CancelToken>()
        .cloned()
        .unwrap_or_else(CancelToken::never)
}

#[cfg(test)]
#[path = "tests/cancel_scope_2376.rs"]
mod cancel_scope_2376;
