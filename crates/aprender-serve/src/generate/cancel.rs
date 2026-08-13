//! Cooperative cancellation for autoregressive generation loops.
//!
//! aprender#2376(3): `apr serve` had **no cancellation machinery at all**. Every
//! decode loop in this crate ran `for _ in 0..max_tokens` with no exit other than
//! EOS, and every HTTP handler called that loop *synchronously* from inside its
//! `async fn`. A synchronous call has no `.await` inside it, so the task never
//! yields while generating — which means axum dropping the response future when the
//! client goes away cannot interrupt it. One abandoned request therefore kept a core
//! pinned at ~250% CPU for the remaining life of the process, with zero open
//! connections.
//!
//! Dropping a future only cancels work that is *suspended at an await point*. The
//! fix is therefore two-sided and both sides are required:
//!
//! 1. the decode loop must **poll** a cancellation signal once per token
//!    ([`CancelToken::is_cancelled`]), and
//! 2. the handler must run that loop off-task (`spawn_blocking`) while holding a
//!    [`CancelOnDrop`] guard, so the drop that axum performs on disconnect actually
//!    reaches the loop.
//!
//! Doing only (1) leaves nothing to set the flag. Doing only (2) leaves the flag
//! set with nobody reading it. See `crates/aprender-serve/src/api/cancel_scope.rs`
//! for the handler half.
//!
//! # Cost when nobody cancels
//!
//! [`CancelToken::default`] (== [`CancelToken::never`]) holds `None`, so
//! `is_cancelled()` is a null check with no atomic and no allocation. Every existing
//! caller that constructs a config with `..Default::default()` keeps that.
//!
//! # Contract
//!
//! `contracts/apr-serve-cancellation-v1.yaml`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Shared state behind a live token.
#[derive(Debug)]
struct Inner {
    /// Set by [`CancelToken::cancel`] — e.g. from [`CancelOnDrop`] when axum drops
    /// the response future.
    cancelled: AtomicBool,
    /// How many times [`CancelToken::is_cancelled`] has been called. This is the
    /// loop's *observed* poll count, which is what lets a falsifier assert the loop
    /// polls once per token rather than merely trusting that it polls at all.
    polls: AtomicUsize,
    /// Poll index at which the token starts reporting cancelled regardless of
    /// `cancelled` — i.e. a deterministic work budget of `budget` further polls.
    /// [`usize::MAX`] means "no budget", which is the normal production token.
    budget: usize,
}

/// A cooperative cancellation signal read by generation loops.
///
/// Cloning is cheap (`Arc` bump) and all clones observe the same state.
#[derive(Clone, Debug, Default)]
pub struct CancelToken {
    inner: Option<Arc<Inner>>,
}

impl CancelToken {
    /// A token that can be cancelled.
    #[must_use]
    pub fn new() -> Self {
        Self::with_budget(usize::MAX)
    }

    /// A token that is never cancelled, allocates nothing, and costs a null check
    /// per poll. This is the default so that adding a `cancel` field to a config
    /// struct changes no existing behaviour.
    #[must_use]
    pub fn never() -> Self {
        Self { inner: None }
    }

    /// A token that reports cancelled once it has been polled `budget` times —
    /// a deterministic work budget of `budget` decode steps.
    ///
    /// This is how the falsifiers cancel *at a known token index* without a sleep,
    /// a second thread, or any other source of flake: the loop polls once per token,
    /// so a budget of `n` stops generation after exactly `n` tokens.
    #[must_use]
    pub fn with_budget(budget: usize) -> Self {
        Self {
            inner: Some(Arc::new(Inner {
                cancelled: AtomicBool::new(false),
                polls: AtomicUsize::new(0),
                budget,
            })),
        }
    }

    /// Request cancellation. Idempotent, and callable from any thread.
    pub fn cancel(&self) {
        if let Some(inner) = &self.inner {
            inner.cancelled.store(true, Ordering::Release);
        }
    }

    /// Poll the signal. **Counts as one poll** — see [`CancelToken::polls`].
    ///
    /// Generation loops call this once per decode step and `break` when it is true.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        match &self.inner {
            None => false,
            Some(inner) => {
                let seen = inner.polls.fetch_add(1, Ordering::Relaxed);
                inner.cancelled.load(Ordering::Acquire) || seen >= inner.budget
            },
        }
    }

    /// Read the cancelled flag **without** counting a poll or spending budget.
    ///
    /// For assertions and for reporting; loops must use [`CancelToken::is_cancelled`].
    #[must_use]
    pub fn peek_cancelled(&self) -> bool {
        match &self.inner {
            None => false,
            Some(inner) => inner.cancelled.load(Ordering::Acquire),
        }
    }

    /// Number of times [`CancelToken::is_cancelled`] has been called.
    ///
    /// A [`CancelToken::never`] token reports 0 because it has no state; that is the
    /// point — it is the zero-cost variant.
    #[must_use]
    pub fn polls(&self) -> usize {
        match &self.inner {
            None => 0,
            Some(inner) => inner.polls.load(Ordering::Relaxed),
        }
    }

    /// Arm a guard that cancels this token when it is dropped.
    ///
    /// Hold the guard inside the async response future. Axum drops that future when
    /// the client disconnects, which drops the guard, which sets the flag, which the
    /// decode loop observes at its next token boundary.
    #[must_use]
    pub fn cancel_on_drop(&self) -> CancelOnDrop {
        CancelOnDrop {
            token: self.clone(),
            armed: true,
        }
    }
}

/// Cancels its [`CancelToken`] on drop, unless [`CancelOnDrop::disarm`] ran first.
///
/// # Why it has to be disarmable
///
/// The original version fired unconditionally, on the reasoning that "firing after
/// generation already finished is harmless — the loop is gone and the flag is read
/// by nobody". That is true for a handler that generates *inside* its own future,
/// and false for every streaming handler: those hand the decode loop to a
/// background task and RETURN the SSE response immediately, so the middleware
/// future completes while generation is still starting. An unconditional guard
/// therefore cancelled the loop before its first token, and
/// `POST /v1/chat/completions` with `"stream":true` answered
/// `text/event-stream` carrying an opening chunk, a terminal chunk and **no
/// content deltas at all**.
///
/// So the guard now distinguishes the two exits it always had:
/// - dropped while the future is still running (the client went away) → cancel;
/// - disarmed by a future that ran to completion → do nothing.
///
/// A streaming request abandoned mid-body is still cancelled, by a different and
/// pre-existing mechanism: hyper drops the response body, which drops the SSE
/// receiver, which makes the generator's `on_token` send fail, which breaks the
/// decode loop.
#[derive(Debug)]
pub struct CancelOnDrop {
    token: CancelToken,
    armed: bool,
}

impl CancelOnDrop {
    /// The token this guard will cancel.
    #[must_use]
    pub fn token(&self) -> &CancelToken {
        &self.token
    }

    /// Give up the right to cancel: this guard's drop becomes a no-op.
    ///
    /// Call it on the path where the work this guard protects has *completed* or
    /// been handed to something else that can stop it. Anything else (an early
    /// return, a panic, a dropped future) leaves the guard armed.
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.token.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_token_is_free_and_never_cancels() {
        let t = CancelToken::never();
        t.cancel();
        assert!(!t.is_cancelled(), "never() must not be cancellable");
        assert!(!t.peek_cancelled());
        assert_eq!(t.polls(), 0, "never() must hold no state");
    }

    #[test]
    fn default_is_never() {
        let t = CancelToken::default();
        t.cancel();
        assert!(
            !t.is_cancelled(),
            "Default must be the zero-cost never() token so adding a cancel field \
             changes no existing caller's behaviour"
        );
    }

    #[test]
    fn cancel_is_observed_by_clones() {
        let t = CancelToken::new();
        let clone = t.clone();
        assert!(!clone.peek_cancelled());
        t.cancel();
        assert!(
            clone.peek_cancelled(),
            "clones must share state or a spawn_blocking worker cannot see the guard fire"
        );
    }

    #[test]
    fn polls_counts_only_is_cancelled() {
        let t = CancelToken::new();
        assert_eq!(t.polls(), 0);
        let _ = t.is_cancelled();
        let _ = t.is_cancelled();
        assert_eq!(t.polls(), 2);
        let _ = t.peek_cancelled();
        assert_eq!(t.polls(), 2, "peek must not consume budget");
    }

    #[test]
    fn budget_trips_after_exactly_n_polls() {
        let t = CancelToken::with_budget(3);
        assert!(!t.is_cancelled(), "poll 0 within budget");
        assert!(!t.is_cancelled(), "poll 1 within budget");
        assert!(!t.is_cancelled(), "poll 2 within budget");
        assert!(t.is_cancelled(), "poll 3 exhausts a budget of 3");
        assert!(t.is_cancelled(), "and stays cancelled");
    }

    #[test]
    fn zero_budget_trips_immediately() {
        let t = CancelToken::with_budget(0);
        assert!(t.is_cancelled(), "a budget of 0 permits no work at all");
    }

    #[test]
    fn new_token_has_no_budget() {
        let t = CancelToken::new();
        for i in 0..10_000 {
            assert!(
                !t.is_cancelled(),
                "poll {i} must not trip an unbudgeted token"
            );
        }
    }

    #[test]
    fn guard_cancels_token_on_drop() {
        let t = CancelToken::new();
        {
            let _guard = t.cancel_on_drop();
            assert!(
                !t.peek_cancelled(),
                "the guard must not cancel while it is still alive"
            );
        }
        assert!(
            t.peek_cancelled(),
            "dropping the guard is what turns an axum client-disconnect into a stop signal"
        );
    }

    #[test]
    fn guard_exposes_the_token_it_will_cancel() {
        let t = CancelToken::new();
        let guard = t.cancel_on_drop();
        assert!(!guard.token().peek_cancelled());
        drop(guard);
        assert!(t.peek_cancelled());
    }

    /// aprender#2375(1): a disarmed guard must NOT cancel. The streaming handlers
    /// return their response while the decode loop is still starting, so a guard
    /// that fired on normal completion killed the generation before its first
    /// token and the SSE body carried no content deltas.
    #[test]
    fn disarmed_guard_does_not_cancel_on_drop() {
        let t = CancelToken::new();
        {
            let mut guard = t.cancel_on_drop();
            guard.disarm();
        }
        assert!(
            !t.peek_cancelled(),
            "a disarmed guard must leave the token alive: work handed to a \
             background task outlives the future that armed the guard"
        );
    }

    /// The guard is armed by default, so forgetting to disarm keeps the
    /// disconnect behaviour rather than silently losing it.
    #[test]
    fn guard_is_armed_until_disarmed() {
        let t = CancelToken::new();
        let mut guard = t.cancel_on_drop();
        drop(t.cancel_on_drop()); // a second, still-armed guard cancels
        assert!(t.peek_cancelled());
        guard.disarm();
    }
}
