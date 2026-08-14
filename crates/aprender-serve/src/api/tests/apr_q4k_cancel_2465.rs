//! Falsifiers for aprender#2465(1) — the APR Q4K CUDA scheduler had no
//! cancellation at all. This is aprender#2376(3) still live on the one decode
//! path the original fix skipped.
//!
//! Contract: `contracts/apr-serve-cancellation-v1.yaml`
//! (FALSIFY-SERVE-CANCEL-009/010/011).
//!
//! # What these assert, and what they refuse to assert
//!
//! Token counts and forward-pass counts — **observed work**. Never "the flag was
//! set": the shipped defect is exactly compatible with the flag being set and
//! nobody reading it, so a test shaped that way passes against the broken code.
//!
//! Each falsifier runs the **uncancelled control first**. Without it, a test can
//! pass because generation produced nothing at all, which is indistinguishable
//! from "cancellation worked".
//!
//! # Why these drive `q4k_decode` rather than `generate_q4k`
//!
//! `generate_q4k` needs a live `CudaExecutor` and a GPU-resident Q4K model, so it
//! cannot run in the default test job. `q4k_decode` **is** its decode loop — the
//! same control flow, with only the CUDA forward pass moved behind a closure —
//! and it is not feature-gated, so these run under `cargo test -p aprender-serve
//! --lib` with no `cuda` feature and no GPU. FALSIFY-SERVE-CANCEL-011 covers the
//! part that cannot be executed here: that the three `#[cfg(feature = "cuda")]`
//! submission sites hand the loop the request's live token.

use std::cell::RefCell;

use crate::api::apr_q4k_scheduler::q4k_decode;
use crate::generate::CancelToken;

/// Records every decode step the loop actually performed.
///
/// The step count is the falsifiable quantity: it is one GPU forward pass per
/// entry, i.e. the work an abandoned request was burning.
#[derive(Default)]
struct StepLog {
    positions: RefCell<Vec<usize>>,
}

impl StepLog {
    fn count(&self) -> usize {
        self.positions.borrow().len()
    }
}

/// A deterministic stand-in for `forward_token_apr_q4k` + sampling: emits a
/// strictly increasing token sequence so a cancelled run is comparable to the
/// uncancelled one token by token.
fn run(
    first_token: u32,
    prompt_len: usize,
    max_tokens: usize,
    eos_ids: &[u32],
    cancel: &CancelToken,
    log: &StepLog,
) -> Vec<u32> {
    q4k_decode(
        first_token,
        prompt_len,
        max_tokens,
        eos_ids,
        cancel,
        |token, position, _step| {
            log.positions.borrow_mut().push(position);
            Ok(token.wrapping_add(1))
        },
    )
    .expect("the fake decode step never fails")
}

// ---------------------------------------------------------------------------
// FALSIFY-SERVE-CANCEL-009 — the Q4K decode loop stops at the cancel point
// ---------------------------------------------------------------------------

/// Pre-fix behaviour: `AprQ4kRequest` had no `cancel` field and the loop's only
/// exit was EOS, so an abandoned `/v1/chat/completions`, `/v1/completions`,
/// `/generate`, `/api/chat` or `/api/generate` request ran the full `max_tokens`
/// on the GPU for a client that had already hung up.
#[test]
fn q4k_scheduler_decode_stops_at_the_cancel_point_not_max_tokens() {
    const PROMPT_LEN: usize = 5;
    const FIRST_TOKEN: u32 = 100;
    const MAX_TOKENS: usize = 64;
    const BUDGET: usize = 8;

    // Uncancelled control FIRST. If this does not run the full budget then the
    // cancelled assertion below is not measuring cancellation.
    let control_log = StepLog::default();
    let uncancelled = run(
        FIRST_TOKEN,
        PROMPT_LEN,
        MAX_TOKENS,
        &[],
        &CancelToken::never(),
        &control_log,
    );
    assert_eq!(
        uncancelled.len(),
        MAX_TOKENS,
        "control: with no cancellation the Q4K loop must emit its full {MAX_TOKENS}-token \
         budget (the prefill token plus {} decode steps)",
        MAX_TOKENS - 1
    );
    assert_eq!(
        control_log.count(),
        MAX_TOKENS - 1,
        "control: the uncancelled loop must perform one forward pass per decode step"
    );
    assert_eq!(
        *control_log.positions.borrow(),
        (PROMPT_LEN..PROMPT_LEN + MAX_TOKENS - 1).collect::<Vec<_>>(),
        "control: decode positions must continue contiguously from the end of the prompt"
    );

    // Cancelled: the token trips after BUDGET polls, and the loop polls once per
    // decode step, so it stops after exactly BUDGET steps.
    let token = CancelToken::with_budget(BUDGET);
    let cancelled_log = StepLog::default();
    let cancelled = run(
        FIRST_TOKEN,
        PROMPT_LEN,
        MAX_TOKENS,
        &[],
        &token,
        &cancelled_log,
    );

    assert_eq!(
        cancelled.len(),
        BUDGET + 1,
        "the Q4K loop must stop at the cancel point ({BUDGET} decode steps after the \
         prefill token), not run to max_tokens ({MAX_TOKENS}); it emitted {} tokens",
        cancelled.len()
    );
    assert_eq!(
        cancelled_log.count(),
        BUDGET,
        "the cancelled run must perform exactly {BUDGET} GPU forward passes; it \
         performed {}",
        cancelled_log.count()
    );
    assert_eq!(
        token.polls(),
        BUDGET + 1,
        "the loop must poll exactly once per decode step ({BUDGET} polls that returned \
         false, plus the one that returned true and broke the loop)"
    );
    assert_eq!(
        cancelled,
        uncancelled[..cancelled.len()].to_vec(),
        "a cancelled run must be a strict prefix of the uncancelled run: cancelling \
         stops work, it does not change the tokens already produced"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-SERVE-CANCEL-010 — the poll is at the TOP of the loop body
// ---------------------------------------------------------------------------

/// A request whose client is already gone must cost **zero** GPU forward passes.
///
/// This is what distinguishes a poll at the top of the loop body from a poll at
/// the bottom: the latter costs one wasted forward pass per cancelled request,
/// and on a 7B Q4K model that is not free.
#[test]
fn q4k_scheduler_decode_cancelled_before_start_does_no_forward_passes() {
    const PROMPT_LEN: usize = 3;
    const FIRST_TOKEN: u32 = 42;
    const MAX_TOKENS: usize = 64;

    // Control first: the same call with a live, uncancelled token does the work.
    let control_log = StepLog::default();
    let uncancelled = run(
        FIRST_TOKEN,
        PROMPT_LEN,
        MAX_TOKENS,
        &[],
        &CancelToken::new(),
        &control_log,
    );
    assert_eq!(
        control_log.count(),
        MAX_TOKENS - 1,
        "control: an uncancelled request must perform all {} forward passes",
        MAX_TOKENS - 1
    );
    assert_eq!(
        uncancelled.len(),
        MAX_TOKENS,
        "control: an uncancelled request must emit the full budget"
    );

    let token = CancelToken::new();
    token.cancel();
    let log = StepLog::default();
    let out = run(FIRST_TOKEN, PROMPT_LEN, MAX_TOKENS, &[], &token, &log);

    assert_eq!(
        log.count(),
        0,
        "an already-cancelled request must perform no forward passes at all; it \
         performed {}",
        log.count()
    );
    assert_eq!(
        out,
        vec![FIRST_TOKEN],
        "the response may still carry the token already sampled from the prefill \
         logits, and nothing more"
    );
}

// ---------------------------------------------------------------------------
// The refactor must not have changed the pre-existing EOS exit
// ---------------------------------------------------------------------------

/// ALB-109's configurable EOS still ends the loop, and the EOS token is still the
/// last token in the output. Guards the extraction of the loop into `q4k_decode`.
#[test]
fn q4k_scheduler_decode_still_stops_at_eos() {
    const PROMPT_LEN: usize = 2;
    const FIRST_TOKEN: u32 = 10;
    const MAX_TOKENS: usize = 64;
    // The fake step emits 11, 12, 13 …, so this is reached after 4 steps.
    const EOS: u32 = 14;

    let log = StepLog::default();
    let out = run(
        FIRST_TOKEN,
        PROMPT_LEN,
        MAX_TOKENS,
        &[EOS],
        &CancelToken::never(),
        &log,
    );

    assert_eq!(
        out,
        vec![10, 11, 12, 13, 14],
        "the loop must stop once EOS is produced, with EOS as the final token"
    );
    assert_eq!(
        log.count(),
        4,
        "reaching EOS from token 10 takes exactly 4 decode steps"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-SERVE-CANCEL-011 — every submission site hands over a LIVE token
// ---------------------------------------------------------------------------

/// The three `AprQ4kRequest` construction sites are all `#[cfg(feature = "cuda")]`,
/// so no test in the default job can execute them. Adding the `cancel` field makes
/// omitting it a compile error, but it does not stop a site from passing
/// `CancelToken::never()` — which is precisely the shipped defect, spelled
/// explicitly. This reads the sources and requires each one to forward the
/// request's own token.
///
/// The count is asserted per file so that a rename or a moved handler shows up as
/// a failure rather than as a search that quietly matched nothing.
#[test]
fn every_apr_q4k_submission_site_forwards_the_request_cancel_token() {
    // Each entry: (path relative to this crate, how many submissions it must have).
    const SITES: [(&str, usize); 3] = [
        ("src/api/cuda_chat_backend.rs", 1),
        ("src/api/gpu_completions_handler.rs", 1),
        ("src/api/batch.rs", 1),
    ];

    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

    for (rel, expected) in SITES {
        let path = crate_root.join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

        // Every struct-literal construction, i.e. `AprQ4kRequest {` — the `use`
        // import is `AprQ4kRequest;` and does not match.
        let bodies: Vec<&str> = src
            .split("AprQ4kRequest {")
            .skip(1)
            .map(|rest| {
                let end = rest.find("})").unwrap_or(rest.len());
                &rest[..end]
            })
            .collect();

        assert_eq!(
            bodies.len(),
            expected,
            "{rel} must construct AprQ4kRequest exactly {expected} time(s); found {}. \
             If a submission site moved, update this list — a search that matches \
             nothing must not pass as clean.",
            bodies.len()
        );

        for body in bodies {
            assert!(
                body.contains("cancel: cancel.clone()"),
                "{rel} submits an AprQ4kRequest without forwarding the request's \
                 CancelToken (aprender#2465(1)). The Q4K scheduler decodes on its own \
                 thread, so a dropped response future cannot reach it and there is no \
                 per-token send to fail — the token is the only thing that stops it. \
                 Offending literal:\n{body}"
            );
        }
    }
}
