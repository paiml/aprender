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
//! # What these drive (#4269 M2b)
//!
//! Since M2b a Q4K request decodes through the one engine: `generate_q4k` builds a
//! [`Session`](crate::session::Session) over [`AprQ4kForward`] and calls
//! `generate` with [`q4k_generate_config`]. These drive that same session, the
//! same `ArchForward` impl and the same config; only the per-token
//! [`Q4kStep`] is a counting fake instead of `CudaQ4kStep`, so they run under
//! `cargo test -p aprender-serve --lib` with no `cuda` feature and no GPU.
//! FALSIFY-SERVE-CANCEL-011 covers the part that cannot be executed here: that
//! the three `#[cfg(feature = "cuda")]` submission sites hand the session the
//! request's live token.
//!
//! The session emits no token before its first poll, so a request cancelled
//! before decode returns nothing (the pre-M2b loop returned the prefill's token
//! unpolled), and a request cancelled mid-decode returns one token per poll that
//! passed.

use std::cell::RefCell;
use std::rc::Rc;

use crate::api::apr_q4k_forward::{AprQ4kForward, AprQ4kSession, Q4kStep};
use crate::api::apr_q4k_scheduler::q4k_generate_config;
use crate::error::Result;
use crate::generate::CancelToken;

const VOCAB: usize = 1024;

/// A deterministic stand-in for `forward_token_apr_q4k`: the logits put all
/// their mass on `token + 1`, so a greedy run emits a strictly increasing
/// sequence and a cancelled run is comparable to the uncancelled one token by
/// token. Every step is recorded — one entry is one GPU forward pass, the work
/// an abandoned request was burning.
#[derive(Default)]
struct Counting {
    log: Rc<RefCell<Log>>,
    /// When set, every step returns these logits instead (the sampled tests).
    fixed: Option<Vec<f32>>,
}

/// What the step saw, read back after the session is gone.
#[derive(Default, Clone)]
struct Log {
    positions: Vec<usize>,
    resets: usize,
}

impl Q4kStep for Counting {
    fn reset(&mut self) {
        self.log.borrow_mut().resets += 1;
    }

    fn step(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        self.log.borrow_mut().positions.push(position);
        if let Some(fixed) = &self.fixed {
            return Ok(fixed.clone());
        }
        let mut logits = vec![0.0; VOCAB];
        logits[(token as usize + 1) % VOCAB] = 1.0;
        Ok(logits)
    }
}

/// One Q4K request as `generate_q4k` runs it: prompt `1..=prompt_len`, a fresh
/// session, the scheduler's config. Returns the generated tokens (what
/// `AprQ4kResponse::output_tokens` carries) and the step log.
fn run(
    prompt_len: usize,
    max_tokens: usize,
    temperature: f32,
    seed: u64,
    eos_ids: &[u32],
    cancel: &CancelToken,
    step: Counting,
) -> (Vec<u32>, Log) {
    let log = Rc::clone(&step.log);
    let prompt: Vec<u32> = (1..=prompt_len as u32).collect();
    let mut session = AprQ4kSession::new(AprQ4kForward::new(step, true, Some(4096)));
    let config = q4k_generate_config(max_tokens, temperature, seed, eos_ids, cancel);
    let turn = session
        .generate(&prompt, &config, &mut |_| true)
        .expect("the fake step never fails");
    let generated = turn.tokens[prompt.len()..].to_vec();
    let seen = log.borrow().clone();
    (generated, seen)
}

/// The decode forwards: every step after the prompt's prefill.
fn decode_steps(log: &Log, prompt_len: usize) -> Vec<usize> {
    log.positions[prompt_len..].to_vec()
}

// ---------------------------------------------------------------------------
// FALSIFY-SERVE-CANCEL-009 — the Q4K decode stops at the cancel point
// ---------------------------------------------------------------------------

/// Pre-fix behaviour: `AprQ4kRequest` had no `cancel` field and the loop's only
/// exit was EOS, so an abandoned `/v1/chat/completions`, `/v1/completions`,
/// `/generate`, `/api/chat` or `/api/generate` request ran the full `max_tokens`
/// on the GPU for a client that had already hung up.
#[test]
fn q4k_scheduler_decode_stops_at_the_cancel_point_not_max_tokens() {
    const PROMPT_LEN: usize = 5;
    const MAX_TOKENS: usize = 64;
    const BUDGET: usize = 8;

    // Uncancelled control FIRST. If this does not run the full budget then the
    // cancelled assertion below is not measuring cancellation.
    let (uncancelled, control) = run(
        PROMPT_LEN,
        MAX_TOKENS,
        0.0,
        42,
        &[],
        &CancelToken::never(),
        Counting::default(),
    );
    assert_eq!(
        uncancelled.len(),
        MAX_TOKENS,
        "control: with no cancellation the Q4K session must emit its full \
         {MAX_TOKENS}-token budget"
    );
    assert_eq!(
        control.positions,
        (0..PROMPT_LEN + MAX_TOKENS - 1).collect::<Vec<_>>(),
        "control: one forward per prompt token, then one per decode step at \
         positions contiguous from the end of the prompt"
    );

    // Cancelled: the token trips after BUDGET polls, and the session polls once
    // per generated token, so it emits exactly BUDGET.
    let token = CancelToken::with_budget(BUDGET);
    let (cancelled, log) = run(
        PROMPT_LEN,
        MAX_TOKENS,
        0.0,
        42,
        &[],
        &token,
        Counting::default(),
    );
    assert_eq!(
        cancelled.len(),
        BUDGET,
        "the Q4K session must stop at the cancel point ({BUDGET} tokens), not run to \
         max_tokens ({MAX_TOKENS}); it emitted {} tokens",
        cancelled.len()
    );
    assert_eq!(
        decode_steps(&log, PROMPT_LEN).len(),
        BUDGET,
        "the cancelled run must perform exactly {BUDGET} decode forward passes (one \
         after each emitted token); it performed {}",
        decode_steps(&log, PROMPT_LEN).len()
    );
    assert_eq!(
        token.polls(),
        BUDGET + 1,
        "the session must poll exactly once per generated token ({BUDGET} polls that \
         returned false, plus the one that returned true and broke the loop)"
    );
    assert_eq!(
        cancelled,
        uncancelled[..cancelled.len()].to_vec(),
        "a cancelled run must be a strict prefix of the uncancelled run: cancelling \
         stops work, it does not change the tokens already produced"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-SERVE-CANCEL-010 — a request already cancelled does no decode work
// ---------------------------------------------------------------------------

/// A request whose client is already gone must cost **zero** decode forward
/// passes: only the prompt's prefill, which ran before the first poll.
#[test]
fn q4k_scheduler_decode_cancelled_before_start_does_no_forward_passes() {
    const PROMPT_LEN: usize = 3;
    const MAX_TOKENS: usize = 64;

    // Control first: the same call with a live, uncancelled token does the work.
    let (uncancelled, control) = run(
        PROMPT_LEN,
        MAX_TOKENS,
        0.0,
        42,
        &[],
        &CancelToken::new(),
        Counting::default(),
    );
    assert_eq!(
        decode_steps(&control, PROMPT_LEN).len(),
        MAX_TOKENS - 1,
        "control: an uncancelled request must perform all {} decode forward passes",
        MAX_TOKENS - 1
    );
    assert_eq!(
        uncancelled.len(),
        MAX_TOKENS,
        "control: an uncancelled request must emit the full budget"
    );

    let token = CancelToken::new();
    token.cancel();
    let (out, log) = run(
        PROMPT_LEN,
        MAX_TOKENS,
        0.0,
        42,
        &[],
        &token,
        Counting::default(),
    );
    assert_eq!(
        decode_steps(&log, PROMPT_LEN),
        Vec::<usize>::new(),
        "an already-cancelled request must perform no decode forward passes at all"
    );
    assert_eq!(
        out,
        Vec::<u32>::new(),
        "an already-cancelled request emits nothing: the session polls before its \
         first token"
    );
}

// ---------------------------------------------------------------------------
// The port must not have changed the pre-existing exits and token choice
// ---------------------------------------------------------------------------

/// ALB-109's configurable EOS still ends the decode, and the EOS token is still
/// the last token in the output.
#[test]
fn q4k_scheduler_decode_still_stops_at_eos() {
    // Prompt 1..=5, so the greedy fake emits 6, 7, 8, ...
    let (out, log) = run(
        5,
        64,
        0.0,
        42,
        &[9],
        &CancelToken::never(),
        Counting::default(),
    );
    assert_eq!(
        out,
        vec![6, 7, 8, 9],
        "EOS must end the decode and be emitted"
    );
    assert_eq!(
        decode_steps(&log, 5).len(),
        3,
        "no forward pass after the EOS token"
    );
}

/// The Q4K path has always decoded greedily at `temperature <= 0.01`; the
/// session alone would sample there.
#[test]
fn q4k_temperature_at_or_below_the_greedy_threshold_decodes_greedily() {
    let (out, _) = run(
        5,
        8,
        0.01,
        7,
        &[],
        &CancelToken::never(),
        Counting::default(),
    );
    assert_eq!(out, (6..14).collect::<Vec<u32>>());
}

/// A fresh session per request starts from a reset cache and prefills from 0.
#[test]
fn q4k_request_prefills_from_a_reset_cache() {
    let (_, log) = run(
        4,
        2,
        0.0,
        42,
        &[],
        &CancelToken::never(),
        Counting::default(),
    );
    assert_eq!(log.resets, 1);
    assert_eq!(log.positions, vec![0, 1, 2, 3, 4]);
}

// ---------------------------------------------------------------------------
// #3786 — a sampled Q4K request draws from the request's seeded RNG
// ---------------------------------------------------------------------------

const LOGITS: [f32; 6] = [1.0, 0.9, 1.1, 0.95, 1.05, 0.85];

fn sampled(seed: u64) -> Vec<u32> {
    let step = Counting {
        fixed: Some(LOGITS.to_vec()),
        ..Counting::default()
    };
    run(3, 32, 1.0, seed, &[], &CancelToken::never(), step).0
}

/// The same seed reproduces the draws byte for byte: the wall-clock sampler the
/// Q4K path once had could not, whatever the request said.
#[test]
fn the_same_seed_reproduces_the_sampled_tokens() {
    assert_eq!(sampled(7), sampled(7));
}

/// A different seed changes them, so the seed is actually read.
#[test]
fn a_different_seed_changes_the_sampled_tokens() {
    assert_ne!(sampled(7), sampled(8));
}

/// It is a draw, not the argmax (index 2) in disguise.
#[test]
fn a_sampled_step_draws_off_the_argmax() {
    assert!(sampled(7).iter().any(|&t| t != 2));
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
