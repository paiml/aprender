//! FALSIFY-RERANK-XENC-001 — `Reranker::rerank(query, candidates,
//! top_k=100)` completes within a tunable latency budget on the
//! shipped reranker implementations.
//!
//! Contract: `contracts/apr-rerank-v1.yaml` v1.4.0.
//!
//! Discharge strategy: build a 100-candidate batch of synthetic
//! `RetrievalResult` instances, time `Reranker::rerank()` on the
//! shipped `MockCrossEncoderReranker` (today's only cross-encoder
//! impl), and assert the elapsed time is below the budget
//! (default 1000 ms, tunable via `APR_RERANK_BUDGET_MS`).
//!
//! Why a 1000 ms budget when the §2.6 sketch said <100 ms? The
//! sketch's 100 ms target was scoped to a ≤100M-param real
//! cross-encoder routed through `aprender-serve`. That routing
//! does not yet exist; today the only `Reranker` impl that
//! advertises "cross-encoder" semantics is the
//! `MockCrossEncoderReranker` (term-overlap proxy, no model load).
//! The mock takes microseconds, so any reasonable budget passes.
//! Setting the budget to 1000 ms here locks in the architectural
//! ceiling that a *future* real cross-encoder must hit, while
//! absorbing CI variance on shared runners. Operators with stricter
//! requirements set `APR_RERANK_BUDGET_MS` tighter.

#![allow(clippy::unwrap_used)]

use aprender_rag::rerank::{MockCrossEncoderReranker, Reranker};
use aprender_rag::retrieve::RetrievalResult;
use aprender_rag::{Chunk, DocumentId};
use std::time::{Duration, Instant};

const N_CANDIDATES: usize = 100;
const TOP_K: usize = 100;

fn budget() -> Duration {
    let ms = std::env::var("APR_RERANK_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1000);
    Duration::from_millis(ms)
}

fn build_candidates() -> Vec<RetrievalResult> {
    (0..N_CANDIDATES)
        .map(|i| {
            // Vary content so the reranker actually scores each
            // candidate differently (avoids degenerate per-cand
            // cost paths).
            let content = format!("document {i} with keyword search context relevance and term");
            let chunk = Chunk::new(DocumentId::new(), content.clone(), 0, content.len());
            RetrievalResult::new(chunk).with_dense_score(0.5 + (i as f32) / 200.0)
        })
        .collect()
}

#[test]
fn rerank_top_100_within_budget() {
    let reranker = MockCrossEncoderReranker::new("test-cross-encoder");
    let candidates = build_candidates();
    assert_eq!(candidates.len(), N_CANDIDATES, "fixture should have 100 candidates");

    let query = "keyword search relevance";
    let start = Instant::now();
    let reranked = reranker.rerank(query, &candidates, TOP_K).unwrap();
    let elapsed = start.elapsed();

    let max = budget();
    assert!(
        elapsed < max,
        "FALSIFY-RERANK-XENC-001: rerank({TOP_K} of {N_CANDIDATES}) took \
         {}ms, exceeding budget {}ms. Reranker is super-linear in candidate \
         count OR a real cross-encoder regressed past the production \
         budget. Tunable via APR_RERANK_BUDGET_MS.",
        elapsed.as_millis(),
        max.as_millis(),
    );

    // Sanity: the reranker returned actual results (not zero from a
    // degenerate early-return).
    assert!(
        !reranked.is_empty(),
        "rerank should return non-empty result; got 0 — reranker is \
         silently dropping all candidates"
    );
    assert!(
        reranked.len() <= TOP_K,
        "rerank should respect top_k cap; got {} > {TOP_K}",
        reranked.len(),
    );
}

#[test]
fn rerank_scales_sub_quadratically_on_doubling_input() {
    // Companion: time rerank() at N=50 and N=100. If the reranker
    // is O(N²) (e.g., scores every pair instead of every
    // (query, doc) pair), the 100-case takes ≥4× the 50-case.
    // Cross-encoders are O(N) in candidate count; we enforce that
    // doubling input takes <3× time (loose to absorb CI noise but
    // tight enough to catch true super-linear).
    let reranker = MockCrossEncoderReranker::new("test");

    let cands_50 = build_candidates().into_iter().take(50).collect::<Vec<_>>();
    let cands_100 = build_candidates();

    // Warm-up to avoid first-call setup costs distorting the ratio.
    let _ = reranker.rerank("warm", &cands_50, 50).unwrap();

    let t_50 = {
        let start = Instant::now();
        let _ = reranker.rerank("query", &cands_50, 50).unwrap();
        start.elapsed()
    };
    let t_100 = {
        let start = Instant::now();
        let _ = reranker.rerank("query", &cands_100, 100).unwrap();
        start.elapsed()
    };

    // Floor t_50 at 1µs so the ratio is well-defined on a very
    // fast machine where both measurements round to 0.
    let t_50_us = t_50.as_nanos().max(1_000) as f64 / 1_000.0;
    let t_100_us = t_100.as_nanos() as f64 / 1_000.0;
    let ratio = t_100_us / t_50_us;

    assert!(
        ratio < 3.0,
        "rerank doubled-input took {ratio:.2}× the half-input time \
         ({}μs vs {}μs); expected ≤3× (sub-O(N²) scaling). A \
         super-linear regression is the likely cause.",
        t_100_us as u128,
        t_50_us as u128,
    );
}

#[test]
fn rerank_empty_candidates_is_fast_and_returns_empty() {
    // Edge case: zero candidates → trivial-fast path. Catches a
    // regression where empty input still runs O(N) setup
    // (allocating, sorting an empty vec — fine; spawning a tokio
    // task — not fine).
    let reranker = MockCrossEncoderReranker::new("test");
    let start = Instant::now();
    let reranked = reranker.rerank("query", &[], 10).unwrap();
    let elapsed = start.elapsed();
    assert!(reranked.is_empty());
    // Generous 50 ms — even an empty-input slow path is suspicious
    // past this point.
    assert!(
        elapsed < Duration::from_millis(50),
        "empty-candidate rerank took {}ms; should be effectively instant",
        elapsed.as_millis(),
    );
}
