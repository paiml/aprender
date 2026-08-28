//! PERF-034 — the production dense decode sampler must be BIT-IDENTICAL after the
//! full-vocabulary sort was replaced by an O(V) partial selection.
//!
//! `OwnedQuantizedModel::sample_topk_with_draw` is the sampler behind `apr run`,
//! `/v1/chat/completions`, `/api/chat` and `/api/generate` on every dense GGUF model.
//! It used to build a fresh `Vec<(usize, f32)>` over the whole vocabulary and
//! **full-sort 152,064 entries per token** to keep `top_k = 40`. It now selects with
//! `select_nth_unstable_by` into a reusable per-thread buffer.
//!
//! That is a performance change that must have **zero** behavioural surface, so this
//! module pins the pre-PERF-034 implementation verbatim as `legacy_sample_topk_with_draw`
//! and asserts the shipping sampler agrees with it token-for-token, over multi-step
//! autoregressive streams where any single divergence compounds.
//!
//! RED-on-regression: swap `retain_top_k_sorted`'s comparator for a plain
//! `partial_cmp` (dropping the index tiebreak) and the tie-heavy cases below fail —
//! that tiebreak is the whole reason an *unstable* selection can stand in for a
//! *stable* sort.

use crate::gguf::OwnedQuantizedModel;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// The dense sampler exactly as it shipped before PERF-034.
///
/// Copied verbatim from `generate_quantized.rs` at `62d23d8d1`. Do not "clean this
/// up" — its value is that it is the old code, not that it is good code.
fn legacy_sample_topk_with_draw(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    r: f32,
) -> u32 {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    if top_k > 0 && top_k < indexed.len() {
        indexed.truncate(top_k);
    }

    if top_p > 0.0 && top_p < 1.0 {
        let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
        let exp_vals: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_val).exp()).collect();
        let total: f32 = exp_vals.iter().sum();
        if total > 0.0 {
            let mut cumulative = 0.0;
            let mut cutoff = indexed.len();
            for (i, &ev) in exp_vals.iter().enumerate() {
                cumulative += ev / total;
                if cumulative >= top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }
    }

    let max_val = indexed.first().map_or(0.0, |(_, v)| *v);
    let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_val).exp()).sum();
    let probs: Vec<(usize, f32)> = indexed
        .iter()
        .map(|(i, v)| (*i, (v - max_val).exp() / exp_sum))
        .collect();

    let mut cumulative = 0.0;
    for &(idx, prob) in &probs {
        cumulative += prob;
        if cumulative >= r {
            return idx as u32;
        }
    }

    probs.last().map_or(0, |(idx, _)| *idx as u32)
}

/// Reproducible pseudo-logits. An LCG, so the corpus is identical on every host and
/// every run — a sampling equivalence test must not itself be a source of variance.
fn pseudo_logits(vocab: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..vocab)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f32 / 4_294_967_296.0f32).mul_add(24.0, -12.0)
        })
        .collect()
}

/// Drive an autoregressive stream through both samplers and demand identical tokens.
///
/// Two `StdRng`s seeded identically stay in lockstep because both paths consume
/// exactly one `f32` draw per step — so this also proves the seeded public entry
/// point (`sample_topk_seeded`, the OpenAI `seed` contract) is unchanged, not just
/// the internal helper.
///
/// The sampled token is folded back into the next step's logits, so a single
/// divergence at step `i` shows up as a different token at every later step too.
fn assert_streams_agree(vocab: usize, steps: usize, temperature: f32, top_k: usize, top_p: f32) {
    for seed in 0..4u64 {
        let mut rng_legacy = StdRng::seed_from_u64(seed);
        let mut rng_new = StdRng::seed_from_u64(seed);
        let mut legacy_stream = Vec::with_capacity(steps);
        let mut new_stream = Vec::with_capacity(steps);
        let mut feedback = seed;

        for step in 0..steps {
            let logits = pseudo_logits(vocab, feedback ^ (step as u64));

            let r: f32 = rng_legacy.random();
            let legacy = legacy_sample_topk_with_draw(&logits, temperature, top_k, top_p, r);
            let fresh = OwnedQuantizedModel::sample_topk_seeded(
                &logits,
                temperature,
                top_k,
                top_p,
                &mut rng_new,
            );

            assert_eq!(
                legacy, fresh,
                "diverged at step {step} (seed={seed}, vocab={vocab}, T={temperature}, \
                 top_k={top_k}, top_p={top_p})"
            );
            legacy_stream.push(legacy);
            new_stream.push(fresh);
            feedback = u64::from(fresh).wrapping_add(1);
        }

        assert_eq!(legacy_stream, new_stream, "stream mismatch at seed {seed}");
        // Guard the guard: a stream of one repeated token would make the comparison
        // above pass trivially without exercising the selection at all.
        assert!(
            new_stream
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                > 1,
            "degenerate stream {new_stream:?} — this test would pass even if broken"
        );
    }
}

#[test]
fn perf034_identical_stream_default_sampling() {
    // The shipped defaults: top_k = 40, top_p = 1.0 (nucleus branch skipped).
    assert_streams_agree(4096, 24, 0.7, 40, 1.0);
}

#[test]
fn perf034_identical_stream_with_nucleus() {
    assert_streams_agree(4096, 24, 0.8, 40, 0.9);
}

#[test]
fn perf034_identical_stream_top_k_disabled() {
    // `top_k = 0` means "disabled" — the whole vocabulary stays as candidates, so
    // this is the path that still needs a full sort. It must not have changed either.
    assert_streams_agree(2048, 16, 1.0, 0, 0.95);
}

#[test]
fn perf034_identical_stream_top_k_exceeds_vocab() {
    assert_streams_agree(512, 16, 1.0, 4096, 1.0);
}

#[test]
fn perf034_identical_stream_low_temperature() {
    // Near-greedy: the CDF walk terminates on the first candidate, so the *order* of
    // the top-k is what decides the token. Most sensitive case to a tie-order change.
    assert_streams_agree(4096, 24, 0.05, 40, 1.0);
}

#[test]
fn perf034_identical_at_production_vocabulary() {
    // Qwen2.5's real vocabulary. Fewer steps because the LEGACY reference full-sorts
    // 152,064 entries per call and this runs in a debug build.
    assert_streams_agree(152_064, 4, 0.7, 40, 1.0);
}

#[test]
fn perf034_identical_when_logits_are_heavily_tied() {
    // A repetition penalty or a grammar mask leaves a large block of equal logits.
    // The stable sort broke those ties by index; an unstable selection only
    // reproduces that because the comparator now says so explicitly. `top_k = 97/98`
    // deliberately cuts *inside* the tied block.
    //
    // The draw `r` is swept by varying the RNG seed rather than passed directly:
    // `sample_topk_with_draw` is private, so the only honest comparison is through
    // the public seeded entry point, with the reference fed the same draw.
    let mut logits = vec![f32::NEG_INFINITY; 3000];
    for (i, slot) in logits.iter_mut().enumerate().step_by(31) {
        *slot = ((i % 7) as f32) * 0.5;
    }
    for top_k in [1usize, 5, 40, 97, 98, 500, 3000] {
        for seed in 0..24u64 {
            let mut rng_legacy = StdRng::seed_from_u64(seed);
            let mut rng_new = StdRng::seed_from_u64(seed);
            let r: f32 = rng_legacy.random();
            let legacy = legacy_sample_topk_with_draw(&logits, 1.0, top_k, 1.0, r);
            let fresh =
                OwnedQuantizedModel::sample_topk_seeded(&logits, 1.0, top_k, 1.0, &mut rng_new);
            assert_eq!(legacy, fresh, "top_k={top_k} seed={seed} r={r}");
        }
    }
}

#[test]
fn perf034_identical_on_all_equal_logits() {
    // Every candidate tied: the token is decided *entirely* by tie order.
    let logits = vec![0.25f32; 1024];
    for top_k in [1usize, 2, 40, 1024, 2048] {
        let mut rng_a = StdRng::seed_from_u64(7);
        let mut rng_b = StdRng::seed_from_u64(7);
        for _ in 0..8 {
            let r: f32 = rng_a.random();
            let legacy = legacy_sample_topk_with_draw(&logits, 1.0, top_k, 1.0, r);
            let fresh =
                OwnedQuantizedModel::sample_topk_seeded(&logits, 1.0, top_k, 1.0, &mut rng_b);
            assert_eq!(legacy, fresh, "top_k={top_k}");
        }
    }
}

#[test]
fn perf034_scratch_reuse_does_not_leak_state_between_calls() {
    // The candidate buffer is now thread-local and reused. A call with a small
    // vocabulary following a call with a large one must not see stale entries.
    let big = pseudo_logits(8192, 11);
    let small = pseudo_logits(64, 11);

    let mut rng = StdRng::seed_from_u64(3);
    let _ = OwnedQuantizedModel::sample_topk_seeded(&big, 0.7, 40, 1.0, &mut rng);

    let mut rng_after = StdRng::seed_from_u64(99);
    let after = OwnedQuantizedModel::sample_topk_seeded(&small, 0.7, 40, 1.0, &mut rng_after);

    let mut rng_clean = StdRng::seed_from_u64(99);
    let clean = OwnedQuantizedModel::sample_topk_seeded(&small, 0.7, 40, 1.0, &mut rng_clean);

    assert_eq!(after, clean, "a prior large-vocab call changed the result");
    assert!(
        (after as usize) < small.len(),
        "returned token {after} is outside the 64-entry vocabulary — stale scratch"
    );
}
