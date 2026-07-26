//! FALSIFY-SAMPLE-TOPK-ZERO-001 — `top_k = 0` must mean "disabled", not "no candidates".
//!
//! llama.cpp and Ollama both use `top_k = 0` to mean *disable top-k filtering*, and
//! aprender's Ollama-compat (`/api/chat`, `/api/generate`) and OpenAI-compat
//! (`/v1/chat/completions`) surfaces pass the value straight through to the dense
//! sampler. Before the fix, `sample_topk_with_draw` ran an unguarded
//! `indexed.truncate(top_k)`: with `top_k == 0` that empties the candidate vector,
//! the softmax/inverse-CDF loop has nothing to iterate, and the fallthrough
//! `probs.last().map_or(0, ..)` returns **token 0** — on every step, forever, so the
//! user sees a stream of `!!!!!!`.
//!
//! RED-on-bug / GREEN-on-fix: with `truncate(top_k)` unguarded these assertions fail
//! (every draw returns 0); with the `top_k > 0 && top_k < len` guard they pass.

use crate::gguf::OwnedQuantizedModel;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Logits whose unique argmax is index 5. Deliberately NOT index 0, so a sampler
/// that collapses to the fallthrough is distinguishable from one that is merely greedy.
const LOGITS: [f32; 8] = [-9.0, -8.0, -7.0, -6.0, -5.0, 10.0, -4.0, -3.0];
const ARGMAX: u32 = 5;

/// The exact defect: `top_k = 0` returned token 0 on every draw.
#[test]
fn topk_zero_is_disabled_not_empty() {
    // 10.0 vs the next-highest -3.0 is a ~13-logit gap; after temperature 0.7 the
    // softmax mass on index 5 is >0.999999, so ANY uniform draw must select it.
    // A single unlucky draw therefore cannot explain a failure here.
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let tok = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 0.7, 0, 1.0, &mut rng);
        assert_eq!(
            tok, ARGMAX,
            "FALSIFY-SAMPLE-TOPK-ZERO-001: top_k=0 (llama.cpp/Ollama 'disabled') \
             returned token {tok}, expected {ARGMAX}. Returning 0 means truncate(0) \
             emptied the candidate set and the inverse-CDF fell through to \
             probs.last().map_or(0, ..) — the '!!!!!!' garbage-output defect."
        );
    }
}

/// `top_k = 0` must be equivalent to "keep every candidate", i.e. the same
/// distribution as `top_k >= vocab`. Oracle comparison, not a point assertion:
/// this stays honest even if the argmax or the RNG changes.
#[test]
fn topk_zero_matches_topk_full_vocab() {
    for seed in 0..64u64 {
        let mut rng_zero = StdRng::seed_from_u64(seed);
        let mut rng_full = StdRng::seed_from_u64(seed);
        let a = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 1.0, 0, 1.0, &mut rng_zero);
        let b =
            OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 1.0, LOGITS.len(), 1.0, &mut rng_full);
        assert_eq!(
            a, b,
            "FALSIFY-SAMPLE-TOPK-ZERO-001: top_k=0 must behave as 'disabled' \
             (identical to top_k=vocab_len); got {a} vs {b} at seed {seed}"
        );
    }
}

/// `top_k` larger than the vocabulary must not panic or change behaviour —
/// the guard's `top_k < indexed.len()` arm has to be a no-op, not a truncation.
#[test]
fn topk_larger_than_vocab_is_safe() {
    let mut rng = StdRng::seed_from_u64(7);
    let tok = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 0.7, 9999, 1.0, &mut rng);
    assert_eq!(tok, ARGMAX, "top_k > vocab_len must clamp harmlessly");
}

/// Ordinary top-k still filters: with `top_k = 1` only the argmax is reachable.
/// Guards against a "fix" that simply disables truncation altogether.
#[test]
fn topk_one_is_still_greedy() {
    for seed in 0..16u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let tok = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 1.0, 1, 1.0, &mut rng);
        assert_eq!(tok, ARGMAX, "top_k=1 must always return the argmax");
    }
}
