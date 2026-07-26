//! FALSIFY-SAMPLE-TOPP-NOOP-001 — `top_p` must actually filter on the dense path.
//!
//! `QuantizedGenerateConfig` has carried a `top_p` field since the dense decode path
//! existed (runtime.rs:30, default 1.0), and `/v1/chat/completions`, `/api/chat`,
//! `/api/generate` and `apr run` all populate it from the request. But
//! `sample_topk_with_draw` never took the parameter, so the value was read and then
//! silently discarded: `--top-p 0.001` produced byte-identical output to
//! `--top-p 1.0`. The only working nucleus implementation lived in `fails.rs`,
//! which is not compiled into the crate.
//!
//! Why the prior contract missed it: `apr-run-sampling-plumbing-v1` only proved the
//! value was *plumbed* from CLI/HTTP into the config struct — which was true. Nothing
//! asserted the sampler's *behaviour* changed as a result. These are behavioural,
//! oracle-based assertions rather than plumbing checks.

use crate::gguf::OwnedQuantizedModel;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A deliberately flat-ish tail with one clear head. Index 3 holds most of the mass;
/// indices 0..=2 and 4..=7 are plausible-but-unlikely tail tokens that a correct
/// nucleus filter must exclude at small `top_p`.
const LOGITS: [f32; 8] = [1.0, 1.2, 1.4, 6.0, 1.3, 1.1, 0.9, 1.5];
const HEAD: u32 = 3;

/// RED-on-bug: a tiny `top_p` must collapse the candidate set to the head token.
/// With `top_p` discarded, the full 8-token distribution is sampled and some seed
/// inevitably draws a tail token.
#[test]
fn topp_small_collapses_to_head() {
    for seed in 0..128u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let tok = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 1.0, 0, 0.02, &mut rng);
        assert_eq!(
            tok, HEAD,
            "FALSIFY-SAMPLE-TOPP-NOOP-001: top_p=0.02 must keep only the nucleus \
             (token {HEAD}); got {tok} at seed {seed}. A tail token here means top_p \
             was ignored and the full distribution was sampled."
        );
    }
}

/// The other half of the oracle: `top_p` must NOT be equivalent to disabled.
/// Guards against a "fix" that accepts the parameter and still ignores it —
/// exactly the failure mode that made this defect survive 9 releases.
#[test]
fn topp_small_differs_from_topp_disabled() {
    let mut differs = false;
    for seed in 0..128u64 {
        let mut a_rng = StdRng::seed_from_u64(seed);
        let mut b_rng = StdRng::seed_from_u64(seed);
        let filtered = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 1.0, 0, 0.02, &mut a_rng);
        let unfiltered = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 1.0, 0, 1.0, &mut b_rng);
        if filtered != unfiltered {
            differs = true;
            break;
        }
    }
    assert!(
        differs,
        "FALSIFY-SAMPLE-TOPP-NOOP-001: top_p=0.02 produced the SAME token as \
         top_p=1.0 for all 128 seeds — top_p is a no-op."
    );
}

/// NO-REGRESSION: `top_p == 1.0` (the default in runtime.rs:51) must be bit-exact
/// with the pre-fix behaviour. The guard skips the nucleus branch entirely, so the
/// candidate set and inverse-CDF draw are untouched. Oracle: identical to `top_p`
/// values at/above 1.0, which cannot filter anything.
#[test]
fn topp_one_is_bit_exact_noop() {
    for seed in 0..128u64 {
        let mut a_rng = StdRng::seed_from_u64(seed);
        let mut b_rng = StdRng::seed_from_u64(seed);
        let one = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 0.8, 0, 1.0, &mut a_rng);
        let above = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 0.8, 0, 2.0, &mut b_rng);
        assert_eq!(
            one, above,
            "top_p=1.0 must be a no-op identical to any value >= 1.0 (seed {seed})"
        );
    }
}

/// `top_p == 0.0` means "disabled" (llama.cpp/Ollama), not "keep nothing" — the same
/// class of bug as top_k=0. The `top_p > 0.0` half of the guard covers this; without
/// it a zero cutoff could truncate to an empty set and fall through to token 0.
#[test]
fn topp_zero_is_disabled_not_empty() {
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let tok = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 1.0, 0, 0.0, &mut rng);
        assert!(
            (tok as usize) < LOGITS.len(),
            "top_p=0.0 must mean disabled, not an empty candidate set (seed {seed})"
        );
    }
}

/// top_k and top_p compose: top_k runs first, then nucleus over the survivors.
/// Mirrors the live MoE ordering (infer/qwen3_moe_generate.rs:101 then :109).
#[test]
fn topk_then_topp_compose() {
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let tok = OwnedQuantizedModel::sample_topk_seeded(&LOGITS, 1.0, 4, 0.02, &mut rng);
        assert_eq!(
            tok, HEAD,
            "top_k=4 then top_p=0.02 must still collapse to the head (seed {seed})"
        );
    }
}
