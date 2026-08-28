//! PERF-041: the admission predicate of `contracts/batch-admission-v1.yaml`,
//! stated once, in one place, in a module that is **not** feature-gated.
//!
//! Two reasons it lives here rather than inside a scheduler:
//!
//! 1. `api::cuda_batch_scheduler` and `api::iteration_scheduler` are both
//!    declared `#[cfg(feature = "cuda")]` in `api/mod.rs`. Anything defined
//!    inside them is a dark target: `cargo test -p aprender-serve --lib` (what
//!    `workspace-test` runs) never compiles it. The comment above
//!    `pub mod apr_q4k_scheduler` in that same file records this exact lesson
//!    already — "Gating the whole module put the only cancellation-free decode
//!    loop in the crate outside every CI test job."
//! 2. `batch-admission-v1.yaml` F-BATCH-002 names the filter
//!    `cargo test -p aprender-serve --features cuda --lib batch_admission`, and
//!    its own `test_harness_status` records that the filter matched **zero**
//!    tests — a filter that selects nothing prints `test result: ok. 0 passed`
//!    and is indistinguishable from a pass. This module is what that filter
//!    selects. Being ungated, it is also selected without `--features cuda`.
//!
//! The soundness obligation this discharges, quoted from the contract:
//!
//! ```text
//! fast_path(batch) <=> (len(batch) == 1 and channel_empty)
//! ```

/// Is the single-request fast path admissible for this batch?
///
/// The fast path (`generate_single_request` → `generate_gpu_resident_streaming`)
/// replays a captured CUDA decode graph and measures ~138 tok/s. The batched
/// path (`batched_decode_step` → `forward_batched_to_token_ids`) launches every
/// kernel eagerly — `BATCHED_GRAPH` is opt-in and documented as "still 25%
/// slower than eager due to capture overhead" — and measures ~46 tok/s per slot.
/// Taking the batched path when nothing else is pending therefore costs a lone
/// client roughly 3x, which is what F-BATCH-004 forbids.
///
/// `force_batched` is the F-BATCH-004 mutation knob ("force the batched path
/// unconditionally; the c=1 case must turn RED"), exposed via
/// [`force_batched_path`] so the penalty can be *measured* rather than inferred.
/// It is false in every build that does not set the environment variable, so
/// production behaviour is unchanged.
#[must_use]
pub fn fast_path_eligible(batch_len: usize, channel_empty: bool, force_batched: bool) -> bool {
    !force_batched && batch_len == 1 && channel_empty
}

/// Environment name of the F-BATCH-004 mutation knob.
pub const FORCE_BATCHED_ENV: &str = "APR_FORCE_BATCHED_PATH";

/// Read the mutation knob once per process.
///
/// PERF-041 measures `serialization_index(c) = wall(c) / wall(1)` where the
/// denominator is taken on the fast path and every numerator on the batched
/// path. That makes the index a product of two independent factors:
///
/// ```text
/// serialization_index(c) = path_penalty * scaling_index(c)
///   path_penalty   = wall_batched(1) / wall_fast(1)
///   scaling_index  = wall(c)         / wall_batched(1)
/// ```
///
/// Only `scaling_index` is serialization. `wall_batched(1)` is the one term of
/// that decomposition no recorded run contains, and this knob is how it is
/// obtained: set `APR_FORCE_BATCHED_PATH=1` and re-run the c=1 band.
#[must_use]
pub fn force_batched_path() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| parse_force_flag(std::env::var(FORCE_BATCHED_ENV).ok().as_deref()))
}

/// The arming rule, separated from the environment so it can be tested.
///
/// Exactly the string `"1"` arms the knob. Anything else — unset, empty, `"0"`,
/// `"true"` — leaves production behaviour alone. Deliberately strict: a knob
/// that a stray `APR_FORCE_BATCHED_PATH=0` could arm would silently make every
/// c=1 measurement on this box a measurement of the penalty path.
#[must_use]
pub fn parse_force_flag(raw: Option<&str>) -> bool {
    raw == Some("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// KANI-BATCH-001's `cheaper_alternative`, written out: the predicate is
    /// boolean over two inputs, so the exhaustive table *is* the proof. No
    /// solver, no floats, no loops.
    #[test]
    fn batch_admission_fast_path_taken_exactly_when_idle() {
        // (batch_len_is_one, channel_empty) -> fast_path
        assert!(
            fast_path_eligible(1, true, false),
            "idle server, one request: the fast path MUST be reachable. \
             F-BATCH-002/004 — deleting it regresses c=1 by ~3x."
        );
        assert!(
            !fast_path_eligible(1, false, false),
            "one request but a peer is already pending: joining is the whole \
             point of admission (F-BATCH-003)."
        );
        assert!(
            !fast_path_eligible(2, true, false),
            "a formed batch is never the single-request path."
        );
        assert!(
            !fast_path_eligible(2, false, false),
            "a formed batch with more pending is never the single-request path."
        );
    }

    /// An empty batch is not a batch. Stated because `batch_len == 1` alone
    /// would let `0` through a `!= 1`-shaped rewrite without any test noticing.
    #[test]
    fn batch_admission_rejects_empty_batch() {
        assert!(!fast_path_eligible(0, true, false));
        assert!(!fast_path_eligible(0, false, false));
    }

    /// The F-BATCH-004 mutation, executed rather than described: forcing the
    /// batched path must make the fast path unreachable in **every** one of the
    /// four assignments, including the idle one that is otherwise the only
    /// `true`.
    #[test]
    fn batch_admission_force_batched_suppresses_every_case() {
        for batch_len in [0usize, 1, 2, 4] {
            for channel_empty in [true, false] {
                assert!(
                    !fast_path_eligible(batch_len, channel_empty, true),
                    "APR_FORCE_BATCHED_PATH must suppress the fast path at \
                     batch_len={batch_len}, channel_empty={channel_empty}"
                );
            }
        }
    }

    /// Only the exact string `"1"` arms the knob. The `"0"` and `"true"` rows
    /// are the ones that matter: either arming on them would turn every c=1
    /// band measured on a box that exports the variable into a measurement of
    /// the penalty path, reported as if it were the production fast path.
    #[test]
    fn batch_admission_only_literal_one_arms_the_knob() {
        assert!(parse_force_flag(Some("1")));
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("TRUE"),
            Some("yes"),
            Some("2"),
            Some("01"),
            Some(" 1"),
            Some("1 "),
        ] {
            assert!(
                !parse_force_flag(raw),
                "{raw:?} must NOT arm {FORCE_BATCHED_ENV}"
            );
        }
    }
}
