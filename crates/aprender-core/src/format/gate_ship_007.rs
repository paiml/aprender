// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-007 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-007 | No unwrap() in new code (enforced by .clippy.toml) |
// merge`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.1.0 PROPOSED
// (FALSIFY-GATE-SHIP-007 — wired in the same PR as this file lands).
//
// GATE-SHIP-007 is the *merge-time* hygiene gate for introduced `.unwrap()`
// calls. `.clippy.toml` lists `unwrap` under `disallowed-methods`, so
// `cargo clippy -- -D warnings` fails on any offending line. The tool-
// emitted lint produces an integer "offending-call count" — this gate
// says that integer MUST be zero in new code.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given a u32 count of offending `.unwrap()` calls, the verdict is `Pass`
// iff `count == 0` and `Fail` otherwise. The tool-level portion
// (actually running `cargo clippy -- -D warnings` on the new-code diff
// and parsing the lint output) is intentionally out of scope here; what
// this file proves is that the compound gate's *threshold shape*
// (zero-tolerance) cannot be silently relaxed to "≤ N" without breaking
// this test.
//
// Zero-tolerance shape: `u32` precludes negative inputs by construction.
// A boundary mutation `count == 0 → count < N` would still Fail this
// test for any N > 0 at count=1. Symmetric twin of GATE-SHIP-010
// (advisory count) at the same `0` floor.

/// Maximum tolerated count of newly-introduced `.unwrap()` calls in a
/// diff. Zero-tolerance: any `.unwrap()` in new code blocks merge.
///
/// Derivation: `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §6 row GATE-SHIP-007 + `.clippy.toml` `disallowed-methods` entry for
/// `unwrap`. Changing this value from 0 is a contract amendment that
/// requires a spec update, a contract bump, and a CLAUDE.md §
/// "unwrap() banned" policy change.
pub const AC_GATE_SHIP_007_MAX_TOLERATED_UNWRAP_COUNT: u32 = 0;

/// Binary verdict for FALSIFY-GATE-SHIP-007 / GATE-SHIP-007.
/// `Pass` iff `count == 0`. `Fail` otherwise (any non-zero count is a
/// merge-blocker per `.clippy.toml`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip007Verdict {
    /// No `.unwrap()` calls introduced in the diff. The
    /// `cargo clippy -- -D warnings` disallowed-methods rule passes.
    /// Merge gate is green.
    Pass,
    /// At least one `.unwrap()` call was introduced in the diff.
    /// Merge is blocked until every call is rewritten as `expect(…)`
    /// or `ok_or_else(|| ...)?`.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-007 /
/// GATE-SHIP-007: zero-tolerance `.unwrap()` count threshold.
///
/// Zero-tolerance rationale: `.clippy.toml` bans `.unwrap()` via the
/// `disallowed-methods` registry; a single call is a -D warnings lint
/// error and fails CI. The integer "offending-call count" emitted by
/// clippy must therefore be zero. Softening to "≤ N" would silently
/// allow drift one-PR-at-a-time.
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_007::{
///     verdict_from_unwrap_count, GateShip007Verdict,
/// };
///
/// // No .unwrap() in new code → Pass.
/// assert_eq!(
///     verdict_from_unwrap_count(0),
///     GateShip007Verdict::Pass
/// );
///
/// // Single .unwrap() → Fail (merge blocked).
/// assert_eq!(
///     verdict_from_unwrap_count(1),
///     GateShip007Verdict::Fail
/// );
/// ```
#[must_use]
pub const fn verdict_from_unwrap_count(count: u32) -> GateShip007Verdict {
    if count == AC_GATE_SHIP_007_MAX_TOLERATED_UNWRAP_COUNT {
        GateShip007Verdict::Pass
    } else {
        GateShip007Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-007 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_007_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-007 algorithm-level PARTIAL discharge: prove
    /// the zero-tolerance `.unwrap()` count threshold. Any edit that
    /// softens `== 0` to `<= N` for any positive N must break this
    /// test before a merge.
    #[test]
    fn falsify_gate_ship_007_unwrap_zero_tolerance() {
        // Section 1: zero-count boundary → Pass. The happy path: a
        // clean diff with no new `.unwrap()` calls.
        assert_eq!(
            verdict_from_unwrap_count(0),
            GateShip007Verdict::Pass,
            "count = 0 must Pass (clean diff)",
        );

        // Section 2: one-count adjacent boundary → Fail. The sharpest
        // counter-example: a single `.unwrap()` must Fail. A softening
        // from `== 0` to `<= 1` would flip this to Pass.
        assert_eq!(
            verdict_from_unwrap_count(1),
            GateShip007Verdict::Fail,
            "count = 1 must Fail (single .unwrap() blocks merge)",
        );

        // Section 3: clear Fail band — various multi-unwrap counts.
        // Catches any relaxation to a mid-range tolerance.
        for &n in &[2_u32, 10, 100, u32::MAX] {
            assert_eq!(
                verdict_from_unwrap_count(n),
                GateShip007Verdict::Fail,
                "count = {n} must Fail (zero-tolerance band)",
            );
        }

        // Section 4: monotonicity sweep 0..=256 — exactly one input
        // yields Pass (count = 0). This catches mutations that flip
        // the inequality direction (e.g., `!= 0` → `>= 0` would Pass
        // all counts).
        for n in 0_u32..=256 {
            let verdict = verdict_from_unwrap_count(n);
            let expected = if n == 0 {
                GateShip007Verdict::Pass
            } else {
                GateShip007Verdict::Fail
            };
            assert_eq!(
                verdict, expected,
                "count = {n} expected {expected:?}, got {verdict:?}",
            );
        }

        // Section 5: provenance pin — zero-tolerance is load-bearing
        // and lockstepped with `.clippy.toml` + CLAUDE.md. If the
        // tolerance is ever softened (e.g., migrating to a code-smell
        // budget), this constant and every consumer must move
        // together.
        assert_eq!(
            AC_GATE_SHIP_007_MAX_TOLERATED_UNWRAP_COUNT, 0,
            "max tolerated .unwrap() count is 0 \
             (spec §6 GATE-SHIP-007; .clippy.toml disallowed-methods)",
        );
    }
}
