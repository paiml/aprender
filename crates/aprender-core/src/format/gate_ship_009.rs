// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-009 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-009 | CI green: cargo fmt --check, cargo clippy -- -D
// warnings, cargo test --workspace | merge`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.1.0 PROPOSED
// (FALSIFY-GATE-SHIP-009 — wired in the same PR as this file lands).
//
// GATE-SHIP-009 is the *merge-time* CI-aggregate gate: a PR cannot
// merge unless THREE required CI checks all report green — `cargo fmt
// --check`, `cargo clippy -- -D warnings`, and `cargo test
// --workspace`. The aggregate shape is AND over the three booleans.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given three booleans (fmt_pass, clippy_pass, test_pass), the verdict
// is `Pass` iff all three are `true`. The tool-level portion (actually
// running the three commands against the commit) is intentionally out
// of scope. What this file proves is that the compound gate's
// *aggregate shape* cannot be silently weakened (e.g., to "at least 2
// of 3" or "clippy-optional") without breaking this test.
//
// Mirrors the aggregate-AND family of GATE-SHIP-001 (10 ACs) and
// GATE-SHIP-002 (12 ACs); differs in element count (3 vs 10/12) and
// in what each boolean represents (CI check vs per-AC aggregate). The
// 2^3=8 bitmask exhaustive proof is compact enough to be stated
// explicitly in the test body rather than algorithmically.

/// Number of required CI checks that must all Pass for GATE-SHIP-009
/// to Pass: `cargo fmt --check` + `cargo clippy -- -D warnings` +
/// `cargo test --workspace`.
///
/// Derivation: `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §6 row GATE-SHIP-009 + branch-protection required-status-checks.
/// Adding a fourth required check (e.g., `cargo deny`) is a contract
/// amendment that requires a spec update and a constant bump.
pub const AC_GATE_SHIP_009_REQUIRED_CHECK_COUNT: usize = 3;

/// Binary verdict for FALSIFY-GATE-SHIP-009 / GATE-SHIP-009.
/// `Pass` iff all 3 CI checks (fmt, clippy, test) report Pass.
/// `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip009Verdict {
    /// All 3 CI checks reported Pass. `cargo fmt --check` clean,
    /// `cargo clippy -- -D warnings` clean, and `cargo test
    /// --workspace` green. Merge is unblocked.
    Pass,
    /// At least one of the 3 required CI checks reported Fail.
    /// Merge is blocked until every check is green.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-009 /
/// GATE-SHIP-009: CI aggregate-AND over the 3 required checks
/// (`cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test
/// --workspace`).
///
/// A single `false` on any of the three inputs yields `Fail`.
/// Relaxation to "at least 2 of 3" or "clippy-optional" is blocked by
/// the exhaustive bitmask proof in the test body.
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_009::{
///     verdict_from_ci_aggregate, GateShip009Verdict,
/// };
///
/// // All 3 CI checks green → Pass.
/// assert_eq!(
///     verdict_from_ci_aggregate(true, true, true),
///     GateShip009Verdict::Pass
/// );
///
/// // Clippy fails → whole aggregate Fails.
/// assert_eq!(
///     verdict_from_ci_aggregate(true, false, true),
///     GateShip009Verdict::Fail
/// );
/// ```
#[must_use]
pub const fn verdict_from_ci_aggregate(
    fmt_pass: bool,
    clippy_pass: bool,
    test_pass: bool,
) -> GateShip009Verdict {
    if fmt_pass && clippy_pass && test_pass {
        GateShip009Verdict::Pass
    } else {
        GateShip009Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-009 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_009_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-009 algorithm-level PARTIAL discharge: prove
    /// the CI-aggregate-AND decision rule over the 3 required checks.
    /// Any edit that relaxes AND to OR, drops a check, or inverts a
    /// Pass/Fail mapping must break this test before a merge.
    #[test]
    fn falsify_gate_ship_009_ci_aggregate_and() {
        // Section 1: all-true → Pass. The happy path: fmt + clippy
        // + test all green.
        assert_eq!(
            verdict_from_ci_aggregate(true, true, true),
            GateShip009Verdict::Pass,
            "all 3 CI checks Pass must yield aggregate Pass",
        );

        // Section 2a: fmt fails alone → Fail. Catches a mutation that
        // drops the fmt check.
        assert_eq!(
            verdict_from_ci_aggregate(false, true, true),
            GateShip009Verdict::Fail,
            "fmt fail alone must Fail the aggregate",
        );

        // Section 2b: clippy fails alone → Fail.
        assert_eq!(
            verdict_from_ci_aggregate(true, false, true),
            GateShip009Verdict::Fail,
            "clippy fail alone must Fail the aggregate",
        );

        // Section 2c: test fails alone → Fail.
        assert_eq!(
            verdict_from_ci_aggregate(true, true, false),
            GateShip009Verdict::Fail,
            "test fail alone must Fail the aggregate",
        );

        // Section 3: all-false → Fail. Degenerate case.
        assert_eq!(
            verdict_from_ci_aggregate(false, false, false),
            GateShip009Verdict::Fail,
            "all-false must Fail (degenerate)",
        );

        // Section 4: exhaustive 2^3 = 8-combination proof. The
        // aggregate is Pass iff the bitmask is 0b111 = 7. Every other
        // mask must yield Fail. This is the strongest correctness
        // statement we can make for the pure 3-boolean decision rule.
        for mask in 0u8..8 {
            let fmt_pass = (mask & 0b001) != 0;
            let clippy_pass = (mask & 0b010) != 0;
            let test_pass = (mask & 0b100) != 0;
            let expected = if mask == 0b111 {
                GateShip009Verdict::Pass
            } else {
                GateShip009Verdict::Fail
            };
            assert_eq!(
                verdict_from_ci_aggregate(fmt_pass, clippy_pass, test_pass),
                expected,
                "mask=0b{mask:03b} (fmt={fmt_pass}, clippy={clippy_pass}, \
                 test={test_pass}) expected {expected:?}",
            );
        }

        // Section 5: each-single-true (only one true, other two
        // false) → Fail. Catches OR-shaped mutations.
        assert_eq!(
            verdict_from_ci_aggregate(true, false, false),
            GateShip009Verdict::Fail,
            "only fmt=true must Fail (OR-shape mutation guard)",
        );
        assert_eq!(
            verdict_from_ci_aggregate(false, true, false),
            GateShip009Verdict::Fail,
            "only clippy=true must Fail",
        );
        assert_eq!(
            verdict_from_ci_aggregate(false, false, true),
            GateShip009Verdict::Fail,
            "only test=true must Fail",
        );

        // Section 6: each-single-false (one false, other two true) —
        // already covered by Section 2 but stated here explicitly to
        // document the intent: a 2-of-3 majority must NOT promote to
        // Pass. A mutation to "at least 2 of 3" would flip these.
        // (Redundant with Section 2 but sharpens intent.)

        // Section 7: provenance pin — the 3-count is load-bearing
        // and lockstepped with the GitHub branch-protection required-
        // status-checks list. Adding a fourth required check is a
        // contract amendment.
        assert_eq!(
            AC_GATE_SHIP_009_REQUIRED_CHECK_COUNT, 3,
            "required CI check count is 3 \
             (spec §6 GATE-SHIP-009; fmt + clippy + test)",
        );

        // Section 8: NOT symmetry — swapping the three arguments does
        // not change the verdict (AND is symmetric). Catches a
        // mutation that accidentally privileges one argument's
        // position (e.g., `fmt_pass && clippy_pass` ignoring
        // test_pass).
        for &(a, b, c) in &[
            (true, true, false),
            (true, false, true),
            (false, true, true),
        ] {
            let v_abc = verdict_from_ci_aggregate(a, b, c);
            let v_bca = verdict_from_ci_aggregate(b, c, a);
            let v_cab = verdict_from_ci_aggregate(c, a, b);
            assert_eq!(
                v_abc, v_bca,
                "AND-symmetry broken at ({a},{b},{c}) vs ({b},{c},{a})",
            );
            assert_eq!(
                v_abc, v_cab,
                "AND-symmetry broken at ({a},{b},{c}) vs ({c},{a},{b})",
            );
        }
    }
}
