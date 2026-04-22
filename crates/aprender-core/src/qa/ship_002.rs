// SHIP-TWO-001 AC-SHIP1-002 / FALSIFY-SHIP-002 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md
// Contract: contracts/qwen2-e2e-verification-v1.yaml (GATE-QW2E-SHIP-002 —
// wired in the same PR as this file lands).
//
// AC-SHIP1-002 states that the MODEL-1 teacher
// (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) must emit syntactically
// valid Python on the canonical prompt `def fib(n):` via
// `apr run <model>.safetensors`. The falsification test
// FALSIFY-SHIP-002 parses the emitted completion with
// `rustpython`/`ruff` and flags any parse error as a ship-blocker.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given a count of syntax errors observed on the canonical prompt, the
// verdict is `Pass` iff `syntax_errors ≤
// AC_SHIP1_002_MAX_TOLERATED_SYNTAX_ERRORS` (= 0). Because the spec
// text is strict — "emits valid Python" with no tolerance allowance —
// any non-zero error count on the single canonical prompt is a Fail.
// The compute-heavy portion of the AC (actually running the teacher
// and parsing its output) is intentionally out of scope here.
//
// Mirrors the MODEL-2 pattern set by SHIP-017 (GATE-ARCH-370M-005 in
// `crates/aprender-train/src/models/llama_370m.rs`), which also binds
// AC-SHIP2-007 to a `verdict_from_syntax_error_count` const fn. SHIP-017
// tolerates ≤ 1 error across 100 held-out prompts; SHIP-002 is the
// MODEL-1 twin with a tighter rule (0 errors on the single canonical
// prompt) because the 7B teacher should be essentially flawless on
// the canonical `def fib(n):` completion. Authored self-contained
// because SHIP-017 PR #1004 is not yet on main; once it lands, the
// two `verdict_from_syntax_error_count_*` fns should be deduplicated
// into a single parameterized helper.
//
// MODEL-1 is now at 6/10 AC-SHIP1 items touched (SHIP-008 + SHIP-009
// + SHIP-006 + SHIP-007 + SHIP-005 + SHIP-002).

/// Spec-authorized tolerance for syntax errors on the canonical
/// AC-SHIP1-002 prompt `def fib(n):`. The spec text — "emits valid
/// Python" — carries no noise allowance, so a single syntax error
/// is a ship-blocker. Holding this as a const locks the threshold
/// at compile time and makes any silent widening (e.g. to 1) a
/// test-breaking edit.
pub const AC_SHIP1_002_MAX_TOLERATED_SYNTAX_ERRORS: usize = 0;

/// Binary verdict for FALSIFY-SHIP-002 / GATE-QW2E-SHIP-002.
/// `Pass` iff the observed syntax-error count is at or below the
/// spec tolerance (0). `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship002Verdict {
    /// `syntax_errors <= AC_SHIP1_002_MAX_TOLERATED_SYNTAX_ERRORS`.
    Pass,
    /// `syntax_errors > AC_SHIP1_002_MAX_TOLERATED_SYNTAX_ERRORS`.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-SHIP-002 / GATE-QW2E-SHIP-002
/// / AC-SHIP1-002: the teacher must emit syntactically valid Python on
/// the canonical `def fib(n):` prompt. The input is an integer count
/// of syntax errors produced by the downstream Python AST parse; this
/// function is purely the threshold arbiter.
///
/// Declared `const fn` so the decision rule is evaluable at compile
/// time, matching MODEL-2 SHIP-017's shape exactly (modulo the
/// different tolerance constant).
#[must_use]
pub const fn verdict_from_syntax_error_count(syntax_errors: usize) -> Ship002Verdict {
    if syntax_errors <= AC_SHIP1_002_MAX_TOLERATED_SYNTAX_ERRORS {
        Ship002Verdict::Pass
    } else {
        Ship002Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-SHIP-002 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod ship_002_tests {
    use super::*;

    /// FALSIFY-SHIP-002 algorithm-level PARTIAL discharge: prove the
    /// integer threshold rule for AC-SHIP1-002 Python syntax validity.
    /// Any edit that widens the zero-tolerance rule (e.g. to 1), flips
    /// the inequality direction, or drifts the constant must break
    /// this test before a live `apr run` parse harness runs.
    #[test]
    fn falsify_ship_002_python_syntax_error_threshold_logic() {
        // Section 1: zero errors → Pass (the trivial unanimous-parse
        // case). This is the only scenario where AC-SHIP1-002 can
        // ship.
        assert_eq!(
            verdict_from_syntax_error_count(0),
            Ship002Verdict::Pass,
            "zero syntax errors on the canonical prompt must Pass",
        );

        // Section 2: exactly-one error → Fail. This is the boundary
        // case; MODEL-2 SHIP-017 tolerates 1, but MODEL-1's spec text
        // says "emits valid Python" with no slack, so 1 error is a
        // ship-blocker. Silently bumping the tolerance to 1 must
        // break this.
        assert_eq!(
            verdict_from_syntax_error_count(1),
            Ship002Verdict::Fail,
            "one syntax error on the canonical prompt must Fail (no tolerance)",
        );

        // Section 3: many-errors / clear-Fail band. Any count strictly
        // above the tolerance is a Fail, with no exception. Spot-check
        // at 2, 10, 100 to cover plausible ranges of a deeply-broken
        // completion.
        for errors in [2usize, 10, 100] {
            assert_eq!(
                verdict_from_syntax_error_count(errors),
                Ship002Verdict::Fail,
                "{errors} syntax errors must Fail (above zero tolerance)",
            );
        }

        // Section 4: monotonicity — raising the error count can only
        // worsen the verdict. Once a Fail is observed, no higher
        // count may flip back to Pass. Sweep 0..=256 to exercise the
        // full small-integer range including the boundary.
        let mut seen_fail = false;
        for errors in 0..=256usize {
            let v = verdict_from_syntax_error_count(errors);
            if v == Ship002Verdict::Fail {
                seen_fail = true;
            } else if seen_fail {
                panic!(
                    "monotonicity broken: errors={errors} flipped back to Pass after Fail"
                );
            }
        }

        // Section 5: extreme-value sanity guard. `usize::MAX` — a
        // pathological telemetry overflow — must still cleanly
        // classify as Fail under the unsigned ≥ rule (it is strictly
        // above the 0 tolerance).
        assert_eq!(
            verdict_from_syntax_error_count(usize::MAX),
            Ship002Verdict::Fail,
            "usize::MAX syntax errors must Fail (sanity guard)",
        );

        // Section 6: provenance pin — the tolerance constant is
        // load-bearing and lockstepped with the spec. If AC-SHIP1-002
        // ever widens "emits valid Python" to "emits ≤ N errors", the
        // constant and this test must move together.
        assert_eq!(
            AC_SHIP1_002_MAX_TOLERATED_SYNTAX_ERRORS, 0,
            "tolerance is 0 syntax errors (spec §4.2 AC-SHIP1-002)",
        );
    }
}
