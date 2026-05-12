// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-010 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-010 | cargo deny check advisories — zero vulnerabilities
// in weight/tokenizer dependencies | merge`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.1.0 PROPOSED
// (FALSIFY-GATE-SHIP-010 — wired in the same PR as this file lands).
//
// GATE-SHIP-010 is the *merge-time* security-audit gate: `cargo deny
// check advisories` MUST report zero vulnerabilities in the workspace
// dependency tree. The tool emits an integer "advisory count"; the
// gate enforces that count == 0.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given a u32 count of open advisories, the verdict is `Pass` iff
// `count == 0` and `Fail` otherwise. The tool-level portion (actually
// running `cargo deny check advisories` on the workspace) is
// intentionally out of scope. Zero-tolerance twin of GATE-SHIP-007
// (unwrap count) at the same `0` floor, different semantic domain.

/// Maximum tolerated count of open security advisories in the workspace
/// dependency tree. Zero-tolerance: any open advisory blocks merge.
///
/// Derivation: `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §6 row GATE-SHIP-010 + `deny.toml` severity policy. Changing this
/// value from 0 is a security-policy amendment that requires a spec
/// update, a contract bump, and an INFOSEC review.
pub const AC_GATE_SHIP_010_MAX_TOLERATED_ADVISORY_COUNT: u32 = 0;

/// Binary verdict for FALSIFY-GATE-SHIP-010 / GATE-SHIP-010.
/// `Pass` iff `count == 0`. `Fail` otherwise (any open advisory is a
/// merge-blocker).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip010Verdict {
    /// No open security advisories in the workspace dependency tree.
    /// `cargo deny check advisories` is green. Merge gate is green.
    Pass,
    /// At least one open security advisory in the workspace
    /// dependency tree. Merge is blocked until every advisory is
    /// resolved (upgrade, replace, or document via `[advisories]
    /// ignore`).
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-010 /
/// GATE-SHIP-010: zero-tolerance security-advisory count threshold.
///
/// Zero-tolerance rationale: weight/tokenizer dependencies ship with
/// published artifacts; a CVE in one of them can compromise every
/// downstream consumer. Softening to "≤ N" would silently allow
/// drift one PR at a time.
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_010::{
///     verdict_from_advisory_count, GateShip010Verdict,
/// };
///
/// // No open advisories → Pass.
/// assert_eq!(
///     verdict_from_advisory_count(0),
///     GateShip010Verdict::Pass
/// );
///
/// // Single open advisory → Fail (merge blocked).
/// assert_eq!(
///     verdict_from_advisory_count(1),
///     GateShip010Verdict::Fail
/// );
/// ```
#[must_use]
pub const fn verdict_from_advisory_count(count: u32) -> GateShip010Verdict {
    if count == AC_GATE_SHIP_010_MAX_TOLERATED_ADVISORY_COUNT {
        GateShip010Verdict::Pass
    } else {
        GateShip010Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-010 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_010_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-010 algorithm-level PARTIAL discharge: prove
    /// the zero-tolerance advisory-count threshold. Any edit that
    /// softens `== 0` to `<= N` for any positive N must break this
    /// test before a merge.
    #[test]
    fn falsify_gate_ship_010_advisory_zero_tolerance() {
        // Section 1: zero-count boundary → Pass. The happy path: a
        // clean `cargo deny check advisories` run.
        assert_eq!(
            verdict_from_advisory_count(0),
            GateShip010Verdict::Pass,
            "count = 0 must Pass (clean deny audit)",
        );

        // Section 2: one-count adjacent boundary → Fail. Sharpest
        // counter-example: a single open advisory blocks merge.
        assert_eq!(
            verdict_from_advisory_count(1),
            GateShip010Verdict::Fail,
            "count = 1 must Fail (single open CVE blocks merge)",
        );

        // Section 3: Fail band — realistic multi-advisory scenarios
        // (e.g., a transitive dep with several stacked CVEs, or a
        // pre-upgrade audit reporting many issues).
        for &n in &[2_u32, 10, 100, u32::MAX] {
            assert_eq!(
                verdict_from_advisory_count(n),
                GateShip010Verdict::Fail,
                "count = {n} must Fail (zero-tolerance security band)",
            );
        }

        // Section 4: monotonicity sweep 0..=256 — exactly one input
        // yields Pass (count = 0). Catches mutations that flip the
        // inequality direction (e.g., `!= 0` → `>= 0` would Pass all
        // counts).
        for n in 0_u32..=256 {
            let verdict = verdict_from_advisory_count(n);
            let expected = if n == 0 {
                GateShip010Verdict::Pass
            } else {
                GateShip010Verdict::Fail
            };
            assert_eq!(
                verdict, expected,
                "count = {n} expected {expected:?}, got {verdict:?}",
            );
        }

        // Section 5: provenance pin — zero-tolerance is load-bearing
        // and lockstepped with `deny.toml` severity policy. Softening
        // this (e.g., to ignore low-severity informationals) is an
        // infosec-review-gated amendment.
        assert_eq!(
            AC_GATE_SHIP_010_MAX_TOLERATED_ADVISORY_COUNT, 0,
            "max tolerated advisory count is 0 \
             (spec §6 GATE-SHIP-010; deny.toml severity policy)",
        );
    }
}
