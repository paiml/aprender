// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-008 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-008 | Contract density: every new public fn has #[contract]
// | merge`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.1.0 PROPOSED
// (FALSIFY-GATE-SHIP-008 — wired in the same PR as this file lands).
//
// GATE-SHIP-008 is the *merge-time* contract-density gate: every newly
// introduced public function MUST carry a `#[contract]` annotation
// (provable-contracts-macros). The measured density is the ratio
// contracted_new_fns / total_new_public_fns. The spec requires 100%
// coverage (ratio == 1.0) on new code; a single uncovered public fn
// blocks merge.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given (contracted_new_fns, total_new_public_fns, min_density), the
// verdict is `Pass` iff `total_new_public_fns > 0` (divide-by-zero
// guard), `min_density.is_finite()` and in `[0.0, 1.0]`,
// `contracted_new_fns <= total_new_public_fns` (sanity guard), AND
// `(contracted_new_fns as f32 / total_new_public_fns as f32) >=
// min_density`. The tool-level portion (actually running `pmat density`
// or `grep -c #\[contract\]` on the new-code diff) is intentionally out
// of scope.
//
// Zero-total guard rationale: an empty diff (no new public fns) yields
// a 0/0 division that is conservatively Fail here — a merge gate that
// silently passes on zero-input is indistinguishable from a merge gate
// that was not run. The caller must ensure total_new_public_fns > 0
// before invoking.

/// Minimum contract-density ratio required on new code: 1.0 (100%
/// coverage). Every newly-introduced public function MUST carry a
/// `#[contract]` annotation for the merge gate to Pass.
///
/// Derivation: `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §6 row GATE-SHIP-008 + `.pmat-gates.toml` contract-density policy.
/// Softening this to 0.95 or 0.9 would silently allow contract-less
/// public APIs to ship.
pub const AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE: f32 = 1.0;

/// Binary verdict for FALSIFY-GATE-SHIP-008 / GATE-SHIP-008.
/// `Pass` iff all inputs are well-formed AND the contract density is
/// at or above the threshold. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip008Verdict {
    /// Well-formed inputs AND contract density ≥ threshold. Every new
    /// public fn has a `#[contract]` annotation (for threshold = 1.0)
    /// or enough do (for lower thresholds on legacy code). Merge is
    /// green.
    Pass,
    /// Any of: zero total (empty diff — conservative Fail, no evidence
    /// of coverage); `contracted_new_fns > total_new_public_fns`
    /// (sanity violation); `min_density` non-finite or out of `[0.0,
    /// 1.0]`; measured density below threshold. Merge blocked.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-008 /
/// GATE-SHIP-008: contract-density threshold check on new public fns.
///
/// Conservative-Fail guards:
///
///   - `total_new_public_fns == 0` → Fail (no evidence; gate cannot
///     distinguish "clean diff" from "not-run").
///   - `contracted_new_fns > total_new_public_fns` → Fail (sanity
///     violation; a sensor bug reported more contracted than total).
///   - `!min_density.is_finite()` → Fail (NaN / ±∞ are never valid
///     thresholds).
///   - `min_density < 0.0 || min_density > 1.0` → Fail (contract
///     drift; density is a ratio in the unit interval).
///   - measured ratio `< min_density` → Fail.
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_008::{
///     verdict_from_contract_density, GateShip008Verdict,
///     AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE,
/// };
///
/// // 10/10 contracted at 100% threshold → Pass.
/// assert_eq!(
///     verdict_from_contract_density(
///         10, 10, AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE,
///     ),
///     GateShip008Verdict::Pass
/// );
///
/// // 9/10 at 100% threshold → Fail (merge blocked).
/// assert_eq!(
///     verdict_from_contract_density(
///         9, 10, AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE,
///     ),
///     GateShip008Verdict::Fail
/// );
/// ```
#[must_use]
pub fn verdict_from_contract_density(
    contracted_new_fns: u32,
    total_new_public_fns: u32,
    min_density: f32,
) -> GateShip008Verdict {
    if total_new_public_fns == 0 {
        return GateShip008Verdict::Fail;
    }
    if contracted_new_fns > total_new_public_fns {
        return GateShip008Verdict::Fail;
    }
    if !min_density.is_finite() {
        return GateShip008Verdict::Fail;
    }
    if !(0.0_f32..=1.0_f32).contains(&min_density) {
        return GateShip008Verdict::Fail;
    }
    let measured = contracted_new_fns as f32 / total_new_public_fns as f32;
    if measured >= min_density {
        GateShip008Verdict::Pass
    } else {
        GateShip008Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-008 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_008_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-008 algorithm-level PARTIAL discharge: prove
    /// the contract-density threshold rule. Any edit that relaxes
    /// `>=` to `>`, widens the divide-by-zero guard to Pass, or
    /// skips the range checks must break this test.
    #[test]
    fn falsify_gate_ship_008_contract_density_threshold() {
        // Section 1: 10/10 at min = 1.0 → Pass. The happy path for
        // 100% coverage on a 10-fn diff.
        assert_eq!(
            verdict_from_contract_density(10, 10, AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE,),
            GateShip008Verdict::Pass,
            "10/10 at threshold 1.0 must Pass (full coverage)",
        );

        // Section 2: 9/10 at min = 1.0 → Fail. The sharpest
        // counter-example: a single uncovered public fn blocks merge
        // at the 100% floor.
        assert_eq!(
            verdict_from_contract_density(9, 10, AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE,),
            GateShip008Verdict::Fail,
            "9/10 at threshold 1.0 must Fail (single uncovered fn)",
        );

        // Section 3: 10/10 at min = 0.9 → Pass. Legacy-code threshold
        // case: higher density than required still Passes.
        assert_eq!(
            verdict_from_contract_density(10, 10, 0.9),
            GateShip008Verdict::Pass,
            "10/10 at threshold 0.9 must Pass (above threshold)",
        );
        // And 9/10 at min = 0.9 → Pass (exactly on threshold,
        // inclusive `>=`).
        assert_eq!(
            verdict_from_contract_density(9, 10, 0.9),
            GateShip008Verdict::Pass,
            "9/10 at threshold 0.9 must Pass (inclusive boundary)",
        );
        // 8/10 at min = 0.9 → Fail.
        assert_eq!(
            verdict_from_contract_density(8, 10, 0.9),
            GateShip008Verdict::Fail,
            "8/10 at threshold 0.9 must Fail (below 0.9 floor)",
        );

        // Section 4: zero-total divide-by-zero guard → Fail. Empty
        // diff is indistinguishable from gate-not-run; conservative
        // Fail.
        assert_eq!(
            verdict_from_contract_density(0, 0, 1.0),
            GateShip008Verdict::Fail,
            "0/0 must Fail conservatively (no evidence of coverage)",
        );
        assert_eq!(
            verdict_from_contract_density(5, 0, 0.9),
            GateShip008Verdict::Fail,
            "5/0 must Fail (nonsensical input + divide-by-zero guard)",
        );

        // Section 5: sanity guard — contracted > total is impossible.
        // A sensor bug reporting more contracted than total must Fail.
        assert_eq!(
            verdict_from_contract_density(11, 10, 1.0),
            GateShip008Verdict::Fail,
            "11/10 must Fail (contracted > total is a sensor bug)",
        );
        assert_eq!(
            verdict_from_contract_density(100, 10, 0.5),
            GateShip008Verdict::Fail,
            "100/10 must Fail (gross sensor bug)",
        );

        // Section 6: non-finite and out-of-range thresholds → Fail.
        for &bad in &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                verdict_from_contract_density(10, 10, bad),
                GateShip008Verdict::Fail,
                "10/10 with non-finite threshold ({bad}) must Fail",
            );
        }
        // Out-of-range thresholds.
        assert_eq!(
            verdict_from_contract_density(10, 10, -0.01),
            GateShip008Verdict::Fail,
            "threshold -0.01 must Fail (below 0.0)",
        );
        assert_eq!(
            verdict_from_contract_density(10, 10, 1.01),
            GateShip008Verdict::Fail,
            "threshold 1.01 must Fail (above 1.0)",
        );
        assert_eq!(
            verdict_from_contract_density(10, 10, 2.0),
            GateShip008Verdict::Fail,
            "threshold 2.0 must Fail (above 1.0 — nonsensical density)",
        );

        // Section 7: provenance pin — the 100% floor is load-bearing
        // and lockstepped with spec §6 + `.pmat-gates.toml`. If the
        // policy ever softens to 0.95, this constant and every
        // consumer must move together.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                AC_GATE_SHIP_008_MIN_CONTRACT_DENSITY_NEW_CODE, 1.0_f32,
                "min contract density for new code is 1.0 \
                 (spec §6 GATE-SHIP-008; .pmat-gates.toml contract-density policy)",
            );
        }
    }
}
