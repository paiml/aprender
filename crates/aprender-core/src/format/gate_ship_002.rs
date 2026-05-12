// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-002 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-002 | MODEL-2: all 12 AC-SHIP2-* PASS | MODEL-2 publish`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.0.0 PROPOSED
// (FALSIFY-GATE-SHIP-002 — wired in the same PR as this file lands).
//
// GATE-SHIP-002 states that the MODEL-2 sovereign 370M model
// (`paiml/llama-370m-sovereign-apache-v1`) cannot ship unless every
// one of the 12 AC-SHIP2-* acceptance criteria (§4.4 rows AC-SHIP2-001
// through AC-SHIP2-012) reports Pass. The compound gate's decision
// rule is aggregate-AND over the 12 per-AC booleans.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given any 12-long boolean array (one per AC-SHIP2 item), the verdict
// is `Pass` iff every AC is `true` AND the array has exactly 12 entries.
// An 11-long input (AC removal) or a 13-long input (AC addition), or
// any single `false` AC, yields `Fail`. The compute-heavy portion
// (actually running all 12 per-AC harnesses against real 370M weights
// on RTX 4090) is intentionally out of scope here.
//
// Twins with GATE-SHIP-001 (MODEL-1 aggregate-AND over 10 ACs);
// differs only in element count (12 vs 10) and the exhaustive bitmask
// proof size (4096 vs 1024). Together they fix the "aggregate-AND over
// N booleans" pattern at two widths for both model publish gates.

/// Number of AC-SHIP2-* acceptance criteria that MODEL-2 must pass for
/// GATE-SHIP-002 to Pass.
///
/// Derivation: `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §4.4 enumerates 12 rows AC-SHIP2-001 through AC-SHIP2-012
/// (architecture scaffold / tokenizer build / corpus ingest /
/// checkpoint cadence / pretrain loop / decode tok/s / `apr qa` /
/// HumanEval / cosine parity / decode floor / chat template /
/// provenance). The literal integer 12 is bound here so that drift
/// in either direction (adding AC-SHIP2-013 without updating
/// GATE-SHIP-002; removing an AC-SHIP2-* without falsifying this
/// test) is caught at compile+test time.
pub const AC_GATE_SHIP_002_MODEL_2_AC_COUNT: usize = 12;

/// Binary verdict for FALSIFY-GATE-SHIP-002 / GATE-SHIP-002.
/// `Pass` iff every AC in the input slice is `true` AND the slice has
/// exactly `AC_GATE_SHIP_002_MODEL_2_AC_COUNT` entries. `Fail`
/// otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip002Verdict {
    /// All 12 AC-SHIP2-* items reported Pass. The MODEL-2 sovereign
    /// has cleared every §4.4 acceptance criterion and may proceed
    /// to publish (subject to GATE-SHIP-003..006 cross-cutting gates).
    Pass,
    /// At least one of: slice length ≠ 12; any AC reported `false`.
    /// MODEL-2 publish is blocked.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-002 /
/// GATE-SHIP-002: aggregate-AND over the 12 AC-SHIP2-* booleans.
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_002::{
///     verdict_from_model2_ac_aggregate, GateShip002Verdict,
/// };
///
/// // Happy path: all 12 AC-SHIP2-* items Pass.
/// let all_pass = [true; 12];
/// assert_eq!(
///     verdict_from_model2_ac_aggregate(&all_pass),
///     GateShip002Verdict::Pass
/// );
///
/// // Any length ≠ 12 Fails.
/// assert_eq!(
///     verdict_from_model2_ac_aggregate(&[true; 11]),
///     GateShip002Verdict::Fail
/// );
/// ```
#[must_use]
pub const fn verdict_from_model2_ac_aggregate(ac_passes: &[bool]) -> GateShip002Verdict {
    if ac_passes.len() != AC_GATE_SHIP_002_MODEL_2_AC_COUNT {
        return GateShip002Verdict::Fail;
    }
    let mut i = 0;
    while i < ac_passes.len() {
        if !ac_passes[i] {
            return GateShip002Verdict::Fail;
        }
        i += 1;
    }
    GateShip002Verdict::Pass
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-002 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_002_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-002 algorithm-level PARTIAL discharge: prove
    /// the aggregate-AND decision rule binding the 12 AC-SHIP2-* items
    /// to the MODEL-2 publish criterion.
    #[test]
    fn falsify_gate_ship_002_model2_ac_aggregate() {
        // Section 1: all-Pass 12-long array yields Pass.
        let all_pass = [true; 12];
        assert_eq!(
            verdict_from_model2_ac_aggregate(&all_pass),
            GateShip002Verdict::Pass,
            "all 12 AC-SHIP2 items Pass must yield aggregate Pass",
        );

        // Section 2: single-AC-flip — each of the 12 positions flipped
        // to Fail individually must cause the aggregate to Fail. This
        // enforces the AND shape (not OR, not majority, not 11-of-12).
        for flip_idx in 0..12 {
            let mut acs = [true; 12];
            acs[flip_idx] = false;
            assert_eq!(
                verdict_from_model2_ac_aggregate(&acs),
                GateShip002Verdict::Fail,
                "flipping AC-SHIP2-{:03} to Fail must break the aggregate",
                flip_idx + 1,
            );
        }

        // Section 3: length-drift counter-examples.

        // 3a: 0-length empty → Fail.
        assert_eq!(
            verdict_from_model2_ac_aggregate(&[]),
            GateShip002Verdict::Fail,
            "empty AC array must Fail — contract-drift guard",
        );

        // 3b: 11-long all-true → Fail (AC removed without spec update).
        assert_eq!(
            verdict_from_model2_ac_aggregate(&[true; 11]),
            GateShip002Verdict::Fail,
            "11-long all-true must Fail — AC-removal drift guard",
        );

        // 3c: 13-long all-true → Fail (AC added without spec update).
        assert_eq!(
            verdict_from_model2_ac_aggregate(&[true; 13]),
            GateShip002Verdict::Fail,
            "13-long all-true must Fail — AC-addition drift guard",
        );

        // 3d: 24-long all-true → Fail. Double-count guard (e.g.,
        // accidental concatenation of MODEL-1 and MODEL-2 ACs, or
        // double-pass concat).
        assert_eq!(
            verdict_from_model2_ac_aggregate(&[true; 24]),
            GateShip002Verdict::Fail,
            "24-long all-true must Fail — double-count guard",
        );

        // Section 4: exhaustive 2^12=4096-combination proof. The
        // aggregate is Pass iff the bitmask is all-ones (0xFFF).
        // Every other mask must yield Fail.
        for mask in 0u16..4096 {
            let acs: [bool; 12] = core::array::from_fn(|i| (mask >> i) & 1 == 1);
            let expected = if mask == 0xFFF {
                GateShip002Verdict::Pass
            } else {
                GateShip002Verdict::Fail
            };
            assert_eq!(
                verdict_from_model2_ac_aggregate(&acs),
                expected,
                "mask=0b{mask:012b} expected {expected:?}",
            );
        }

        // Section 5: monotonicity — once Pass, any flip Pass→Fail.
        let pass_state = [true; 12];
        for flip_idx in 0..12 {
            let mut mutated = pass_state;
            mutated[flip_idx] = false;
            assert_ne!(
                verdict_from_model2_ac_aggregate(&mutated),
                verdict_from_model2_ac_aggregate(&pass_state),
                "Pass→Fail monotonicity broken at idx {flip_idx}",
            );
        }

        // Section 6: provenance pin — the required AC count is load-
        // bearing and lockstepped with the spec §4.4. If the table
        // ever grows to 13 rows or shrinks to 11, this constant (and
        // this test) must move together.
        assert_eq!(
            AC_GATE_SHIP_002_MODEL_2_AC_COUNT, 12,
            "MODEL-2 compound ship gate requires exactly 12 AC-SHIP2 items \
             (spec §4.4 AC-SHIP2-001 through AC-SHIP2-012; §6 GATE-SHIP-002)",
        );
    }
}
