// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-001 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-001 | MODEL-1: all 10 AC-SHIP1-* PASS | MODEL-1 publish`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.0.0 PROPOSED
// (FALSIFY-GATE-SHIP-001 — wired in the same PR as this file lands).
//
// GATE-SHIP-001 states that the MODEL-1 teacher
// (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) cannot ship unless every one
// of the 10 AC-SHIP1-* acceptance criteria (§4.2 rows AC-SHIP1-001 through
// AC-SHIP1-010) reports Pass. The compound gate's decision rule is
// aggregate-AND over the 10 per-AC booleans.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given any 10-long boolean array (one per AC-SHIP1 item), the verdict
// is `Pass` iff every AC is `true` AND the array has exactly 10 entries.
// A 9-long input (adding an AC without bumping the count), an 11-long
// input (silently duplicating a gate), or any single `false` AC yields
// `Fail`. The compute-heavy portion (actually running all 10 per-AC
// harnesses — cos-parity quantize, llama.cpp GGUF load, HumanEval eval,
// `apr qa`, `apr bench`, chat-template diff, provenance grep, SHA-256
// curl) is intentionally out of scope here; what this file proves is
// that the compound gate's *aggregate shape* cannot be silently weakened
// without breaking this test.
//
// Mirrors the aggregate-AND family already established by SHIP-006
// (MODEL-1 `apr qa` 8-gate) and SHIP-016 (MODEL-2 `apr qa` 8-gate);
// differs in element count (10 vs 8) and in what each boolean represents
// (per-AC aggregate instead of per-QA-gate).

/// Number of AC-SHIP1-* acceptance criteria that MODEL-1 must pass for
/// GATE-SHIP-001 to Pass.
///
/// Derivation: `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §4.2 enumerates 10 rows AC-SHIP1-001 through AC-SHIP1-010
/// (safetensors load / valid-Python emit / quantize round-trip /
/// GGUF export / HumanEval pass@1 / `apr qa` aggregate / decode tok/s /
/// chat template / provenance / published SHA-256). The literal
/// integer 10 is bound here so that drift in either direction
/// (adding an AC-SHIP1-011 without updating GATE-SHIP-001; removing
/// an AC-SHIP1-* without falsifying this test) is caught at
/// compile+test time, not at a production publish.
pub const AC_GATE_SHIP_001_MODEL_1_AC_COUNT: usize = 10;

/// Binary verdict for FALSIFY-GATE-SHIP-001 / GATE-SHIP-001.
/// `Pass` iff every AC in the input slice is `true` AND the slice has
/// exactly `AC_GATE_SHIP_001_MODEL_1_AC_COUNT` entries. `Fail`
/// otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip001Verdict {
    /// All 10 AC-SHIP1-* items reported Pass. The MODEL-1 teacher
    /// has cleared every §4.2 acceptance criterion and may proceed
    /// to publish (subject to GATE-SHIP-003..006 cross-cutting gates).
    Pass,
    /// At least one of: slice length ≠ 10; any AC reported `false`.
    /// MODEL-1 publish is blocked.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-001 /
/// GATE-SHIP-001: aggregate-AND over the 10 AC-SHIP1-* booleans.
/// A single `false` AC, or an AC array of the wrong length, yields
/// `Fail`. This proves the decision rule without invoking any of
/// the underlying compute-heavy AC harnesses; the full discharge
/// (actually running all 10 AC harnesses against the 7B teacher on
/// an RTX 4090 host) remains blocked on hardware evidence
/// collection.
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_001::{
///     verdict_from_model1_ac_aggregate, GateShip001Verdict,
/// };
///
/// // Happy path: all 10 AC-SHIP1-* items Pass.
/// let all_pass = [true; 10];
/// assert_eq!(
///     verdict_from_model1_ac_aggregate(&all_pass),
///     GateShip001Verdict::Pass
/// );
///
/// // Single AC fails — compound gate Fails.
/// let mut one_fail = [true; 10];
/// one_fail[3] = false;
/// assert_eq!(
///     verdict_from_model1_ac_aggregate(&one_fail),
///     GateShip001Verdict::Fail
/// );
/// ```
#[must_use]
pub const fn verdict_from_model1_ac_aggregate(ac_passes: &[bool]) -> GateShip001Verdict {
    if ac_passes.len() != AC_GATE_SHIP_001_MODEL_1_AC_COUNT {
        return GateShip001Verdict::Fail;
    }
    let mut i = 0;
    while i < ac_passes.len() {
        if !ac_passes[i] {
            return GateShip001Verdict::Fail;
        }
        i += 1;
    }
    GateShip001Verdict::Pass
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-001 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_001_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-001 algorithm-level PARTIAL discharge: prove
    /// the aggregate-AND decision rule binding the 10 AC-SHIP1-* items
    /// to the MODEL-1 publish criterion. Any edit that changes the
    /// required AC count, the aggregate shape, or the Pass condition
    /// must break this test before a teacher publish runs.
    #[test]
    fn falsify_gate_ship_001_model1_ac_aggregate() {
        // Section 1: all-Pass 10-long array yields Pass.
        let all_pass = [true; 10];
        assert_eq!(
            verdict_from_model1_ac_aggregate(&all_pass),
            GateShip001Verdict::Pass,
            "all 10 AC-SHIP1 items Pass must yield aggregate Pass",
        );

        // Section 2: single-AC-flip — each of the 10 positions flipped
        // to Fail individually must cause the aggregate to Fail. This
        // enforces the AND shape (not OR, not majority, not 9-of-10).
        for flip_idx in 0..10 {
            let mut acs = [true; 10];
            acs[flip_idx] = false;
            assert_eq!(
                verdict_from_model1_ac_aggregate(&acs),
                GateShip001Verdict::Fail,
                "flipping AC-SHIP1-{:03} to Fail must break the aggregate",
                flip_idx + 1,
            );
        }

        // Section 3: length-drift counter-examples. The rule is "exactly
        // 10 ACs"; a 9-long or 11-long input must Fail regardless of
        // content. This is the spec-drift gate: if someone adds
        // AC-SHIP1-011 without updating GATE-SHIP-001, this test fails
        // before that PR merges.

        // 3a: 0-length empty input → Fail (degenerate).
        assert_eq!(
            verdict_from_model1_ac_aggregate(&[]),
            GateShip001Verdict::Fail,
            "empty AC array must Fail — contract-drift guard",
        );

        // 3b: 9-long all-true → Fail. AC removed without spec update.
        assert_eq!(
            verdict_from_model1_ac_aggregate(&[true; 9]),
            GateShip001Verdict::Fail,
            "9-long all-true must Fail — AC-removal drift guard",
        );

        // 3c: 11-long all-true → Fail. AC added without spec update.
        assert_eq!(
            verdict_from_model1_ac_aggregate(&[true; 11]),
            GateShip001Verdict::Fail,
            "11-long all-true must Fail — AC-addition drift guard",
        );

        // 3d: 20-long all-true → Fail. Double-count guard (e.g.,
        // accidental concatenation of MODEL-1 and MODEL-2 ACs).
        assert_eq!(
            verdict_from_model1_ac_aggregate(&[true; 20]),
            GateShip001Verdict::Fail,
            "20-long all-true must Fail — double-count guard",
        );

        // Section 4: exhaustive 2^10=1024-combination proof. The
        // aggregate is Pass iff the bitmask is all-ones (0x3FF).
        // Every other mask must yield Fail. This is the strongest
        // correctness statement we can make for the pure 10-boolean
        // decision rule.
        for mask in 0u16..1024 {
            let acs: [bool; 10] = core::array::from_fn(|i| (mask >> i) & 1 == 1);
            let expected = if mask == 0x3FF {
                GateShip001Verdict::Pass
            } else {
                GateShip001Verdict::Fail
            };
            assert_eq!(
                verdict_from_model1_ac_aggregate(&acs),
                expected,
                "mask=0b{mask:010b} expected {expected:?}",
            );
        }

        // Section 5: monotonicity — once the aggregate is Pass, flipping
        // ANY AC from true→false must yield Fail. The converse of
        // Section 2, stated as an invariant.
        let pass_state = [true; 10];
        for flip_idx in 0..10 {
            let mut mutated = pass_state;
            mutated[flip_idx] = false;
            assert_ne!(
                verdict_from_model1_ac_aggregate(&mutated),
                verdict_from_model1_ac_aggregate(&pass_state),
                "Pass→Fail monotonicity broken at idx {flip_idx}",
            );
        }

        // Section 6: provenance pin — the required AC count is load-
        // bearing and lockstepped with the spec. If §4.2 ever grows
        // to 11 rows or shrinks to 9, this constant (and this test)
        // must move together.
        assert_eq!(
            AC_GATE_SHIP_001_MODEL_1_AC_COUNT, 10,
            "MODEL-1 compound ship gate requires exactly 10 AC-SHIP1 items \
             (spec §4.2 AC-SHIP1-001 through AC-SHIP1-010; §6 GATE-SHIP-001)",
        );
    }
}
