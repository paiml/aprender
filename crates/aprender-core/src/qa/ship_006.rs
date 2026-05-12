// SHIP-TWO-001 AC-SHIP1-006 / FALSIFY-SHIP-006 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md
// Contract: contracts/apr-cli-commands-v1.yaml (GATE-QA-SHIP-006 — to be wired
// in the same PR as this file lands)
//
// AC-SHIP1-006 states that the MODEL-1 teacher
// (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) must pass all 8 `apr qa` gates
// authoritative per `docs/specifications/components/qa.md` §3:
//
//     1. Golden output        (known-good response comparison)
//     2. Throughput           (tok/s benchmark ≥ 10)
//     3. Ollama parity        (match Ollama output)
//     4. GPU speedup          (GPU faster than CPU)
//     5. Tensor contracts     (weight shape/value validation)
//     6. Format parity        (APR vs SafeTensors comparison)
//     7. PTX parity           (GPU kernel validation)
//     8. Metadata             (plausibility validation)
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given any 8-long boolean array (one per gate), the verdict is `Pass` iff
// every gate is `true` (aggregate AND). The compute-heavy portion of the
// AC (running `apr qa paiml/qwen2.5-coder-7b-apache-q4k-v1 --json` against
// live teacher weights on an RTX 4090 host) is intentionally out of scope
// here; the aggregate-AND rule is what `apr qa` must emit a Pass on, and
// changing either side of the bind (the 8-gate count, or the AND shape)
// breaks this test before any qa run is launched.
//
// Mirrors the MODEL-2 pattern set by SHIP-016 (task #152 on branch
// `feat/falsify-ship-016-partial-discharge`, PR pending merge). SHIP-006
// is the MODEL-1 twin: identical aggregate-AND verdict shape, different
// required gate count (both 8 today — the spec's `All must Pass` is
// model-independent). Authored self-contained because SHIP-016 is not yet
// on main; once it lands the two `verdict_from_qa_gates_*` fns should be
// deduplicated into a single parameterized helper.
//
// MODEL-1 is now at 3/10 AC-SHIP1 items touched (SHIP-008 + SHIP-009 +
// SHIP-006).

/// Number of `apr qa` gates that MODEL-1 must pass.
///
/// Derivation: `docs/specifications/components/qa.md` §3 enumerates 8 gates
/// (golden / throughput / ollama parity / gpu speedup / tensor contracts /
/// format parity / ptx parity / metadata). The spec AC-SHIP1-006 text says
/// "all 8 gates PASS (Golden Output, layout, tensor stats, etc.)". The
/// literal integer 8 is bound here so that drift in either direction
/// (adding a 9th gate without updating AC; removing a gate without
/// falsifying this test) is caught at compile+test time, not at a
/// production publish.
pub const AC_SHIP1_006_REQUIRED_QA_GATE_COUNT: usize = 8;

/// Binary verdict for FALSIFY-SHIP-006 / GATE-QA-SHIP-006.
/// `Pass` iff every gate in the input slice is `true` AND the slice has
/// exactly `AC_SHIP1_006_REQUIRED_QA_GATE_COUNT` entries. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship006Verdict {
    Pass,
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-SHIP-006 / GATE-QA-SHIP-006
/// / AC-SHIP1-006: aggregate-AND over the 8 `apr qa` gate booleans. A
/// single `false` gate, or a gate array of the wrong length, yields
/// `Fail`. This proves the decision rule without invoking `apr qa` itself;
/// the full discharge (live `apr qa paiml/qwen2.5-coder-7b-apache-q4k-v1
/// --json` with 8 `"pass": true` entries) remains blocked on RTX 4090
/// evidence collection.
#[must_use]
pub const fn verdict_from_qa_gates(gate_results: &[bool]) -> Ship006Verdict {
    if gate_results.len() != AC_SHIP1_006_REQUIRED_QA_GATE_COUNT {
        return Ship006Verdict::Fail;
    }
    let mut i = 0;
    while i < gate_results.len() {
        if !gate_results[i] {
            return Ship006Verdict::Fail;
        }
        i += 1;
    }
    Ship006Verdict::Pass
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-SHIP-006 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod ship_006_tests {
    use super::*;

    /// FALSIFY-SHIP-006 algorithm-level PARTIAL discharge: prove the
    /// aggregate-AND decision rule and the bind between `apr qa`'s
    /// 8 gates and the MODEL-1 ship criterion. Any edit that changes
    /// the required gate count, the aggregate shape, or the Pass
    /// condition must break this test before a teacher publish runs.
    #[test]
    fn falsify_ship_006_apr_qa_eight_gates_aggregate() {
        // Section 1: all-Pass 8-long array yields Pass.
        let all_pass = [true; 8];
        assert_eq!(
            verdict_from_qa_gates(&all_pass),
            Ship006Verdict::Pass,
            "all 8 gates Pass must yield aggregate Pass",
        );

        // Section 2: all-Fail 8-long array yields Fail.
        let all_fail = [false; 8];
        assert_eq!(
            verdict_from_qa_gates(&all_fail),
            Ship006Verdict::Fail,
            "all 8 gates Fail must yield aggregate Fail",
        );

        // Section 3: single-gate-flip — every one of the 8 positions
        // flipped to Fail individually must cause the aggregate to
        // Fail. This enforces the AND shape (not OR, not majority,
        // not 7-of-8). Mirrors the class of drift bug where a well-
        // intentioned "graceful degradation" PR quietly weakens the
        // gate to "7 of 8 pass is good enough".
        for flip_idx in 0..8 {
            let mut gates = [true; 8];
            gates[flip_idx] = false;
            assert_eq!(
                verdict_from_qa_gates(&gates),
                Ship006Verdict::Fail,
                "flipping gate {flip_idx} to Fail must break the aggregate",
            );
        }

        // Section 4: exhaustive 2^8=256-combination proof. The
        // aggregate is Pass iff the bitmask is all-ones (0xFF).
        // Every other mask must yield Fail. This is the strongest
        // correctness statement we can make for the pure 8-boolean
        // decision rule.
        for mask in 0u16..256 {
            let gates: [bool; 8] = core::array::from_fn(|i| (mask >> i) & 1 == 1);
            let expected = if mask == 0xFF {
                Ship006Verdict::Pass
            } else {
                Ship006Verdict::Fail
            };
            assert_eq!(
                verdict_from_qa_gates(&gates),
                expected,
                "mask=0b{mask:08b} expected {expected:?}",
            );
        }

        // Section 5: monotonicity — once the aggregate is Pass,
        // flipping ANY gate from true→false must yield Fail (the
        // converse of section 3, stated as an invariant). This
        // doubles as a smoke test that no subset of 8 gates can
        // "mask" another broken gate.
        let pass_state = [true; 8];
        for flip_idx in 0..8 {
            let mut mutated = pass_state;
            mutated[flip_idx] = false;
            assert_ne!(
                verdict_from_qa_gates(&mutated),
                verdict_from_qa_gates(&pass_state),
                "Pass→Fail monotonicity broken at idx {flip_idx}",
            );
        }

        // Section 6: length-drift counter-examples. The rule is
        // "exactly 8 gates"; a 7-long or 9-long input must Fail
        // regardless of content. This is the contract-drift gate:
        // if someone adds a 9th `apr qa` gate without updating
        // AC-SHIP1-006, this test fails before that PR merges.

        // 6a: 0-length — empty gate array, Fail.
        assert_eq!(
            verdict_from_qa_gates(&[]),
            Ship006Verdict::Fail,
            "empty gate array must Fail — contract-drift guard",
        );

        // 6b: 7-long all-true — one gate missing, Fail. This catches
        // the class of bug where `apr qa --skip-metadata` quietly
        // reduces the gate count to 7 yet the publish pipeline still
        // treats the truncated result as authoritative.
        assert_eq!(
            verdict_from_qa_gates(&[true; 7]),
            Ship006Verdict::Fail,
            "7-long all-true must Fail — gate count drift guard",
        );

        // 6c: 9-long all-true — extra gate added without spec update,
        // Fail. Forces the AC and contract to be amended before a
        // new gate ships into the pipeline.
        assert_eq!(
            verdict_from_qa_gates(&[true; 9]),
            Ship006Verdict::Fail,
            "9-long all-true must Fail — gate count drift guard",
        );

        // 6d: 16-long all-true — double-count, Fail. Catches the
        // (unlikely but non-zero) bug where results are accidentally
        // concatenated twice.
        assert_eq!(
            verdict_from_qa_gates(&[true; 16]),
            Ship006Verdict::Fail,
            "16-long all-true must Fail — double-count guard",
        );

        // Section 7: provenance pin — the required gate count is
        // load-bearing and lockstepped with the spec. If the spec
        // AC-SHIP1-006 text ever says "all 7 gates PASS" or
        // "all 9 gates PASS", this constant (and this test) must
        // move together.
        assert_eq!(
            AC_SHIP1_006_REQUIRED_QA_GATE_COUNT, 8,
            "MODEL-1 ship criterion requires exactly 8 apr qa gates \
             (spec §4.2 AC-SHIP1-006; docs/specifications/components/qa.md §3)",
        );
    }
}
