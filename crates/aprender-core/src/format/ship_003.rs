// SHIP-TWO-001 AC-SHIP1-003 / FALSIFY-SHIP-003 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md
// Contract: contracts/qwen2-e2e-verification-v1.yaml (FALSIFY-QW2E-SHIP-003 —
// wired in the same PR as this file lands).
//
// AC-SHIP1-003 states that the MODEL-1 teacher must survive an
// `apr convert --quantize q4_k_m` round-trip with **every per-layer
// weight tensor preserving cosine similarity ≥ 0.999** between the
// original f32/f16 weights and the dequantized q4_k_m weights. The
// spec's falsification rule is phrased as "any layer cos < 0.999"
// — an aggregate-AND over the per-layer cosine vector.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`
// by binding two pure verdict fns:
//
//   1. `verdict_from_cosine_similarity(sim, threshold) -> Ship003Verdict`
//      — the single-layer rule. `Pass` iff `sim >= threshold` AND the
//      similarity is a finite value in the cosine range `[-1.0, 1.0]`
//      AND the threshold is a finite value in `[-1.0, 1.0]`. Conservative
//      Fail on any non-finite or out-of-range input (a harness bug must
//      never silently ship a broken quantizer).
//
//   2. `verdict_from_per_layer_cosines(sims, threshold) -> Ship003Verdict`
//      — the aggregate-AND rule. `Pass` iff the per-layer vector is
//      non-empty AND every layer's verdict is Pass. `Fail` on the
//      first failing layer, or on the empty input (a round-trip that
//      produced zero per-layer measurements is evidence of a broken
//      harness, not a quantizer success).
//
// The compute-heavy portion of the AC (running `apr convert --quantize
// q4_k_m` against the 7B teacher and measuring per-layer cosines across
// all 28 transformer blocks × {q,k,v,o,gate,up,down,norm} projections)
// is intentionally out of scope here. The threshold rule and the
// aggregate combinator are what the compute harness must emit a Pass
// on, and changing either side of the bind (the 0.999 floor, the range
// guard, the aggregate-AND shape, or the non-empty requirement) breaks
// this test before a single quantization run is launched.
//
// Mirrors the single-number-threshold shape of SHIP-007 / SHIP-020
// (AC_SHIP?_???_MIN_DECODE_TPS_*) and the aggregate-AND shape of
// SHIP-016 (`verdict_from_qa_gates(&[bool])`). Unlike the integer-
// threshold ships (SHIP-005 pass@1, SHIP-017/002 syntax error count),
// the cosine-similarity floor is a float in `[0.0, 1.0]` and demands
// an explicit range guard on top of the usual non-finite / monotonicity
// rules. MODEL-1 is now at 6/10 AC-SHIP1 items touched (SHIP-008 +
// SHIP-009 + SHIP-006 + SHIP-007 + SHIP-005 + SHIP-010) before this
// lands; SHIP-003 brings it to 7/10.

/// Minimum per-layer cosine similarity threshold for the MODEL-1
/// `apr convert --quantize q4_k_m` round-trip. Any transformer-block
/// weight tensor whose cosine similarity between the original f32/f16
/// value and the dequantized q4_k_m value falls below 0.999 fails
/// FALSIFY-SHIP-003.
///
/// Lockstep with `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §4.2 row AC-SHIP1-003: "Convert to APR via `apr convert --quantize
/// q4_k_m`; round-trip weights match (cos ≥ 0.999)".
///
/// f32-exact: `0.999` is representable in IEEE-754 binary32 as the
/// closest-round value `0x3F7FBE77` = 0.99900001287460327148... — the
/// ULP-below neighbour is `0x3F7FBE76` = 0.99899995326995849609...,
/// which the mutation survey uses as a sharp counter-example.
pub const AC_SHIP1_003_MIN_COSINE_SIMILARITY: f32 = 0.999;

/// Binary verdict for FALSIFY-SHIP-003 / AC-SHIP1-003.
/// `Pass` iff every input is well-formed AND the cosine similarity
/// is at or above the threshold (single-layer) OR every per-layer
/// verdict is Pass (aggregate-AND). `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship003Verdict {
    /// Well-formed inputs AND `sim ≥ threshold` (single-layer) OR
    /// every layer Passes (aggregate). The round-trip preserves the
    /// AC-SHIP1-003 floor on this layer / all layers.
    Pass,
    /// Any of: `sim < threshold`; non-finite `sim` or `threshold`;
    /// `sim` or `threshold` outside `[-1.0, 1.0]`; empty per-layer
    /// vector (aggregate); any layer Fails (aggregate).
    Fail,
}

/// Single-layer verdict rule for FALSIFY-SHIP-003 / AC-SHIP1-003:
/// per-layer cosine similarity threshold check. Returns
/// [`Ship003Verdict::Fail`] conservatively for any malformed input so
/// that a harness bug (NaN propagation through the dot product, a
/// quantizer that silently emits a sentinel value, a telemetry bug
/// that clips to infinity) cannot silently ship a failing quantized
/// model.
///
/// Conservative-Fail guards:
///
///   - `!sim.is_finite()` → Fail (NaN / ±∞ are never valid cosines).
///   - `!threshold.is_finite()` → Fail (contract drift — no real
///     cosine floor can be non-finite).
///   - `sim` outside `[-1.0, 1.0]` → Fail (mathematical nonsense; a
///     real cosine between two vectors cannot exceed 1.0 or undercut
///     −1.0 — numerical drift slightly above 1.0 is still rejected
///     because it indicates a broken inner-product routine).
///   - `threshold` outside `[-1.0, 1.0]` → Fail (contract drift — the
///     threshold is a cosine and must live in the cosine range).
#[must_use]
pub fn verdict_from_cosine_similarity(sim: f32, threshold: f32) -> Ship003Verdict {
    if !sim.is_finite() {
        return Ship003Verdict::Fail;
    }
    if !threshold.is_finite() {
        return Ship003Verdict::Fail;
    }
    if !(-1.0_f32..=1.0_f32).contains(&sim) {
        return Ship003Verdict::Fail;
    }
    if !(-1.0_f32..=1.0_f32).contains(&threshold) {
        return Ship003Verdict::Fail;
    }
    if sim >= threshold {
        Ship003Verdict::Pass
    } else {
        Ship003Verdict::Fail
    }
}

/// Aggregate-AND verdict rule for FALSIFY-SHIP-003 / AC-SHIP1-003:
/// "any layer cos < 0.999" is a ship-blocker, so the whole model is
/// a `Pass` iff **every** per-layer verdict is a `Pass`.
///
/// Conservative-Fail on empty input: a round-trip that produced zero
/// per-layer measurements is evidence of a broken
/// harness (the tensor-walker never ran, or the convert step silently
/// dropped every layer) — not a quantizer success. Mirrors the
/// aggregate-AND shape of SHIP-016 `verdict_from_qa_gates` and avoids
/// the "vacuously true" trap.
#[must_use]
pub fn verdict_from_per_layer_cosines(sims: &[f32], threshold: f32) -> Ship003Verdict {
    if sims.is_empty() {
        return Ship003Verdict::Fail;
    }
    for &sim in sims {
        if verdict_from_cosine_similarity(sim, threshold) == Ship003Verdict::Fail {
            return Ship003Verdict::Fail;
        }
    }
    Ship003Verdict::Pass
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-SHIP-003 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod ship_003_tests {
    use super::*;

    /// FALSIFY-SHIP-003 single-layer algorithm-level PARTIAL discharge:
    /// prove the cosine-threshold rule for AC-SHIP1-003. Any edit that
    /// changes the 0.999 floor, the inequality direction, the range
    /// guard `[-1.0, 1.0]`, or the non-finite handling must break this
    /// test before a teacher `apr convert --quantize q4_k_m` quality
    /// run is launched.
    #[test]
    fn falsify_ship_003_cosine_similarity_threshold_logic() {
        // Section 1: exact boundary 0.999 → Pass. `0.999` in f32 is
        // `0x3F7FBE77` = 0.99900001287460327148... — this IS the const
        // value, so the boundary test is `sim == threshold` and must
        // Pass.
        assert_eq!(
            verdict_from_cosine_similarity(
                AC_SHIP1_003_MIN_COSINE_SIMILARITY,
                AC_SHIP1_003_MIN_COSINE_SIMILARITY
            ),
            Ship003Verdict::Pass,
            "sim == threshold at the boundary must Pass",
        );

        // Section 2: one ULP below the threshold → Fail. This is the
        // sharpest single-layer counter-example — a quantizer that
        // lands at the next representable float below 0.999 must fail
        // the ship gate. `f32::from_bits` yields `0x3F7FBE76` =
        // 0.99899995326995849609... which is strictly less than the
        // stored value of `AC_SHIP1_003_MIN_COSINE_SIMILARITY`.
        let one_ulp_below = f32::from_bits(AC_SHIP1_003_MIN_COSINE_SIMILARITY.to_bits() - 1);
        assert!(
            one_ulp_below < AC_SHIP1_003_MIN_COSINE_SIMILARITY,
            "harness sanity: ULP-below must really be less than threshold",
        );
        assert_eq!(
            verdict_from_cosine_similarity(one_ulp_below, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "one ULP below 0.999 must Fail (sharpest counter-example)",
        );

        // Section 3: safely above — 0.9999, 0.99999, 1.0 all Pass.
        assert_eq!(
            verdict_from_cosine_similarity(0.9999, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Pass,
            "0.9999 must Pass the 0.999 floor",
        );
        assert_eq!(
            verdict_from_cosine_similarity(1.0, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Pass,
            "identical cosine 1.0 must Pass (identity quantizer)",
        );

        // Section 4: safely below — 0.998, 0.5, 0.0, -0.5, -1.0 all Fail.
        assert_eq!(
            verdict_from_cosine_similarity(0.998, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "0.998 must Fail (0.001 below floor)",
        );
        assert_eq!(
            verdict_from_cosine_similarity(0.5, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "0.5 must Fail",
        );
        assert_eq!(
            verdict_from_cosine_similarity(0.0, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "orthogonal (0.0) must Fail — worst-case quantizer",
        );
        assert_eq!(
            verdict_from_cosine_similarity(-1.0, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "antipodal (-1.0) must Fail — sign-flipped quantizer",
        );

        // Section 5: monotonicity — sweep sim over a range crossing
        // the threshold, verdict flips from Fail → Pass exactly once
        // and never flips back. If a future refactor introduces a
        // non-monotonic guard (e.g. accidentally rejecting sim == 1.0
        // as "suspiciously perfect"), this test breaks.
        let mut seen_pass = false;
        // Sweep 0.990 .. 1.000 in 1e-4 steps (101 points — enough to
        // straddle the 0.999 boundary and surface any non-monotonic
        // mutant).
        for i in 0..=100 {
            #[allow(clippy::cast_precision_loss)]
            let sim = 0.990 + (i as f32) * 0.0001;
            let v = verdict_from_cosine_similarity(sim, AC_SHIP1_003_MIN_COSINE_SIMILARITY);
            if v == Ship003Verdict::Pass {
                seen_pass = true;
            } else if seen_pass {
                panic!("monotonicity broken: sim={sim} flipped back to Fail after Pass");
            }
        }

        // Section 6: non-finite similarity guard. A dot-product
        // routine that divides by zero norm, or a tensor with Inf
        // entries, must produce a conservative Fail rather than
        // silently flipping to Pass via NaN comparison semantics.
        assert_eq!(
            verdict_from_cosine_similarity(f32::NAN, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "NaN sim must Fail conservatively",
        );
        assert_eq!(
            verdict_from_cosine_similarity(f32::INFINITY, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "+∞ sim must Fail conservatively",
        );
        assert_eq!(
            verdict_from_cosine_similarity(f32::NEG_INFINITY, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "-∞ sim must Fail conservatively",
        );
        // Non-finite threshold (contract drift) must also Fail.
        assert_eq!(
            verdict_from_cosine_similarity(1.0, f32::NAN),
            Ship003Verdict::Fail,
            "NaN threshold must Fail conservatively",
        );
        assert_eq!(
            verdict_from_cosine_similarity(1.0, f32::INFINITY),
            Ship003Verdict::Fail,
            "+∞ threshold must Fail conservatively",
        );

        // Section 7: range guard — cosine similarity must live in
        // `[-1.0, 1.0]`. Any value outside that range is mathematical
        // nonsense (typically a broken inner-product routine that
        // didn't normalize, or a precision-loss bug) and must Fail
        // even if it's "above" the threshold.
        assert_eq!(
            verdict_from_cosine_similarity(1.0001, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "sim > 1.0 must Fail (range guard, not Pass)",
        );
        assert_eq!(
            verdict_from_cosine_similarity(2.0, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "sim = 2.0 must Fail (range guard)",
        );
        assert_eq!(
            verdict_from_cosine_similarity(-1.0001, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "sim < -1.0 must Fail (range guard)",
        );
        // Out-of-range threshold (contract drift) must also Fail.
        assert_eq!(
            verdict_from_cosine_similarity(1.0, 1.5),
            Ship003Verdict::Fail,
            "threshold > 1.0 must Fail (contract drift)",
        );
        assert_eq!(
            verdict_from_cosine_similarity(1.0, -2.0),
            Ship003Verdict::Fail,
            "threshold < -1.0 must Fail (contract drift)",
        );

        // Section 8: provenance pin — the 0.999 constant is load-
        // bearing and lockstepped with the spec. If AC-SHIP1-003 ever
        // changes the floor (say to 0.9995 for a tighter ship gate),
        // this const must move in lockstep and the rest of the survey
        // follows.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                AC_SHIP1_003_MIN_COSINE_SIMILARITY, 0.999,
                "minimum cosine similarity is 0.999 (spec §4.2 AC-SHIP1-003)",
            );
        }
    }

    /// FALSIFY-SHIP-003 aggregate-AND algorithm-level PARTIAL discharge:
    /// prove the per-layer aggregate combinator. Any edit that changes
    /// the "any layer < 0.999 fails the whole model" shape, or the
    /// conservative empty-Fail, must break this test.
    #[test]
    fn falsify_ship_003_per_layer_aggregate_and() {
        // Section 1: all-Pass vector → Pass. Sweep includes exact
        // boundary (0.999), well-above (0.9999), and identity (1.0).
        assert_eq!(
            verdict_from_per_layer_cosines(
                &[
                    AC_SHIP1_003_MIN_COSINE_SIMILARITY,
                    0.9999,
                    1.0,
                    0.9995,
                    0.99999,
                ],
                AC_SHIP1_003_MIN_COSINE_SIMILARITY
            ),
            Ship003Verdict::Pass,
            "vector of all Pass must be aggregate Pass",
        );

        // Section 2: single-layer failure → aggregate Fail. A
        // realistic 28-layer Qwen2.5-7B model would have 28 blocks
        // × 7 projections = 196 cosines; we simulate by inserting
        // one Fail into an otherwise Pass vector.
        let one_ulp_below = f32::from_bits(AC_SHIP1_003_MIN_COSINE_SIMILARITY.to_bits() - 1);
        let mut mostly_pass = [AC_SHIP1_003_MIN_COSINE_SIMILARITY; 196];
        mostly_pass[100] = one_ulp_below;
        assert_eq!(
            verdict_from_per_layer_cosines(&mostly_pass, AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "195 Pass + 1 Fail must be aggregate Fail (any-layer rule)",
        );

        // Section 3: all-Fail vector → aggregate Fail.
        assert_eq!(
            verdict_from_per_layer_cosines(
                &[0.5, 0.7, 0.8, 0.9],
                AC_SHIP1_003_MIN_COSINE_SIMILARITY
            ),
            Ship003Verdict::Fail,
            "all-Fail must be aggregate Fail",
        );

        // Section 4: empty vector → conservative Fail. A round-trip
        // that produced zero per-layer measurements is a harness bug,
        // not a quantizer success. Avoid the "vacuously true"
        // aggregate-AND trap.
        assert_eq!(
            verdict_from_per_layer_cosines(&[], AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "empty per-layer vector must Fail conservatively",
        );

        // Section 5: single-element vector — behaves as single-layer.
        // Smoke test that aggregate doesn't introduce off-by-one or
        // special-case the len==1 path.
        assert_eq!(
            verdict_from_per_layer_cosines(&[1.0], AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Pass,
            "single-element all-Pass must be aggregate Pass",
        );
        assert_eq!(
            verdict_from_per_layer_cosines(&[0.5], AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "single-element Fail must be aggregate Fail",
        );

        // Section 6: first-layer Fail short-circuits — any tainted
        // value (NaN, out-of-range) in position 0 must produce Fail
        // without reading later entries. This also proves the
        // aggregate routes through `verdict_from_cosine_similarity`
        // rather than re-implementing the single-layer rule.
        assert_eq!(
            verdict_from_per_layer_cosines(
                &[f32::NAN, 1.0, 1.0],
                AC_SHIP1_003_MIN_COSINE_SIMILARITY
            ),
            Ship003Verdict::Fail,
            "first-layer NaN must Fail even if remaining layers Pass",
        );
        assert_eq!(
            verdict_from_per_layer_cosines(&[2.0, 1.0, 1.0], AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "first-layer out-of-range must Fail even if remaining layers Pass",
        );

        // Section 7: last-layer Fail is still caught — proves the
        // aggregate doesn't short-circuit to Pass after seeing any
        // Pass (a classic mutation: replace `== Fail` with `== Pass`
        // in the early-return, which would flip the whole aggregate).
        assert_eq!(
            verdict_from_per_layer_cosines(&[1.0, 1.0, 0.5], AC_SHIP1_003_MIN_COSINE_SIMILARITY),
            Ship003Verdict::Fail,
            "last-layer Fail must still Fail the aggregate",
        );
    }

    /// FALSIFY-QW2E-SHIP-003 YAML binding: parses
    /// `qwen2-e2e-verification-v1.yaml` and asserts the
    /// FALSIFY-QW2E-SHIP-003 falsification block has been promoted from
    /// `PARTIAL_ALGORITHM_LEVEL` (v1.4.0) → `DISCHARGED` (v1.10.0) via
    /// live `apr diff` 339-tensor cosine sweep on noah-Lambda-Vector
    /// RTX 4090. Falsifier: if the contract is edited to drop the
    /// live-evidence block or downgrade the discharge marker, this
    /// test fails before any compute runs.
    #[test]
    fn falsify_ship_003_yaml_binding_pins_discharged_status() {
        const CONTRACT_YAML: &str =
            include_str!("../../../../contracts/qwen2-e2e-verification-v1.yaml");

        let doc: serde_yaml::Value = serde_yaml::from_str(CONTRACT_YAML)
            .expect("qwen2-e2e-verification-v1.yaml must parse as YAML");

        let falsifications = doc["falsification_tests"]
            .as_sequence()
            .expect("falsification_tests must be a sequence");
        let block = falsifications
            .iter()
            .find(|b| b["id"].as_str() == Some("FALSIFY-QW2E-SHIP-003"))
            .expect("FALSIFY-QW2E-SHIP-003 must exist in qwen2-e2e-verification-v1");

        assert_eq!(
            block["discharge_status"].as_str(),
            Some("DISCHARGED"),
            "FALSIFY-QW2E-SHIP-003 must advertise DISCHARGED \
             (live `apr diff` 339-tensor cosine sweep at v1.10.0; \
              previous PARTIAL_ALGORITHM_LEVEL at v1.4.0)",
        );
        assert!(
            block["discharged_evidence"].is_mapping(),
            "FALSIFY-QW2E-SHIP-003 DISCHARGED status requires a discharged_evidence \
             block documenting host, command, cosine summary, and live verdicts",
        );
        assert_eq!(
            block["discharged_evidence"]["host"].as_str(),
            Some("noah-Lambda-Vector"),
            "discharged_evidence.host must pin to lambda-labs RTX 4090",
        );
        assert_eq!(
            block["discharged_evidence"]["aggregate_verdict"].as_str(),
            Some("Pass"),
            "discharged_evidence.aggregate_verdict must equal Pass",
        );
        assert_eq!(
            block["discharged_evidence"]["tensors_compared"].as_u64(),
            Some(339),
            "discharged_evidence.tensors_compared must equal 339 (Qwen2.5-Coder-7B canonical)",
        );
        assert_eq!(
            block["discharged_evidence"]["cosine_summary"]["below_threshold_count"].as_u64(),
            Some(0),
            "discharged_evidence.cosine_summary.below_threshold_count must equal 0 \
             (zero tensors below the 0.999 floor)",
        );
        let live_evidence = block["discharged_evidence"]["evidence_discharged_by_live"]
            .as_sequence()
            .expect(
                "FALSIFY-QW2E-SHIP-003 DISCHARGED requires \
                 discharged_evidence.evidence_discharged_by_live (live RTX 4090 evidence list)",
            );
        assert!(
            !live_evidence.is_empty(),
            "evidence_discharged_by_live must list at least one live observation",
        );
    }
}
