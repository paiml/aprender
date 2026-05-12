// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-003 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-003 | Both models: apr qa Golden Output never regresses
// post-quantize | publish`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.0.0 PROPOSED
// (FALSIFY-GATE-SHIP-003 — wired in the same PR as this file lands).
//
// GATE-SHIP-003 states that for both MODEL-1 and MODEL-2, the
// `apr qa` Golden Output gate MUST produce byte-identical emissions
// before and after a quantization round-trip (`apr convert --quantize
// q4_k_m`). Any drift in the emitted bytes is a ship-blocker: the
// Golden Output gate is the stack's last-line defence against silent
// quality regressions in the distilled/quantized checkpoint.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given two byte slices representing pre-quantize and post-quantize
// Golden Output emissions, the verdict is `Pass` iff BOTH are non-empty
// (empty = no Golden Output recorded = no regression proof possible)
// AND they are byte-by-byte equal. The compute-heavy portion (actually
// running `apr qa <model>.apr` on the pre-quantize and post-quantize
// checkpoints to produce the two byte streams) is intentionally out of
// scope here; what this file proves is that the compound gate's
// *comparison shape* cannot be silently relaxed (e.g., to a Unicode-
// folded or case-insensitive compare) without breaking this test.
//
// Conservative-Fail rationale for empty inputs: if the Golden Output
// gate was SKIPPED (tokenizer missing, feature flag off, etc.), there
// is NO evidence that the model emits the canonical bytes. A missing
// Golden Output is treated as a ship-blocker per apr-model-qa-v1.yaml
// `FALSIFY-EX-001` / `--require-golden-output` promotion. Here we
// surface that semantics at the decision-rule layer: empty slice on
// either side → Fail.

/// Binary verdict for FALSIFY-GATE-SHIP-003 / GATE-SHIP-003.
/// `Pass` iff both pre-quantize and post-quantize Golden Output byte
/// streams are non-empty AND byte-by-byte equal. `Fail` otherwise
/// (length mismatch, any byte difference, empty on either side).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip003Verdict {
    /// Both byte streams are non-empty and byte-identical. The
    /// quantization round-trip preserves Golden Output and the
    /// compound gate passes.
    Pass,
    /// At least one of: either side is empty (no Golden Output
    /// recorded — conservative Fail); lengths differ; any byte
    /// position differs. MODEL-* publish is blocked.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-003 /
/// GATE-SHIP-003: pre-quantize vs post-quantize Golden Output byte-
/// identity check.
///
/// Conservative-Fail guards:
///
///   - Either side empty → Fail. An empty Golden Output means the
///     `apr qa` gate was SKIPPED (tokenizer missing, feature flag off,
///     etc.); we cannot prove "no regression" without evidence on both
///     sides, so we conservatively Fail to block the publish.
///   - Lengths differ → Fail (would not even be byte-equal, but we
///     short-circuit on length for speed AND for a clearer failure
///     signal).
///   - Any byte position differs → Fail.
///
/// This is *byte-identity*, not Unicode-folded equality: a whitespace
/// drift, a trailing newline, a BOM promotion are all ship-blockers.
/// The Golden Output gate is the single byte-exact guard against
/// quantization-induced drift; relaxing this rule would let a
/// quantizer that silently corrupts the prompt template pass.
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_003::{
///     verdict_from_golden_output_diff, GateShip003Verdict,
/// };
///
/// // Byte-identical Golden Output → Pass.
/// let pre = b"42\n".to_vec();
/// let post = b"42\n".to_vec();
/// assert_eq!(
///     verdict_from_golden_output_diff(&pre, &post),
///     GateShip003Verdict::Pass
/// );
///
/// // Single-byte drift → Fail.
/// let post_drift = b"43\n".to_vec();
/// assert_eq!(
///     verdict_from_golden_output_diff(&pre, &post_drift),
///     GateShip003Verdict::Fail
/// );
/// ```
#[must_use]
pub fn verdict_from_golden_output_diff(
    pre_quantize: &[u8],
    post_quantize: &[u8],
) -> GateShip003Verdict {
    if pre_quantize.is_empty() || post_quantize.is_empty() {
        return GateShip003Verdict::Fail;
    }
    if pre_quantize.len() != post_quantize.len() {
        return GateShip003Verdict::Fail;
    }
    if pre_quantize == post_quantize {
        GateShip003Verdict::Pass
    } else {
        GateShip003Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-003 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_003_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-003 algorithm-level PARTIAL discharge: prove
    /// the byte-identity comparison rule binding pre-quantize and
    /// post-quantize Golden Output byte streams to the compound
    /// regression gate. Any edit that relaxes the comparison (case-
    /// insensitive, trim-whitespace, Unicode-fold) or silently accepts
    /// empty input must break this test.
    #[test]
    fn falsify_gate_ship_003_golden_output_byte_identity() {
        // Section 1: byte-identical non-empty → Pass. Baseline.
        let canonical =
            b"```python\ndef fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n```\n";
        assert_eq!(
            verdict_from_golden_output_diff(canonical, canonical),
            GateShip003Verdict::Pass,
            "byte-identical canonical Golden Output must Pass",
        );

        // Section 2: length-mismatch — various drift shapes.
        let shorter = b"```python\ndef fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n``";
        let longer =
            b"```python\ndef fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n```\n\n";
        assert_eq!(
            verdict_from_golden_output_diff(canonical, shorter),
            GateShip003Verdict::Fail,
            "length mismatch (pre longer) must Fail",
        );
        assert_eq!(
            verdict_from_golden_output_diff(shorter, canonical),
            GateShip003Verdict::Fail,
            "length mismatch (post longer) must Fail",
        );
        assert_eq!(
            verdict_from_golden_output_diff(canonical, longer),
            GateShip003Verdict::Fail,
            "length mismatch (trailing newline added) must Fail",
        );

        // Section 3: single-byte flip at various positions — sharpest
        // possible Fail counter-examples. Any mutation that relaxes
        // `==` to "close-enough" or "starts-with" would flip these.
        for flip_pos in [0, 5, canonical.len() / 2, canonical.len() - 1] {
            let mut mutated = canonical.to_vec();
            mutated[flip_pos] ^= 0x01;
            assert_eq!(
                verdict_from_golden_output_diff(canonical, &mutated),
                GateShip003Verdict::Fail,
                "single-byte flip at position {flip_pos} must Fail",
            );
        }

        // Section 4: both-empty — conservative Fail. The rule is "prove
        // no regression"; empty Golden Output on both sides means the
        // `apr qa` gate was SKIPPED (tokenizer missing, feature flag
        // off). We cannot prove no regression without evidence, so we
        // conservatively Fail to block the publish. This mirrors
        // apr-model-qa-v1.yaml FALSIFY-EX-001 (`--require-golden-output`
        // promotes SKIPPED to Fail).
        assert_eq!(
            verdict_from_golden_output_diff(b"", b""),
            GateShip003Verdict::Fail,
            "both-empty must Fail — no Golden Output recorded = no regression proof",
        );

        // Section 5: one-empty — also conservative Fail. A partial
        // SKIP (pre recorded but post not, or vice versa) is still a
        // missing-evidence state.
        assert_eq!(
            verdict_from_golden_output_diff(canonical, b""),
            GateShip003Verdict::Fail,
            "post empty must Fail — missing post-quantize evidence",
        );
        assert_eq!(
            verdict_from_golden_output_diff(b"", canonical),
            GateShip003Verdict::Fail,
            "pre empty must Fail — missing pre-quantize evidence",
        );

        // Section 6: large identical — Pass for a 10_000-byte stream
        // (stress-test the byte-by-byte comparison path; catches any
        // O(1) slice-pointer-equality shortcut that would silently
        // accept aliased-but-not-equal buffers).
        let large: Vec<u8> = (0..10_000).map(|i| (i & 0xFF) as u8).collect();
        let large_copy = large.clone();
        assert_eq!(
            verdict_from_golden_output_diff(&large, &large_copy),
            GateShip003Verdict::Pass,
            "10_000-byte identical streams must Pass (byte-by-byte depth guard)",
        );
        // A mid-stream single-byte flip in the 10_000-byte stream
        // must still Fail.
        let mut large_mutated = large.clone();
        large_mutated[5000] ^= 0x01;
        assert_eq!(
            verdict_from_golden_output_diff(&large, &large_mutated),
            GateShip003Verdict::Fail,
            "mid-stream (idx 5000) single-byte flip must Fail",
        );
    }
}
