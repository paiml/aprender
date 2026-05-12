// SHIP-TWO-001 — `apr-cli-operations-v1` algorithm-level PARTIAL
// discharge for FALSIFY-OPS-003.
//
// Contract: `contracts/apr-cli-operations-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What FALSIFY-OPS-003 says
//
//   rule: Greedy decoding is deterministic
//   prediction: Two runs with temperature=0 produce identical output
//   test: Run same prompt twice with temperature=0, assert output equality
//   if_fails: Non-deterministic codepath in greedy sampling
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two stdout byte slices from
// `apr run --temperature 0 ...` invoked twice with the same
// prompt, Pass iff:
//
//   run_a is non-empty AND
//   run_b is non-empty AND
//   run_a == run_b (byte-identical)
//
// Same shape as `bpe_inv_006` (encode determinism), applied to
// the inference output instead of token IDs. Catches:
// - HashMap iteration order in argmax-tiebreak.
// - Race condition in multi-threaded sampling.
// - Stochastic codepath that ignores temperature=0.

/// Binary verdict for `FALSIFY-OPS-003`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ops003Verdict {
    /// Both outputs are non-empty AND byte-identical.
    Pass,
    /// One or more of:
    /// - Either output is empty (caller error — `apr run` silent
    ///   regression).
    /// - Outputs differ in any byte (non-deterministic codepath
    ///   in greedy sampling).
    Fail,
}

/// Pure verdict function for `FALSIFY-OPS-003`.
///
/// Inputs:
/// - `run_a`: stdout bytes from first `apr run --temperature 0`
///   invocation.
/// - `run_b`: stdout bytes from second `apr run --temperature 0`
///   invocation with the same prompt.
///
/// Pass iff:
/// 1. `!run_a.is_empty()`,
/// 2. `!run_b.is_empty()`,
/// 3. `run_a == run_b` (byte-identical).
///
/// Otherwise `Fail`.
#[must_use]
pub fn verdict_from_greedy_decoding_pair(
    run_a: &[u8],
    run_b: &[u8],
) -> Ops003Verdict {
    if run_a.is_empty() || run_b.is_empty() {
        return Ops003Verdict::Fail;
    }
    if run_a == run_b {
        Ops003Verdict::Pass
    } else {
        Ops003Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — identical greedy outputs.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_short_identical_output() {
        let a = b"4\n";
        let b = b"4\n";
        let v = verdict_from_greedy_decoding_pair(a, b);
        assert_eq!(v, Ops003Verdict::Pass);
    }

    #[test]
    fn pass_realistic_2_plus_2_response() {
        // Canonical apr run --temperature 0 'What is 2+2?' output.
        let response = b"4";
        let v = verdict_from_greedy_decoding_pair(response, response);
        assert_eq!(v, Ops003Verdict::Pass);
    }

    #[test]
    fn pass_long_identical_response() {
        let long = vec![b'x'; 10_000];
        let v = verdict_from_greedy_decoding_pair(&long, &long);
        assert_eq!(v, Ops003Verdict::Pass);
    }

    #[test]
    fn pass_with_special_chars() {
        let a = b"```python\nprint('hello')\n```";
        let v = verdict_from_greedy_decoding_pair(a, a);
        assert_eq!(v, Ops003Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — single-byte drift (non-determinism).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_first_byte_differs() {
        let a = b"4\n";
        let b = b"5\n";
        let v = verdict_from_greedy_decoding_pair(a, b);
        assert_eq!(
            v,
            Ops003Verdict::Fail,
            "single-byte drift must Fail (non-determinism)"
        );
    }

    #[test]
    fn fail_last_byte_differs() {
        let a = b"def foo():\n    pass\n";
        let b = b"def foo():\n    pass ";
        let v = verdict_from_greedy_decoding_pair(a, b);
        assert_eq!(v, Ops003Verdict::Fail);
    }

    #[test]
    fn fail_middle_byte_differs() {
        let a = b"hello world";
        let b = b"hellp world";
        let v = verdict_from_greedy_decoding_pair(a, b);
        assert_eq!(v, Ops003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — length mismatch.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_a_longer() {
        let a = b"hello world";
        let b = b"hello";
        let v = verdict_from_greedy_decoding_pair(a, b);
        assert_eq!(v, Ops003Verdict::Fail);
    }

    #[test]
    fn fail_b_longer() {
        let a = b"hello";
        let b = b"hello world";
        let v = verdict_from_greedy_decoding_pair(a, b);
        assert_eq!(v, Ops003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — empty inputs.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_a_empty() {
        let v = verdict_from_greedy_decoding_pair(&[], b"output");
        assert_eq!(v, Ops003Verdict::Fail);
    }

    #[test]
    fn fail_b_empty() {
        let v = verdict_from_greedy_decoding_pair(b"output", &[]);
        assert_eq!(v, Ops003Verdict::Fail);
    }

    #[test]
    fn fail_both_empty() {
        let v = verdict_from_greedy_decoding_pair(&[], &[]);
        assert_eq!(
            v,
            Ops003Verdict::Fail,
            "both empty must Fail (vacuous Pass refused)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Symmetry property.
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_is_symmetric_pass() {
        let same = b"identical";
        let v_ab = verdict_from_greedy_decoding_pair(same, same);
        let v_ba = verdict_from_greedy_decoding_pair(same, same);
        assert_eq!(v_ab, v_ba);
        assert_eq!(v_ab, Ops003Verdict::Pass);
    }

    #[test]
    fn verdict_is_symmetric_fail() {
        let a = b"foo";
        let b = b"bar";
        let v_ab = verdict_from_greedy_decoding_pair(a, b);
        let v_ba = verdict_from_greedy_decoding_pair(b, a);
        assert_eq!(v_ab, v_ba);
        assert_eq!(v_ab, Ops003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Position sweep — drift at every position must Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_at_every_drift_position() {
        let baseline = b"the quick brown fox jumps over the lazy dog";
        for pos in 0..baseline.len() {
            let mut drift = baseline.to_vec();
            drift[pos] ^= 0x01;
            let v = verdict_from_greedy_decoding_pair(baseline, &drift);
            assert_eq!(
                v,
                Ops003Verdict::Fail,
                "drift at position {pos} must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — apr run scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_apr_run_arithmetic_same_answer() {
        // Run twice with --temperature 0; both produce "4".
        let response = b"4";
        let v = verdict_from_greedy_decoding_pair(response, response);
        assert_eq!(v, Ops003Verdict::Pass);
    }

    #[test]
    fn fail_apr_run_hashmap_tiebreak_drift() {
        // Realistic regression: argmax tiebreak picks different
        // token id between runs due to HashMap iteration order.
        let run_a = b"The answer is 4";
        let run_b = b"The answer is 5"; // tiebreak chose differently
        let v = verdict_from_greedy_decoding_pair(run_a, run_b);
        assert_eq!(
            v,
            Ops003Verdict::Fail,
            "argmax tiebreak drift must Fail"
        );
    }

    #[test]
    fn fail_apr_run_floating_point_associativity() {
        // Realistic regression: parallel logit reduction order
        // varies between runs, producing different argmax.
        let run_a = b"def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)";
        let run_b = b"def factorial(n):\n    return 1 if n <= 0 else n * factorial(n - 1)";
        let v = verdict_from_greedy_decoding_pair(run_a, run_b);
        assert_eq!(v, Ops003Verdict::Fail);
    }
}
