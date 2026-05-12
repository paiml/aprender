// SHIP-TWO-001 — `apr-cli-operations-v1` algorithm-level PARTIAL
// discharge for FALSIFY-OPS-005.
//
// Contract: `contracts/apr-cli-operations-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What FALSIFY-OPS-005 says
//
//   rule: Concurrent inference results independent
//   prediction: Result of request A is identical whether B runs
//               concurrently or not
//   test: Run request A alone, then with concurrent B, diff results
//   if_fails: KV cache contamination between concurrent requests
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two stdout byte slices for the same prompt
// — one captured running alone, one captured with a concurrent
// peer — Pass iff:
//
//   alone_output is non-empty AND
//   concurrent_output is non-empty AND
//   alone_output == concurrent_output (byte-identical)
//
// Same shape as `bpe_inv_006` (encode determinism) and `ops_003`
// (greedy determinism), applied to concurrent-vs-isolated
// inference. Catches KV cache contamination, batch-id leakage,
// global-mutex-violation regressions.

/// Binary verdict for `FALSIFY-OPS-005`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ops005Verdict {
    /// Both outputs are non-empty AND byte-identical.
    Pass,
    /// One or more of:
    /// - Either output is empty (caller error — `apr run` silent
    ///   regression).
    /// - Outputs differ in any byte (KV cache contamination,
    ///   batch-id leakage, or other concurrent-state corruption).
    Fail,
}

/// Pure verdict function for `FALSIFY-OPS-005`.
///
/// Inputs:
/// - `alone_output`: stdout from `apr run` invoked alone.
/// - `concurrent_output`: stdout from `apr run` invoked with the
///   same prompt while a peer request is in flight.
///
/// Pass iff both non-empty AND byte-identical.
#[must_use]
pub fn verdict_from_concurrent_isolation(
    alone_output: &[u8],
    concurrent_output: &[u8],
) -> Ops005Verdict {
    if alone_output.is_empty() || concurrent_output.is_empty() {
        return Ops005Verdict::Fail;
    }
    if alone_output == concurrent_output {
        Ops005Verdict::Pass
    } else {
        Ops005Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — concurrent-isolated agreement.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_identical_outputs() {
        let same = b"4";
        let v = verdict_from_concurrent_isolation(same, same);
        assert_eq!(v, Ops005Verdict::Pass);
    }

    #[test]
    fn pass_long_identical_response() {
        let long = vec![b'x'; 5000];
        let v = verdict_from_concurrent_isolation(&long, &long);
        assert_eq!(v, Ops005Verdict::Pass);
    }

    #[test]
    fn pass_realistic_apr_run_arithmetic() {
        let response = b"The answer is 4.";
        let v = verdict_from_concurrent_isolation(response, response);
        assert_eq!(v, Ops005Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — contamination (single-byte drift).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_first_byte_differs() {
        let alone = b"4";
        let concurrent = b"5";
        let v = verdict_from_concurrent_isolation(alone, concurrent);
        assert_eq!(
            v,
            Ops005Verdict::Fail,
            "concurrent contamination must Fail"
        );
    }

    #[test]
    fn fail_kv_cache_leakage_drift() {
        // Realistic regression: KV cache from peer request leaked
        // into A's tail tokens.
        let alone = b"def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)";
        let concurrent = b"def factorial(n):\n    return 1 if n <= 0 else n * factorial(n - 1)";
        let v = verdict_from_concurrent_isolation(alone, concurrent);
        assert_eq!(v, Ops005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — empty inputs.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_alone_empty() {
        let v = verdict_from_concurrent_isolation(&[], b"output");
        assert_eq!(v, Ops005Verdict::Fail);
    }

    #[test]
    fn fail_concurrent_empty() {
        let v = verdict_from_concurrent_isolation(b"output", &[]);
        assert_eq!(v, Ops005Verdict::Fail);
    }

    #[test]
    fn fail_both_empty() {
        let v = verdict_from_concurrent_isolation(&[], &[]);
        assert_eq!(v, Ops005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Symmetry property.
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_is_symmetric_pass() {
        let same = b"identical";
        let v_ab = verdict_from_concurrent_isolation(same, same);
        let v_ba = verdict_from_concurrent_isolation(same, same);
        assert_eq!(v_ab, v_ba);
        assert_eq!(v_ab, Ops005Verdict::Pass);
    }

    #[test]
    fn verdict_is_symmetric_fail() {
        let a = b"foo";
        let b = b"bar";
        let v_ab = verdict_from_concurrent_isolation(a, b);
        let v_ba = verdict_from_concurrent_isolation(b, a);
        assert_eq!(v_ab, v_ba);
        assert_eq!(v_ab, Ops005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Length mismatch.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_alone_longer() {
        let v = verdict_from_concurrent_isolation(b"longer text", b"short");
        assert_eq!(v, Ops005Verdict::Fail);
    }

    #[test]
    fn fail_concurrent_longer() {
        let v = verdict_from_concurrent_isolation(b"short", b"longer text");
        assert_eq!(v, Ops005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — concurrent-inference regression classes.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_batch_id_leakage() {
        // Realistic: batch_id confusion causes A's tokens to come
        // from B's logits.
        let alone = b"hello world";
        let concurrent = b"hello WORLD";
        let v = verdict_from_concurrent_isolation(alone, concurrent);
        assert_eq!(v, Ops005Verdict::Fail);
    }

    #[test]
    fn fail_global_mutex_release_during_softmax() {
        // Realistic: mutex released mid-softmax allows peer to
        // overwrite logits buffer.
        let alone = b"output A";
        let concurrent = b"output X";
        let v = verdict_from_concurrent_isolation(alone, concurrent);
        assert_eq!(v, Ops005Verdict::Fail);
    }
}
