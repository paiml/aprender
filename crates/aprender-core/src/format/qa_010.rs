// SHIP-TWO-001 — `apr-cli-qa-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QA-010.
//
// Contract: `contracts/apr-cli-qa-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI QA gates).
//
// ## What FALSIFY-QA-010 says
//
//   rule: cross-subcommand architecture consistency
//   prediction: "inspect and check report same architecture"
//   test: "diff <(apr inspect --json M | jq .architecture)
//                <(apr check M 2>&1 | grep -i arch)"
//   if_fails: "architecture disagrees across subcommands"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two architecture-string slices extracted
// from `apr inspect --json` and `apr check`, Pass iff:
//
//   inspect_arch is non-empty AND
//   check_arch is non-empty AND
//   inspect_arch == check_arch (byte-identical)
//
// Positive byte-equality verdict (cf. `bpe_inv_006` encode
// determinism). Different subcommands MUST agree on the model
// architecture; disagreement signals a stale parser, a feature-flag
// drift, or a registry mismatch.

/// Binary verdict for `FALSIFY-QA-010`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qa010Verdict {
    /// Both architecture strings are non-empty AND byte-identical.
    Pass,
    /// One or more of:
    /// - Either string is empty (caller error — extraction failed).
    /// - Strings differ in any byte (architecture disagreement).
    Fail,
}

/// Pure verdict function for `FALSIFY-QA-010`.
///
/// Inputs:
/// - `inspect_arch`: architecture string extracted from
///   `apr inspect --json M | jq .architecture`.
/// - `check_arch`: architecture string extracted from
///   `apr check M | grep -i arch`.
///
/// Pass iff both non-empty AND `inspect_arch == check_arch`.
///
/// Otherwise `Fail`.
#[must_use]
pub fn verdict_from_cross_subcmd_arch(
    inspect_arch: &[u8],
    check_arch: &[u8],
) -> Qa010Verdict {
    if inspect_arch.is_empty() || check_arch.is_empty() {
        return Qa010Verdict::Fail;
    }
    if inspect_arch == check_arch {
        Qa010Verdict::Pass
    } else {
        Qa010Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — agreement at canonical architectures.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_qwen2_agreement() {
        let v = verdict_from_cross_subcmd_arch(b"qwen2", b"qwen2");
        assert_eq!(v, Qa010Verdict::Pass);
    }

    #[test]
    fn pass_llama_agreement() {
        let v = verdict_from_cross_subcmd_arch(b"llama", b"llama");
        assert_eq!(v, Qa010Verdict::Pass);
    }

    #[test]
    fn pass_qwen3_moe_agreement() {
        let v = verdict_from_cross_subcmd_arch(b"qwen3-moe", b"qwen3-moe");
        assert_eq!(v, Qa010Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — architecture disagreement.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_qwen2_vs_qwen3() {
        let v = verdict_from_cross_subcmd_arch(b"qwen2", b"qwen3");
        assert_eq!(
            v,
            Qa010Verdict::Fail,
            "different architectures must Fail"
        );
    }

    #[test]
    fn fail_llama_vs_qwen() {
        let v = verdict_from_cross_subcmd_arch(b"llama", b"qwen2");
        assert_eq!(v, Qa010Verdict::Fail);
    }

    #[test]
    fn fail_one_byte_difference() {
        let v = verdict_from_cross_subcmd_arch(b"qwen2", b"qwen3");
        assert_eq!(v, Qa010Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — case mismatch (case-sensitive).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_case_difference() {
        let v = verdict_from_cross_subcmd_arch(b"qwen2", b"QWEN2");
        assert_eq!(
            v,
            Qa010Verdict::Fail,
            "case mismatch must Fail (one subcmd normalized, other didn't)"
        );
    }

    #[test]
    fn fail_inspect_lowercase_check_titlecase() {
        let v = verdict_from_cross_subcmd_arch(b"llama", b"Llama");
        assert_eq!(v, Qa010Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — whitespace / formatting differences.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_trailing_whitespace() {
        let v = verdict_from_cross_subcmd_arch(b"qwen2", b"qwen2 ");
        assert_eq!(v, Qa010Verdict::Fail);
    }

    #[test]
    fn fail_leading_whitespace() {
        let v = verdict_from_cross_subcmd_arch(b"qwen2", b" qwen2");
        assert_eq!(v, Qa010Verdict::Fail);
    }

    #[test]
    fn fail_with_quotes() {
        // jq sometimes emits values with quotes; `grep` strips them.
        let v = verdict_from_cross_subcmd_arch(b"\"qwen2\"", b"qwen2");
        assert_eq!(v, Qa010Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — empty inputs.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_inspect_arch_empty() {
        let v = verdict_from_cross_subcmd_arch(&[], b"qwen2");
        assert_eq!(v, Qa010Verdict::Fail);
    }

    #[test]
    fn fail_check_arch_empty() {
        let v = verdict_from_cross_subcmd_arch(b"qwen2", &[]);
        assert_eq!(v, Qa010Verdict::Fail);
    }

    #[test]
    fn fail_both_empty() {
        let v = verdict_from_cross_subcmd_arch(&[], &[]);
        assert_eq!(
            v,
            Qa010Verdict::Fail,
            "both empty must Fail (vacuous Pass refused)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Symmetry — verdict is symmetric in (inspect, check).
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_is_symmetric_pass() {
        let v_ic = verdict_from_cross_subcmd_arch(b"qwen2", b"qwen2");
        let v_ci = verdict_from_cross_subcmd_arch(b"qwen2", b"qwen2");
        assert_eq!(v_ic, v_ci);
        assert_eq!(v_ic, Qa010Verdict::Pass);
    }

    #[test]
    fn verdict_is_symmetric_fail() {
        let v_ic = verdict_from_cross_subcmd_arch(b"qwen2", b"llama");
        let v_ci = verdict_from_cross_subcmd_arch(b"llama", b"qwen2");
        assert_eq!(v_ic, v_ci);
        assert_eq!(v_ic, Qa010Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full architecture string family.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_realistic_full_architecture_strings() {
        let v = verdict_from_cross_subcmd_arch(
            b"qwen2.5-coder-7b",
            b"qwen2.5-coder-7b",
        );
        assert_eq!(v, Qa010Verdict::Pass);
    }

    #[test]
    fn fail_realistic_inspect_full_check_short() {
        // Common drift: inspect emits full name, check emits short.
        let v = verdict_from_cross_subcmd_arch(
            b"qwen2.5-coder-7b",
            b"qwen2",
        );
        assert_eq!(v, Qa010Verdict::Fail);
    }

    #[test]
    fn fail_inspect_says_qwen3_check_says_qwen2() {
        // Worst case: stale check parser gives wrong arch.
        let v = verdict_from_cross_subcmd_arch(b"qwen3-moe", b"qwen2");
        assert_eq!(v, Qa010Verdict::Fail);
    }

    #[test]
    fn pass_albor_llama_370m() {
        let v = verdict_from_cross_subcmd_arch(
            b"albor-llama-370m",
            b"albor-llama-370m",
        );
        assert_eq!(v, Qa010Verdict::Pass);
    }
}
