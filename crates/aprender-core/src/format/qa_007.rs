// SHIP-TWO-001 — `apr-cli-qa-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QA-007.
//
// Contract: `contracts/apr-cli-qa-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI QA gates).
//
// ## What FALSIFY-QA-007 says
//
//   rule: --json flag changes output
//   prediction: "json output differs from default"
//   test: "diff <(apr inspect model) <(apr inspect --json model) |
//          grep -q ."
//   if_fails: "--json flag is a no-op"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two byte slices (`default_output`,
// `json_output`) from running the same apr command twice (once
// without --json, once with), Pass iff:
//
//   default_output is non-empty AND
//   json_output is non-empty AND
//   default_output != json_output
//
// Inverse-equality verdict: the two outputs MUST differ. Same
// shape catches the regression where --json is parsed but doesn't
// route to a different formatter.

/// Binary verdict for `FALSIFY-QA-007`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qa007Verdict {
    /// Both outputs are non-empty AND `default_output != json_output`.
    Pass,
    /// One or more of:
    /// - `default_output.is_empty()` (caller error — apr without
    ///   --json silently failed).
    /// - `json_output.is_empty()` (caller error — apr with --json
    ///   silently failed).
    /// - `default_output == json_output` (regression — --json
    ///   flag is a no-op; both formats route to the same writer).
    Fail,
}

/// Pure verdict function for `FALSIFY-QA-007`.
///
/// Inputs:
/// - `default_output`: stdout bytes from `apr <subcmd> <args>`.
/// - `json_output`: stdout bytes from `apr <subcmd> --json <args>`.
///
/// Pass iff:
/// 1. `!default_output.is_empty()`,
/// 2. `!json_output.is_empty()`,
/// 3. `default_output != json_output`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Default and JSON outputs differ (typical) — `Pass`:
/// ```
/// use aprender::format::qa_007::{
///     verdict_from_json_flag_diff, Qa007Verdict,
/// };
/// let default_out = b"Model: qwen2.5-coder\nTensors: 339\n";
/// let json_out = b"{\"model\":\"qwen2.5-coder\",\"tensors\":339}\n";
/// let v = verdict_from_json_flag_diff(default_out, json_out);
/// assert_eq!(v, Qa007Verdict::Pass);
/// ```
///
/// --json flag is a no-op (regression) — `Fail`:
/// ```
/// use aprender::format::qa_007::{
///     verdict_from_json_flag_diff, Qa007Verdict,
/// };
/// let same = b"Model: qwen2.5-coder\nTensors: 339\n";
/// let v = verdict_from_json_flag_diff(same, same);
/// assert_eq!(v, Qa007Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_json_flag_diff(
    default_output: &[u8],
    json_output: &[u8],
) -> Qa007Verdict {
    if default_output.is_empty() || json_output.is_empty() {
        return Qa007Verdict::Fail;
    }
    if default_output != json_output {
        Qa007Verdict::Pass
    } else {
        Qa007Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — default and JSON outputs differ.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_realistic_inspect_default_vs_json() {
        let default_out = b"Model: qwen2.5-coder-7b\nTensors: 339\nQuant: Q4_K\n";
        let json_out = b"{\"model\":\"qwen2.5-coder-7b\",\"tensors\":339,\"quant\":\"Q4_K\"}\n";
        let v = verdict_from_json_flag_diff(default_out, json_out);
        assert_eq!(v, Qa007Verdict::Pass);
    }

    #[test]
    fn pass_one_byte_difference() {
        // Smallest possible difference still passes.
        let default_out = b"Model: A";
        let json_out = b"Model: B";
        let v = verdict_from_json_flag_diff(default_out, json_out);
        assert_eq!(v, Qa007Verdict::Pass);
    }

    #[test]
    fn pass_completely_different_lengths() {
        let default_out = b"short";
        let json_out = b"a much longer JSON-formatted output that wraps the data";
        let v = verdict_from_json_flag_diff(default_out, json_out);
        assert_eq!(v, Qa007Verdict::Pass);
    }

    #[test]
    fn pass_realistic_apr_qa_outputs() {
        // `apr qa model` text vs `apr qa --json model` JSON.
        let default_out = b"  PASS  T-001: integrity\n  PASS  T-002: signature\n";
        let json_out = b"{\"results\":[{\"id\":\"T-001\",\"verdict\":\"PASS\"},{\"id\":\"T-002\",\"verdict\":\"PASS\"}]}\n";
        let v = verdict_from_json_flag_diff(default_out, json_out);
        assert_eq!(v, Qa007Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — --json is a no-op (the regression).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_json_flag_no_op() {
        // Both outputs identical: --json flag was parsed but
        // ignored.
        let same = b"Model: qwen2.5-coder\nTensors: 339\n";
        let v = verdict_from_json_flag_diff(same, same);
        assert_eq!(
            v,
            Qa007Verdict::Fail,
            "--json no-op must Fail (both outputs identical)"
        );
    }

    #[test]
    fn fail_byte_identical_long_output() {
        // Long but identical output.
        let same = b"\
{\"model\":\"qwen2.5-coder-7b-apache-q4k-v1\",\"tensors\":339,\"layers\":28,\"hidden\":3584,\"heads\":28,\"kv_heads\":4,\"vocab\":152064}
";
        let v = verdict_from_json_flag_diff(same, same);
        assert_eq!(v, Qa007Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — empty inputs.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_default_empty() {
        let v = verdict_from_json_flag_diff(&[], b"some json");
        assert_eq!(
            v,
            Qa007Verdict::Fail,
            "empty default output must Fail (apr silent regression)"
        );
    }

    #[test]
    fn fail_json_empty() {
        let v = verdict_from_json_flag_diff(b"some text", &[]);
        assert_eq!(
            v,
            Qa007Verdict::Fail,
            "empty json output must Fail"
        );
    }

    #[test]
    fn fail_both_empty() {
        let v = verdict_from_json_flag_diff(&[], &[]);
        assert_eq!(
            v,
            Qa007Verdict::Fail,
            "both empty must Fail (vacuous Pass refused)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Edge — single-byte non-trivial differences.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_one_byte_appended_to_json() {
        // JSON output has trailing newline that default lacks.
        let default_out = b"output";
        let json_out = b"output\n";
        let v = verdict_from_json_flag_diff(default_out, json_out);
        assert_eq!(v, Qa007Verdict::Pass);
    }

    #[test]
    fn pass_one_byte_prepended() {
        let default_out = b"data";
        let json_out = b"{data";
        let v = verdict_from_json_flag_diff(default_out, json_out);
        assert_eq!(v, Qa007Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 5: Symmetry — verdict is symmetric in (default, json).
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_is_symmetric_pass() {
        let a = b"text format";
        let b = b"{\"json\":true}";
        let ab = verdict_from_json_flag_diff(a, b);
        let ba = verdict_from_json_flag_diff(b, a);
        assert_eq!(ab, ba);
        assert_eq!(ab, Qa007Verdict::Pass);
    }

    #[test]
    fn verdict_is_symmetric_fail_identical() {
        let same = b"identical";
        let v1 = verdict_from_json_flag_diff(same, same);
        let v2 = verdict_from_json_flag_diff(same, same);
        assert_eq!(v1, v2);
        assert_eq!(v1, Qa007Verdict::Fail);
    }

    #[test]
    fn verdict_is_symmetric_fail_one_empty() {
        let real = b"data";
        let ae = verdict_from_json_flag_diff(real, &[]);
        let eb = verdict_from_json_flag_diff(&[], real);
        assert_eq!(ae, eb);
        assert_eq!(ae, Qa007Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Domain — inverse-equality property.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_outputs_differ_at_canonical_lengths() {
        let lengths: Vec<usize> = vec![1, 10, 100, 1000];
        for len in lengths {
            let a = vec![b'a'; len];
            let b = vec![b'b'; len];
            let v_diff = verdict_from_json_flag_diff(&a, &b);
            assert_eq!(v_diff, Qa007Verdict::Pass, "len={len} differing");

            let v_same = verdict_from_json_flag_diff(&a, &a);
            assert_eq!(v_same, Qa007Verdict::Fail, "len={len} identical");
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — 3 apr subcommands' default vs --json contrast.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_apr_inspect_text_vs_json() {
        // Per QA-007 contract test wording.
        let default_out = b"Model: qwen2.5-coder-7b-apache-q4k-v1\nTensors: 339\n";
        let json_out = b"{\"model\":\"qwen2.5-coder-7b-apache-q4k-v1\",\"tensors\":339}";
        let v = verdict_from_json_flag_diff(default_out, json_out);
        assert_eq!(v, Qa007Verdict::Pass);
    }

    #[test]
    fn pass_apr_diff_text_vs_json() {
        let default_out = b"Tensor 0: cosine 0.9999\nTensor 1: cosine 0.9998\n";
        let json_out = b"{\"tensors\":[{\"idx\":0,\"cos\":0.9999},{\"idx\":1,\"cos\":0.9998}]}";
        let v = verdict_from_json_flag_diff(default_out, json_out);
        assert_eq!(v, Qa007Verdict::Pass);
    }

    #[test]
    fn fail_apr_validate_json_unimplemented() {
        // Hypothetical regression: `apr validate --json` was added
        // to the CLI but the formatter wasn't routed; both outputs
        // print the text format.
        let same = b"VALIDATION: PASS\n";
        let v = verdict_from_json_flag_diff(same, same);
        assert_eq!(
            v,
            Qa007Verdict::Fail,
            "--json formatter unimplemented must Fail"
        );
    }
}
