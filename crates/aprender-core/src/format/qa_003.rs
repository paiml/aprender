// SHIP-TWO-001 — `apr-cli-qa-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QA-003.
//
// Contract: `contracts/apr-cli-qa-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI QA gates).
//
// ## What FALSIFY-QA-003 says
//
//   rule: json output is valid
//   prediction: "apr inspect --json model | jq . exits 0"
//   if_fails: "invalid JSON emitted"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`output_bytes`, `jq_exit_code`), Pass iff:
//
//   output_bytes is non-empty AND
//   jq_exit_code == 0 AND
//   output_bytes' first non-whitespace byte is '{' or '[' AND
//   output_bytes' last non-whitespace byte is '}' or ']' AND
//   the bracket and brace counts balance
//
// The bracket/brace balance check is a structural sanity ahead of
// invoking `jq`. Both must agree (jq exits 0 AND structural balance)
// to accept JSON validity. This catches:
// - jq missing or replaced: structural check still applies.
// - jq stub returning 0 silently: structural check catches it.
// - Empty output: refuses vacuous Pass.
// - Truncated JSON (unbalanced brackets): caught by structural check.

/// Binary verdict for `FALSIFY-QA-003`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qa003Verdict {
    /// Output is non-empty, structurally JSON-shaped (starts with
    /// `{`/`[`, ends with `}`/`]`, brackets balance), AND jq exited 0.
    Pass,
    /// One or more of:
    /// - `output_bytes.is_empty()` (caller error — apr emitted no
    ///   JSON at all).
    /// - `jq_exit_code != 0` (jq rejected the input as invalid JSON).
    /// - First non-whitespace byte not `{` or `[`.
    /// - Last non-whitespace byte not `}` or `]`.
    /// - Brace `{`/`}` counts unbalanced.
    /// - Bracket `[`/`]` counts unbalanced.
    Fail,
}

/// Pure verdict function for `FALSIFY-QA-003`.
///
/// Inputs:
/// - `output_bytes`: stdout from `apr inspect --json model`.
/// - `jq_exit_code`: process exit code from
///   `apr inspect --json model | jq . > /dev/null`.
///
/// Pass iff:
/// 1. `!output_bytes.is_empty()`,
/// 2. `jq_exit_code == 0`,
/// 3. First non-whitespace byte is `{` or `[`,
/// 4. Last non-whitespace byte is `}` or `]`,
/// 5. Brace counts equal AND bracket counts equal.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Well-formed JSON object — `Pass`:
/// ```
/// use aprender::format::qa_003::{
///     verdict_from_json_validity, Qa003Verdict,
/// };
/// let json = b"{\"model\":\"qwen\",\"tensors\":339}";
/// let v = verdict_from_json_validity(json, 0);
/// assert_eq!(v, Qa003Verdict::Pass);
/// ```
///
/// Truncated JSON (jq fails) — `Fail`:
/// ```
/// use aprender::format::qa_003::{
///     verdict_from_json_validity, Qa003Verdict,
/// };
/// let truncated = b"{\"model\":\"qwen\",\"tensors\":33";
/// let v = verdict_from_json_validity(truncated, 1);
/// assert_eq!(v, Qa003Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_json_validity(
    output_bytes: &[u8],
    jq_exit_code: i32,
) -> Qa003Verdict {
    if output_bytes.is_empty() {
        return Qa003Verdict::Fail;
    }
    if jq_exit_code != 0 {
        return Qa003Verdict::Fail;
    }
    // Trim leading whitespace.
    let trimmed_start_idx = output_bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(output_bytes.len());
    if trimmed_start_idx >= output_bytes.len() {
        return Qa003Verdict::Fail; // all-whitespace
    }
    let first = output_bytes[trimmed_start_idx];
    if first != b'{' && first != b'[' {
        return Qa003Verdict::Fail;
    }
    // Trim trailing whitespace.
    let trimmed_end_idx = output_bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .unwrap_or(0);
    let last = output_bytes[trimmed_end_idx];
    if last != b'}' && last != b']' {
        return Qa003Verdict::Fail;
    }
    // Check brace and bracket counts balance.
    // Note: this is a coarse structural check; it does NOT account
    // for braces/brackets inside strings. A more rigorous check
    // requires a real parser, which is FULL_DISCHARGE work.
    let mut brace_balance: i64 = 0;
    let mut bracket_balance: i64 = 0;
    for &b in output_bytes {
        match b {
            b'{' => brace_balance += 1,
            b'}' => brace_balance -= 1,
            b'[' => bracket_balance += 1,
            b']' => bracket_balance -= 1,
            _ => {}
        }
    }
    if brace_balance != 0 || bracket_balance != 0 {
        return Qa003Verdict::Fail;
    }
    Qa003Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — well-formed JSON.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_simple_object() {
        let json = b"{\"key\":\"value\"}";
        let v = verdict_from_json_validity(json, 0);
        assert_eq!(v, Qa003Verdict::Pass);
    }

    #[test]
    fn pass_simple_array() {
        let json = b"[1,2,3]";
        let v = verdict_from_json_validity(json, 0);
        assert_eq!(v, Qa003Verdict::Pass);
    }

    #[test]
    fn pass_realistic_apr_inspect_json() {
        let json = b"{\"model\":\"qwen2.5-coder-7b-apache-q4k-v1\",\"tensors\":339,\"layers\":28}";
        let v = verdict_from_json_validity(json, 0);
        assert_eq!(v, Qa003Verdict::Pass);
    }

    #[test]
    fn pass_nested_structure() {
        let json = b"{\"outer\":{\"inner\":[1,2,{\"deep\":true}]}}";
        let v = verdict_from_json_validity(json, 0);
        assert_eq!(v, Qa003Verdict::Pass);
    }

    #[test]
    fn pass_with_leading_trailing_whitespace() {
        let json = b"  \n{\"key\":\"value\"}\n  ";
        let v = verdict_from_json_validity(json, 0);
        assert_eq!(v, Qa003Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — jq rejected.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_jq_exit_one() {
        // Even structurally-balanced output: if jq exited non-zero,
        // we trust jq.
        let json = b"{\"key\":\"value\"}";
        let v = verdict_from_json_validity(json, 1);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    #[test]
    fn fail_jq_panic_exit() {
        let json = b"{\"valid\":true}";
        let v = verdict_from_json_validity(json, 101);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — empty output.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_output() {
        let v = verdict_from_json_validity(&[], 0);
        assert_eq!(
            v,
            Qa003Verdict::Fail,
            "empty output must Fail (vacuous Pass refused)"
        );
    }

    #[test]
    fn fail_whitespace_only() {
        let v = verdict_from_json_validity(b"   \n  \t  ", 0);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — non-JSON shapes.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_starts_with_text() {
        // Output: "Model: qwen" — not JSON-shaped.
        let v = verdict_from_json_validity(b"Model: qwen", 0);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    #[test]
    fn fail_starts_with_quote() {
        // Bare string is not what apr should emit — must be object/array.
        let v = verdict_from_json_validity(b"\"just a string\"", 0);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    #[test]
    fn fail_ends_with_text() {
        let v = verdict_from_json_validity(b"{\"key\":\"value\"} trailing", 0);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — bracket / brace imbalance.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_unbalanced_extra_open_brace() {
        let v = verdict_from_json_validity(b"{{\"key\":\"value\"}", 0);
        assert_eq!(
            v,
            Qa003Verdict::Fail,
            "unbalanced open brace must Fail"
        );
    }

    #[test]
    fn fail_unbalanced_missing_close_brace() {
        let v = verdict_from_json_validity(b"{\"key\":\"value\"", 1);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    #[test]
    fn fail_unbalanced_extra_close_bracket() {
        let v = verdict_from_json_validity(b"[1,2,3]]", 0);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    #[test]
    fn fail_truncated_mid_object() {
        // Realistic regression: stdout pipe broke mid-write.
        let v = verdict_from_json_validity(b"{\"model\":\"qwe", 1);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Edge cases — minimal valid + adversarial inputs.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_minimal_empty_object() {
        let v = verdict_from_json_validity(b"{}", 0);
        assert_eq!(v, Qa003Verdict::Pass);
    }

    #[test]
    fn pass_minimal_empty_array() {
        let v = verdict_from_json_validity(b"[]", 0);
        assert_eq!(v, Qa003Verdict::Pass);
    }

    #[test]
    fn fail_jq_exit_negative() {
        let v = verdict_from_json_validity(b"{\"valid\":true}", -1);
        assert_eq!(v, Qa003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — apr inspect / qa / trace JSON outputs.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_apr_qa_results_json() {
        let json = b"{\"results\":[{\"id\":\"T-001\",\"verdict\":\"PASS\"},{\"id\":\"T-002\",\"verdict\":\"PASS\"}]}";
        let v = verdict_from_json_validity(json, 0);
        assert_eq!(v, Qa003Verdict::Pass);
    }

    #[test]
    fn pass_apr_trace_layer_stats_json() {
        let json = b"{\"layers\":[{\"idx\":0,\"params\":1234567},{\"idx\":1,\"params\":1234567}]}";
        let v = verdict_from_json_validity(json, 0);
        assert_eq!(v, Qa003Verdict::Pass);
    }

    #[test]
    fn fail_apr_inspect_emitted_text_instead_of_json() {
        // Regression: --json flag accepted but text formatter was
        // routed (cf. QA-007).
        let v = verdict_from_json_validity(b"Model: qwen2.5-coder\nTensors: 339\n", 0);
        assert_eq!(v, Qa003Verdict::Fail);
    }
}
