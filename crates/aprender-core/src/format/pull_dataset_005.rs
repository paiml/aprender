// SHIP-TWO-001 MODEL-2 — `apr-cli-pull-dataset-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-PULL-DATASET-005.
//
// Contract: `contracts/apr-cli-pull-dataset-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pull (P1.1).
//
// ## What FALSIFY-APR-PULL-DATASET-005 says
//
//   rule: model-path backward compatible
//   prediction: "`apr pull paiml/qwen2.5-coder-7b-apache-q4k-v1`
//                (model path, no dataset asset-type) still works"
//   test: "apr pull paiml/qwen2.5-coder-7b-apache-q4k-v1 --dry-run
//          | grep -q 'qwen2.5-coder-7b-apache-q4k-v1'"
//   if_fails: "extension regressed existing model puller — breaks
//              AC-EX-005, AC-SHIP1-006"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given the bytes of stdout from
// `apr pull <model> --dry-run` and the expected model-name
// substring, Pass iff:
//
//   stdout is non-empty AND
//   stdout contains the model_name as a substring (utf-8) AND
//   model_name is non-empty
//
// Substring containment (not equality) matches the contract's
// `grep -q 'qwen2.5-coder-7b-apache-q4k-v1'` test wording.
// Empty inputs refused as caller error.

/// Binary verdict for `FALSIFY-APR-PULL-DATASET-005`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDataset005Verdict {
    /// Both inputs are non-empty AND `dry_run_stdout` contains
    /// `model_name` as a (UTF-8) substring.
    Pass,
    /// One or more of:
    /// - `dry_run_stdout.is_empty()` (caller error — `apr pull`
    ///   failed to emit anything; almost certainly regressed).
    /// - `model_name.is_empty()` (caller error — no name to match).
    /// - `dry_run_stdout` does NOT contain `model_name`
    ///   (the regression: model-path puller dropped the name from
    ///   its dry-run output, signaling the legacy code path is
    ///   broken).
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-PULL-DATASET-005`.
///
/// Inputs:
/// - `dry_run_stdout`: stdout bytes captured from
///   `apr pull <model> --dry-run`.
/// - `model_name`: the expected substring (e.g.,
///   `b"qwen2.5-coder-7b-apache-q4k-v1"`).
///
/// Pass iff:
/// 1. `!dry_run_stdout.is_empty()`,
/// 2. `!model_name.is_empty()`,
/// 3. `dry_run_stdout` contains `model_name` as a contiguous byte
///    subsequence.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Stdout contains the model name — `Pass`:
/// ```
/// use aprender::format::pull_dataset_005::{
///     verdict_from_model_path_backward_compat, PullDataset005Verdict,
/// };
/// let stdout = b"DRY RUN: would pull paiml/qwen2.5-coder-7b-apache-q4k-v1 to ~/.cache/apr/";
/// let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
/// assert_eq!(v, PullDataset005Verdict::Pass);
/// ```
///
/// Stdout missing the model name (legacy puller regressed) — `Fail`:
/// ```
/// use aprender::format::pull_dataset_005::{
///     verdict_from_model_path_backward_compat, PullDataset005Verdict,
/// };
/// let stdout = b"ERROR: dataset asset-type required";
/// let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
/// assert_eq!(v, PullDataset005Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_model_path_backward_compat(
    dry_run_stdout: &[u8],
    model_name: &[u8],
) -> PullDataset005Verdict {
    if dry_run_stdout.is_empty() || model_name.is_empty() {
        return PullDataset005Verdict::Fail;
    }
    if contains_subsequence(dry_run_stdout, model_name) {
        PullDataset005Verdict::Pass
    } else {
        PullDataset005Verdict::Fail
    }
}

/// Returns `true` iff `needle` appears as a contiguous subsequence
/// of `haystack`. Pure, allocation-free.
#[must_use]
fn contains_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — canonical substring matches.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_canonical_qwen_in_dry_run_output() {
        let stdout = b"DRY RUN: would pull paiml/qwen2.5-coder-7b-apache-q4k-v1 to ~/.cache/apr/models/";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(v, PullDataset005Verdict::Pass);
    }

    #[test]
    fn pass_albor_llama_model_name() {
        let stdout = b"Pulling paiml/albor-llama-370m-python-v1 (1.5 GB)";
        let v = verdict_from_model_path_backward_compat(stdout, b"albor-llama-370m-python-v1");
        assert_eq!(v, PullDataset005Verdict::Pass);
    }

    #[test]
    fn pass_needle_at_start() {
        let stdout = b"qwen2.5-coder-7b-apache-q4k-v1 is being pulled";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(v, PullDataset005Verdict::Pass);
    }

    #[test]
    fn pass_needle_at_end() {
        let stdout = b"Pulling model: qwen2.5-coder-7b-apache-q4k-v1";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(v, PullDataset005Verdict::Pass);
    }

    #[test]
    fn pass_needle_equals_haystack() {
        let v = verdict_from_model_path_backward_compat(b"qwen", b"qwen");
        assert_eq!(v, PullDataset005Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — needle missing (regression).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_legacy_puller_regressed() {
        // The exact regression: dataset asset-type now required.
        let stdout = b"ERROR: dataset asset-type required";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(
            v,
            PullDataset005Verdict::Fail,
            "legacy puller error message must Fail"
        );
    }

    #[test]
    fn fail_completely_unrelated_output() {
        let stdout = b"hello world";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen");
        assert_eq!(v, PullDataset005Verdict::Fail);
    }

    #[test]
    fn fail_partial_substring_only() {
        // "qwen2.5-coder" appears but the full model name does not.
        let stdout = b"Pulling qwen2.5-coder-something-else";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(v, PullDataset005Verdict::Fail);
    }

    #[test]
    fn fail_one_byte_off() {
        // Off-by-one in the model name: "v2" vs "v1".
        let stdout = b"Pulling qwen2.5-coder-7b-apache-q4k-v2";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(v, PullDataset005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — empty inputs.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_stdout() {
        let v = verdict_from_model_path_backward_compat(&[], b"qwen");
        assert_eq!(
            v,
            PullDataset005Verdict::Fail,
            "empty stdout must Fail (apr pull silent)"
        );
    }

    #[test]
    fn fail_empty_model_name() {
        let v = verdict_from_model_path_backward_compat(b"some output", &[]);
        assert_eq!(v, PullDataset005Verdict::Fail);
    }

    #[test]
    fn fail_both_empty() {
        let v = verdict_from_model_path_backward_compat(&[], &[]);
        assert_eq!(v, PullDataset005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — needle longer than haystack.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_needle_longer_than_haystack() {
        // 32-byte name, 5-byte stdout.
        let v = verdict_from_model_path_backward_compat(b"short", b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(v, PullDataset005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Edge — case-sensitivity (model names are case-sensitive on HF).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_case_mismatch() {
        // HF model paths are case-sensitive; "QWEN" != "qwen".
        let stdout = b"Pulling QWEN2.5-CODER-7B-APACHE-Q4K-V1";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(
            v,
            PullDataset005Verdict::Fail,
            "case mismatch must Fail (HF paths case-sensitive)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — multi-line stdout with the model name embedded.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_multiline_stdout() {
        let stdout = b"\
Resolving paiml/qwen2.5-coder-7b-apache-q4k-v1
Files: 339 tensors, 8.04 GB total
Cache: ~/.cache/apr/models/qwen2.5-coder-7b-apache-q4k-v1/
DRY RUN: no files downloaded.
";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(v, PullDataset005Verdict::Pass);
    }

    #[test]
    fn pass_with_progress_indicators() {
        let stdout = b"[*] paiml/qwen2.5-coder-7b-apache-q4k-v1 [12%]";
        let v = verdict_from_model_path_backward_compat(stdout, b"qwen2.5-coder-7b-apache-q4k-v1");
        assert_eq!(v, PullDataset005Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 7: Domain — substring property at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_at_every_offset_in_haystack() {
        // The needle "abc" should be found at every offset where it
        // appears.
        let needle = b"abc";
        for offset in 0..10 {
            let mut stdout = vec![b'.'; 10 + offset];
            stdout.extend_from_slice(needle);
            stdout.extend_from_slice(b"...padding...");
            let v = verdict_from_model_path_backward_compat(&stdout, needle);
            assert_eq!(
                v,
                PullDataset005Verdict::Pass,
                "needle at offset {offset} must Pass"
            );
        }
    }

    #[test]
    fn fail_needle_one_byte_short_in_overlapping_pattern() {
        // Haystack contains "ababab" — needle "ababaX" should
        // NOT match, even though there's near-overlap.
        let stdout = b"ababababab";
        let v = verdict_from_model_path_backward_compat(stdout, b"ababaX");
        assert_eq!(v, PullDataset005Verdict::Fail);
    }
}
