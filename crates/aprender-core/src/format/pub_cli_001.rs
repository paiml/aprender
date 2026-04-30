// SHIP-TWO-001 — `apr-cli-publish-v1` algorithm-level PARTIAL
// discharge for FALSIFY-PUB-CLI-001.
//
// Contract: `contracts/apr-cli-publish-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI publish gate; cross-cutting requirement for MODEL-1 +
// MODEL-2 shipping).
//
// ## What FALSIFY-PUB-CLI-001 says
//
//   rule: default features don't pull old deps
//   prediction: "default features contain no inference/training/code/cuda"
//   test: "grep '^default = ' crates/apr-cli/Cargo.toml |
//          grep -qvE 'inference|training|code|cuda'"
//   if_fails: "cargo install aprender will hit cyclic dep chain"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given the bytes of the `default = […]` line from
// `apr-cli/Cargo.toml`, Pass iff:
//
//   default_line is non-empty AND
//   default_line does NOT contain ANY of:
//   - "inference"
//   - "training"
//   - "code"
//   - "cuda"
//
// Substring containment matches the contract's
// `grep -qvE 'inference|training|code|cuda'` semantics. Even one
// of the four forbidden tokens trips the gate — they each pull a
// heavy/cyclic dep chain that breaks `cargo install aprender`.

/// Forbidden substrings in the `default = […]` features line.
///
/// Per contract: each of these triggers a cyclic or heavy-binary
/// dep chain when included in the published `aprender` crate's
/// default features. They MUST be opt-in, never default.
pub const AC_PUB_CLI_001_FORBIDDEN_SUBSTRINGS: &[&[u8]] = &[
    b"inference",
    b"training",
    b"code",
    b"cuda",
];

/// Binary verdict for `FALSIFY-PUB-CLI-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubCli001Verdict {
    /// `default = […]` line is non-empty AND contains none of the
    /// four forbidden substrings.
    Pass,
    /// One or more of:
    /// - `default_line.is_empty()` (caller error — Cargo.toml
    ///   parsing produced no default-features line).
    /// - `default_line` contains any of `inference`, `training`,
    ///   `code`, or `cuda` (regression — published crate would
    ///   pull cyclic/heavy deps).
    Fail,
}

/// Pure verdict function for `FALSIFY-PUB-CLI-001`.
///
/// Inputs:
/// - `default_line`: bytes of the literal `default = […]` line
///   from `crates/apr-cli/Cargo.toml`. The caller is expected to
///   extract this line via `grep '^default = '` or equivalent.
///
/// Pass iff:
/// 1. `!default_line.is_empty()`,
/// 2. `default_line` does NOT contain any of the four forbidden
///    substrings (case-sensitive byte match).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Clean default features — `Pass`:
/// ```
/// use aprender::format::pub_cli_001::{
///     verdict_from_default_features_string, PubCli001Verdict,
/// };
/// let line = b"default = [\"format\", \"pull\", \"qa\"]";
/// let v = verdict_from_default_features_string(line);
/// assert_eq!(v, PubCli001Verdict::Pass);
/// ```
///
/// `inference` snuck into defaults (cyclic dep risk) — `Fail`:
/// ```
/// use aprender::format::pub_cli_001::{
///     verdict_from_default_features_string, PubCli001Verdict,
/// };
/// let line = b"default = [\"format\", \"inference\", \"pull\"]";
/// let v = verdict_from_default_features_string(line);
/// assert_eq!(v, PubCli001Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_default_features_string(default_line: &[u8]) -> PubCli001Verdict {
    if default_line.is_empty() {
        return PubCli001Verdict::Fail;
    }
    for forbidden in AC_PUB_CLI_001_FORBIDDEN_SUBSTRINGS {
        if contains_subsequence(default_line, forbidden) {
            return PubCli001Verdict::Fail;
        }
    }
    PubCli001Verdict::Pass
}

/// Returns `true` iff `needle` appears as a contiguous subsequence
/// of `haystack`. Same primitive as in `pull_dataset_001` and
/// `pull_dataset_005`.
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
    // Section 1: Provenance pin — the four forbidden substrings.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_forbidden_substrings_count_is_four() {
        assert_eq!(AC_PUB_CLI_001_FORBIDDEN_SUBSTRINGS.len(), 4);
    }

    #[test]
    fn provenance_forbidden_substrings_are_canonical() {
        assert_eq!(AC_PUB_CLI_001_FORBIDDEN_SUBSTRINGS[0], b"inference");
        assert_eq!(AC_PUB_CLI_001_FORBIDDEN_SUBSTRINGS[1], b"training");
        assert_eq!(AC_PUB_CLI_001_FORBIDDEN_SUBSTRINGS[2], b"code");
        assert_eq!(AC_PUB_CLI_001_FORBIDDEN_SUBSTRINGS[3], b"cuda");
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — clean default features.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_minimal_default_features() {
        let line = b"default = [\"format\", \"qa\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Pass);
    }

    #[test]
    fn pass_canonical_apr_default_features() {
        // Realistic clean defaults: format readers, pull, qa, tools.
        let line = b"default = [\"format\", \"pull\", \"qa\", \"tools\", \"validate\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Pass);
    }

    #[test]
    fn pass_empty_array_default() {
        let line = b"default = []";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — each forbidden substring (one at a time).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_inference_in_defaults() {
        let line = b"default = [\"format\", \"inference\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(
            v,
            PubCli001Verdict::Fail,
            "inference in default features must Fail (cyclic dep risk)"
        );
    }

    #[test]
    fn fail_training_in_defaults() {
        let line = b"default = [\"training\", \"format\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Fail);
    }

    #[test]
    fn fail_code_in_defaults() {
        let line = b"default = [\"format\", \"code\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Fail);
    }

    #[test]
    fn fail_cuda_in_defaults() {
        let line = b"default = [\"format\", \"cuda\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — multiple forbidden substrings.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_inference_and_cuda() {
        let line = b"default = [\"inference\", \"cuda\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Fail);
    }

    #[test]
    fn fail_all_four_forbidden() {
        let line = b"default = [\"inference\", \"training\", \"code\", \"cuda\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — empty input (caller error).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_line() {
        let v = verdict_from_default_features_string(&[]);
        assert_eq!(
            v,
            PubCli001Verdict::Fail,
            "empty line must Fail (Cargo.toml parsing failed)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Edge cases — substring matches anywhere in the line.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_inference_as_dep_value() {
        // Even as a non-features dep value (unusual but possible),
        // any substring match trips the gate.
        let line = b"default = [\"realizar-inference-stub\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(
            v,
            PubCli001Verdict::Fail,
            "substring match anywhere in line trips gate"
        );
    }

    #[test]
    fn fail_code_as_part_of_other_token() {
        // "qcode" or "decode" both contain "code" as substring.
        // This is an intentional conservative match — anything
        // resembling 'code' is suspect in default features per
        // contract.
        let line = b"default = [\"qcode\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Fail);
    }

    #[test]
    fn fail_cuda_as_dep_path() {
        let line = b"default = [\"realizar/cuda-kernels\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — actual apr-cli/Cargo.toml format.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_realistic_with_long_features_array() {
        // Plausible long but clean defaults.
        let line = b"default = [\"format\", \"pull\", \"qa\", \"validate\", \"diff\", \"inspect\", \"tensors\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Pass);
    }

    #[test]
    fn fail_realistic_with_inference_added() {
        // The exact regression class: realistic defaults + inference.
        let line = b"default = [\"format\", \"pull\", \"qa\", \"inference\", \"validate\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Fail);
    }

    #[test]
    fn pass_features_with_safe_substrings() {
        // "format" and "validate" don't contain any forbidden tokens.
        let line = b"default = [\"format\", \"validate-strict\"]";
        let v = verdict_from_default_features_string(line);
        assert_eq!(v, PubCli001Verdict::Pass);
    }
}
