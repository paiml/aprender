//! Include / exclude glob filtering for `apr pull` selective download
//! (CRUX-A-04).
//!
//! Contract: `contracts/crux-A-04-v1.yaml`.
//!
//! Pure classifier — takes a remote file list plus `--include` /
//! `--exclude` glob patterns and returns the subset that should be
//! downloaded. No I/O, no network, no filesystem access. The
//! integration-level claim ("`apr pull` actually downloads exactly
//! this set") is discharged by a separate network-gated harness.
//!
//! Formula (from contract): `Selected(R, I, X) = (if I == ∅ then R else
//! R ∩ I) \ X`. `--exclude` wins over `--include` for any overlap.
//! Glob semantics are fnmatch-style (`*`, `?`, `[abc]`, `**`) to match
//! `huggingface_hub`'s `snapshot_download(allow_patterns, ignore_patterns)`.

use glob::Pattern;

/// Error returned when a user-supplied glob is syntactically invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlobFilterError {
    /// The pattern could not be parsed by `glob::Pattern::new`. The
    /// offending pattern is included for operator-visible diagnostics.
    InvalidPattern(String),
}

impl std::fmt::Display for GlobFilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlobFilterError::InvalidPattern(p) => {
                write!(f, "invalid glob pattern: {p:?}")
            }
        }
    }
}

impl std::error::Error for GlobFilterError {}

/// Return true iff `path` matches ANY of the given compiled patterns.
fn any_match(path: &str, patterns: &[Pattern]) -> bool {
    patterns.iter().any(|p| p.matches(path))
}

/// Compile a slice of raw glob strings into `glob::Pattern`s, failing
/// on the first malformed pattern.
fn compile_patterns(raw: &[&str]) -> Result<Vec<Pattern>, GlobFilterError> {
    raw.iter()
        .map(|s| {
            Pattern::new(s)
                .map_err(|_| GlobFilterError::InvalidPattern((*s).to_string()))
        })
        .collect()
}

/// Select the subset of `files` that should be downloaded, per the
/// CRUX-A-04 `glob_selection_set` formula.
///
/// - Empty `include` means "take everything".
/// - `exclude` wins over `include` for any overlap.
/// - Iteration order is preserved (stable, deterministic output).
///
/// CRUX-A-04 ALGO-001/002/003 sub-claim of FALSIFY-001/002/003: the
/// selection function matches the contract formula exactly, which is
/// the algorithm-level precondition for the integration-level
/// `apr pull --include/--exclude` download-set check.
pub fn select_files<S: AsRef<str>>(
    files: &[S],
    include: &[&str],
    exclude: &[&str],
) -> Result<Vec<String>, GlobFilterError> {
    let inc = compile_patterns(include)?;
    let exc = compile_patterns(exclude)?;

    let mut out = Vec::with_capacity(files.len());
    for f in files {
        let path = f.as_ref();
        let included = inc.is_empty() || any_match(path, &inc);
        let excluded = any_match(path, &exc);
        if included && !excluded {
            out.push(path.to_string());
        }
    }
    Ok(out)
}

/// Return true iff a single `path` would be selected under the given
/// globs. Convenience wrapper; identical semantics to `select_files`
/// on a one-element list.
pub fn is_selected(
    path: &str,
    include: &[&str],
    exclude: &[&str],
) -> Result<bool, GlobFilterError> {
    let inc = compile_patterns(include)?;
    let exc = compile_patterns(exclude)?;
    let included = inc.is_empty() || any_match(path, &inc);
    let excluded = any_match(path, &exc);
    Ok(included && !excluded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_repo() -> Vec<&'static str> {
        // Approximates the gpt2 file tree used in the FALSIFY tests.
        vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
            "README.md",
        ]
    }

    #[test]
    fn empty_include_empty_exclude_selects_everything() {
        // Contract: empty --include means "take everything", empty
        // --exclude means "drop nothing".
        let files = sample_repo();
        let got = select_files(&files, &[], &[]).unwrap();
        assert_eq!(got.len(), files.len());
    }

    #[test]
    fn include_safetensors_falsify_001_algorithm_sub_claim() {
        // CRUX-A-04 ALGO-001 sub-claim of FALSIFY-001: `--include
        // '*.safetensors'` MUST restrict the output to only files
        // whose path matches the glob. Matches the `find ! -name`
        // check in the shell falsification test.
        let files = sample_repo();
        let got = select_files(&files, &["*.safetensors"], &[]).unwrap();
        assert_eq!(got, vec!["model.safetensors"]);
        for f in &got {
            assert!(f.ends_with(".safetensors"), "unexpected leak: {f}");
        }
    }

    #[test]
    fn exclude_bin_falsify_002_algorithm_sub_claim() {
        // CRUX-A-04 ALGO-002 sub-claim of FALSIFY-002: `--exclude
        // '*.bin'` MUST drop all files whose path matches the glob.
        let files = sample_repo();
        let got = select_files(&files, &[], &["*.bin"]).unwrap();
        assert!(!got.iter().any(|f| f.ends_with(".bin")));
        // All non-.bin files should be retained.
        assert_eq!(got.len(), files.len() - 1);
    }

    #[test]
    fn exclude_wins_over_include_falsify_003_algorithm_sub_claim() {
        // CRUX-A-04 ALGO-003 sub-claim of FALSIFY-003: precedence rule
        // — `--include '*.json' --exclude 'config.json'` keeps *.json
        // files EXCEPT config.json.
        let files = sample_repo();
        let got =
            select_files(&files, &["*.json"], &["config.json"]).unwrap();
        assert!(!got.iter().any(|f| f == "config.json"));
        // But other .json files survive.
        assert!(got.iter().any(|f| f == "tokenizer.json"));
        assert!(got.iter().any(|f| f == "vocab.json"));
        // And no non-json file leaks in.
        for f in &got {
            assert!(f.ends_with(".json"), "non-json leak: {f}");
        }
    }

    #[test]
    fn multiple_include_globs_union_semantics() {
        let files = sample_repo();
        let got =
            select_files(&files, &["*.safetensors", "*.bin"], &[]).unwrap();
        assert_eq!(got.len(), 2);
        assert!(got.contains(&"model.safetensors".to_string()));
        assert!(got.contains(&"pytorch_model.bin".to_string()));
    }

    #[test]
    fn multiple_exclude_globs_union_semantics() {
        let files = sample_repo();
        let got = select_files(&files, &[], &["*.bin", "*.h5"]).unwrap();
        assert!(!got.iter().any(|f| f.ends_with(".bin")));
        assert!(!got.iter().any(|f| f.ends_with(".h5")));
    }

    #[test]
    fn question_mark_matches_single_char() {
        let files = vec!["a.json", "ab.json", "b.json"];
        let got = select_files(&files, &["?.json"], &[]).unwrap();
        assert!(got.contains(&"a.json".to_string()));
        assert!(got.contains(&"b.json".to_string()));
        assert!(!got.contains(&"ab.json".to_string()));
    }

    #[test]
    fn recursive_glob_matches_subdirs() {
        // HuggingFace repos commonly nest weights under subdirs
        // (`model-00001-of-00002.safetensors` is flat; LoRA adapters
        // often live under `adapters/`). Confirm `**` behaves.
        let files = vec![
            "adapters/lora.safetensors",
            "adapters/nested/deep.safetensors",
            "model.safetensors",
        ];
        let got =
            select_files(&files, &["adapters/**/*.safetensors"], &[]).unwrap();
        assert!(got.contains(&"adapters/lora.safetensors".to_string()));
        assert!(got.contains(&"adapters/nested/deep.safetensors".to_string()));
        assert!(!got.contains(&"model.safetensors".to_string()));
    }

    #[test]
    fn invalid_include_glob_is_error() {
        // `glob` 0.3 rejects unclosed `[` as invalid.
        let files = vec!["a.json"];
        let err = select_files(&files, &["a["], &[]).unwrap_err();
        match err {
            GlobFilterError::InvalidPattern(p) => assert_eq!(p, "a["),
        }
    }

    #[test]
    fn invalid_exclude_glob_is_error() {
        let files = vec!["a.json"];
        assert!(select_files(&files, &[], &["a["]).is_err());
    }

    #[test]
    fn selection_is_deterministic() {
        // Same inputs → same output across invocations. Matches
        // `download_idempotence` invariant in the contract.
        let files = sample_repo();
        let a = select_files(&files, &["*.json"], &["config.json"]).unwrap();
        let b = select_files(&files, &["*.json"], &["config.json"]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn selection_preserves_input_order() {
        // Deterministic iteration order eases downstream manifest
        // generation — the output must match the input order.
        let files = vec!["z.json", "a.json", "m.json"];
        let got = select_files(&files, &["*.json"], &[]).unwrap();
        assert_eq!(got, vec!["z.json", "a.json", "m.json"]);
    }

    #[test]
    fn empty_repo_produces_empty_output() {
        let files: Vec<&str> = vec![];
        let got = select_files(&files, &["*.safetensors"], &[]).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn include_matches_no_file_produces_empty() {
        // If `--include` matches nothing, output is empty — NOT a
        // fallback to "everything". Matches the formula exactly.
        let files = sample_repo();
        let got = select_files(&files, &["*.nonexistent"], &[]).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn is_selected_agrees_with_select_files() {
        let files = sample_repo();
        let inc = ["*.json"];
        let exc = ["config.json"];
        let whole = select_files(&files, &inc, &exc).unwrap();
        for f in &files {
            let single = is_selected(f, &inc, &exc).unwrap();
            assert_eq!(
                single,
                whole.iter().any(|w| w == f),
                "disagreement on {f}",
            );
        }
    }

    #[test]
    fn falsify_001_gpt2_shape_only_safetensors_survive() {
        // CRUX-A-04 FALSIFY-001 algorithm-level: the shell test's
        // predicate is `find ! -name '*.safetensors' | grep -q .`.
        // Algorithm-level equivalent: every output path ends with
        // .safetensors AND model.safetensors is present.
        let files = sample_repo();
        let got = select_files(&files, &["*.safetensors"], &[]).unwrap();
        assert!(!got.is_empty());
        assert!(got.iter().all(|f| f.ends_with(".safetensors")));
        assert!(got.contains(&"model.safetensors".to_string()));
    }

    #[test]
    fn falsify_002_gpt2_shape_zero_bin_files() {
        // CRUX-A-04 FALSIFY-002 algorithm-level: shell test counts
        // .bin files and asserts 0.
        let files = sample_repo();
        let got = select_files(&files, &[], &["*.bin"]).unwrap();
        assert_eq!(got.iter().filter(|f| f.ends_with(".bin")).count(), 0);
    }

    #[test]
    fn falsify_003_gpt2_shape_config_dropped_others_kept() {
        // CRUX-A-04 FALSIFY-003 algorithm-level: shell test asserts
        // `! -f "$TMP/config.json"` AND `ls "$TMP"/*.json` succeeds.
        let files = sample_repo();
        let got =
            select_files(&files, &["*.json"], &["config.json"]).unwrap();
        assert!(!got.iter().any(|f| f == "config.json"));
        assert!(got.iter().any(|f| f.ends_with(".json")));
    }
}
