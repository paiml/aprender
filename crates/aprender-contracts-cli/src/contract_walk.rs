//! Shared recursive contract directory walker.
//!
//! Multiple `pv` subcommands need to load every YAML contract under
//! a directory tree (including subdirectories like `contracts/aprender/`,
//! `contracts/trueno/`, `contracts/patterns/`). This module provides the
//! canonical walker so commands agree on what counts as a contract and
//! which sidecar files to skip.
//!
//! PVL-1 (PMAT-1099): a corpus with ZERO contracts is REFUSED, never
//! reported. Measured before this change (PV-LEAN-AUDIT-001, 2026-09-10):
//! `pv lint /nonexistent-path` exited 0 with `Result: PASS` over 0 contracts,
//! and `pv proof-status <empty dir>` printed `Proof Status (0 contracts)` at
//! exit 0 — a gate that measures nothing and reports PASS. [`collect_corpus`]
//! is the entry point every reporting subcommand uses: it returns
//! [`ZeroContracts`] (exit [`ZERO_CONTRACTS_EXIT`]) for an empty or missing
//! directory, and treats a single `.yaml` file as a one-contract corpus so
//! `pv <cmd> <file>` reports that file instead of walking nothing.
//!
//! ONE definition of "empty". The set of contract files is the one `pv lint`
//! walks — [`provable_contracts::lint::collect_yaml_files`] (the
//! `is_contract_yaml` rule plus the skipped sidecar directories) — so no two
//! commands can disagree on what is there. A contract file that fails to
//! parse is NOT "no contract": it was measured, and it failed
//! ([`ParseErrors`], exit 1, every file named). The second review quorum on
//! #3093 measured `proof-status` and `lint --diff` refusing an
//! unparsable-only directory as "0 contracts" (exit 2) while `lint` failed
//! the same directory with `contracts: 1, errors: 1` (exit 1); this walker's
//! old private rule also skipped every `*playbook*` stem, and the corpus holds
//! real contracts named that way.

use std::fmt;
use std::path::{Path, PathBuf};

use provable_contracts::lint::collect_yaml_files;
use provable_contracts::schema::{parse_contract, Contract};

/// Exit status of a refused empty corpus.
///
/// 1 means "the corpus was measured and failed"; 2 means "nothing was
/// measured" — the invocation itself is wrong, which is also what clap
/// returns for a usage error. A caller treating both as failure is
/// unchanged; one that wants to tell them apart now can.
pub const ZERO_CONTRACTS_EXIT: i32 = 2;

/// The corpus under `path` holds no contract (after `filter`, when set).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroContracts {
    /// The directory (or file) that was asked for.
    pub path: PathBuf,
    /// A `--kind` filter that emptied a non-empty corpus, if any.
    pub filter: Option<String>,
}

impl fmt::Display for ZeroContracts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0 contracts under {}", self.path.display())?;
        if let Some(k) = &self.filter {
            write!(f, " (after --kind {k})")?;
        }
        Ok(())
    }
}

impl std::error::Error for ZeroContracts {}

/// Contract files under `path` that failed to parse: measured, and failed
/// (exit 1) — never a silent skip, never "0 contracts".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseErrors {
    /// The directory that was asked for.
    pub path: PathBuf,
    /// Contract files seen (parsed + failed).
    pub files: usize,
    /// `(file, error)` per failure, in walk order.
    pub errors: Vec<(PathBuf, String)>,
}

impl fmt::Display for ParseErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} of {} contract files under {} failed to parse",
            self.errors.len(),
            self.files,
            self.path.display()
        )?;
        for (file, err) in &self.errors {
            write!(f, "\n  {}: {err}", file.display())?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseErrors {}

/// Refuse an empty corpus: `Err(ZeroContracts)` when `corpus` is empty.
pub fn require_contracts<T>(
    path: &Path,
    corpus: &[T],
    filter: Option<&str>,
) -> Result<(), ZeroContracts> {
    if corpus.is_empty() {
        return Err(ZeroContracts {
            path: path.to_path_buf(),
            filter: filter.map(str::to_string),
        });
    }
    Ok(())
}

/// Is there at least one contract file under `path`? The one definition of a
/// non-empty corpus, answered WITHOUT parsing (diff mode asks this before it
/// says "nothing changed"). A file path is a one-contract corpus.
pub fn has_contract_files(path: &Path) -> bool {
    if path.is_file() {
        return true;
    }
    let mut files = Vec::new();
    collect_yaml_files(path, &mut files);
    !files.is_empty()
}

/// Load the corpus at `path` and refuse an empty one.
///
/// - a directory is walked by [`walk_contracts`]; every contract file must
///   parse — any failure is [`ParseErrors`] (exit 1), never a silent skip;
/// - a single `.yaml` file is a one-contract corpus (a parse error is the
///   file's own error, exit 1 — it WAS measured);
/// - a missing path, or a directory without a contract file, is
///   [`ZeroContracts`] (exit [`ZERO_CONTRACTS_EXIT`]).
///
/// The result is sorted by stem.
pub fn collect_corpus(path: &Path) -> Result<Vec<(String, Contract)>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    let mut errors = Vec::new();
    if path.is_dir() {
        walk_contracts(path, &mut out, &mut errors);
    } else if path.is_file() {
        out.push((stem_of(path), parse_contract(path)?));
    }
    if !errors.is_empty() {
        return Err(ParseErrors {
            path: path.to_path_buf(),
            files: out.len() + errors.len(),
            errors,
        }
        .into());
    }
    require_contracts(path, &out, None)?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Exit status for a `dispatch` error: [`ZERO_CONTRACTS_EXIT`] for a refused
/// empty corpus, 1 for everything else (a parse failure included).
pub fn exit_code_for(err: &(dyn std::error::Error + 'static)) -> i32 {
    if err.downcast_ref::<ZeroContracts>().is_some() {
        ZERO_CONTRACTS_EXIT
    } else {
        1
    }
}

fn stem_of(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// Walk `dir` recursively with `pv lint`'s file rule: every contract file is
/// parsed into `out` as `(stem, contract)`, or recorded in `errors` as
/// `(file, error)`. Nothing is dropped.
pub fn walk_contracts(
    dir: &Path,
    out: &mut Vec<(String, Contract)>,
    errors: &mut Vec<(PathBuf, String)>,
) {
    let mut files = Vec::new();
    collect_yaml_files(dir, &mut files);
    for path in files {
        match parse_contract(&path) {
            Ok(c) => out.push((stem_of(&path), c)),
            Err(e) => errors.push((path, e.to_string())),
        }
    }
}

/// Walk `dir` and collect every PARSEABLE contract, dropping the rest.
///
/// Answers "what is here" for callers that do not report over the result;
/// anything that REPORTS goes through [`collect_corpus`], which refuses an
/// empty corpus and fails a broken file.
pub fn collect_contracts(dir: &Path, out: &mut Vec<(String, Contract)>) {
    let mut dropped = Vec::new();
    walk_contracts(dir, out, &mut dropped);
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = "metadata:\n  version: \"1.0.0\"\n  description: \"t\"\n  references: [\"x\"]\nequations:\n  eq1:\n    formula: \"f(x)=x\"\nproof_obligations: []\nfalsification_tests: []\nkani_harnesses: []\n";

    #[test]
    fn collect_from_missing_dir_is_empty() {
        let mut out = Vec::new();
        collect_contracts(Path::new("/nonexistent/path/to/contracts"), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn collect_from_real_contracts_dir() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        if !dir.exists() {
            return; // skip on CI without contracts
        }
        let mut out = Vec::new();
        collect_contracts(&dir, &mut out);
        // Should find hundreds of contracts including subdirs.
        assert!(
            out.len() > 100,
            "expected > 100 contracts across the tree, got {}",
            out.len()
        );
        // Every entry must have a non-empty stem, and the sidecar is skipped.
        for (stem, _) in &out {
            assert!(!stem.is_empty(), "empty stem in collected contracts");
            assert_ne!(stem, "binding", "binding sidecar was not skipped");
        }
        // ONE file rule: the walker sees exactly the files `pv lint` walks — every
        // one of them parses in this corpus (lint's validate gate: 0 errors), so
        // the counts are equal; a `*playbook*` stem is a real contract, not a sidecar.
        let mut files = Vec::new();
        provable_contracts::lint::collect_yaml_files(&dir, &mut files);
        assert_eq!(
            out.len(),
            files.len(),
            "walker and lint disagree on the corpus"
        );
        assert!(
            out.iter().any(|(stem, _)| stem.contains("playbook")),
            "the corpus's playbook-named contracts were dropped"
        );
    }

    #[test]
    fn collect_skips_binding_yaml() {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        std::fs::write(
            tmp.path().join("binding.yaml"),
            "crates: []\nbindings: []\n",
        )
        .expect("fixture file is writable");
        std::fs::write(tmp.path().join("real-contract-v1.yaml"), MINIMAL)
            .expect("fixture file is writable");

        let mut out = Vec::new();
        collect_contracts(tmp.path(), &mut out);

        let stems: Vec<_> = out.iter().map(|(s, _)| s.as_str()).collect();
        assert!(!stems.contains(&"binding"), "binding should be skipped");
    }

    #[test]
    fn collect_recurses_into_subdirs() {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        let sub = tmp.path().join("sub");
        std::fs::create_dir_all(&sub).expect("fixture subdirectory is creatable");
        std::fs::write(tmp.path().join("top-v1.yaml"), MINIMAL).expect("fixture file is writable");
        std::fs::write(sub.join("nested-v1.yaml"), MINIMAL).expect("fixture file is writable");

        let mut out = Vec::new();
        collect_contracts(tmp.path(), &mut out);

        let stems: Vec<_> = out.iter().map(|(s, _)| s.clone()).collect();
        assert!(stems.contains(&"top-v1".to_string()));
        assert!(stems.contains(&"nested-v1".to_string()));
    }

    // ---- PVL-1 (PMAT-1099): an empty corpus is refused, never reported ----

    #[test]
    fn zero_contracts_names_the_path_and_the_filter() {
        let plain = ZeroContracts {
            path: PathBuf::from("/x/contracts"),
            filter: None,
        };
        assert_eq!(plain.to_string(), "0 contracts under /x/contracts");
        let filtered = ZeroContracts {
            path: PathBuf::from("/x/contracts"),
            filter: Some("kernel".to_string()),
        };
        assert_eq!(
            filtered.to_string(),
            "0 contracts under /x/contracts (after --kind kernel)"
        );
    }

    #[test]
    fn collect_corpus_refuses_a_missing_path_with_exit_2() {
        let err = collect_corpus(Path::new("/nonexistent/path/to/contracts"))
            .expect_err("a missing path is an empty corpus");
        assert!(
            err.downcast_ref::<ZeroContracts>().is_some(),
            "not a ZeroContracts: {err}"
        );
        assert_eq!(exit_code_for(err.as_ref()), ZERO_CONTRACTS_EXIT);
        assert_eq!(
            err.to_string(),
            "0 contracts under /nonexistent/path/to/contracts"
        );
    }

    #[test]
    fn collect_corpus_refuses_an_empty_dir_and_a_sidecar_only_dir() {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        assert!(
            collect_corpus(tmp.path()).is_err(),
            "empty dir must be refused"
        );
        std::fs::write(
            tmp.path().join("binding.yaml"),
            "crates: []\nbindings: []\n",
        )
        .expect("fixture file is writable");
        assert!(
            collect_corpus(tmp.path()).is_err(),
            "a directory holding only sidecars is an empty corpus"
        );
    }

    #[test]
    fn collect_corpus_treats_a_file_as_a_one_contract_corpus() {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        let file = tmp.path().join("solo-v1.yaml");
        std::fs::write(&file, MINIMAL).expect("fixture file is writable");
        let corpus = collect_corpus(&file).expect("one file is one contract");
        assert_eq!(corpus.len(), 1);
        assert_eq!(corpus[0].0, "solo-v1");
    }

    #[test]
    fn collect_corpus_propagates_a_parse_error_at_exit_1() {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        let file = tmp.path().join("garbage.yaml");
        std::fs::write(&file, "{{{ not yaml at all: [\n").expect("fixture file is writable");
        let err = collect_corpus(&file).expect_err("garbage is a parse error, not an empty corpus");
        assert!(
            err.downcast_ref::<ZeroContracts>().is_none(),
            "a parse error is not ZeroContracts"
        );
        assert_eq!(exit_code_for(err.as_ref()), 1);
    }

    #[test]
    fn require_contracts_passes_a_non_empty_corpus() {
        assert!(require_contracts(Path::new("/x"), &[1], None).is_ok());
        assert_eq!(
            require_contracts::<u8>(Path::new("/x"), &[], Some("kernel")),
            Err(ZeroContracts {
                path: PathBuf::from("/x"),
                filter: Some("kernel".to_string()),
            })
        );
    }
}
