// `apr-docs-v1` algorithm-level PARTIAL discharge for the 5
// FALSIFY-README gates (install command, apr run example, no apr-cli
// reference, no stale repos, mdbook build).
//
// Contract: `contracts/apr-docs-v1.yaml`.
//
// ## Disambiguation
//
// `readme-claims-v1.yaml` (task #256) is a sibling contract that pins
// quantitative claims (crate count, contract count, CLI command count,
// cookbook link). Despite both contracts sharing the FALSIFY-README-*
// prefix in some descriptions, the two contracts cover different
// invariants. This file binds `apr-docs-v1` only.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Five pure decision predicates over the README text and an mdbook exit
// code. Live discharge is `make tier3` running the falsification tests
// directly (`grep -q "cargo install aprender" README.md`, `mdbook build
// book/`). Algorithm-level pinning prevents drift on the install-command
// brand and the no-stale-repos invariant.

/// Required install command per FALSIFY-README-001.
pub const AC_DOCS_REQUIRED_INSTALL: &str = "cargo install aprender";

/// Required CLI usage example per FALSIFY-README-002.
pub const AC_DOCS_REQUIRED_USAGE: &str = "apr run";

/// Forbidden install command per FALSIFY-README-003.
pub const AC_DOCS_FORBIDDEN_INSTALL: &str = "cargo install apr-cli";

/// Repo names that MUST NOT appear as `cargo install <name>` per
/// FALSIFY-README-004 (post-APR-MONO-consolidation).
pub const AC_DOCS_FORBIDDEN_REPOS: [&str; 4] = ["trueno", "realizar", "entrenar", "batuta"];

// =============================================================================
// FALSIFY-README-001 — `cargo install aprender` present
// =============================================================================

/// Verdict for FALSIFY-README-001.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallCommandVerdict {
    /// README contains `cargo install aprender`.
    Pass,
    /// String absent.
    Fail,
}

#[must_use]
pub fn verdict_from_install_command(readme: &str) -> InstallCommandVerdict {
    if readme.contains(AC_DOCS_REQUIRED_INSTALL) {
        InstallCommandVerdict::Pass
    } else {
        InstallCommandVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-README-002 — `apr run` example present
// =============================================================================

/// Verdict for FALSIFY-README-002.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AprRunExampleVerdict {
    /// README contains `apr run`.
    Pass,
    /// String absent.
    Fail,
}

#[must_use]
pub fn verdict_from_apr_run_example(readme: &str) -> AprRunExampleVerdict {
    if readme.contains(AC_DOCS_REQUIRED_USAGE) {
        AprRunExampleVerdict::Pass
    } else {
        AprRunExampleVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-README-003 — `cargo install apr-cli` ABSENT
// =============================================================================

/// Verdict for FALSIFY-README-003.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoAprCliInstallVerdict {
    /// README does NOT contain `cargo install apr-cli`.
    Pass,
    /// Forbidden string is present.
    Fail,
}

#[must_use]
pub fn verdict_from_no_apr_cli_install(readme: &str) -> NoAprCliInstallVerdict {
    if readme.contains(AC_DOCS_FORBIDDEN_INSTALL) {
        NoAprCliInstallVerdict::Fail
    } else {
        NoAprCliInstallVerdict::Pass
    }
}

// =============================================================================
// FALSIFY-README-004 — no stale repo install commands
// =============================================================================

/// Verdict for FALSIFY-README-004.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoStaleReposVerdict {
    /// README does not contain `cargo install <stale_repo>` for any of
    /// {trueno, realizar, entrenar, batuta}.
    Pass,
    /// At least one stale repo `cargo install` reference present.
    Fail,
}

#[must_use]
pub fn verdict_from_no_stale_repos(readme: &str) -> NoStaleReposVerdict {
    for repo in AC_DOCS_FORBIDDEN_REPOS {
        let needle = format!("cargo install {repo}");
        // Match the gate's `\b` boundary: `cargo install batuta-something` MUST
        // not trigger; only `cargo install batuta` (followed by whitespace,
        // newline, or end-of-text) does.
        if let Some(pos) = readme.find(&needle) {
            let after = &readme[pos + needle.len()..];
            // Word boundary: next char is not alphanumeric, underscore, or hyphen.
            let boundary_ok = after
                .chars()
                .next()
                .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_' && c != '-');
            if boundary_ok {
                return NoStaleReposVerdict::Fail;
            }
        }
    }
    NoStaleReposVerdict::Pass
}

// =============================================================================
// FALSIFY-README-005 — `mdbook build book/` exits 0
// =============================================================================

/// Verdict for FALSIFY-README-005.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MdbookBuildVerdict {
    /// `mdbook build book/` exited 0.
    Pass,
    /// Non-zero exit.
    Fail,
}

#[must_use]
pub fn verdict_from_mdbook_build(exit_code: i32) -> MdbookBuildVerdict {
    if exit_code == 0 {
        MdbookBuildVerdict::Pass
    } else {
        MdbookBuildVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_install_command_string() {
        assert_eq!(AC_DOCS_REQUIRED_INSTALL, "cargo install aprender");
    }

    #[test]
    fn provenance_required_usage_string() {
        assert_eq!(AC_DOCS_REQUIRED_USAGE, "apr run");
    }

    #[test]
    fn provenance_forbidden_install_string() {
        assert_eq!(AC_DOCS_FORBIDDEN_INSTALL, "cargo install apr-cli");
    }

    #[test]
    fn provenance_forbidden_repos_count_4() {
        assert_eq!(AC_DOCS_FORBIDDEN_REPOS.len(), 4);
        assert!(AC_DOCS_FORBIDDEN_REPOS.contains(&"trueno"));
        assert!(AC_DOCS_FORBIDDEN_REPOS.contains(&"realizar"));
        assert!(AC_DOCS_FORBIDDEN_REPOS.contains(&"entrenar"));
        assert!(AC_DOCS_FORBIDDEN_REPOS.contains(&"batuta"));
    }

    // -------------------------------------------------------------------------
    // Section 2: README-001 install command.
    // -------------------------------------------------------------------------
    #[test]
    fn r001_pass_install_present() {
        let r = "## Install\n```bash\ncargo install aprender\n```\n";
        assert_eq!(verdict_from_install_command(r), InstallCommandVerdict::Pass);
    }

    #[test]
    fn r001_fail_install_absent() {
        let r = "## Install\nDownload the binary from releases.\n";
        assert_eq!(verdict_from_install_command(r), InstallCommandVerdict::Fail);
    }

    #[test]
    fn r001_fail_empty_readme() {
        assert_eq!(verdict_from_install_command(""), InstallCommandVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: README-002 apr run example.
    // -------------------------------------------------------------------------
    #[test]
    fn r002_pass_apr_run_present() {
        let r = "## Quickstart\n```bash\napr run hf://Qwen/Qwen3-Coder-30B-A3B\n```";
        assert_eq!(verdict_from_apr_run_example(r), AprRunExampleVerdict::Pass);
    }

    #[test]
    fn r002_fail_apr_run_absent() {
        let r = "## Quickstart\nSee the docs site for usage.";
        assert_eq!(verdict_from_apr_run_example(r), AprRunExampleVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: README-003 no apr-cli install.
    // -------------------------------------------------------------------------
    #[test]
    fn r003_pass_no_apr_cli_install() {
        let r = "## Install\n`cargo install aprender`\n";
        assert_eq!(verdict_from_no_apr_cli_install(r), NoAprCliInstallVerdict::Pass);
    }

    #[test]
    fn r003_fail_apr_cli_install_present() {
        let r = "## Install\n`cargo install apr-cli`\n";
        assert_eq!(verdict_from_no_apr_cli_install(r), NoAprCliInstallVerdict::Fail);
    }

    #[test]
    fn r003_pass_apr_cli_mentioned_but_not_installed() {
        // The crate name appears in module paths but not as a cargo install.
        let r = "The internal `apr-cli` crate provides command logic.";
        assert_eq!(verdict_from_no_apr_cli_install(r), NoAprCliInstallVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 5: README-004 no stale repos.
    // -------------------------------------------------------------------------
    #[test]
    fn r004_pass_only_aprender() {
        let r = "## Install\n`cargo install aprender`";
        assert_eq!(verdict_from_no_stale_repos(r), NoStaleReposVerdict::Pass);
    }

    #[test]
    fn r004_fail_install_trueno() {
        let r = "## Install\n`cargo install trueno`";
        assert_eq!(verdict_from_no_stale_repos(r), NoStaleReposVerdict::Fail);
    }

    #[test]
    fn r004_fail_install_realizar() {
        let r = "Install: cargo install realizar";
        assert_eq!(verdict_from_no_stale_repos(r), NoStaleReposVerdict::Fail);
    }

    #[test]
    fn r004_fail_install_entrenar() {
        let r = "$ cargo install entrenar\n";
        assert_eq!(verdict_from_no_stale_repos(r), NoStaleReposVerdict::Fail);
    }

    #[test]
    fn r004_fail_install_batuta() {
        let r = "## Install\n```bash\ncargo install batuta\n```";
        assert_eq!(verdict_from_no_stale_repos(r), NoStaleReposVerdict::Fail);
    }

    #[test]
    fn r004_pass_stale_name_in_history_context() {
        // Migration/history context — the gate explicitly allows stale names
        // to appear; only `cargo install <name>\b` is forbidden.
        let r = "Pre-APR-MONO, separate crates trueno, realizar, entrenar, batuta were installed individually.";
        assert_eq!(verdict_from_no_stale_repos(r), NoStaleReposVerdict::Pass);
    }

    #[test]
    fn r004_pass_stale_name_with_suffix() {
        // Word boundary: `cargo install trueno-foo` MUST pass.
        let r = "cargo install trueno-extras";
        assert_eq!(verdict_from_no_stale_repos(r), NoStaleReposVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 6: README-005 mdbook build.
    // -------------------------------------------------------------------------
    #[test]
    fn r005_pass_exit_zero() {
        assert_eq!(verdict_from_mdbook_build(0), MdbookBuildVerdict::Pass);
    }

    #[test]
    fn r005_fail_exit_nonzero() {
        assert_eq!(verdict_from_mdbook_build(1), MdbookBuildVerdict::Fail);
    }

    #[test]
    fn r005_fail_exit_signal() {
        assert_eq!(verdict_from_mdbook_build(137), MdbookBuildVerdict::Fail);
    }

    #[test]
    fn r005_fail_exit_negative() {
        assert_eq!(verdict_from_mdbook_build(-1), MdbookBuildVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — a healthy README + clean book build passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_full_healthy_readme_passes_all_5() {
        let readme = "\
# aprender

Next-generation ML framework in pure Rust.

## Install

```bash
cargo install aprender
```

## Quickstart

```bash
apr run hf://Qwen/Qwen3-Coder-30B-A3B
```

## History

Pre-APR-MONO, the framework was split across trueno, realizar, entrenar,
and batuta. These are now subsumed by aprender.
";
        assert_eq!(verdict_from_install_command(readme), InstallCommandVerdict::Pass);
        assert_eq!(verdict_from_apr_run_example(readme), AprRunExampleVerdict::Pass);
        assert_eq!(verdict_from_no_apr_cli_install(readme), NoAprCliInstallVerdict::Pass);
        assert_eq!(verdict_from_no_stale_repos(readme), NoStaleReposVerdict::Pass);
        assert_eq!(verdict_from_mdbook_build(0), MdbookBuildVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // The exact regression class: bad README + book builds broken.
        let bad = "## Install\n`cargo install apr-cli`\n## Train\n`cargo install trueno`";
        assert_eq!(verdict_from_install_command(bad), InstallCommandVerdict::Fail);
        assert_eq!(verdict_from_apr_run_example(bad), AprRunExampleVerdict::Fail);
        assert_eq!(verdict_from_no_apr_cli_install(bad), NoAprCliInstallVerdict::Fail);
        assert_eq!(verdict_from_no_stale_repos(bad), NoStaleReposVerdict::Fail);
        assert_eq!(verdict_from_mdbook_build(1), MdbookBuildVerdict::Fail);
    }
}
