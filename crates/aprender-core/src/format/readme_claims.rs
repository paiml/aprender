// SHIP-TWO-001 — `readme-claims-v1` algorithm-level PARTIAL discharge
// for FALSIFY-README-001..004 (closes 4/4).
//
// Contract: `contracts/readme-claims-v1.yaml`.
//
// Bundles 4 verdict fns over README-vs-repo-state consistency. The
// runtime-level falsifier `bash scripts/check_readme_claims.sh`
// already exists per contract evidence — this file pins the
// *decision rule* against drift in:
//
//   - workspace crate count (claimed N == observed N)
//   - contract YAML count (claimed M == observed M)
//   - apr CLI subcommand count (claimed K == observed K)
//   - apr-cookbook link presence (≥ 1 hit in README)

// ===========================================================================
// README-001 — Workspace crate count claim matches filesystem
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readme001Verdict { Pass, Fail }

/// Pass iff `claimed == observed` AND both are non-zero.
#[must_use]
pub fn verdict_from_crate_count(claimed: u64, observed: u64) -> Readme001Verdict {
    if claimed == 0 || observed == 0 { return Readme001Verdict::Fail; }
    if claimed == observed { Readme001Verdict::Pass } else { Readme001Verdict::Fail }
}

// ===========================================================================
// README-002 — Contract YAML count claim matches filesystem
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readme002Verdict { Pass, Fail }

/// Pass iff `claimed == observed` AND both are non-zero.
#[must_use]
pub fn verdict_from_contract_count(claimed: u64, observed: u64) -> Readme002Verdict {
    if claimed == 0 || observed == 0 { return Readme002Verdict::Fail; }
    if claimed == observed { Readme002Verdict::Pass } else { Readme002Verdict::Fail }
}

// ===========================================================================
// README-003 — CLI subcommand count claim matches `apr --help`
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readme003Verdict { Pass, Fail }

/// Pass iff `claimed == observed` AND `observed >= MIN_REASONABLE`
/// (guards against an empty-help regression — apr is documented as a
/// 58-subcommand CLI in `CLAUDE.md`).
pub const AC_README_003_MIN_REASONABLE: u64 = 30;

#[must_use]
pub fn verdict_from_cli_command_count(claimed: u64, observed: u64) -> Readme003Verdict {
    if observed < AC_README_003_MIN_REASONABLE { return Readme003Verdict::Fail; }
    if claimed == observed { Readme003Verdict::Pass } else { Readme003Verdict::Fail }
}

// ===========================================================================
// README-004 — `apr-cookbook` link is present in README
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readme004Verdict { Pass, Fail }

/// Pass iff `readme_text` contains the literal substring `apr-cookbook`.
/// Matches both `[apr-cookbook](url)` markdown links and bare path
/// references like `../apr-cookbook/...`.
#[must_use]
pub fn verdict_from_cookbook_link_presence(readme_text: &str) -> Readme004Verdict {
    if readme_text.contains("apr-cookbook") {
        Readme004Verdict::Pass
    } else {
        Readme004Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- README-001 --------------------------------------------------------

    #[test] fn r001_pass_match() { assert_eq!(verdict_from_crate_count(70, 70), Readme001Verdict::Pass); }
    #[test] fn r001_fail_off_by_one_high() { assert_eq!(verdict_from_crate_count(70, 71), Readme001Verdict::Fail); }
    #[test] fn r001_fail_off_by_one_low() { assert_eq!(verdict_from_crate_count(70, 69), Readme001Verdict::Fail); }
    #[test] fn r001_fail_claimed_zero() { assert_eq!(verdict_from_crate_count(0, 70), Readme001Verdict::Fail); }
    #[test] fn r001_fail_observed_zero() { assert_eq!(verdict_from_crate_count(70, 0), Readme001Verdict::Fail); }

    // ----- README-002 --------------------------------------------------------

    #[test] fn r002_pass_match() { assert_eq!(verdict_from_contract_count(405, 405), Readme002Verdict::Pass); }
    #[test] fn r002_fail_stale_low() { assert_eq!(verdict_from_contract_count(400, 405), Readme002Verdict::Fail); }
    #[test] fn r002_fail_zero_observed() { assert_eq!(verdict_from_contract_count(405, 0), Readme002Verdict::Fail); }
    #[test] fn r002_fail_zero_claimed() { assert_eq!(verdict_from_contract_count(0, 405), Readme002Verdict::Fail); }

    // ----- README-003 --------------------------------------------------------

    #[test] fn r003_pass_match() { assert_eq!(verdict_from_cli_command_count(58, 58), Readme003Verdict::Pass); }
    #[test] fn r003_pass_match_at_min() {
        assert_eq!(
            verdict_from_cli_command_count(AC_README_003_MIN_REASONABLE, AC_README_003_MIN_REASONABLE),
            Readme003Verdict::Pass
        );
    }
    #[test] fn r003_fail_observed_below_min() {
        // Empty-help regression: only 5 subcommands listed.
        assert_eq!(verdict_from_cli_command_count(58, 5), Readme003Verdict::Fail);
    }
    #[test] fn r003_fail_off_by_one() { assert_eq!(verdict_from_cli_command_count(58, 59), Readme003Verdict::Fail); }
    #[test] fn r003_fail_subcommand_dropped() { assert_eq!(verdict_from_cli_command_count(58, 57), Readme003Verdict::Fail); }
    #[test] fn r003_provenance_min() { assert_eq!(AC_README_003_MIN_REASONABLE, 30); }

    // ----- README-004 --------------------------------------------------------

    #[test] fn r004_pass_markdown_link() {
        let readme = "See the [apr-cookbook](https://github.com/paiml/apr-cookbook) for recipes.";
        assert_eq!(verdict_from_cookbook_link_presence(readme), Readme004Verdict::Pass);
    }

    #[test] fn r004_pass_relative_path() {
        let readme = "Local check: `../apr-cookbook/recipes/qwen.md`.";
        assert_eq!(verdict_from_cookbook_link_presence(readme), Readme004Verdict::Pass);
    }

    #[test] fn r004_pass_bare_mention() {
        // The contract just requires the literal substring.
        let readme = "Coordinated with apr-cookbook.";
        assert_eq!(verdict_from_cookbook_link_presence(readme), Readme004Verdict::Pass);
    }

    #[test] fn r004_fail_link_removed() {
        let readme = "# aprender\n\nA Rust ML framework.\n";
        assert_eq!(verdict_from_cookbook_link_presence(readme), Readme004Verdict::Fail);
    }

    #[test] fn r004_fail_link_renamed_without_sync() {
        // Someone replaced `apr-cookbook` with `apr-recipes` and forgot
        // to update the link.
        let readme = "See [apr-recipes](https://example.com).";
        assert_eq!(verdict_from_cookbook_link_presence(readme), Readme004Verdict::Fail);
    }

    #[test] fn r004_fail_empty() {
        assert_eq!(verdict_from_cookbook_link_presence(""), Readme004Verdict::Fail);
    }
}
