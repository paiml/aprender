// SHIP-TWO-001 — `apr-cli-commands-v1` algorithm-level PARTIAL
// discharge for FALSIFY-CLI-001..004 (closes 4/4 sweep).
//
// Contract: `contracts/apr-cli-commands-v1.yaml`.
// Spec: SHIP-TWO-001 (apr CLI command registry — single source of truth).

use std::collections::HashSet;

// ===========================================================================
// FALSIFY-CLI-001 — Every registered command responds to --help
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cli001Verdict { Pass, Fail }

/// Pass iff every command in `registered` is present in `responds_to_help`
/// (set equality direction). `registered` MUST be non-empty.
#[must_use]
pub fn verdict_from_help_responsive<S1, S2>(
    registered: &HashSet<String, S1>,
    responds_to_help: &HashSet<String, S2>,
) -> Cli001Verdict
where
    S1: std::hash::BuildHasher,
    S2: std::hash::BuildHasher,
{
    if registered.is_empty() { return Cli001Verdict::Fail; }
    for cmd in registered {
        if !responds_to_help.contains(cmd) { return Cli001Verdict::Fail; }
    }
    Cli001Verdict::Pass
}

// ===========================================================================
// FALSIFY-CLI-002 — No undocumented command in `apr --help`
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cli002Verdict { Pass, Fail }

/// Pass iff every command in `apr_help_output` is present in
/// `registered`. Catches drift where apr exposes a command that the
/// registry doesn't track.
#[must_use]
pub fn verdict_from_no_undocumented_commands<S1, S2>(
    apr_help_output: &HashSet<String, S1>,
    registered: &HashSet<String, S2>,
) -> Cli002Verdict
where
    S1: std::hash::BuildHasher,
    S2: std::hash::BuildHasher,
{
    if apr_help_output.is_empty() { return Cli002Verdict::Fail; }
    for cmd in apr_help_output {
        if !registered.contains(cmd) { return Cli002Verdict::Fail; }
    }
    Cli002Verdict::Pass
}

// ===========================================================================
// FALSIFY-CLI-003 — Exit codes restricted to {0, 1, 2}
// ===========================================================================

pub const AC_CLI_003_ALLOWED_EXIT_CODES: [i32; 3] = [0, 1, 2];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cli003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_canonical_exit_code(exit_code: i32) -> Cli003Verdict {
    if AC_CLI_003_ALLOWED_EXIT_CODES.contains(&exit_code) {
        Cli003Verdict::Pass
    } else {
        Cli003Verdict::Fail
    }
}

// ===========================================================================
// FALSIFY-CLI-004 — `<cmd> --help` MUST NOT panic
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpOutcome {
    /// Help text printed cleanly with exit 0.
    PrintedSuccess,
    /// Non-zero exit but no panic (acceptable).
    NonZeroExitNoPanic,
    /// Process panicked (typically exit 101 on Rust).
    Panicked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cli004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_help_outcome(outcome: HelpOutcome) -> Cli004Verdict {
    match outcome {
        HelpOutcome::PrintedSuccess | HelpOutcome::NonZeroExitNoPanic => Cli004Verdict::Pass,
        HelpOutcome::Panicked => Cli004Verdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> HashSet<String> {
        items.iter().map(|i| (*i).to_string()).collect()
    }

    // ----- CLI-001 -----------------------------------------------------------

    #[test] fn cli001_pass_full_help_coverage() {
        let reg = s(&["run", "serve", "validate"]);
        let help = s(&["run", "serve", "validate"]);
        assert_eq!(verdict_from_help_responsive(&reg, &help), Cli001Verdict::Pass);
    }

    #[test] fn cli001_pass_help_superset() {
        // Registry is the gating contract; help having extra cmds is
        // a CLI-002 failure but not a CLI-001 failure.
        let reg = s(&["run", "serve"]);
        let help = s(&["run", "serve", "extra"]);
        assert_eq!(verdict_from_help_responsive(&reg, &help), Cli001Verdict::Pass);
    }

    #[test] fn cli001_fail_missing_help() {
        let reg = s(&["run", "serve", "validate"]);
        let help = s(&["run", "serve"]); // validate missing
        assert_eq!(verdict_from_help_responsive(&reg, &help), Cli001Verdict::Fail);
    }

    #[test] fn cli001_fail_empty_registry() {
        let reg = HashSet::new();
        let help = s(&["run"]);
        assert_eq!(verdict_from_help_responsive(&reg, &help), Cli001Verdict::Fail);
    }

    // ----- CLI-002 -----------------------------------------------------------

    #[test] fn cli002_pass_help_subset() {
        let help = s(&["run", "serve"]);
        let reg = s(&["run", "serve", "validate"]);
        assert_eq!(verdict_from_no_undocumented_commands(&help, &reg), Cli002Verdict::Pass);
    }

    #[test] fn cli002_fail_undocumented() {
        let help = s(&["run", "serve", "secret_cmd"]);
        let reg = s(&["run", "serve"]);
        assert_eq!(verdict_from_no_undocumented_commands(&help, &reg), Cli002Verdict::Fail);
    }

    #[test] fn cli002_fail_empty_help() {
        let help = HashSet::new();
        let reg = s(&["run"]);
        assert_eq!(verdict_from_no_undocumented_commands(&help, &reg), Cli002Verdict::Fail);
    }

    // ----- CLI-003 -----------------------------------------------------------

    #[test] fn cli003_pass_success() { assert_eq!(verdict_from_canonical_exit_code(0), Cli003Verdict::Pass); }
    #[test] fn cli003_pass_runtime_error() { assert_eq!(verdict_from_canonical_exit_code(1), Cli003Verdict::Pass); }
    #[test] fn cli003_pass_usage_error() { assert_eq!(verdict_from_canonical_exit_code(2), Cli003Verdict::Pass); }
    #[test] fn cli003_fail_panic_exit() {
        // Rust panic exit 101 — outside the canonical set.
        assert_eq!(verdict_from_canonical_exit_code(101), Cli003Verdict::Fail);
    }
    #[test] fn cli003_fail_signal_exit() {
        // SIGKILL via 128+9.
        assert_eq!(verdict_from_canonical_exit_code(137), Cli003Verdict::Fail);
    }
    #[test] fn cli003_fail_negative() {
        assert_eq!(verdict_from_canonical_exit_code(-1), Cli003Verdict::Fail);
    }
    #[test] fn cli003_fail_3() {
        assert_eq!(verdict_from_canonical_exit_code(3), Cli003Verdict::Fail);
    }

    // ----- CLI-004 -----------------------------------------------------------

    #[test] fn cli004_pass_printed() {
        assert_eq!(verdict_from_help_outcome(HelpOutcome::PrintedSuccess), Cli004Verdict::Pass);
    }
    #[test] fn cli004_pass_nonzero_no_panic() {
        assert_eq!(verdict_from_help_outcome(HelpOutcome::NonZeroExitNoPanic), Cli004Verdict::Pass);
    }
    #[test] fn cli004_fail_panicked() {
        // The regression: --help triggered a panic.
        assert_eq!(verdict_from_help_outcome(HelpOutcome::Panicked), Cli004Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_exit_codes() {
        assert_eq!(AC_CLI_003_ALLOWED_EXIT_CODES, [0, 1, 2]);
    }
}
