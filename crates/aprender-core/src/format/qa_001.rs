// SHIP-TWO-001 — `apr-cli-qa-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QA-001.
//
// Contract: `contracts/apr-cli-qa-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI QA gates; cross-cutting requirement for MODEL-1 + MODEL-2
// shipping).
//
// ## What FALSIFY-QA-001 says
//
//   rule: all 58 commands respond to --help
//   prediction: "every registered command exits 0 on --help"
//   test: "cargo test -p apr-cli --test cli_commands"
//   if_fails: "phantom subcommand or broken dispatch"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`commands_tested`,
// `commands_passing_help_smoke`), Pass iff:
//
//   commands_tested == AC_QA_001_EXPECTED_COMMAND_COUNT (58) AND
//   commands_passing_help_smoke == commands_tested
//
// The 58-count is pinned per the canonical apr CLI surface.
// Strict equality on both axes catches:
// - Phantom subcommand: tested = 59, passing = 59 → fails
//   (over-count: someone added a subcommand without bumping the
//   contract).
// - Broken dispatch: tested = 58, passing = 57 → fails (one
//   subcommand's --help broke).
// - Missing subcommand: tested = 57 → fails (a subcommand is
//   missing from the registry).

/// Expected number of registered apr CLI commands.
///
/// Per spec §26.8 / CLAUDE.md: the canonical apr CLI surface is
/// 58 subcommands (57 original + `mcp` added in PR #864).
/// Pinning this constant catches a regression where a subcommand
/// is added/dropped without the contract being bumped.
pub const AC_QA_001_EXPECTED_COMMAND_COUNT: u64 = 58;

/// Binary verdict for `FALSIFY-QA-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qa001Verdict {
    /// `commands_tested == 58` AND every tested command exited
    /// 0 on `--help`.
    Pass,
    /// One or more of:
    /// - `commands_tested != 58` (count drift — phantom or
    ///   missing subcommand).
    /// - `commands_passing_help_smoke != commands_tested` (one
    ///   or more --help calls failed; broken dispatch).
    /// - `commands_passing_help_smoke > commands_tested`
    ///   (counter corruption — partition violation).
    Fail,
}

/// Pure verdict function for `FALSIFY-QA-001`.
///
/// Inputs:
/// - `commands_tested`: number of commands the test harness
///   invoked with `--help`.
/// - `commands_passing_help_smoke`: number of those that exited
///   0.
///
/// Pass iff:
/// 1. `commands_tested == 58`,
/// 2. `commands_passing_help_smoke == commands_tested`,
/// 3. `commands_passing_help_smoke <= commands_tested` (counter
///    sanity).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// All 58 commands respond — `Pass`:
/// ```
/// use aprender::format::qa_001::{
///     verdict_from_help_subcommand_smoke, Qa001Verdict,
/// };
/// let v = verdict_from_help_subcommand_smoke(58, 58);
/// assert_eq!(v, Qa001Verdict::Pass);
/// ```
///
/// One subcommand's --help broke — `Fail`:
/// ```
/// use aprender::format::qa_001::{
///     verdict_from_help_subcommand_smoke, Qa001Verdict,
/// };
/// let v = verdict_from_help_subcommand_smoke(58, 57);
/// assert_eq!(v, Qa001Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_help_subcommand_smoke(
    commands_tested: u64,
    commands_passing_help_smoke: u64,
) -> Qa001Verdict {
    if commands_tested != AC_QA_001_EXPECTED_COMMAND_COUNT {
        return Qa001Verdict::Fail;
    }
    if commands_passing_help_smoke > commands_tested {
        return Qa001Verdict::Fail;
    }
    if commands_passing_help_smoke == commands_tested {
        Qa001Verdict::Pass
    } else {
        Qa001Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — 58-command canonical count.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_expected_command_count_is_58() {
        assert_eq!(AC_QA_001_EXPECTED_COMMAND_COUNT, 58);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — canonical 58 / 58.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_all_58_commands_help_smoke() {
        let v = verdict_from_help_subcommand_smoke(58, 58);
        assert_eq!(v, Qa001Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — broken dispatch (passing < tested).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_command_broken() {
        let v = verdict_from_help_subcommand_smoke(58, 57);
        assert_eq!(
            v,
            Qa001Verdict::Fail,
            "1 broken --help must Fail (broken dispatch)"
        );
    }

    #[test]
    fn fail_handful_broken() {
        let v = verdict_from_help_subcommand_smoke(58, 50);
        assert_eq!(v, Qa001Verdict::Fail);
    }

    #[test]
    fn fail_all_broken() {
        let v = verdict_from_help_subcommand_smoke(58, 0);
        assert_eq!(v, Qa001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — count drift (tested != 58).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_phantom_subcommand_added_57_to_59() {
        // A subcommand was added without bumping the contract.
        let v = verdict_from_help_subcommand_smoke(59, 59);
        assert_eq!(
            v,
            Qa001Verdict::Fail,
            "phantom subcommand (59 tested) must Fail"
        );
    }

    #[test]
    fn fail_subcommand_dropped() {
        // A subcommand was removed without bumping the contract.
        let v = verdict_from_help_subcommand_smoke(57, 57);
        assert_eq!(
            v,
            Qa001Verdict::Fail,
            "missing subcommand (57 tested) must Fail"
        );
    }

    #[test]
    fn fail_zero_tested() {
        // Test harness produced no work — registry empty?
        let v = verdict_from_help_subcommand_smoke(0, 0);
        assert_eq!(v, Qa001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — counter / partition violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_passing_exceeds_tested() {
        // Counter corruption: passing > tested.
        let v = verdict_from_help_subcommand_smoke(58, 100);
        assert_eq!(
            v,
            Qa001Verdict::Fail,
            "passing > tested must Fail (partition violation)"
        );
    }

    #[test]
    fn fail_passing_huge_with_canonical_tested() {
        let v = verdict_from_help_subcommand_smoke(58, u64::MAX);
        assert_eq!(v, Qa001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep — passing-count sweep at fixed 58 tested.
    // -------------------------------------------------------------------------
    #[test]
    fn passing_count_sweep_at_58_tested() {
        let probes: Vec<(u64, Qa001Verdict)> = vec![
            (0, Qa001Verdict::Fail),
            (1, Qa001Verdict::Fail),
            (29, Qa001Verdict::Fail),
            (56, Qa001Verdict::Fail),
            (57, Qa001Verdict::Fail), // off-by-one
            (58, Qa001Verdict::Pass), // canonical pass
            (59, Qa001Verdict::Fail), // partition violation
            (100, Qa001Verdict::Fail),
        ];
        for (passing, expected) in probes {
            let v = verdict_from_help_subcommand_smoke(58, passing);
            assert_eq!(
                v, expected,
                "tested=58 passing={passing} expected {expected:?}"
            );
        }
    }

    #[test]
    fn tested_count_sweep_at_perfect_help_passing() {
        // Sweep tested-count when all tested commands pass.
        let probes: Vec<(u64, Qa001Verdict)> = vec![
            (0, Qa001Verdict::Fail),
            (1, Qa001Verdict::Fail),
            (50, Qa001Verdict::Fail),
            (57, Qa001Verdict::Fail),
            (58, Qa001Verdict::Pass), // exact pin
            (59, Qa001Verdict::Fail),
            (100, Qa001Verdict::Fail),
        ];
        for (n, expected) in probes {
            let v = verdict_from_help_subcommand_smoke(n, n);
            assert_eq!(v, expected, "tested={n} passing={n} expected {expected:?}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Composite — exact-match property at canonical scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_canonical_58_with_all_passing() {
        // Property: only the exact (58, 58) cell passes; all
        // others fail.
        let cases: Vec<(u64, u64, Qa001Verdict)> = vec![
            (58, 58, Qa001Verdict::Pass), // canonical pass
            (58, 57, Qa001Verdict::Fail), // 1 broken
            (57, 57, Qa001Verdict::Fail), // missing 1
            (59, 59, Qa001Verdict::Fail), // phantom
            (58, 0, Qa001Verdict::Fail),  // all broken
            (0, 0, Qa001Verdict::Fail),   // empty registry
            (58, 59, Qa001Verdict::Fail), // partition violation
        ];
        for (tested, passing, expected) in cases {
            let v = verdict_from_help_subcommand_smoke(tested, passing);
            assert_eq!(
                v, expected,
                "tested={tested} passing={passing} expected {expected:?}"
            );
        }
    }
}
