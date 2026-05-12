// SHIP-TWO-001 — `apr-cli-qa-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QA-008.
//
// Contract: `contracts/apr-cli-qa-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI QA gates).
//
// ## What FALSIFY-QA-008 says
//
//   rule: no phantom subcommands
//   prediction: "all advertised commands have real implementations"
//   test: `apr --help | awk '/^  [a-z]/{print $1}' |
//          while read c; do
//            apr $c --help >/dev/null 2>&1 || echo PHANTOM:$c
//          done | grep -q PHANTOM && exit 1 || exit 0`
//   if_fails: "subcommand listed but not implemented"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`commands_advertised`, `phantom_count`),
// Pass iff:
//
//   commands_advertised > 0 AND
//   phantom_count == 0 AND
//   phantom_count <= commands_advertised
//
// Zero-tolerance: a single phantom subcommand (advertised but
// without a real `--help`-responding implementation) is enough to
// trip the gate. Distinct from QA-001 which checks the registered
// count matches 58 — QA-008 checks the *registry-vs-impl* gap.

/// Binary verdict for `FALSIFY-QA-008`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qa008Verdict {
    /// `commands_advertised > 0` AND zero phantom subcommands
    /// observed.
    Pass,
    /// One or more of:
    /// - `commands_advertised == 0` (caller error — apr --help
    ///   listed no subcommands).
    /// - `phantom_count > 0` (one or more advertised commands
    ///   has no working --help).
    /// - `phantom_count > commands_advertised` (counter
    ///   corruption — partition violation).
    Fail,
}

/// Pure verdict function for `FALSIFY-QA-008`.
///
/// Inputs:
/// - `commands_advertised`: count of subcommand names extracted
///   from `apr --help` output (line awk pattern).
/// - `phantom_count`: count of those subcommands whose
///   `apr <name> --help` exited non-zero.
///
/// Pass iff:
/// 1. `commands_advertised > 0`,
/// 2. `phantom_count == 0`,
/// 3. `phantom_count <= commands_advertised` (counter sanity).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 58 commands advertised, all real — `Pass`:
/// ```
/// use aprender::format::qa_008::{
///     verdict_from_phantom_scan, Qa008Verdict,
/// };
/// let v = verdict_from_phantom_scan(58, 0);
/// assert_eq!(v, Qa008Verdict::Pass);
/// ```
///
/// 1 phantom in 58 — `Fail` (one is enough):
/// ```
/// use aprender::format::qa_008::{
///     verdict_from_phantom_scan, Qa008Verdict,
/// };
/// let v = verdict_from_phantom_scan(58, 1);
/// assert_eq!(v, Qa008Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_phantom_scan(
    commands_advertised: u64,
    phantom_count: u64,
) -> Qa008Verdict {
    if commands_advertised == 0 {
        return Qa008Verdict::Fail;
    }
    if phantom_count > commands_advertised {
        return Qa008Verdict::Fail;
    }
    if phantom_count == 0 {
        Qa008Verdict::Pass
    } else {
        Qa008Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — clean implementations.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_canonical_58_no_phantoms() {
        let v = verdict_from_phantom_scan(58, 0);
        assert_eq!(v, Qa008Verdict::Pass);
    }

    #[test]
    fn pass_one_command_zero_phantoms() {
        // Minimal CLI with one advertised command.
        let v = verdict_from_phantom_scan(1, 0);
        assert_eq!(v, Qa008Verdict::Pass);
    }

    #[test]
    fn pass_huge_clean_cli() {
        let v = verdict_from_phantom_scan(1_000_000, 0);
        assert_eq!(v, Qa008Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — phantom subcommands (zero-tolerance).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_phantom_in_58() {
        let v = verdict_from_phantom_scan(58, 1);
        assert_eq!(
            v,
            Qa008Verdict::Fail,
            "one phantom must Fail (no tolerance)"
        );
    }

    #[test]
    fn fail_handful_of_phantoms() {
        let v = verdict_from_phantom_scan(58, 5);
        assert_eq!(v, Qa008Verdict::Fail);
    }

    #[test]
    fn fail_half_phantoms() {
        let v = verdict_from_phantom_scan(58, 29);
        assert_eq!(v, Qa008Verdict::Fail);
    }

    #[test]
    fn fail_all_phantoms() {
        // Catastrophic: every advertised command is phantom.
        let v = verdict_from_phantom_scan(58, 58);
        assert_eq!(v, Qa008Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — empty advertised (caller error).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_advertised() {
        // `apr --help` listed no subcommands — registry empty.
        let v = verdict_from_phantom_scan(0, 0);
        assert_eq!(
            v,
            Qa008Verdict::Fail,
            "zero advertised must Fail (vacuous Pass refused)"
        );
    }

    #[test]
    fn fail_zero_advertised_with_phantoms() {
        // Counter corruption: zero advertised, nonzero phantoms.
        let v = verdict_from_phantom_scan(0, 5);
        assert_eq!(v, Qa008Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — partition violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_phantoms_exceed_advertised() {
        let v = verdict_from_phantom_scan(58, 100);
        assert_eq!(
            v,
            Qa008Verdict::Fail,
            "phantoms > advertised must Fail (partition violation)"
        );
    }

    #[test]
    fn fail_huge_phantoms_with_smaller_advertised() {
        let v = verdict_from_phantom_scan(58, u64::MAX);
        assert_eq!(v, Qa008Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Boundary sweep — phantom-count sweep at 58 advertised.
    // -------------------------------------------------------------------------
    #[test]
    fn phantom_count_sweep_at_58_advertised() {
        let advertised = 58_u64;
        let probes: Vec<(u64, Qa008Verdict)> = vec![
            (0, Qa008Verdict::Pass),
            (1, Qa008Verdict::Fail),
            (5, Qa008Verdict::Fail),
            (29, Qa008Verdict::Fail),
            (57, Qa008Verdict::Fail),
            (58, Qa008Verdict::Fail),
            (59, Qa008Verdict::Fail), // partition violation
        ];
        for (phantoms, expected) in probes {
            let v = verdict_from_phantom_scan(advertised, phantoms);
            assert_eq!(
                v, expected,
                "advertised=58 phantoms={phantoms} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Domain — zero-tolerance property.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_phantom_count_is_exactly_zero() {
        for advertised in [1_u64, 10, 58, 100, 1_000_000] {
            let v_pass = verdict_from_phantom_scan(advertised, 0);
            assert_eq!(v_pass, Qa008Verdict::Pass, "advertised={advertised}");

            let v_fail = verdict_from_phantom_scan(advertised, 1);
            assert_eq!(
                v_fail,
                Qa008Verdict::Fail,
                "advertised={advertised} with one phantom"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — apr CLI scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_apr_clean_canonical_release() {
        // Canonical 58-command release with no phantoms.
        let v = verdict_from_phantom_scan(58, 0);
        assert_eq!(v, Qa008Verdict::Pass);
    }

    #[test]
    fn fail_apr_with_unwired_subcommand() {
        // Realistic regression: a subcommand was added to the
        // registry but the dispatch arm was forgotten — `apr <cmd>`
        // returns "unimplemented" exit 1.
        let v = verdict_from_phantom_scan(58, 1);
        assert_eq!(v, Qa008Verdict::Fail);
    }

    #[test]
    fn pass_minimal_features_build() {
        // Minimal-features build advertises fewer commands but all
        // are real.
        let v = verdict_from_phantom_scan(20, 0);
        assert_eq!(v, Qa008Verdict::Pass);
    }
}
