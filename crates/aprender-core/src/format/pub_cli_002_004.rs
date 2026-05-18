// SHIP-TWO-001 — `apr-cli-publish-v1` algorithm-level PARTIAL
// discharge for FALSIFY-PUB-CLI-002 AND FALSIFY-PUB-CLI-004.
//
// Contract: `contracts/apr-cli-publish-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// (apr CLI publish gate; cross-cutting requirement for MODEL-1 +
// MODEL-2 shipping).
//
// ## What FALSIFY-PUB-CLI-002 says
//
//   rule: cargo install aprender works
//   prediction: "cargo install aprender exits 0 on clean machine"
//   if_fails: "golden path broken"
//
// ## What FALSIFY-PUB-CLI-004 says
//
//   rule: full features compile locally
//   prediction: "cargo check -p apr-cli --all-features exits 0"
//   if_fails: "workspace build broken"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule for BOTH gates: given `exit_code` of either
// `cargo install aprender` (PUB-CLI-002) or
// `cargo check -p apr-cli --all-features` (PUB-CLI-004), Pass iff:
//
//   exit_code == 0
//
// Both gates share an identical verdict shape: pure exit-code
// equality. They differ only in which subprocess produces the
// exit code. Composing them into one verdict module reflects the
// shared algorithm; the integration-level wiring (which command
// to invoke, how to capture the exit) is FULL_DISCHARGE.

/// Binary verdict for `FALSIFY-PUB-CLI-002` and `FALSIFY-PUB-CLI-004`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubCli002004Verdict {
    /// `exit_code == 0` (cargo command succeeded).
    Pass,
    /// `exit_code != 0` (compile/install failed; possibly with
    /// negative or out-of-range value indicating a panic /
    /// signal-driven exit).
    Fail,
}

/// Pure verdict function for `FALSIFY-PUB-CLI-002` and
/// `FALSIFY-PUB-CLI-004`.
///
/// Inputs:
/// - `exit_code`: process exit code of the relevant cargo command:
///     * PUB-CLI-002: `cargo install aprender --force`
///     * PUB-CLI-004: `cargo check -p apr-cli --all-features`
///
/// Pass iff `exit_code == 0`. Otherwise `Fail`.
///
/// # Examples
///
/// Cargo command succeeded — `Pass`:
/// ```
/// use aprender::format::pub_cli_002_004::{
///     verdict_from_cargo_command_exit, PubCli002004Verdict,
/// };
/// let v = verdict_from_cargo_command_exit(0);
/// assert_eq!(v, PubCli002004Verdict::Pass);
/// ```
///
/// Cargo command failed — `Fail`:
/// ```
/// use aprender::format::pub_cli_002_004::{
///     verdict_from_cargo_command_exit, PubCli002004Verdict,
/// };
/// let v = verdict_from_cargo_command_exit(101);
/// assert_eq!(v, PubCli002004Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_cargo_command_exit(exit_code: i32) -> PubCli002004Verdict {
    if exit_code == 0 {
        PubCli002004Verdict::Pass
    } else {
        PubCli002004Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — only exit 0 passes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_exit_zero() {
        let v = verdict_from_cargo_command_exit(0);
        assert_eq!(v, PubCli002004Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — non-zero exit codes (cargo failure modes).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_exit_one_compile_error() {
        // Cargo's typical "compile error" exit code.
        let v = verdict_from_cargo_command_exit(1);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    #[test]
    fn fail_exit_two_clap_error() {
        let v = verdict_from_cargo_command_exit(2);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    #[test]
    fn fail_exit_101_rustc_panic() {
        // Rust panic exit code.
        let v = verdict_from_cargo_command_exit(101);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    #[test]
    fn fail_exit_137_sigkill() {
        // 128 + 9 = SIGKILL (e.g., OOM during link).
        let v = verdict_from_cargo_command_exit(137);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    #[test]
    fn fail_exit_143_sigterm() {
        // 128 + 15 = SIGTERM.
        let v = verdict_from_cargo_command_exit(143);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — domain edges (negative, max).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_exit_negative_one() {
        let v = verdict_from_cargo_command_exit(-1);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    #[test]
    fn fail_exit_i32_min() {
        let v = verdict_from_cargo_command_exit(i32::MIN);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    #[test]
    fn fail_exit_i32_max() {
        let v = verdict_from_cargo_command_exit(i32::MAX);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    #[test]
    fn fail_exit_255_posix_max() {
        let v = verdict_from_cargo_command_exit(255);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    #[test]
    fn fail_exit_256_above_posix_cap() {
        // 256 is the post-truncation wrap of 0; in shell, this
        // becomes 0. But on raw exit_code from waitpid, 256 is a
        // valid distinct value indicating a signal-related state
        // bit. Either way, non-zero must Fail.
        let v = verdict_from_cargo_command_exit(256);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Boundary sweep — sweep around zero.
    // -------------------------------------------------------------------------
    #[test]
    fn exit_code_sweep_around_zero() {
        let probes: Vec<(i32, PubCli002004Verdict)> = vec![
            (-3, PubCli002004Verdict::Fail),
            (-2, PubCli002004Verdict::Fail),
            (-1, PubCli002004Verdict::Fail),
            (0, PubCli002004Verdict::Pass),
            (1, PubCli002004Verdict::Fail),
            (2, PubCli002004Verdict::Fail),
            (3, PubCli002004Verdict::Fail),
        ];
        for (exit, expected) in probes {
            let v = verdict_from_cargo_command_exit(exit);
            assert_eq!(v, expected, "exit={exit} expected {expected:?}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 5: Property — Pass iff exit_code == 0 across all i32.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_exit_is_exactly_zero_at_canonical_codes() {
        // Every i32 must Fail except 0.
        for exit in [
            i32::MIN,
            -1000,
            -1,
            0,
            1,
            2,
            101,
            137,
            143,
            255,
            1000,
            i32::MAX,
        ] {
            let v = verdict_from_cargo_command_exit(exit);
            let expected = if exit == 0 {
                PubCli002004Verdict::Pass
            } else {
                PubCli002004Verdict::Fail
            };
            assert_eq!(v, expected, "exit={exit}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — both gates' canonical scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_pub_cli_002_clean_install() {
        // `cargo install aprender --force` succeeded.
        let v = verdict_from_cargo_command_exit(0);
        assert_eq!(v, PubCli002004Verdict::Pass);
    }

    #[test]
    fn pass_pub_cli_004_clean_check() {
        // `cargo check -p apr-cli --all-features` succeeded.
        let v = verdict_from_cargo_command_exit(0);
        assert_eq!(v, PubCli002004Verdict::Pass);
    }

    #[test]
    fn fail_pub_cli_002_cyclic_dep_chain() {
        // The if_fails class for PUB-CLI-002: "cargo install will
        // hit cyclic dep chain". Cargo's typical exit for this.
        let v = verdict_from_cargo_command_exit(101);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    #[test]
    fn fail_pub_cli_004_workspace_build_broken() {
        // The if_fails class for PUB-CLI-004: "workspace build broken".
        let v = verdict_from_cargo_command_exit(1);
        assert_eq!(v, PubCli002004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Symmetry — gate 002 and gate 004 share the verdict.
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_is_gate_agnostic() {
        // Whether the exit_code came from PUB-CLI-002's
        // `cargo install` OR PUB-CLI-004's `cargo check`,
        // the verdict is identical for the same exit_code.
        // This is the algorithm-level identity.
        for exit in [0_i32, 1, 2, 101, 137, -1, i32::MAX] {
            let v_install = verdict_from_cargo_command_exit(exit);
            let v_check = verdict_from_cargo_command_exit(exit);
            assert_eq!(
                v_install, v_check,
                "verdict must be gate-agnostic for exit={exit}"
            );
        }
    }
}
