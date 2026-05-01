// SHIP-TWO-001 — `apr-cli-distill-train-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-DISTILL-TRAIN-007 + 008.
// Closes 9/9 distill-train sweep.
//
// Contract: `contracts/apr-cli-distill-train-v1.yaml`.
//
// Both gates are exit-code-only verdicts on different cargo
// commands. Same shape as `pub_cli_002_004` and `cmd_safety`'s
// CMD-SAFETY-003 (exit-code conjunctive bundles).

/// Binary verdict for both `FALSIFY-APR-DISTILL-TRAIN-007` and
/// `FALSIFY-APR-DISTILL-TRAIN-008`. Same algorithm-level shape:
/// pure exit-code-equals-zero predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillTrain007008Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for both `FALSIFY-APR-DISTILL-TRAIN-007`
/// (`pv validate contracts/apr-cli-distill-train-v1.yaml` exits 0)
/// AND `FALSIFY-APR-DISTILL-TRAIN-008` (`cargo test -p apr-cli
/// --test cli_commands registered_commands` exits 0).
///
/// Pass iff `exit_code == 0`. Otherwise `Fail`.
#[must_use]
pub fn verdict_from_pv_or_cargo_exit(exit_code: i32) -> DistillTrain007008Verdict {
    if exit_code == 0 {
        DistillTrain007008Verdict::Pass
    } else {
        DistillTrain007008Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_exit_zero() {
        assert_eq!(verdict_from_pv_or_cargo_exit(0), DistillTrain007008Verdict::Pass);
    }

    #[test]
    fn fail_exit_one() {
        assert_eq!(verdict_from_pv_or_cargo_exit(1), DistillTrain007008Verdict::Fail);
    }

    #[test]
    fn fail_exit_two_clap() {
        assert_eq!(verdict_from_pv_or_cargo_exit(2), DistillTrain007008Verdict::Fail);
    }

    #[test]
    fn fail_panic_101() {
        assert_eq!(verdict_from_pv_or_cargo_exit(101), DistillTrain007008Verdict::Fail);
    }

    #[test]
    fn fail_negative_exit() {
        assert_eq!(verdict_from_pv_or_cargo_exit(-1), DistillTrain007008Verdict::Fail);
    }

    #[test]
    fn fail_i32_max() {
        assert_eq!(verdict_from_pv_or_cargo_exit(i32::MAX), DistillTrain007008Verdict::Fail);
    }

    #[test]
    fn fail_i32_min() {
        assert_eq!(verdict_from_pv_or_cargo_exit(i32::MIN), DistillTrain007008Verdict::Fail);
    }

    #[test]
    fn pass_iff_exit_is_exactly_zero_at_canonical_codes() {
        for exit in [-1000_i32, -1, 0, 1, 2, 101, 137, 143, 255, 1000, i32::MAX, i32::MIN] {
            let v = verdict_from_pv_or_cargo_exit(exit);
            let expected = if exit == 0 {
                DistillTrain007008Verdict::Pass
            } else {
                DistillTrain007008Verdict::Fail
            };
            assert_eq!(v, expected, "exit={exit}");
        }
    }

    #[test]
    fn verdict_is_gate_agnostic() {
        // Both TRAIN-007 (pv validate) and TRAIN-008 (cargo test)
        // share the verdict for any given exit_code.
        for exit in [0_i32, 1, 2, 101] {
            let v_pv = verdict_from_pv_or_cargo_exit(exit);
            let v_cargo = verdict_from_pv_or_cargo_exit(exit);
            assert_eq!(v_pv, v_cargo);
        }
    }
}
