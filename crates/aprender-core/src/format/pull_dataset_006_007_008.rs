// SHIP-TWO-001 — `apr-cli-pull-dataset-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-PULL-DATASET-006 + 007 + 008.
// Closes 8/8 sweep.
//
// Contract: `contracts/apr-cli-pull-dataset-v1.yaml`.

// ===========================================================================
// PULL-DATASET-006 + 007 — exit-code-zero (3-surface drift / pv validate)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDataset006007Verdict {
    Pass,
    Fail,
}

/// Pure verdict function shared between PULL-DATASET-006 (cargo
/// test cli_commands registered_commands) and PULL-DATASET-007
/// (pv validate).
///
/// Pass iff `exit_code == 0`.
#[must_use]
pub fn verdict_from_cargo_or_pv_exit(exit_code: i32) -> PullDataset006007Verdict {
    if exit_code == 0 {
        PullDataset006007Verdict::Pass
    } else {
        PullDataset006007Verdict::Fail
    }
}

// ===========================================================================
// PULL-DATASET-008 — deprecated namespace count == 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDataset008Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-PULL-DATASET-008`.
///
/// Pass iff `deprecated_match_count == 0`.
///
/// Per `feedback_stack_tool_extension_not_cli_shim`: `apr` is
/// canonical post-APR-MONO. Substring matches for
/// `batuta hf pull` or `huggingface-cli download` (after filtering
/// out negative-context lines like "deprecated", "wrong",
/// "MUST NOT", "muda", "forbidden") indicate the spec/scripts
/// reverted to old/non-stack tooling.
#[must_use]
pub fn verdict_from_deprecated_namespace_count(deprecated_match_count: u64) -> PullDataset008Verdict {
    if deprecated_match_count == 0 {
        PullDataset008Verdict::Pass
    } else {
        PullDataset008Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // PULL-DATASET-006 + 007 -----------------------------------------------------
    #[test]
    fn p006_007_pass_exit_zero() {
        assert_eq!(verdict_from_cargo_or_pv_exit(0), PullDataset006007Verdict::Pass);
    }

    #[test]
    fn p006_007_fail_exit_one() {
        assert_eq!(verdict_from_cargo_or_pv_exit(1), PullDataset006007Verdict::Fail);
    }

    #[test]
    fn p006_007_fail_panic_101() {
        assert_eq!(verdict_from_cargo_or_pv_exit(101), PullDataset006007Verdict::Fail);
    }

    #[test]
    fn p006_007_fail_negative() {
        assert_eq!(verdict_from_cargo_or_pv_exit(-1), PullDataset006007Verdict::Fail);
    }

    #[test]
    fn p006_007_pass_iff_exit_is_zero() {
        for exit in [-1_i32, 0, 1, 2, 101] {
            let v = verdict_from_cargo_or_pv_exit(exit);
            let expected = if exit == 0 {
                PullDataset006007Verdict::Pass
            } else {
                PullDataset006007Verdict::Fail
            };
            assert_eq!(v, expected, "exit={exit}");
        }
    }

    // PULL-DATASET-008 -----------------------------------------------------------
    #[test]
    fn p008_pass_zero_deprecated_matches() {
        assert_eq!(verdict_from_deprecated_namespace_count(0), PullDataset008Verdict::Pass);
    }

    #[test]
    fn p008_fail_one_deprecated_match() {
        assert_eq!(verdict_from_deprecated_namespace_count(1), PullDataset008Verdict::Fail);
    }

    #[test]
    fn p008_fail_many_deprecated_matches() {
        assert_eq!(verdict_from_deprecated_namespace_count(100), PullDataset008Verdict::Fail);
    }

    #[test]
    fn p008_pass_at_huge_zero_count() {
        // Sanity: u64::MAX still passes if it's exactly 0.
        // (Trivially: this is just the same Pass case; included
        //  to document that the verdict has no upper bound.)
        assert_eq!(verdict_from_deprecated_namespace_count(0), PullDataset008Verdict::Pass);
    }
}
