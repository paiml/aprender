// SHIP-TWO-001 — `apr-cli-publish-extra-v1` algorithm-level
// PARTIAL discharge for FALSIFY-PUB-EXTRA-001 + 007.
//
// Contract: `contracts/apr-cli-publish-extra-v1.yaml`.

// ===========================================================================
// PUB-EXTRA-001 — apr publish --manifest dry-run accepts valid YAML
// ===========================================================================

/// Binary verdict for `FALSIFY-PUB-EXTRA-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubExtra001Verdict {
    /// `apr publish --manifest <yaml> --dry-run` exited 0.
    Pass,
    /// Non-zero exit (manifest consumption broken — ship blocked).
    Fail,
}

/// Pure verdict function for `FALSIFY-PUB-EXTRA-001`.
///
/// Pass iff `exit_code == 0`.
#[must_use]
pub fn verdict_from_manifest_dry_run_exit(exit_code: i32) -> PubExtra001Verdict {
    if exit_code == 0 {
        PubExtra001Verdict::Pass
    } else {
        PubExtra001Verdict::Fail
    }
}

// ===========================================================================
// PUB-EXTRA-007 — HF repo contains apr + safetensors + gguf (3-format)
// ===========================================================================

/// Required number of distinct format extensions in the published HF
/// repo.
///
/// Per contract: `\\.(apr|safetensors|gguf)$` matched count must equal
/// 3 — one of each ecosystem's tooling.
pub const AC_PUB_EXTRA_007_REQUIRED_FORMATS: u32 = 3;

/// Binary verdict for `FALSIFY-PUB-EXTRA-007`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubExtra007Verdict {
    /// HF repo file listing contains exactly one .apr, one
    /// .safetensors, and one .gguf file.
    Pass,
    /// Counts differ from `(1, 1, 1)`. Includes:
    /// - any format missing
    /// - any format duplicated (two .apr, etc.)
    /// - extra files of irrelevant extensions don't matter for
    ///   this verdict; only the three target counts.
    Fail,
}

/// Pure verdict function for `FALSIFY-PUB-EXTRA-007`.
///
/// Pass iff `apr_count == 1 AND safetensors_count == 1 AND
/// gguf_count == 1`.
#[must_use]
pub fn verdict_from_three_format_repo(
    apr_count: u32,
    safetensors_count: u32,
    gguf_count: u32,
) -> PubExtra007Verdict {
    if apr_count == 1 && safetensors_count == 1 && gguf_count == 1 {
        PubExtra007Verdict::Pass
    } else {
        PubExtra007Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // PUB-EXTRA-001 tests
    // -------------------------------------------------------------------------
    #[test]
    fn p001_pass_dry_run_exit_zero() {
        let v = verdict_from_manifest_dry_run_exit(0);
        assert_eq!(v, PubExtra001Verdict::Pass);
    }

    #[test]
    fn p001_fail_dry_run_exit_one() {
        let v = verdict_from_manifest_dry_run_exit(1);
        assert_eq!(v, PubExtra001Verdict::Fail);
    }

    #[test]
    fn p001_fail_clap_error_exit_two() {
        let v = verdict_from_manifest_dry_run_exit(2);
        assert_eq!(v, PubExtra001Verdict::Fail);
    }

    #[test]
    fn p001_fail_panic_101() {
        let v = verdict_from_manifest_dry_run_exit(101);
        assert_eq!(v, PubExtra001Verdict::Fail);
    }

    #[test]
    fn p001_fail_negative_exit() {
        let v = verdict_from_manifest_dry_run_exit(-1);
        assert_eq!(v, PubExtra001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // PUB-EXTRA-007 tests
    // -------------------------------------------------------------------------
    #[test]
    fn p007_provenance_required_formats_is_3() {
        assert_eq!(AC_PUB_EXTRA_007_REQUIRED_FORMATS, 3);
    }

    #[test]
    fn p007_pass_canonical_one_each() {
        let v = verdict_from_three_format_repo(1, 1, 1);
        assert_eq!(v, PubExtra007Verdict::Pass);
    }

    #[test]
    fn p007_fail_apr_missing() {
        let v = verdict_from_three_format_repo(0, 1, 1);
        assert_eq!(v, PubExtra007Verdict::Fail);
    }

    #[test]
    fn p007_fail_safetensors_missing() {
        let v = verdict_from_three_format_repo(1, 0, 1);
        assert_eq!(v, PubExtra007Verdict::Fail);
    }

    #[test]
    fn p007_fail_gguf_missing() {
        let v = verdict_from_three_format_repo(1, 1, 0);
        assert_eq!(v, PubExtra007Verdict::Fail);
    }

    #[test]
    fn p007_fail_all_missing() {
        let v = verdict_from_three_format_repo(0, 0, 0);
        assert_eq!(v, PubExtra007Verdict::Fail);
    }

    #[test]
    fn p007_fail_duplicate_apr() {
        let v = verdict_from_three_format_repo(2, 1, 1);
        assert_eq!(v, PubExtra007Verdict::Fail);
    }

    #[test]
    fn p007_fail_two_each() {
        // Multiple shipped versions in same repo (sharded?)
        let v = verdict_from_three_format_repo(2, 2, 2);
        assert_eq!(v, PubExtra007Verdict::Fail);
    }

    #[test]
    fn matrix_only_one_one_one_passes() {
        // Exhaustive 0..=2 cube: only (1,1,1) passes.
        for a in 0..=2 {
            for s in 0..=2 {
                for g in 0..=2 {
                    let v = verdict_from_three_format_repo(a, s, g);
                    let expected = if a == 1 && s == 1 && g == 1 {
                        PubExtra007Verdict::Pass
                    } else {
                        PubExtra007Verdict::Fail
                    };
                    assert_eq!(v, expected, "apr={a} safetensors={s} gguf={g}");
                }
            }
        }
    }
}
