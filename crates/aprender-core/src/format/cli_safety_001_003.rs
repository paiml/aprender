// `apr-cli-safety-v1` algorithm-level PARTIAL discharge for
// FALSIFY-CLI-001..003.
//
// Contract: `contracts/apr-cli-safety-v1.yaml`.
//
// Module name `cli_safety_001_003` disambiguates from
// `apr-cli-commands-v1` which also uses FALSIFY-CLI-001..004 (already
// bound at task #278 in the `cli_001_004` module).
//
// CLI-001: validate exits non-zero on score < 50 (no silent failure)
// CLI-002: --offline rejects hf:// network sources before any IO
// CLI-003: encrypt rejects .enc input (idempotency / no double encryption)

/// CLI-001: validation score threshold.
pub const AC_CLI_VALIDATE_SCORE_THRESHOLD: u32 = 50;
/// CLI-001: exit code for low-score validation result.
pub const AC_CLI_VALIDATE_LOW_SCORE_EXIT: i32 = 5;

/// CLI-002: prefixes that count as "network sources" for the offline
/// guard.
pub const AC_CLI_NETWORK_PREFIXES: &[&str] = &["hf://", "http://", "https://"];

/// CLI-003: file extensions already-encrypted (subject to reject).
pub const AC_CLI_ENCRYPTED_EXTENSIONS: &[&str] = &[".enc"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliSafetyVerdict {
    Pass,
    Fail,
}

/// CLI-001: validate exit-code monotonicity.
///
/// Pass iff:
///   - score < 50 → exit_code == 5
///   - score >= 50 → exit_code == 0
#[must_use]
pub fn verdict_from_validate_exit(score: u32, exit_code: i32) -> CliSafetyVerdict {
    if score < AC_CLI_VALIDATE_SCORE_THRESHOLD {
        if exit_code == AC_CLI_VALIDATE_LOW_SCORE_EXIT {
            CliSafetyVerdict::Pass
        } else {
            CliSafetyVerdict::Fail
        }
    } else if exit_code == 0 {
        CliSafetyVerdict::Pass
    } else {
        CliSafetyVerdict::Fail
    }
}

/// CLI-002: --offline rejects network sources.
///
/// Pass iff:
///   - offline_flag == true AND source has a network prefix → rejected_before_io
///   - offline_flag == false → not constrained, vacuous Pass
#[must_use]
pub fn verdict_from_offline_blocks_network(
    offline_flag: bool,
    source: &str,
    rejected_before_io: bool,
) -> CliSafetyVerdict {
    if !offline_flag {
        return CliSafetyVerdict::Pass;
    }
    if source.is_empty() {
        return CliSafetyVerdict::Fail;
    }
    let is_network = AC_CLI_NETWORK_PREFIXES
        .iter()
        .any(|p| source.starts_with(p));
    if is_network && !rejected_before_io {
        return CliSafetyVerdict::Fail;
    }
    if !is_network {
        // Local source, offline mode → no network access, vacuously Pass.
        return CliSafetyVerdict::Pass;
    }
    CliSafetyVerdict::Pass
}

/// CLI-003: encrypt rejects already-encrypted input.
///
/// Pass iff:
///   - input ends with `.enc` → encrypt_was_rejected == true
///   - input doesn't end with `.enc` → not constrained, vacuous Pass
#[must_use]
pub fn verdict_from_encrypt_rejects_enc(
    input_path: &str,
    encrypt_was_rejected: bool,
) -> CliSafetyVerdict {
    if input_path.is_empty() {
        return CliSafetyVerdict::Fail;
    }
    let already_encrypted = AC_CLI_ENCRYPTED_EXTENSIONS
        .iter()
        .any(|ext| input_path.ends_with(ext));
    if already_encrypted && !encrypt_was_rejected {
        return CliSafetyVerdict::Fail;
    }
    CliSafetyVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_CLI_VALIDATE_SCORE_THRESHOLD, 50);
        assert_eq!(AC_CLI_VALIDATE_LOW_SCORE_EXIT, 5);
        assert_eq!(AC_CLI_NETWORK_PREFIXES, &["hf://", "http://", "https://"]);
        assert_eq!(AC_CLI_ENCRYPTED_EXTENSIONS, &[".enc"]);
    }

    // -----------------------------------------------------------------
    // Section 2: CLI-001 validate exit.
    // -----------------------------------------------------------------
    #[test]
    fn fcli001_pass_low_score_exits_5() {
        let v = verdict_from_validate_exit(3, 5);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    #[test]
    fn fcli001_pass_high_score_exits_0() {
        let v = verdict_from_validate_exit(95, 0);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    #[test]
    fn fcli001_pass_at_threshold_exits_0() {
        // score=50 is "high" per `< 50` rule
        let v = verdict_from_validate_exit(50, 0);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    #[test]
    fn fcli001_fail_low_score_exits_0() {
        // The exact regression — silent failure.
        let v = verdict_from_validate_exit(3, 0);
        assert_eq!(v, CliSafetyVerdict::Fail);
    }

    #[test]
    fn fcli001_fail_low_score_exits_other() {
        let v = verdict_from_validate_exit(3, 1);
        assert_eq!(v, CliSafetyVerdict::Fail);
    }

    #[test]
    fn fcli001_fail_high_score_exits_5() {
        // false positive — model is good but apr says invalid
        let v = verdict_from_validate_exit(95, 5);
        assert_eq!(v, CliSafetyVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: CLI-002 offline blocks network.
    // -----------------------------------------------------------------
    #[test]
    fn fcli002_pass_offline_hf_rejected() {
        let v = verdict_from_offline_blocks_network(true, "hf://test/x", true);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    #[test]
    fn fcli002_pass_offline_https_rejected() {
        let v = verdict_from_offline_blocks_network(true, "https://example.com/x", true);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    #[test]
    fn fcli002_pass_offline_local_path() {
        let v = verdict_from_offline_blocks_network(true, "/tmp/model.gguf", true);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    #[test]
    fn fcli002_pass_online_no_constraint() {
        let v = verdict_from_offline_blocks_network(false, "hf://test/x", false);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    #[test]
    fn fcli002_fail_offline_hf_not_rejected() {
        // Privacy violation — offline mode contacted HF
        let v = verdict_from_offline_blocks_network(true, "hf://test/x", false);
        assert_eq!(v, CliSafetyVerdict::Fail);
    }

    #[test]
    fn fcli002_fail_empty_source() {
        let v = verdict_from_offline_blocks_network(true, "", true);
        assert_eq!(v, CliSafetyVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: CLI-003 encrypt rejects .enc.
    // -----------------------------------------------------------------
    #[test]
    fn fcli003_pass_enc_rejected() {
        let v = verdict_from_encrypt_rejects_enc("model.enc", true);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    #[test]
    fn fcli003_pass_plain_input_unchecked() {
        let v = verdict_from_encrypt_rejects_enc("model.apr", false);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    #[test]
    fn fcli003_fail_enc_not_rejected() {
        // Double-encryption regression class
        let v = verdict_from_encrypt_rejects_enc("model.enc", false);
        assert_eq!(v, CliSafetyVerdict::Fail);
    }

    #[test]
    fn fcli003_fail_empty_path() {
        let v = verdict_from_encrypt_rejects_enc("", true);
        assert_eq!(v, CliSafetyVerdict::Fail);
    }

    #[test]
    fn fcli003_pass_path_with_dot_enc_in_dir() {
        // "/foo.enc/model.apr" doesn't end with .enc — vacuous pass
        let v = verdict_from_encrypt_rejects_enc("/foo.enc/model.apr", false);
        assert_eq!(v, CliSafetyVerdict::Pass);
    }

    // -----------------------------------------------------------------
    // Section 5: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_001_score_band() {
        for score in [0_u32, 49, 50, 51, 99, 100] {
            for exit in [0_i32, 5, 1] {
                let v = verdict_from_validate_exit(score, exit);
                let expected = if score < 50 {
                    if exit == 5 {
                        CliSafetyVerdict::Pass
                    } else {
                        CliSafetyVerdict::Fail
                    }
                } else if exit == 0 {
                    CliSafetyVerdict::Pass
                } else {
                    CliSafetyVerdict::Fail
                };
                assert_eq!(v, expected, "score={score} exit={exit}");
            }
        }
    }

    #[test]
    fn mutation_survey_002_network_prefix_band() {
        for src in [
            "hf://x",
            "http://x",
            "https://x",
            "file:///tmp/x",
            "/local/path",
            "s3://bucket/x", // not in our network-prefix list
        ] {
            // offline=true, rejected=true
            let v = verdict_from_offline_blocks_network(true, src, true);
            assert_eq!(v, CliSafetyVerdict::Pass, "src={src}");
        }
    }

    // -----------------------------------------------------------------
    // Section 6: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_3() {
        let v1 = verdict_from_validate_exit(85, 0);
        let v2 = verdict_from_offline_blocks_network(true, "hf://model/x", true);
        let v3 = verdict_from_encrypt_rejects_enc("model.enc", true);
        assert_eq!(v1, CliSafetyVerdict::Pass);
        assert_eq!(v2, CliSafetyVerdict::Pass);
        assert_eq!(v3, CliSafetyVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_3_failures() {
        // Three regression classes:
        //  1: silent-failure — apr validate said "invalid" but exit 0
        //  2: privacy violation — offline mode hit HF anyway
        //  3: double-encryption — apr encrypt model.enc didn't reject
        let v1 = verdict_from_validate_exit(3, 0);
        let v2 = verdict_from_offline_blocks_network(true, "hf://x", false);
        let v3 = verdict_from_encrypt_rejects_enc("model.enc", false);
        assert_eq!(v1, CliSafetyVerdict::Fail);
        assert_eq!(v2, CliSafetyVerdict::Fail);
        assert_eq!(v3, CliSafetyVerdict::Fail);
    }
}
