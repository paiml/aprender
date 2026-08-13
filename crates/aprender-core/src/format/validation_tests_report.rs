use super::*;

// ============================================================================
// ValidationReport Tests
// ============================================================================

#[cfg(test)]
mod tests_report {
    use super::*;

    #[test]
    fn test_report_grade_a_plus() {
        let mut report = ValidationReport::new();
        for i in 1..=95 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        assert_eq!(report.grade(), "A+");
        assert_eq!(report.total_score, 95);
    }

    /// #1866: `F` is earned by FAILING checks, never by a short checklist.
    ///
    /// This test used to assert that 50 checks — every one of them PASSING —
    /// graded `F`, because the grade banded 50 awarded points against a fixed
    /// 100-point denominator. That assertion was the defect written down as a
    /// test: it is the same arithmetic that graded a healthy `.apr` file `F`
    /// at 3/100.
    #[test]
    fn test_report_grade_f() {
        let mut report = ValidationReport::new();
        for i in 1..=50 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        assert_eq!(
            report.grade(),
            "A+",
            "50 of 50 checks passed; a short checklist is not a failing model"
        );
        assert_eq!(report.total_score, 50);

        // One genuine failure, and only then, is F.
        report.add_check(ValidationCheck {
            id: 51,
            name: "broken",
            category: Category::Structure,
            status: CheckStatus::Fail("magic bytes".to_string()),
            points: 0,
        });
        assert_eq!(report.grade(), "F");
    }

    /// #1866: the threshold is a percentage of the checks that RAN.
    ///
    /// It used to compare raw awarded points, so `passed(95)` was unreachable
    /// for any real `.apr` file (ceiling 4 points) and `apr import` reported
    /// "completed with warnings" on every successful import.
    #[test]
    fn test_report_passed_threshold() {
        let mut report = ValidationReport::new();
        for i in 1..=90 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        assert!(report.passed(90));
        assert!(report.passed(95), "90/90 checks passed — that is 100%");

        // A tenth of the suite failing puts it under the bar.
        for i in 91..=100 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Fail("bad".to_string()),
                points: 0,
            });
        }
        assert!(!report.passed(95), "90/100 = 90% must not clear 95");
    }

    /// A threshold cannot be cleared against a score that was never measured.
    #[test]
    fn passed_fails_closed_when_no_check_ran() {
        let mut report = ValidationReport::new();
        for i in 1..=25 {
            push_check(&mut report, i, CheckStatus::Skip("Not implemented".into()));
        }
        assert!(
            !report.passed(0),
            "nothing ran, so no threshold — not even 0 — was demonstrated"
        );
        assert_eq!(report.grade(), "N/A");
    }

    #[test]
    fn test_report_failed_checks() {
        let mut report = ValidationReport::new();
        report.add_check(ValidationCheck {
            id: 1,
            name: "pass",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
        report.add_check(ValidationCheck {
            id: 2,
            name: "fail",
            category: Category::Structure,
            status: CheckStatus::Fail("reason".to_string()),
            points: 0,
        });

        let failed = report.failed_checks();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].id, 2);
    }

    #[test]
    fn test_category_scores() {
        let mut report = ValidationReport::new();
        report.add_check(ValidationCheck {
            id: 1,
            name: "struct1",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
        report.add_check(ValidationCheck {
            id: 26,
            name: "physics1",
            category: Category::Physics,
            status: CheckStatus::Pass,
            points: 1,
        });

        assert_eq!(report.category_scores.get(&Category::Structure), Some(&1));
        assert_eq!(report.category_scores.get(&Category::Physics), Some(&1));
    }

    // ====================================================================
    // Coverage: CheckStatus methods
    // ====================================================================

    #[test]
    fn test_check_status_is_pass() {
        assert!(CheckStatus::Pass.is_pass());
        assert!(!CheckStatus::Pass.is_fail());
    }

    #[test]
    fn test_check_status_is_fail() {
        let fail = CheckStatus::Fail("bad".to_string());
        assert!(fail.is_fail());
        assert!(!fail.is_pass());
    }

    #[test]
    fn test_check_status_skip_not_pass_not_fail() {
        let skip = CheckStatus::Skip("n/a".to_string());
        assert!(!skip.is_pass());
        assert!(!skip.is_fail());
    }

    // ====================================================================
    // Coverage: Category methods
    // ====================================================================

    #[test]
    fn test_category_letter() {
        assert_eq!(Category::Structure.letter(), 'A');
        assert_eq!(Category::Physics.letter(), 'B');
        assert_eq!(Category::Tooling.letter(), 'C');
        assert_eq!(Category::Conversion.letter(), 'D');
    }

    #[test]
    fn test_category_name() {
        assert_eq!(Category::Structure.name(), "Format & Structural Integrity");
        assert_eq!(Category::Physics.name(), "Tensor Physics & Statistics");
        assert_eq!(Category::Tooling.name(), "Tooling & Operations");
        assert_eq!(Category::Conversion.name(), "Conversion & Interoperability");
    }

    // ====================================================================
    // Coverage: AprHeader flag methods
    // ====================================================================

    #[test]
    fn test_apr_header_is_compressed() {
        let header = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 2,
            version_minor: 0,
            flags: 0x01, // compressed bit
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(header.is_compressed());
        assert!(!header.is_signed());
        assert!(!header.is_encrypted());
    }

    #[test]
    fn test_apr_header_is_signed() {
        let header = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 2,
            version_minor: 0,
            flags: 0x20, // signed bit
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(!header.is_compressed());
        assert!(header.is_signed());
        assert!(!header.is_encrypted());
    }

    #[test]
    fn test_apr_header_is_encrypted() {
        let header = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 2,
            version_minor: 0,
            flags: 0x10, // encrypted bit
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(!header.is_compressed());
        assert!(!header.is_signed());
        assert!(header.is_encrypted());
    }

    #[test]
    fn test_apr_header_supported_versions() {
        // v1.0, v1.1, v1.2 supported
        for minor in 0..=2 {
            let h = AprHeader {
                magic: [0x41, 0x50, 0x52, 0x00],
                version_major: 1,
                version_minor: minor,
                flags: 0,
                metadata_offset: 0,
                metadata_size: 0,
                index_offset: 0,
                index_size: 0,
                data_offset: 0,
            };
            assert!(h.is_supported_version(), "v1.{minor} should be supported");
        }
        // v2.0 supported
        let h = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 2,
            version_minor: 0,
            flags: 0,
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(h.is_supported_version());
        // v3.0 not supported
        let h = AprHeader {
            magic: [0x41, 0x50, 0x52, 0x00],
            version_major: 3,
            version_minor: 0,
            flags: 0,
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(!h.is_supported_version());
    }

    // ====================================================================
    // Coverage: ValidationReport::failed_checks
    // ====================================================================

    #[test]
    fn test_report_failed_checks_mixed() {
        let mut report = ValidationReport::new();
        report.add_check(ValidationCheck {
            id: 1,
            name: "pass_check",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
        report.add_check(ValidationCheck {
            id: 2,
            name: "fail_check",
            category: Category::Structure,
            status: CheckStatus::Fail("bad".to_string()),
            points: 0,
        });
        report.add_check(ValidationCheck {
            id: 3,
            name: "skip_check",
            category: Category::Physics,
            status: CheckStatus::Skip("n/a".to_string()),
            points: 0,
        });
        let failed = report.failed_checks();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].id, 2);
    }

    // ====================================================================
    // Coverage: CheckStatus::Warn variant
    // ====================================================================

    #[test]
    fn test_check_status_warn() {
        let warn = CheckStatus::Warn("warning message".to_string());
        assert!(!warn.is_pass());
        assert!(!warn.is_fail());
    }

    // ====================================================================
    // Coverage: AprHeader::is_valid_magic
    // ====================================================================

    #[test]
    fn test_apr_header_is_valid_magic_true() {
        let header = AprHeader {
            magic: *b"APR\0",
            version_major: 1,
            version_minor: 0,
            flags: 0,
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(header.is_valid_magic());
    }

    #[test]
    fn test_apr_header_is_valid_magic_false() {
        let header = AprHeader {
            magic: *b"GGUF",
            version_major: 1,
            version_minor: 0,
            flags: 0,
            metadata_offset: 0,
            metadata_size: 0,
            index_offset: 0,
            index_size: 0,
            data_offset: 0,
        };
        assert!(!header.is_valid_magic());
    }

    // ====================================================================
    // Coverage: AprHeader::parse error path
    // ====================================================================

    #[test]
    fn test_apr_header_parse_too_small() {
        let data = vec![0u8; 16]; // Too small
        let result = AprHeader::parse(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = format!("{:?}", err);
        assert!(err_msg.contains("Header too small"));
    }

    // ====================================================================
    // Coverage: ValidationCheck with Warn status
    // ====================================================================

    #[test]
    fn test_validation_check_with_warn_status() {
        let check = ValidationCheck {
            id: 11,
            name: "unknown_flags",
            category: Category::Structure,
            status: CheckStatus::Warn("Unknown flag bits".to_string()),
            points: 0,
        };
        assert!(!check.status.is_pass());
        assert!(!check.status.is_fail());
    }

    // ====================================================================
    // Coverage: AprValidator::validate (tensor validation path)
    // ====================================================================

    #[test]
    fn test_apr_validator_validate_tensors() {
        let mut validator = AprValidator::new();
        // Add some tensor stats
        validator.add_tensor_stats(TensorStats::compute("test.weight", &vec![1.0f32; 100]));
        let report = validator.validate();
        // Should have run tensor validation
        assert!(!report.checks.is_empty());
    }

    // ====================================================================
    // Coverage: AprValidator Default trait
    // ====================================================================

    #[test]
    fn test_apr_validator_default() {
        let validator = AprValidator::default();
        assert!(validator.report().checks.is_empty());
    }

    // ====================================================================
    // Coverage: ValidationReport Default trait
    // ====================================================================

    #[test]
    fn test_validation_report_default() {
        let report = ValidationReport::default();
        assert!(report.checks.is_empty());
        assert_eq!(report.total_score, 0);
    }

    // ====================================================================
    // Coverage: File too small for magic bytes
    // ====================================================================

    #[test]
    fn test_check_magic_file_too_small() {
        let data = vec![0u8; 2]; // Only 2 bytes
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(check.status.is_fail());
    }

    // ====================================================================
    // Coverage: GGUF file too small for version check
    // ====================================================================

    #[test]
    fn test_gguf_version_file_too_small() {
        // GGUF magic but not enough bytes for version
        let mut data = vec![0u8; 6]; // Less than 8 bytes
        data[0..4].copy_from_slice(b"GGUF");
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_fail());
    }

    // ====================================================================
    // Coverage: Unknown flags warning path (check 11)
    // ====================================================================

    #[test]
    fn test_check_11_unknown_flags_warn() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR\0");
        data[4] = 1; // version major
                     // Set unknown flag bits (beyond bit 7)
        data[9] = 0x01; // This sets bit 8 which is unknown
        let mut validator = AprValidator::new();
        validator.validate_bytes(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 11)
            .unwrap();
        // Should be a warning for unknown flags
        assert!(matches!(check.status, CheckStatus::Warn(_)));
    }

    // ====================================================================
    // Coverage: TensorStats with only NaN/Inf values
    // ====================================================================

    #[test]
    fn test_tensor_stats_all_nan() {
        let data = vec![f32::NAN, f32::NAN, f32::NAN];
        let stats = TensorStats::compute("nan_tensor", &data);
        assert_eq!(stats.nan_count, 3);
        assert_eq!(stats.mean, 0.0); // No valid values, mean defaults to 0
        assert_eq!(stats.std, 0.0); // No valid values, std defaults to 0
    }

    #[test]
    fn test_tensor_stats_all_inf() {
        let data = vec![f32::INFINITY, f32::NEG_INFINITY];
        let stats = TensorStats::compute("inf_tensor", &data);
        assert_eq!(stats.inf_count, 2);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.min, 0.0); // min/max default when all inf
        assert_eq!(stats.max, 0.0);
    }

    #[test]
    fn test_tensor_stats_single_value() {
        let data = vec![42.0f32];
        let stats = TensorStats::compute("single", &data);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.mean, 42.0);
        assert_eq!(stats.std, 0.0); // std with single value is 0
        assert_eq!(stats.min, 42.0);
        assert_eq!(stats.max, 42.0);
    }

    // ========================================================================
    // #1866: implemented_score_pct + implemented_max
    //
    // Contract: apr-validate-quality-threshold-v1
    // ========================================================================

    fn push_check(report: &mut ValidationReport, id: u8, status: CheckStatus) {
        let points = if matches!(status, CheckStatus::Pass) { 1 } else { 0 };
        report.add_check(ValidationCheck {
            id,
            name: "test",
            category: Category::Structure,
            status,
            points,
        });
    }

    /// FALSIFY-VALIDATE-QUALITY-001: fully-stubbed suite returns None.
    #[test]
    fn test_implemented_score_pct_none_when_all_stubbed() {
        let mut report = ValidationReport::new();
        for i in 1..=25 {
            push_check(&mut report, i, CheckStatus::Skip("Not implemented".into()));
        }
        assert_eq!(report.implemented_max(), 0);
        assert_eq!(report.implemented_score_pct(), None);
    }

    /// FALSIFY-VALIDATE-QUALITY-002: 3 Pass + 22 Skip returns 100% (matches #1866 reproducer).
    #[test]
    fn test_implemented_score_pct_100_when_all_pass() {
        let mut report = ValidationReport::new();
        for i in 1..=3 {
            push_check(&mut report, i, CheckStatus::Pass);
        }
        for i in 4..=25 {
            push_check(&mut report, i, CheckStatus::Skip("Not implemented".into()));
        }
        assert_eq!(report.implemented_max(), 3);
        assert_eq!(report.implemented_score_pct(), Some(100.0));
        // Per #1866: this is the 1.5B Q4K APR case — must not fail the gate.
    }

    /// Half-implemented half-failing: implemented_pct = 50% — gate fires at < 50.
    #[test]
    fn test_implemented_score_pct_mixed() {
        let mut report = ValidationReport::new();
        push_check(&mut report, 1, CheckStatus::Pass);
        push_check(&mut report, 2, CheckStatus::Pass);
        push_check(&mut report, 3, CheckStatus::Fail("bad".into()));
        push_check(&mut report, 4, CheckStatus::Fail("bad".into()));
        for i in 5..=25 {
            push_check(&mut report, i, CheckStatus::Skip("Not implemented".into()));
        }
        assert_eq!(report.implemented_max(), 4);
        let pct = report.implemented_score_pct().expect("some");
        assert!((pct - 50.0).abs() < f64::EPSILON, "expected 50.0, got {pct}");
    }

    /// Below-threshold case: 1/4 pass = 25% — gate must fire.
    #[test]
    fn test_implemented_score_pct_below_threshold() {
        let mut report = ValidationReport::new();
        push_check(&mut report, 1, CheckStatus::Pass);
        push_check(&mut report, 2, CheckStatus::Fail("bad".into()));
        push_check(&mut report, 3, CheckStatus::Fail("bad".into()));
        push_check(&mut report, 4, CheckStatus::Fail("bad".into()));
        let pct = report.implemented_score_pct().expect("some");
        assert!(pct < 50.0, "expected < 50, got {pct}");
    }

    /// FALSIFIER (#2394 finding 12): a score must carry the denominator it was
    /// measured against.
    ///
    /// `apr validate` printed `✓ VALID 3/100 points` on a healthy model. 97 of
    /// those 100 checks are `Skip("Not implemented")` stubs that never ran, so
    /// "100" was never the denominator being measured against — the line
    /// understates a healthy model and would equally overstate a sick one.
    /// `ImplementedScore` cannot be displayed without both numbers.
    #[test]
    fn implemented_score_reports_the_checks_that_actually_ran() {
        let mut report = ValidationReport::new();
        for id in 1..=3u8 {
            report.add_check(ValidationCheck {
                id,
                name: "implemented",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        for id in 4..=100u8 {
            report.add_check(ValidationCheck {
                id,
                name: "stub",
                category: Category::Physics,
                status: CheckStatus::Skip("Not implemented".to_string()),
                points: 0,
            });
        }

        let score = report.implemented_score();
        assert_eq!(score.passed, 3);
        assert_eq!(score.ran, 3, "97 stubs did not run and must not count");
        assert_eq!(score.not_implemented(), 97);

        let rendered = score.to_string();
        assert!(
            !rendered.contains("3/100"),
            "the score must not be printed against a denominator that never ran: {rendered}"
        );
        assert!(rendered.contains("3/3"), "{rendered}");
        assert!(rendered.contains("97"), "{rendered}");
    }

    /// A failing check RAN, so it belongs in the denominator — otherwise a
    /// model could score 1/1 while a check was actively failing.
    #[test]
    fn implemented_score_counts_failures_in_the_denominator() {
        let mut report = ValidationReport::new();
        report.add_check(ValidationCheck {
            id: 1,
            name: "passing",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
        report.add_check(ValidationCheck {
            id: 2,
            name: "failing",
            category: Category::Structure,
            status: CheckStatus::Fail("bad".to_string()),
            points: 0,
        });

        let score = report.implemented_score();
        assert_eq!(score.passed, 1);
        assert_eq!(score.ran, 2, "a FAILED check ran and must count against us");
        assert_eq!(score.not_implemented(), 0);
        assert_eq!(score.to_string(), "1/2 checks that ran");
    }

    // ========================================================================
    // #1866 FALSIFIER: a healthy model must not be graded F, and the grade,
    // the pass flag and the human verdict must not contradict each other.
    //
    // Contract: apr-validate-quality-threshold-v1
    // ========================================================================

    /// The exact report `apr validate` builds for
    /// `/home/noah/models/qwen2.5-coder-0.5b-instruct.apr` — a model
    /// `apr qa` passes and `apr run` answers correctly from. Checks 1/2/3
    /// PASS, check 11 WARNs on an unknown flag bit, and 22 checks are
    /// `Skip("Not implemented")` stubs.
    fn healthy_apr_report() -> ValidationReport {
        let mut report = ValidationReport::new();
        push_check(&mut report, 1, CheckStatus::Pass);
        push_check(&mut report, 2, CheckStatus::Pass);
        push_check(&mut report, 3, CheckStatus::Pass);
        push_check(
            &mut report,
            11,
            CheckStatus::Warn("Unknown flag bits: 0x00000100".into()),
        );
        push_check(
            &mut report,
            4,
            CheckStatus::Skip("Footer not implemented".into()),
        );
        for id in 5..=25 {
            push_check(&mut report, id, CheckStatus::Skip("Not implemented".into()));
        }
        report
    }

    /// FALSIFY-VALIDATE-QUALITY-004 (#1866): a healthy model must score above
    /// the F band.
    ///
    /// Before the fix this asserted `"F"` on a working model, at exit 0, while
    /// the same report advertised `passed: true` and printed `✓ VALID`.
    #[test]
    fn healthy_model_is_not_graded_f() {
        let report = healthy_apr_report();

        assert_eq!(report.implemented_score().ran, 4);
        assert_eq!(report.implemented_score().not_implemented(), 22);
        assert_ne!(
            report.grade(),
            "F",
            "a model with zero failed checks was graded F: {:?}",
            report.implemented_score()
        );
        assert_eq!(report.grade(), "C+", "3 of the 4 checks that ran passed");
    }

    /// FALSIFY-VALIDATE-QUALITY-005 (#1866): grade, pass flag and human
    /// verdict are one decision, not three.
    ///
    /// `grade() == "F"` must hold EXACTLY when a check that ran reported a
    /// failure — which is also `!is_valid()`, the predicate behind the
    /// `VALID` / `INVALID` badge. Exhaustive over every arrangement of
    /// Pass/Fail/Warn/Skip across four checks (256 reports).
    #[test]
    fn grade_is_f_exactly_when_a_check_failed() {
        let statuses = [
            CheckStatus::Pass,
            CheckStatus::Fail("bad".into()),
            CheckStatus::Warn("advisory".into()),
            CheckStatus::Skip("Not implemented".into()),
        ];
        // Every arrangement of four checks, enumerated as the base-4 digits
        // of 0..256 — one loop instead of four nested ones.
        for shape in 0..statuses.len().pow(4) {
            let picks = [shape % 4, (shape / 4) % 4, (shape / 16) % 4, (shape / 64) % 4];
            let mut report = ValidationReport::new();
            for (i, idx) in picks.into_iter().enumerate() {
                push_check(&mut report, i as u8 + 1, statuses[idx].clone());
            }
            let grade = report.grade();
            let any_failed = !report.is_valid();

            if report.implemented_score().ran == 0 {
                assert_eq!(grade, "N/A", "nothing ran but was graded {grade}");
                continue;
            }
            assert_eq!(
                grade == "F",
                any_failed,
                "grade {grade} disagrees with the VALID/INVALID verdict for {picks:?}"
            );
            // The JSON `passed` flag is `is_valid()` too, so a passing report
            // can never carry an F.
            assert!(
                any_failed || grade != "F",
                "passing report graded F for {picks:?}"
            );
        }
    }

    /// A category in which nothing is declared must report `ran == 0`, so the
    /// renderer can say "not implemented" instead of drawing `0/25` — a zero
    /// for something that was never measured (#1866).
    #[test]
    fn category_score_distinguishes_never_ran_from_scored_zero() {
        let report = healthy_apr_report(); // Structure only.

        let structure = report.category_score(Category::Structure);
        assert_eq!(structure.ran, 4);
        assert_eq!(structure.passed, 3);

        for empty in [Category::Physics, Category::Tooling, Category::Conversion] {
            let score = report.category_score(empty);
            assert_eq!(score.declared, 0, "{empty:?} declares no checks here");
            assert_eq!(
                score.pct(),
                None,
                "{empty:?} ran nothing, so it has no score — not a score of zero"
            );
        }
    }
}
