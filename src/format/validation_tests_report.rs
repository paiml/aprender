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
        assert_eq!(report.grade(), "F");
        assert_eq!(report.total_score, 50);
    }

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
        assert!(!report.passed(95));
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
}
