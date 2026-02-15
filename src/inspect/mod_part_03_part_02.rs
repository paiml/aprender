
    #[test]
    fn test_header_flags_roundtrip() {
        let flags = HeaderFlags {
            compressed: true,
            signed: false,
            encrypted: true,
            streaming: false,
            licensed: true,
            quantized: false,
        };

        let byte = flags.to_byte();
        let restored = HeaderFlags::from_byte(byte);

        assert_eq!(flags.compressed, restored.compressed);
        assert_eq!(flags.signed, restored.signed);
        assert_eq!(flags.encrypted, restored.encrypted);
        assert_eq!(flags.streaming, restored.streaming);
        assert_eq!(flags.licensed, restored.licensed);
        assert_eq!(flags.quantized, restored.quantized);
    }

    #[test]
    fn test_header_flags_list() {
        let flags = HeaderFlags {
            compressed: true,
            signed: true,
            encrypted: false,
            streaming: false,
            licensed: false,
            quantized: false,
        };

        let list = flags.flag_list();
        assert!(list.contains(&"COMPRESSED"));
        assert!(list.contains(&"SIGNED"));
        assert!(!list.contains(&"ENCRYPTED"));
    }

    #[test]
    fn test_header_inspection() {
        let header = HeaderInspection::new();
        assert!(header.is_valid());
        assert_eq!(header.magic_string(), "APRN");
        assert_eq!(header.version_string(), "1.0");
        assert!((header.compression_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_stats_from_slice() {
        let weights = vec![1.0_f32, 2.0, 3.0, 0.0, 5.0];
        let stats = WeightStats::from_slice(&weights);

        assert_eq!(stats.count, 5);
        assert!((stats.min - 0.0).abs() < 0.001);
        assert!((stats.max - 5.0).abs() < 0.001);
        assert!((stats.mean - 2.2).abs() < 0.001);
        assert_eq!(stats.zero_count, 1);
        assert_eq!(stats.nan_count, 0);
        assert!((stats.sparsity - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_weight_stats_empty() {
        let stats = WeightStats::from_slice(&[]);
        assert_eq!(stats.count, 0);
        assert!(!stats.has_issues());
    }

    #[test]
    fn test_weight_stats_with_nan() {
        let weights = vec![1.0_f32, f32::NAN, 3.0];
        let stats = WeightStats::from_slice(&weights);

        assert_eq!(stats.nan_count, 1);
        assert!(stats.has_issues());
        assert_eq!(stats.health_status(), WeightHealth::Critical);
    }

    #[test]
    fn test_weight_health() {
        assert_eq!(
            WeightHealth::Healthy.description(),
            "Weights are within normal parameters"
        );
    }

    #[test]
    fn test_inspection_warning() {
        let warning =
            InspectionWarning::new("W001", "Test warning").with_recommendation("Fix the issue");

        let display = format!("{}", warning);
        assert!(display.contains("W001"));
        assert!(display.contains("Test warning"));
        assert!(display.contains("Fix the issue"));
    }

    #[test]
    fn test_inspection_error() {
        let error = InspectionError::new("E001", "Test error", true);
        let display = format!("{}", error);
        assert!(display.contains("FATAL"));
        assert!(display.contains("E001"));
    }

    #[test]
    fn test_diff_item() {
        let item = DiffItem::new("version", "1.0", "2.0");
        let display = format!("{}", item);
        assert!(display.contains("version"));
        assert!(display.contains("1.0"));
        assert!(display.contains("2.0"));
    }

    #[test]
    fn test_weight_diff_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let diff = WeightDiff::from_slices(&a, &b);

        assert!(diff.is_identical());
        assert!((diff.cosine_similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_diff_different() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0, 4.0];
        let diff = WeightDiff::from_slices(&a, &b);

        assert!(!diff.is_identical());
        assert_eq!(diff.changed_count, 1);
        assert!((diff.max_diff - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_diff_result() {
        let mut diff = DiffResult::new("model_a.apr", "model_b.apr");
        assert!(diff.is_identical());

        diff.header_diff
            .push(DiffItem::new("version", "1.0", "2.0"));
        assert!(!diff.is_identical());
        assert_eq!(diff.diff_count(), 1);
    }

    #[test]
    fn test_inspect_options_default() {
        let opts = InspectOptions::default();
        assert!(opts.include_weights);
        assert!(opts.include_quality);
    }

    #[test]
    fn test_inspect_options_quick() {
        let opts = InspectOptions::quick();
        assert!(!opts.include_weights);
        assert!(!opts.include_quality);
    }

    #[test]
    fn test_inspection_result() {
        let header = HeaderInspection::new();
        let metadata = MetadataInspection::new("LinearRegression");
        let result = InspectionResult::new(header, metadata);

        assert!(!result.has_issues());
        assert!(result.is_valid());
    }

    #[test]
    fn test_training_info() {
        let info = TrainingInfo::new();
        assert!(info.trained_at.is_none());
        assert!(info.dataset_name.is_none());
    }

    #[test]
    fn test_license_info() {
        let info = LicenseInfo::new("MIT");
        assert_eq!(info.license_type, "MIT");
        assert!(!info.has_restrictions());
    }

    #[test]
    fn test_metadata_inspection() {
        let meta = MetadataInspection::new("RandomForest");
        assert_eq!(meta.model_type_name, "RandomForest");
        assert!(!meta.has_training_info());
        assert!(!meta.is_licensed());
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_inspection_result_with_warnings() {
        let header = HeaderInspection::new();
        let metadata = MetadataInspection::new("Test");
        let mut result = InspectionResult::new(header, metadata);

        result
            .warnings
            .push(InspectionWarning::new("W001", "Test warning"));
        assert!(result.has_issues());
        assert!(result.is_valid()); // Still valid, just has warnings
        assert_eq!(result.issue_count(), 1);
    }

    #[test]
    fn test_inspection_result_with_errors() {
        let header = HeaderInspection::new();
        let metadata = MetadataInspection::new("Test");
        let mut result = InspectionResult::new(header, metadata);

        result
            .errors
            .push(InspectionError::new("E001", "Test error", true));
        assert!(result.has_issues());
        assert!(!result.is_valid()); // Invalid due to errors
        assert_eq!(result.issue_count(), 1);
    }

    #[test]
    fn test_inspection_result_with_both() {
        let header = HeaderInspection::new();
        let metadata = MetadataInspection::new("Test");
        let mut result = InspectionResult::new(header, metadata);

        result
            .warnings
            .push(InspectionWarning::new("W001", "Warning"));
        result
            .errors
            .push(InspectionError::new("E001", "Error", false));
        assert_eq!(result.issue_count(), 2);
    }

    #[test]
    fn test_header_inspection_compression_ratio_nonzero() {
        let mut header = HeaderInspection::new();
        header.compressed_size = 500;
        header.uncompressed_size = 1000;
        assert!((header.compression_ratio() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_header_inspection_invalid_magic() {
        let mut header = HeaderInspection::new();
        header.magic_valid = false;
        assert!(!header.is_valid());
    }

    #[test]
    fn test_header_inspection_unsupported_version() {
        let mut header = HeaderInspection::new();
        header.version_supported = false;
        assert!(!header.is_valid());
    }

    #[test]
    fn test_header_flags_all_set() {
        let flags = HeaderFlags {
            compressed: true,
            signed: true,
            encrypted: true,
            streaming: true,
            licensed: true,
            quantized: true,
        };
        let list = flags.flag_list();
        assert_eq!(list.len(), 6);
        assert!(list.contains(&"COMPRESSED"));
        assert!(list.contains(&"SIGNED"));
        assert!(list.contains(&"ENCRYPTED"));
        assert!(list.contains(&"STREAMING"));
        assert!(list.contains(&"LICENSED"));
        assert!(list.contains(&"QUANTIZED"));
    }

    #[test]
    fn test_header_flags_byte_roundtrip_all() {
        let byte = 0x3F; // All 6 flags set
        let flags = HeaderFlags::from_byte(byte);
        assert!(flags.compressed);
        assert!(flags.signed);
        assert!(flags.encrypted);
        assert!(flags.streaming);
        assert!(flags.licensed);
        assert!(flags.quantized);
        assert_eq!(flags.to_byte(), byte);
    }

    #[test]
    fn test_header_flags_empty() {
        let flags = HeaderFlags::default();
        assert!(flags.flag_list().is_empty());
        assert_eq!(flags.to_byte(), 0);
    }

    #[test]
    fn test_metadata_with_training_info() {
        let mut meta = MetadataInspection::new("Model");
        meta.training_info = Some(TrainingInfo::new());
        assert!(meta.has_training_info());
    }

    #[test]
    fn test_metadata_with_license_info() {
        let mut meta = MetadataInspection::new("Model");
        meta.license_info = Some(LicenseInfo::new("MIT"));
        assert!(meta.is_licensed());
    }

    #[test]
    fn test_training_info_default() {
        let info = TrainingInfo::default();
        assert!(info.trained_at.is_none());
        assert!(info.duration.is_none());
        assert!(info.dataset_name.is_none());
        assert!(info.n_samples.is_none());
        assert!(info.final_loss.is_none());
        assert!(info.framework.is_none());
        assert!(info.framework_version.is_none());
    }

    #[test]
    fn test_license_info_with_restrictions() {
        let mut info = LicenseInfo::new("Proprietary");
        info.restrictions.push("No commercial use".to_string());
        info.restrictions.push("No redistribution".to_string());
        assert!(info.has_restrictions());
        assert_eq!(info.restrictions.len(), 2);
    }

    #[test]
    fn test_weight_stats_with_inf() {
        let weights = vec![1.0_f32, f32::INFINITY, 3.0];
        let stats = WeightStats::from_slice(&weights);
        assert_eq!(stats.inf_count, 1);
        assert!(stats.has_issues());
        assert_eq!(stats.health_status(), WeightHealth::Critical);
    }

    #[test]
    fn test_weight_stats_high_sparsity() {
        let weights = vec![0.0_f32; 100];
        let stats = WeightStats::from_slice(&weights);
        assert!((stats.sparsity - 1.0).abs() < 0.001);
        assert_eq!(stats.health_status(), WeightHealth::Warning);
    }

    #[test]
    fn test_weight_stats_low_variance() {
        let weights = vec![1.0_f32; 100];
        let stats = WeightStats::from_slice(&weights);
        assert!(stats.std < 1e-10);
        assert_eq!(stats.health_status(), WeightHealth::Warning);
    }

    #[test]
    fn test_weight_stats_single_element() {
        let weights = vec![5.0_f32];
        let stats = WeightStats::from_slice(&weights);
        assert_eq!(stats.count, 1);
        assert!((stats.mean - 5.0).abs() < 0.001);
        assert!((stats.std - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_health_descriptions() {
        assert!(!WeightHealth::Healthy.description().is_empty());
        assert!(!WeightHealth::Warning.description().is_empty());
        assert!(!WeightHealth::Critical.description().is_empty());
        assert!(WeightHealth::Warning.description().contains("potential"));
        assert!(WeightHealth::Critical.description().contains("critical"));
    }

    #[test]
    fn test_inspection_warning_without_recommendation() {
        let warning = InspectionWarning::new("W001", "Test warning");
        assert!(warning.recommendation.is_none());
        let display = format!("{}", warning);
        assert!(display.contains("W001"));
        assert!(!display.contains("Recommendation"));
    }

    #[test]
    fn test_inspection_error_nonfatal() {
        let error = InspectionError::new("E001", "Non-fatal error", false);
        assert!(!error.fatal);
        let display = format!("{}", error);
        assert!(display.contains("ERROR"));
        assert!(!display.contains("FATAL"));
    }

    #[test]
    fn test_diff_result_with_weight_diff() {
        let mut diff = DiffResult::new("a.apr", "b.apr");
        diff.weight_diff = Some(WeightDiff::empty());
        assert!(diff.is_identical()); // Empty diff means identical

        diff.weight_diff = Some(WeightDiff {
            changed_count: 5,
            max_diff: 0.1,
            mean_diff: 0.05,
            l2_distance: 0.2,
            cosine_similarity: 0.99,
        });
        assert!(!diff.is_identical());
        assert_eq!(diff.diff_count(), 1); // One weight diff entry
    }

    #[test]
    fn test_weight_diff_empty_or_mismatched() {
        let empty1: Vec<f32> = vec![];
        let empty2: Vec<f32> = vec![];
        let diff = WeightDiff::from_slices(&empty1, &empty2);
        assert!(diff.is_identical());

        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0]; // Different lengths
        let diff = WeightDiff::from_slices(&a, &b);
        assert!(diff.is_identical()); // Empty diff due to length mismatch
    }

    #[test]
    fn test_weight_diff_zero_norms() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let diff = WeightDiff::from_slices(&a, &b);
        assert!((diff.cosine_similarity - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_inspect_options_new() {
        let opts = InspectOptions::new();
        assert!(opts.include_weights);
        assert!(opts.include_quality);
    }
