
    #[test]
    fn tensor_stats_compute_mixed_nan_inf_valid() {
        let stats = TensorStats::compute(&[1.0, f32::NAN, f32::INFINITY, 2.0, f32::NEG_INFINITY]);
        assert_eq!(stats.len, 5);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 2);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 2.0);
    }

    #[test]
    fn tensor_stats_zero_pct_empty() {
        let stats = TensorStats::compute(&[]);
        assert_eq!(stats.zero_pct(), 0.0);
    }

    #[test]
    fn tensor_stats_zero_pct_with_zeros() {
        let stats = TensorStats::compute(&[0.0, 0.0, 1.0, 2.0]);
        // 2 out of 4 are near-zero = 50%
        assert!((stats.zero_pct() - 50.0).abs() < 0.01);
    }

    // ================================================================
    // ContractValidationError Display and Error trait
    // ================================================================

    #[test]
    fn contract_validation_error_display_format() {
        let err = ContractValidationError {
            tensor_name: "embedding".to_string(),
            rule_id: "F-DATA-QUALITY-001".to_string(),
            message: "DENSITY FAILURE: 94.5% zeros".to_string(),
        };
        let display = format!("{err}");
        assert_eq!(
            display,
            "[F-DATA-QUALITY-001] Tensor 'embedding': DENSITY FAILURE: 94.5% zeros"
        );
    }

    #[test]
    fn contract_validation_error_implements_std_error() {
        let err = ContractValidationError {
            tensor_name: "weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
            message: "Shape mismatch".to_string(),
        };
        // Verify it implements std::error::Error
        let std_err: &dyn std::error::Error = &err;
        let display_via_error = format!("{std_err}");
        assert!(display_via_error.contains("Shape mismatch"));
        // source() should return None (no wrapped error)
        assert!(std_err.source().is_none());
    }

    #[test]
    fn contract_validation_error_clone() {
        let err = ContractValidationError {
            tensor_name: "test".to_string(),
            rule_id: "F-001".to_string(),
            message: "fail".to_string(),
        };
        let cloned = err.clone();
        assert_eq!(cloned.tensor_name, err.tensor_name);
        assert_eq!(cloned.rule_id, err.rule_id);
        assert_eq!(cloned.message, err.message);
    }

    #[test]
    fn contract_validation_error_debug() {
        let err = ContractValidationError {
            tensor_name: "test".to_string(),
            rule_id: "F-001".to_string(),
            message: "fail".to_string(),
        };
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("ContractValidationError"));
        assert!(debug_str.contains("test"));
    }
