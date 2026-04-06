
    #[test]
    fn test_inspect_options_full() {
        let opts = InspectOptions::full();
        assert!(opts.include_weights);
        assert!(opts.include_quality);
        assert!(opts.verbose);
        assert_eq!(opts.max_weights, usize::MAX);
    }

    #[test]
    fn test_diff_item_display() {
        let item = DiffItem::new("field", "old", "new");
        let display = format!("{}", item);
        assert!(display.contains("field"));
        assert!(display.contains("old"));
        assert!(display.contains("new"));
        assert!(display.contains("->"));
    }

    #[test]
    fn test_weight_stats_norms() {
        let weights = vec![1.0_f32, 2.0, 3.0];
        let stats = WeightStats::from_slice(&weights);
        // L1 norm = |1| + |2| + |3| = 6
        assert!((stats.l1_norm - 6.0).abs() < 0.001);
        // L2 norm = sqrt(1 + 4 + 9) = sqrt(14)
        assert!((stats.l2_norm - 14.0_f64.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_header_inspection_default() {
        let header = HeaderInspection::default();
        assert_eq!(header.magic_string(), "APRN");
        assert_eq!(header.version_string(), "1.0");
    }

    #[test]
    fn test_weight_diff_diff_count_zero() {
        let diff = WeightDiff::empty();
        assert_eq!(diff.diff_count(), 0);
    }

    #[test]
    fn test_weight_diff_diff_count_nonzero() {
        let diff = WeightDiff {
            changed_count: 10,
            max_diff: 0.5,
            mean_diff: 0.2,
            l2_distance: 1.0,
            cosine_similarity: 0.95,
        };
        assert_eq!(diff.diff_count(), 1);
    }
