
    #[test]
    fn test_histogram_bayesian_large_dataset() {
        // Larger dataset to test O(n²) scaling
        let mut data = Vec::new();
        for i in 0..50 {
            data.push(i as f32 / 10.0);
        }
        let v = Vector::from_slice(&data);
        let stats = DescriptiveStats::new(&v);

        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for large dataset");

        // Should complete in reasonable time and produce valid result
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);

        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 50);
    }

    // =========================================================================
    // Additional coverage: Debug/Clone/PartialEq on structs and enums
    // =========================================================================

    #[test]
    fn test_descriptive_stats_debug() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        let debug_str = format!("{:?}", stats);
        assert!(
            debug_str.contains("DescriptiveStats"),
            "Debug output should contain type name"
        );
    }

    #[test]
    fn test_five_number_summary_clone_partial_eq() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let summary = stats.five_number_summary().expect("Should compute summary");
        let cloned = summary.clone();
        assert_eq!(summary, cloned);
    }

    #[test]
    fn test_five_number_summary_debug() {
        let summary = FiveNumberSummary {
            min: 1.0,
            q1: 2.0,
            median: 3.0,
            q3: 4.0,
            max: 5.0,
        };
        let debug_str = format!("{:?}", summary);
        assert!(debug_str.contains("FiveNumberSummary"));
        assert!(debug_str.contains("min"));
    }

    #[test]
    fn test_histogram_clone_partial_eq() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram(3).expect("Should compute histogram");
        let cloned = hist.clone();
        assert_eq!(hist, cloned);
    }

    #[test]
    fn test_histogram_debug() {
        let hist = Histogram {
            bins: vec![0.0, 1.0, 2.0],
            counts: vec![5, 3],
            density: None,
        };
        let debug_str = format!("{:?}", hist);
        assert!(debug_str.contains("Histogram"));
        assert!(debug_str.contains("bins"));
        assert!(debug_str.contains("counts"));
    }

    #[test]
    fn test_histogram_with_density() {
        let hist = Histogram {
            bins: vec![0.0, 1.0, 2.0],
            counts: vec![5, 3],
            density: Some(vec![0.625, 0.375]),
        };
        let cloned = hist.clone();
        assert_eq!(hist, cloned);
        assert!(hist.density.is_some());
    }

    #[test]
    fn test_bin_method_clone_copy_partial_eq_debug() {
        let method = BinMethod::FreedmanDiaconis;
        let cloned = method;
        assert_eq!(method, cloned);

        let method2 = BinMethod::Sturges;
        assert_ne!(method, method2);

        let debug_str = format!("{:?}", BinMethod::Bayesian);
        assert!(debug_str.contains("Bayesian"));

        // Test all variants for Clone/Copy/PartialEq
        let methods = [
            BinMethod::FreedmanDiaconis,
            BinMethod::Sturges,
            BinMethod::Scott,
            BinMethod::SquareRoot,
            BinMethod::Bayesian,
        ];
        for &m in &methods {
            let copied = m;
            assert_eq!(m, copied);
        }
    }

    // =========================================================================
    // Additional coverage: error paths in percentiles, five_number_summary, iqr
    // =========================================================================

    #[test]
    fn test_percentiles_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.percentiles(&[50.0]).is_err());
    }

    #[test]
    fn test_percentiles_invalid_value() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.percentiles(&[101.0]).is_err());
        assert!(stats.percentiles(&[-1.0]).is_err());
    }

    #[test]
    fn test_percentiles_boundary_values() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let p = stats
            .percentiles(&[0.0, 100.0])
            .expect("Should handle boundary percentiles");
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 5.0);
    }

    #[test]
    fn test_five_number_summary_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.five_number_summary().is_err());
    }

    #[test]
    fn test_iqr_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.iqr().is_err());
    }

    #[test]
    fn test_iqr_single_element() {
        let v = Vector::from_slice(&[42.0]);
        let stats = DescriptiveStats::new(&v);
        let iqr_val = stats.iqr().expect("Should compute IQR for single element");
        assert!(
            (iqr_val - 0.0).abs() < 1e-10,
            "IQR of single element should be 0"
        );
    }

    // =========================================================================
    // Additional coverage: histogram_method error/edge paths
    // =========================================================================

    #[test]
    fn test_histogram_method_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram_method(BinMethod::Sturges).is_err());
        assert!(stats.histogram_method(BinMethod::Scott).is_err());
        assert!(stats.histogram_method(BinMethod::FreedmanDiaconis).is_err());
        assert!(stats.histogram_method(BinMethod::SquareRoot).is_err());
        assert!(stats.histogram_method(BinMethod::Bayesian).is_err());
    }

    #[test]
    fn test_histogram_auto_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram_auto().is_err());
    }

    #[test]
    fn test_histogram_freedman_diaconis_zero_iqr() {
        // All same values -> IQR = 0 -> error in Freedman-Diaconis
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let result = stats.histogram_method(BinMethod::FreedmanDiaconis);
        assert!(
            result.is_err(),
            "FreedmanDiaconis should fail when IQR is zero"
        );
    }

    #[test]
    fn test_histogram_scott_zero_std() {
        // All same values -> std = 0 -> error in Scott
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let result = stats.histogram_method(BinMethod::Scott);
        assert!(result.is_err(), "Scott should fail when std is zero");
    }

    #[test]
    fn test_histogram_scott_normal_data() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Scott)
            .expect("Scott method should succeed");
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.counts.iter().sum::<usize>(), 10);
    }

    // =========================================================================
    // Additional coverage: histogram_edges edge cases
    // =========================================================================

    #[test]
    fn test_histogram_edges_empty_data() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram_edges(&[0.0, 1.0, 2.0]).is_err());
    }

    #[test]
    fn test_histogram_edges_out_of_range_values() {
        // Some values fall outside the bin range and should be skipped
        let v = Vector::from_slice(&[-10.0, 1.0, 2.0, 3.0, 100.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_edges(&[0.0, 2.5, 5.0])
            .expect("Should handle out-of-range values");
        // Only 1.0, 2.0 are in [0.0, 2.5), 3.0 is in [2.5, 5.0]
        // -10.0 and 100.0 are out of range
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 3, "Only 3 values should be in range");
    }

    #[test]
    fn test_histogram_edges_single_bin() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_edges(&[0.0, 10.0])
            .expect("Should handle single bin");
        assert_eq!(hist.counts.len(), 1);
        assert_eq!(hist.counts[0], 3);
    }

    #[test]
    fn test_histogram_edges_value_on_boundary() {
        // Value exactly on an interior edge goes to the left bin (except last bin)
        let v = Vector::from_slice(&[2.5]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_edges(&[0.0, 2.5, 5.0])
            .expect("Should handle boundary value");
        // 2.5 is at the boundary between bins 0 and 1
        // Bin 0: [0.0, 2.5), Bin 1: [2.5, 5.0]
        // 2.5 goes to bin 1 (last bin is closed on both sides)
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 1);
    }

    // =========================================================================
    // Additional coverage: bayesian_blocks_edges edge cases
    // =========================================================================

    #[test]
    fn test_bayesian_blocks_single_element() {
        let v = Vector::from_slice(&[42.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian should handle single element");
        assert!(hist.bins.len() >= 2);
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn test_bayesian_blocks_two_elements() {
        let v = Vector::from_slice(&[1.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian should handle two elements");
        assert!(hist.bins.len() >= 2);
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 2);
    }

    // =========================================================================
    // Additional coverage: quantile interpolation paths
    // =========================================================================

    #[test]
    fn test_quantile_quartiles() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let stats = DescriptiveStats::new(&v);
        let q1 = stats.quantile(0.25).expect("Q1 should compute");
        let q3 = stats.quantile(0.75).expect("Q3 should compute");
        assert!(q1 < q3, "Q1 should be less than Q3");
        assert!(q1 >= 1.0 && q1 <= 8.0);
        assert!(q3 >= 1.0 && q3 <= 8.0);
    }

    #[test]
    fn test_quantile_unsorted_data() {
        // Data is not sorted; quantile should still work correctly
        let v = Vector::from_slice(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        let stats = DescriptiveStats::new(&v);
        let median = stats.quantile(0.5).expect("Median should compute");
        assert_eq!(median, 3.0, "Median of [1,2,3,4,5] should be 3.0");
    }

    #[test]
    fn test_percentiles_single_element() {
        let v = Vector::from_slice(&[7.0]);
        let stats = DescriptiveStats::new(&v);
        let p = stats
            .percentiles(&[0.0, 25.0, 50.0, 75.0, 100.0])
            .expect("Should compute percentiles for single element");
        for &val in &p {
            assert_eq!(
                val, 7.0,
                "All percentiles of single element should be that element"
            );
        }
    }

    #[test]
    fn test_five_number_summary_single_element() {
        let v = Vector::from_slice(&[42.0]);
        let stats = DescriptiveStats::new(&v);
        let summary = stats
            .five_number_summary()
            .expect("Should compute for single element");
        assert_eq!(summary.min, 42.0);
        assert_eq!(summary.q1, 42.0);
        assert_eq!(summary.median, 42.0);
        assert_eq!(summary.q3, 42.0);
        assert_eq!(summary.max, 42.0);
    }

    #[test]
    fn test_five_number_summary_two_elements() {
        let v = Vector::from_slice(&[1.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let summary = stats
            .five_number_summary()
            .expect("Should compute for two elements");
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 5.0);
        assert_eq!(summary.median, 3.0);
    }

    #[test]
    fn test_histogram_single_bin() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram(1).expect("Should work with 1 bin");
        assert_eq!(hist.bins.len(), 2);
        assert_eq!(hist.counts.len(), 1);
        assert_eq!(hist.counts[0], 5);
    }

    #[test]
    fn test_five_number_summary_partial_eq_ne() {
        let s1 = FiveNumberSummary {
            min: 1.0,
            q1: 2.0,
            median: 3.0,
            q3: 4.0,
            max: 5.0,
        };
        let s2 = FiveNumberSummary {
            min: 1.0,
            q1: 2.0,
            median: 3.0,
            q3: 4.0,
            max: 6.0, // different max
        };
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_histogram_partial_eq_ne() {
        let h1 = Histogram {
            bins: vec![0.0, 1.0],
            counts: vec![5],
            density: None,
        };
        let h2 = Histogram {
            bins: vec![0.0, 2.0], // different edge
            counts: vec![5],
            density: None,
        };
        assert_ne!(h1, h2);
    }
