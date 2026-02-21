
    #[test]
    fn test_quantile_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.quantile(0.5).is_err());
    }

    #[test]
    fn test_quantile_single_element() {
        let v = Vector::from_slice(&[42.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats
                .quantile(0.0)
                .expect("quantile should succeed for single element"),
            42.0
        );
        assert_eq!(
            stats
                .quantile(0.5)
                .expect("quantile should succeed for single element"),
            42.0
        );
        assert_eq!(
            stats
                .quantile(1.0)
                .expect("quantile should succeed for single element"),
            42.0
        );
    }

    #[test]
    fn test_quantile_two_elements() {
        let v = Vector::from_slice(&[1.0, 2.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats
                .quantile(0.0)
                .expect("quantile should succeed for two elements"),
            1.0
        );
        assert_eq!(
            stats
                .quantile(0.5)
                .expect("quantile should succeed for two elements"),
            1.5
        );
        assert_eq!(
            stats
                .quantile(1.0)
                .expect("quantile should succeed for two elements"),
            2.0
        );
    }

    #[test]
    fn test_quantile_odd_length() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats
                .quantile(0.5)
                .expect("quantile should succeed for odd length data"),
            3.0
        ); // exact median
    }

    #[test]
    fn test_quantile_even_length() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats
                .quantile(0.5)
                .expect("quantile should succeed for even length data"),
            2.5
        ); // interpolated median
    }

    #[test]
    fn test_quantile_edge_cases() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats.quantile(0.0).expect("min quantile should succeed"),
            1.0
        ); // min
        assert_eq!(
            stats.quantile(1.0).expect("max quantile should succeed"),
            5.0
        ); // max
    }

    #[test]
    fn test_quantile_invalid() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.quantile(-0.1).is_err());
        assert!(stats.quantile(1.1).is_err());
    }

    #[test]
    fn test_percentiles() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let p = stats
            .percentiles(&[25.0, 50.0, 75.0])
            .expect("percentiles should succeed for valid inputs");
        assert_eq!(p, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_five_number_summary() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let summary = stats
            .five_number_summary()
            .expect("five-number summary should succeed for valid data");

        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.q1, 2.0);
        assert_eq!(summary.median, 3.0);
        assert_eq!(summary.q3, 4.0);
        assert_eq!(summary.max, 5.0);
    }

    #[test]
    fn test_iqr() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(stats.iqr().expect("IQR should succeed for valid data"), 2.0);
    }

    // Histogram tests

    #[test]
    fn test_histogram_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram(3).is_err());
    }

    #[test]
    fn test_histogram_zero_bins() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram(0).is_err());
    }

    #[test]
    fn test_histogram_fixed_bins() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram(3)
            .expect("histogram should succeed for valid inputs");

        assert_eq!(hist.bins.len(), 4); // n_bins + 1
        assert_eq!(hist.counts.len(), 3);
        assert_eq!(hist.counts.iter().sum::<usize>(), 5); // Total count
    }

    #[test]
    fn test_histogram_uniform_distribution() {
        // Uniform distribution should have roughly equal counts
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram(5)
            .expect("histogram should succeed for uniform distribution");

        assert_eq!(hist.bins.len(), 6);
        assert_eq!(hist.counts.len(), 5);
        // Each bin should have exactly 2 values
        for count in hist.counts {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_histogram_all_same_value() {
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram(3)
            .expect("histogram should succeed for constant data");

        assert_eq!(hist.bins.len(), 2);
        assert_eq!(hist.counts.len(), 1);
        assert_eq!(hist.counts[0], 4);
    }

    #[test]
    fn test_histogram_sturges() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Sturges)
            .expect("histogram with Sturges method should succeed");

        // n = 8, so n_bins = ceil(log2(8)) + 1 = 3 + 1 = 4
        assert_eq!(hist.bins.len(), 5);
        assert_eq!(hist.counts.len(), 4);
        assert_eq!(hist.counts.iter().sum::<usize>(), 8);
    }

    #[test]
    fn test_histogram_square_root() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::SquareRoot)
            .expect("histogram with SquareRoot method should succeed");

        // n = 9, so n_bins = ceil(sqrt(9)) = 3
        assert_eq!(hist.bins.len(), 4);
        assert_eq!(hist.counts.len(), 3);
        assert_eq!(hist.counts.iter().sum::<usize>(), 9);
    }

    #[test]
    fn test_histogram_freedman_diaconis() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::FreedmanDiaconis)
            .expect("histogram with FreedmanDiaconis method should succeed");

        // Should have at least 1 bin
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);
        assert_eq!(hist.counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_histogram_auto() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_auto()
            .expect("auto histogram should succeed");

        // Auto uses Freedman-Diaconis by default
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);
    }

    #[test]
    fn test_histogram_edges_custom() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_edges(&[0.0, 2.5, 5.0, 10.0])
            .expect("histogram with custom edges should succeed");

        assert_eq!(hist.bins.len(), 4);
        assert_eq!(hist.counts.len(), 3);
        // Standard histogram convention:
        // Bin 0: [0.0, 2.5) -> values 1.0, 2.0
        // Bin 1: [2.5, 5.0) -> values 3.0, 4.0
        // Bin 2: [5.0, 10.0] -> value 5.0 (last bin is closed on both sides)
        assert_eq!(hist.counts[0], 2);
        assert_eq!(hist.counts[1], 2);
        assert_eq!(hist.counts[2], 1);
    }

    #[test]
    fn test_histogram_edges_invalid() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);

        // Too few edges
        assert!(stats.histogram_edges(&[1.0]).is_err());

        // Non-sorted edges
        assert!(stats.histogram_edges(&[5.0, 1.0, 10.0]).is_err());

        // Non-strictly increasing
        assert!(stats.histogram_edges(&[1.0, 5.0, 5.0, 10.0]).is_err());
    }

    // Bayesian Blocks tests
    #[test]
    fn test_histogram_bayesian_basic() {
        // Basic test: algorithm should run and produce valid histogram
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed");

        // Should produce valid histogram
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);

        // Bins should be sorted
        for i in 1..hist.bins.len() {
            assert!(hist.bins[i] > hist.bins[i - 1]);
        }

        // Counts should sum to number of data points
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_histogram_bayesian_uniform_data() {
        // Uniform data should produce relatively few bins
        let v = Vector::from_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for uniform data");

        // Uniform distribution should not need many bins
        assert!(hist.bins.len() <= 10); // Should be much fewer than 20
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);
    }

    #[test]
    fn test_histogram_bayesian_change_point_detection() {
        // Data with clear change points: two distinct clusters
        let v = Vector::from_slice(&[
            // Cluster 1: around 1-2
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, // Cluster 2: around 9-10
            9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0,
        ]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for clustered data");

        // Should detect the gap and create appropriate bins
        // Should have at least 2 bins to capture both clusters
        assert!(hist.bins.len() >= 3);

        // Verify bins cover the data range
        assert!(hist.bins[0] <= 1.0);
        assert!(
            *hist
                .bins
                .last()
                .expect("histogram should have at least one bin edge")
                >= 10.0
        );
    }

    #[test]
    fn test_histogram_bayesian_small_dataset() {
        // Small dataset - should still work
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for small dataset");

        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);

        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_histogram_bayesian_reproducibility() {
        // Same data should give same result (deterministic algorithm)
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0]);
        let stats = DescriptiveStats::new(&v);

        let hist1 = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("first Bayesian histogram should succeed");
        let hist2 = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("second Bayesian histogram should succeed");

        // Should produce identical results
        assert_eq!(hist1.bins.len(), hist2.bins.len());
        for (b1, b2) in hist1.bins.iter().zip(hist2.bins.iter()) {
            assert!((b1 - b2).abs() < 1e-6);
        }
        assert_eq!(hist1.counts, hist2.counts);
    }

    #[test]
    fn test_histogram_bayesian_single_value() {
        // All same value - should handle gracefully
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for constant data");

        // Should create at least 1 bin
        assert!(hist.bins.len() >= 2); // n+1 edges for n bins
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);

        // All values should be in bins
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_histogram_bayesian_vs_fixed_width() {
        // Compare Bayesian Blocks with fixed-width methods
        // Data with non-uniform distribution
        let v = Vector::from_slice(&[
            1.0, 1.5, 2.0, 2.5, 3.0, // Dense cluster
            10.0, 15.0, 20.0, // Sparse region
            30.0, 30.5, 31.0, 31.5, 32.0, // Another dense cluster
        ]);
        let stats = DescriptiveStats::new(&v);

        let hist_bayesian = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed");
        let hist_sturges = stats
            .histogram_method(BinMethod::Sturges)
            .expect("Sturges histogram should succeed");

        // Both should be valid
        assert!(hist_bayesian.bins.len() >= 2);
        assert!(hist_sturges.bins.len() >= 2);

        // Bayesian should adapt to data structure
        // (exact comparison depends on implementation, so we just verify it works)
        assert_eq!(hist_bayesian.bins.len(), hist_bayesian.counts.len() + 1);
    }
