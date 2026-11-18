
    fn brick_with_samples(samples: &[f64]) -> BrickTiming {
        let mut b = BrickTiming::new("s", 1.0);
        b.samples = samples.to_vec();
        b
    }

    #[test]
    fn cv_percent_from_samples_degenerate_inputs_are_zero() {
        assert_eq!(cv_percent_from_samples(&[]), 0.0);
        assert_eq!(cv_percent_from_samples(&[5.0]), 0.0);
        assert_eq!(cv_percent_from_samples(&[0.0, 0.0]), 0.0);
        assert_eq!(cv_percent_from_samples(&[-1.0, -3.0]), 0.0); // mean <= 0
        assert_eq!(cv_percent_from_samples(&[10.0, 10.0, 10.0]), 0.0);
    }

    #[test]
    fn cv_percent_from_samples_uses_sample_stdev() {
        // mean 5, sum sq dev 32, sample variance 32/7, stdev/mean*100.
        let want = (32.0_f64 / 7.0).sqrt() / 5.0 * 100.0;
        let got = cv_percent_from_samples(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        // two samples: stdev = |a-b|/sqrt(2)
        let got2 = cv_percent_from_samples(&[1.0, 3.0]);
        assert!((got2 - (2.0_f64.sqrt() / 2.0 * 100.0)).abs() < 1e-9);
    }

    #[test]
    fn cv_percent_from_samples_nan_does_not_panic() {
        assert!(cv_percent_from_samples(&[1.0, f64::NAN, 3.0]).is_nan());
    }

    #[test]
    fn percentiles_from_brick_empty_and_single() {
        assert_eq!(percentiles_from_brick(&brick_with_samples(&[])), (0.0, 0.0));
        assert_eq!(percentiles_from_brick(&brick_with_samples(&[7.0])), (7.0, 7.0));
    }

    #[test]
    fn percentiles_from_brick_two_samples() {
        // p50 index = 2/2 = 1; p99 index = floor(1.98) = 1.
        assert_eq!(percentiles_from_brick(&brick_with_samples(&[9.0, 1.0])), (9.0, 9.0));
    }

    #[test]
    fn percentiles_from_brick_hundred_samples_unsorted() {
        // 1..=100 reversed: sorted[50] = 51, sorted[floor(99.0)=99] = 100.
        let rev: Vec<f64> = (1..=100).rev().map(f64::from).collect();
        assert_eq!(percentiles_from_brick(&brick_with_samples(&rev)), (51.0, 100.0));
    }

    #[test]
    fn percentiles_from_brick_nan_does_not_panic() {
        let (p50, p99) = percentiles_from_brick(&brick_with_samples(&[3.0, f64::NAN, 1.0]));
        assert!(p50.is_finite() || p50.is_nan());
        assert!(p99.is_finite() || p99.is_nan());
    }
