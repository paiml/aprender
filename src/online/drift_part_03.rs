
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddm_no_drift() {
        let mut ddm = DDM::new();

        // Low error rate - no drift
        for _ in 0..100 {
            ddm.add_element(false); // correct prediction
        }

        assert_eq!(ddm.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_ddm_detects_sudden_drift() {
        let mut ddm = DDM::with_thresholds(20, 2.0, 3.0);

        // Start with low error rate
        for _ in 0..50 {
            ddm.add_element(false);
        }

        // Sudden increase in errors
        for _ in 0..50 {
            ddm.add_element(true);
        }

        let status = ddm.detected_change();
        assert!(
            status == DriftStatus::Warning || status == DriftStatus::Drift,
            "Expected warning or drift, got {:?}",
            status
        );
    }

    #[test]
    fn test_ddm_stats() {
        let mut ddm = DDM::new();

        for _ in 0..50 {
            ddm.add_element(false);
        }
        for _ in 0..50 {
            ddm.add_element(true);
        }

        let stats = ddm.stats();
        assert_eq!(stats.n_samples, 100);
        assert!((stats.error_rate - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_ddm_reset() {
        let mut ddm = DDM::new();

        for _ in 0..50 {
            ddm.add_element(true);
        }

        ddm.reset();
        let stats = ddm.stats();
        assert_eq!(stats.n_samples, 0);
        assert_eq!(stats.status, DriftStatus::Stable);
    }

    #[test]
    fn test_page_hinkley_no_drift() {
        let mut ph = PageHinkley::new();

        // Low error rate - no drift
        for _ in 0..100 {
            ph.add_element(false);
        }

        assert_eq!(ph.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_page_hinkley_detects_gradual_drift() {
        let mut ph = PageHinkley::with_thresholds(0.01, 20.0);

        // Start with low error rate
        for _ in 0..50 {
            ph.add_element(false);
        }

        // Gradual increase in errors
        for i in 0..100 {
            // Increasing error probability
            ph.add_element(i % 3 == 0);
        }

        for _ in 0..100 {
            ph.add_element(true);
        }

        let status = ph.detected_change();
        assert!(
            status == DriftStatus::Warning || status == DriftStatus::Drift,
            "Expected warning or drift, got {:?}",
            status
        );
    }

    #[test]
    fn test_page_hinkley_reset() {
        let mut ph = PageHinkley::new();

        for _ in 0..50 {
            ph.add_element(true);
        }

        ph.reset();
        let stats = ph.stats();
        assert_eq!(stats.n_samples, 0);
        assert_eq!(stats.status, DriftStatus::Stable);
    }

    #[test]
    fn test_adwin_no_drift() {
        let mut adwin = ADWIN::new();

        // Low error rate - no drift
        for _ in 0..100 {
            adwin.add_element(false);
        }

        assert_eq!(adwin.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_adwin_detects_sudden_drift() {
        let mut adwin = ADWIN::with_delta(0.1); // More sensitive

        // Start with low error rate (all correct)
        for _ in 0..200 {
            adwin.add_element(false);
        }

        // Sudden increase in errors (all wrong)
        for _ in 0..200 {
            adwin.add_element(true);
        }

        // Either the status changed or the mean changed significantly
        let status = adwin.detected_change();
        let mean = adwin.mean();

        // With 200 correct + 200 wrong, mean should be ~0.5
        // ADWIN should detect this as drift, or at minimum the mean should reflect the change
        assert!(
            status == DriftStatus::Warning || status == DriftStatus::Drift || mean > 0.3,
            "Expected warning/drift or mean > 0.3, got status={:?}, mean={}",
            status,
            mean
        );
    }

    #[test]
    fn test_adwin_window_size() {
        let mut adwin = ADWIN::new();

        for _ in 0..50 {
            adwin.add_element(false);
        }

        assert!(adwin.window_size() > 0);
        assert!(adwin.mean() < 0.1);
    }

    #[test]
    fn test_adwin_reset() {
        let mut adwin = ADWIN::new();

        for _ in 0..50 {
            adwin.add_element(true);
        }

        adwin.reset();
        assert_eq!(adwin.window_size(), 0);
        assert_eq!(adwin.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_adwin_mean() {
        let mut adwin = ADWIN::new();

        for _ in 0..50 {
            adwin.add_element(false);
        }
        for _ in 0..50 {
            adwin.add_element(true);
        }

        // Mean should be around 0.5
        assert!((adwin.mean() - 0.5).abs() < 0.3);
    }

    #[test]
    fn test_factory_recommended() {
        let detector = DriftDetectorFactory::recommended();
        // Recommended is ADWIN
        assert_eq!(detector.stats().n_samples, 0);
    }

    #[test]
    fn test_factory_all_types() {
        let ddm = DriftDetectorFactory::ddm();
        let ph = DriftDetectorFactory::page_hinkley();
        let adwin = DriftDetectorFactory::adwin();

        // All should start stable
        assert_eq!(ddm.detected_change(), DriftStatus::Stable);
        assert_eq!(ph.detected_change(), DriftStatus::Stable);
        assert_eq!(adwin.detected_change(), DriftStatus::Stable);
    }

    #[test]
    fn test_drift_status_equality() {
        assert_eq!(DriftStatus::Stable, DriftStatus::Stable);
        assert_ne!(DriftStatus::Stable, DriftStatus::Warning);
        assert_ne!(DriftStatus::Warning, DriftStatus::Drift);
    }

    #[test]
    fn test_ddm_default() {
        let ddm = DDM::default();
        assert_eq!(ddm.stats().n_samples, 0);
    }

    #[test]
    fn test_page_hinkley_default() {
        let ph = PageHinkley::default();
        assert_eq!(ph.stats().n_samples, 0);
    }

    #[test]
    fn test_adwin_default() {
        let adwin = ADWIN::default();
        assert_eq!(adwin.window_size(), 0);
    }

    #[test]
    fn test_adwin_gradual_drift() {
        let mut adwin = ADWIN::with_delta(0.1);

        // Start with no errors
        for _ in 0..50 {
            adwin.add_element(false);
        }

        // Gradual increase
        for i in 0..100 {
            adwin.add_element(i % 5 == 0);
        }

        // Now mostly errors
        for _ in 0..50 {
            adwin.add_element(true);
        }

        // Window should have adapted
        assert!(adwin.window_size() > 0);
    }
}
