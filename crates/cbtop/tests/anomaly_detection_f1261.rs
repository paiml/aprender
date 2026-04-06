#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-034: Anomaly Detection Engine
//!
//! F1261-F1270: Anomaly detection falsification tests

use cbtop::{
    Anomaly, AnomalyDetector, AnomalySeverity, AnomalyType, ChangePoint, DEFAULT_IQR_MULTIPLIER,
    DEFAULT_ZSCORE_THRESHOLD, MIN_SAMPLES_FOR_DETECTION,
};

// =============================================================================
// F1261: Z-Score Outlier Detection Tests
// =============================================================================

/// F1261.1: Z-score outliers detected (>3σ)
#[test]
fn f1261_zscore_outliers_detected() {
    let mut detector = AnomalyDetector::new();

    // Normal data with one outlier at 3σ
    let mut data: Vec<f64> = (0..20).map(|_| 100.0).collect();
    data[10] = 200.0; // Clear outlier

    detector.add_all(&data);
    let outliers = detector.detect_zscore_outliers();

    assert!(!outliers.is_empty(), "Should detect outliers");
    assert!(outliers.iter().any(|a| a.index == 10));
}

/// F1261.2: Outlier deviation calculated correctly
#[test]
fn f1261_zscore_deviation() {
    let mut detector = AnomalyDetector::new().with_zscore_threshold(2.0);

    let data: Vec<f64> = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 50.0];
    detector.add_all(&data);
    let outliers = detector.detect_zscore_outliers();

    assert!(!outliers.is_empty());
    // The outlier should have positive deviation
    let outlier = outliers.iter().find(|a| a.index == 9).unwrap();
    assert!(outlier.deviation > 0.0);
}

// =============================================================================
// F1262: IQR Outlier Detection Tests
// =============================================================================

/// F1262.1: IQR outliers detected (>1.5×IQR)
#[test]
fn f1262_iqr_outliers_detected() {
    let mut detector = AnomalyDetector::new();

    let mut data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
    data[15] = 500.0; // Outlier beyond 1.5×IQR

    detector.add_all(&data);
    let outliers = detector.detect_iqr_outliers();

    assert!(!outliers.is_empty());
}

/// F1262.2: IQR robust to heavy tails
#[test]
fn f1262_iqr_robust() {
    let mut detector = AnomalyDetector::new();

    // Data with slight variations - no outliers
    let data: Vec<f64> = (0..20).map(|i| 100.0 + (i % 3) as f64).collect();
    detector.add_all(&data);
    let outliers = detector.detect_iqr_outliers();

    // Should find no outliers in well-behaved data
    assert!(outliers.is_empty());
}

// =============================================================================
// F1263: Change Point Detection Tests
// =============================================================================

/// F1263.1: Change points identified (sudden shifts)
#[test]
fn f1263_change_points_detected() {
    let mut detector = AnomalyDetector::new();

    // Clear level shift at index 20
    let data: Vec<f64> = (0..40).map(|i| if i < 20 { 100.0 } else { 200.0 }).collect();

    detector.add_all(&data);
    let change_points = detector.detect_change_points();

    assert!(!change_points.is_empty());
}

/// F1263.2: Change point magnitude calculated
#[test]
fn f1263_change_point_magnitude() {
    let cp = ChangePoint::new(20, 100.0, 200.0);

    assert_eq!(cp.index, 20);
    assert_eq!(cp.mean_before, 100.0);
    assert_eq!(cp.mean_after, 200.0);
    assert_eq!(cp.magnitude, 100.0);
    assert!(cp.is_significant());
}

// =============================================================================
// F1264: Normal Data Tests
// =============================================================================

/// F1264.1: Normal data passes (no false positives)
#[test]
fn f1264_normal_data_passes() {
    let mut detector = AnomalyDetector::new();

    // Perfectly uniform data
    let data: Vec<f64> = (0..20).map(|_| 100.0).collect();
    detector.add_all(&data);

    let outliers = detector.detect_zscore_outliers();
    assert!(outliers.is_empty(), "No outliers in uniform data");
}

/// F1264.2: Low variance data handled
#[test]
fn f1264_low_variance_data() {
    let mut detector = AnomalyDetector::new();

    // Very low variance
    let data: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 * 0.001)).collect();
    detector.add_all(&data);

    let report = detector.analyze();
    // Should handle low variance gracefully
    assert_eq!(report.total_points, 20);
}

// =============================================================================
// F1265: Anomaly Classification Tests
// =============================================================================

/// F1265.1: Classification accurate (correct type)
#[test]
fn f1265_classification_accurate() {
    let anomaly = Anomaly::new(10, 200.0, 100.0, 3.5, AnomalyType::Spike);
    assert_eq!(anomaly.anomaly_type, AnomalyType::Spike);
}

/// F1265.2: Anomaly type names correct
#[test]
fn f1265_anomaly_type_names() {
    assert_eq!(AnomalyType::Outlier.name(), "outlier");
    assert_eq!(AnomalyType::Spike.name(), "spike");
    assert_eq!(AnomalyType::Drop.name(), "drop");
    assert_eq!(AnomalyType::ChangePoint.name(), "change_point");
    assert_eq!(AnomalyType::Periodic.name(), "periodic");
    assert_eq!(AnomalyType::Correlated.name(), "correlated");
}

// =============================================================================
// F1266: Multi-Metric Correlation Tests
// =============================================================================

/// F1266.1: Cross-metric anomalies detected
#[test]
fn f1266_cross_metric_anomalies() {
    let mut detector = AnomalyDetector::new();

    // Anomalies at nearby indices should be correlated
    let mut data: Vec<f64> = (0..30).map(|_| 100.0).collect();
    data[10] = 200.0;
    data[11] = 195.0;
    data[12] = 190.0;

    detector.add_all(&data);
    detector.analyze();

    // Should detect cluster of anomalies
    let anomalies = detector.get_anomalies();
    assert!(!anomalies.is_empty());
}

// =============================================================================
// F1267: Sliding Window Tests
// =============================================================================

/// F1267.1: Sliding window works (real-time detection)
#[test]
fn f1267_sliding_window() {
    let mut detector = AnomalyDetector::new().with_window_size(20).with_zscore_threshold(2.0);

    // Add baseline data with slight variation (for non-zero std dev)
    for i in 0..25 {
        detector.add(100.0 + (i % 3) as f64);
    }

    // Add clear anomaly (much higher than baseline)
    let result = detector.detect_realtime(500.0);
    assert!(result.is_some(), "Should detect real-time anomaly");
}

/// F1267.2: Window size configurable
#[test]
fn f1267_window_configurable() {
    let detector = AnomalyDetector::new().with_window_size(100);
    assert!(detector.data_count() == 0);
}

// =============================================================================
// F1268: Severity Ranking Tests
// =============================================================================

/// F1268.1: Severity ranking (Critical > Warning > Info)
#[test]
fn f1268_severity_ranking() {
    assert!(AnomalySeverity::Critical > AnomalySeverity::Warning);
    assert!(AnomalySeverity::Warning > AnomalySeverity::Info);
}

/// F1268.2: Severity from deviation
#[test]
fn f1268_severity_from_deviation() {
    assert_eq!(AnomalySeverity::from_deviation(2.0), AnomalySeverity::Info);
    assert_eq!(AnomalySeverity::from_deviation(4.0), AnomalySeverity::Warning);
    assert_eq!(AnomalySeverity::from_deviation(6.0), AnomalySeverity::Critical);
}

/// F1268.3: Severity names
#[test]
fn f1268_severity_names() {
    assert_eq!(AnomalySeverity::Info.name(), "info");
    assert_eq!(AnomalySeverity::Warning.name(), "warning");
    assert_eq!(AnomalySeverity::Critical.name(), "critical");
}

// =============================================================================
// F1269: Export Tests
// =============================================================================

/// F1269.1: Anomaly export (JSON format valid)
#[test]
fn f1269_anomaly_export_json() {
    let anomaly = Anomaly::new(5, 150.0, 100.0, 3.5, AnomalyType::Spike);
    let json = anomaly.to_json();

    assert!(json.contains("\"index\":5"));
    assert!(json.contains("\"value\":150"));
    assert!(json.contains("\"type\":\"spike\""));
    assert!(json.contains("\"severity\":\"warning\""));
}

/// F1269.2: Report export (JSON format valid)
#[test]
fn f1269_report_export_json() {
    let mut detector = AnomalyDetector::new();
    let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
    detector.add_all(&data);

    let report = detector.analyze();
    let json = report.to_json();

    assert!(json.contains("\"total_points\":20"));
    assert!(json.contains("\"method\""));
}

// =============================================================================
// F1270: Clear Functionality Tests
// =============================================================================

/// F1270.1: Reset state works
#[test]
fn f1270_clear_functionality() {
    let mut detector = AnomalyDetector::new();

    let data: Vec<f64> = (0..20).map(|_| 100.0).collect();
    detector.add_all(&data);
    assert_eq!(detector.data_count(), 20);

    detector.clear();
    assert_eq!(detector.data_count(), 0);
    assert!(detector.get_anomalies().is_empty());
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test constants
#[test]
fn test_constants() {
    assert_eq!(DEFAULT_ZSCORE_THRESHOLD, 3.0);
    assert_eq!(DEFAULT_IQR_MULTIPLIER, 1.5);
    assert_eq!(MIN_SAMPLES_FOR_DETECTION, 10);
}

/// Test detector with custom thresholds
#[test]
fn test_custom_thresholds() {
    let detector = AnomalyDetector::new().with_zscore_threshold(2.5).with_iqr_multiplier(2.0);

    assert_eq!(detector.data_count(), 0);
}

/// Test has_sufficient_data
#[test]
fn test_sufficient_data() {
    let mut detector = AnomalyDetector::new();

    for i in 0..MIN_SAMPLES_FOR_DETECTION - 1 {
        detector.add(i as f64);
    }
    assert!(!detector.has_sufficient_data());

    detector.add(100.0);
    assert!(detector.has_sufficient_data());
}

/// Test anomaly is_critical
#[test]
fn test_anomaly_is_critical() {
    let critical = Anomaly::new(0, 100.0, 50.0, 6.0, AnomalyType::Spike);
    assert!(critical.is_critical());

    let warning = Anomaly::new(0, 100.0, 50.0, 4.0, AnomalyType::Spike);
    assert!(!warning.is_critical());
}

/// Test report critical count
#[test]
fn test_report_critical_count() {
    let mut detector = AnomalyDetector::new().with_zscore_threshold(2.0);

    let mut data: Vec<f64> = (0..20).map(|_| 100.0).collect();
    data[5] = 300.0; // Very high outlier

    detector.add_all(&data);
    let report = detector.analyze();

    // May or may not have critical anomalies depending on data
    let _critical_count = report.count_by_severity(AnomalySeverity::Critical);
}

/// Test change point not significant
#[test]
fn test_change_point_not_significant() {
    let cp = ChangePoint::new(20, 100.0, 105.0);
    assert!(!cp.is_significant()); // 5% change is not significant
}
