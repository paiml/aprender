use super::*;

#[test]
fn test_anomaly_severity() {
    assert_eq!(AnomalySeverity::from_deviation(2.0), AnomalySeverity::Info);
    assert_eq!(AnomalySeverity::from_deviation(3.5), AnomalySeverity::Warning);
    assert_eq!(AnomalySeverity::from_deviation(6.0), AnomalySeverity::Critical);
}

#[test]
fn test_anomaly_type_names() {
    assert_eq!(AnomalyType::Outlier.name(), "outlier");
    assert_eq!(AnomalyType::Spike.name(), "spike");
    assert_eq!(AnomalyType::ChangePoint.name(), "change_point");
}

#[test]
fn test_detector_creation() {
    let detector = AnomalyDetector::new();
    assert_eq!(detector.data_count(), 0);
    assert!(!detector.has_sufficient_data());
}

#[test]
fn test_zscore_detection() {
    let mut detector = AnomalyDetector::new();

    // Normal data with one outlier
    let mut data: Vec<f64> = (0..20).map(|_| 100.0).collect();
    data[10] = 200.0; // Outlier

    detector.add_all(&data);
    let outliers = detector.detect_zscore_outliers();

    assert!(!outliers.is_empty());
    assert!(outliers.iter().any(|a| a.index == 10));
}

#[test]
fn test_iqr_detection() {
    let mut detector = AnomalyDetector::new();

    let mut data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
    data[15] = 500.0; // Outlier

    detector.add_all(&data);
    let outliers = detector.detect_iqr_outliers();

    assert!(!outliers.is_empty());
}

#[test]
fn test_change_point_detection() {
    let mut detector = AnomalyDetector::new();

    // Data with clear change point at index 20
    let data: Vec<f64> = (0..40).map(|i| if i < 20 { 100.0 } else { 200.0 }).collect();

    detector.add_all(&data);
    let change_points = detector.detect_change_points();

    assert!(!change_points.is_empty());
}

#[test]
fn test_anomaly_export_json() {
    let anomaly = Anomaly::new(5, 150.0, 100.0, 3.5, AnomalyType::Spike);
    let json = anomaly.to_json();

    assert!(json.contains("\"index\":5"));
    assert!(json.contains("\"type\":\"spike\""));
    assert!(json.contains("\"severity\":\"warning\""));
}

#[test]
fn test_report_generation() {
    let mut detector = AnomalyDetector::new();

    let data: Vec<f64> = (0..30).map(|i| 100.0 + (i % 5) as f64).collect();
    detector.add_all(&data);

    let report = detector.analyze();
    assert_eq!(report.total_points, 30);
}
