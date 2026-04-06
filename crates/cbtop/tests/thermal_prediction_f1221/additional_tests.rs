//! thermal_prediction_f1221 - Part 2

use cbtop::{analyze_thermal, RiskCategory, ThermalAnalyzer, ThermalPrediction};

// =============================================================================
// F1227: Insufficient Data Handling Tests
// =============================================================================

/// F1227.1: Returns None with insufficient samples
#[test]
fn f1227_insufficient_samples() {
    let mut analyzer = ThermalAnalyzer::new(10);

    analyzer.add(60.0, 0.0);
    analyzer.add(65.0, 1.0);

    // Only 2 samples, need at least 3
    assert!(!analyzer.has_sufficient_samples());
    assert!(analyzer.predict_trend(10.0).is_none());
    assert!(analyzer.calculate_trend().is_none());
}

/// F1227.2: Empty analyzer returns None
#[test]
fn f1227_empty_analyzer() {
    let analyzer = ThermalAnalyzer::new(10);

    assert!(analyzer.current_temperature().is_none());
    assert!(analyzer.average_temperature().is_none());
    assert!(analyzer.temperature_range().is_none());
    assert!(analyzer.predict_trend(10.0).is_none());
}

// =============================================================================
// F1228: Throttle Threshold Configuration Tests
// =============================================================================

/// F1228.1: Custom threshold works
#[test]
fn f1228_custom_threshold() {
    let mut analyzer = ThermalAnalyzer::new(10).with_threshold(95.0);

    for i in 0..5 {
        analyzer.add(85.0 + i as f64, i as f64);
    }

    let risk = analyzer.throttle_risk().unwrap();

    // With 95°C threshold, 89°C should be moderate risk
    assert!(risk.threshold_c == 95.0);
    assert!(risk.margin_c > 0.0);
}

/// F1228.2: Default threshold is 85°C
#[test]
fn f1228_default_threshold() {
    let mut analyzer = ThermalAnalyzer::new(10);

    for i in 0..5 {
        analyzer.add(70.0, i as f64);
    }

    let risk = analyzer.throttle_risk().unwrap();
    assert_eq!(risk.threshold_c, 85.0);
}

// =============================================================================
// F1229: Continuous Prediction Update Tests
// =============================================================================

/// F1229.1: Prediction updates with new samples
#[test]
fn f1229_prediction_updates() {
    let mut analyzer = ThermalAnalyzer::new(10);

    // Initial increasing trend
    for i in 0..5 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    let pred1 = analyzer.predict_trend(10.0).unwrap();

    // Add samples showing cooling
    for i in 5..10 {
        analyzer.add(65.0 - (i - 5) as f64, i as f64);
    }

    let pred2 = analyzer.predict_trend(10.0).unwrap();

    // Second prediction should show cooling (lower predicted temp)
    assert!(pred2.predicted_temp_c < pred1.predicted_temp_c);
}

/// F1229.2: Real-time update simulation
#[test]
fn f1229_realtime_update() {
    let mut analyzer = ThermalAnalyzer::new(20);
    let mut predictions = Vec::new();

    // Simulate real-time updates every second
    for i in 0..15 {
        analyzer.add(60.0 + i as f64 * 0.5, i as f64);

        if analyzer.has_sufficient_samples() {
            predictions.push(analyzer.predict_trend(5.0).unwrap().predicted_temp_c);
        }
    }

    // All predictions should be increasing
    for window in predictions.windows(2) {
        assert!(window[1] >= window[0] - 0.1); // Allow small tolerance
    }
}

// =============================================================================
// F1230: Thermal Variance Isolation Tests
// =============================================================================

/// F1230.1: Variance contribution percentage calculated
#[test]
fn f1230_variance_contribution() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Strong thermal correlation
    for i in 0..10 {
        let temp = 60.0 + i as f64;
        let latency = 100.0 + i as f64 * 5.0;
        analyzer.add_with_latency(temp, i as f64, latency);
    }

    let variance = analyzer.thermal_variance().unwrap();

    // Contribution should be between 0-100%
    assert!(variance.contribution_percent >= 0.0);
    assert!(variance.contribution_percent <= 100.0);
    // With strong correlation, should be high
    assert!(variance.contribution_percent > 50.0);
}

/// F1230.2: Temperature range tracked
#[test]
fn f1230_temp_range_tracked() {
    let mut analyzer = ThermalAnalyzer::new(20);

    for i in 0..10 {
        analyzer.add(60.0 + i as f64 * 2.0, i as f64);
    }

    let variance = analyzer.thermal_variance().unwrap();

    // Range should be 18°C (60 to 78)
    assert!((variance.temp_range_c - 18.0).abs() < 0.1);
    // Average should be 69°C
    assert!((variance.avg_temp_c - 69.0).abs() < 0.1);
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test prediction will_throttle
#[test]
fn test_will_throttle() {
    let prediction = ThermalPrediction {
        predicted_temp_c: 90.0,
        horizon_sec: 10.0,
        trend_slope: 1.0,
        confidence: 0.95,
        sample_count: 10,
    };

    assert!(prediction.will_throttle(85.0));
    assert!(!prediction.will_throttle(95.0));
}

/// Test time_to_throttle
#[test]
fn test_time_to_throttle() {
    let prediction = ThermalPrediction {
        predicted_temp_c: 90.0,
        horizon_sec: 10.0,
        trend_slope: 2.0, // 2°C/sec
        confidence: 0.95,
        sample_count: 10,
    };

    // Current at 80°C, threshold at 85°C, 2°C/sec = 2.5 seconds
    let time = prediction.time_to_throttle(80.0, 85.0).unwrap();
    assert!((time - 2.5).abs() < 0.1);
}

/// Test risk category names
#[test]
fn test_risk_category_names() {
    assert_eq!(RiskCategory::Low.name(), "low");
    assert_eq!(RiskCategory::Moderate.name(), "moderate");
    assert_eq!(RiskCategory::High.name(), "high");
    assert_eq!(RiskCategory::Critical.name(), "critical");
}

/// Test analyzer with custom cooling rate
#[test]
fn test_custom_cooling_rate() {
    let mut analyzer = ThermalAnalyzer::new(10).with_cooling_rate(1.0); // 1°C/sec

    for i in 0..5 {
        analyzer.add(90.0, i as f64);
    }

    let cooldown = analyzer.recommended_cooldown().unwrap();

    // Need to cool from 90 to 75 (15°C) at 1°C/sec = 15 seconds
    assert!((cooldown.duration_sec - 15.0).abs() < 0.1);
}

/// Test analyze_thermal convenience function
#[test]
fn test_analyze_thermal_function() {
    let samples = vec![(60.0, 0.0), (65.0, 1.0), (70.0, 2.0), (75.0, 3.0), (80.0, 4.0)];

    let prediction = analyze_thermal(&samples, 5.0).unwrap();

    // Trend is 5°C/sec, current is 80°C, 5 seconds ahead = 105°C
    // (but prediction is from current + slope * horizon)
    assert!(prediction.trend_slope > 4.0 && prediction.trend_slope < 6.0);
    assert!(prediction.predicted_temp_c > 100.0);
}

/// Test thermal correlation has_thermal_impact
#[test]
fn test_has_thermal_impact() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Strong correlation
    for i in 0..10 {
        analyzer.add_with_latency(60.0 + i as f64, i as f64, 100.0 + i as f64 * 5.0);
    }

    let corr = analyzer.correlation_to_latency().unwrap();
    assert!(corr.has_thermal_impact());
}

/// Test clear samples
#[test]
fn test_clear_samples() {
    let mut analyzer = ThermalAnalyzer::new(10);

    for i in 0..5 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    assert_eq!(analyzer.sample_count(), 5);

    analyzer.clear();
    assert_eq!(analyzer.sample_count(), 0);
}

/// Test samples access
#[test]
fn test_samples_access() {
    let mut analyzer = ThermalAnalyzer::new(10);

    analyzer.add(60.0, 0.0);
    analyzer.add(65.0, 1.0);

    let samples = analyzer.samples();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].temperature_c, 60.0);
}
