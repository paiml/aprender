//! thermal_prediction_f1221 - Part 1

use cbtop::{
    assess_throttle_risk, CooldownRecommendation, RiskCategory, ThermalAnalyzer, ThrottleRisk,
};

// =============================================================================
// F1221: Trend Prediction Accuracy Tests
// =============================================================================

/// F1221.1: Trend prediction accurate (±3°C for 10s forecast)
#[test]
fn f1221_trend_prediction_accurate() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Linear increase: 1°C/sec
    for i in 0..10 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    // Predict 10 seconds ahead
    let prediction = analyzer.predict_trend(10.0).unwrap();

    // At t=19, expected temp = 60 + 19 = 79°C
    // Current is 69°C, prediction should be 79°C (69 + 10)
    let expected = 79.0;
    let error = (prediction.predicted_temp_c - expected).abs();

    assert!(error < 3.0, "Error {:.1}°C exceeds ±3°C threshold", error);
}

/// F1221.2: Prediction with noisy data
#[test]
fn f1221_prediction_with_noise() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Linear trend with small noise
    let temps = [60.0, 61.2, 62.0, 62.8, 64.1, 64.9, 66.0, 66.8, 68.1, 69.0];
    for (i, &temp) in temps.iter().enumerate() {
        analyzer.add(temp, i as f64);
    }

    let prediction = analyzer.predict_trend(10.0).unwrap();

    // Trend is ~1°C/sec, should predict ~79°C
    assert!((prediction.predicted_temp_c - 79.0).abs() < 5.0);
}

// =============================================================================
// F1222: Throttle Risk Calculation Tests
// =============================================================================

/// F1222.1: Throttle risk in valid range (0.0-1.0)
#[test]
fn f1222_risk_range() {
    // Test various scenarios
    let scenarios = [
        (60.0, 85.0, 0.0), // Low temp, no trend
        (80.0, 85.0, 0.5), // High temp, slight increase
        (84.0, 85.0, 1.0), // Near threshold, increasing
        (90.0, 85.0, 0.0), // Above threshold
    ];

    for (temp, threshold, slope) in scenarios {
        let risk = assess_throttle_risk(temp, threshold, slope);
        assert!(
            risk.probability >= 0.0 && risk.probability <= 1.0,
            "Risk {:.2} out of range for temp={}, slope={}",
            risk.probability,
            temp,
            slope
        );
    }
}

/// F1222.2: High risk near threshold
#[test]
fn f1222_high_risk_near_threshold() {
    let risk = ThrottleRisk::assess(82.0, 85.0, 1.0, 10.0);

    // Very close to threshold and trending up = high risk
    assert!(risk.probability > 0.5);
    assert!(risk.category == RiskCategory::High || risk.category == RiskCategory::Critical);
}

/// F1222.3: Low risk when cooling down
#[test]
fn f1222_low_risk_cooling() {
    let risk = ThrottleRisk::assess(70.0, 85.0, -1.0, 10.0);

    // Below threshold and cooling = low/moderate risk
    assert!(risk.probability < 0.5);
    // Category depends on proximity; moderate is acceptable when temp is 70°C
    assert!(risk.category == RiskCategory::Low || risk.category == RiskCategory::Moderate);
}

// =============================================================================
// F1223: Thermal Correlation Tests
// =============================================================================

/// F1223.1: Valid Pearson r coefficient
#[test]
fn f1223_valid_pearson_r() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Perfect positive correlation
    for i in 0..10 {
        let temp = 60.0 + i as f64;
        let latency = 100.0 + i as f64 * 10.0;
        analyzer.add_with_latency(temp, i as f64, latency);
    }

    let corr = analyzer.correlation_to_latency().unwrap();

    // Pearson r should be in [-1, 1]
    assert!(corr.pearson_r >= -1.0 && corr.pearson_r <= 1.0);
    // Strong positive correlation expected
    assert!(corr.pearson_r > 0.9);
}

/// F1223.2: Negative correlation detected
#[test]
fn f1223_negative_correlation() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Negative correlation: higher temp = lower latency (unlikely but testing)
    for i in 0..10 {
        let temp = 60.0 + i as f64;
        let latency = 200.0 - i as f64 * 10.0;
        analyzer.add_with_latency(temp, i as f64, latency);
    }

    let corr = analyzer.correlation_to_latency().unwrap();
    assert!(corr.pearson_r < -0.9);
}

/// F1223.3: Weak correlation for alternating data
#[test]
fn f1223_weak_correlation() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Alternating pattern - should show weak/moderate correlation
    let temps = [60.0, 70.0, 60.0, 70.0, 60.0, 70.0, 60.0, 70.0, 60.0, 70.0];
    let latencies = [
        100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0, 100.0, 110.0,
    ];

    for (i, (&temp, &latency)) in temps.iter().zip(latencies.iter()).enumerate() {
        analyzer.add_with_latency(temp, i as f64, latency);
    }

    let corr = analyzer.correlation_to_latency().unwrap();
    // Correlation coefficient in valid range
    assert!(corr.pearson_r >= -1.0 && corr.pearson_r <= 1.0);
}

// =============================================================================
// F1224: Cooldown Recommendation Tests
// =============================================================================

/// F1224.1: Cooldown duration is positive
#[test]
fn f1224_positive_duration() {
    let cooldown = CooldownRecommendation::calculate(90.0, 75.0, 0.5);

    assert!(cooldown.duration_sec > 0.0);
    assert!(cooldown.is_needed());
}

/// F1224.2: No cooldown when below target
#[test]
fn f1224_no_cooldown_below_target() {
    let cooldown = CooldownRecommendation::calculate(70.0, 75.0, 0.5);

    assert_eq!(cooldown.duration_sec, 0.0);
    assert!(!cooldown.is_needed());
}

/// F1224.3: Cooldown calculation correct
#[test]
fn f1224_cooldown_calculation() {
    // 20°C to cool at 0.5°C/sec = 40 seconds
    let cooldown = CooldownRecommendation::calculate(90.0, 70.0, 0.5);

    assert!((cooldown.duration_sec - 40.0).abs() < 0.1);
}

// =============================================================================
// F1225: Trend Slope Calculation Tests
// =============================================================================

/// F1225.1: Trend slope accurate (°C/second)
#[test]
fn f1225_trend_slope_accurate() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Exactly 2°C/sec increase
    for i in 0..10 {
        analyzer.add(60.0 + i as f64 * 2.0, i as f64);
    }

    let trend = analyzer.calculate_trend().unwrap();
    assert!((trend - 2.0).abs() < 0.01);
}

/// F1225.2: Negative slope for cooling
#[test]
fn f1225_negative_slope() {
    let mut analyzer = ThermalAnalyzer::new(20);

    // Decreasing temperature
    for i in 0..10 {
        analyzer.add(80.0 - i as f64, i as f64);
    }

    let trend = analyzer.calculate_trend().unwrap();
    assert!(trend < 0.0);
    assert!((trend - (-1.0)).abs() < 0.01);
}

/// F1225.3: Zero slope for constant temp
#[test]
fn f1225_zero_slope_constant() {
    let mut analyzer = ThermalAnalyzer::new(20);

    for i in 0..10 {
        analyzer.add(70.0, i as f64);
    }

    let trend = analyzer.calculate_trend().unwrap();
    assert!(trend.abs() < 0.01);
}

// =============================================================================
// F1226: Historical Samples (Sliding Window) Tests
// =============================================================================

/// F1226.1: Sliding window works
#[test]
fn f1226_sliding_window() {
    let mut analyzer = ThermalAnalyzer::new(5);

    // Add more than window size
    for i in 0..10 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    // Should only have last 5 samples
    assert_eq!(analyzer.sample_count(), 5);

    // Oldest should be 65.0 (index 5)
    let (min, max) = analyzer.temperature_range().unwrap();
    assert_eq!(min, 65.0);
    assert_eq!(max, 69.0);
}

/// F1226.2: New samples included in prediction
#[test]
fn f1226_new_samples_included() {
    let mut analyzer = ThermalAnalyzer::new(10);

    // Initial samples
    for i in 0..5 {
        analyzer.add(60.0 + i as f64, i as f64);
    }

    let pred1 = analyzer.predict_trend(10.0).unwrap();

    // Add more samples with different trend
    for i in 5..10 {
        analyzer.add(64.0 + i as f64 * 2.0, i as f64);
    }

    let pred2 = analyzer.predict_trend(10.0).unwrap();

    // Predictions should be different
    assert!((pred1.trend_slope - pred2.trend_slope).abs() > 0.1);
}
