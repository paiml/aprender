use super::*;

#[test]
fn test_thermal_sample() {
    let sample = ThermalSample::new(65.0, 1.0);
    assert_eq!(sample.temperature_c, 65.0);
    assert!(sample.latency_us.is_none());

    let sample_with_latency = ThermalSample::with_latency(70.0, 2.0, 100.0);
    assert_eq!(sample_with_latency.latency_us, Some(100.0));
}

#[test]
fn test_analyzer_basic() {
    let mut analyzer = ThermalAnalyzer::new(10);

    analyzer.add(60.0, 0.0);
    analyzer.add(65.0, 1.0);
    analyzer.add(70.0, 2.0);

    assert_eq!(analyzer.sample_count(), 3);
    assert!(analyzer.has_sufficient_samples());
    assert_eq!(analyzer.current_temperature(), Some(70.0));
}

#[test]
fn test_trend_calculation() {
    let mut analyzer = ThermalAnalyzer::new(10);

    // Linear increase: 5°C/sec
    analyzer.add(60.0, 0.0);
    analyzer.add(65.0, 1.0);
    analyzer.add(70.0, 2.0);
    analyzer.add(75.0, 3.0);

    let trend = analyzer.calculate_trend().unwrap();
    assert!((trend - 5.0).abs() < 0.1);
}

#[test]
fn test_constant_temperature() {
    let mut analyzer = ThermalAnalyzer::new(10);

    analyzer.add(70.0, 0.0);
    analyzer.add(70.0, 1.0);
    analyzer.add(70.0, 2.0);

    let trend = analyzer.calculate_trend().unwrap();
    assert!(trend.abs() < 0.1); // No trend
}

#[test]
fn test_prediction() {
    let mut analyzer = ThermalAnalyzer::new(10);

    // 5°C/sec increase
    analyzer.add(60.0, 0.0);
    analyzer.add(65.0, 1.0);
    analyzer.add(70.0, 2.0);

    let prediction = analyzer.predict_trend(10.0).unwrap();

    // At t=12, should be 70 + 5*10 = 120°C
    assert!((prediction.predicted_temp_c - 120.0).abs() < 1.0);
    assert!((prediction.trend_slope - 5.0).abs() < 0.1);
}

#[test]
fn test_throttle_risk() {
    let risk = ThrottleRisk::assess(80.0, 85.0, 1.0, 10.0);

    // Close to threshold and trending up
    assert!(risk.probability > 0.5);
    assert_eq!(risk.margin_c, 5.0);
}

#[test]
fn test_risk_category() {
    assert_eq!(RiskCategory::from_probability(0.1), RiskCategory::Low);
    assert_eq!(RiskCategory::from_probability(0.3), RiskCategory::Moderate);
    assert_eq!(RiskCategory::from_probability(0.6), RiskCategory::High);
    assert_eq!(RiskCategory::from_probability(0.9), RiskCategory::Critical);
}

#[test]
fn test_cooldown_recommendation() {
    let cooldown = CooldownRecommendation::calculate(
        90.0, // Current temp
        75.0, // Target temp
        0.5,  // Cooling rate
    );

    // Need to cool 15°C at 0.5°C/sec = 30 seconds
    assert!((cooldown.duration_sec - 30.0).abs() < 0.1);
    assert!(cooldown.is_needed());
}

#[test]
fn test_no_cooldown_needed() {
    let cooldown = CooldownRecommendation::calculate(
        70.0, // Current temp
        75.0, // Target temp (already below target)
        0.5,
    );

    assert_eq!(cooldown.duration_sec, 0.0);
    assert!(!cooldown.is_needed());
}

#[test]
fn test_thermal_correlation() {
    let mut analyzer = ThermalAnalyzer::new(10);

    // Positive correlation: higher temp = higher latency
    analyzer.add_with_latency(60.0, 0.0, 100.0);
    analyzer.add_with_latency(65.0, 1.0, 110.0);
    analyzer.add_with_latency(70.0, 2.0, 120.0);
    analyzer.add_with_latency(75.0, 3.0, 130.0);
    analyzer.add_with_latency(80.0, 4.0, 140.0);
    analyzer.add_with_latency(85.0, 5.0, 150.0);

    let corr = analyzer.correlation_to_latency().unwrap();

    assert!(corr.pearson_r > 0.9); // Strong positive correlation
    assert!(corr.is_significant);
    assert!(corr.has_thermal_impact());
}

#[test]
fn test_insufficient_samples() {
    let mut analyzer = ThermalAnalyzer::new(10);

    analyzer.add(60.0, 0.0);
    analyzer.add(65.0, 1.0);

    assert!(!analyzer.has_sufficient_samples());
    assert!(analyzer.predict_trend(10.0).is_none());
}

#[test]
fn test_sliding_window() {
    let mut analyzer = ThermalAnalyzer::new(3);

    analyzer.add(60.0, 0.0);
    analyzer.add(65.0, 1.0);
    analyzer.add(70.0, 2.0);
    assert_eq!(analyzer.sample_count(), 3);

    analyzer.add(75.0, 3.0);
    assert_eq!(analyzer.sample_count(), 3); // Still 3

    // Oldest sample (60.0) should be gone
    let (min, _) = analyzer.temperature_range().unwrap();
    assert_eq!(min, 65.0);
}
