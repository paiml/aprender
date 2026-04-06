use super::*;

#[test]
fn test_system_context() {
    let ctx = SystemContext::new()
        .with_cpu_temp(70.0)
        .with_memory(60.0)
        .with_cpu_freq(3500.0, 4000.0);

    assert_eq!(ctx.cpu_temp_c, 70.0);
    assert_eq!(ctx.memory_percent, 60.0);
    assert!((ctx.freq_utilization() - 0.875).abs() < 0.001);
}

#[test]
fn test_context_json() {
    let ctx = SystemContext::new().with_cpu_temp(65.0);
    let json = ctx.to_json();

    assert!(json.contains("\"cpu_temp_c\":65"));
}

#[test]
fn test_predictor_creation() {
    let predictor = ContextRegressionPredictor::new();
    assert_eq!(predictor.baseline_count("test"), 0);
}

#[test]
fn test_cold_start_margin() {
    let predictor = ContextRegressionPredictor::new();
    let ctx = SystemContext::new();

    let threshold = predictor.compute_threshold("test", &ctx);
    assert_eq!(threshold.final_percent, DEFAULT_COLD_START_MARGIN);
    assert!(threshold.confidence < 0.5);
}

#[test]
fn test_learned_threshold() {
    let mut predictor = ContextRegressionPredictor::new();

    // Add baseline entries
    for i in 0..10 {
        let ctx = SystemContext::new().with_timestamp(i as u64 * 86400);
        predictor.add_baseline("latency", 100.0 + (i % 3) as f64, ctx);
    }

    let current = SystemContext::new().with_timestamp(100 * 86400);
    let threshold = predictor.compute_threshold("latency", &current);

    assert!(threshold.final_percent < DEFAULT_COLD_START_MARGIN);
    assert!(threshold.confidence > 0.1);
}

#[test]
fn test_regression_check() {
    let mut predictor = ContextRegressionPredictor::new();

    for i in 0..10 {
        let ctx = SystemContext::new();
        predictor.add_baseline("throughput", 1000.0 + (i % 5) as f64, ctx);
    }

    let ctx = SystemContext::new();

    // No regression
    let check = predictor.check_regression("throughput", 1002.0, &ctx);
    assert!(!check.is_regression);

    // Clear regression (50% drop)
    let check = predictor.check_regression("throughput", 500.0, &ctx);
    assert!(check.is_regression);
}

#[test]
fn test_trend_detection() {
    let mut predictor = ContextRegressionPredictor::new();

    // Add increasing trend
    for i in 0..20 {
        let ctx = SystemContext::new().with_timestamp(i as u64 * 86400);
        predictor.add_baseline("metric", 100.0 + i as f64 * 2.0, ctx);
    }

    let trend = predictor.detect_trend("metric").unwrap();
    assert!(trend.slope_per_day > 0.0);
    assert_eq!(trend.direction, "increasing");
}
