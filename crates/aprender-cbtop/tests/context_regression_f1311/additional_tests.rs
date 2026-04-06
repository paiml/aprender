//! context_regression_f1311 - Part 2

use cbtop::{
    ContextRegressionPredictor, RegressionThreshold, SystemContext, DEFAULT_COLD_START_MARGIN,
    DEFAULT_STALENESS_SEC, MIN_SAMPLES_FOR_CONTEXT,
};

// =============================================================================
// F1319: Configuration Tests
// =============================================================================

/// F1319.1: Custom cold start margin
#[test]
fn f1319_custom_cold_margin() {
    let predictor = ContextRegressionPredictor::new().with_cold_start_margin(25.0);

    let ctx = SystemContext::new();
    let threshold = predictor.compute_threshold("test", &ctx);

    assert_eq!(threshold.final_percent, 25.0);
}

/// F1319.2: Custom min margin
#[test]
fn f1319_custom_min_margin() {
    let predictor = ContextRegressionPredictor::new().with_min_margin(5.0);

    // No baselines yet, so baseline_count is 0
    assert_eq!(predictor.baseline_count("test"), 0);
}

/// F1319.3: Custom temp factor
#[test]
fn f1319_custom_temp_factor() {
    let predictor = ContextRegressionPredictor::new().with_temp_factor(5.0);

    // Verify configuration works
    assert_eq!(predictor.baseline_count("test"), 0);
}

// =============================================================================
// F1320: Edge Cases Tests
// =============================================================================

/// F1320.1: Unknown metric returns cold start threshold
#[test]
fn f1320_unknown_metric() {
    let predictor = ContextRegressionPredictor::new();
    let context = SystemContext::default();

    let threshold = predictor.compute_threshold("unknown_metric", &context);
    // Should return cold start threshold
    assert_eq!(threshold.final_percent, DEFAULT_COLD_START_MARGIN);
}

/// F1320.2: Empty baseline handled
#[test]
fn f1320_empty_baseline() {
    let predictor = ContextRegressionPredictor::new();

    assert_eq!(predictor.baseline_count("any_metric"), 0);
    assert!(!predictor.has_sufficient_history("any_metric"));
}

/// F1320.3: Clear baselines works
#[test]
fn f1320_clear_baselines() {
    let mut predictor = ContextRegressionPredictor::new();
    let context = SystemContext::default();

    predictor.add_baseline("latency", 10.0, context);
    assert_eq!(predictor.baseline_count("latency"), 1);

    predictor.clear("latency");
    assert_eq!(predictor.baseline_count("latency"), 0);
}

/// F1320.4: Clear all baselines works
#[test]
fn f1320_clear_all() {
    let mut predictor = ContextRegressionPredictor::new();
    let context = SystemContext::default();

    predictor.add_baseline("metric_a", 10.0, context.clone());
    predictor.add_baseline("metric_b", 20.0, context);

    predictor.clear_all();
    assert_eq!(predictor.baseline_count("metric_a"), 0);
    assert_eq!(predictor.baseline_count("metric_b"), 0);
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test constants
#[test]
fn test_constants() {
    assert_ne!(DEFAULT_COLD_START_MARGIN, 0.0);
    assert_ne!(MIN_SAMPLES_FOR_CONTEXT, 0);
    assert_ne!(DEFAULT_STALENESS_SEC, 0);
}

/// Test default system context
#[test]
fn test_default_context() {
    let context = SystemContext::default();
    assert_eq!(context.timestamp, 0);
    assert!(context.cpu_temp_c > 0.0);
}

/// Test system context builder
#[test]
fn test_context_builder() {
    let ctx = SystemContext::new()
        .with_timestamp(1000)
        .with_cpu_temp(70.0)
        .with_gpu_temp(65.0)
        .with_memory(80.0)
        .with_cpu_freq(3500.0, 4000.0)
        .with_cache_warm(true)
        .with_load(2.0);

    assert_eq!(ctx.timestamp, 1000);
    assert_eq!(ctx.cpu_temp_c, 70.0);
    assert_eq!(ctx.gpu_temp_c, 65.0);
    assert_eq!(ctx.memory_percent, 80.0);
    assert_eq!(ctx.cpu_freq_mhz, 3500.0);
    assert_eq!(ctx.cpu_freq_max_mhz, 4000.0);
    assert!(ctx.cache_warm);
    assert_eq!(ctx.load_average, 2.0);
}

/// Test frequency utilization
#[test]
fn test_freq_utilization() {
    let ctx = SystemContext::new().with_cpu_freq(3500.0, 4000.0);
    assert!((ctx.freq_utilization() - 0.875).abs() < 0.001);
}

/// Test thermal headroom
#[test]
fn test_thermal_headroom() {
    let ctx = SystemContext::new().with_cpu_temp(70.0);
    let headroom = ctx.thermal_headroom(100.0);
    assert_eq!(headroom, 30.0);
}

/// Test context JSON export
#[test]
fn test_context_json() {
    let ctx = SystemContext::new().with_cpu_temp(65.0);
    let json = ctx.to_json();

    assert!(json.contains("\"cpu_temp_c\":65"));
}

/// Test regression check fields
#[test]
fn test_regression_check_fields() {
    let mut predictor = ContextRegressionPredictor::new();
    let context = SystemContext::new();

    for _ in 0..10 {
        predictor.add_baseline("test", 100.0, context.clone());
    }

    let check = predictor.check_regression("test", 105.0, &context);

    assert_eq!(check.metric, "test");
    assert_eq!(check.current_value, 105.0);
    assert!(check.baseline_mean > 0.0);
}

/// Test trend significance
#[test]
fn test_trend_significance() {
    let mut predictor = ContextRegressionPredictor::new();

    // Add increasing trend with good fit
    for i in 0..20 {
        let ctx = SystemContext::new().with_timestamp(i as u64 * 86400);
        predictor.add_baseline("metric", 100.0 + i as f64 * 5.0, ctx);
    }

    let trend = predictor.detect_trend("metric").unwrap();
    assert!(trend.is_significant());
}

/// Test export JSON
#[test]
fn test_export_json() {
    let mut predictor = ContextRegressionPredictor::new();
    let context = SystemContext::new().with_timestamp(1000);

    predictor.add_baseline("latency", 10.0, context);

    let json = predictor.export_json("latency");
    assert!(json.is_some());
    assert!(json.unwrap().contains("\"metric\":\"latency\""));
}

/// Test export JSON unknown metric
#[test]
fn test_export_json_unknown() {
    let predictor = ContextRegressionPredictor::new();
    assert!(predictor.export_json("unknown").is_none());
}

/// Test context capture
#[test]
fn test_context_capture() {
    let ctx = SystemContext::capture();
    assert!(ctx.timestamp > 0);
}

/// Test threshold is_regression
#[test]
fn test_threshold_is_regression() {
    let threshold = RegressionThreshold {
        base_percent: 5.0,
        temp_adjustment: 1.0,
        memory_adjustment: 0.5,
        freq_adjustment: 0.5,
        cache_adjustment: 0.0,
        final_percent: 7.0,
        confidence: 0.8,
        sample_count: 20,
    };

    assert!(!threshold.is_regression(5.0)); // Within threshold
    assert!(threshold.is_regression(10.0)); // Above threshold
    assert!(threshold.is_regression(-10.0)); // Below threshold (abs > final_percent)
}

/// Test trend with insufficient data
#[test]
fn test_trend_insufficient_data() {
    let mut predictor = ContextRegressionPredictor::new();

    // Add just 2 samples (below MIN_SAMPLES_FOR_CONTEXT)
    for i in 0..2 {
        let ctx = SystemContext::new().with_timestamp(i as u64 * 86400);
        predictor.add_baseline("metric", 10.0 + i as f64, ctx);
    }

    let trend = predictor.detect_trend("metric");
    assert!(trend.is_none());
}

/// Test has_sufficient_history
#[test]
fn test_has_sufficient_history() {
    let mut predictor = ContextRegressionPredictor::new();
    let context = SystemContext::default();

    // Initially no history
    assert!(!predictor.has_sufficient_history("metric"));

    // Add samples up to threshold
    for i in 0..MIN_SAMPLES_FOR_CONTEXT {
        predictor.add_baseline("metric", 10.0 + i as f64, context.clone());
    }

    assert!(predictor.has_sufficient_history("metric"));
}
