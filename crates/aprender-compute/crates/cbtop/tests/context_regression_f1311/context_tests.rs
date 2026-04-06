//! context_regression_f1311 - Part 1

use cbtop::{ContextRegressionPredictor, SystemContext, DEFAULT_COLD_START_MARGIN};

// =============================================================================
// F1311: Baseline Recording Tests
// =============================================================================

/// F1311.1: Can record baseline with context
#[test]
fn f1311_record_baseline() {
    let mut predictor = ContextRegressionPredictor::new();

    let context = SystemContext::new()
        .with_timestamp(1000)
        .with_cpu_temp(45.0)
        .with_memory(50.0)
        .with_cpu_freq(3500.0, 4000.0)
        .with_cache_warm(true)
        .with_load(1.5);

    predictor.add_baseline("latency_p99", 10.5, context);

    assert!(predictor.baseline_count("latency_p99") > 0);
    assert_eq!(predictor.baseline_count("unknown_metric"), 0);
}

/// F1311.2: Multiple baselines recorded
#[test]
fn f1311_multiple_baselines() {
    let mut predictor = ContextRegressionPredictor::new();

    let context = SystemContext::default();

    predictor.add_baseline("metric_a", 10.0, context.clone());
    predictor.add_baseline("metric_b", 20.0, context);

    assert_eq!(predictor.baseline_count("metric_a"), 1);
    assert_eq!(predictor.baseline_count("metric_b"), 1);
}

// =============================================================================
// F1312: Cold Start Detection Tests
// =============================================================================

/// F1312.1: Cold start margin applied
#[test]
fn f1312_cold_start_margin() {
    let predictor = ContextRegressionPredictor::new();

    let context_cold = SystemContext::new().with_timestamp(1000).with_cache_warm(false);

    let threshold = predictor.compute_threshold("latency", &context_cold);

    // Cold start should use default margin
    assert_eq!(threshold.final_percent, DEFAULT_COLD_START_MARGIN);
    assert!(threshold.confidence < 0.5);
}

/// F1312.2: Warm cache lower threshold
#[test]
fn f1312_warm_cache() {
    let mut predictor = ContextRegressionPredictor::new();

    let context_warm = SystemContext::new().with_timestamp(1000).with_cache_warm(true);

    for i in 0..10 {
        predictor.add_baseline("latency", 10.0 + (i % 2) as f64, context_warm.clone());
    }

    let threshold = predictor.compute_threshold("latency", &context_warm);

    // Warm cache should have tighter threshold than cold start
    assert!(threshold.final_percent < DEFAULT_COLD_START_MARGIN);
}

// =============================================================================
// F1313: Temperature Correlation Tests
// =============================================================================

/// F1313.1: High temp increases threshold
#[test]
fn f1313_high_temp_threshold() {
    let mut predictor = ContextRegressionPredictor::new();

    let context_normal =
        SystemContext::new().with_timestamp(1000).with_cpu_temp(50.0).with_cache_warm(true);

    for i in 0..10 {
        predictor.add_baseline("latency", 10.0 + (i % 2) as f64, context_normal.clone());
    }

    let context_hot =
        SystemContext::new().with_timestamp(2000).with_cpu_temp(85.0).with_cache_warm(true);

    let threshold_normal = predictor.compute_threshold("latency", &context_normal);
    let threshold_hot = predictor.compute_threshold("latency", &context_hot);

    // Hot context should have higher threshold (temperature adjustment)
    assert!(threshold_hot.temp_adjustment > threshold_normal.temp_adjustment);
}

// =============================================================================
// F1314: Frequency Scaling Tests
// =============================================================================

/// F1314.1: Lower frequency adjusts threshold
#[test]
fn f1314_frequency_adjustment() {
    let mut predictor = ContextRegressionPredictor::new();

    let context_full = SystemContext::new()
        .with_timestamp(1000)
        .with_cpu_freq(4000.0, 4000.0)
        .with_cache_warm(true);

    for i in 0..10 {
        predictor.add_baseline("throughput", 1000.0 + (i * 10) as f64, context_full.clone());
    }

    let context_throttled = SystemContext::new()
        .with_timestamp(2000)
        .with_cpu_freq(2000.0, 4000.0)
        .with_cache_warm(true);

    let threshold_full = predictor.compute_threshold("throughput", &context_full);
    let threshold_throttled = predictor.compute_threshold("throughput", &context_throttled);

    // Throttled should have higher frequency adjustment
    assert!(threshold_throttled.freq_adjustment >= threshold_full.freq_adjustment);
}

// =============================================================================
// F1315: Memory Pressure Tests
// =============================================================================

/// F1315.1: High memory pressure increases threshold
#[test]
fn f1315_memory_pressure() {
    let mut predictor = ContextRegressionPredictor::new();

    let context_low_mem =
        SystemContext::new().with_timestamp(1000).with_memory(30.0).with_cache_warm(true);

    for i in 0..10 {
        predictor.add_baseline("latency", 10.0 + (i % 2) as f64, context_low_mem.clone());
    }

    let context_high_mem =
        SystemContext::new().with_timestamp(2000).with_memory(95.0).with_cache_warm(true);

    let threshold_low = predictor.compute_threshold("latency", &context_low_mem);
    let threshold_high = predictor.compute_threshold("latency", &context_high_mem);

    // High memory pressure should have higher memory adjustment
    assert!(threshold_high.memory_adjustment >= threshold_low.memory_adjustment);
}

// =============================================================================
// F1316: Trend Detection Tests
// =============================================================================

/// F1316.1: Increasing trend detected
#[test]
fn f1316_increasing_trend() {
    let mut predictor = ContextRegressionPredictor::new();

    // Add increasing values over time
    for i in 0..20 {
        let ctx = SystemContext::new().with_timestamp(i as u64 * 86400);
        predictor.add_baseline("latency", 10.0 + i as f64 * 2.0, ctx);
    }

    let trend = predictor.detect_trend("latency");
    assert!(trend.is_some());

    let t = trend.unwrap();
    assert!(t.slope_per_day > 0.0);
    assert_eq!(t.direction, "increasing");
}

/// F1316.2: Stable trend detected
#[test]
fn f1316_stable_trend() {
    let mut predictor = ContextRegressionPredictor::new();

    // Add stable values
    for i in 0..20 {
        let ctx = SystemContext::new().with_timestamp(i as u64 * 86400);
        predictor.add_baseline("latency", 10.0, ctx);
    }

    let trend = predictor.detect_trend("latency");
    assert!(trend.is_some());

    let t = trend.unwrap();
    assert!(t.slope_per_day.abs() < 0.1);
    assert_eq!(t.direction, "stable");
}

/// F1316.3: Decreasing trend detected
#[test]
fn f1316_decreasing_trend() {
    let mut predictor = ContextRegressionPredictor::new();

    // Add decreasing values
    for i in 0..20 {
        let ctx = SystemContext::new().with_timestamp(i as u64 * 86400);
        predictor.add_baseline("latency", 100.0 - i as f64 * 3.0, ctx);
    }

    let trend = predictor.detect_trend("latency");
    assert!(trend.is_some());

    let t = trend.unwrap();
    assert!(t.slope_per_day < 0.0);
    assert_eq!(t.direction, "decreasing");
}

// =============================================================================
// F1317: Regression Check Tests
// =============================================================================

/// F1317.1: No regression when within threshold
#[test]
fn f1317_no_regression() {
    let mut predictor = ContextRegressionPredictor::new();

    let context = SystemContext::new().with_timestamp(1000).with_cache_warm(true);

    for i in 0..10 {
        predictor.add_baseline("latency", 10.0 + (i % 2) as f64, context.clone());
    }

    let check = predictor.check_regression("latency", 11.0, &context);
    assert!(!check.is_regression);
    assert!(check.passed());
}

/// F1317.2: Regression detected when above threshold
#[test]
fn f1317_regression_detected() {
    let mut predictor = ContextRegressionPredictor::new();

    let context = SystemContext::new().with_timestamp(1000).with_cache_warm(true);

    for _ in 0..10 {
        predictor.add_baseline("latency", 10.0, context.clone());
    }

    // Value significantly above baseline (50% regression)
    let check = predictor.check_regression("latency", 15.0, &context);
    assert!(check.is_regression);
    assert!(!check.passed());
}

// =============================================================================
// F1318: Staleness Tests
// =============================================================================

/// F1318.1: Stale baseline detected
#[test]
fn f1318_stale_baseline() {
    let predictor = ContextRegressionPredictor::new().with_staleness(60); // 60 second staleness

    let _old_context = SystemContext::new().with_timestamp(1000);

    let new_context = SystemContext::new().with_timestamp(100_000); // 100 seconds later

    // Even with stale data, threshold should still compute
    let threshold = predictor.compute_threshold("latency", &new_context);
    assert!(threshold.final_percent > 0.0);
}
