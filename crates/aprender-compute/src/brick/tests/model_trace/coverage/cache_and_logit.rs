use super::super::super::super::*;

// =========================================================================
// F282-F283, F288: KvCache, LogitEvolution, and related tests
// =========================================================================

/// F282: KvCacheStateTrace utilization and window exhaustion
#[test]
fn test_f282_kv_cache_utilization() {
    // Test utilization calculation
    let mut trace = KvCacheStateTrace::new(50, 2048);
    trace.valid_positions = 1024;
    assert!((trace.utilization() - 0.5).abs() < 0.01);

    // Test window exhaustion
    assert!(!trace.is_window_exhausted());
    trace.valid_positions = 2048;
    assert!(trace.is_window_exhausted());

    // Test session thrashing detection
    let mut session = KvCacheSessionTrace::default();
    for step in 0..100 {
        let mut s = KvCacheStateTrace::new(step, 2048);
        s.valid_positions = step + 1;
        s.evictions_this_step = if step > 50 { 3 } else { 0 };
        session.add_step(s);
    }
    // 50 steps * 3 evictions = 150 evictions in last 50 steps
    assert!(session.has_thrashing(50, 0.5));
}

/// F283: LogitEvolutionTrace compute_rank edge cases
#[test]
fn test_f283_logit_rank_edge_cases() {
    // Single element
    let single = vec![5.0];
    assert_eq!(LogitEvolutionTrace::compute_rank(&single, 0), 0);

    // All same values
    let same = vec![3.0, 3.0, 3.0, 3.0];
    let rank = LogitEvolutionTrace::compute_rank(&same, 2);
    assert_eq!(rank, 0); // All tied at highest

    // Negative values
    let negative = vec![-5.0, -3.0, -1.0, -10.0];
    assert_eq!(LogitEvolutionTrace::compute_rank(&negative, 2), 0); // -1.0 is highest
    assert_eq!(LogitEvolutionTrace::compute_rank(&negative, 3), 3); // -10.0 is lowest
}

/// F288: LogitEvolutionTrace::finalize
#[test]
fn test_f288_logit_evolution_finalize() {
    let mut trace = LogitEvolutionTrace::new(100, 0.7, 0.9);

    // Track a token
    let token = trace.track_token(42, "hello".to_string());
    token.record_layer(0.5, 500);
    token.record_layer(1.0, 200);
    token.record_layer(5.0, 1);

    // Finalize with this token selected
    trace.finalize(42);
    // Decisive layer should be set based on token's evolution
    // The jump from 200 to 1 is the biggest - verify finalize completes without panic
    let _ = trace.decisive_layer;

    // Finalize with non-tracked token
    let mut trace2 = LogitEvolutionTrace::new(100, 0.7, 0.9);
    trace2.finalize(999); // Token not tracked
                          // Should not panic, just won't find decisive layer
}

/// F291: has_recency_bias with query_pos == 0
#[test]
fn test_f291_recency_bias_edge_case() {
    // Query position 0 - should always return false
    let weights = vec![0.8, 0.2];
    let trace = AttentionWeightTrace::from_weights(0, 0, 0, &weights, 2);
    assert!(!trace.has_recency_bias(5, 0.5)); // query_pos == 0, returns false
}

/// F292: LayerActivationTrace::new default values
#[test]
fn test_f292_layer_activation_trace_defaults() {
    let layer = LayerActivationTrace::new(5);
    assert_eq!(layer.layer_idx, 5);
    assert_eq!(layer.residual_ratio, 0.0);
    assert!(!layer.has_anomaly()); // All stats are default, no anomaly
    assert!(layer.anomaly_description().is_none());
}
