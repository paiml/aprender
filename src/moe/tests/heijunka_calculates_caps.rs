//! Tests for HeijunkaController

use crate::moe::{HeijunkaConfig, HeijunkaController, LoadSheddingDecision};

#[test]
fn test_heijunka_calculates_optimal_concurrency() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 100,
    });

    // Little's Law: L = lambda * W
    // arrival_rate = 10 req/s, latency = 100ms = 0.1s
    // Optimal = 10 * 0.1 = 1
    let concurrency = controller.optimal_concurrency(10.0, 100.0);
    assert_eq!(concurrency, 1);
}

#[test]
fn test_heijunka_caps_at_max_concurrency() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 10,
    });

    let concurrency = controller.optimal_concurrency(1000.0, 100.0);
    assert_eq!(concurrency, 10);
}

#[test]
fn test_heijunka_load_leveling_decision() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 50,
    });

    let decision = controller.should_shed_load(150.0, 50);
    assert!(decision.shed_load);

    let decision = controller.should_shed_load(50.0, 10);
    assert!(!decision.shed_load);
}

#[test]
fn test_heijunka_target_latency_getter() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 150.0,
        max_concurrency: 50,
    });

    assert!((controller.target_latency_ms() - 150.0).abs() < f64::EPSILON);
}

#[test]
fn test_heijunka_minimum_concurrency_floor() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 100,
    });

    let concurrency = controller.optimal_concurrency(0.001, 1.0);
    assert_eq!(concurrency, 1);

    let concurrency = controller.optimal_concurrency(0.0, 100.0);
    assert_eq!(concurrency, 1);
}

#[test]
fn test_heijunka_recommended_concurrency_bounds() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 10,
    });

    let decision = controller.should_shed_load(1000.0, 10);
    assert!(decision.shed_load);
    assert!(decision.recommended_concurrency >= 1);
    assert!(decision.recommended_concurrency <= 10);

    let decision = controller.should_shed_load(10.0, 5);
    assert!(!decision.shed_load);
    assert!(decision.recommended_concurrency >= 1);
    assert!(decision.recommended_concurrency <= 10);
}

#[test]
fn test_heijunka_no_shed_when_under_max_concurrency() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 50,
    });

    let decision = controller.should_shed_load(200.0, 30);
    assert!(!decision.shed_load);
}

#[test]
fn test_heijunka_config_debug() {
    let config = HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 50,
    };
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("target_latency_ms: 100.0"));
    assert!(debug_str.contains("max_concurrency: 50"));
}

#[test]
fn test_heijunka_config_clone() {
    let config = HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 50,
    };
    let cloned = config.clone();
    assert!((config.target_latency_ms - cloned.target_latency_ms).abs() < f64::EPSILON);
    assert_eq!(config.max_concurrency, cloned.max_concurrency);
}

#[test]
fn test_load_shedding_decision_debug() {
    let decision = LoadSheddingDecision {
        shed_load: true,
        recommended_concurrency: 5,
    };
    let debug_str = format!("{:?}", decision);
    assert!(debug_str.contains("shed_load: true"));
    assert!(debug_str.contains("recommended_concurrency: 5"));
}

#[test]
fn test_load_shedding_decision_clone() {
    let decision = LoadSheddingDecision {
        shed_load: true,
        recommended_concurrency: 5,
    };
    let cloned = decision.clone();
    assert_eq!(decision.shed_load, cloned.shed_load);
    assert_eq!(
        decision.recommended_concurrency,
        cloned.recommended_concurrency
    );
}

#[test]
fn test_heijunka_high_arrival_rate() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 50.0,
        max_concurrency: 100,
    });

    // 500 req/s with 50ms latency = 500 * 0.05 = 25 concurrent
    let concurrency = controller.optimal_concurrency(500.0, 50.0);
    assert_eq!(concurrency, 25);
}

#[test]
fn test_heijunka_very_low_latency() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 10.0,
        max_concurrency: 50,
    });

    // 100 req/s with 1ms latency = 100 * 0.001 = 0.1 -> ceil = 1
    let concurrency = controller.optimal_concurrency(100.0, 1.0);
    assert_eq!(concurrency, 1);
}

#[test]
fn test_heijunka_shed_load_boundary() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 10,
    });

    // Exactly at target latency and max concurrency - should not shed
    let decision = controller.should_shed_load(100.0, 10);
    assert!(!decision.shed_load);

    // Slightly over latency and at max - should shed
    let decision = controller.should_shed_load(100.1, 10);
    assert!(decision.shed_load);
}

#[test]
fn test_heijunka_recommended_concurrency_calculation() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 100.0,
        max_concurrency: 50,
    });

    // Current latency is 2x target, current concurrency is 20
    // ratio = 100/200 = 0.5
    // recommended = ceil(20 * 0.5) = 10
    let decision = controller.should_shed_load(200.0, 20);
    assert_eq!(decision.recommended_concurrency, 10);
}

#[test]
fn test_heijunka_extreme_values() {
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 1.0,
        max_concurrency: 1000,
    });

    // Test with very high arrival rate
    let concurrency = controller.optimal_concurrency(10000.0, 10.0);
    assert_eq!(concurrency, 100);

    // Test with very low values
    let controller = HeijunkaController::new(HeijunkaConfig {
        target_latency_ms: 1000.0,
        max_concurrency: 5,
    });
    let decision = controller.should_shed_load(500.0, 3);
    assert!(!decision.shed_load);
}
