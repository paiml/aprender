//! Tests for MOE routers - CapacityFactorRouter and PowerOfTwoChoicesRouter

use crate::error::RealizarError;
use crate::moe::{CapacityConfig, CapacityFactorRouter, PowerOfTwoChoicesRouter, PowerOfTwoConfig};

// ============================================================================
// CapacityFactorRouter Tests
// ============================================================================

#[test]
fn test_route_to_best_expert() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 10,
        num_experts: 4,
    });
    let scores = vec![0.1, 0.5, 0.3, 0.1];
    assert_eq!(router.route(&scores).expect("test"), 1);
}

#[test]
fn test_fallback_when_primary_full() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 1,
        num_experts: 4,
    });
    router.record_start(1); // Fill expert 1
    let scores = vec![0.1, 0.5, 0.3, 0.1];
    assert_eq!(router.route(&scores).expect("test"), 2); // Falls back to #2
}

#[test]
fn test_queue_depth_tracking() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 10,
        num_experts: 2,
    });
    assert_eq!(router.queue_depth(0), 0);
    router.record_start(0);
    assert_eq!(router.queue_depth(0), 1);
    router.record_end(0);
    assert_eq!(router.queue_depth(0), 0);
}

#[test]
fn test_wrong_score_count_error() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 10,
        num_experts: 4,
    });
    let scores = vec![0.5, 0.5]; // Wrong count
    assert!(router.route(&scores).is_err());
}

#[test]
fn test_capacity_factor_router_single_expert_at_capacity() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 1,
        num_experts: 1,
    });
    router.record_start(0); // Fill the only expert

    let scores = vec![0.9];
    let result = router.route(&scores);

    assert!(result.is_err());
    match result {
        Err(RealizarError::ExpertCapacityExceeded {
            expert_id,
            queue_depth,
            capacity,
        }) => {
            assert_eq!(expert_id, 0);
            assert_eq!(queue_depth, 1);
            assert_eq!(capacity, 1);
        },
        _ => panic!("Expected ExpertCapacityExceeded error"),
    }
}

#[test]
fn test_capacity_factor_router_both_top_experts_at_capacity() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 1,
        num_experts: 4,
    });

    router.record_start(1);
    router.record_start(2);

    let scores = vec![0.1, 0.9, 0.8, 0.2];
    let result = router.route(&scores);

    if let Ok(expert_id) = result {
        assert!(expert_id < 4, "Expert ID should be valid");
    }
}

#[test]
fn test_capacity_factor_top_k_with_nan_scores() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 10,
        num_experts: 3,
    });

    let scores = vec![f32::NAN, 0.5, 0.3];
    let result = router.route(&scores);
    assert!(result.is_ok());
}

#[test]
fn test_capacity_factor_all_equal_scores() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 10,
        num_experts: 4,
    });

    let scores = vec![0.5, 0.5, 0.5, 0.5];
    let result = router.route(&scores);
    assert!(result.is_ok());
    assert!(result.expect("test value should be present") < 4);
}

#[test]
fn test_capacity_config_debug() {
    let config = CapacityConfig {
        capacity: 10,
        num_experts: 4,
    };
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("capacity: 10"));
    assert!(debug_str.contains("num_experts: 4"));
}

#[test]
fn test_capacity_config_clone() {
    let config = CapacityConfig {
        capacity: 10,
        num_experts: 4,
    };
    let cloned = config.clone();
    assert_eq!(config.capacity, cloned.capacity);
    assert_eq!(config.num_experts, cloned.num_experts);
}

// ============================================================================
// PowerOfTwoChoicesRouter Tests
// ============================================================================

#[test]
fn test_power_of_two_choices_selects_least_loaded() {
    let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
        num_experts: 4,
        capacity: 100,
    });

    for _ in 0..50 {
        router.record_start(1);
    }

    let scores = vec![0.1, 0.9, 0.8, 0.1];
    let choice = router.route(&scores).expect("test");
    assert_eq!(choice, 2);
}

#[test]
fn test_power_of_two_choices_equal_load_picks_best_score() {
    let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
        num_experts: 4,
        capacity: 100,
    });

    let scores = vec![0.1, 0.9, 0.8, 0.1];
    let choice = router.route(&scores).expect("test");
    assert_eq!(choice, 1);
}

#[test]
fn test_power_of_two_choices_respects_capacity() {
    let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
        num_experts: 2,
        capacity: 5,
    });

    for _ in 0..5 {
        router.record_start(0);
        router.record_start(1);
    }

    let scores = vec![0.9, 0.8];
    let result = router.route(&scores);
    assert!(result.is_err());
}

#[test]
fn test_power_of_two_choices_wrong_score_count() {
    let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
        num_experts: 4,
        capacity: 10,
    });

    let scores = vec![0.5, 0.5];
    let result = router.route(&scores);

    assert!(result.is_err());
    match result {
        Err(RealizarError::MoeError(msg)) => {
            assert!(msg.contains("Expected 4 scores, got 2"));
        },
        _ => panic!("Expected MoeError"),
    }
}

#[test]
fn test_power_of_two_choices_queue_depth_tracking() {
    let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
        num_experts: 3,
        capacity: 10,
    });

    assert_eq!(router.queue_depth(0), 0);
    assert_eq!(router.queue_depth(1), 0);

    router.record_start(1);
    router.record_start(1);
    assert_eq!(router.queue_depth(1), 2);

    router.record_end(1);
    assert_eq!(router.queue_depth(1), 1);

    router.record_end(1);
    assert_eq!(router.queue_depth(1), 0);
}

#[test]
fn test_power_of_two_choices_with_nan_scores() {
    let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
        num_experts: 3,
        capacity: 10,
    });

    let scores = vec![f32::NAN, 0.5, 0.3];
    let result = router.route(&scores);
    assert!(result.is_ok());
}

#[test]
fn test_power_of_two_single_expert() {
    let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
        num_experts: 1,
        capacity: 5,
    });

    let scores = vec![0.9];
    let result = router.route(&scores);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present"), 0);

    for _ in 0..5 {
        router.record_start(0);
    }

    let result = router.route(&scores);
    assert!(result.is_err());
}

#[test]
fn test_power_of_two_config_debug() {
    let config = PowerOfTwoConfig {
        num_experts: 8,
        capacity: 20,
    };
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("num_experts: 8"));
    assert!(debug_str.contains("capacity: 20"));
}

#[test]
fn test_power_of_two_config_clone() {
    let config = PowerOfTwoConfig {
        num_experts: 8,
        capacity: 20,
    };
    let cloned = config.clone();
    assert_eq!(config.num_experts, cloned.num_experts);
    assert_eq!(config.capacity, cloned.capacity);
}

// ============================================================================
// Additional Router Coverage Tests
// ============================================================================

#[test]
fn test_capacity_factor_multiple_record_start_end() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 10,
        num_experts: 3,
    });

    // Multiple starts
    router.record_start(0);
    router.record_start(0);
    router.record_start(0);
    assert_eq!(router.queue_depth(0), 3);

    // Mixed operations
    router.record_end(0);
    assert_eq!(router.queue_depth(0), 2);

    router.record_start(0);
    assert_eq!(router.queue_depth(0), 3);

    // Clear all
    router.record_end(0);
    router.record_end(0);
    router.record_end(0);
    assert_eq!(router.queue_depth(0), 0);
}

#[test]
fn test_power_of_two_all_experts_partial_load() {
    let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
        num_experts: 4,
        capacity: 10,
    });

    // Load experts with different amounts
    router.record_start(0); // 1
    router.record_start(1);
    router.record_start(1); // 2
    router.record_start(2);
    router.record_start(2);
    router.record_start(2); // 3
    router.record_start(3);
    router.record_start(3);
    router.record_start(3);
    router.record_start(3); // 4

    // Scores favor experts 3 and 2
    let scores = vec![0.1, 0.2, 0.8, 0.9];
    let choice = router.route(&scores).expect("should route");
    // Should pick expert 2 (load=3) over expert 3 (load=4)
    assert_eq!(choice, 2);
}

#[test]
fn test_capacity_factor_negative_scores() {
    let router = CapacityFactorRouter::new(CapacityConfig {
        capacity: 10,
        num_experts: 3,
    });

    let scores = vec![-0.5, -0.1, -0.3];
    let result = router.route(&scores);
    assert!(result.is_ok());
    // -0.1 is the highest (least negative)
    assert_eq!(result.expect("test value should be present"), 1);
}

#[test]
fn test_power_of_two_tie_breaking_with_load() {
    let router = PowerOfTwoChoicesRouter::new(PowerOfTwoConfig {
        num_experts: 4,
        capacity: 10,
    });

    // Equal load on all
    let scores = vec![0.5, 0.5, 0.3, 0.3];
    let choice = router.route(&scores).expect("should route");
    // Should pick first of the tied top-2 (index 0 or 1)
    assert!(choice == 0 || choice == 1);
}
