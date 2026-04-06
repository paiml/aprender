pub(crate) use super::*;

pub(super) fn make_result(model_id: &str, size: u64, success: f64) -> EvalResult {
    let mut result = EvalResult::new(model_id, "test", size);
    result.overall_success_rate = success;
    result.avg_turns_to_success = 1.5;
    result
}

#[test]
fn test_empty_frontier() {
    let results: Vec<EvalResult> = vec![];
    let frontier = compute_pareto_frontier(&results);
    assert!(frontier.is_empty());
}

#[test]
fn test_single_point() {
    let results = vec![make_result("model1", 1000, 0.9)];
    let frontier = compute_pareto_frontier(&results);

    assert_eq!(frontier.len(), 1);
    assert!(frontier[0].is_pareto_optimal);
}

#[test]
fn test_dominated_point() {
    // model1: smaller AND more accurate -> dominates model2
    let results = vec![
        make_result("model1", 1000, 0.95),
        make_result("model2", 2000, 0.90),
    ];
    let frontier = compute_pareto_frontier(&results);

    assert_eq!(frontier.len(), 1);
    assert_eq!(frontier[0].model_id, "model1");
}

#[test]
fn test_incomparable_points() {
    // model1: smaller but less accurate
    // model2: larger but more accurate
    // Neither dominates the other
    let results = vec![
        make_result("model1", 1000, 0.80),
        make_result("model2", 2000, 0.95),
    ];
    let frontier = compute_pareto_frontier(&results);

    assert_eq!(frontier.len(), 2);
    assert!(frontier.iter().all(|p| p.is_pareto_optimal));
}

#[test]
fn test_three_point_frontier() {
    // Classic Pareto curve
    let results = vec![
        make_result("small", 1000, 0.70),
        make_result("medium", 2000, 0.85),
        make_result("large", 4000, 0.95),
    ];
    let frontier = compute_pareto_frontier(&results);

    assert_eq!(frontier.len(), 3);
    // Should be sorted by size
    assert_eq!(frontier[0].model_id, "small");
    assert_eq!(frontier[1].model_id, "medium");
    assert_eq!(frontier[2].model_id, "large");
}

#[test]
fn test_dominated_in_middle() {
    // model2 is dominated by model1 (same success, larger)
    let results = vec![
        make_result("model1", 1000, 0.90),
        make_result("model2", 2000, 0.90), // Dominated
        make_result("model3", 3000, 0.95),
    ];
    let frontier = compute_pareto_frontier(&results);

    assert_eq!(frontier.len(), 2);
    assert!(frontier.iter().any(|p| p.model_id == "model1"));
    assert!(frontier.iter().any(|p| p.model_id == "model3"));
    assert!(!frontier.iter().any(|p| p.model_id == "model2"));
}

#[test]
fn test_dominates_function() {
    let a = ParetoPoint {
        model_id: "a".to_string(),
        size_bytes: 1000,
        success_rate: 0.9,
        avg_turns: 1.0,
        is_pareto_optimal: false,
    };
    let b = ParetoPoint {
        model_id: "b".to_string(),
        size_bytes: 2000,
        success_rate: 0.8,
        avg_turns: 1.5,
        is_pareto_optimal: false,
    };

    assert!(dominates(&a, &b)); // a dominates b (smaller AND more accurate)
    assert!(!dominates(&b, &a)); // b does not dominate a
}

#[test]
fn test_find_knee_point() {
    let frontier = vec![
        ParetoPoint {
            model_id: "p1".to_string(),
            size_bytes: 1000,
            success_rate: 0.60,
            avg_turns: 2.0,
            is_pareto_optimal: true,
        },
        ParetoPoint {
            model_id: "p2".to_string(),
            size_bytes: 2000,
            success_rate: 0.85,
            avg_turns: 1.5,
            is_pareto_optimal: true,
        },
        ParetoPoint {
            model_id: "p3".to_string(),
            size_bytes: 5000,
            success_rate: 0.90,
            avg_turns: 1.2,
            is_pareto_optimal: true,
        },
        ParetoPoint {
            model_id: "p4".to_string(),
            size_bytes: 10000,
            success_rate: 0.92,
            avg_turns: 1.1,
            is_pareto_optimal: true,
        },
    ];

    let knee = find_knee_point(&frontier);
    assert!(knee.is_some());
    // The knee should be around p2 or p3 (biggest improvement per size)
}

#[test]
fn test_find_knee_point_small() {
    let frontier = vec![ParetoPoint {
        model_id: "only".to_string(),
        size_bytes: 1000,
        success_rate: 0.9,
        avg_turns: 1.0,
        is_pareto_optimal: true,
    }];

    let knee = find_knee_point(&frontier);
    assert!(knee.is_some());
    assert_eq!(knee.unwrap().model_id, "only");
}

#[test]
fn test_generate_recommendations() {
    let frontier = vec![
        ParetoPoint {
            model_id: "small".to_string(),
            size_bytes: 1000,
            success_rate: 0.70,
            avg_turns: 2.0,
            is_pareto_optimal: true,
        },
        ParetoPoint {
            model_id: "medium".to_string(),
            size_bytes: 2000,
            success_rate: 0.85,
            avg_turns: 1.5,
            is_pareto_optimal: true,
        },
        ParetoPoint {
            model_id: "large".to_string(),
            size_bytes: 4000,
            success_rate: 0.95,
            avg_turns: 1.0,
            is_pareto_optimal: true,
        },
    ];

    let thresholds = vec![0.80, 0.90];
    let recs = generate_recommendations(&frontier, &thresholds);

    assert_eq!(recs.smallest.as_ref().unwrap().model_id, "small");
    assert_eq!(recs.most_accurate.as_ref().unwrap().model_id, "large");

    // At 80% threshold, medium is smallest qualifying
    assert_eq!(recs.by_threshold[0].1.as_ref().unwrap().model_id, "medium");
    // At 90% threshold, large is smallest qualifying
    assert_eq!(recs.by_threshold[1].1.as_ref().unwrap().model_id, "large");
}

#[test]
fn test_hypervolume() {
    let frontier = vec![
        ParetoPoint {
            model_id: "p1".to_string(),
            size_bytes: 1000,
            success_rate: 0.7,
            avg_turns: 1.0,
            is_pareto_optimal: true,
        },
        ParetoPoint {
            model_id: "p2".to_string(),
            size_bytes: 2000,
            success_rate: 0.9,
            avg_turns: 1.0,
            is_pareto_optimal: true,
        },
    ];

    let reference = (5000, 0.5);
    let hv = compute_hypervolume(&frontier, reference);

    // Should be positive (frontier dominates some area)
    assert!(hv >= 0.0);
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_hypervolume_empty() {
    let frontier: Vec<ParetoPoint> = vec![];
    let hv = compute_hypervolume(&frontier, (5000, 0.5));
    assert!((hv - 0.0).abs() < 0.001);
}

#[test]
fn test_hypervolume_better_than_reference() {
    // Point with success rate higher than reference - should skip
    let frontier = vec![ParetoPoint {
        model_id: "p1".to_string(),
        size_bytes: 1000,
        success_rate: 0.9, // Higher than ref_success of 0.5
        avg_turns: 1.0,
        is_pareto_optimal: true,
    }];

    let reference = (5000, 0.5);
    let hv = compute_hypervolume(&frontier, reference);
    assert!(hv >= 0.0); // Should still work
}

#[test]
fn test_find_knee_point_two_points() {
    let frontier = vec![
        ParetoPoint {
            model_id: "p1".to_string(),
            size_bytes: 1000,
            success_rate: 0.7,
            avg_turns: 1.0,
            is_pareto_optimal: true,
        },
        ParetoPoint {
            model_id: "p2".to_string(),
            size_bytes: 2000,
            success_rate: 0.9,
            avg_turns: 1.0,
            is_pareto_optimal: true,
        },
    ];

    let knee = find_knee_point(&frontier);
    assert!(knee.is_some());
    // With 2 points, returns first
    assert_eq!(knee.unwrap().model_id, "p1");
}

#[test]
fn test_find_knee_point_empty() {
    let frontier: Vec<ParetoPoint> = vec![];
    let knee = find_knee_point(&frontier);
    assert!(knee.is_none());
}

#[test]
fn test_find_knee_uniform_size() {
    // All same size - size_range is 0
    let frontier = vec![
        ParetoPoint {
            model_id: "p1".to_string(),
            size_bytes: 1000,
            success_rate: 0.7,
            avg_turns: 1.0,
            is_pareto_optimal: true,
        },
        ParetoPoint {
            model_id: "p2".to_string(),
            size_bytes: 1000,
            success_rate: 0.8,
            avg_turns: 1.0,
            is_pareto_optimal: true,
        },
        ParetoPoint {
            model_id: "p3".to_string(),
            size_bytes: 1000,
            success_rate: 0.9,
            avg_turns: 1.0,
            is_pareto_optimal: true,
        },
    ];

    let knee = find_knee_point(&frontier);
    assert!(knee.is_some());
    // Should return first due to tiny range
    assert_eq!(knee.unwrap().model_id, "p1");
}

#[test]
fn test_generate_recommendations_empty() {
    let frontier: Vec<ParetoPoint> = vec![];
    let thresholds = vec![0.8, 0.9];
    let recs = generate_recommendations(&frontier, &thresholds);

    assert!(recs.smallest.is_none());
    assert!(recs.most_accurate.is_none());
    assert!(recs.best_tradeoff.is_none());
}

#[test]
fn test_dominates_edge_cases() {
    let a = ParetoPoint {
        model_id: "a".to_string(),
        size_bytes: 1000,
        success_rate: 0.9,
        avg_turns: 1.0,
        is_pareto_optimal: false,
    };
    let same = ParetoPoint {
        model_id: "same".to_string(),
        size_bytes: 1000,
        success_rate: 0.9,
        avg_turns: 1.0,
        is_pareto_optimal: false,
    };

    // Equal points don't dominate each other
    assert!(!dominates(&a, &same));
    assert!(!dominates(&same, &a));
}

#[test]
fn test_model_recommendations_debug() {
    let recs = ModelRecommendations {
        smallest: None,
        most_accurate: None,
        best_tradeoff: None,
        by_threshold: vec![],
    };
    let debug_str = format!("{:?}", recs);
    assert!(debug_str.contains("ModelRecommendations"));
}

#[test]
fn test_model_recommendations_clone() {
    let recs = ModelRecommendations {
        smallest: Some(ParetoPoint {
            model_id: "test".to_string(),
            size_bytes: 1000,
            success_rate: 0.9,
            avg_turns: 1.0,
            is_pareto_optimal: true,
        }),
        most_accurate: None,
        best_tradeoff: None,
        by_threshold: vec![(0.8, None)],
    };
    let cloned = recs.clone();
    assert_eq!(cloned.smallest.as_ref().unwrap().model_id, "test");
}
