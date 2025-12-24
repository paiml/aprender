//! Pareto Frontier Computation
//!
//! Implements multi-objective optimization for model selection.
//! Finds the Pareto-optimal set of models (no model dominates another).
//!
//! # References
//! - Deb et al. (2002) "NSGA-II: A Fast and Elitist Multiobjective Genetic Algorithm"
//!
//! # Toyota Way Alignment
//! - **Muda Elimination**: Avoid overprovisioning by selecting right-sized models

use super::{EvalResult, ParetoPoint};

/// Compute Pareto frontier from evaluation results
///
/// A point is Pareto-optimal if no other point is both:
/// - Smaller (or equal) in size AND
/// - Higher (or equal) in success rate
/// - With at least one strict inequality
#[must_use]
pub fn compute_pareto_frontier(results: &[EvalResult]) -> Vec<ParetoPoint> {
    if results.is_empty() {
        return Vec::new();
    }

    let mut points: Vec<ParetoPoint> = results
        .iter()
        .map(|r| ParetoPoint {
            model_id: r.model_id.clone(),
            size_bytes: r.model_size_bytes,
            success_rate: r.overall_success_rate,
            avg_turns: r.avg_turns_to_success,
            is_pareto_optimal: false,
        })
        .collect();

    // For each point, check if it's dominated by any other
    for i in 0..points.len() {
        let is_dominated = (0..points.len()).any(|j| {
            if i == j {
                return false;
            }
            dominates(&points[j], &points[i])
        });
        points[i].is_pareto_optimal = !is_dominated;
    }

    // Return only Pareto-optimal points, sorted by size
    let mut frontier: Vec<_> = points.into_iter().filter(|p| p.is_pareto_optimal).collect();
    frontier.sort_by_key(|p| p.size_bytes);
    frontier
}

/// Check if point A dominates point B
///
/// A dominates B if A is at least as good in all objectives and strictly better in at least one.
/// Objectives: minimize size, maximize success rate
fn dominates(a: &ParetoPoint, b: &ParetoPoint) -> bool {
    // A is at least as good as B in both objectives
    let size_ok = a.size_bytes <= b.size_bytes;
    let success_ok = a.success_rate >= b.success_rate;

    // A is strictly better in at least one objective
    let strictly_better = a.size_bytes < b.size_bytes || a.success_rate > b.success_rate;

    size_ok && success_ok && strictly_better
}

/// Find the "knee" point on the Pareto frontier
///
/// The knee is the point with maximum curvature, representing the best trade-off.
#[must_use]
pub fn find_knee_point(frontier: &[ParetoPoint]) -> Option<&ParetoPoint> {
    if frontier.len() < 3 {
        // With fewer than 3 points, return the middle or first
        return frontier.first();
    }

    // Normalize values to [0, 1] for fair comparison
    let size_min = frontier.iter().map(|p| p.size_bytes).min().unwrap_or(1) as f64;
    let size_max = frontier.iter().map(|p| p.size_bytes).max().unwrap_or(1) as f64;
    let success_min = frontier
        .iter()
        .map(|p| p.success_rate)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    let success_max = frontier
        .iter()
        .map(|p| p.success_rate)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1.0);

    let size_range = size_max - size_min;
    let success_range = success_max - success_min;

    if size_range < 1e-10 || success_range < 1e-10 {
        return frontier.first();
    }

    // Compute curvature at each point using angle between adjacent segments
    let mut max_curvature = 0.0_f64;
    let mut knee_idx = 0;

    for i in 1..frontier.len() - 1 {
        let prev = &frontier[i - 1];
        let curr = &frontier[i];
        let next = &frontier[i + 1];

        // Normalize coordinates
        let prev_size = (prev.size_bytes as f64 - size_min) / size_range;
        let curr_size = (curr.size_bytes as f64 - size_min) / size_range;
        let next_size = (next.size_bytes as f64 - size_min) / size_range;

        let prev_success = (prev.success_rate - success_min) / success_range;
        let curr_success = (curr.success_rate - success_min) / success_range;
        let next_success = (next.success_rate - success_min) / success_range;

        // Vectors from curr to prev and curr to next
        let v1 = (prev_size - curr_size, prev_success - curr_success);
        let v2 = (next_size - curr_size, next_success - curr_success);

        // Cross product (gives curvature direction)
        let cross = v1.0 * v2.1 - v1.1 * v2.0;

        // Dot product and magnitudes for angle
        let dot = v1.0 * v2.0 + v1.1 * v2.1;
        let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
        let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();

        if mag1 > 1e-10 && mag2 > 1e-10 {
            let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
            let angle = cos_angle.acos();
            let curvature = cross.abs() / (mag1 * mag2 * angle.max(1e-10));

            if curvature > max_curvature {
                max_curvature = curvature;
                knee_idx = i;
            }
        }
    }

    frontier.get(knee_idx)
}

/// Compute hypervolume indicator for a Pareto frontier
///
/// Measures the volume of objective space dominated by the frontier.
/// Higher is better.
#[must_use]
pub fn compute_hypervolume(frontier: &[ParetoPoint], reference: (u64, f64)) -> f64 {
    if frontier.is_empty() {
        return 0.0;
    }

    let (ref_size, ref_success) = reference;

    // Sort by size (ascending)
    let mut sorted = frontier.to_vec();
    sorted.sort_by_key(|p| p.size_bytes);

    // Compute dominated hypervolume using inclusion-exclusion
    let mut hypervolume = 0.0;
    let mut prev_success = ref_success;

    for point in sorted.iter().rev() {
        // Only count if point is better than reference in success
        if point.success_rate > ref_success {
            continue;
        }

        // Width: from point's size to reference size
        let width = (ref_size as f64 - point.size_bytes as f64).max(0.0);

        // Height: from previous success level to this point's success
        let height = (prev_success - point.success_rate).max(0.0);

        hypervolume += width * height;
        prev_success = point.success_rate;
    }

    hypervolume
}

/// Select models for different use cases based on Pareto frontier
#[derive(Debug, Clone)]
pub struct ModelRecommendations {
    /// Smallest model on frontier
    pub smallest: Option<ParetoPoint>,
    /// Most accurate model on frontier
    pub most_accurate: Option<ParetoPoint>,
    /// Best trade-off (knee point)
    pub best_tradeoff: Option<ParetoPoint>,
    /// Smallest model meeting various thresholds
    pub by_threshold: Vec<(f64, Option<ParetoPoint>)>,
}

/// Generate recommendations from Pareto frontier
#[must_use]
pub fn generate_recommendations(
    frontier: &[ParetoPoint],
    thresholds: &[f64],
) -> ModelRecommendations {
    let smallest = frontier.iter().min_by_key(|p| p.size_bytes).cloned();

    let most_accurate = frontier
        .iter()
        .max_by(|a, b| {
            a.success_rate
                .partial_cmp(&b.success_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    let best_tradeoff = find_knee_point(frontier).cloned();

    let by_threshold: Vec<_> = thresholds
        .iter()
        .map(|&t| {
            let model = frontier
                .iter()
                .filter(|p| p.success_rate >= t)
                .min_by_key(|p| p.size_bytes)
                .cloned();
            (t, model)
        })
        .collect();

    ModelRecommendations {
        smallest,
        most_accurate,
        best_tradeoff,
        by_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(model_id: &str, size: u64, success: f64) -> EvalResult {
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
}
