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
#[path = "pareto_tests.rs"]
mod tests;
