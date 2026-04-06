//! Quality summary statistics
//!
//! Contains `QualitySummary` - summary statistics for stack quality.

use serde::{Deserialize, Serialize};

use super::component::ComponentQuality;
use super::types::QualityGrade;

// ============================================================================
// Quality Summary
// ============================================================================

/// Summary statistics for stack quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySummary {
    /// Total components checked
    pub total_components: usize,
    /// Components with A+ grade
    pub a_plus_count: usize,
    /// Components with A grade
    pub a_count: usize,
    /// Components with A- grade
    pub a_minus_count: usize,
    /// Components below A- threshold
    pub below_threshold_count: usize,
    /// Components missing hero image
    pub missing_hero_count: usize,
    /// Average rust score
    pub avg_rust_score: f64,
    /// Average repo score
    pub avg_repo_score: f64,
    /// Average readme score
    pub avg_readme_score: f64,
}

impl QualitySummary {
    /// Calculate summary from component list
    pub fn from_components(components: &[ComponentQuality]) -> Self {
        let total = components.len();
        if total == 0 {
            return Self {
                total_components: 0,
                a_plus_count: 0,
                a_count: 0,
                a_minus_count: 0,
                below_threshold_count: 0,
                missing_hero_count: 0,
                avg_rust_score: 0.0,
                avg_repo_score: 0.0,
                avg_readme_score: 0.0,
            };
        }

        let a_plus_count = components.iter().filter(|c| c.grade == QualityGrade::APlus).count();
        let a_count = components.iter().filter(|c| c.grade == QualityGrade::A).count();
        let a_minus_count = components.iter().filter(|c| c.grade == QualityGrade::AMinus).count();
        let below_threshold_count =
            components.iter().filter(|c| !c.grade.is_release_ready()).count();
        let missing_hero_count = components.iter().filter(|c| !c.hero_image.valid).count();

        let avg_rust_score =
            components.iter().map(|c| c.rust_score.value as f64).sum::<f64>() / total as f64;
        let avg_repo_score =
            components.iter().map(|c| c.repo_score.value as f64).sum::<f64>() / total as f64;
        let avg_readme_score =
            components.iter().map(|c| c.readme_score.value as f64).sum::<f64>() / total as f64;

        Self {
            total_components: total,
            a_plus_count,
            a_count,
            a_minus_count,
            below_threshold_count,
            missing_hero_count,
            avg_rust_score,
            avg_repo_score,
            avg_readme_score,
        }
    }
}
