//! Component quality assessment
//!
//! Contains `ComponentQuality` - quality assessment for a single component.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::types::{QualityGrade, QualityIssue, Score, StackLayer};
use crate::stack::hero_image::HeroImageResult;

// ============================================================================
// Component Quality
// ============================================================================

/// Quality assessment for a single component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentQuality {
    /// Component name
    pub name: String,
    /// Layer in the stack
    pub layer: StackLayer,
    /// Repository path
    pub path: PathBuf,
    /// Rust project score
    pub rust_score: Score,
    /// Repository score
    pub repo_score: Score,
    /// README score
    pub readme_score: Score,
    /// Hero image result
    pub hero_image: HeroImageResult,
    /// Stack Quality Index (0-100)
    pub sqi: f64,
    /// Overall grade
    pub grade: QualityGrade,
    /// Whether release is allowed
    pub release_ready: bool,
    /// Issues found
    pub issues: Vec<QualityIssue>,
}

impl ComponentQuality {
    /// Create a new component quality assessment
    pub fn new(
        name: impl Into<String>,
        path: PathBuf,
        rust_score: Score,
        repo_score: Score,
        readme_score: Score,
        hero_image: HeroImageResult,
    ) -> Self {
        let name = name.into();
        let layer = StackLayer::from_component(&name);
        let sqi = Self::calculate_sqi(&rust_score, &repo_score, &readme_score, &hero_image);
        let grade = QualityGrade::from_sqi(sqi);
        let issues = Self::collect_issues(&rust_score, &repo_score, &readme_score, &hero_image);
        let release_ready = grade.is_release_ready() && hero_image.valid;

        Self {
            name,
            layer,
            path,
            rust_score,
            repo_score,
            readme_score,
            hero_image,
            sqi,
            grade,
            release_ready,
            issues,
        }
    }

    /// Calculate Stack Quality Index
    /// SQI = (0.40 x Rust) + (0.30 x Repo) + (0.20 x README) + (0.10 x Hero)
    pub fn calculate_sqi(
        rust: &Score,
        repo: &Score,
        readme: &Score,
        hero: &HeroImageResult,
    ) -> f64 {
        let rust_normalized = rust.normalized();
        let repo_normalized = repo.normalized();
        let readme_normalized = readme.normalized();
        let hero_normalized = if hero.valid { 100.0 } else { 0.0 };

        (0.40 * rust_normalized)
            + (0.30 * repo_normalized)
            + (0.20 * readme_normalized)
            + (0.10 * hero_normalized)
    }

    /// Collect issues based on scores
    fn collect_issues(
        rust: &Score,
        repo: &Score,
        readme: &Score,
        hero: &HeroImageResult,
    ) -> Vec<QualityIssue> {
        let mut issues = Vec::new();

        // Check Rust score (A- threshold = 85)
        if rust.value < 85 {
            issues.push(QualityIssue::score_below_threshold("rust_project", rust.value, 85));
        }

        // Check Repo score (A- threshold = 85)
        if repo.value < 85 {
            issues.push(QualityIssue::score_below_threshold("repo", repo.value, 85));
        }

        // Check README score (A- threshold = 14)
        if readme.value < 14 {
            issues.push(QualityIssue::score_below_threshold("readme", readme.value, 14));
        }

        // Check hero image
        if !hero.valid {
            issues.push(QualityIssue::missing_hero_image());
        }

        issues
    }
}
