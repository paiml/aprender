//! Stack quality report
//!
//! Contains `StackQualityReport` - complete quality report for the stack.

use serde::{Deserialize, Serialize};

use super::component::ComponentQuality;
use super::summary::QualitySummary;
use super::types::QualityGrade;

// ============================================================================
// Stack Quality Report
// ============================================================================

/// Complete quality report for the stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackQualityReport {
    /// Timestamp of the report
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Individual component assessments
    pub components: Vec<ComponentQuality>,
    /// Summary statistics
    pub summary: QualitySummary,
    /// Overall Stack Quality Index
    pub stack_quality_index: f64,
    /// Overall grade
    pub overall_grade: QualityGrade,
    /// Whether all components are release-ready
    pub release_ready: bool,
    /// Components that block release
    pub blocked_components: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl StackQualityReport {
    /// Create report from component list
    pub fn from_components(components: Vec<ComponentQuality>) -> Self {
        let summary = QualitySummary::from_components(&components);

        // Calculate overall SQI as average of component SQIs
        let sqi = if components.is_empty() {
            0.0
        } else {
            components.iter().map(|c| c.sqi).sum::<f64>() / components.len() as f64
        };

        let grade = QualityGrade::from_sqi(sqi);
        let blocked: Vec<String> =
            components.iter().filter(|c| !c.release_ready).map(|c| c.name.clone()).collect();

        let release_ready = blocked.is_empty();
        let recommendations = Self::generate_recommendations(&components, &summary);

        Self {
            timestamp: chrono::Utc::now(),
            components,
            summary,
            stack_quality_index: sqi,
            overall_grade: grade,
            release_ready,
            blocked_components: blocked,
            recommendations,
        }
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(
        components: &[ComponentQuality],
        summary: &QualitySummary,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        if summary.missing_hero_count > 0 {
            recs.push(format!("Add hero images to {} components", summary.missing_hero_count));
        }

        if summary.below_threshold_count > 0 {
            recs.push(format!(
                "Improve quality scores for {} components below A- threshold",
                summary.below_threshold_count
            ));
        }

        // Specific recommendations for blocked components
        for comp in components.iter().filter(|c| !c.release_ready) {
            if comp.rust_score.value < 85 {
                recs.push(format!("{}: Improve test coverage and documentation", comp.name));
            }
            if !comp.hero_image.valid {
                recs.push(format!("{}: Add hero.png to docs/ directory", comp.name));
            }
        }

        recs
    }

    /// Check if all components meet strict A+ requirement
    pub fn is_all_a_plus(&self) -> bool {
        self.components.iter().all(|c| c.grade.is_a_plus())
    }
}
