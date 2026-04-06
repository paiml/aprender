//! PMAT Code Quality Metrics (ENT-005)
//!
//! Provides structured code quality metrics following PMAT methodology:
//! - Line coverage from cargo-llvm-cov
//! - Mutation score from cargo-mutants
//! - Clippy warning counts
//! - PMAT grade (A/B/C/D/F)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors for quality metrics parsing
#[derive(Debug, Error)]
pub enum QualityError {
    #[error("Failed to parse coverage output: {0}")]
    CoverageParseError(String),

    #[error("Failed to parse mutation output: {0}")]
    MutationParseError(String),

    #[error("Invalid metric value: {0}")]
    InvalidMetric(String),
}

/// Result type for quality operations
pub type Result<T> = std::result::Result<T, QualityError>;

/// PMAT quality grade
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PmatGrade {
    /// Excellent: coverage >= 95%, mutation >= 85%
    A,
    /// Good: coverage >= 85%, mutation >= 75%
    B,
    /// Acceptable: coverage >= 75%, mutation >= 65%
    C,
    /// Poor: coverage >= 60%, mutation >= 50%
    D,
    /// Failing: below D thresholds
    F,
}

impl PmatGrade {
    /// Calculate grade from coverage and mutation scores
    pub fn from_scores(coverage: f64, mutation: f64) -> Self {
        if coverage >= 95.0 && mutation >= 85.0 {
            Self::A
        } else if coverage >= 85.0 && mutation >= 75.0 {
            Self::B
        } else if coverage >= 75.0 && mutation >= 65.0 {
            Self::C
        } else if coverage >= 60.0 && mutation >= 50.0 {
            Self::D
        } else {
            Self::F
        }
    }

    /// Returns true if this grade meets or exceeds the target
    pub fn meets_target(&self, target: Self) -> bool {
        self.as_numeric() >= target.as_numeric()
    }

    /// Convert grade to numeric value for comparison (A=4, B=3, C=2, D=1, F=0)
    fn as_numeric(&self) -> u8 {
        match self {
            Self::A => 4,
            Self::B => 3,
            Self::C => 2,
            Self::D => 1,
            Self::F => 0,
        }
    }
}

impl std::fmt::Display for PmatGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
            Self::C => write!(f, "C"),
            Self::D => write!(f, "D"),
            Self::F => write!(f, "F"),
        }
    }
}

/// Code quality metrics from PMAT analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeQualityMetrics {
    /// Line coverage percentage (0.0 - 100.0)
    pub coverage_percent: f64,

    /// Mutation testing score percentage (0.0 - 100.0)
    pub mutation_score: f64,

    /// Number of clippy warnings
    pub clippy_warnings: u32,

    /// Computed PMAT grade
    pub pmat_grade: PmatGrade,

    /// Timestamp when metrics were collected
    pub timestamp: DateTime<Utc>,
}

impl CodeQualityMetrics {
    /// Create new metrics with current timestamp
    pub fn new(coverage_percent: f64, mutation_score: f64, clippy_warnings: u32) -> Self {
        let pmat_grade = PmatGrade::from_scores(coverage_percent, mutation_score);
        Self {
            coverage_percent,
            mutation_score,
            clippy_warnings,
            pmat_grade,
            timestamp: Utc::now(),
        }
    }

    /// Create metrics with explicit timestamp
    pub fn with_timestamp(
        coverage_percent: f64,
        mutation_score: f64,
        clippy_warnings: u32,
        timestamp: DateTime<Utc>,
    ) -> Self {
        let pmat_grade = PmatGrade::from_scores(coverage_percent, mutation_score);
        Self { coverage_percent, mutation_score, clippy_warnings, pmat_grade, timestamp }
    }

    /// Parse metrics from cargo-llvm-cov and cargo-mutants output
    ///
    /// # Arguments
    ///
    /// * `coverage` - JSON output from `cargo llvm-cov --json`
    /// * `mutants` - JSON output from `cargo mutants --json`
    ///
    /// # Example
    ///
    /// ```ignore
    /// let coverage_json = r#"{"data":[{"totals":{"lines":{"percent":85.5}}}]}"#;
    /// let mutants_json = r#"{"total_mutants":100,"caught":75}"#;
    /// let metrics = CodeQualityMetrics::from_cargo_output(coverage_json, mutants_json, 0)?;
    /// ```
    pub fn from_cargo_output(coverage: &str, mutants: &str, clippy_warnings: u32) -> Result<Self> {
        let coverage_percent = Self::parse_coverage(coverage)?;
        let mutation_score = Self::parse_mutants(mutants)?;

        Ok(Self::new(coverage_percent, mutation_score, clippy_warnings))
    }

    /// Parse coverage percentage from cargo-llvm-cov JSON output
    fn parse_coverage(json: &str) -> Result<f64> {
        // cargo llvm-cov --json format:
        // {"data":[{"totals":{"lines":{"percent":85.5}}}]}
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| QualityError::CoverageParseError(e.to_string()))?;

        value
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|d| d.get("totals"))
            .and_then(|t| t.get("lines"))
            .and_then(|l| l.get("percent"))
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                QualityError::CoverageParseError("Missing lines.percent field".to_string())
            })
    }

    /// Parse mutation score from cargo-mutants JSON output
    fn parse_mutants(json: &str) -> Result<f64> {
        // cargo mutants --json format:
        // {"total_mutants":100,"caught":75,"missed":20,"timeout":5}
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| QualityError::MutationParseError(e.to_string()))?;

        let total =
            value.get("total_mutants").and_then(serde_json::Value::as_u64).ok_or_else(|| {
                QualityError::MutationParseError("Missing total_mutants field".to_string())
            })?;

        if total == 0 {
            return Ok(0.0);
        }

        let caught = value
            .get("caught")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| QualityError::MutationParseError("Missing caught field".to_string()))?;

        Ok((caught as f64 / total as f64) * 100.0)
    }

    /// Check if metrics meet minimum thresholds
    ///
    /// # Arguments
    ///
    /// * `min_coverage` - Minimum coverage percentage (0.0 - 100.0)
    /// * `min_mutation` - Minimum mutation score percentage (0.0 - 100.0)
    ///
    /// # Example
    ///
    /// ```
    /// use entrenar::quality::CodeQualityMetrics;
    ///
    /// let metrics = CodeQualityMetrics::new(90.0, 80.0, 0);
    /// assert!(metrics.meets_threshold(85.0, 75.0));
    /// assert!(!metrics.meets_threshold(95.0, 85.0));
    /// ```
    pub fn meets_threshold(&self, min_coverage: f64, min_mutation: f64) -> bool {
        self.coverage_percent >= min_coverage && self.mutation_score >= min_mutation
    }

    /// Check if metrics meet the target PMAT grade
    pub fn meets_grade(&self, target: PmatGrade) -> bool {
        self.pmat_grade.meets_target(target)
    }

    /// Returns true if there are no clippy warnings
    pub fn is_clippy_clean(&self) -> bool {
        self.clippy_warnings == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pmat_grade_from_scores_a() {
        assert_eq!(PmatGrade::from_scores(95.0, 85.0), PmatGrade::A);
        assert_eq!(PmatGrade::from_scores(100.0, 100.0), PmatGrade::A);
        assert_eq!(PmatGrade::from_scores(99.0, 90.0), PmatGrade::A);
    }

    #[test]
    fn test_pmat_grade_from_scores_b() {
        assert_eq!(PmatGrade::from_scores(85.0, 75.0), PmatGrade::B);
        assert_eq!(PmatGrade::from_scores(94.9, 84.9), PmatGrade::B);
        assert_eq!(PmatGrade::from_scores(90.0, 80.0), PmatGrade::B);
    }

    #[test]
    fn test_pmat_grade_from_scores_c() {
        assert_eq!(PmatGrade::from_scores(75.0, 65.0), PmatGrade::C);
        assert_eq!(PmatGrade::from_scores(84.9, 74.9), PmatGrade::C);
    }

    #[test]
    fn test_pmat_grade_from_scores_d() {
        assert_eq!(PmatGrade::from_scores(60.0, 50.0), PmatGrade::D);
        assert_eq!(PmatGrade::from_scores(74.9, 64.9), PmatGrade::D);
    }

    #[test]
    fn test_pmat_grade_from_scores_f() {
        assert_eq!(PmatGrade::from_scores(59.9, 49.9), PmatGrade::F);
        assert_eq!(PmatGrade::from_scores(0.0, 0.0), PmatGrade::F);
        assert_eq!(PmatGrade::from_scores(90.0, 40.0), PmatGrade::F);
    }

    #[test]
    fn test_pmat_grade_meets_target() {
        assert!(PmatGrade::A.meets_target(PmatGrade::A));
        assert!(PmatGrade::A.meets_target(PmatGrade::B));
        assert!(PmatGrade::B.meets_target(PmatGrade::C));
        assert!(!PmatGrade::B.meets_target(PmatGrade::A));
        assert!(!PmatGrade::F.meets_target(PmatGrade::D));
    }

    #[test]
    fn test_pmat_grade_display() {
        assert_eq!(format!("{}", PmatGrade::A), "A");
        assert_eq!(format!("{}", PmatGrade::F), "F");
    }

    #[test]
    fn test_code_quality_metrics_new() {
        let metrics = CodeQualityMetrics::new(90.0, 80.0, 5);

        assert!((metrics.coverage_percent - 90.0).abs() < f64::EPSILON);
        assert!((metrics.mutation_score - 80.0).abs() < f64::EPSILON);
        assert_eq!(metrics.clippy_warnings, 5);
        assert_eq!(metrics.pmat_grade, PmatGrade::B);
    }

    #[test]
    fn test_code_quality_metrics_meets_threshold() {
        let metrics = CodeQualityMetrics::new(90.0, 80.0, 0);

        // Should pass these thresholds
        assert!(metrics.meets_threshold(85.0, 75.0));
        assert!(metrics.meets_threshold(90.0, 80.0));

        // Should fail these thresholds
        assert!(!metrics.meets_threshold(95.0, 85.0));
        assert!(!metrics.meets_threshold(90.0, 85.0));
        assert!(!metrics.meets_threshold(95.0, 80.0));
    }

    #[test]
    fn test_code_quality_metrics_meets_threshold_edge_cases() {
        // Exactly at threshold
        let metrics = CodeQualityMetrics::new(85.0, 75.0, 0);
        assert!(metrics.meets_threshold(85.0, 75.0));

        // Just below threshold
        let metrics = CodeQualityMetrics::new(84.99, 74.99, 0);
        assert!(!metrics.meets_threshold(85.0, 75.0));
    }

    #[test]
    fn test_code_quality_metrics_meets_grade() {
        let metrics = CodeQualityMetrics::new(90.0, 80.0, 0);

        assert!(metrics.meets_grade(PmatGrade::B));
        assert!(metrics.meets_grade(PmatGrade::C));
        assert!(!metrics.meets_grade(PmatGrade::A));
    }

    #[test]
    fn test_code_quality_metrics_is_clippy_clean() {
        let clean = CodeQualityMetrics::new(90.0, 80.0, 0);
        let warnings = CodeQualityMetrics::new(90.0, 80.0, 5);

        assert!(clean.is_clippy_clean());
        assert!(!warnings.is_clippy_clean());
    }

    #[test]
    fn test_from_cargo_output_valid() {
        let coverage_json = r#"{"data":[{"totals":{"lines":{"percent":85.5}}}]}"#;
        let mutants_json = r#"{"total_mutants":100,"caught":75,"missed":20,"timeout":5}"#;

        let metrics = CodeQualityMetrics::from_cargo_output(coverage_json, mutants_json, 0)
            .expect("operation should succeed");

        assert!((metrics.coverage_percent - 85.5).abs() < f64::EPSILON);
        assert!((metrics.mutation_score - 75.0).abs() < f64::EPSILON);
        // 85.5% coverage >= 85% and 75% mutation >= 75% = Grade B
        assert_eq!(metrics.pmat_grade, PmatGrade::B);
    }

    #[test]
    fn test_from_cargo_output_perfect_scores() {
        let coverage_json = r#"{"data":[{"totals":{"lines":{"percent":100.0}}}]}"#;
        let mutants_json = r#"{"total_mutants":50,"caught":50,"missed":0,"timeout":0}"#;

        let metrics = CodeQualityMetrics::from_cargo_output(coverage_json, mutants_json, 0)
            .expect("operation should succeed");

        assert!((metrics.coverage_percent - 100.0).abs() < f64::EPSILON);
        assert!((metrics.mutation_score - 100.0).abs() < f64::EPSILON);
        assert_eq!(metrics.pmat_grade, PmatGrade::A);
    }

    #[test]
    fn test_from_cargo_output_zero_mutants() {
        let coverage_json = r#"{"data":[{"totals":{"lines":{"percent":90.0}}}]}"#;
        let mutants_json = r#"{"total_mutants":0,"caught":0,"missed":0,"timeout":0}"#;

        let metrics = CodeQualityMetrics::from_cargo_output(coverage_json, mutants_json, 0)
            .expect("operation should succeed");

        assert!((metrics.mutation_score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_cargo_output_invalid_coverage() {
        let coverage_json = r#"{"invalid": "json"}"#;
        let mutants_json = r#"{"total_mutants":100,"caught":75}"#;

        let result = CodeQualityMetrics::from_cargo_output(coverage_json, mutants_json, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_cargo_output_invalid_mutants() {
        let coverage_json = r#"{"data":[{"totals":{"lines":{"percent":85.5}}}]}"#;
        let mutants_json = r#"{"invalid": "json"}"#;

        let result = CodeQualityMetrics::from_cargo_output(coverage_json, mutants_json, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_code_quality_metrics_serialization() {
        let metrics = CodeQualityMetrics::new(90.0, 80.0, 0);
        let json = serde_json::to_string(&metrics).expect("JSON serialization should succeed");
        let parsed: CodeQualityMetrics =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert!((parsed.coverage_percent - metrics.coverage_percent).abs() < f64::EPSILON);
        assert_eq!(parsed.pmat_grade, metrics.pmat_grade);
    }
}
