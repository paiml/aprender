//! Data quality assessment for ML pipelines
//!
//! Detects data quality issues including missing values, outliers,
//! duplicates, and schema problems.
//!
//! # 100-Point Quality Scoring System (GH-6)
//!
//! Based on the Toyota Way principles of Jidoka (built-in quality) and
//! the Doctest Corpus QA Checklist for Publication.
//!
//! ## Severity Weights
//! - **Critical (2.0x)**: Blocks publication - data integrity failures
//! - **High (1.5x)**: Major issues requiring immediate attention
//! - **Medium (1.0x)**: Standard issues to address before publication
//! - **Low (0.5x)**: Minor issues, informational
//!
//! ## Letter Grades
//! - **A (95-100)**: Publish immediately
//! - **B (85-94)**: Publish with documented caveats
//! - **C (70-84)**: Remediation required before publication
//! - **D (50-69)**: Major rework needed
//! - **F (<50)**: Do not publish
//!
//! # Example
//!
//! ```ignore
//! use alimentar::quality::{QualityChecker, QualityScore};
//!
//! let checker = QualityChecker::new()
//!     .max_null_ratio(0.1)
//!     .max_duplicate_ratio(0.05);
//!
//! let report = checker.check(&dataset)?;
//! let score = QualityScore::from_report(&report);
//! println!("Grade: {} ({})", score.grade, score.score);
//! ```
//!
//! # References
//! - [1] Batini & Scannapieco (2016). Data and Information Quality.
//! - [6] Hynes et al. (2017). The Data Linter. NIPS Workshop on ML Systems.

// Statistical computation and internal methods
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::unused_self)]
#![allow(clippy::if_not_else)]

mod checks;
pub mod decontaminate;
mod profiles;
mod scoring;

#[cfg(test)]
mod tests;

// Re-export scoring types
// Re-export check types
pub use checks::{
    ColumnQuality, NumericStats, QualityChecker, QualityIssue, QualityReport, QualityThresholds,
    TextColumnStats,
};
// Re-export decontamination types
pub use decontaminate::{
    check_contamination, ngram_overlap, ContaminationResult, DecontaminationReport,
};
// Re-export profile types
pub use profiles::QualityProfile;
pub use scoring::{ChecklistItem, LetterGrade, QualityScore, Severity, SeverityStats};
