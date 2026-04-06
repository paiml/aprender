//! Stack Quality Matrix
//!
//! Implements the quality enforcement system for PAIML stack components.
//! Ensures all components meet A+ quality standards before release.
//!
//! ## Quality Dimensions
//!
//! - **Rust Project Score** (pmat rust-project-score): 105-114 for A+
//! - **Repository Score** (pmat repo-score): 95-110 for A+
//! - **README Score**: 18-20 for A+
//! - **Hero Image**: Present and valid

mod component;
mod report;
mod summary;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use component::ComponentQuality;
pub use report::StackQualityReport;
pub use summary::QualitySummary;
pub use types::{IssueSeverity, QualityGrade, QualityIssue, Score, StackLayer};

// Re-export hero image types for backward compatibility
pub use super::hero_image::{HeroImageResult, ImageFormat};

// Re-export QualityChecker from dedicated module
pub use super::quality_checker::QualityChecker;

// Re-export format functions from quality_format module
pub use super::quality_format::{format_report_json, format_report_text};
