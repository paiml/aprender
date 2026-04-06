//! Output Generation
//!
//! Generates reports and test files from analysis results.

mod fkr_generator;
mod html_report;

pub use fkr_generator::generate_fkr_tests;
pub use html_report::generate_html_report;

use crate::bugs::BugRegistry;
use crate::falsification::FalsificationReport;

/// Analysis result combining all analyses
#[derive(Debug)]
pub struct AnalysisResult {
    /// Module name
    pub module_name: String,
    /// Falsification score (0-100)
    pub falsification_score: f64,
    /// Confidence (0-1)
    pub confidence: f64,
    /// Falsification report
    pub falsification_report: FalsificationReport,
    /// Detected bugs
    pub bugs: BugRegistry,
}

impl AnalysisResult {
    /// Create from analysis components
    pub fn new(
        module_name: &str,
        falsification_report: FalsificationReport,
        bugs: BugRegistry,
    ) -> Self {
        Self {
            module_name: module_name.to_string(),
            falsification_score: falsification_report.score,
            confidence: falsification_report.confidence,
            falsification_report,
            bugs,
        }
    }
}
