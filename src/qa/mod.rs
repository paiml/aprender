//! Model Quality Assurance Module (`aprender::qa`)
//!
//! Provides a 100-point adversarial QA checklist for production model validation.
//! Separates *model quality* (aprender) from *code quality* (certeza).
//!
//! # Toyota Way Alignment
//! - **Jidoka**: `Severity::Blocker` stops the deployment line
//! - **Poka-yoke**: Type-safe category enums prevent misconfiguration
//!
//! # Example
//! ```
//! use aprender::qa::{QaChecklist, QaCategory, Severity};
//!
//! let checklist = QaChecklist::default();
//! assert_eq!(QaChecklist::max_score(), 100);
//! ```

pub mod adversarial;
pub mod docs;
pub mod fairness;
pub mod robustness;
pub mod security;
pub mod velocity;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// 100-point QA checklist for model validation
#[derive(Debug, Clone)]
pub struct QaChecklist {
    /// Model under test
    pub model_path: PathBuf,
    /// Test dataset (required for most checks)
    pub test_data: Option<PathBuf>,
    /// Protected attributes for fairness testing
    pub protected_attrs: Vec<String>,
    /// Latency SLA for performance testing
    pub latency_sla: Duration,
    /// Memory budget for resource testing
    pub memory_budget: usize,
    /// Maximum turns before failure (for multi-turn evals)
    pub max_turns: u32,
}

impl Default for QaChecklist {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            test_data: None,
            protected_attrs: Vec::new(),
            latency_sla: Duration::from_millis(100),
            memory_budget: 512 * 1024 * 1024, // 512 MB
            max_turns: 5,
        }
    }
}

impl QaChecklist {
    /// Create a new QA checklist for a model
    #[must_use]
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            ..Default::default()
        }
    }

    /// Set test data path
    #[must_use]
    pub fn with_test_data(mut self, path: PathBuf) -> Self {
        self.test_data = Some(path);
        self
    }

    /// Set protected attributes for fairness testing
    #[must_use]
    pub fn with_protected_attrs(mut self, attrs: Vec<String>) -> Self {
        self.protected_attrs = attrs;
        self
    }

    /// Set latency SLA
    #[must_use]
    pub fn with_latency_sla(mut self, sla: Duration) -> Self {
        self.latency_sla = sla;
        self
    }

    /// Set memory budget
    #[must_use]
    pub fn with_memory_budget(mut self, budget: usize) -> Self {
        self.memory_budget = budget;
        self
    }

    /// Maximum possible score (always 100)
    #[must_use]
    pub const fn max_score() -> u8 {
        100
    }

    /// Get points allocation per category
    #[must_use]
    pub fn category_points() -> HashMap<QaCategory, u8> {
        let mut points = HashMap::new();
        points.insert(QaCategory::Robustness, 20);
        points.insert(QaCategory::EdgeCases, 15);
        points.insert(QaCategory::DistributionShift, 15);
        points.insert(QaCategory::Fairness, 15);
        points.insert(QaCategory::Privacy, 10);
        points.insert(QaCategory::Latency, 10);
        points.insert(QaCategory::Memory, 10);
        points.insert(QaCategory::Reproducibility, 5);
        points
    }
}

/// QA report with 100-point scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaReport {
    /// Model identifier
    pub model_id: String,
    /// Individual category scores
    pub categories: HashMap<QaCategory, CategoryScore>,
    /// Total score (0-100)
    pub total_score: u8,
    /// Pass/fail determination
    pub passed: bool,
    /// Blocking issues (must fix)
    pub blockers: Vec<QaIssue>,
    /// Warnings (should fix)
    pub warnings: Vec<QaIssue>,
}

impl QaReport {
    /// Create a new empty QA report
    #[must_use]
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            categories: HashMap::new(),
            total_score: 0,
            passed: false,
            blockers: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add a category score
    pub fn add_category(&mut self, category: QaCategory, score: CategoryScore) {
        self.categories.insert(category, score);
        self.recalculate_total();
    }

    /// Add a blocking issue
    pub fn add_blocker(&mut self, issue: QaIssue) {
        self.blockers.push(issue);
        self.passed = false;
    }

    /// Add a warning
    pub fn add_warning(&mut self, issue: QaIssue) {
        self.warnings.push(issue);
    }

    /// Recalculate total score from categories
    fn recalculate_total(&mut self) {
        let earned: u16 = self
            .categories
            .values()
            .map(|s| u16::from(s.points_earned))
            .sum();
        let possible: u16 = self
            .categories
            .values()
            .map(|s| u16::from(s.points_possible))
            .sum();

        self.total_score = if possible > 0 {
            ((earned * 100) / possible).min(100) as u8
        } else {
            0
        };

        // Pass if score >= 80 and no blockers
        self.passed = self.total_score >= 80 && self.blockers.is_empty();
    }

    /// Check if the model is production-ready
    #[must_use]
    pub fn is_production_ready(&self) -> bool {
        self.passed && self.total_score >= 90
    }
}

/// QA category enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QaCategory {
    /// Adversarial robustness (FGSM, PGD, noise)
    Robustness,
    /// Edge cases (NaN, Inf, empty, max-size)
    EdgeCases,
    /// Out-of-distribution detection
    DistributionShift,
    /// Fairness metrics (disparate impact, EOD)
    Fairness,
    /// Privacy (membership inference)
    Privacy,
    /// Latency (P50, P95, P99)
    Latency,
    /// Memory (peak, leaks)
    Memory,
    /// Reproducibility (determinism)
    Reproducibility,
}

impl QaCategory {
    /// Get all categories
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Robustness,
            Self::EdgeCases,
            Self::DistributionShift,
            Self::Fairness,
            Self::Privacy,
            Self::Latency,
            Self::Memory,
            Self::Reproducibility,
        ]
    }

    /// Get display name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Robustness => "Robustness",
            Self::EdgeCases => "Edge Cases",
            Self::DistributionShift => "Distribution Shift",
            Self::Fairness => "Fairness",
            Self::Privacy => "Privacy",
            Self::Latency => "Latency",
            Self::Memory => "Memory",
            Self::Reproducibility => "Reproducibility",
        }
    }
}

/// Score for a single category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryScore {
    /// Points earned
    pub points_earned: u8,
    /// Points possible
    pub points_possible: u8,
    /// Tests passed
    pub tests_passed: u32,
    /// Tests failed
    pub tests_failed: u32,
    /// Detailed test results
    pub details: Vec<TestResult>,
}

impl CategoryScore {
    /// Create a new category score
    #[must_use]
    pub fn new(points_possible: u8) -> Self {
        Self {
            points_earned: 0,
            points_possible,
            tests_passed: 0,
            tests_failed: 0,
            details: Vec::new(),
        }
    }

    /// Add a test result
    pub fn add_result(&mut self, result: TestResult) {
        if result.passed {
            self.tests_passed += 1;
        } else {
            self.tests_failed += 1;
        }
        self.details.push(result);
    }

    /// Calculate earned points based on pass rate
    pub fn finalize(&mut self) {
        let total = self.tests_passed + self.tests_failed;
        if total > 0 {
            let pass_rate = f64::from(self.tests_passed) / f64::from(total);
            self.points_earned = (f64::from(self.points_possible) * pass_rate).round() as u8;
        }
    }

    /// Get pass rate as percentage
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        let total = self.tests_passed + self.tests_failed;
        if total > 0 {
            f64::from(self.tests_passed) / f64::from(total) * 100.0
        } else {
            0.0
        }
    }
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Pass/fail
    pub passed: bool,
    /// Optional message
    pub message: Option<String>,
    /// Duration
    pub duration: Duration,
}

impl TestResult {
    /// Create a passing test result
    #[must_use]
    pub fn pass(name: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            passed: true,
            message: None,
            duration,
        }
    }

    /// Create a failing test result
    #[must_use]
    pub fn fail(name: impl Into<String>, message: impl Into<String>, duration: Duration) -> Self {
        Self {
            name: name.into(),
            passed: false,
            message: Some(message.into()),
            duration,
        }
    }
}

/// QA issue (blocker or warning)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaIssue {
    /// Category
    pub category: QaCategory,
    /// Severity level
    pub severity: Severity,
    /// Issue message
    pub message: String,
    /// Remediation suggestion
    pub remediation: String,
}

impl QaIssue {
    /// Create a new QA issue
    #[must_use]
    pub fn new(
        category: QaCategory,
        severity: Severity,
        message: impl Into<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self {
            category,
            severity,
            message: message.into(),
            remediation: remediation.into(),
        }
    }

    /// Create a blocker issue
    #[must_use]
    pub fn blocker(
        category: QaCategory,
        message: impl Into<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self::new(category, Severity::Blocker, message, remediation)
    }

    /// Create a warning issue
    #[must_use]
    pub fn warning(
        category: QaCategory,
        message: impl Into<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self::new(category, Severity::Warning, message, remediation)
    }
}

/// Issue severity (Toyota Way: Jidoka - stop the line)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Blocks production deployment (Andon cord)
    Blocker,
    /// Should fix before production
    Critical,
    /// Recommended improvement
    Warning,
    /// Informational only
    Info,
}

impl Severity {
    /// Check if this severity should block deployment
    #[must_use]
    pub const fn is_blocking(&self) -> bool {
        matches!(self, Self::Blocker)
    }

    /// Check if this severity requires human review
    #[must_use]
    pub const fn requires_review(&self) -> bool {
        matches!(self, Self::Blocker | Self::Critical)
    }
}

/// Jidoka enforcement points in the loading pipeline
#[derive(Debug, Clone)]
pub enum JidokaStop {
    /// Header magic/version mismatch - stop immediately
    InvalidHeader,
    /// Signature verification failed - stop, alert security
    SignatureFailed,
    /// Checksum mismatch - stop, data corrupted
    ChecksumFailed,
    /// WCET budget exceeded - stop, unsafe for deployment
    WcetViolation,
    /// Fairness threshold breached - stop, ethical concern
    FairnessViolation,
    /// Model score below threshold - stop, quality gate
    QualityGateFailed {
        /// Actual score
        score: u8,
        /// Required threshold
        threshold: u8,
    },
}

impl JidokaStop {
    /// All stops are non-recoverable without human intervention (Andon cord)
    #[must_use]
    pub const fn requires_human_review(&self) -> bool {
        true
    }

    /// Get description of the stop
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::InvalidHeader => "Invalid file header".to_string(),
            Self::SignatureFailed => "Signature verification failed".to_string(),
            Self::ChecksumFailed => "Checksum mismatch - data corrupted".to_string(),
            Self::WcetViolation => "WCET budget exceeded".to_string(),
            Self::FairnessViolation => "Fairness threshold breached".to_string(),
            Self::QualityGateFailed { score, threshold } => {
                format!("Quality gate failed: {score}/100 < {threshold}/100")
            }
        }
    }
}

#[cfg(test)]
#[path = "qa_tests.rs"]
mod tests;
