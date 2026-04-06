//! Falsification Checklist Types
//!
//! Types for representing checklist items, results, and evaluation status.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Severity level for checklist items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Project FAIL - blocks release entirely
    Critical,
    /// Requires remediation before release
    Major,
    /// Documented limitation
    Minor,
    /// Clarification needed
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Critical => write!(f, "CRITICAL"),
            Severity::Major => write!(f, "MAJOR"),
            Severity::Minor => write!(f, "MINOR"),
            Severity::Info => write!(f, "INFO"),
        }
    }
}

/// Result status for a checklist item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckStatus {
    /// All criteria passed
    Pass,
    /// Partial evidence, minor issues
    Partial,
    /// Rejection criteria met - claim falsified
    Fail,
    /// Check could not be performed
    Skipped,
}

impl CheckStatus {
    /// Get the score contribution for this status.
    pub fn score(&self) -> f64 {
        match self {
            CheckStatus::Pass => 1.0,
            CheckStatus::Partial => 0.5,
            CheckStatus::Fail | CheckStatus::Skipped => 0.0,
        }
    }
}

impl std::fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckStatus::Pass => write!(f, "PASS"),
            CheckStatus::Partial => write!(f, "PARTIAL"),
            CheckStatus::Fail => write!(f, "FAIL"),
            CheckStatus::Skipped => write!(f, "SKIPPED"),
        }
    }
}

/// A single checklist item result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckItem {
    /// Item ID (e.g., "AI-01", "SDG-05")
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// The claim being tested
    pub claim: String,

    /// Severity level
    pub severity: Severity,

    /// Evaluation status
    pub status: CheckStatus,

    /// Evidence collected
    pub evidence: Vec<Evidence>,

    /// Rejection reason (if failed)
    pub rejection_reason: Option<String>,

    /// TPS principle mapping
    pub tps_principle: String,

    /// Duration of check in milliseconds
    pub duration_ms: u64,
}

impl CheckItem {
    /// Create a new check item.
    pub fn new(id: impl Into<String>, name: impl Into<String>, claim: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            claim: claim.into(),
            severity: Severity::Major,
            status: CheckStatus::Skipped,
            evidence: Vec::new(),
            rejection_reason: None,
            tps_principle: String::new(),
            duration_ms: 0,
        }
    }

    /// Set severity level.
    pub fn with_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    /// Set TPS principle.
    pub fn with_tps(mut self, principle: impl Into<String>) -> Self {
        self.tps_principle = principle.into();
        self
    }

    /// Mark as passed.
    pub fn pass(mut self) -> Self {
        self.status = CheckStatus::Pass;
        self
    }

    /// Mark as failed with reason.
    pub fn fail(mut self, reason: impl Into<String>) -> Self {
        self.status = CheckStatus::Fail;
        self.rejection_reason = Some(reason.into());
        self
    }

    /// Mark as partial.
    pub fn partial(mut self, reason: impl Into<String>) -> Self {
        self.status = CheckStatus::Partial;
        self.rejection_reason = Some(reason.into());
        self
    }

    /// Add evidence.
    pub fn with_evidence(mut self, evidence: Evidence) -> Self {
        self.evidence.push(evidence);
        self
    }

    /// Set duration.
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Record elapsed time from a start instant.
    pub fn finish_timed(self, start: std::time::Instant) -> Self {
        let ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.with_duration(ms)
    }

    /// Check if this is a critical failure.
    pub fn is_critical_failure(&self) -> bool {
        self.severity == Severity::Critical && self.status == CheckStatus::Fail
    }
}

/// Evidence collected for a check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Type of evidence
    pub evidence_type: EvidenceType,

    /// Description
    pub description: String,

    /// Raw data (if applicable)
    pub data: Option<String>,

    /// File paths involved
    pub files: Vec<PathBuf>,
}

impl Evidence {
    /// Create file audit evidence.
    pub fn file_audit(description: impl Into<String>, files: Vec<PathBuf>) -> Self {
        Self {
            evidence_type: EvidenceType::FileAudit,
            description: description.into(),
            data: None,
            files,
        }
    }

    /// Create dependency audit evidence.
    pub fn dependency_audit(description: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            evidence_type: EvidenceType::DependencyAudit,
            description: description.into(),
            data: Some(data.into()),
            files: Vec::new(),
        }
    }

    /// Create schema validation evidence.
    pub fn schema_validation(description: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            evidence_type: EvidenceType::SchemaValidation,
            description: description.into(),
            data: Some(data.into()),
            files: Vec::new(),
        }
    }

    /// Create test result evidence.
    pub fn test_result(description: impl Into<String>, passed: bool) -> Self {
        Self {
            evidence_type: EvidenceType::TestResult,
            description: description.into(),
            data: Some(if passed { "PASSED" } else { "FAILED" }.to_string()),
            files: Vec::new(),
        }
    }
}

/// Type of evidence collected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// File existence/content audit
    FileAudit,
    /// Dependency tree audit
    DependencyAudit,
    /// YAML/config schema validation
    SchemaValidation,
    /// Test execution result
    TestResult,
    /// Static analysis result
    StaticAnalysis,
    /// Coverage measurement
    Coverage,
}

/// Complete checklist evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecklistResult {
    /// Project path evaluated
    pub project_path: PathBuf,

    /// Evaluation timestamp
    pub timestamp: String,

    /// Results by section
    pub sections: HashMap<String, Vec<CheckItem>>,

    /// Overall score (0-100)
    pub score: f64,

    /// TPS assessment grade
    pub grade: TpsGrade,

    /// Whether any critical items failed
    pub has_critical_failure: bool,

    /// Total items evaluated
    pub total_items: usize,

    /// Items passed
    pub passed_items: usize,

    /// Items failed
    pub failed_items: usize,
}

impl ChecklistResult {
    /// Create a new checklist result.
    pub fn new(project_path: &Path) -> Self {
        Self {
            project_path: project_path.to_path_buf(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            sections: HashMap::new(),
            score: 0.0,
            grade: TpsGrade::StopTheLine,
            has_critical_failure: false,
            total_items: 0,
            passed_items: 0,
            failed_items: 0,
        }
    }

    /// Add a section of results.
    pub fn add_section(&mut self, name: impl Into<String>, items: Vec<CheckItem>) {
        self.sections.insert(name.into(), items);
    }

    /// Finalize the result, calculating scores.
    pub fn finalize(&mut self) {
        let mut total_score = 0.0;
        let mut total_items = 0;
        let mut passed = 0;
        let mut failed = 0;
        let mut has_critical = false;

        for items in self.sections.values() {
            for item in items {
                total_items += 1;
                total_score += item.status.score();

                match item.status {
                    CheckStatus::Pass => passed += 1,
                    CheckStatus::Fail => {
                        failed += 1;
                        if item.severity == Severity::Critical {
                            has_critical = true;
                        }
                    }
                    CheckStatus::Partial => {}
                    CheckStatus::Skipped => {}
                }
            }
        }

        self.total_items = total_items;
        self.passed_items = passed;
        self.failed_items = failed;
        self.has_critical_failure = has_critical;

        if total_items > 0 {
            self.score = (total_score / total_items as f64) * 100.0;
        }

        self.grade = TpsGrade::from_score(self.score, has_critical);
    }

    /// Check if the project passes.
    pub fn passes(&self) -> bool {
        !self.has_critical_failure && self.grade.passes()
    }

    /// Get summary string.
    pub fn summary(&self) -> String {
        format!(
            "{}: {:.1}% ({}/{} passed) - {}",
            self.grade,
            self.score,
            self.passed_items,
            self.total_items,
            if self.passes() { "RELEASE OK" } else { "BLOCKED" }
        )
    }
}

/// TPS-aligned assessment grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TpsGrade {
    /// 95-100%: "Good Thinking, Good Products" - Release
    ToyotaStandard,
    /// 85-94%: Beta/Preview with documented issues
    KaizenRequired,
    /// 70-84%: Significant revision, no release
    AndonWarning,
    /// <70% or critical failure: Major rework, halt development
    StopTheLine,
}

impl TpsGrade {
    /// Determine grade from score and critical failure status.
    pub fn from_score(score: f64, has_critical_failure: bool) -> Self {
        if has_critical_failure {
            return TpsGrade::StopTheLine;
        }

        if score >= 95.0 {
            TpsGrade::ToyotaStandard
        } else if score >= 85.0 {
            TpsGrade::KaizenRequired
        } else if score >= 70.0 {
            TpsGrade::AndonWarning
        } else {
            TpsGrade::StopTheLine
        }
    }

    /// Check if this grade allows release.
    pub fn passes(&self) -> bool {
        matches!(self, TpsGrade::ToyotaStandard | TpsGrade::KaizenRequired)
    }
}

impl std::fmt::Display for TpsGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TpsGrade::ToyotaStandard => write!(f, "Toyota Standard"),
            TpsGrade::KaizenRequired => write!(f, "Kaizen Required"),
            TpsGrade::AndonWarning => write!(f, "Andon Warning"),
            TpsGrade::StopTheLine => write!(f, "STOP THE LINE"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // FALS-TYP-001: Severity Display
    // =========================================================================

    #[test]
    fn test_fals_typ_001_severity_display() {
        assert_eq!(format!("{}", Severity::Critical), "CRITICAL");
        assert_eq!(format!("{}", Severity::Major), "MAJOR");
        assert_eq!(format!("{}", Severity::Minor), "MINOR");
        assert_eq!(format!("{}", Severity::Info), "INFO");
    }

    // =========================================================================
    // FALS-TYP-002: CheckStatus Scoring
    // =========================================================================

    #[test]
    fn test_fals_typ_002_check_status_scores() {
        assert_eq!(CheckStatus::Pass.score(), 1.0);
        assert_eq!(CheckStatus::Partial.score(), 0.5);
        assert_eq!(CheckStatus::Fail.score(), 0.0);
        assert_eq!(CheckStatus::Skipped.score(), 0.0);
    }

    // =========================================================================
    // FALS-TYP-003: CheckItem Builder
    // =========================================================================

    #[test]
    fn test_fals_typ_003_check_item_builder() {
        let item = CheckItem::new("AI-01", "Declarative YAML", "Project offers YAML config")
            .with_severity(Severity::Critical)
            .with_tps("Poka-Yoke")
            .pass();

        assert_eq!(item.id, "AI-01");
        assert_eq!(item.severity, Severity::Critical);
        assert_eq!(item.status, CheckStatus::Pass);
        assert_eq!(item.tps_principle, "Poka-Yoke");
    }

    #[test]
    fn test_fals_typ_003_check_item_fail() {
        let item = CheckItem::new("AI-02", "Zero Scripting", "No Python/JS")
            .with_severity(Severity::Critical)
            .fail("Found .py files in src/");

        assert_eq!(item.status, CheckStatus::Fail);
        assert!(item.rejection_reason.is_some());
        assert!(item.is_critical_failure());
    }

    // =========================================================================
    // FALS-TYP-004: Evidence Types
    // =========================================================================

    #[test]
    fn test_fals_typ_004_evidence_file_audit() {
        let evidence =
            Evidence::file_audit("Found 3 Python files", vec![PathBuf::from("src/main.py")]);

        assert_eq!(evidence.evidence_type, EvidenceType::FileAudit);
        assert_eq!(evidence.files.len(), 1);
    }

    #[test]
    fn test_fals_typ_004_evidence_dependency_audit() {
        let evidence = Evidence::dependency_audit("No pyo3 found", "cargo tree output");

        assert_eq!(evidence.evidence_type, EvidenceType::DependencyAudit);
        assert!(evidence.data.is_some());
    }

    // =========================================================================
    // FALS-TYP-005: TpsGrade Determination
    // =========================================================================

    #[test]
    fn test_fals_typ_005_tps_grade_toyota_standard() {
        let grade = TpsGrade::from_score(95.0, false);
        assert_eq!(grade, TpsGrade::ToyotaStandard);
        assert!(grade.passes());
    }

    #[test]
    fn test_fals_typ_005_tps_grade_kaizen() {
        let grade = TpsGrade::from_score(90.0, false);
        assert_eq!(grade, TpsGrade::KaizenRequired);
        assert!(grade.passes());
    }

    #[test]
    fn test_fals_typ_005_tps_grade_andon() {
        let grade = TpsGrade::from_score(75.0, false);
        assert_eq!(grade, TpsGrade::AndonWarning);
        assert!(!grade.passes());
    }

    #[test]
    fn test_fals_typ_005_tps_grade_stop_line() {
        let grade = TpsGrade::from_score(50.0, false);
        assert_eq!(grade, TpsGrade::StopTheLine);
        assert!(!grade.passes());
    }

    #[test]
    fn test_fals_typ_005_critical_failure_stops_line() {
        // Even with 100% score, critical failure = stop
        let grade = TpsGrade::from_score(100.0, true);
        assert_eq!(grade, TpsGrade::StopTheLine);
        assert!(!grade.passes());
    }

    // =========================================================================
    // FALS-TYP-006: ChecklistResult Finalization
    // =========================================================================

    #[test]
    fn test_fals_typ_006_checklist_result_finalize() {
        let mut result = ChecklistResult::new(Path::new("."));

        let items = vec![
            CheckItem::new("T-01", "Test 1", "Claim 1").pass(),
            CheckItem::new("T-02", "Test 2", "Claim 2").pass(),
            CheckItem::new("T-03", "Test 3", "Claim 3").fail("Failed"),
        ];

        result.add_section("Test Section", items);
        result.finalize();

        assert_eq!(result.total_items, 3);
        assert_eq!(result.passed_items, 2);
        assert_eq!(result.failed_items, 1);
        // (1.0 + 1.0 + 0.0) / 3 * 100 = 66.67%
        assert!((result.score - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_fals_typ_006_critical_failure_detection() {
        let mut result = ChecklistResult::new(Path::new("."));

        let items = vec![CheckItem::new("AI-01", "Test", "Claim")
            .with_severity(Severity::Critical)
            .fail("Critical failure")];

        result.add_section("Critical", items);
        result.finalize();

        assert!(result.has_critical_failure);
        assert!(!result.passes());
        assert_eq!(result.grade, TpsGrade::StopTheLine);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_check_status_display() {
        assert_eq!(format!("{}", CheckStatus::Pass), "PASS");
        assert_eq!(format!("{}", CheckStatus::Partial), "PARTIAL");
        assert_eq!(format!("{}", CheckStatus::Fail), "FAIL");
        assert_eq!(format!("{}", CheckStatus::Skipped), "SKIPPED");
    }

    #[test]
    fn test_check_item_partial() {
        let item = CheckItem::new("T-01", "Test", "Claim").partial("Missing docs");

        assert_eq!(item.status, CheckStatus::Partial);
        assert_eq!(item.rejection_reason, Some("Missing docs".to_string()));
    }

    #[test]
    fn test_check_item_with_evidence() {
        let evidence = Evidence::file_audit("Found file", vec![PathBuf::from("test.rs")]);
        let item = CheckItem::new("T-01", "Test", "Claim").with_evidence(evidence);

        assert_eq!(item.evidence.len(), 1);
        assert_eq!(item.evidence[0].evidence_type, EvidenceType::FileAudit);
    }

    #[test]
    fn test_check_item_with_duration() {
        let item = CheckItem::new("T-01", "Test", "Claim").with_duration(150);

        assert_eq!(item.duration_ms, 150);
    }

    #[test]
    fn test_check_item_is_not_critical_failure() {
        let item = CheckItem::new("T-01", "Test", "Claim")
            .with_severity(Severity::Minor)
            .fail("Minor issue");

        assert!(!item.is_critical_failure());
    }

    #[test]
    fn test_evidence_schema_validation() {
        let evidence = Evidence::schema_validation("Config valid", "schema: valid");

        assert_eq!(evidence.evidence_type, EvidenceType::SchemaValidation);
        assert_eq!(evidence.description, "Config valid");
        assert_eq!(evidence.data, Some("schema: valid".to_string()));
    }

    #[test]
    fn test_evidence_test_result_passed() {
        let evidence = Evidence::test_result("Unit tests", true);

        assert_eq!(evidence.evidence_type, EvidenceType::TestResult);
        assert_eq!(evidence.data, Some("PASSED".to_string()));
    }

    #[test]
    fn test_evidence_test_result_failed() {
        let evidence = Evidence::test_result("Integration tests", false);

        assert_eq!(evidence.evidence_type, EvidenceType::TestResult);
        assert_eq!(evidence.data, Some("FAILED".to_string()));
    }

    #[test]
    fn test_checklist_result_summary() {
        let mut result = ChecklistResult::new(Path::new("/test"));
        result.add_section(
            "section",
            vec![
                CheckItem::new("T-01", "Test 1", "Claim 1").pass(),
                CheckItem::new("T-02", "Test 2", "Claim 2").pass(),
            ],
        );
        result.finalize();

        let summary = result.summary();
        assert!(summary.contains("Toyota Standard"));
        assert!(summary.contains("2/2"));
        assert!(summary.contains("RELEASE OK"));
    }

    #[test]
    fn test_checklist_result_summary_blocked() {
        let mut result = ChecklistResult::new(Path::new("/test"));
        result.add_section(
            "section",
            vec![
                CheckItem::new("T-01", "Test 1", "Claim 1").fail("Failed"),
                CheckItem::new("T-02", "Test 2", "Claim 2").fail("Failed"),
            ],
        );
        result.finalize();

        let summary = result.summary();
        assert!(summary.contains("BLOCKED"));
    }

    #[test]
    fn test_tps_grade_display() {
        assert_eq!(format!("{}", TpsGrade::ToyotaStandard), "Toyota Standard");
        assert_eq!(format!("{}", TpsGrade::KaizenRequired), "Kaizen Required");
        assert_eq!(format!("{}", TpsGrade::AndonWarning), "Andon Warning");
        assert_eq!(format!("{}", TpsGrade::StopTheLine), "STOP THE LINE");
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(Severity::Critical, Severity::Critical);
        assert_ne!(Severity::Critical, Severity::Major);
        assert_ne!(Severity::Major, Severity::Minor);
        assert_ne!(Severity::Minor, Severity::Info);
    }

    #[test]
    fn test_check_status_equality() {
        assert_eq!(CheckStatus::Pass, CheckStatus::Pass);
        assert_ne!(CheckStatus::Pass, CheckStatus::Fail);
    }

    #[test]
    fn test_evidence_type_equality() {
        assert_eq!(EvidenceType::FileAudit, EvidenceType::FileAudit);
        assert_ne!(EvidenceType::FileAudit, EvidenceType::TestResult);
    }

    #[test]
    fn test_checklist_result_empty() {
        let mut result = ChecklistResult::new(Path::new("."));
        result.finalize();

        assert_eq!(result.total_items, 0);
        assert_eq!(result.score, 0.0);
        assert!(!result.has_critical_failure);
    }

    #[test]
    fn test_check_item_default_values() {
        let item = CheckItem::new("ID", "Name", "Claim");

        assert_eq!(item.severity, Severity::Major);
        assert_eq!(item.status, CheckStatus::Skipped);
        assert!(item.evidence.is_empty());
        assert!(item.rejection_reason.is_none());
        assert!(item.tps_principle.is_empty());
        assert_eq!(item.duration_ms, 0);
    }

    // =========================================================================
    // Coverage gap: finish_timed
    // =========================================================================

    #[test]
    fn test_check_item_finish_timed() {
        let start = std::time::Instant::now();
        // Small delay to ensure non-zero duration
        std::thread::sleep(std::time::Duration::from_millis(1));
        let item = CheckItem::new("T-01", "Timed", "Timed claim").finish_timed(start);
        assert!(item.duration_ms >= 1, "Duration should be at least 1ms");
    }

    // =========================================================================
    // Coverage gap: finalize with Skipped + Partial items (line 319-320)
    // =========================================================================

    #[test]
    fn test_checklist_result_finalize_with_all_statuses() {
        let mut result = ChecklistResult::new(Path::new("/test"));

        let items = vec![
            CheckItem::new("P-01", "Pass", "Pass claim").pass(),
            CheckItem::new("F-01", "Fail", "Fail claim")
                .with_severity(Severity::Major)
                .fail("Failed"),
            CheckItem::new("PT-01", "Partial", "Partial claim").partial("Partial reason"),
            CheckItem::new("S-01", "Skipped", "Skipped claim"),
        ];

        result.add_section("Mixed", items);
        result.finalize();

        assert_eq!(result.total_items, 4);
        assert_eq!(result.passed_items, 1);
        assert_eq!(result.failed_items, 1);
        // Score: (1.0 + 0.0 + 0.5 + 0.0) / 4 * 100 = 37.5%
        assert!((result.score - 37.5).abs() < 0.1);
        // Major failure (not critical) — has_critical_failure should be false
        assert!(!result.has_critical_failure);
    }

    // =========================================================================
    // Coverage gap: passes() with critical failure + high score
    // =========================================================================

    #[test]
    fn test_checklist_result_passes_critical_blocks() {
        let mut result = ChecklistResult::new(Path::new("/test"));

        // All pass except one critical failure
        let items = vec![
            CheckItem::new("P-01", "Pass 1", "Claim 1").pass(),
            CheckItem::new("P-02", "Pass 2", "Claim 2").pass(),
            CheckItem::new("P-03", "Pass 3", "Claim 3").pass(),
            CheckItem::new("P-04", "Pass 4", "Claim 4").pass(),
            CheckItem::new("C-01", "Critical Fail", "Critical claim")
                .with_severity(Severity::Critical)
                .fail("Critical issue"),
        ];

        result.add_section("Test", items);
        result.finalize();

        // Score is 80% but critical failure blocks
        assert!(result.has_critical_failure);
        assert!(!result.passes());
    }
}
