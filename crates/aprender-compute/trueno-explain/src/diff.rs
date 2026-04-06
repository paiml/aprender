//! Analysis diff and regression detection
//!
//! Compares two analysis reports to detect performance regressions.
//! Supports CI integration with exit codes for automated gating.

use crate::analyzer::AnalysisReport;
use serde::{Deserialize, Serialize};

/// Regression severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Informational change (no action needed)
    Info,
    /// Minor regression (review recommended)
    Warning,
    /// Major regression (CI should fail)
    Critical,
}

/// A detected change between baseline and current
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    /// What changed
    pub metric: String,
    /// Baseline value
    pub baseline: f32,
    /// Current value
    pub current: f32,
    /// Percentage change (positive = regression)
    pub percent_change: f32,
    /// Severity of the change
    pub severity: Severity,
}

/// Result of comparing two analyses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffReport {
    /// Name of the analysis
    pub name: String,
    /// List of detected changes
    pub changes: Vec<Change>,
    /// Whether any regressions were detected
    pub has_regression: bool,
    /// Summary message
    pub summary: String,
}

/// Thresholds for regression detection
#[derive(Debug, Clone)]
pub struct DiffThresholds {
    /// Register increase warning threshold (percentage)
    pub register_increase_warning: f32,
    /// Register increase critical threshold (percentage)
    pub register_increase_critical: f32,
    /// Instruction count increase warning threshold (percentage)
    pub instruction_increase_warning: f32,
    /// Instruction count increase critical threshold (percentage)
    pub instruction_increase_critical: f32,
    /// Occupancy decrease warning threshold (percentage points)
    pub occupancy_decrease_warning: f32,
    /// Occupancy decrease critical threshold (percentage points)
    pub occupancy_decrease_critical: f32,
    /// Warning count increase that triggers concern
    pub warning_count_increase: u32,
}

impl Default for DiffThresholds {
    fn default() -> Self {
        Self {
            register_increase_warning: 10.0,     // 10% more registers = warning
            register_increase_critical: 25.0,    // 25% more = critical
            instruction_increase_warning: 15.0,  // 15% more instructions = warning
            instruction_increase_critical: 50.0, // 50% more = critical
            occupancy_decrease_warning: 10.0,    // 10pp occupancy drop = warning
            occupancy_decrease_critical: 25.0,   // 25pp = critical
            warning_count_increase: 2,           // 2+ new warnings = concern
        }
    }
}

/// Classify severity based on a value exceeding warning/critical thresholds.
/// Returns the severity and whether this constitutes a regression (critical).
fn classify_severity(value: f32, warning_threshold: f32, critical_threshold: f32) -> Severity {
    if value > critical_threshold {
        Severity::Critical
    } else if value > warning_threshold {
        Severity::Warning
    } else {
        Severity::Info
    }
}

/// Compare a percent-change metric (registers, instructions) between baseline and current.
/// Pushes a `Change` if the difference is significant. Returns true if a critical regression.
fn compare_percent_metric(
    metric: &str,
    baseline_val: f32,
    current_val: f32,
    warning_threshold: f32,
    critical_threshold: f32,
    changes: &mut Vec<Change>,
) -> bool {
    if baseline_val <= 0.0 {
        return false;
    }
    let percent = (current_val - baseline_val) / baseline_val * 100.0;
    if percent.abs() <= 0.1 {
        return false;
    }
    let severity = classify_severity(percent, warning_threshold, critical_threshold);
    let is_critical = severity == Severity::Critical;
    changes.push(Change {
        metric: metric.to_string(),
        baseline: baseline_val,
        current: current_val,
        percent_change: percent,
        severity,
    });
    is_critical
}

/// Compare occupancy (absolute percentage-point drop). Returns true if critical regression.
fn compare_occupancy(
    baseline: &AnalysisReport,
    current: &AnalysisReport,
    thresholds: &DiffThresholds,
    changes: &mut Vec<Change>,
) -> bool {
    let baseline_occ = baseline.estimated_occupancy * 100.0;
    let current_occ = current.estimated_occupancy * 100.0;
    let occ_diff = baseline_occ - current_occ; // Positive = regression (drop)
    if occ_diff.abs() <= 0.1 {
        return false;
    }
    // Occupancy uses >= (absolute pp thresholds) unlike percent metrics which use >
    let severity = if occ_diff >= thresholds.occupancy_decrease_critical {
        Severity::Critical
    } else if occ_diff >= thresholds.occupancy_decrease_warning {
        Severity::Warning
    } else {
        Severity::Info
    };
    let is_critical = severity == Severity::Critical;
    changes.push(Change {
        metric: "estimated_occupancy".to_string(),
        baseline: baseline_occ,
        current: current_occ,
        percent_change: -occ_diff, // Negative change = regression
        severity,
    });
    is_critical
}

/// Compare warning counts between baseline and current.
fn compare_warning_counts(
    baseline: &AnalysisReport,
    current: &AnalysisReport,
    thresholds: &DiffThresholds,
    changes: &mut Vec<Change>,
) {
    let baseline_warns = baseline.warnings.len() as u32;
    let current_warns = current.warnings.len() as u32;
    if current_warns <= baseline_warns {
        return;
    }
    let increase = current_warns - baseline_warns;
    let severity = if increase >= thresholds.warning_count_increase {
        Severity::Warning
    } else {
        Severity::Info
    };
    let percent_change =
        if baseline_warns > 0 { (increase as f32 / baseline_warns as f32) * 100.0 } else { 100.0 };
    changes.push(Change {
        metric: "muda_warnings".to_string(),
        baseline: baseline_warns as f32,
        current: current_warns as f32,
        percent_change,
        severity,
    });
}

/// Generate a human-readable summary from the list of changes.
fn build_summary(changes: &[Change]) -> String {
    let critical_count = changes.iter().filter(|c| c.severity == Severity::Critical).count();
    let warning_count = changes.iter().filter(|c| c.severity == Severity::Warning).count();

    if critical_count > 0 {
        format!("{} critical regression(s), {} warning(s)", critical_count, warning_count)
    } else if warning_count > 0 {
        format!("{} warning(s), no critical regressions", warning_count)
    } else if changes.is_empty() {
        "No significant changes detected".to_string()
    } else {
        format!("{} minor change(s)", changes.len())
    }
}

/// Compare two analysis reports
#[must_use]
pub fn compare_reports(
    baseline: &AnalysisReport,
    current: &AnalysisReport,
    thresholds: &DiffThresholds,
) -> DiffReport {
    let mut changes = Vec::new();

    // Compare register usage
    let reg_critical = compare_percent_metric(
        "register_count",
        baseline.registers.total() as f32,
        current.registers.total() as f32,
        thresholds.register_increase_warning,
        thresholds.register_increase_critical,
        &mut changes,
    );

    // Compare instruction count
    let inst_critical = compare_percent_metric(
        "instruction_count",
        baseline.instruction_count as f32,
        current.instruction_count as f32,
        thresholds.instruction_increase_warning,
        thresholds.instruction_increase_critical,
        &mut changes,
    );

    // Compare estimated occupancy
    let occ_critical = compare_occupancy(baseline, current, thresholds, &mut changes);

    // Compare warning counts
    compare_warning_counts(baseline, current, thresholds, &mut changes);

    let has_regression = reg_critical || inst_critical || occ_critical;
    let summary = build_summary(&changes);

    DiffReport { name: current.name.clone(), changes, has_regression, summary }
}

/// Format diff report as text
#[must_use]
pub fn format_diff_text(report: &DiffReport) -> String {
    let mut output = String::new();

    output.push_str(&format!("╔══ Diff Report: {} ══╗\n", report.name));
    output.push_str(&format!("Summary: {}\n\n", report.summary));

    if report.changes.is_empty() {
        output.push_str("  No changes detected.\n");
    } else {
        for change in &report.changes {
            let icon = match change.severity {
                Severity::Critical => "❌",
                Severity::Warning => "⚠️",
                Severity::Info => "ℹ️",
            };
            let direction = if change.percent_change > 0.0 { "↑" } else { "↓" };
            output.push_str(&format!(
                "{} {}: {} → {} ({}{:.1}%)\n",
                icon,
                change.metric,
                change.baseline,
                change.current,
                direction,
                change.percent_change.abs()
            ));
        }
    }

    if report.has_regression {
        output.push_str("\n🚨 REGRESSION DETECTED - CI should fail\n");
    }

    output
}

/// Format diff report as JSON
#[must_use]
pub fn format_diff_json(report: &DiffReport) -> String {
    serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{MemoryPattern, MudaType, MudaWarning, RegisterUsage, RooflineMetric};

    fn make_warning() -> MudaWarning {
        MudaWarning {
            muda_type: MudaType::Transport,
            description: "Test warning".to_string(),
            impact: "Minor".to_string(),
            line: None,
            suggestion: None,
        }
    }

    fn make_report(name: &str, regs: u32, inst: u32, occ: f32, warns: usize) -> AnalysisReport {
        AnalysisReport {
            name: name.to_string(),
            target: "test".to_string(),
            registers: RegisterUsage {
                f32_regs: regs,
                f64_regs: 0,
                pred_regs: 0,
                ..Default::default()
            },
            memory: MemoryPattern::default(),
            roofline: RooflineMetric::default(),
            warnings: (0..warns).map(|_| make_warning()).collect(),
            instruction_count: inst,
            estimated_occupancy: occ,
        }
    }

    #[test]
    fn test_no_changes() {
        let baseline = make_report("test", 32, 100, 0.75, 1);
        let current = make_report("test", 32, 100, 0.75, 1);
        let thresholds = DiffThresholds::default();

        let report = compare_reports(&baseline, &current, &thresholds);

        assert!(!report.has_regression);
        assert!(report.changes.is_empty());
    }

    #[test]
    fn test_register_increase_warning() {
        let baseline = make_report("test", 32, 100, 0.75, 1);
        let current = make_report("test", 36, 100, 0.75, 1); // 12.5% increase
        let thresholds = DiffThresholds::default();

        let report = compare_reports(&baseline, &current, &thresholds);

        assert!(!report.has_regression);
        assert!(report
            .changes
            .iter()
            .any(|c| c.metric == "register_count" && c.severity == Severity::Warning));
    }

    #[test]
    fn test_register_increase_critical() {
        let baseline = make_report("test", 32, 100, 0.75, 1);
        let current = make_report("test", 48, 100, 0.75, 1); // 50% increase
        let thresholds = DiffThresholds::default();

        let report = compare_reports(&baseline, &current, &thresholds);

        assert!(report.has_regression);
        assert!(report
            .changes
            .iter()
            .any(|c| c.metric == "register_count" && c.severity == Severity::Critical));
    }

    #[test]
    fn test_occupancy_decrease() {
        let baseline = make_report("test", 32, 100, 0.75, 1);
        let current = make_report("test", 32, 100, 0.50, 1); // 25pp decrease
        let thresholds = DiffThresholds::default();

        let report = compare_reports(&baseline, &current, &thresholds);

        assert!(report.has_regression);
    }

    #[test]
    fn test_warning_count_increase() {
        let baseline = make_report("test", 32, 100, 0.75, 1);
        let current = make_report("test", 32, 100, 0.75, 4); // 3 new warnings
        let thresholds = DiffThresholds::default();

        let report = compare_reports(&baseline, &current, &thresholds);

        assert!(report.changes.iter().any(|c| c.metric == "muda_warnings"));
    }

    #[test]
    fn test_format_text() {
        let baseline = make_report("test", 32, 100, 0.75, 1);
        let current = make_report("test", 40, 100, 0.75, 1);
        let thresholds = DiffThresholds::default();

        let report = compare_reports(&baseline, &current, &thresholds);
        let text = format_diff_text(&report);

        assert!(text.contains("Diff Report"));
        assert!(text.contains("register_count"));
    }

    #[test]
    fn test_format_json() {
        let baseline = make_report("test", 32, 100, 0.75, 1);
        let current = make_report("test", 32, 100, 0.75, 1);
        let thresholds = DiffThresholds::default();

        let report = compare_reports(&baseline, &current, &thresholds);
        let json = format_diff_json(&report);

        assert!(json.contains("\"name\": \"test\""));
    }

    /// F086: Diff detects register regression
    #[test]
    fn f086_diff_detects_register_regression() {
        let baseline = make_report("gemm", 32, 500, 0.75, 0);
        let current = make_report("gemm", 64, 500, 0.75, 0); // 100% increase
        let thresholds = DiffThresholds::default();

        let report = compare_reports(&baseline, &current, &thresholds);

        assert!(report.has_regression, "Should detect register regression");
        assert!(
            report
                .changes
                .iter()
                .any(|c| c.metric == "register_count" && c.severity == Severity::Critical),
            "Register increase should be critical"
        );
    }

    /// F089: Diff returns exit code on regression
    #[test]
    fn f089_diff_exit_code_on_regression() {
        let baseline = make_report("gemm", 32, 500, 0.75, 0);
        let current = make_report("gemm", 64, 800, 0.50, 5); // Multiple regressions
        let thresholds = DiffThresholds::default();

        let report = compare_reports(&baseline, &current, &thresholds);

        // In real CLI, has_regression determines exit code
        assert!(report.has_regression, "Should have regression for CI fail");
    }
}
