//! Verification report generation and formatting.
//!
//! Provides visual output of pipeline verification results
//! in the style of probar pixel-perfect TUI rendering.

use super::{StageResult, StageStatus};
use std::fmt::Write;
use std::time::Duration;

/// Report of pipeline verification results.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// Pipeline name
    pipeline_name: String,
    /// Results for each stage
    results: Vec<StageResult>,
    /// Total verification duration
    duration: Option<Duration>,
}

impl VerifyReport {
    /// Create a new empty report.
    pub fn new(pipeline_name: impl Into<String>) -> Self {
        Self {
            pipeline_name: pipeline_name.into(),
            results: Vec::new(),
            duration: None,
        }
    }

    /// Add a stage result.
    pub fn add_result(&mut self, result: StageResult) {
        self.results.push(result);
    }

    /// Set the total duration.
    pub fn set_duration(&mut self, duration: Duration) {
        self.duration = Some(duration);
    }

    /// Get the pipeline name.
    pub fn pipeline_name(&self) -> &str {
        &self.pipeline_name
    }

    /// Get all results.
    pub fn results(&self) -> &[StageResult] {
        &self.results
    }

    /// Check if all stages passed.
    pub fn all_passed(&self) -> bool {
        self.results
            .iter()
            .all(|r| r.status().is_passed() || r.status() == StageStatus::Skipped)
            && self.results.iter().any(|r| r.status().is_passed())
    }

    /// Get the first failure, if any.
    pub fn first_failure(&self) -> Option<&StageResult> {
        self.results.iter().find(|r| r.status().is_failed())
    }

    /// Count passed stages.
    pub fn passed_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.status().is_passed())
            .count()
    }

    /// Count failed stages.
    pub fn failed_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.status().is_failed())
            .count()
    }

    /// Count skipped stages.
    pub fn skipped_count(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.status() == StageStatus::Skipped)
            .count()
    }

    /// Generate a summary line.
    pub fn summary(&self) -> String {
        let total = self.results.len();
        let passed = self.passed_count();
        let failed = self.failed_count();
        let skipped = self.skipped_count();

        if self.all_passed() {
            format!("\x1b[32m✓ All {passed} stages passed\x1b[0m")
        } else {
            format!(
                "\x1b[31m✗ {failed}/{total} stages failed\x1b[0m (passed: {passed}, skipped: {skipped})"
            )
        }
    }

    /// Render the report as a visual table.
    ///
    /// This produces output suitable for terminal display with
    /// Unicode box drawing characters and ANSI color codes.
    pub fn render(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(
            "\n╔══════════════════════════════════════════════════════════════════════════════╗\n",
        );
        output.push_str(
            "║                    APRENDER PIPELINE VERIFICATION                            ║\n",
        );
        let _ = writeln!(
            output,
            "║                    Pipeline: {:48}║",
            self.pipeline_name
        );
        output.push_str(
            "╠══════════════════════════════════════════════════════════════════════════════╣\n",
        );

        // Table header
        output.push_str(
            "║                                                                              ║\n",
        );
        output.push_str("║  ┌─────┬─────────────────┬────────┬────────────────────┬────────────────────┬─────────┐  ║\n");
        output.push_str("║  │  #  │ Stage           │ Status │ Our Value          │ Ground Truth       │  Delta  │  ║\n");
        output.push_str("║  ├─────┼─────────────────┼────────┼────────────────────┼────────────────────┼─────────┤  ║\n");

        // Stage rows
        for (i, result) in self.results.iter().enumerate() {
            let letter = (b'A' + i as u8) as char;
            let status_icon = result.status().icon();
            let status_color = result.status().color();
            const RESET: &str = "\x1b[0m";

            let our_val = result.our_stats().map_or_else(
                || "N/A".to_string(),
                |s| format!("μ={:+.4} σ={:.4}", s.mean(), s.std()),
            );

            let gt_val = result.gt_stats().map_or_else(
                || "(no ground truth)".to_string(),
                |s| format!("μ={:+.4} σ={:.4}", s.mean(), s.std()),
            );

            let delta_str = result.delta().map_or_else(
                || "  N/A  ".to_string(),
                |d| format!("{:>6.1}%", d.percent()),
            );

            let _ = writeln!(
                output,
                "║  │  {letter}  │ {:15} │   {status_color}{status_icon}{RESET}    │ {our_val:18} │ {gt_val:18} │{delta_str} │  ║",
                result.name()
            );
        }

        output.push_str("║  └─────┴─────────────────┴────────┴────────────────────┴────────────────────┴─────────┘  ║\n");
        output.push_str(
            "║                                                                              ║\n",
        );

        // Diagnosis section
        if let Some(failure) = self.first_failure() {
            output.push_str(
                "╠══════════════════════════════════════════════════════════════════════════════╣\n",
            );
            output.push_str(
                "║                              DIAGNOSIS                                       ║\n",
            );
            output.push_str(
                "╠══════════════════════════════════════════════════════════════════════════════╣\n",
            );

            for line in failure.diagnose() {
                let _ = writeln!(output, "║  {line:72}  ║");
            }
        }

        // Footer with summary
        output.push_str(
            "╠══════════════════════════════════════════════════════════════════════════════╣\n",
        );
        let summary = self.summary();
        let _ = writeln!(output, "║  {summary:72}  ║");

        if let Some(duration) = self.duration {
            let duration_str = format!("{:.2}ms", duration.as_secs_f64() * 1000.0);
            let _ = writeln!(output, "║  Duration: {duration_str:62}║");
        }

        output.push_str(
            "╚══════════════════════════════════════════════════════════════════════════════╝\n",
        );

        output
    }

    /// Render as minimal one-line summary.
    pub fn render_oneline(&self) -> String {
        let icons: String = self
            .results
            .iter()
            .map(|r| r.status().icon())
            .collect::<Vec<_>>()
            .join("");

        let status = if self.all_passed() { "PASS" } else { "FAIL" };
        let name = &self.pipeline_name;
        format!("[{icons}] {name} - {status}")
    }
}

impl std::fmt::Display for VerifyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::Delta;

    #[test]
    fn test_all_passed() {
        let mut report = VerifyReport::new("test");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::passed("b", Delta::from_percent(2.0)));
        assert!(report.all_passed());
    }

    #[test]
    fn test_has_failure() {
        let mut report = VerifyReport::new("test");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::failed("b", Delta::from_percent(90.0)));
        assert!(!report.all_passed());
    }

    #[test]
    fn test_first_failure() {
        let mut report = VerifyReport::new("test");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::failed("b", Delta::from_percent(90.0)));
        report.add_result(StageResult::skipped("c"));

        let first = report.first_failure().unwrap();
        assert_eq!(first.name(), "b");
    }

    #[test]
    fn test_counts() {
        let mut report = VerifyReport::new("test");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::failed("b", Delta::from_percent(90.0)));
        report.add_result(StageResult::skipped("c"));
        report.add_result(StageResult::skipped("d"));

        assert_eq!(report.passed_count(), 1);
        assert_eq!(report.failed_count(), 1);
        assert_eq!(report.skipped_count(), 2);
    }

    #[test]
    fn test_render_oneline() {
        let mut report = VerifyReport::new("whisper-tiny");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::failed("b", Delta::from_percent(90.0)));
        report.add_result(StageResult::skipped("c"));

        let line = report.render_oneline();
        assert!(line.contains("✓"));
        assert!(line.contains("✗"));
        assert!(line.contains("○"));
        assert!(line.contains("FAIL"));
    }

    #[test]
    fn test_summary() {
        let mut report = VerifyReport::new("test");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::passed("b", Delta::from_percent(2.0)));

        let summary = report.summary();
        assert!(summary.contains("passed"));
        assert!(summary.contains("2"));
    }

    #[test]
    fn test_render() {
        use std::time::Duration;

        let mut report = VerifyReport::new("test-pipeline");
        report.add_result(StageResult::passed("stage_a", Delta::from_percent(1.0)));
        report.add_result(StageResult::failed("stage_b", Delta::from_percent(50.0)));
        report.set_duration(Duration::from_millis(100));

        let rendered = report.render();
        assert!(rendered.contains("APRENDER PIPELINE VERIFICATION"));
        assert!(rendered.contains("test-pipeline"));
        assert!(rendered.contains("stage_a"));
        assert!(rendered.contains("stage_b"));
        assert!(rendered.contains("Duration"));
    }

    #[test]
    fn test_render_all_passed() {
        let mut report = VerifyReport::new("success-pipeline");
        report.add_result(StageResult::passed("a", Delta::from_percent(0.5)));

        let rendered = report.render();
        assert!(rendered.contains("success-pipeline"));
        assert!(!rendered.contains("DIAGNOSIS")); // No failures
    }

    #[test]
    fn test_render_with_diagnosis() {
        use crate::verify::GroundTruth;

        let mut report = VerifyReport::new("failing-pipeline");

        // Create a failed result with diagnosis info
        let our = GroundTruth::from_stats(0.5, 1.0);
        let gt = GroundTruth::from_stats(-0.5, 1.0);
        let delta = Delta::compute(&our, &gt);
        let failed = StageResult::failed("bad_stage", delta);
        report.add_result(failed);

        let rendered = report.render();
        assert!(rendered.contains("DIAGNOSIS"));
        assert!(rendered.contains("bad_stage"));
    }

    #[test]
    fn test_render_with_stats() {
        use crate::verify::GroundTruth;

        let mut report = VerifyReport::new("stats-pipeline");

        let our = GroundTruth::from_stats(0.1, 0.5);
        let gt = GroundTruth::from_stats(0.0, 0.5);
        let delta = Delta::compute(&our, &gt);

        let result = StageResult::passed("stage_a", delta);
        report.add_result(result);

        let rendered = report.render();
        assert!(rendered.contains("stage_a"));
        // Stats are rendered - check for μ= and σ=
        // The render method accesses our_stats and gt_stats
    }

    #[test]
    fn test_display_trait() {
        let mut report = VerifyReport::new("display-test");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));

        let display_output = format!("{}", report);
        assert!(display_output.contains("APRENDER PIPELINE VERIFICATION"));
    }

    #[test]
    fn test_all_passed_only_skipped() {
        let mut report = VerifyReport::new("only-skipped");
        report.add_result(StageResult::skipped("a"));
        report.add_result(StageResult::skipped("b"));

        // all_passed should be false if only skipped (no passes)
        assert!(!report.all_passed());
    }

    #[test]
    fn test_summary_with_failure() {
        let mut report = VerifyReport::new("fail-summary");
        report.add_result(StageResult::passed("a", Delta::from_percent(1.0)));
        report.add_result(StageResult::failed("b", Delta::from_percent(50.0)));
        report.add_result(StageResult::skipped("c"));

        let summary = report.summary();
        assert!(summary.contains("failed"));
        assert!(summary.contains("1")); // 1 failed
        assert!(summary.contains("skipped"));
    }
}
