//! Output formatting and progress reporting

use console::{style, Style, Term};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Output format for test results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON output
    Json,
    /// TAP (Test Anything Protocol)
    Tap,
}

/// Progress reporter for test execution
#[derive(Debug)]
pub struct ProgressReporter {
    term: Term,
    progress_bar: Option<ProgressBar>,
    /// Whether to use colors
    pub use_color: bool,
    /// Quiet mode
    pub quiet: bool,
}

impl Default for ProgressReporter {
    fn default() -> Self {
        Self::new(true, false)
    }
}

impl ProgressReporter {
    /// Create a new progress reporter
    #[must_use]
    pub fn new(use_color: bool, quiet: bool) -> Self {
        Self {
            term: Term::stderr(),
            progress_bar: None,
            use_color,
            quiet,
        }
    }

    /// Start a progress bar for multiple tests
    pub fn start_progress(&mut self, total: u64, message: &str) {
        if self.quiet {
            return;
        }

        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("=>-"),
        );
        pb.set_message(message.to_string());
        self.progress_bar = Some(pb);
    }

    /// Increment progress
    pub fn increment(&self, delta: u64) {
        if let Some(ref pb) = self.progress_bar {
            pb.inc(delta);
        }
    }

    /// Update progress message
    pub fn set_message(&self, message: &str) {
        if let Some(ref pb) = self.progress_bar {
            pb.set_message(message.to_string());
        }
    }

    /// Finish progress bar
    pub fn finish(&self) {
        if let Some(ref pb) = self.progress_bar {
            pb.finish_with_message("Done");
        }
    }

    /// Print a success message
    pub fn success(&self, message: &str) {
        if self.quiet {
            return;
        }

        let prefix = if self.use_color {
            style("✓").green().bold().to_string()
        } else {
            "PASS".to_string()
        };

        let _ = self.term.write_line(&format!("{prefix} {message}"));
    }

    /// Print a failure message
    pub fn failure(&self, message: &str) {
        // Always print failures, even in quiet mode
        let prefix = if self.use_color {
            style("✗").red().bold().to_string()
        } else {
            "FAIL".to_string()
        };

        let _ = self.term.write_line(&format!("{prefix} {message}"));
    }

    /// Print a warning message
    pub fn warning(&self, message: &str) {
        if self.quiet {
            return;
        }

        let prefix = if self.use_color {
            style("⚠").yellow().bold().to_string()
        } else {
            "WARN".to_string()
        };

        let _ = self.term.write_line(&format!("{prefix} {message}"));
    }

    /// Print an info message
    pub fn info(&self, message: &str) {
        if self.quiet {
            return;
        }

        let prefix = if self.use_color {
            style("ℹ").blue().bold().to_string()
        } else {
            "INFO".to_string()
        };

        let _ = self.term.write_line(&format!("{prefix} {message}"));
    }

    /// Print a section header
    pub fn header(&self, title: &str) {
        if self.quiet {
            return;
        }

        let styled = if self.use_color {
            style(title).bold().underlined().to_string()
        } else {
            format!("=== {title} ===")
        };

        let _ = self.term.write_line("");
        let _ = self.term.write_line(&styled);
    }

    /// Print test summary
    pub fn summary(&self, passed: usize, failed: usize, skipped: usize, duration: Duration) {
        if self.quiet && failed == 0 {
            return;
        }

        let _ = self.term.write_line("");

        let total = passed + failed + skipped;
        let duration_secs = duration.as_secs_f64();

        if self.use_color {
            let passed_style = Style::new().green().bold();
            let failed_style = Style::new().red().bold();
            let skipped_style = Style::new().yellow();

            let status = if failed > 0 {
                failed_style.apply_to("FAILED")
            } else {
                passed_style.apply_to("PASSED")
            };

            let _ = self.term.write_line(&format!(
                "{} {} tests in {:.2}s ({} passed, {} failed, {} skipped)",
                status,
                total,
                duration_secs,
                passed_style.apply_to(passed),
                if failed > 0 {
                    failed_style.apply_to(failed).to_string()
                } else {
                    failed.to_string()
                },
                skipped_style.apply_to(skipped)
            ));
        } else {
            let status = if failed > 0 { "FAILED" } else { "PASSED" };
            let _ = self.term.write_line(&format!(
                "{status} {total} tests in {duration_secs:.2}s ({passed} passed, {failed} failed, {skipped} skipped)"
            ));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod output_format_tests {
        use super::*;

        #[test]
        fn test_default_format() {
            let format = OutputFormat::default();
            assert_eq!(format, OutputFormat::Text);
        }

        #[test]
        fn test_format_variants() {
            let _ = OutputFormat::Text;
            let _ = OutputFormat::Json;
            let _ = OutputFormat::Tap;
        }
    }

    mod progress_reporter_tests {
        use super::*;

        #[test]
        fn test_new_reporter() {
            let reporter = ProgressReporter::new(true, false);
            assert!(reporter.use_color);
            assert!(!reporter.quiet);
        }

        #[test]
        fn test_default_reporter() {
            let reporter = ProgressReporter::default();
            assert!(reporter.use_color);
            assert!(!reporter.quiet);
        }

        #[test]
        fn test_quiet_reporter() {
            let reporter = ProgressReporter::new(false, true);
            assert!(reporter.quiet);
        }

        #[test]
        fn test_success_message() {
            let reporter = ProgressReporter::new(false, false);
            reporter.success("Test passed");
            // No panic = success
        }

        #[test]
        fn test_failure_message() {
            let reporter = ProgressReporter::new(false, false);
            reporter.failure("Test failed");
            // No panic = success
        }

        #[test]
        fn test_warning_message() {
            let reporter = ProgressReporter::new(false, false);
            reporter.warning("Test warning");
            // No panic = success
        }

        #[test]
        fn test_info_message() {
            let reporter = ProgressReporter::new(false, false);
            reporter.info("Test info");
            // No panic = success
        }

        #[test]
        fn test_header() {
            let reporter = ProgressReporter::new(false, false);
            reporter.header("Test Header");
            // No panic = success
        }

        #[test]
        fn test_summary_passed() {
            let reporter = ProgressReporter::new(false, false);
            reporter.summary(10, 0, 2, Duration::from_secs(5));
            // No panic = success
        }

        #[test]
        fn test_summary_failed() {
            let reporter = ProgressReporter::new(false, false);
            reporter.summary(8, 2, 0, Duration::from_secs(3));
            // No panic = success
        }

        #[test]
        fn test_progress_bar() {
            let mut reporter = ProgressReporter::new(false, false);
            reporter.start_progress(10, "Running tests");
            reporter.increment(1);
            reporter.set_message("test_1");
            reporter.increment(1);
            reporter.finish();
            // No panic = success
        }

        #[test]
        fn test_quiet_mode_suppresses_output() {
            let mut reporter = ProgressReporter::new(false, true);
            reporter.start_progress(10, "Running tests");
            reporter.success("hidden");
            reporter.warning("hidden");
            reporter.info("hidden");
            reporter.header("hidden");
            // Failure is still printed
            reporter.failure("shown");
            // No panic = success
        }

        #[test]
        fn test_color_mode_messages() {
            let reporter = ProgressReporter::new(true, false);
            reporter.success("Pass with color");
            reporter.failure("Fail with color");
            reporter.warning("Warn with color");
            reporter.info("Info with color");
            reporter.header("Header with color");
        }

        #[test]
        fn test_summary_all_skipped() {
            let reporter = ProgressReporter::new(false, false);
            reporter.summary(0, 0, 5, Duration::from_secs(1));
        }

        #[test]
        fn test_summary_mixed() {
            let reporter = ProgressReporter::new(true, false);
            reporter.summary(5, 3, 2, Duration::from_millis(500));
        }

        #[test]
        fn test_progress_without_start() {
            let reporter = ProgressReporter::new(false, false);
            // These should not panic even without start_progress
            reporter.increment(1);
            reporter.set_message("test");
            reporter.finish();
        }

        #[test]
        fn test_debug() {
            let reporter = ProgressReporter::new(true, false);
            let debug = format!("{reporter:?}");
            assert!(debug.contains("ProgressReporter"));
        }
    }

    mod output_format_additional_tests {
        use super::*;

        #[test]
        fn test_clone() {
            let format = OutputFormat::Json;
            let cloned = format;
            assert_eq!(format, cloned);
        }

        #[test]
        fn test_debug() {
            let debug = format!("{:?}", OutputFormat::Text);
            assert!(debug.contains("Text"));
        }

        #[test]
        fn test_serialize() {
            let format = OutputFormat::Json;
            let json = serde_json::to_string(&format).unwrap();
            assert!(json.contains("Json"));
        }
    }
}
