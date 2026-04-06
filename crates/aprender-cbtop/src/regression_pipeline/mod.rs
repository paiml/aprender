//! CI/CD Regression Detection Pipeline (PMAT-047)
//!
//! Automated performance regression detection for CI/CD pipelines.
//!
//! # Design
//!
//! - Git integration for baseline retrieval
//! - Automated benchmark execution
//! - Statistical regression detection
//! - PR status reporting
//! - Artifact storage for historical comparison
//!
//! # Falsification (FKR-048)
//!
//! H₀: Pipeline cannot detect regressions within 60 seconds
//! Test: Run pipeline with known regression, verify detection and timing

mod analysis;
mod metrics;
mod types;

pub use analysis::{MetricRegression, RegressionAnalysis};
pub use metrics::{BenchmarkMetric, BenchmarkResults};
pub use types::{
    GitRef, PipelineConfig, PipelineError, PipelineResult, PipelineStatus, StatusCheck,
};

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// CI/CD regression detection pipeline
#[derive(Debug)]
pub struct RegressionPipeline {
    /// Pipeline configuration
    config: PipelineConfig,
    /// Current status
    status: PipelineStatus,
    /// Start time
    start_time: Option<Instant>,
    /// Cached baselines
    baseline_cache: HashMap<String, BenchmarkResults>,
    /// Run history
    history: Vec<RegressionAnalysis>,
}

impl RegressionPipeline {
    /// Create a new regression pipeline
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            status: PipelineStatus::Pending,
            start_time: None,
            baseline_cache: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Run the pipeline for a git ref
    pub fn run(&mut self, git_ref: &GitRef) -> PipelineResult<RegressionAnalysis> {
        self.status = PipelineStatus::Running;
        self.start_time = Some(Instant::now());

        // Get baseline (from base branch)
        let baseline = self.get_or_create_baseline()?;

        // Run benchmarks for current ref
        let current = self.run_benchmarks(git_ref)?;

        // Analyze regressions
        let analysis = self.analyze_regressions(&baseline, &current);

        // Update status based on analysis
        self.status = analysis.status;

        // Store in history
        self.history.push(analysis.clone());

        // Update PR status if configured
        if let Some(pr_num) = self.extract_pr_number(git_ref) {
            let _ = self.update_pr_status(pr_num, &analysis);
        }

        Ok(analysis)
    }

    /// Get or create baseline from base branch
    fn get_or_create_baseline(&mut self) -> PipelineResult<BenchmarkResults> {
        let base_branch = self.config.base_branch.clone();

        // Check cache first
        if let Some(cached) = self.baseline_cache.get(&base_branch) {
            return Ok(cached.clone());
        }

        // Run baseline benchmarks
        let baseline = self.run_benchmarks(&GitRef::Branch(base_branch.clone()))?;

        // Cache for future use
        self.baseline_cache.insert(base_branch, baseline.clone());

        Ok(baseline)
    }

    /// Run benchmarks for a git ref
    fn run_benchmarks(&self, git_ref: &GitRef) -> PipelineResult<BenchmarkResults> {
        // Simulate benchmark execution
        let start = Instant::now();

        // In production, this would:
        // 1. Checkout git ref
        // 2. Run benchmark command
        // 3. Parse results

        let commit = match git_ref {
            GitRef::Commit(sha) => sha.clone(),
            GitRef::Branch(name) => format!("{}-head", name),
            GitRef::Tag(name) => format!("tag-{}", name),
            GitRef::PullRequest(num) => format!("pr-{}", num),
        };

        let branch = match git_ref {
            GitRef::Branch(name) => name.clone(),
            GitRef::Commit(_) | GitRef::Tag(_) | GitRef::PullRequest(_) => "detached".to_string(),
        };

        let mut results = BenchmarkResults::new(commit, branch);

        // Simulate metrics (in production, parse from benchmark output)
        results.add_metric(BenchmarkMetric::new(
            "latency_p50",
            vec![100.0, 102.0, 98.0, 101.0, 99.0],
            "\u{03bc}s",
        ));
        results.add_metric(BenchmarkMetric::new(
            "latency_p99",
            vec![200.0, 210.0, 195.0, 205.0, 198.0],
            "\u{03bc}s",
        ));
        results.add_metric(BenchmarkMetric::new(
            "throughput",
            vec![10000.0, 10100.0, 9900.0, 10050.0, 9950.0],
            "ops/s",
        ));

        results.duration_ms = start.elapsed().as_millis() as u64;

        Ok(results)
    }

    /// Analyze regressions between baseline and current
    fn analyze_regressions(
        &self,
        baseline: &BenchmarkResults,
        current: &BenchmarkResults,
    ) -> RegressionAnalysis {
        let start = Instant::now();
        let mut regressions = Vec::new();

        // Compare each metric
        for baseline_metric in &baseline.metrics {
            if let Some(current_metric) = current.get_metric(&baseline_metric.name) {
                let regression = MetricRegression::from_metrics(
                    baseline_metric,
                    current_metric,
                    self.config.regression_threshold_percent,
                    self.config.warning_threshold_percent,
                );
                regressions.push(regression);
            }
        }

        // Determine overall status
        let has_regressions = regressions.iter().any(|r| r.is_regression);
        let has_warnings = regressions.iter().any(|r| r.is_warning);

        let status = if has_regressions {
            PipelineStatus::Failed
        } else if has_warnings {
            PipelineStatus::Warning
        } else {
            PipelineStatus::Passed
        };

        // Build summary
        let summary = self.build_summary(&regressions, status);

        RegressionAnalysis {
            baseline: baseline.clone(),
            current: current.clone(),
            regressions,
            status,
            analysis_duration_ms: start.elapsed().as_millis() as u64,
            summary,
        }
    }

    /// Build summary message
    fn build_summary(&self, regressions: &[MetricRegression], status: PipelineStatus) -> String {
        let regression_count = regressions.iter().filter(|r| r.is_regression).count();
        let warning_count = regressions.iter().filter(|r| r.is_warning).count();

        match status {
            PipelineStatus::Passed => {
                "All benchmarks passed. No performance regressions detected.".to_string()
            }
            PipelineStatus::Warning => {
                format!(
                    "Performance warnings detected: {} metrics show minor degradation ({:.1}%-{:.1}%)",
                    warning_count,
                    self.config.warning_threshold_percent,
                    self.config.regression_threshold_percent
                )
            }
            PipelineStatus::Failed => {
                format!(
                    "Performance regression detected: {} metrics degraded by >{:.1}%",
                    regression_count, self.config.regression_threshold_percent
                )
            }
            PipelineStatus::Pending
            | PipelineStatus::Running
            | PipelineStatus::Cancelled
            | PipelineStatus::Error => "Pipeline status unknown".to_string(),
        }
    }

    /// Extract PR number from git ref
    fn extract_pr_number(&self, git_ref: &GitRef) -> Option<u64> {
        match git_ref {
            GitRef::PullRequest(num) => Some(*num),
            GitRef::Branch(_) | GitRef::Commit(_) | GitRef::Tag(_) => None,
        }
    }

    /// Update PR status check
    fn update_pr_status(&self, pr_num: u64, analysis: &RegressionAnalysis) -> PipelineResult<()> {
        let _status_check = StatusCheck {
            name: "Performance Regression Check".to_string(),
            state: analysis.status,
            description: analysis.summary.clone(),
            target_url: self.config.artifact_path.clone().into(),
            context: "cbtop/regression".to_string(),
        };

        // In production, use GitHub API:
        // POST /repos/{owner}/{repo}/statuses/{sha}
        let _ = pr_num;

        Ok(())
    }

    /// Get current status
    pub fn status(&self) -> PipelineStatus {
        self.status
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Option<Duration> {
        self.start_time.map(|t| t.elapsed())
    }

    /// Get configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Get run history
    pub fn history(&self) -> &[RegressionAnalysis] {
        &self.history
    }

    /// Clear baseline cache
    pub fn clear_baseline_cache(&mut self) {
        self.baseline_cache.clear();
    }

    /// Store artifact for a run
    pub fn store_artifact(&self, analysis: &RegressionAnalysis) -> PipelineResult<String> {
        // In production, serialize and store to artifact_path
        let artifact_id = format!(
            "{}-{}",
            analysis.current.commit, analysis.current.timestamp_ns
        );

        Ok(artifact_id)
    }

    /// Generate markdown report
    pub fn generate_report(&self, analysis: &RegressionAnalysis) -> String {
        let mut report = String::new();

        // Header
        report.push_str("# Performance Regression Report\n\n");

        // Summary
        report.push_str(&format!("## Summary\n\n{}\n\n", analysis.summary));

        // Status badge
        let badge = match analysis.status {
            PipelineStatus::Passed => "![Passed](https://img.shields.io/badge/status-passed-green)",
            PipelineStatus::Warning => {
                "![Warning](https://img.shields.io/badge/status-warning-yellow)"
            }
            PipelineStatus::Failed => "![Failed](https://img.shields.io/badge/status-failed-red)",
            PipelineStatus::Pending
            | PipelineStatus::Running
            | PipelineStatus::Cancelled
            | PipelineStatus::Error => {
                "![Unknown](https://img.shields.io/badge/status-unknown-gray)"
            }
        };
        report.push_str(&format!("{}\n\n", badge));

        // Comparison table
        report.push_str("## Metric Comparison\n\n");
        report.push_str("| Metric | Baseline | Current | Change | Status |\n");
        report.push_str("|--------|----------|---------|--------|--------|\n");

        for regression in &analysis.regressions {
            let status = if regression.is_regression {
                "\u{274c} Regression"
            } else if regression.is_warning {
                "\u{26a0}\u{fe0f} Warning"
            } else if regression.percent_change.abs() > 1.0 {
                "\u{2705} Improved"
            } else {
                "\u{2796} Stable"
            };

            report.push_str(&format!(
                "| {} | {:.2} {} | {:.2} {} | {:+.1}% | {} |\n",
                regression.name,
                regression.baseline_mean,
                regression.unit,
                regression.current_mean,
                regression.unit,
                regression.percent_change,
                status
            ));
        }

        report.push('\n');

        // Details
        report.push_str("## Details\n\n");
        report.push_str(&format!(
            "- **Baseline commit:** {}\n",
            analysis.baseline.commit
        ));
        report.push_str(&format!(
            "- **Current commit:** {}\n",
            analysis.current.commit
        ));
        report.push_str(&format!(
            "- **Analysis time:** {}ms\n",
            analysis.analysis_duration_ms
        ));

        report
    }
}

/// Default timeout in seconds
pub const DEFAULT_TIMEOUT_SEC: u64 = 600;

/// Default regression threshold
pub const DEFAULT_REGRESSION_THRESHOLD: f64 = 5.0;

/// Default warning threshold
pub const DEFAULT_WARNING_THRESHOLD: f64 = 2.0;

#[cfg(test)]
mod tests;
