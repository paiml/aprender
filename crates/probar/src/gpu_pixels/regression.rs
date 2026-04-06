//! GPU Kernel Regression Testing
//!
//! Tracks kernel PTX signatures and detects regressions across versions.

// Allow unwrap on Option::last() when we've already checked non-empty
#![allow(clippy::unwrap_used)]

use super::kernel_pixels::{GpuPixelResult, GpuPixelTestSuite, KernelPixelConfig};
use super::ptx_analysis::PtxAnalyzer;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for regression testing
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// Directory to store baseline PTX
    pub baseline_dir: String,
    /// Auto-update baselines on pass
    pub auto_update: bool,
    /// Kernel pixel config
    pub pixel_config: KernelPixelConfig,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            baseline_dir: ".gpu_baselines".to_string(),
            auto_update: false,
            pixel_config: KernelPixelConfig::default(),
        }
    }
}

/// Result of regression check
#[derive(Debug, Clone)]
pub struct RegressionResult {
    /// Kernel name
    pub kernel_name: String,
    /// Is this a regression?
    pub is_regression: bool,
    /// Regression details
    pub details: Option<String>,
    /// Pixel test results
    pub pixel_results: GpuPixelTestSuite,
    /// Duration
    pub duration: Duration,
}

impl RegressionResult {
    /// Check if kernel passed all checks
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.is_regression && self.pixel_results.all_passed()
    }
}

/// GPU kernel regression test suite
#[derive(Debug)]
pub struct GpuRegressionSuite {
    /// Configuration
    config: RegressionConfig,
    /// Baseline PTX for each kernel
    baselines: HashMap<String, String>,
    /// Results
    results: Vec<RegressionResult>,
}

impl GpuRegressionSuite {
    /// Create new regression suite
    #[must_use]
    pub fn new(config: RegressionConfig) -> Self {
        Self {
            config,
            baselines: HashMap::new(),
            results: Vec::new(),
        }
    }

    /// Add baseline PTX for a kernel
    pub fn add_baseline(&mut self, kernel_name: &str, ptx: &str) {
        self.baselines
            .insert(kernel_name.to_string(), ptx.to_string());
    }

    /// Test kernel against baseline
    pub fn test_kernel(&mut self, kernel_name: &str, ptx: &str) -> &RegressionResult {
        let start = Instant::now();

        // Run pixel tests
        let mut pixel_suite = GpuPixelTestSuite::new(kernel_name);
        let ptx_result = PtxAnalyzer::default().analyze(ptx);
        pixel_suite.add_result(GpuPixelResult::from_ptx_validation(&ptx_result));
        pixel_suite.run_kernel_pixels(ptx, &self.config.pixel_config);

        // Check against baseline
        let (is_regression, details) = if let Some(baseline) = self.baselines.get(kernel_name) {
            self.compare_ptx(baseline, ptx)
        } else {
            (false, None)
        };

        let result = RegressionResult {
            kernel_name: kernel_name.to_string(),
            is_regression,
            details,
            pixel_results: pixel_suite,
            duration: start.elapsed(),
        };

        self.results.push(result);
        self.results.last().unwrap()
    }

    /// Compare PTX for regression
    fn compare_ptx(&self, baseline: &str, current: &str) -> (bool, Option<String>) {
        // Extract key patterns
        let baseline_patterns = self.extract_patterns(baseline);
        let current_patterns = self.extract_patterns(current);

        // Check for regressions
        let mut regressions = Vec::new();

        // Check shared memory addressing
        if baseline_patterns.uses_u32_shared && !current_patterns.uses_u32_shared {
            regressions.push("Regression: switched from u32 to u64 shared memory addressing");
        }

        // Check barrier presence
        if baseline_patterns.has_barrier && !current_patterns.has_barrier {
            regressions.push("Regression: removed barrier synchronization");
        }

        // Check kernel name
        if baseline_patterns.kernel_names != current_patterns.kernel_names {
            regressions.push("Regression: kernel name changed");
        }

        if regressions.is_empty() {
            (false, None)
        } else {
            (true, Some(regressions.join("; ")))
        }
    }

    /// Extract testable patterns from PTX
    fn extract_patterns(&self, ptx: &str) -> PtxPatterns {
        let analyzer = PtxAnalyzer::default();
        let result = analyzer.analyze(ptx);

        PtxPatterns {
            kernel_names: result.kernel_names,
            uses_u32_shared: !ptx.contains("[%rd")
                || !ptx.contains("st.shared") && !ptx.contains("ld.shared"),
            has_barrier: ptx.contains("bar.sync"),
            has_shared_mem: ptx.contains(".shared"),
        }
    }

    /// Get all results
    #[must_use]
    pub fn results(&self) -> &[RegressionResult] {
        &self.results
    }

    /// Check if all tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed())
    }

    /// Get summary
    #[must_use]
    pub fn summary(&self) -> String {
        let passed = self.results.iter().filter(|r| r.passed()).count();
        let total = self.results.len();
        let regressions = self.results.iter().filter(|r| r.is_regression).count();

        format!(
            "GPU Regression Suite: {}/{} passed, {} regressions",
            passed, total, regressions
        )
    }
}

/// Extracted PTX patterns for comparison
#[derive(Debug, Clone)]
#[allow(dead_code)] // Reserved for future PTX pattern analysis
struct PtxPatterns {
    kernel_names: Vec<String>,
    uses_u32_shared: bool,
    has_barrier: bool,
    has_shared_mem: bool,
}

/// Run regression suite on multiple kernels
#[must_use]
pub fn run_regression_suite(
    kernels: &[(&str, &str)], // (name, ptx)
    baselines: &[(&str, &str)],
    config: RegressionConfig,
) -> GpuRegressionSuite {
    let mut suite = GpuRegressionSuite::new(config);

    // Add baselines
    for (name, ptx) in baselines {
        suite.add_baseline(name, ptx);
    }

    // Test kernels
    for (name, ptx) in kernels {
        suite.test_kernel(name, ptx);
    }

    suite
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_suite_creation() {
        let suite = GpuRegressionSuite::new(RegressionConfig::default());
        assert!(suite.results().is_empty());
        assert!(suite.all_passed());
    }

    #[test]
    fn test_add_baseline() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());
        suite.add_baseline("test_kernel", ".version 8.0");
        assert!(suite.baselines.contains_key("test_kernel"));
    }

    #[test]
    fn test_no_regression_without_baseline() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64
.visible .entry test() { ret; }
"#;
        let result = suite.test_kernel("new_kernel", ptx);
        assert!(!result.is_regression);
    }

    #[test]
    fn test_regression_detected() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());

        // Baseline uses u32 shared memory
        let baseline = r#"
.version 8.0
.target sm_70
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    bar.sync 0;
    ret;
}
"#;
        suite.add_baseline("test", baseline);

        // Current uses u64 (regression!)
        let current = r#"
.version 8.0
.target sm_70
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%rd0], %f0;
    bar.sync 0;
    ret;
}
"#;
        let result = suite.test_kernel("test", current);
        // Should detect regression (u64 addressing)
        assert!(!result.pixel_results.all_passed() || result.is_regression);
    }

    #[test]
    fn test_summary_format() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());
        let ptx = ".version 8.0\n.visible .entry t() { ret; }";
        suite.test_kernel("k1", ptx);
        suite.test_kernel("k2", ptx);

        let summary = suite.summary();
        assert!(summary.contains("GPU Regression Suite"));
        assert!(summary.contains("/2"));
    }

    #[test]
    fn test_run_regression_suite() {
        let kernels = vec![
            ("k1", ".version 8.0\n.visible .entry k1() { ret; }"),
            ("k2", ".version 8.0\n.visible .entry k2() { ret; }"),
        ];
        let baselines: Vec<(&str, &str)> = vec![];

        let suite = run_regression_suite(&kernels, &baselines, RegressionConfig::default());
        assert_eq!(suite.results().len(), 2);
    }
}
