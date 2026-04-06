//! Kernel Pixel Tests: Atomic verification units for GPU kernels
//!
//! Each "pixel" test verifies a single, indivisible property of a kernel.

// Static regexes are always valid - compile-time constants
#![allow(clippy::unwrap_used)]

use super::ptx_analysis::{PtxBugClass, PtxValidationResult};
use std::time::{Duration, Instant};

/// Configuration for kernel pixel tests
#[derive(Debug, Clone)]
pub struct KernelPixelConfig {
    /// Test degenerate dimensions (tile=1, seq=1)
    pub test_degenerate_dims: bool,
    /// Test boundary conditions
    pub test_boundaries: bool,
    /// Strict PTX validation
    pub strict_ptx: bool,
    /// Timeout for individual pixel tests
    pub timeout: Duration,
}

impl Default for KernelPixelConfig {
    fn default() -> Self {
        Self {
            test_degenerate_dims: true,
            test_boundaries: true,
            strict_ptx: true,
            timeout: Duration::from_secs(5),
        }
    }
}

/// Result of a single GPU pixel test
#[derive(Debug, Clone)]
pub struct GpuPixelResult {
    /// Test name
    pub name: String,
    /// Did test pass?
    pub passed: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Test duration
    pub duration: Duration,
    /// Bug class if applicable
    pub bug_class: Option<PtxBugClass>,
}

impl GpuPixelResult {
    /// Create passing result
    #[must_use]
    pub fn pass(name: &str, duration: Duration) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            error: None,
            duration,
            bug_class: None,
        }
    }

    /// Create failing result
    #[must_use]
    pub fn fail(name: &str, error: &str, duration: Duration) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            error: Some(error.to_string()),
            duration,
            bug_class: None,
        }
    }

    /// Create failing result with bug class
    #[must_use]
    pub fn fail_with_bug(name: &str, error: &str, bug: PtxBugClass, duration: Duration) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            error: Some(error.to_string()),
            duration,
            bug_class: Some(bug),
        }
    }

    /// Create from PTX validation result
    #[must_use]
    pub fn from_ptx_validation(result: &PtxValidationResult) -> Self {
        let start = Instant::now();
        if result.is_valid() {
            Self::pass("ptx_validation", start.elapsed())
        } else {
            let first_bug = result.bugs.first();
            let error = first_bug
                .map(|b| format!("{}: {}", b.class, b.message))
                .unwrap_or_else(|| "Unknown PTX error".to_string());
            let bug_class = first_bug.map(|b| b.class.clone());
            Self {
                name: "ptx_validation".to_string(),
                passed: false,
                error: Some(error),
                duration: start.elapsed(),
                bug_class,
            }
        }
    }
}

/// A single GPU pixel test
#[derive(Debug, Clone)]
pub struct GpuPixelTest {
    /// Test name
    pub name: String,
    /// Test description
    pub description: String,
    /// Bug class this test catches
    pub catches: PtxBugClass,
}

impl GpuPixelTest {
    /// Create new pixel test
    #[must_use]
    pub fn new(name: &str, description: &str, catches: PtxBugClass) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            catches,
        }
    }
}

/// Suite of GPU pixel tests for a kernel
#[derive(Debug, Clone)]
pub struct GpuPixelTestSuite {
    /// Kernel name being tested
    pub kernel_name: String,
    /// Test results
    pub results: Vec<GpuPixelResult>,
    /// Total duration
    pub duration: Duration,
}

impl GpuPixelTestSuite {
    /// Create new suite for kernel
    #[must_use]
    pub fn new(kernel_name: &str) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            results: Vec::new(),
            duration: Duration::ZERO,
        }
    }

    /// Add a test result
    pub fn add_result(&mut self, result: GpuPixelResult) {
        self.duration += result.duration;
        self.results.push(result);
    }

    /// Check if all tests passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Get passed count
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Get failed count
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Get failures
    #[must_use]
    pub fn failures(&self) -> Vec<&GpuPixelResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Run kernel-specific pixel tests
    pub fn run_kernel_pixels(&mut self, ptx: &str, config: &KernelPixelConfig) {
        let start = Instant::now();

        // Pixel: shared memory addressing
        self.add_result(self.pixel_shared_mem_addressing(ptx));

        // Pixel: kernel entry exists
        self.add_result(self.pixel_kernel_entry_exists(ptx));

        // Pixel: loop structure
        self.add_result(self.pixel_loop_structure(ptx, config.strict_ptx));

        // Pixel: barrier synchronization (if shared memory used)
        if ptx.contains(".shared") {
            self.add_result(self.pixel_barrier_sync(ptx));
        }

        self.duration = start.elapsed();
    }

    /// Pixel test: shared memory uses 32-bit addressing
    fn pixel_shared_mem_addressing(&self, ptx: &str) -> GpuPixelResult {
        let start = Instant::now();
        let regex = regex::Regex::new(r"[sl]t\.shared\.[^\[]+\[%rd\d+\]").unwrap();

        if regex.is_match(ptx) {
            GpuPixelResult::fail_with_bug(
                "shared_mem_u32_addressing",
                "Shared memory uses 64-bit addressing (should be 32-bit)",
                PtxBugClass::SharedMemU64Addressing,
                start.elapsed(),
            )
        } else {
            GpuPixelResult::pass("shared_mem_u32_addressing", start.elapsed())
        }
    }

    /// Pixel test: kernel entry point exists
    fn pixel_kernel_entry_exists(&self, ptx: &str) -> GpuPixelResult {
        let start = Instant::now();
        let regex = regex::Regex::new(r"\.visible\s+\.entry\s+\w+").unwrap();

        if regex.is_match(ptx) {
            GpuPixelResult::pass("kernel_entry_exists", start.elapsed())
        } else {
            GpuPixelResult::fail_with_bug(
                "kernel_entry_exists",
                "No kernel entry point found",
                PtxBugClass::MissingEntryPoint,
                start.elapsed(),
            )
        }
    }

    /// Pixel test: loop branches go to start, not end
    fn pixel_loop_structure(&self, ptx: &str, strict: bool) -> GpuPixelResult {
        let start = Instant::now();

        if !strict {
            return GpuPixelResult::pass("loop_structure", start.elapsed());
        }

        // Check for unconditional branches to _end labels
        let branch_regex = regex::Regex::new(r"^\s+bra\s+(\w*_end\w*);").unwrap();
        for line in ptx.lines() {
            if branch_regex.is_match(line) && !line.trim().starts_with('@') {
                return GpuPixelResult::fail_with_bug(
                    "loop_structure",
                    "Unconditional branch to loop end (should branch to start)",
                    PtxBugClass::LoopBranchToEnd,
                    start.elapsed(),
                );
            }
        }

        GpuPixelResult::pass("loop_structure", start.elapsed())
    }

    /// Pixel test: barrier sync present when using shared memory
    fn pixel_barrier_sync(&self, ptx: &str) -> GpuPixelResult {
        let start = Instant::now();

        if ptx.contains("bar.sync") {
            GpuPixelResult::pass("barrier_sync", start.elapsed())
        } else {
            GpuPixelResult::fail_with_bug(
                "barrier_sync",
                "Shared memory used but no bar.sync found",
                PtxBugClass::MissingBarrierSync,
                start.elapsed(),
            )
        }
    }

    /// Generate summary report
    #[must_use]
    pub fn summary(&self) -> String {
        let status = if self.all_passed() { "PASS" } else { "FAIL" };
        format!(
            "[{}] {} - {}/{} passed ({:?})",
            status,
            self.kernel_name,
            self.passed_count(),
            self.results.len(),
            self.duration
        )
    }
}

/// Standard GPU pixel tests catalog
pub fn standard_pixel_tests() -> Vec<GpuPixelTest> {
    vec![
        GpuPixelTest::new(
            "shared_mem_u32_addressing",
            "Verify shared memory uses 32-bit addressing",
            PtxBugClass::SharedMemU64Addressing,
        ),
        GpuPixelTest::new(
            "loop_branch_to_start",
            "Verify loop branches go to start label, not end",
            PtxBugClass::LoopBranchToEnd,
        ),
        GpuPixelTest::new(
            "barrier_sync_present",
            "Verify barrier sync exists when using shared memory",
            PtxBugClass::MissingBarrierSync,
        ),
        GpuPixelTest::new(
            "kernel_entry_exists",
            "Verify kernel has entry point",
            PtxBugClass::MissingEntryPoint,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suite_all_passed() {
        let mut suite = GpuPixelTestSuite::new("test_kernel");
        suite.add_result(GpuPixelResult::pass("test1", Duration::from_millis(1)));
        suite.add_result(GpuPixelResult::pass("test2", Duration::from_millis(2)));
        assert!(suite.all_passed());
        assert_eq!(suite.passed_count(), 2);
        assert_eq!(suite.failed_count(), 0);
    }

    #[test]
    fn test_suite_has_failure() {
        let mut suite = GpuPixelTestSuite::new("test_kernel");
        suite.add_result(GpuPixelResult::pass("test1", Duration::from_millis(1)));
        suite.add_result(GpuPixelResult::fail(
            "test2",
            "error",
            Duration::from_millis(2),
        ));
        assert!(!suite.all_passed());
        assert_eq!(suite.passed_count(), 1);
        assert_eq!(suite.failed_count(), 1);
    }

    #[test]
    fn test_pixel_shared_mem_u64_fails() {
        let ptx = "st.shared.f32 [%rd5], %f0;";
        let suite = GpuPixelTestSuite::new("test");
        let result = suite.pixel_shared_mem_addressing(ptx);
        assert!(!result.passed);
        assert_eq!(result.bug_class, Some(PtxBugClass::SharedMemU64Addressing));
    }

    #[test]
    fn test_pixel_shared_mem_u32_passes() {
        let ptx = "st.shared.f32 [%r5], %f0;";
        let suite = GpuPixelTestSuite::new("test");
        let result = suite.pixel_shared_mem_addressing(ptx);
        assert!(result.passed);
    }

    #[test]
    fn test_standard_pixel_tests() {
        let tests = standard_pixel_tests();
        assert!(!tests.is_empty());
        assert!(tests.iter().any(|t| t.name == "shared_mem_u32_addressing"));
    }

    #[test]
    fn test_summary_format() {
        let mut suite = GpuPixelTestSuite::new("gemm_tiled");
        suite.add_result(GpuPixelResult::pass("test1", Duration::from_millis(1)));
        let summary = suite.summary();
        assert!(summary.contains("PASS"));
        assert!(summary.contains("gemm_tiled"));
    }
}
