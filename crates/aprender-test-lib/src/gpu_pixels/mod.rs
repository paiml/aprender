//! GPU Pixel Testing: Atomic verification of CUDA kernel correctness
//!
//! Inspired by pixel-level GUI testing, GPU pixel testing verifies the smallest
//! testable units of GPU kernel behavior to catch bugs like:
//! - Shared memory addressing (u64 vs u32)
//! - Loop branch direction (START vs END)
//! - Kernel name mismatches
//! - Tile/thread bounds violations
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    GPU Pixel Testing                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │   ┌────────────┐    ┌────────────┐    ┌────────────┐            │
//! │   │ PTX Static │    │ Kernel     │    │ Regression │            │
//! │   │ Analysis   │───►│ Pixel      │───►│ Detection  │            │
//! │   │            │    │ Tests      │    │            │            │
//! │   └────────────┘    └────────────┘    └────────────┘            │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod kernel_pixels;
mod ptx_analysis;
mod regression;

pub use kernel_pixels::{
    standard_pixel_tests, GpuPixelResult, GpuPixelTest, GpuPixelTestSuite, KernelPixelConfig,
};
pub use ptx_analysis::{PtxAnalyzer, PtxBug, PtxBugClass, PtxValidationResult};
pub use regression::{
    run_regression_suite, GpuRegressionSuite, RegressionConfig, RegressionResult,
};

/// Quick validation of PTX for common GPU kernel bugs
///
/// # Example
/// ```ignore
/// use probar::gpu_pixels::validate_ptx;
///
/// let ptx = kernel.emit_ptx();
/// let result = validate_ptx(&ptx);
/// assert!(result.is_valid(), "PTX bugs: {:?}", result.bugs);
/// ```
#[must_use]
pub fn validate_ptx(ptx: &str) -> PtxValidationResult {
    PtxAnalyzer::default().analyze(ptx)
}

/// Run all GPU pixel tests for a kernel
///
/// # Example
/// ```ignore
/// use probar::gpu_pixels::run_kernel_pixels;
///
/// let results = run_kernel_pixels("gemm_tiled", &ptx, &config);
/// assert!(results.all_passed());
/// ```
pub fn run_kernel_pixels(
    kernel_name: &str,
    ptx: &str,
    config: &KernelPixelConfig,
) -> GpuPixelTestSuite {
    let mut suite = GpuPixelTestSuite::new(kernel_name);

    // PTX static analysis
    let ptx_result = validate_ptx(ptx);
    suite.add_result(GpuPixelResult::from_ptx_validation(&ptx_result));

    // Kernel-specific pixel tests
    suite.run_kernel_pixels(ptx, config);

    suite
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_ptx_empty() {
        let result = validate_ptx("");
        assert!(!result.is_valid());
    }

    #[test]
    fn test_validate_ptx_minimal_valid() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry test_kernel() {
    ret;
}
"#;
        let result = validate_ptx(ptx);
        assert!(result.is_valid());
    }

    #[test]
    fn test_shared_memory_u64_bug_detected() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry buggy_kernel() {
    .reg .u64 %rd<5>;
    .reg .f32 %f<2>;
    .shared .align 16 .b8 smem[4096];

    st.shared.f32 [%rd0], %f0;
    ret;
}
"#;
        let result = validate_ptx(ptx);
        assert!(!result.is_valid());
        assert!(result
            .bugs
            .iter()
            .any(|b| matches!(b.class, PtxBugClass::SharedMemU64Addressing)));
    }

    #[test]
    fn test_shared_memory_u32_valid() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry valid_kernel() {
    .reg .u32 %r<5>;
    .reg .f32 %f<2>;
    .shared .align 16 .b8 smem[4096];

    st.shared.f32 [%r0], %f0;
    ret;
}
"#;
        let result = validate_ptx(ptx);
        // Should not have shared memory addressing bug
        assert!(!result
            .bugs
            .iter()
            .any(|b| matches!(b.class, PtxBugClass::SharedMemU64Addressing)));
    }

    // ========================================================================
    // Additional tests for run_kernel_pixels function
    // ========================================================================

    #[test]
    fn test_run_kernel_pixels_valid_kernel() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry gemm_tiled() {
    ret;
}
"#;
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("gemm_tiled", ptx, &config);

        assert_eq!(suite.kernel_name, "gemm_tiled");
        assert!(suite.all_passed());
        assert_eq!(suite.failed_count(), 0);
    }

    #[test]
    fn test_run_kernel_pixels_with_u64_bug() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry buggy() {
    .shared .b8 smem[1024];
    st.shared.f32 [%rd0], %f0;
    ret;
}
"#;
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("buggy", ptx, &config);

        // Should detect bugs
        assert!(!suite.all_passed());
        assert!(suite.failed_count() > 0);
    }

    #[test]
    fn test_run_kernel_pixels_no_entry_point() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64
"#;
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("missing_entry", ptx, &config);

        // Should fail PTX validation (no entry point)
        assert!(!suite.all_passed());
        // Verify it detected the missing entry point
        let failures = suite.failures();
        assert!(!failures.is_empty());
    }

    #[test]
    fn test_run_kernel_pixels_with_shared_memory_needs_barrier() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry test_kernel() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    ret;
}
"#;
        // With strict mode, missing barrier should be detected
        let config = KernelPixelConfig {
            strict_ptx: true,
            ..Default::default()
        };
        let suite = run_kernel_pixels("test_kernel", ptx, &config);

        // Should fail due to missing barrier
        assert!(!suite.all_passed());
    }

    #[test]
    fn test_run_kernel_pixels_with_barrier_sync() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry test_kernel() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    bar.sync 0;
    ret;
}
"#;
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("test_kernel", ptx, &config);

        // Should pass all tests (valid shared mem addressing + barrier)
        assert!(suite.all_passed());
    }

    #[test]
    fn test_run_kernel_pixels_non_strict_mode() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry test_kernel() {
    ret;
}
"#;
        let config = KernelPixelConfig {
            test_degenerate_dims: false,
            test_boundaries: false,
            strict_ptx: false,
            timeout: std::time::Duration::from_secs(1),
        };
        let suite = run_kernel_pixels("test_kernel", ptx, &config);

        assert!(suite.all_passed());
    }

    // ========================================================================
    // Tests for KernelPixelConfig
    // ========================================================================

    #[test]
    fn test_kernel_pixel_config_default() {
        let config = KernelPixelConfig::default();
        assert!(config.test_degenerate_dims);
        assert!(config.test_boundaries);
        assert!(config.strict_ptx);
        assert_eq!(config.timeout, std::time::Duration::from_secs(5));
    }

    #[test]
    fn test_kernel_pixel_config_custom() {
        let config = KernelPixelConfig {
            test_degenerate_dims: false,
            test_boundaries: false,
            strict_ptx: false,
            timeout: std::time::Duration::from_millis(500),
        };
        assert!(!config.test_degenerate_dims);
        assert!(!config.test_boundaries);
        assert!(!config.strict_ptx);
        assert_eq!(config.timeout, std::time::Duration::from_millis(500));
    }

    #[test]
    fn test_kernel_pixel_config_clone() {
        let config = KernelPixelConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.test_degenerate_dims, config.test_degenerate_dims);
        assert_eq!(cloned.test_boundaries, config.test_boundaries);
        assert_eq!(cloned.strict_ptx, config.strict_ptx);
        assert_eq!(cloned.timeout, config.timeout);
    }

    #[test]
    fn test_kernel_pixel_config_debug() {
        let config = KernelPixelConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("KernelPixelConfig"));
        assert!(debug_str.contains("test_degenerate_dims"));
        assert!(debug_str.contains("strict_ptx"));
    }

    // ========================================================================
    // Tests for GpuPixelResult
    // ========================================================================

    #[test]
    fn test_gpu_pixel_result_pass() {
        let result = GpuPixelResult::pass("test", std::time::Duration::from_millis(10));
        assert!(result.passed);
        assert!(result.error.is_none());
        assert!(result.bug_class.is_none());
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_gpu_pixel_result_fail() {
        let result = GpuPixelResult::fail(
            "test",
            "error message",
            std::time::Duration::from_millis(10),
        );
        assert!(!result.passed);
        assert_eq!(result.error, Some("error message".to_string()));
        assert!(result.bug_class.is_none());
    }

    #[test]
    fn test_gpu_pixel_result_fail_with_bug() {
        let result = GpuPixelResult::fail_with_bug(
            "test",
            "shared mem bug",
            PtxBugClass::SharedMemU64Addressing,
            std::time::Duration::from_millis(10),
        );
        assert!(!result.passed);
        assert!(result.error.is_some());
        assert_eq!(result.bug_class, Some(PtxBugClass::SharedMemU64Addressing));
    }

    #[test]
    fn test_gpu_pixel_result_from_valid_ptx_validation() {
        let ptx = r#"
.version 8.0
.target sm_70
.visible .entry test() { ret; }
"#;
        let validation = validate_ptx(ptx);
        let result = GpuPixelResult::from_ptx_validation(&validation);
        assert!(result.passed);
        assert_eq!(result.name, "ptx_validation");
    }

    #[test]
    fn test_gpu_pixel_result_from_invalid_ptx_validation() {
        // Empty PTX is invalid (no entry point)
        let validation = validate_ptx("");
        let result = GpuPixelResult::from_ptx_validation(&validation);
        assert!(!result.passed);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_gpu_pixel_result_from_ptx_with_bug() {
        let ptx = "st.shared.f32 [%rd0], %f0;";
        let validation = validate_ptx(ptx);
        let result = GpuPixelResult::from_ptx_validation(&validation);
        assert!(!result.passed);
        // Should have the bug class set
        assert!(result.bug_class.is_some());
    }

    // ========================================================================
    // Tests for GpuPixelTestSuite
    // ========================================================================

    #[test]
    fn test_gpu_pixel_test_suite_new() {
        let suite = GpuPixelTestSuite::new("my_kernel");
        assert_eq!(suite.kernel_name, "my_kernel");
        assert!(suite.results.is_empty());
        assert_eq!(suite.duration, std::time::Duration::ZERO);
    }

    #[test]
    fn test_gpu_pixel_test_suite_add_result() {
        let mut suite = GpuPixelTestSuite::new("test");
        let result = GpuPixelResult::pass("t1", std::time::Duration::from_millis(5));
        suite.add_result(result);
        assert_eq!(suite.results.len(), 1);
        assert_eq!(suite.duration, std::time::Duration::from_millis(5));
    }

    #[test]
    fn test_gpu_pixel_test_suite_failures() {
        let mut suite = GpuPixelTestSuite::new("test");
        suite.add_result(GpuPixelResult::pass(
            "t1",
            std::time::Duration::from_millis(1),
        ));
        suite.add_result(GpuPixelResult::fail(
            "t2",
            "error",
            std::time::Duration::from_millis(1),
        ));
        suite.add_result(GpuPixelResult::fail(
            "t3",
            "error",
            std::time::Duration::from_millis(1),
        ));

        let failures = suite.failures();
        assert_eq!(failures.len(), 2);
        assert!(!failures[0].passed);
        assert!(!failures[1].passed);
    }

    #[test]
    fn test_gpu_pixel_test_suite_summary_pass() {
        let mut suite = GpuPixelTestSuite::new("gemm");
        suite.add_result(GpuPixelResult::pass(
            "t1",
            std::time::Duration::from_millis(1),
        ));
        let summary = suite.summary();
        assert!(summary.contains("[PASS]"));
        assert!(summary.contains("gemm"));
        assert!(summary.contains("1/1"));
    }

    #[test]
    fn test_gpu_pixel_test_suite_summary_fail() {
        let mut suite = GpuPixelTestSuite::new("buggy");
        suite.add_result(GpuPixelResult::fail(
            "t1",
            "error",
            std::time::Duration::from_millis(1),
        ));
        let summary = suite.summary();
        assert!(summary.contains("[FAIL]"));
        assert!(summary.contains("buggy"));
        assert!(summary.contains("0/1"));
    }

    #[test]
    fn test_gpu_pixel_test_suite_run_kernel_pixels_without_shared() {
        let mut suite = GpuPixelTestSuite::new("test");
        let ptx = r#"
.version 8.0
.target sm_70
.visible .entry test() { ret; }
"#;
        let config = KernelPixelConfig::default();
        suite.run_kernel_pixels(ptx, &config);

        // Should have run: shared_mem_addressing, kernel_entry_exists, loop_structure
        // (no barrier_sync because no .shared keyword)
        assert!(suite.results.len() >= 3);
    }

    #[test]
    fn test_gpu_pixel_test_suite_run_kernel_pixels_with_shared() {
        let mut suite = GpuPixelTestSuite::new("test");
        let ptx = r#"
.version 8.0
.target sm_70
.visible .entry test() {
    .shared .b8 smem[1024];
    bar.sync 0;
    ret;
}
"#;
        let config = KernelPixelConfig::default();
        suite.run_kernel_pixels(ptx, &config);

        // Should have run: shared_mem_addressing, kernel_entry_exists, loop_structure, barrier_sync
        assert!(suite.results.len() >= 4);
    }

    // ========================================================================
    // Tests for GpuPixelTest
    // ========================================================================

    #[test]
    fn test_gpu_pixel_test_new() {
        let test = GpuPixelTest::new(
            "my_test",
            "Tests something important",
            PtxBugClass::MissingBarrierSync,
        );
        assert_eq!(test.name, "my_test");
        assert_eq!(test.description, "Tests something important");
        assert_eq!(test.catches, PtxBugClass::MissingBarrierSync);
    }

    #[test]
    fn test_gpu_pixel_test_clone() {
        let test = GpuPixelTest::new("test", "desc", PtxBugClass::InvalidSyntax);
        let cloned = test.clone();
        assert_eq!(cloned.name, test.name);
        assert_eq!(cloned.description, test.description);
        assert_eq!(cloned.catches, test.catches);
    }

    #[test]
    fn test_gpu_pixel_test_debug() {
        let test = GpuPixelTest::new("test", "desc", PtxBugClass::LoopBranchToEnd);
        let debug_str = format!("{:?}", test);
        assert!(debug_str.contains("GpuPixelTest"));
        assert!(debug_str.contains("test"));
    }

    // ========================================================================
    // Tests for standard_pixel_tests
    // ========================================================================

    #[test]
    fn test_standard_pixel_tests_count() {
        let tests = standard_pixel_tests();
        assert_eq!(tests.len(), 4);
    }

    #[test]
    fn test_standard_pixel_tests_all_variants() {
        let tests = standard_pixel_tests();
        let names: Vec<&str> = tests.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"shared_mem_u32_addressing"));
        assert!(names.contains(&"loop_branch_to_start"));
        assert!(names.contains(&"barrier_sync_present"));
        assert!(names.contains(&"kernel_entry_exists"));
    }

    // ========================================================================
    // Tests for PtxBugClass Display
    // ========================================================================

    #[test]
    fn test_ptx_bug_class_display_shared_mem() {
        assert_eq!(
            format!("{}", PtxBugClass::SharedMemU64Addressing),
            "shared_mem_u64"
        );
    }

    #[test]
    fn test_ptx_bug_class_display_loop_branch() {
        assert_eq!(
            format!("{}", PtxBugClass::LoopBranchToEnd),
            "loop_branch_to_end"
        );
    }

    #[test]
    fn test_ptx_bug_class_display_missing_barrier() {
        assert_eq!(
            format!("{}", PtxBugClass::MissingBarrierSync),
            "missing_barrier"
        );
    }

    #[test]
    fn test_ptx_bug_class_display_non_inplace() {
        assert_eq!(
            format!("{}", PtxBugClass::NonInPlaceLoopAccumulator),
            "non_inplace_accum"
        );
    }

    #[test]
    fn test_ptx_bug_class_display_invalid_syntax() {
        assert_eq!(format!("{}", PtxBugClass::InvalidSyntax), "invalid_syntax");
    }

    #[test]
    fn test_ptx_bug_class_display_missing_entry() {
        assert_eq!(
            format!("{}", PtxBugClass::MissingEntryPoint),
            "missing_entry"
        );
    }

    // ========================================================================
    // Tests for PtxValidationResult
    // ========================================================================

    #[test]
    fn test_ptx_validation_result_is_valid_with_kernels_no_bugs() {
        let result = PtxValidationResult {
            bugs: vec![],
            kernel_names: vec!["kernel1".to_string()],
            lines_analyzed: 10,
        };
        assert!(result.is_valid());
    }

    #[test]
    fn test_ptx_validation_result_invalid_no_kernels() {
        let result = PtxValidationResult {
            bugs: vec![],
            kernel_names: vec![],
            lines_analyzed: 10,
        };
        assert!(!result.is_valid());
    }

    #[test]
    fn test_ptx_validation_result_invalid_with_bugs() {
        let result = PtxValidationResult {
            bugs: vec![ptx_analysis::PtxBug {
                class: PtxBugClass::InvalidSyntax,
                line: 1,
                instruction: "bad".to_string(),
                message: "error".to_string(),
            }],
            kernel_names: vec!["kernel".to_string()],
            lines_analyzed: 10,
        };
        assert!(!result.is_valid());
    }

    #[test]
    fn test_ptx_validation_result_bug_count() {
        let result = PtxValidationResult {
            bugs: vec![
                ptx_analysis::PtxBug {
                    class: PtxBugClass::SharedMemU64Addressing,
                    line: 1,
                    instruction: "st".to_string(),
                    message: "err".to_string(),
                },
                ptx_analysis::PtxBug {
                    class: PtxBugClass::SharedMemU64Addressing,
                    line: 2,
                    instruction: "st".to_string(),
                    message: "err".to_string(),
                },
                ptx_analysis::PtxBug {
                    class: PtxBugClass::MissingBarrierSync,
                    line: 3,
                    instruction: String::new(),
                    message: "err".to_string(),
                },
            ],
            kernel_names: vec!["k".to_string()],
            lines_analyzed: 10,
        };
        assert_eq!(result.bug_count(&PtxBugClass::SharedMemU64Addressing), 2);
        assert_eq!(result.bug_count(&PtxBugClass::MissingBarrierSync), 1);
        assert_eq!(result.bug_count(&PtxBugClass::InvalidSyntax), 0);
    }

    #[test]
    fn test_ptx_validation_result_has_bug() {
        let result = PtxValidationResult {
            bugs: vec![ptx_analysis::PtxBug {
                class: PtxBugClass::LoopBranchToEnd,
                line: 1,
                instruction: "bra".to_string(),
                message: "err".to_string(),
            }],
            kernel_names: vec!["k".to_string()],
            lines_analyzed: 5,
        };
        assert!(result.has_bug(&PtxBugClass::LoopBranchToEnd));
        assert!(!result.has_bug(&PtxBugClass::SharedMemU64Addressing));
    }

    // ========================================================================
    // Tests for PtxAnalyzer
    // ========================================================================

    #[test]
    fn test_ptx_analyzer_default() {
        let analyzer = PtxAnalyzer::default();
        assert!(!analyzer.strict);
    }

    #[test]
    fn test_ptx_analyzer_strict() {
        let analyzer = PtxAnalyzer::strict();
        assert!(analyzer.strict);
    }

    #[test]
    fn test_ptx_analyzer_analyze_empty() {
        let analyzer = PtxAnalyzer::default();
        let result = analyzer.analyze("");
        assert!(result.kernel_names.is_empty());
        assert!(result.bugs.is_empty()); // Empty string has no bugs (only whitespace/empty)
                                         // lines_analyzed is the count of lines returned by .lines()
                                         // For empty string, .lines() returns 0 elements
        assert_eq!(result.lines_analyzed, 0);
    }

    #[test]
    fn test_ptx_analyzer_analyze_multiple_bugs() {
        let ptx = r#"
.version 8.0
.target sm_70
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%rd0], %f0;
    ret;
}
"#;
        let analyzer = PtxAnalyzer::strict();
        let result = analyzer.analyze(ptx);

        // Should find: SharedMemU64Addressing, MissingBarrierSync
        assert!(result.has_bug(&PtxBugClass::SharedMemU64Addressing));
        assert!(result.has_bug(&PtxBugClass::MissingBarrierSync));
    }

    #[test]
    fn test_ptx_analyzer_ld_shared_detection() {
        let ptx = "ld.shared.f32 %f0, [%r0];";
        let analyzer = PtxAnalyzer::default();
        let result = analyzer.analyze(ptx);
        // ld.shared doesn't match the regex [sl]t\.shared
        assert!(!result.has_bug(&PtxBugClass::SharedMemU64Addressing));
    }

    // ========================================================================
    // Tests for RegressionConfig
    // ========================================================================

    #[test]
    fn test_regression_config_default() {
        let config = RegressionConfig::default();
        assert_eq!(config.baseline_dir, ".gpu_baselines");
        assert!(!config.auto_update);
    }

    #[test]
    fn test_regression_config_custom() {
        let config = RegressionConfig {
            baseline_dir: "/custom/path".to_string(),
            auto_update: true,
            pixel_config: KernelPixelConfig {
                strict_ptx: false,
                ..Default::default()
            },
        };
        assert_eq!(config.baseline_dir, "/custom/path");
        assert!(config.auto_update);
        assert!(!config.pixel_config.strict_ptx);
    }

    #[test]
    fn test_regression_config_clone() {
        let config = RegressionConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.baseline_dir, config.baseline_dir);
        assert_eq!(cloned.auto_update, config.auto_update);
    }

    #[test]
    fn test_regression_config_debug() {
        let config = RegressionConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("RegressionConfig"));
        assert!(debug_str.contains("baseline_dir"));
    }

    // ========================================================================
    // Tests for RegressionResult
    // ========================================================================

    #[test]
    fn test_regression_result_passed_true() {
        let suite = GpuPixelTestSuite::new("test");
        let result = RegressionResult {
            kernel_name: "test".to_string(),
            is_regression: false,
            details: None,
            pixel_results: suite,
            duration: std::time::Duration::from_millis(10),
        };
        assert!(result.passed());
    }

    #[test]
    fn test_regression_result_passed_false_regression() {
        let suite = GpuPixelTestSuite::new("test");
        let result = RegressionResult {
            kernel_name: "test".to_string(),
            is_regression: true,
            details: Some("Regression detected".to_string()),
            pixel_results: suite,
            duration: std::time::Duration::from_millis(10),
        };
        assert!(!result.passed());
    }

    #[test]
    fn test_regression_result_passed_false_pixel_failure() {
        let mut suite = GpuPixelTestSuite::new("test");
        suite.add_result(GpuPixelResult::fail(
            "t1",
            "error",
            std::time::Duration::from_millis(1),
        ));
        let result = RegressionResult {
            kernel_name: "test".to_string(),
            is_regression: false,
            details: None,
            pixel_results: suite,
            duration: std::time::Duration::from_millis(10),
        };
        assert!(!result.passed());
    }

    #[test]
    fn test_regression_result_clone() {
        let suite = GpuPixelTestSuite::new("test");
        let result = RegressionResult {
            kernel_name: "kernel".to_string(),
            is_regression: true,
            details: Some("details".to_string()),
            pixel_results: suite,
            duration: std::time::Duration::from_millis(5),
        };
        let cloned = result.clone();
        assert_eq!(cloned.kernel_name, result.kernel_name);
        assert_eq!(cloned.is_regression, result.is_regression);
        assert_eq!(cloned.details, result.details);
    }

    #[test]
    fn test_regression_result_debug() {
        let suite = GpuPixelTestSuite::new("test");
        let result = RegressionResult {
            kernel_name: "test".to_string(),
            is_regression: false,
            details: None,
            pixel_results: suite,
            duration: std::time::Duration::from_millis(1),
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("RegressionResult"));
        assert!(debug_str.contains("kernel_name"));
    }

    // ========================================================================
    // Tests for GpuRegressionSuite
    // ========================================================================

    #[test]
    fn test_gpu_regression_suite_new() {
        let suite = GpuRegressionSuite::new(RegressionConfig::default());
        assert!(suite.results().is_empty());
        assert!(suite.all_passed());
    }

    #[test]
    fn test_gpu_regression_suite_add_baseline() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());
        suite.add_baseline("kernel1", "ptx content");
        // Baseline is added internally
    }

    #[test]
    fn test_gpu_regression_suite_test_kernel_no_baseline() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());
        let ptx = ".version 8.0\n.visible .entry k() { ret; }";
        let result = suite.test_kernel("new_kernel", ptx);
        assert!(!result.is_regression);
    }

    #[test]
    fn test_gpu_regression_suite_test_kernel_with_matching_baseline() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());
        let ptx = r#"
.version 8.0
.visible .entry test() { ret; }
"#;
        suite.add_baseline("test", ptx);
        let result = suite.test_kernel("test", ptx);
        assert!(!result.is_regression);
    }

    #[test]
    fn test_gpu_regression_suite_barrier_removed_regression() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());

        let baseline = r#"
.version 8.0
.visible .entry test() {
    .shared .b8 smem[1024];
    bar.sync 0;
    ret;
}
"#;
        let current = r#"
.version 8.0
.visible .entry test() {
    .shared .b8 smem[1024];
    ret;
}
"#;
        suite.add_baseline("test", baseline);
        let result = suite.test_kernel("test", current);
        assert!(result.is_regression);
        assert!(result
            .details
            .as_ref()
            .is_some_and(|d| d.contains("barrier")));
    }

    #[test]
    fn test_gpu_regression_suite_kernel_name_changed_regression() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());

        let baseline = ".version 8.0\n.visible .entry old_name() { ret; }";
        let current = ".version 8.0\n.visible .entry new_name() { ret; }";

        suite.add_baseline("test", baseline);
        let result = suite.test_kernel("test", current);
        assert!(result.is_regression);
        assert!(result
            .details
            .as_ref()
            .is_some_and(|d| d.contains("kernel name")));
    }

    #[test]
    fn test_gpu_regression_suite_u64_addressing_regression() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());

        let baseline = r#"
.version 8.0
.visible .entry test() {
    st.shared.f32 [%r0], %f0;
    ret;
}
"#;
        let current = r#"
.version 8.0
.visible .entry test() {
    st.shared.f32 [%rd0], %f0;
    ret;
}
"#;
        suite.add_baseline("test", baseline);
        let result = suite.test_kernel("test", current);
        // Should detect regression (switched from u32 to u64)
        assert!(result.is_regression);
    }

    #[test]
    fn test_gpu_regression_suite_all_passed() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());
        let ptx = ".version 8.0\n.visible .entry k() { ret; }";
        suite.test_kernel("k1", ptx);
        suite.test_kernel("k2", ptx);
        assert!(suite.all_passed());
    }

    #[test]
    fn test_gpu_regression_suite_summary() {
        let mut suite = GpuRegressionSuite::new(RegressionConfig::default());
        let ptx = ".version 8.0\n.visible .entry k() { ret; }";
        suite.test_kernel("k1", ptx);
        suite.test_kernel("k2", ptx);
        let summary = suite.summary();
        assert!(summary.contains("GPU Regression Suite"));
        assert!(summary.contains("2/2"));
        assert!(summary.contains("0 regressions"));
    }

    #[test]
    fn test_gpu_regression_suite_debug() {
        let suite = GpuRegressionSuite::new(RegressionConfig::default());
        let debug_str = format!("{:?}", suite);
        assert!(debug_str.contains("GpuRegressionSuite"));
    }

    // ========================================================================
    // Tests for run_regression_suite function
    // ========================================================================

    #[test]
    fn test_run_regression_suite_empty() {
        let kernels: Vec<(&str, &str)> = vec![];
        let baselines: Vec<(&str, &str)> = vec![];
        let suite = run_regression_suite(&kernels, &baselines, RegressionConfig::default());
        assert!(suite.results().is_empty());
        assert!(suite.all_passed());
    }

    #[test]
    fn test_run_regression_suite_multiple_kernels() {
        let kernels = vec![
            ("k1", ".version 8.0\n.visible .entry k1() { ret; }"),
            ("k2", ".version 8.0\n.visible .entry k2() { ret; }"),
            ("k3", ".version 8.0\n.visible .entry k3() { ret; }"),
        ];
        let baselines: Vec<(&str, &str)> = vec![];
        let suite = run_regression_suite(&kernels, &baselines, RegressionConfig::default());
        assert_eq!(suite.results().len(), 3);
    }

    #[test]
    fn test_run_regression_suite_with_baselines() {
        let baselines = vec![("k1", ".version 8.0\n.visible .entry k1() { ret; }")];
        let kernels = vec![("k1", ".version 8.0\n.visible .entry k1() { ret; }")];
        let suite = run_regression_suite(&kernels, &baselines, RegressionConfig::default());
        assert!(!suite.results()[0].is_regression);
    }

    #[test]
    fn test_run_regression_suite_with_custom_config() {
        let config = RegressionConfig {
            baseline_dir: "/tmp/baselines".to_string(),
            auto_update: true,
            pixel_config: KernelPixelConfig {
                strict_ptx: false,
                ..Default::default()
            },
        };
        let kernels = vec![("k", ".version 8.0\n.visible .entry k() { ret; }")];
        let baselines: Vec<(&str, &str)> = vec![];
        let suite = run_regression_suite(&kernels, &baselines, config);
        assert_eq!(suite.results().len(), 1);
    }

    // ========================================================================
    // Tests for pixel_kernel_entry_exists (via run_kernel_pixels)
    // ========================================================================

    #[test]
    fn test_pixel_kernel_entry_exists_missing() {
        let ptx = ".version 8.0\n.target sm_70\n// no entry point";
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("test", ptx, &config);

        // Should fail because no entry point
        let failures = suite.failures();
        assert!(!failures.is_empty());
        assert!(failures
            .iter()
            .any(|f| f.name.contains("entry")
                || f.error.as_ref().is_some_and(|e| e.contains("entry"))));
    }

    #[test]
    fn test_pixel_kernel_entry_exists_present() {
        let ptx = ".version 8.0\n.visible .entry my_kernel() { ret; }";
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("my_kernel", ptx, &config);

        // Should pass - entry point exists
        assert!(suite.all_passed());
    }

    // ========================================================================
    // Tests for pixel_loop_structure
    // ========================================================================

    #[test]
    fn test_pixel_loop_structure_strict_true_no_loop() {
        let ptx = ".version 8.0\n.visible .entry test() { ret; }";
        let config = KernelPixelConfig {
            strict_ptx: true,
            ..Default::default()
        };
        let suite = run_kernel_pixels("test", ptx, &config);
        // No loop = should pass
        assert!(suite.all_passed());
    }

    #[test]
    fn test_pixel_loop_structure_strict_false() {
        let ptx = r#"
.version 8.0
.visible .entry test() {
    bra some_end;
    ret;
}
"#;
        let config = KernelPixelConfig {
            strict_ptx: false,
            ..Default::default()
        };
        let suite = run_kernel_pixels("test", ptx, &config);
        // Non-strict mode should pass even with potential loop issues
        assert!(suite.all_passed());
    }

    // ========================================================================
    // Tests for pixel_barrier_sync
    // ========================================================================

    #[test]
    fn test_pixel_barrier_sync_present() {
        let ptx = r#"
.version 8.0
.visible .entry test() {
    .shared .b8 smem[1024];
    bar.sync 0;
    ret;
}
"#;
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("test", ptx, &config);
        assert!(suite.all_passed());
    }

    #[test]
    fn test_pixel_barrier_sync_missing() {
        let ptx = r#"
.version 8.0
.visible .entry test() {
    .shared .b8 smem[1024];
    ret;
}
"#;
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("test", ptx, &config);

        // Should fail because no bar.sync with shared memory
        assert!(!suite.all_passed());
        let failures = suite.failures();
        assert!(failures
            .iter()
            .any(|f| f.name.contains("barrier")
                || f.bug_class == Some(PtxBugClass::MissingBarrierSync)));
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn test_validate_ptx_whitespace_only() {
        let result = validate_ptx("   \n\t\n   ");
        assert!(!result.is_valid());
    }

    #[test]
    fn test_run_kernel_pixels_complex_valid() {
        let ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry complex_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 size
) {
    .reg .u32 %r<10>;
    .reg .u64 %rd<5>;
    .reg .f32 %f<20>;
    .shared .align 16 .b8 tile[4096];

    // Load
    ld.param.u64 %rd0, [input_ptr];
    ld.param.u64 %rd1, [output_ptr];

    // Compute with shared memory (u32 addressing)
    st.shared.f32 [%r0], %f0;
    bar.sync 0;
    ld.shared.f32 %f1, [%r1];

    // Store result
    st.global.f32 [%rd1], %f1;

    ret;
}
"#;
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("complex_kernel", ptx, &config);
        assert!(suite.all_passed());
    }

    #[test]
    fn test_run_kernel_pixels_multiple_kernels_in_ptx() {
        let ptx = r#"
.version 8.0
.target sm_70

.visible .entry kernel_a() { ret; }
.visible .entry kernel_b() { ret; }
.visible .entry kernel_c() { ret; }
"#;
        let config = KernelPixelConfig::default();
        let suite = run_kernel_pixels("multi", ptx, &config);
        assert!(suite.all_passed());
    }

    #[test]
    fn test_gpu_pixel_test_suite_duration_accumulates() {
        let mut suite = GpuPixelTestSuite::new("test");
        suite.add_result(GpuPixelResult::pass(
            "t1",
            std::time::Duration::from_millis(10),
        ));
        suite.add_result(GpuPixelResult::pass(
            "t2",
            std::time::Duration::from_millis(20),
        ));
        suite.add_result(GpuPixelResult::pass(
            "t3",
            std::time::Duration::from_millis(30),
        ));

        assert_eq!(suite.duration, std::time::Duration::from_millis(60));
    }

    #[test]
    fn test_gpu_pixel_result_duration() {
        let duration = std::time::Duration::from_secs(1);
        let result = GpuPixelResult::pass("test", duration);
        assert_eq!(result.duration, duration);
    }
}
