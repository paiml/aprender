//! Optimization identification tooling for cbtop
//!
//! Provides systematic performance analysis using the cbtop Library API (HL-007).
//!
//! # Components
//!
//! - [`OptimizationSuite`]: Benchmark suite for baseline collection
//! - [`BottleneckAnalysis`]: Identifies operations performing below expectations
//! - [`RegressionDetector`]: Automated regression detection for CI/CD
//! - [`OptimizationValidator`]: Statistical validation of optimizations
//!
//! # Example
//!
//! ```rust,no_run
//! use cbtop::optimize::{OptimizationSuite, BottleneckAnalysis};
//!
//! // Collect baseline
//! let suite = OptimizationSuite::standard();
//! let baseline = suite.collect_baseline().unwrap();
//!
//! // Analyze bottlenecks
//! let analysis = suite.analyze_bottlenecks(&baseline);
//! for bottleneck in &analysis.severe {
//!     println!("{}: {} - {}", bottleneck.workload, bottleneck.efficiency, bottleneck.recommendation);
//! }
//! ```

mod bottleneck;
mod cpu_detect;
mod regression;
mod stats;
mod suite;
mod validator;

pub use bottleneck::{AnalysisSummary, BottleneckAnalysis, BottleneckEntry, BottleneckSeverity};
pub use cpu_detect::CpuCapabilities;
pub use regression::{RegressionDetector, RegressionEntry, RegressionReport};
pub use suite::{BaselineEntry, BaselineReport, OptimizationSuite, WorkloadConfig};
pub use validator::{OptimizationValidator, ValidationResult};

#[cfg(test)]
mod tests {
    use super::stats::{cv, mean, std_dev, t_test};
    use super::*;

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 0.001);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_std_dev() {
        let samples = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&samples);
        assert!((sd - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_cv() {
        let samples = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        assert_eq!(cv(&samples), 0.0);

        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let c = cv(&samples);
        assert!(c > 0.0);
    }

    #[test]
    fn test_t_test_same_distribution() {
        let a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let b = vec![10.1, 10.9, 10.3, 10.6, 10.4];
        let p = t_test(&a, &b);
        // Same distribution should have high p-value (not significant)
        assert!(p > 0.05);
    }

    #[test]
    fn test_t_test_different_distribution() {
        // Use slightly varying values to have non-zero variance
        let a = vec![10.0, 10.1, 9.9, 10.2, 9.8];
        let b = vec![20.0, 20.1, 19.9, 20.2, 19.8];
        let p = t_test(&a, &b);
        // Different distributions should have low p-value (significant)
        assert!(p < 0.05, "p-value {} should be < 0.05", p);
    }

    #[test]
    fn test_baseline_entry_serialization() {
        let entry = BaselineEntry {
            workload: "dot_product".to_string(),
            size: 1000000,
            backend: "Simd".to_string(),
            gflops: 50.0,
            efficiency: 0.5,
            cv_percent: 5.0,
            score: 85,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: BaselineEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.workload, "dot_product");
        assert_eq!(parsed.gflops, 50.0);
    }

    #[test]
    fn test_optimization_suite_quick() {
        let suite = OptimizationSuite::quick();
        assert_eq!(suite.workloads.len(), 2);
        assert_eq!(suite.sizes.len(), 2);
        assert_eq!(suite.duration, std::time::Duration::from_secs(1));
    }

    #[test]
    fn test_bottleneck_severity() {
        assert_eq!(
            serde_json::to_string(&BottleneckSeverity::Critical).unwrap(),
            "\"Critical\""
        );
    }

    #[test]
    fn test_regression_report_exit_code() {
        let passing = RegressionReport {
            passed: true,
            regressions: vec![],
            improvements: vec![],
            summary: "OK".to_string(),
        };
        assert_eq!(passing.exit_code(), 0);

        let failing = RegressionReport {
            passed: false,
            regressions: vec![RegressionEntry {
                workload: "test".to_string(),
                size: 1000,
                baseline_gflops: 100.0,
                current_gflops: 80.0,
                change_percent: -20.0,
            }],
            improvements: vec![],
            summary: "FAILED".to_string(),
        };
        assert_eq!(failing.exit_code(), 1);
    }

    #[test]
    fn test_validation_result_format() {
        let result = ValidationResult {
            passed: true,
            improvement_percent: 15.0,
            before_gflops: 50.0,
            after_gflops: 57.5,
            before_cv: 3.0,
            after_cv: 2.5,
            p_value: 0.01,
            statistically_significant: true,
        };

        let report = result.format_report();
        assert!(report.contains("PASSED"));
        assert!(report.contains("15.0%"));
    }
}
