// Statistical Regression Detection with Hypothesis Testing
// (Section 6.4 of single-shot-compile-tooling-spec.md)
//
// This module implements statistical regression detection to replace fixed
// percentage thresholds (magic 5%). Uses aprender's t-tests for significance
// and Zeller's Delta Debugging for noise filtering.
//
// Scientific Foundation:
// [7] Zeller, A. (2002). Isolating cause-effect chains from computer programs.
//     FSE-10. Delta Debugging minimizes differences between failing/passing runs.
//
// [9] Heger, C., Happe, J., & Farahbod, R. (2013). Automated root cause isolation
//     of performance regressions. ICPE. Fixed % thresholds yield high false positives.
//
// Key Innovation: No magic numbers. Thresholds adapt to project-specific variance.
//
// Implementation:
// - Uses aprender (crates.io) for statistical hypothesis testing (t-tests)
// - Uses trueno (crates.io) for SIMD-optimized vector statistics
// - Uses aprender's DescriptiveStats for quantiles and median calculation
// - NO custom implementations - leverage existing, well-tested libraries

mod config;
mod noise_filter;
mod statistics;
mod verdict;

pub use config::RegressionConfig;
pub use noise_filter::{filter_noisy_syscalls, SyscallDistribution};
pub use statistics::{compare_distributions, median, StatisticalTest};
pub use verdict::{assess_regression, RegressionAssessment, RegressionVerdict};

#[cfg(test)]
mod tests;
