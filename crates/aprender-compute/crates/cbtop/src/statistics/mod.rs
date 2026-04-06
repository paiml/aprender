//! Statistical Analysis Module (PMAT-024)
//!
//! Implements statistical analysis per F221 for 95% nonparametric confidence
//! intervals, effect size calculation, and bootstrap sampling.
//!
//! # Components
//!
//! | Component | Formula | Use Case |
//! |-----------|---------|----------|
//! | Bootstrap CI | Resampling with replacement | Nonparametric 95% CI |
//! | Cohen's d | (M1-M2) / pooled_std | Effect size magnitude |
//! | Welch's t-test | t-statistic with unequal variances | A/B comparison |
//! | Mann-Whitney U | Nonparametric rank test | Non-normal distributions |
//! | IQR Outlier Filter | Q1 - 1.5*IQR to Q3 + 1.5*IQR | Robust statistics |
//!
//! # Citations
//!
//! - [Efron & Tibshirani 1993] "An Introduction to the Bootstrap"
//! - [Cohen 1988] "Statistical Power Analysis for Behavioral Sciences"
//! - [Hoefler & Belli 2015] "Scientific Benchmarking of Parallel Computing Systems"

mod analysis;
mod comparison;
mod helpers;

pub use analysis::{EffectCategory, EffectSize, StatisticalAnalysis};
pub use comparison::{ComparisonResult, MannWhitneyResult, OutlierFilter};
pub use helpers::{bootstrap_ci, percentile, trimmed_mean};

#[cfg(test)]
mod tests;
