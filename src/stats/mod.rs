//! Traditional descriptive statistics for vector data.
//!
//! This module provides high-level statistical operations built on top of
//! Trueno's SIMD-optimized primitives. Key features:
//!
//! - Quantiles and percentiles using R-7 method (Hyndman & Fan 1996)
//! - Five-number summary (min, Q1, median, Q3, max)
//! - Histograms with multiple bin selection methods
//! - Hypothesis testing (t-tests, chi-square, ANOVA)
//! - Covariance and correlation matrices
//! - Optimized with Toyota Way principles (`QuickSelect` for O(n) quantiles)
//!
//! # Examples
//!
//! ```
//! use aprender::stats::DescriptiveStats;
//! use trueno::Vector;
//!
//! let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
//! let stats = DescriptiveStats::new(&data);
//!
//! assert_eq!(stats.quantile(0.5).expect("median should be computable for valid data"), 3.0); // median
//! assert_eq!(stats.quantile(0.0).expect("min quantile should be computable for valid data"), 1.0); // min
//! assert_eq!(stats.quantile(1.0).expect("max quantile should be computable for valid data"), 5.0); // max
//! ```

pub mod covariance;
pub mod hypothesis;

pub use covariance::{corr, corr_matrix, cov, cov_matrix};
pub use hypothesis::{
    chisquare, f_oneway, ttest_1samp, ttest_ind, ttest_rel, AnovaResult, ChiSquareResult,
    TTestResult,
};

use trueno::Vector;

/// Descriptive statistics computed on a vector of f32 values.
///
/// Holds a reference to the data vector to avoid unnecessary copying.
/// Uses lazy evaluation and caching for repeated computations.
#[derive(Debug)]
pub struct DescriptiveStats<'a> {
    data: &'a Vector<f32>,
}

/// Five-number summary: minimum, Q1, median, Q3, maximum.
///
/// This is the foundation for box plots and outlier detection.
#[derive(Debug, Clone, PartialEq)]
pub struct FiveNumberSummary {
    pub min: f32,
    pub q1: f32,
    pub median: f32,
    pub q3: f32,
    pub max: f32,
}

/// Histogram representation with bin edges and counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    /// Bin edges (length = `n_bins` + 1)
    pub bins: Vec<f32>,
    /// Bin counts (length = `n_bins`)
    pub counts: Vec<usize>,
    /// Normalized density (optional, length = `n_bins`)
    pub density: Option<Vec<f64>>,
}

/// Bin selection methods for histogram construction.
///
/// Different methods are optimal for different data distributions:
/// - `FreedmanDiaconis`: Default for unimodal distributions
/// - `Sturges`: Best for small datasets (n < 200)
/// - `Scott`: Best for smooth, normal-like data
/// - `SquareRoot`: Simple rule of thumb
/// - `Bayesian`: Best for multimodal/heavy-tailed distributions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinMethod {
    FreedmanDiaconis,
    Sturges,
    Scott,
    SquareRoot,
    Bayesian,
}

include!("descriptive_impl.rs");
include!("descriptive_tests.rs");
