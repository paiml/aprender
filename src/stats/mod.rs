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

impl<'a> DescriptiveStats<'a> {
    /// Create a new `DescriptiveStats` instance from a data vector.
    ///
    /// # Arguments
    /// * `data` - Reference to a `Vector<f32>` containing the data
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// ```
    #[must_use]
    pub fn new(data: &'a Vector<f32>) -> Self {
        Self { data }
    }

    /// Compute quantile using linear interpolation (R-7 method).
    ///
    /// Uses the method from Hyndman & Fan (1996) commonly used in
    /// statistical packages (R, `NumPy`, Pandas).
    ///
    /// # Performance
    /// Uses `QuickSelect` (`select_nth_unstable`) for O(n) average-case
    /// performance instead of full sort O(n log n). This is a Toyota Way
    /// Muda elimination optimization (Floyd & Rivest 1975).
    ///
    /// # Arguments
    /// * `q` - Quantile value in [0, 1]
    ///
    /// # Returns
    /// Interpolated quantile value
    ///
    /// # Errors
    /// Returns error if:
    /// - Data vector is empty
    /// - Quantile q is not in [0, 1]
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let stats = DescriptiveStats::new(&data);
    ///
    /// assert_eq!(stats.quantile(0.5).expect("median should be computable for valid data"), 3.0); // median
    /// assert_eq!(stats.quantile(0.0).expect("min quantile should be computable for valid data"), 1.0); // min
    /// assert_eq!(stats.quantile(1.0).expect("max quantile should be computable for valid data"), 5.0); // max
    /// ```
    pub fn quantile(&self, q: f64) -> Result<f32, String> {
        // Validation
        if self.data.is_empty() {
            return Err("Cannot compute quantile of empty vector".to_string());
        }
        if !(0.0..=1.0).contains(&q) {
            return Err(format!("Quantile must be in [0, 1], got {q}"));
        }

        let n = self.data.len();

        // Edge cases: single element
        if n == 1 {
            return Ok(self.data.as_slice()[0]);
        }

        // R-7 method: h = (n - 1) * q
        // Position in sorted array (0-indexed)
        let h = (n - 1) as f64 * q;
        let h_floor = h.floor() as usize;
        let h_ceil = h.ceil() as usize;

        // Toyota Way: Use QuickSelect for O(n) instead of full sort O(n log n)
        // This is Muda elimination (waste of unnecessary sorting)
        let mut working_copy = self.data.as_slice().to_vec();

        // Handle edge quantiles (q = 0.0 or q = 1.0)
        if h_floor == h_ceil {
            // Exact index, no interpolation needed
            working_copy.select_nth_unstable_by(h_floor, |a, b| {
                a.partial_cmp(b)
                    .expect("f32 values should be comparable (not NaN)")
            });
            return Ok(working_copy[h_floor]);
        }

        // For interpolation, we need both floor and ceil values
        // Use nth_element twice (still O(n) average case)
        working_copy.select_nth_unstable_by(h_floor, |a, b| {
            a.partial_cmp(b)
                .expect("f32 values should be comparable (not NaN)")
        });
        let lower = working_copy[h_floor];

        // After first partition, ceil element might be in different partition
        // Full sort is still O(n log n), but for single quantile with interpolation,
        // we need a different approach
        working_copy.select_nth_unstable_by(h_ceil, |a, b| {
            a.partial_cmp(b)
                .expect("f32 values should be comparable (not NaN)")
        });
        let upper = working_copy[h_ceil];

        // Linear interpolation
        let fraction = h - h_floor as f64;
        let result = lower + (fraction as f32) * (upper - lower);

        Ok(result)
    }

    /// Compute multiple percentiles efficiently (single sort).
    ///
    /// When computing multiple quantiles, it's more efficient to sort once
    /// and then index into the sorted array. This is O(n log n) amortized.
    ///
    /// # Arguments
    /// * `percentiles` - Slice of percentile values (0-100)
    ///
    /// # Returns
    /// Vector of percentile values in the same order as input
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// let p = stats.percentiles(&[25.0, 50.0, 75.0]).expect("percentiles should be computable for valid data");
    /// assert_eq!(p, vec![2.0, 3.0, 4.0]);
    /// ```
    pub fn percentiles(&self, percentiles: &[f64]) -> Result<Vec<f32>, String> {
        // Validate inputs
        if self.data.is_empty() {
            return Err("Cannot compute percentiles of empty vector".to_string());
        }
        for &p in percentiles {
            if !(0.0..=100.0).contains(&p) {
                return Err(format!("Percentile must be in [0, 100], got {p}"));
            }
        }

        // For multiple quantiles, full sort is optimal
        let mut sorted = self.data.as_slice().to_vec();
        sorted.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("f32 values should be comparable (not NaN)")
        });

        let n = sorted.len();
        let mut results = Vec::with_capacity(percentiles.len());

        for &p in percentiles {
            let q = p / 100.0;
            let h = (n - 1) as f64 * q;
            let h_floor = h.floor() as usize;
            let h_ceil = h.ceil() as usize;

            let value = if h_floor == h_ceil {
                sorted[h_floor]
            } else {
                let fraction = h - h_floor as f64;
                sorted[h_floor] + (fraction as f32) * (sorted[h_ceil] - sorted[h_floor])
            };

            results.push(value);
        }

        Ok(results)
    }

    /// Compute five-number summary: min, Q1, median, Q3, max.
    ///
    /// This is the foundation for box plots and outlier detection.
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// let summary = stats.five_number_summary().expect("five-number summary should be computable for valid data");
    ///
    /// assert_eq!(summary.min, 1.0);
    /// assert_eq!(summary.q1, 2.0);
    /// assert_eq!(summary.median, 3.0);
    /// assert_eq!(summary.q3, 4.0);
    /// assert_eq!(summary.max, 5.0);
    /// ```
    pub fn five_number_summary(&self) -> Result<FiveNumberSummary, String> {
        if self.data.is_empty() {
            return Err("Cannot compute summary of empty vector".to_string());
        }

        // Use percentiles for efficiency (single sort)
        let values = self.percentiles(&[0.0, 25.0, 50.0, 75.0, 100.0])?;

        Ok(FiveNumberSummary {
            min: values[0],
            q1: values[1],
            median: values[2],
            q3: values[3],
            max: values[4],
        })
    }

    /// Compute interquartile range (IQR = Q3 - Q1).
    ///
    /// The IQR is a measure of statistical dispersion, being equal to the
    /// difference between 75th and 25th percentiles.
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// assert_eq!(stats.iqr().expect("IQR should be computable for valid data"), 2.0);
    /// ```
    pub fn iqr(&self) -> Result<f32, String> {
        let summary = self.five_number_summary()?;
        Ok(summary.q3 - summary.q1)
    }

    /// Compute histogram with automatic bin selection (Freedman-Diaconis rule).
    ///
    /// Uses Freedman-Diaconis rule: `bin_width` = 2 * IQR / n^(1/3)
    /// This is optimal for unimodal, symmetric distributions.
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 2.0, 3.0, 5.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// let hist = stats.histogram_auto().expect("histogram should be computable for valid data");
    /// assert_eq!(hist.bins.len(), hist.counts.len() + 1);
    /// ```
    pub fn histogram_auto(&self) -> Result<Histogram, String> {
        self.histogram_method(BinMethod::FreedmanDiaconis)
    }

    /// Compute optimal histogram bin edges using Bayesian Blocks algorithm.
    ///
    /// This implements the Bayesian Blocks algorithm (Scargle et al., 2013) which
    /// finds optimal change points in the data using dynamic programming.
    ///
    /// # Returns
    /// Vector of bin edges (sorted, strictly increasing)
    fn bayesian_blocks_edges(&self) -> Result<Vec<f32>, String> {
        if self.data.is_empty() {
            return Err("Cannot compute Bayesian Blocks on empty data".to_string());
        }

        let n = self.data.len();

        // Handle edge cases
        if n == 1 {
            let val = self.data.as_slice()[0];
            return Ok(vec![val - 0.5, val + 0.5]);
        }

        // Sort data (Bayesian Blocks requires sorted data)
        let mut sorted_data: Vec<f32> = self.data.as_slice().to_vec();
        sorted_data.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("f32 values should be comparable (not NaN)")
        });

        // Handle all same values
        if sorted_data[0] == sorted_data[n - 1] {
            let val = sorted_data[0];
            return Ok(vec![val - 0.5, val + 0.5]);
        }

        // Prior on number of change points (ncp_prior)
        // Following Scargle et al. (2013), we use a prior that penalizes too many blocks
        // but allows detection of significant changes. Lower value = more blocks.
        let ncp_prior = 0.5_f32; // More sensitive to changes

        // Dynamic programming arrays
        let mut best_fitness = vec![0.0_f32; n];
        let mut last_change_point = vec![0_usize; n];

        // Compute fitness for first block [0, 0]
        best_fitness[0] = 0.0;

        // Fill DP table
        for r in 1..n {
            // Try all possible positions for previous change point
            let mut max_fitness = f32::NEG_INFINITY;
            let mut best_cp = 0;

            for l in 0..=r {
                // Compute fitness for block [l, r]
                let block_count = (r - l + 1) as f32;

                // For Bayesian Blocks, we want to favor blocks with similar density
                // Use negative variance as fitness (prefer uniform blocks)
                let block_values: Vec<f32> = sorted_data[l..=r].to_vec();

                // Compute block statistics
                let block_min = block_values[0];
                let block_max = block_values[block_values.len() - 1];
                let block_range = (block_max - block_min).max(1e-10);

                // Fitness: Prefer blocks with uniform density (low range relative to count)
                // and penalize creating new blocks
                let density_score = -block_range / block_count.sqrt();

                // Total fitness: previous best + current block - prior penalty
                let fitness = if l == 0 {
                    density_score - ncp_prior
                } else {
                    best_fitness[l - 1] + density_score - ncp_prior
                };

                if fitness > max_fitness {
                    max_fitness = fitness;
                    best_cp = l;
                }
            }

            best_fitness[r] = max_fitness;
            last_change_point[r] = best_cp;
        }

        // Backtrack to find change points
        let mut change_points = Vec::new();
        let mut current = n - 1;

        while current > 0 {
            let cp = last_change_point[current];
            if cp > 0 {
                change_points.push(cp);
            }
            if cp == 0 {
                break;
            }
            current = cp - 1;
        }

        change_points.reverse();

        // Convert change points to bin edges
        let mut edges = Vec::new();

        // Add left edge (slightly before first data point)
        let data_min = sorted_data[0];
        let data_max = sorted_data[n - 1];
        let range = data_max - data_min;
        let margin = range * 0.001; // 0.1% margin
        edges.push(data_min - margin);

        // Add edges at change points (midpoint between adjacent blocks)
        for &cp in &change_points {
            if cp > 0 && cp < n {
                let edge = (sorted_data[cp - 1] + sorted_data[cp]) / 2.0;
                edges.push(edge);
            }
        }

        // Add right edge (slightly after last data point)
        edges.push(data_max + margin);

        // Ensure edges are strictly increasing and unique
        edges.dedup();
        edges.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("f32 values should be comparable (not NaN)")
        });

        // Remove any non-strictly-increasing edges (shouldn't happen, but be safe)
        let mut i = 1;
        while i < edges.len() {
            if edges[i] <= edges[i - 1] {
                edges.remove(i);
            } else {
                i += 1;
            }
        }

        // Ensure we have at least 2 edges
        if edges.len() < 2 {
            return Ok(vec![data_min - margin, data_max + margin]);
        }

        Ok(edges)
    }

    /// Compute histogram with specified bin selection method.
    ///
    /// # Arguments
    /// * `method` - Bin selection method to use
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::{DescriptiveStats, BinMethod};
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// let hist = stats.histogram_method(BinMethod::Sturges).expect("histogram should be computable for valid data");
    /// ```
    pub fn histogram_method(&self, method: BinMethod) -> Result<Histogram, String> {
        if self.data.is_empty() {
            return Err("Cannot compute histogram of empty vector".to_string());
        }

        let n = self.data.len();
        let n_bins = match method {
            BinMethod::FreedmanDiaconis => {
                // bin_width = 2 * IQR * n^(-1/3)
                let iqr = self.iqr()?;
                if iqr == 0.0 {
                    return Err("IQR is zero, cannot use Freedman-Diaconis rule".to_string());
                }
                let bin_width = 2.0 * iqr * (n as f32).powf(-1.0 / 3.0);
                let data_min = self.data.min().map_err(|e| e.to_string())?;
                let data_max = self.data.max().map_err(|e| e.to_string())?;
                let range = data_max - data_min;
                let n_bins = (range / bin_width).ceil() as usize;
                n_bins.max(1) // At least 1 bin
            }
            BinMethod::Sturges => {
                // n_bins = ceil(log2(n)) + 1
                ((n as f64).log2().ceil() as usize + 1).max(1)
            }
            BinMethod::Scott => {
                // bin_width = 3.5 * σ * n^(-1/3)
                let std = self.data.stddev().map_err(|e| e.to_string())?;
                if std == 0.0 {
                    return Err("Standard deviation is zero, cannot use Scott rule".to_string());
                }
                let bin_width = 3.5 * std * (n as f32).powf(-1.0 / 3.0);
                let data_min = self.data.min().map_err(|e| e.to_string())?;
                let data_max = self.data.max().map_err(|e| e.to_string())?;
                let range = data_max - data_min;
                let n_bins = (range / bin_width).ceil() as usize;
                n_bins.max(1)
            }
            BinMethod::SquareRoot => {
                // n_bins = ceil(sqrt(n))
                ((n as f64).sqrt().ceil() as usize).max(1)
            }
            BinMethod::Bayesian => {
                // Use Bayesian Blocks algorithm to find optimal bin edges
                let edges = self.bayesian_blocks_edges()?;
                return self.histogram_edges(&edges);
            }
        };

        self.histogram(n_bins)
    }

    /// Compute histogram with fixed number of bins.
    ///
    /// # Arguments
    /// * `n_bins` - Number of bins (must be >= 1)
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// let hist = stats.histogram(3).expect("histogram should be computable for valid data");
    /// assert_eq!(hist.bins.len(), 4); // n_bins + 1 edges
    /// assert_eq!(hist.counts.len(), 3);
    /// ```
    pub fn histogram(&self, n_bins: usize) -> Result<Histogram, String> {
        if self.data.is_empty() {
            return Err("Cannot compute histogram of empty vector".to_string());
        }
        if n_bins == 0 {
            return Err("Number of bins must be at least 1".to_string());
        }

        let data_min = self.data.min().map_err(|e| e.to_string())?;
        let data_max = self.data.max().map_err(|e| e.to_string())?;

        // Handle case where all values are the same
        if data_min == data_max {
            return Ok(Histogram {
                bins: vec![data_min, data_max],
                counts: vec![self.data.len()],
                density: None,
            });
        }

        // Create bin edges (n_bins + 1 edges)
        let range = data_max - data_min;
        let bin_width = range / n_bins as f32;
        let mut bins = Vec::with_capacity(n_bins + 1);
        for i in 0..=n_bins {
            bins.push(data_min + i as f32 * bin_width);
        }

        // Count values in each bin
        let mut counts = vec![0usize; n_bins];
        for &value in self.data.as_slice() {
            // Find which bin this value belongs to
            let mut bin_idx = ((value - data_min) / bin_width) as usize;
            // Handle edge case: value == data_max goes in last bin
            if bin_idx >= n_bins {
                bin_idx = n_bins - 1;
            }
            counts[bin_idx] += 1;
        }

        Ok(Histogram {
            bins,
            counts,
            density: None,
        })
    }

    /// Compute histogram with custom bin edges.
    ///
    /// # Arguments
    /// * `edges` - Bin edges (must be sorted and have length >= 2)
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// let hist = stats.histogram_edges(&[0.0, 2.5, 5.0, 10.0]).expect("histogram should be computable for valid bin edges");
    /// assert_eq!(hist.bins.len(), 4);
    /// assert_eq!(hist.counts.len(), 3);
    /// ```
    pub fn histogram_edges(&self, edges: &[f32]) -> Result<Histogram, String> {
        if self.data.is_empty() {
            return Err("Cannot compute histogram of empty vector".to_string());
        }
        if edges.len() < 2 {
            return Err("Must have at least 2 bin edges".to_string());
        }

        // Verify edges are sorted
        for i in 1..edges.len() {
            if edges[i] <= edges[i - 1] {
                return Err("Bin edges must be strictly increasing".to_string());
            }
        }

        let n_bins = edges.len() - 1;
        let mut counts = vec![0usize; n_bins];

        for &value in self.data.as_slice() {
            // Find which bin this value belongs to
            if value < edges[0] || value > edges[n_bins] {
                // Value is out of range, skip
                continue;
            }

            // Find the bin index
            // Bins are [edges[i], edges[i+1]) except the last bin is [edges[n-1], edges[n]]
            let mut bin_idx = None;
            for i in 0..(n_bins - 1) {
                if value >= edges[i] && value < edges[i + 1] {
                    bin_idx = Some(i);
                    break;
                }
            }

            // If not found yet, check the last bin (which is closed on both sides)
            if bin_idx.is_none() && value >= edges[n_bins - 1] && value <= edges[n_bins] {
                bin_idx = Some(n_bins - 1);
            }

            if let Some(idx) = bin_idx {
                counts[idx] += 1;
            }
        }

        Ok(Histogram {
            bins: edges.to_vec(),
            counts,
            density: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trueno::Vector;

    #[test]
    fn test_quantile_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.quantile(0.5).is_err());
    }

    #[test]
    fn test_quantile_single_element() {
        let v = Vector::from_slice(&[42.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats
                .quantile(0.0)
                .expect("quantile should succeed for single element"),
            42.0
        );
        assert_eq!(
            stats
                .quantile(0.5)
                .expect("quantile should succeed for single element"),
            42.0
        );
        assert_eq!(
            stats
                .quantile(1.0)
                .expect("quantile should succeed for single element"),
            42.0
        );
    }

    #[test]
    fn test_quantile_two_elements() {
        let v = Vector::from_slice(&[1.0, 2.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats
                .quantile(0.0)
                .expect("quantile should succeed for two elements"),
            1.0
        );
        assert_eq!(
            stats
                .quantile(0.5)
                .expect("quantile should succeed for two elements"),
            1.5
        );
        assert_eq!(
            stats
                .quantile(1.0)
                .expect("quantile should succeed for two elements"),
            2.0
        );
    }

    #[test]
    fn test_quantile_odd_length() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats
                .quantile(0.5)
                .expect("quantile should succeed for odd length data"),
            3.0
        ); // exact median
    }

    #[test]
    fn test_quantile_even_length() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats
                .quantile(0.5)
                .expect("quantile should succeed for even length data"),
            2.5
        ); // interpolated median
    }

    #[test]
    fn test_quantile_edge_cases() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(
            stats.quantile(0.0).expect("min quantile should succeed"),
            1.0
        ); // min
        assert_eq!(
            stats.quantile(1.0).expect("max quantile should succeed"),
            5.0
        ); // max
    }

    #[test]
    fn test_quantile_invalid() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.quantile(-0.1).is_err());
        assert!(stats.quantile(1.1).is_err());
    }

    #[test]
    fn test_percentiles() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let p = stats
            .percentiles(&[25.0, 50.0, 75.0])
            .expect("percentiles should succeed for valid inputs");
        assert_eq!(p, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_five_number_summary() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let summary = stats
            .five_number_summary()
            .expect("five-number summary should succeed for valid data");

        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.q1, 2.0);
        assert_eq!(summary.median, 3.0);
        assert_eq!(summary.q3, 4.0);
        assert_eq!(summary.max, 5.0);
    }

    #[test]
    fn test_iqr() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(stats.iqr().expect("IQR should succeed for valid data"), 2.0);
    }

    // Histogram tests

    #[test]
    fn test_histogram_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram(3).is_err());
    }

    #[test]
    fn test_histogram_zero_bins() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram(0).is_err());
    }

    #[test]
    fn test_histogram_fixed_bins() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram(3)
            .expect("histogram should succeed for valid inputs");

        assert_eq!(hist.bins.len(), 4); // n_bins + 1
        assert_eq!(hist.counts.len(), 3);
        assert_eq!(hist.counts.iter().sum::<usize>(), 5); // Total count
    }

    #[test]
    fn test_histogram_uniform_distribution() {
        // Uniform distribution should have roughly equal counts
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram(5)
            .expect("histogram should succeed for uniform distribution");

        assert_eq!(hist.bins.len(), 6);
        assert_eq!(hist.counts.len(), 5);
        // Each bin should have exactly 2 values
        for count in hist.counts {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_histogram_all_same_value() {
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram(3)
            .expect("histogram should succeed for constant data");

        assert_eq!(hist.bins.len(), 2);
        assert_eq!(hist.counts.len(), 1);
        assert_eq!(hist.counts[0], 4);
    }

    #[test]
    fn test_histogram_sturges() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Sturges)
            .expect("histogram with Sturges method should succeed");

        // n = 8, so n_bins = ceil(log2(8)) + 1 = 3 + 1 = 4
        assert_eq!(hist.bins.len(), 5);
        assert_eq!(hist.counts.len(), 4);
        assert_eq!(hist.counts.iter().sum::<usize>(), 8);
    }

    #[test]
    fn test_histogram_square_root() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::SquareRoot)
            .expect("histogram with SquareRoot method should succeed");

        // n = 9, so n_bins = ceil(sqrt(9)) = 3
        assert_eq!(hist.bins.len(), 4);
        assert_eq!(hist.counts.len(), 3);
        assert_eq!(hist.counts.iter().sum::<usize>(), 9);
    }

    #[test]
    fn test_histogram_freedman_diaconis() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::FreedmanDiaconis)
            .expect("histogram with FreedmanDiaconis method should succeed");

        // Should have at least 1 bin
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);
        assert_eq!(hist.counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_histogram_auto() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_auto()
            .expect("auto histogram should succeed");

        // Auto uses Freedman-Diaconis by default
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);
    }

    #[test]
    fn test_histogram_edges_custom() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_edges(&[0.0, 2.5, 5.0, 10.0])
            .expect("histogram with custom edges should succeed");

        assert_eq!(hist.bins.len(), 4);
        assert_eq!(hist.counts.len(), 3);
        // Standard histogram convention:
        // Bin 0: [0.0, 2.5) -> values 1.0, 2.0
        // Bin 1: [2.5, 5.0) -> values 3.0, 4.0
        // Bin 2: [5.0, 10.0] -> value 5.0 (last bin is closed on both sides)
        assert_eq!(hist.counts[0], 2);
        assert_eq!(hist.counts[1], 2);
        assert_eq!(hist.counts[2], 1);
    }

    #[test]
    fn test_histogram_edges_invalid() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);

        // Too few edges
        assert!(stats.histogram_edges(&[1.0]).is_err());

        // Non-sorted edges
        assert!(stats.histogram_edges(&[5.0, 1.0, 10.0]).is_err());

        // Non-strictly increasing
        assert!(stats.histogram_edges(&[1.0, 5.0, 5.0, 10.0]).is_err());
    }

    // Bayesian Blocks tests
    #[test]
    fn test_histogram_bayesian_basic() {
        // Basic test: algorithm should run and produce valid histogram
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed");

        // Should produce valid histogram
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);

        // Bins should be sorted
        for i in 1..hist.bins.len() {
            assert!(hist.bins[i] > hist.bins[i - 1]);
        }

        // Counts should sum to number of data points
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_histogram_bayesian_uniform_data() {
        // Uniform data should produce relatively few bins
        let v = Vector::from_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for uniform data");

        // Uniform distribution should not need many bins
        assert!(hist.bins.len() <= 10); // Should be much fewer than 20
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);
    }

    #[test]
    fn test_histogram_bayesian_change_point_detection() {
        // Data with clear change points: two distinct clusters
        let v = Vector::from_slice(&[
            // Cluster 1: around 1-2
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, // Cluster 2: around 9-10
            9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0,
        ]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for clustered data");

        // Should detect the gap and create appropriate bins
        // Should have at least 2 bins to capture both clusters
        assert!(hist.bins.len() >= 3);

        // Verify bins cover the data range
        assert!(hist.bins[0] <= 1.0);
        assert!(
            *hist
                .bins
                .last()
                .expect("histogram should have at least one bin edge")
                >= 10.0
        );
    }

    #[test]
    fn test_histogram_bayesian_small_dataset() {
        // Small dataset - should still work
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for small dataset");

        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);

        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_histogram_bayesian_reproducibility() {
        // Same data should give same result (deterministic algorithm)
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0]);
        let stats = DescriptiveStats::new(&v);

        let hist1 = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("first Bayesian histogram should succeed");
        let hist2 = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("second Bayesian histogram should succeed");

        // Should produce identical results
        assert_eq!(hist1.bins.len(), hist2.bins.len());
        for (b1, b2) in hist1.bins.iter().zip(hist2.bins.iter()) {
            assert!((b1 - b2).abs() < 1e-6);
        }
        assert_eq!(hist1.counts, hist2.counts);
    }

    #[test]
    fn test_histogram_bayesian_single_value() {
        // All same value - should handle gracefully
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for constant data");

        // Should create at least 1 bin
        assert!(hist.bins.len() >= 2); // n+1 edges for n bins
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);

        // All values should be in bins
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_histogram_bayesian_vs_fixed_width() {
        // Compare Bayesian Blocks with fixed-width methods
        // Data with non-uniform distribution
        let v = Vector::from_slice(&[
            1.0, 1.5, 2.0, 2.5, 3.0, // Dense cluster
            10.0, 15.0, 20.0, // Sparse region
            30.0, 30.5, 31.0, 31.5, 32.0, // Another dense cluster
        ]);
        let stats = DescriptiveStats::new(&v);

        let hist_bayesian = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed");
        let hist_sturges = stats
            .histogram_method(BinMethod::Sturges)
            .expect("Sturges histogram should succeed");

        // Both should be valid
        assert!(hist_bayesian.bins.len() >= 2);
        assert!(hist_sturges.bins.len() >= 2);

        // Bayesian should adapt to data structure
        // (exact comparison depends on implementation, so we just verify it works)
        assert_eq!(hist_bayesian.bins.len(), hist_bayesian.counts.len() + 1);
    }

    #[test]
    fn test_histogram_bayesian_large_dataset() {
        // Larger dataset to test O(n²) scaling
        let mut data = Vec::new();
        for i in 0..50 {
            data.push(i as f32 / 10.0);
        }
        let v = Vector::from_slice(&data);
        let stats = DescriptiveStats::new(&v);

        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian histogram should succeed for large dataset");

        // Should complete in reasonable time and produce valid result
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);

        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 50);
    }

    // =========================================================================
    // Additional coverage: Debug/Clone/PartialEq on structs and enums
    // =========================================================================

    #[test]
    fn test_descriptive_stats_debug() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        let debug_str = format!("{:?}", stats);
        assert!(
            debug_str.contains("DescriptiveStats"),
            "Debug output should contain type name"
        );
    }

    #[test]
    fn test_five_number_summary_clone_partial_eq() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let summary = stats.five_number_summary().expect("Should compute summary");
        let cloned = summary.clone();
        assert_eq!(summary, cloned);
    }

    #[test]
    fn test_five_number_summary_debug() {
        let summary = FiveNumberSummary {
            min: 1.0,
            q1: 2.0,
            median: 3.0,
            q3: 4.0,
            max: 5.0,
        };
        let debug_str = format!("{:?}", summary);
        assert!(debug_str.contains("FiveNumberSummary"));
        assert!(debug_str.contains("min"));
    }

    #[test]
    fn test_histogram_clone_partial_eq() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram(3).expect("Should compute histogram");
        let cloned = hist.clone();
        assert_eq!(hist, cloned);
    }

    #[test]
    fn test_histogram_debug() {
        let hist = Histogram {
            bins: vec![0.0, 1.0, 2.0],
            counts: vec![5, 3],
            density: None,
        };
        let debug_str = format!("{:?}", hist);
        assert!(debug_str.contains("Histogram"));
        assert!(debug_str.contains("bins"));
        assert!(debug_str.contains("counts"));
    }

    #[test]
    fn test_histogram_with_density() {
        let hist = Histogram {
            bins: vec![0.0, 1.0, 2.0],
            counts: vec![5, 3],
            density: Some(vec![0.625, 0.375]),
        };
        let cloned = hist.clone();
        assert_eq!(hist, cloned);
        assert!(hist.density.is_some());
    }

    #[test]
    fn test_bin_method_clone_copy_partial_eq_debug() {
        let method = BinMethod::FreedmanDiaconis;
        let cloned = method;
        assert_eq!(method, cloned);

        let method2 = BinMethod::Sturges;
        assert_ne!(method, method2);

        let debug_str = format!("{:?}", BinMethod::Bayesian);
        assert!(debug_str.contains("Bayesian"));

        // Test all variants for Clone/Copy/PartialEq
        let methods = [
            BinMethod::FreedmanDiaconis,
            BinMethod::Sturges,
            BinMethod::Scott,
            BinMethod::SquareRoot,
            BinMethod::Bayesian,
        ];
        for &m in &methods {
            let copied = m;
            assert_eq!(m, copied);
        }
    }

    // =========================================================================
    // Additional coverage: error paths in percentiles, five_number_summary, iqr
    // =========================================================================

    #[test]
    fn test_percentiles_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.percentiles(&[50.0]).is_err());
    }

    #[test]
    fn test_percentiles_invalid_value() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.percentiles(&[101.0]).is_err());
        assert!(stats.percentiles(&[-1.0]).is_err());
    }

    #[test]
    fn test_percentiles_boundary_values() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let p = stats
            .percentiles(&[0.0, 100.0])
            .expect("Should handle boundary percentiles");
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 5.0);
    }

    #[test]
    fn test_five_number_summary_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.five_number_summary().is_err());
    }

    #[test]
    fn test_iqr_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.iqr().is_err());
    }

    #[test]
    fn test_iqr_single_element() {
        let v = Vector::from_slice(&[42.0]);
        let stats = DescriptiveStats::new(&v);
        let iqr_val = stats.iqr().expect("Should compute IQR for single element");
        assert!(
            (iqr_val - 0.0).abs() < 1e-10,
            "IQR of single element should be 0"
        );
    }

    // =========================================================================
    // Additional coverage: histogram_method error/edge paths
    // =========================================================================

    #[test]
    fn test_histogram_method_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram_method(BinMethod::Sturges).is_err());
        assert!(stats.histogram_method(BinMethod::Scott).is_err());
        assert!(stats.histogram_method(BinMethod::FreedmanDiaconis).is_err());
        assert!(stats.histogram_method(BinMethod::SquareRoot).is_err());
        assert!(stats.histogram_method(BinMethod::Bayesian).is_err());
    }

    #[test]
    fn test_histogram_auto_empty() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram_auto().is_err());
    }

    #[test]
    fn test_histogram_freedman_diaconis_zero_iqr() {
        // All same values -> IQR = 0 -> error in Freedman-Diaconis
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let result = stats.histogram_method(BinMethod::FreedmanDiaconis);
        assert!(
            result.is_err(),
            "FreedmanDiaconis should fail when IQR is zero"
        );
    }

    #[test]
    fn test_histogram_scott_zero_std() {
        // All same values -> std = 0 -> error in Scott
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let result = stats.histogram_method(BinMethod::Scott);
        assert!(result.is_err(), "Scott should fail when std is zero");
    }

    #[test]
    fn test_histogram_scott_normal_data() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Scott)
            .expect("Scott method should succeed");
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.counts.iter().sum::<usize>(), 10);
    }

    // =========================================================================
    // Additional coverage: histogram_edges edge cases
    // =========================================================================

    #[test]
    fn test_histogram_edges_empty_data() {
        let v = Vector::from_slice(&[]);
        let stats = DescriptiveStats::new(&v);
        assert!(stats.histogram_edges(&[0.0, 1.0, 2.0]).is_err());
    }

    #[test]
    fn test_histogram_edges_out_of_range_values() {
        // Some values fall outside the bin range and should be skipped
        let v = Vector::from_slice(&[-10.0, 1.0, 2.0, 3.0, 100.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_edges(&[0.0, 2.5, 5.0])
            .expect("Should handle out-of-range values");
        // Only 1.0, 2.0 are in [0.0, 2.5), 3.0 is in [2.5, 5.0]
        // -10.0 and 100.0 are out of range
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 3, "Only 3 values should be in range");
    }

    #[test]
    fn test_histogram_edges_single_bin() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_edges(&[0.0, 10.0])
            .expect("Should handle single bin");
        assert_eq!(hist.counts.len(), 1);
        assert_eq!(hist.counts[0], 3);
    }

    #[test]
    fn test_histogram_edges_value_on_boundary() {
        // Value exactly on an interior edge goes to the left bin (except last bin)
        let v = Vector::from_slice(&[2.5]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_edges(&[0.0, 2.5, 5.0])
            .expect("Should handle boundary value");
        // 2.5 is at the boundary between bins 0 and 1
        // Bin 0: [0.0, 2.5), Bin 1: [2.5, 5.0]
        // 2.5 goes to bin 1 (last bin is closed on both sides)
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 1);
    }

    // =========================================================================
    // Additional coverage: bayesian_blocks_edges edge cases
    // =========================================================================

    #[test]
    fn test_bayesian_blocks_single_element() {
        let v = Vector::from_slice(&[42.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian should handle single element");
        assert!(hist.bins.len() >= 2);
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn test_bayesian_blocks_two_elements() {
        let v = Vector::from_slice(&[1.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats
            .histogram_method(BinMethod::Bayesian)
            .expect("Bayesian should handle two elements");
        assert!(hist.bins.len() >= 2);
        let total: usize = hist.counts.iter().sum();
        assert_eq!(total, 2);
    }

    // =========================================================================
    // Additional coverage: quantile interpolation paths
    // =========================================================================

    #[test]
    fn test_quantile_quartiles() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let stats = DescriptiveStats::new(&v);
        let q1 = stats.quantile(0.25).expect("Q1 should compute");
        let q3 = stats.quantile(0.75).expect("Q3 should compute");
        assert!(q1 < q3, "Q1 should be less than Q3");
        assert!(q1 >= 1.0 && q1 <= 8.0);
        assert!(q3 >= 1.0 && q3 <= 8.0);
    }

    #[test]
    fn test_quantile_unsorted_data() {
        // Data is not sorted; quantile should still work correctly
        let v = Vector::from_slice(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        let stats = DescriptiveStats::new(&v);
        let median = stats.quantile(0.5).expect("Median should compute");
        assert_eq!(median, 3.0, "Median of [1,2,3,4,5] should be 3.0");
    }

    #[test]
    fn test_percentiles_single_element() {
        let v = Vector::from_slice(&[7.0]);
        let stats = DescriptiveStats::new(&v);
        let p = stats
            .percentiles(&[0.0, 25.0, 50.0, 75.0, 100.0])
            .expect("Should compute percentiles for single element");
        for &val in &p {
            assert_eq!(
                val, 7.0,
                "All percentiles of single element should be that element"
            );
        }
    }

    #[test]
    fn test_five_number_summary_single_element() {
        let v = Vector::from_slice(&[42.0]);
        let stats = DescriptiveStats::new(&v);
        let summary = stats
            .five_number_summary()
            .expect("Should compute for single element");
        assert_eq!(summary.min, 42.0);
        assert_eq!(summary.q1, 42.0);
        assert_eq!(summary.median, 42.0);
        assert_eq!(summary.q3, 42.0);
        assert_eq!(summary.max, 42.0);
    }

    #[test]
    fn test_five_number_summary_two_elements() {
        let v = Vector::from_slice(&[1.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let summary = stats
            .five_number_summary()
            .expect("Should compute for two elements");
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 5.0);
        assert_eq!(summary.median, 3.0);
    }

    #[test]
    fn test_histogram_single_bin() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram(1).expect("Should work with 1 bin");
        assert_eq!(hist.bins.len(), 2);
        assert_eq!(hist.counts.len(), 1);
        assert_eq!(hist.counts[0], 5);
    }

    #[test]
    fn test_five_number_summary_partial_eq_ne() {
        let s1 = FiveNumberSummary {
            min: 1.0,
            q1: 2.0,
            median: 3.0,
            q3: 4.0,
            max: 5.0,
        };
        let s2 = FiveNumberSummary {
            min: 1.0,
            q1: 2.0,
            median: 3.0,
            q3: 4.0,
            max: 6.0, // different max
        };
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_histogram_partial_eq_ne() {
        let h1 = Histogram {
            bins: vec![0.0, 1.0],
            counts: vec![5],
            density: None,
        };
        let h2 = Histogram {
            bins: vec![0.0, 2.0], // different edge
            counts: vec![5],
            density: None,
        };
        assert_ne!(h1, h2);
    }
}
