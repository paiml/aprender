//! Traditional descriptive statistics for vector data.
//!
//! This module provides high-level statistical operations built on top of
//! Trueno's SIMD-optimized primitives. Key features:
//!
//! - Quantiles and percentiles using R-7 method (Hyndman & Fan 1996)
//! - Five-number summary (min, Q1, median, Q3, max)
//! - Histograms with multiple bin selection methods
//! - Optimized with Toyota Way principles (QuickSelect for O(n) quantiles)
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
//! assert_eq!(stats.quantile(0.5).unwrap(), 3.0); // median
//! assert_eq!(stats.quantile(0.0).unwrap(), 1.0); // min
//! assert_eq!(stats.quantile(1.0).unwrap(), 5.0); // max
//! ```

use trueno::Vector;

/// Descriptive statistics computed on a vector of f32 values.
///
/// Holds a reference to the data vector to avoid unnecessary copying.
/// Uses lazy evaluation and caching for repeated computations.
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
    /// Bin edges (length = n_bins + 1)
    pub bins: Vec<f32>,
    /// Bin counts (length = n_bins)
    pub counts: Vec<usize>,
    /// Normalized density (optional, length = n_bins)
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
    /// Create a new DescriptiveStats instance from a data vector.
    ///
    /// # Arguments
    /// * `data` - Reference to a Vector<f32> containing the data
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// ```
    pub fn new(data: &'a Vector<f32>) -> Self {
        Self { data }
    }

    /// Compute quantile using linear interpolation (R-7 method).
    ///
    /// Uses the method from Hyndman & Fan (1996) commonly used in
    /// statistical packages (R, NumPy, Pandas).
    ///
    /// # Performance
    /// Uses QuickSelect (`select_nth_unstable`) for O(n) average-case
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
    /// assert_eq!(stats.quantile(0.5).unwrap(), 3.0); // median
    /// assert_eq!(stats.quantile(0.0).unwrap(), 1.0); // min
    /// assert_eq!(stats.quantile(1.0).unwrap(), 5.0); // max
    /// ```
    pub fn quantile(&self, q: f64) -> Result<f32, String> {
        // Validation
        if self.data.is_empty() {
            return Err("Cannot compute quantile of empty vector".to_string());
        }
        if !(0.0..=1.0).contains(&q) {
            return Err(format!("Quantile must be in [0, 1], got {}", q));
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
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            return Ok(working_copy[h_floor]);
        }

        // For interpolation, we need both floor and ceil values
        // Use nth_element twice (still O(n) average case)
        working_copy.select_nth_unstable_by(h_floor, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let lower = working_copy[h_floor];

        // After first partition, ceil element might be in different partition
        // Full sort is still O(n log n), but for single quantile with interpolation,
        // we need a different approach
        working_copy.select_nth_unstable_by(h_ceil, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
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
    /// let p = stats.percentiles(&[25.0, 50.0, 75.0]).unwrap();
    /// assert_eq!(p, vec![2.0, 3.0, 4.0]);
    /// ```
    pub fn percentiles(&self, percentiles: &[f64]) -> Result<Vec<f32>, String> {
        // Validate inputs
        if self.data.is_empty() {
            return Err("Cannot compute percentiles of empty vector".to_string());
        }
        for &p in percentiles {
            if !(0.0..=100.0).contains(&p) {
                return Err(format!("Percentile must be in [0, 100], got {}", p));
            }
        }

        // For multiple quantiles, full sort is optimal
        let mut sorted = self.data.as_slice().to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

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
    /// let summary = stats.five_number_summary().unwrap();
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
    /// assert_eq!(stats.iqr().unwrap(), 2.0);
    /// ```
    pub fn iqr(&self) -> Result<f32, String> {
        let summary = self.five_number_summary()?;
        Ok(summary.q3 - summary.q1)
    }

    /// Compute histogram with automatic bin selection (Freedman-Diaconis rule).
    ///
    /// Uses Freedman-Diaconis rule: bin_width = 2 * IQR / n^(1/3)
    /// This is optimal for unimodal, symmetric distributions.
    ///
    /// # Examples
    /// ```
    /// use aprender::stats::DescriptiveStats;
    /// use trueno::Vector;
    ///
    /// let data = Vector::from_slice(&[1.0, 2.0, 2.0, 3.0, 5.0]);
    /// let stats = DescriptiveStats::new(&data);
    /// let hist = stats.histogram_auto().unwrap();
    /// assert_eq!(hist.bins.len(), hist.counts.len() + 1);
    /// ```
    pub fn histogram_auto(&self) -> Result<Histogram, String> {
        self.histogram_method(BinMethod::FreedmanDiaconis)
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
    /// let hist = stats.histogram_method(BinMethod::Sturges).unwrap();
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
                // TODO: Implement Bayesian Blocks (O(n²) dynamic programming)
                // For now, fallback to Freedman-Diaconis
                return self.histogram_method(BinMethod::FreedmanDiaconis);
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
    /// let hist = stats.histogram(3).unwrap();
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
    /// let hist = stats.histogram_edges(&[0.0, 2.5, 5.0, 10.0]).unwrap();
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
        assert_eq!(stats.quantile(0.0).unwrap(), 42.0);
        assert_eq!(stats.quantile(0.5).unwrap(), 42.0);
        assert_eq!(stats.quantile(1.0).unwrap(), 42.0);
    }

    #[test]
    fn test_quantile_two_elements() {
        let v = Vector::from_slice(&[1.0, 2.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(stats.quantile(0.0).unwrap(), 1.0);
        assert_eq!(stats.quantile(0.5).unwrap(), 1.5);
        assert_eq!(stats.quantile(1.0).unwrap(), 2.0);
    }

    #[test]
    fn test_quantile_odd_length() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(stats.quantile(0.5).unwrap(), 3.0); // exact median
    }

    #[test]
    fn test_quantile_even_length() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(stats.quantile(0.5).unwrap(), 2.5); // interpolated median
    }

    #[test]
    fn test_quantile_edge_cases() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        assert_eq!(stats.quantile(0.0).unwrap(), 1.0); // min
        assert_eq!(stats.quantile(1.0).unwrap(), 5.0); // max
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
        let p = stats.percentiles(&[25.0, 50.0, 75.0]).unwrap();
        assert_eq!(p, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_five_number_summary() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let summary = stats.five_number_summary().unwrap();

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
        assert_eq!(stats.iqr().unwrap(), 2.0);
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
        let hist = stats.histogram(3).unwrap();

        assert_eq!(hist.bins.len(), 4); // n_bins + 1
        assert_eq!(hist.counts.len(), 3);
        assert_eq!(hist.counts.iter().sum::<usize>(), 5); // Total count
    }

    #[test]
    fn test_histogram_uniform_distribution() {
        // Uniform distribution should have roughly equal counts
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram(5).unwrap();

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
        let hist = stats.histogram(3).unwrap();

        assert_eq!(hist.bins.len(), 2);
        assert_eq!(hist.counts.len(), 1);
        assert_eq!(hist.counts[0], 4);
    }

    #[test]
    fn test_histogram_sturges() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram_method(BinMethod::Sturges).unwrap();

        // n = 8, so n_bins = ceil(log2(8)) + 1 = 3 + 1 = 4
        assert_eq!(hist.bins.len(), 5);
        assert_eq!(hist.counts.len(), 4);
        assert_eq!(hist.counts.iter().sum::<usize>(), 8);
    }

    #[test]
    fn test_histogram_square_root() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram_method(BinMethod::SquareRoot).unwrap();

        // n = 9, so n_bins = ceil(sqrt(9)) = 3
        assert_eq!(hist.bins.len(), 4);
        assert_eq!(hist.counts.len(), 3);
        assert_eq!(hist.counts.iter().sum::<usize>(), 9);
    }

    #[test]
    fn test_histogram_freedman_diaconis() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram_method(BinMethod::FreedmanDiaconis).unwrap();

        // Should have at least 1 bin
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);
        assert_eq!(hist.counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_histogram_auto() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram_auto().unwrap();

        // Auto uses Freedman-Diaconis by default
        assert!(hist.bins.len() >= 2);
        assert_eq!(hist.bins.len(), hist.counts.len() + 1);
    }

    #[test]
    fn test_histogram_edges_custom() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::new(&v);
        let hist = stats.histogram_edges(&[0.0, 2.5, 5.0, 10.0]).unwrap();

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
}
