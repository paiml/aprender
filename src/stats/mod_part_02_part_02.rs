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
}
