impl<'a> DescriptiveStats<'a> {

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
