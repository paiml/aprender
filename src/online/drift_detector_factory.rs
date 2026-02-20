
impl ADWIN {
    /// Create a new ADWIN detector with default delta (0.002)
    ///
    /// The default delta provides good balance between sensitivity and
    /// false positive rate.
    #[must_use]
    pub fn new() -> Self {
        Self::with_delta(0.002)
    }

    /// Create ADWIN with custom confidence parameter
    ///
    /// # Arguments
    /// * `delta` - Confidence parameter (smaller = more sensitive, typical: 0.001-0.1)
    #[must_use]
    pub fn with_delta(delta: f64) -> Self {
        Self {
            delta,
            max_buckets: 5,
            bucket_rows: vec![VecDeque::new(); 32], // log2(max_window)
            total: 0.0,
            count: 0,
            width: 0,
            status: DriftStatus::Stable,
            last_bucket_row: 0,
        }
    }

    /// Get current window size
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.width
    }

    /// Get current mean of window
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total / self.count as f64
        }
    }

    /// Insert element into bucket structure
    fn insert_element(&mut self, value: f64) {
        // Add to first row
        self.bucket_rows[0].push_back(Bucket {
            total: value,
            count: 1,
        });
        self.total += value;
        self.count += 1;
        self.width += 1;

        // Compress if needed
        self.compress_buckets();
    }

    /// Compress buckets when too many in a row
    fn compress_buckets(&mut self) {
        for row in 0..self.bucket_rows.len() - 1 {
            if self.bucket_rows[row].len() > self.max_buckets {
                // Merge last two buckets and promote
                if let (Some(b1), Some(b2)) = (
                    self.bucket_rows[row].pop_front(),
                    self.bucket_rows[row].pop_front(),
                ) {
                    let merged = Bucket {
                        total: b1.total + b2.total,
                        count: b1.count + b2.count,
                    };
                    self.bucket_rows[row + 1].push_back(merged);
                }
            }
        }
    }

    /// Remove oldest elements from window
    fn delete_element(&mut self) {
        // Find non-empty row from the end
        for row in (0..self.bucket_rows.len()).rev() {
            if !self.bucket_rows[row].is_empty() {
                if let Some(bucket) = self.bucket_rows[row].pop_front() {
                    self.total -= bucket.total;
                    self.count -= bucket.count;
                    self.width -= bucket.count;
                    self.last_bucket_row = row;
                    return;
                }
            }
        }
    }

    /// Check if cut is significant using Hoeffding bound
    fn detect_cut(&self, n0: usize, n1: usize, u0: f64, u1: f64) -> bool {
        let n = n0 + n1;
        if n0 == 0 || n1 == 0 {
            return false;
        }

        let m = 1.0 / (n0 as f64) + 1.0 / (n1 as f64);
        let dd = (2.0 / (n0 as f64 * m) * (2.0 * n as f64 / self.delta).ln()).sqrt();
        let epsilon = dd + 2.0 / 3.0 / (n0 as f64) * (2.0 * n as f64 / self.delta).ln();

        (u0 - u1).abs() > epsilon
    }
}

impl DriftDetector for ADWIN {
    fn add_element(&mut self, error: bool) {
        let value = if error { 1.0 } else { 0.0 };
        self.insert_element(value);

        // Check for concept drift by looking for cuts in the window
        self.status = DriftStatus::Stable;

        // Iterate through possible window cuts
        let mut n0 = 0usize;
        let mut u0 = 0.0f64;

        for row in 0..self.bucket_rows.len() {
            for bucket in &self.bucket_rows[row] {
                n0 += bucket.count;
                u0 += bucket.total;

                let n1 = self.count.saturating_sub(n0);
                if n1 == 0 {
                    continue;
                }

                let u1 = self.total - u0;
                let mean0 = u0 / n0.max(1) as f64;
                let mean1 = u1 / n1.max(1) as f64;

                if self.detect_cut(n0, n1, mean0, mean1) {
                    self.status = DriftStatus::Drift;
                    // Remove old part of window
                    while self.width > n1 {
                        self.delete_element();
                    }
                    return;
                }

                // Check for warning (less strict threshold)
                let warning_delta = self.delta * 10.0;
                let m = 1.0 / (n0 as f64) + 1.0 / (n1 as f64);
                let dd =
                    (2.0 / (n0 as f64 * m) * (2.0 * (n0 + n1) as f64 / warning_delta).ln()).sqrt();
                let epsilon =
                    dd + 2.0 / 3.0 / (n0 as f64) * (2.0 * (n0 + n1) as f64 / warning_delta).ln();

                if (mean0 - mean1).abs() > epsilon * 0.7 && self.status == DriftStatus::Stable {
                    self.status = DriftStatus::Warning;
                }
            }
        }
    }

    fn detected_change(&self) -> DriftStatus {
        self.status
    }

    fn reset(&mut self) {
        for row in &mut self.bucket_rows {
            row.clear();
        }
        self.total = 0.0;
        self.count = 0;
        self.width = 0;
        self.status = DriftStatus::Stable;
        self.last_bucket_row = 0;
    }

    fn stats(&self) -> DriftStats {
        DriftStats {
            n_samples: self.count as u64,
            error_rate: self.mean(),
            min_error_rate: 0.0,
            std_dev: 0.0, // ADWIN doesn't track std dev
            status: self.status,
        }
    }
}

/// Factory for creating drift detectors
#[derive(Debug)]
pub struct DriftDetectorFactory;

impl DriftDetectorFactory {
    /// Create the recommended default drift detector (ADWIN)
    ///
    /// Per Toyota Way review: "Use ADWIN as default. While DDM is simpler,
    /// it struggles with gradual drift."
    #[must_use]
    pub fn recommended() -> Box<dyn DriftDetector> {
        Box::new(ADWIN::new())
    }

    /// Create a DDM detector
    #[must_use]
    pub fn ddm() -> Box<dyn DriftDetector> {
        Box::new(DDM::new())
    }

    /// Create a Page-Hinkley detector
    #[must_use]
    pub fn page_hinkley() -> Box<dyn DriftDetector> {
        Box::new(PageHinkley::new())
    }

    /// Create an ADWIN detector
    #[must_use]
    pub fn adwin() -> Box<dyn DriftDetector> {
        Box::new(ADWIN::new())
    }
}
