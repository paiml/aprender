//! Latency histogram with bucket statistics and entropy.

/// Histogram bucket for latency distribution
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Lower bound of bucket (inclusive)
    pub lower: f64,
    /// Upper bound of bucket (exclusive)
    pub upper: f64,
    /// Count of samples in bucket
    pub count: usize,
    /// Percentage of total samples
    pub percentage: f64,
}

/// Latency histogram with statistical properties
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    /// Histogram buckets
    pub buckets: Vec<HistogramBucket>,
    /// Total number of samples
    pub total_samples: usize,
    /// Shannon entropy (0-1 normalized)
    pub entropy: f64,
    /// Index of the mode bucket (most frequent)
    pub mode_bucket: usize,
    /// Number of buckets
    pub bucket_count: usize,
}

impl LatencyHistogram {
    /// Build histogram from samples with specified bucket count
    pub fn build(samples: &[f64], bucket_count: usize) -> Self {
        if samples.is_empty() || bucket_count == 0 {
            return Self {
                buckets: Vec::new(),
                total_samples: 0,
                entropy: 0.0,
                mode_bucket: 0,
                bucket_count: 0,
            };
        }

        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range = max - min;
        let bucket_width = if range > 0.0 {
            range / bucket_count as f64
        } else {
            1.0
        };

        let mut buckets: Vec<HistogramBucket> = (0..bucket_count)
            .map(|i| {
                let lower = min + i as f64 * bucket_width;
                let upper = if i == bucket_count - 1 {
                    max + f64::EPSILON
                } else {
                    min + (i + 1) as f64 * bucket_width
                };
                HistogramBucket {
                    lower,
                    upper,
                    count: 0,
                    percentage: 0.0,
                }
            })
            .collect();

        for &sample in samples {
            let bucket_idx = if range > 0.0 {
                ((sample - min) / bucket_width).floor() as usize
            } else {
                0
            };
            let idx = bucket_idx.min(bucket_count - 1);
            buckets[idx].count += 1;
        }

        let total = samples.len();
        for bucket in &mut buckets {
            bucket.percentage = bucket.count as f64 / total as f64 * 100.0;
        }

        let mode_bucket = buckets
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| b.count)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let entropy = shannon_entropy(&buckets, total);

        Self {
            buckets,
            total_samples: total,
            entropy,
            mode_bucket,
            bucket_count,
        }
    }

    /// Get the mode (most frequent) bucket
    pub fn mode(&self) -> Option<&HistogramBucket> {
        self.buckets.get(self.mode_bucket)
    }

    /// Verify bucket counts sum to total
    pub fn verify_counts(&self) -> bool {
        let sum: usize = self.buckets.iter().map(|b| b.count).sum();
        sum == self.total_samples
    }
}

/// Calculate Shannon entropy of histogram (normalized 0-1)
fn shannon_entropy(buckets: &[HistogramBucket], total: usize) -> f64 {
    if total == 0 || buckets.is_empty() {
        return 0.0;
    }

    let mut entropy = 0.0;
    for bucket in buckets {
        if bucket.count > 0 {
            let p = bucket.count as f64 / total as f64;
            entropy -= p * p.ln();
        }
    }

    let max_entropy = (buckets.len() as f64).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}
