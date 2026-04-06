
impl DispatchMetrics {
    /// Histogram bucket boundaries in microseconds
    pub const BUCKET_BOUNDARIES: [u64; 4] = [100, 500, 1000, 5000];

    /// Create new metrics tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            cpu_dispatches: std::sync::atomic::AtomicUsize::new(0),
            gpu_dispatches: std::sync::atomic::AtomicUsize::new(0),
            cpu_latency_count: std::sync::atomic::AtomicUsize::new(0),
            cpu_latency_sum_us: std::sync::atomic::AtomicU64::new(0),
            gpu_latency_count: std::sync::atomic::AtomicUsize::new(0),
            gpu_latency_sum_us: std::sync::atomic::AtomicU64::new(0),
            cpu_latency_buckets: [
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
            ],
            gpu_latency_buckets: [
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
                std::sync::atomic::AtomicUsize::new(0),
            ],
            cpu_latency_min_us: std::sync::atomic::AtomicU64::new(u64::MAX),
            cpu_latency_max_us: std::sync::atomic::AtomicU64::new(0),
            gpu_latency_min_us: std::sync::atomic::AtomicU64::new(u64::MAX),
            gpu_latency_max_us: std::sync::atomic::AtomicU64::new(0),
            cpu_latency_sum_sq_us: std::sync::atomic::AtomicU64::new(0),
            gpu_latency_sum_sq_us: std::sync::atomic::AtomicU64::new(0),
            start_time_ms: std::sync::atomic::AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            ),
        }
    }

    fn bucket_index(latency_us: u64) -> usize {
        for (i, &boundary) in Self::BUCKET_BOUNDARIES.iter().enumerate() {
            if latency_us < boundary {
                return i;
            }
        }
        4
    }

    /// Record a CPU dispatch
    pub fn record_cpu_dispatch(&self) {
        self.cpu_dispatches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a GPU dispatch
    pub fn record_gpu_dispatch(&self) {
        self.gpu_dispatches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record CPU dispatch latency
    pub fn record_cpu_latency(&self, latency: std::time::Duration) {
        let latency_us = latency.as_micros() as u64;
        self.cpu_latency_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_us
            .fetch_add(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_buckets[Self::bucket_index(latency_us)]
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_min_us
            .fetch_min(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_max_us
            .fetch_max(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_sq_us.fetch_add(
            latency_us * latency_us,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Record GPU dispatch latency
    pub fn record_gpu_latency(&self, latency: std::time::Duration) {
        let latency_us = latency.as_micros() as u64;
        self.gpu_latency_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_us
            .fetch_add(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_buckets[Self::bucket_index(latency_us)]
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_min_us
            .fetch_min(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_max_us
            .fetch_max(latency_us, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_sq_us.fetch_add(
            latency_us * latency_us,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Get CPU dispatch count
    #[must_use]
    pub fn cpu_dispatches(&self) -> usize {
        self.cpu_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get GPU dispatch count
    #[must_use]
    pub fn gpu_dispatches(&self) -> usize {
        self.gpu_dispatches
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total dispatches
    #[must_use]
    pub fn total_dispatches(&self) -> usize {
        self.cpu_dispatches() + self.gpu_dispatches()
    }

    /// Get GPU dispatch ratio
    #[must_use]
    pub fn gpu_ratio(&self) -> f64 {
        let total = self.total_dispatches();
        if total == 0 {
            0.0
        } else {
            self.gpu_dispatches() as f64 / total as f64
        }
    }

    /// Get CPU latency count
    #[must_use]
    pub fn cpu_latency_count(&self) -> usize {
        self.cpu_latency_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get GPU latency count
    #[must_use]
    pub fn gpu_latency_count(&self) -> usize {
        self.gpu_latency_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get mean CPU latency in microseconds
    #[must_use]
    pub fn cpu_latency_mean_us(&self) -> f64 {
        let count = self.cpu_latency_count();
        if count == 0 {
            0.0
        } else {
            self.cpu_latency_sum_us
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / count as f64
        }
    }

    /// Get mean GPU latency in microseconds
    #[must_use]
    pub fn gpu_latency_mean_us(&self) -> f64 {
        let count = self.gpu_latency_count();
        if count == 0 {
            0.0
        } else {
            self.gpu_latency_sum_us
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / count as f64
        }
    }

    /// Get CPU latency sum
    #[must_use]
    pub fn cpu_latency_sum_us(&self) -> u64 {
        self.cpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get GPU latency sum
    #[must_use]
    pub fn gpu_latency_sum_us(&self) -> u64 {
        self.gpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get CPU latency min
    #[must_use]
    pub fn cpu_latency_min_us(&self) -> u64 {
        if self.cpu_latency_count() == 0 {
            0
        } else {
            self.cpu_latency_min_us
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    /// Get CPU latency max
    #[must_use]
    pub fn cpu_latency_max_us(&self) -> u64 {
        self.cpu_latency_max_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get GPU latency min
    #[must_use]
    pub fn gpu_latency_min_us(&self) -> u64 {
        if self.gpu_latency_count() == 0 {
            0
        } else {
            self.gpu_latency_min_us
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    /// Get GPU latency max
    #[must_use]
    pub fn gpu_latency_max_us(&self) -> u64 {
        self.gpu_latency_max_us
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get CPU latency variance
    #[must_use]
    pub fn cpu_latency_variance_us(&self) -> f64 {
        let count = self.cpu_latency_count();
        if count < 2 {
            return 0.0;
        }
        let sum = self
            .cpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let sum_sq = self
            .cpu_latency_sum_sq_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let n = count as f64;
        (sum_sq / n) - (sum / n).powi(2)
    }

    /// Get CPU latency stddev
    #[must_use]
    pub fn cpu_latency_stddev_us(&self) -> f64 {
        self.cpu_latency_variance_us().sqrt()
    }

    /// Get GPU latency variance
    #[must_use]
    pub fn gpu_latency_variance_us(&self) -> f64 {
        let count = self.gpu_latency_count();
        if count < 2 {
            return 0.0;
        }
        let sum = self
            .gpu_latency_sum_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let sum_sq = self
            .gpu_latency_sum_sq_us
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let n = count as f64;
        (sum_sq / n) - (sum / n).powi(2)
    }

    /// Get GPU latency stddev
    #[must_use]
    pub fn gpu_latency_stddev_us(&self) -> f64 {
        self.gpu_latency_variance_us().sqrt()
    }

    /// Get CPU latency histogram buckets
    #[must_use]
    pub fn cpu_latency_buckets(&self) -> [usize; 5] {
        [
            self.cpu_latency_buckets[0].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[1].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[2].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[3].load(std::sync::atomic::Ordering::Relaxed),
            self.cpu_latency_buckets[4].load(std::sync::atomic::Ordering::Relaxed),
        ]
    }

    /// Get GPU latency histogram buckets
    #[must_use]
    pub fn gpu_latency_buckets(&self) -> [usize; 5] {
        [
            self.gpu_latency_buckets[0].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[1].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[2].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[3].load(std::sync::atomic::Ordering::Relaxed),
            self.gpu_latency_buckets[4].load(std::sync::atomic::Ordering::Relaxed),
        ]
    }

    fn estimate_percentile_from_buckets(buckets: &[usize; 5], percentile: f64) -> f64 {
        const BUCKET_UPPER_BOUNDS: [f64; 5] = [100.0, 500.0, 1000.0, 5000.0, 10000.0];
        const BUCKET_LOWER_BOUNDS: [f64; 5] = [0.0, 100.0, 500.0, 1000.0, 5000.0];
        let total: usize = buckets.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let target_rank = (percentile / 100.0) * total as f64;
        let mut cumulative: f64 = 0.0;
        for (i, &count) in buckets.iter().enumerate() {
            let prev_cumulative = cumulative;
            cumulative += count as f64;
            if cumulative >= target_rank {
                if count == 0 {
                    return BUCKET_LOWER_BOUNDS[i];
                }
                let fraction = (target_rank - prev_cumulative) / count as f64;
                return BUCKET_LOWER_BOUNDS[i]
                    + fraction * (BUCKET_UPPER_BOUNDS[i] - BUCKET_LOWER_BOUNDS[i]);
            }
        }
        BUCKET_UPPER_BOUNDS[4]
    }

    /// Get CPU p50 latency
    #[must_use]
    pub fn cpu_latency_p50_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.cpu_latency_buckets(), 50.0)
    }

    /// Get CPU p95 latency
    #[must_use]
    pub fn cpu_latency_p95_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.cpu_latency_buckets(), 95.0)
    }

    /// Get CPU p99 latency
    #[must_use]
    pub fn cpu_latency_p99_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.cpu_latency_buckets(), 99.0)
    }

    /// Get GPU p50 latency
    #[must_use]
    pub fn gpu_latency_p50_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.gpu_latency_buckets(), 50.0)
    }

    /// Get GPU p95 latency
    #[must_use]
    pub fn gpu_latency_p95_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.gpu_latency_buckets(), 95.0)
    }

    /// Get GPU p99 latency
    #[must_use]
    pub fn gpu_latency_p99_us(&self) -> f64 {
        Self::estimate_percentile_from_buckets(&self.gpu_latency_buckets(), 99.0)
    }

    /// Get bucket boundaries as strings
    #[must_use]
    pub fn bucket_boundaries_us(&self) -> Vec<String> {
        vec![
            format!("0-{}", Self::BUCKET_BOUNDARIES[0]),
            format!(
                "{}-{}",
                Self::BUCKET_BOUNDARIES[0],
                Self::BUCKET_BOUNDARIES[1]
            ),
            format!(
                "{}-{}",
                Self::BUCKET_BOUNDARIES[1],
                Self::BUCKET_BOUNDARIES[2]
            ),
            format!(
                "{}-{}",
                Self::BUCKET_BOUNDARIES[2],
                Self::BUCKET_BOUNDARIES[3]
            ),
            format!("{}+", Self::BUCKET_BOUNDARIES[3]),
        ]
    }

    /// Get start time
    #[must_use]
    pub fn start_time_ms(&self) -> u64 {
        self.start_time_ms
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get elapsed seconds
    #[must_use]
    pub fn elapsed_seconds(&self) -> f64 {
        let start = self.start_time_ms();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        (now.saturating_sub(start)) as f64 / 1000.0
    }

    /// Get throughput
    #[must_use]
    pub fn throughput_rps(&self) -> f64 {
        let elapsed = self.elapsed_seconds();
        if elapsed < 0.001 {
            0.0
        } else {
            self.total_dispatches() as f64 / elapsed
        }
    }

    /// Get CPU latency CV
    #[must_use]
    pub fn cpu_latency_cv(&self) -> f64 {
        let mean = self.cpu_latency_mean_us();
        if mean < 0.001 {
            0.0
        } else {
            (self.cpu_latency_stddev_us() / mean) * 100.0
        }
    }

    /// Get GPU latency CV
    #[must_use]
    pub fn gpu_latency_cv(&self) -> f64 {
        let mean = self.gpu_latency_mean_us();
        if mean < 0.001 {
            0.0
        } else {
            (self.gpu_latency_stddev_us() / mean) * 100.0
        }
    }

    /// Get CPU/GPU speedup
    #[must_use]
    pub fn cpu_gpu_speedup(&self) -> f64 {
        let gpu_mean = self.gpu_latency_mean_us();
        if gpu_mean < 0.001 {
            0.0
        } else {
            self.cpu_latency_mean_us() / gpu_mean
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.cpu_dispatches
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_dispatches
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_min_us
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_max_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_min_us
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_max_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.cpu_latency_sum_sq_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.gpu_latency_sum_sq_us
            .store(0, std::sync::atomic::Ordering::Relaxed);
        for bucket in &self.cpu_latency_buckets {
            bucket.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        for bucket in &self.gpu_latency_buckets {
            bucket.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.start_time_ms
            .store(now, std::sync::atomic::Ordering::Relaxed);
    }
}
