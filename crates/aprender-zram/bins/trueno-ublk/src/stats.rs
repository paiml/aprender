//! Statistics module - real-time stats collection and aggregation

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Throughput calculator using exponential moving average
pub struct ThroughputCalculator {
    bytes_total: AtomicU64,
    last_bytes: AtomicU64,
    last_update: std::sync::RwLock<Instant>,
    ema_throughput: std::sync::RwLock<f64>,
    alpha: f64,
}

impl ThroughputCalculator {
    pub fn new() -> Self {
        Self {
            bytes_total: AtomicU64::new(0),
            last_bytes: AtomicU64::new(0),
            last_update: std::sync::RwLock::new(Instant::now()),
            ema_throughput: std::sync::RwLock::new(0.0),
            alpha: 0.3, // EMA smoothing factor
        }
    }

    /// Record bytes processed
    pub fn record_bytes(&self, bytes: u64) {
        self.bytes_total.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Update throughput calculation (call periodically)
    pub fn update(&self) -> f64 {
        let now = Instant::now();
        let mut last_update = self.last_update.write().expect("rwlock poisoned");
        let elapsed = now.duration_since(*last_update);

        if elapsed < Duration::from_millis(100) {
            return *self.ema_throughput.read().expect("rwlock poisoned");
        }

        let current_bytes = self.bytes_total.load(Ordering::Relaxed);
        let last = self.last_bytes.swap(current_bytes, Ordering::Relaxed);
        let bytes_delta = current_bytes.saturating_sub(last);

        let instant_throughput = bytes_delta as f64 / elapsed.as_secs_f64() / 1e9; // GB/s

        let mut ema = self.ema_throughput.write().expect("rwlock poisoned");
        *ema = self.alpha * instant_throughput + (1.0 - self.alpha) * *ema;
        *last_update = now;

        *ema
    }

    /// Get current throughput in GB/s
    pub fn throughput_gbps(&self) -> f64 {
        *self.ema_throughput.read().expect("rwlock poisoned")
    }
}

impl Default for ThroughputCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compression ratio tracker
pub struct RatioTracker {
    orig_bytes: AtomicU64,
    compr_bytes: AtomicU64,
}

impl RatioTracker {
    pub fn new() -> Self {
        Self { orig_bytes: AtomicU64::new(0), compr_bytes: AtomicU64::new(0) }
    }

    /// Record a compression operation
    pub fn record(&self, original: u64, compressed: u64) {
        self.orig_bytes.fetch_add(original, Ordering::Relaxed);
        self.compr_bytes.fetch_add(compressed, Ordering::Relaxed);
    }

    /// Get overall compression ratio
    pub fn ratio(&self) -> f64 {
        let compr = self.compr_bytes.load(Ordering::Relaxed);
        if compr == 0 {
            1.0
        } else {
            self.orig_bytes.load(Ordering::Relaxed) as f64 / compr as f64
        }
    }

    /// Get space savings percentage
    pub fn savings_percent(&self) -> f64 {
        let orig = self.orig_bytes.load(Ordering::Relaxed);
        let compr = self.compr_bytes.load(Ordering::Relaxed);
        if orig == 0 {
            0.0
        } else {
            (1.0 - (compr as f64 / orig as f64)) * 100.0
        }
    }
}

impl Default for RatioTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Entropy distribution tracker
pub struct EntropyTracker {
    low_entropy: AtomicU64,    // < 4 bits/byte (GPU batch)
    medium_entropy: AtomicU64, // 4-7 bits/byte (SIMD)
    high_entropy: AtomicU64,   // > 7 bits/byte (scalar/skip)
    sum: AtomicU64,
    count: AtomicU64,
}

impl EntropyTracker {
    pub fn new() -> Self {
        Self {
            low_entropy: AtomicU64::new(0),
            medium_entropy: AtomicU64::new(0),
            high_entropy: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record a page entropy
    pub fn record(&self, entropy: f64) {
        // Store entropy * 100 as u64 for atomic operations
        let scaled = (entropy * 100.0) as u64;
        self.sum.fetch_add(scaled, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        if entropy < 4.0 {
            self.low_entropy.fetch_add(1, Ordering::Relaxed);
        } else if entropy < 7.0 {
            self.medium_entropy.fetch_add(1, Ordering::Relaxed);
        } else {
            self.high_entropy.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get average entropy
    pub fn average(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            (self.sum.load(Ordering::Relaxed) as f64) / (count as f64) / 100.0
        }
    }

    /// Get distribution percentages
    pub fn distribution(&self) -> (f64, f64, f64) {
        let total = self.low_entropy.load(Ordering::Relaxed)
            + self.medium_entropy.load(Ordering::Relaxed)
            + self.high_entropy.load(Ordering::Relaxed);

        if total == 0 {
            return (0.0, 0.0, 0.0);
        }

        let total = total as f64;
        (
            self.low_entropy.load(Ordering::Relaxed) as f64 / total * 100.0,
            self.medium_entropy.load(Ordering::Relaxed) as f64 / total * 100.0,
            self.high_entropy.load(Ordering::Relaxed) as f64 / total * 100.0,
        )
    }

    /// Get page counts
    pub fn page_counts(&self) -> (u64, u64, u64) {
        (
            self.low_entropy.load(Ordering::Relaxed),
            self.medium_entropy.load(Ordering::Relaxed),
            self.high_entropy.load(Ordering::Relaxed),
        )
    }
}

impl Default for EntropyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throughput_calculator() {
        let calc = ThroughputCalculator::new();
        calc.record_bytes(1_000_000_000); // 1 GB
        std::thread::sleep(Duration::from_millis(150));
        let throughput = calc.update();
        assert!(throughput >= 0.0);
    }

    #[test]
    fn test_ratio_tracker() {
        let tracker = RatioTracker::new();
        tracker.record(1000, 500);
        assert!((tracker.ratio() - 2.0).abs() < 0.01);
        assert!((tracker.savings_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_tracker() {
        let tracker = EntropyTracker::new();
        tracker.record(2.0); // low
        tracker.record(5.0); // medium
        tracker.record(7.5); // high

        let avg = tracker.average();
        assert!((avg - 4.833).abs() < 0.1);

        let (low, med, high) = tracker.distribution();
        assert!((low - 33.3).abs() < 1.0);
        assert!((med - 33.3).abs() < 1.0);
        assert!((high - 33.3).abs() < 1.0);
    }
}
