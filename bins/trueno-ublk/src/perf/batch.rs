//! Page batching for coalesced compression
//!
//! Groups sequential I/O requests into batches of 64-256 pages for:
//!
//! Benefits:
//! - Reduced io_uring submission overhead
//! - Better SIMD utilization (process multiple pages per vector op)
//! - Improved cache locality
//!
//! ## Batch Strategy
//!
//! 1. **Sequential detection**: Pages with consecutive sectors are grouped
//! 2. **Timeout flush**: Partial batches flushed after timeout (avoid latency spikes)
//! 3. **Size threshold**: Flush when batch reaches target size
//!
//! ## Usage
//!
//! ```ignore
//! let config = BatchConfig::default();
//! let mut coalescer = BatchCoalescer::new(config);
//!
//! // Add requests
//! if let Some(batch) = coalescer.add(request) {
//!     // Process batch
//!     process_batch(batch);
//! }
//!
//! // Periodic timeout check
//! if let Some(batch) = coalescer.check_timeout() {
//!     process_batch(batch);
//! }
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Batch configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Minimum batch size (pages)
    pub min_size: usize,

    /// Maximum batch size (pages)
    pub max_size: usize,

    /// Timeout for flushing partial batches
    pub timeout: Duration,

    /// Enable sequential detection (group consecutive sectors)
    pub sequential_detection: bool,

    /// Maximum gap between sectors to still consider sequential (in sectors)
    pub sequential_gap_threshold: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            min_size: 64,
            max_size: 256,
            timeout: Duration::from_micros(100),
            sequential_detection: true,
            sequential_gap_threshold: 0, // Strictly sequential
        }
    }
}

impl BatchConfig {
    /// High-throughput configuration
    pub fn high_throughput() -> Self {
        Self {
            min_size: 128,
            max_size: 256,
            timeout: Duration::from_micros(50),
            sequential_detection: true,
            sequential_gap_threshold: 8, // Allow small gaps
        }
    }

    /// Low-latency configuration (smaller batches, shorter timeout)
    pub fn low_latency() -> Self {
        Self {
            min_size: 32,
            max_size: 64,
            timeout: Duration::from_micros(25),
            sequential_detection: true,
            sequential_gap_threshold: 0,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.min_size == 0 {
            return Err("min_size must be > 0");
        }
        if self.max_size < self.min_size {
            return Err("max_size must be >= min_size");
        }
        if self.max_size > 4096 {
            return Err("max_size must be <= 4096");
        }
        Ok(())
    }
}

/// A page request to be batched
#[derive(Debug, Clone)]
pub struct PageRequest {
    /// Starting sector
    pub sector: u64,

    /// Number of sectors
    pub nr_sectors: u32,

    /// Tag from ublk
    pub tag: u16,

    /// Queue ID
    pub queue_id: u16,

    /// Operation type (read/write)
    pub is_write: bool,

    /// Timestamp when request was received
    pub timestamp: Instant,
}

impl PageRequest {
    /// Create a new page request
    pub fn new(sector: u64, nr_sectors: u32, tag: u16, queue_id: u16, is_write: bool) -> Self {
        Self { sector, nr_sectors, tag, queue_id, is_write, timestamp: Instant::now() }
    }

    /// End sector (exclusive)
    pub fn end_sector(&self) -> u64 {
        self.sector + self.nr_sectors as u64
    }

    /// Byte length
    pub fn byte_len(&self) -> usize {
        self.nr_sectors as usize * 512
    }
}

/// A batch of coalesced page requests
#[derive(Debug, Clone)]
pub struct PageBatch {
    /// Requests in this batch
    pub requests: Vec<PageRequest>,

    /// Whether all requests are sequential
    pub is_sequential: bool,

    /// Starting sector of first request
    pub start_sector: u64,

    /// Total sectors in batch
    pub total_sectors: u64,

    /// Batch creation timestamp
    pub created_at: Instant,

    /// Whether this is a write batch
    pub is_write: bool,
}

impl PageBatch {
    /// Create empty batch
    fn new(is_write: bool) -> Self {
        Self {
            requests: Vec::new(),
            is_sequential: true,
            start_sector: 0,
            total_sectors: 0,
            created_at: Instant::now(),
            is_write,
        }
    }

    /// Number of pages in batch
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Total bytes in batch
    pub fn total_bytes(&self) -> usize {
        self.total_sectors as usize * 512
    }

    /// Average wait time for requests in batch
    pub fn avg_wait_time(&self) -> Duration {
        if self.requests.is_empty() {
            return Duration::ZERO;
        }
        let now = Instant::now();
        let total_wait: Duration =
            self.requests.iter().map(|r| now.duration_since(r.timestamp)).sum();
        total_wait / self.requests.len() as u32
    }

    /// Maximum wait time (oldest request)
    pub fn max_wait_time(&self) -> Duration {
        self.requests.first().map(|r| r.timestamp.elapsed()).unwrap_or(Duration::ZERO)
    }
}

/// Batch coalescer - groups page requests into batches
pub struct BatchCoalescer {
    config: BatchConfig,
    read_pending: VecDeque<PageRequest>,
    write_pending: VecDeque<PageRequest>,
    last_flush: Instant,
    stats: BatchStats,
}

/// Statistics for batch coalescing
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total batches created
    pub batches_created: u64,

    /// Batches flushed due to size threshold
    pub size_flushes: u64,

    /// Batches flushed due to timeout
    pub timeout_flushes: u64,

    /// Total pages batched
    pub pages_batched: u64,

    /// Sequential batches (all pages consecutive)
    pub sequential_batches: u64,

    /// Non-sequential batches
    pub non_sequential_batches: u64,

    /// Minimum batch size
    pub min_batch_size: usize,

    /// Maximum batch size
    pub max_batch_size: usize,
}

impl BatchStats {
    /// Average batch size
    pub fn avg_batch_size(&self) -> f64 {
        if self.batches_created == 0 {
            return 0.0;
        }
        self.pages_batched as f64 / self.batches_created as f64
    }

    /// Sequential batch rate
    pub fn sequential_rate(&self) -> f64 {
        let total = self.sequential_batches + self.non_sequential_batches;
        if total == 0 {
            return 0.0;
        }
        self.sequential_batches as f64 / total as f64
    }
}

impl BatchCoalescer {
    /// Create new batch coalescer
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            read_pending: VecDeque::with_capacity(256),
            write_pending: VecDeque::with_capacity(256),
            last_flush: Instant::now(),
            stats: BatchStats::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Number of pending read requests
    pub fn pending_reads(&self) -> usize {
        self.read_pending.len()
    }

    /// Number of pending write requests
    pub fn pending_writes(&self) -> usize {
        self.write_pending.len()
    }

    /// Add a page request, returns batch if ready
    pub fn add(&mut self, request: PageRequest) -> Option<PageBatch> {
        let is_write = request.is_write;
        let pending = if is_write { &mut self.write_pending } else { &mut self.read_pending };

        pending.push_back(request);

        // Check if batch is ready (size threshold)
        if pending.len() >= self.config.max_size {
            return Some(self.flush_pending(is_write, FlushReason::Size));
        }

        None
    }

    /// Check for timeout and flush if needed
    pub fn check_timeout(&mut self) -> Option<PageBatch> {
        let now = Instant::now();

        // Check reads
        if !self.read_pending.is_empty() {
            if let Some(oldest) = self.read_pending.front() {
                if now.duration_since(oldest.timestamp) >= self.config.timeout {
                    return Some(self.flush_pending(false, FlushReason::Timeout));
                }
            }
        }

        // Check writes
        if !self.write_pending.is_empty() {
            if let Some(oldest) = self.write_pending.front() {
                if now.duration_since(oldest.timestamp) >= self.config.timeout {
                    return Some(self.flush_pending(true, FlushReason::Timeout));
                }
            }
        }

        None
    }

    /// Force flush all pending requests
    pub fn flush_all(&mut self) -> Vec<PageBatch> {
        let mut batches = Vec::new();

        if !self.read_pending.is_empty() {
            batches.push(self.flush_pending(false, FlushReason::Forced));
        }

        if !self.write_pending.is_empty() {
            batches.push(self.flush_pending(true, FlushReason::Forced));
        }

        batches
    }

    /// Flush pending requests of given type
    fn flush_pending(&mut self, is_write: bool, reason: FlushReason) -> PageBatch {
        let pending = if is_write { &mut self.write_pending } else { &mut self.read_pending };

        let mut batch = PageBatch::new(is_write);
        let mut last_end_sector: Option<u64> = None;
        let mut is_sequential = true;

        while let Some(req) = pending.pop_front() {
            // Check sequentiality
            if self.config.sequential_detection {
                if let Some(last_end) = last_end_sector {
                    let gap = if req.sector >= last_end {
                        req.sector - last_end
                    } else {
                        u64::MAX // Out of order
                    };
                    if gap > self.config.sequential_gap_threshold {
                        is_sequential = false;
                    }
                }
                last_end_sector = Some(req.end_sector());
            }

            if batch.requests.is_empty() {
                batch.start_sector = req.sector;
            }
            batch.total_sectors += req.nr_sectors as u64;
            batch.requests.push(req);

            // Stop at max size
            if batch.requests.len() >= self.config.max_size {
                break;
            }
        }

        batch.is_sequential = is_sequential;

        // Update stats
        self.stats.batches_created += 1;
        self.stats.pages_batched += batch.requests.len() as u64;

        if batch.is_sequential {
            self.stats.sequential_batches += 1;
        } else {
            self.stats.non_sequential_batches += 1;
        }

        match reason {
            FlushReason::Size => self.stats.size_flushes += 1,
            FlushReason::Timeout => self.stats.timeout_flushes += 1,
            FlushReason::Forced => {}
        }

        let batch_size = batch.requests.len();
        if self.stats.min_batch_size == 0 || batch_size < self.stats.min_batch_size {
            self.stats.min_batch_size = batch_size;
        }
        if batch_size > self.stats.max_batch_size {
            self.stats.max_batch_size = batch_size;
        }

        self.last_flush = Instant::now();
        batch
    }
}

#[derive(Debug, Clone, Copy)]
enum FlushReason {
    Size,
    Timeout,
    Forced,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // BatchConfig Tests
    // ============================================================================

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.min_size, 64);
        assert_eq!(config.max_size, 256);
        assert_eq!(config.timeout, Duration::from_micros(100));
        assert!(config.sequential_detection);
    }

    #[test]
    fn test_batch_config_high_throughput() {
        let config = BatchConfig::high_throughput();
        assert_eq!(config.min_size, 128);
        assert_eq!(config.max_size, 256);
        assert_eq!(config.sequential_gap_threshold, 8);
    }

    #[test]
    fn test_batch_config_low_latency() {
        let config = BatchConfig::low_latency();
        assert_eq!(config.min_size, 32);
        assert_eq!(config.max_size, 64);
        assert_eq!(config.timeout, Duration::from_micros(25));
    }

    #[test]
    fn test_batch_config_validate_success() {
        let config = BatchConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_batch_config_validate_zero_min() {
        let mut config = BatchConfig::default();
        config.min_size = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_batch_config_validate_max_less_than_min() {
        let mut config = BatchConfig::default();
        config.min_size = 100;
        config.max_size = 50;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_batch_config_validate_max_too_large() {
        let mut config = BatchConfig::default();
        config.max_size = 5000;
        assert!(config.validate().is_err());
    }

    // ============================================================================
    // PageRequest Tests
    // ============================================================================

    #[test]
    fn test_page_request_new() {
        let req = PageRequest::new(100, 8, 5, 0, false);
        assert_eq!(req.sector, 100);
        assert_eq!(req.nr_sectors, 8);
        assert_eq!(req.tag, 5);
        assert_eq!(req.queue_id, 0);
        assert!(!req.is_write);
    }

    #[test]
    fn test_page_request_end_sector() {
        let req = PageRequest::new(100, 8, 0, 0, false);
        assert_eq!(req.end_sector(), 108);
    }

    #[test]
    fn test_page_request_byte_len() {
        let req = PageRequest::new(0, 8, 0, 0, false);
        assert_eq!(req.byte_len(), 4096); // 8 sectors * 512 bytes
    }

    // ============================================================================
    // PageBatch Tests
    // ============================================================================

    #[test]
    fn test_page_batch_empty() {
        let batch = PageBatch::new(false);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.total_bytes(), 0);
    }

    #[test]
    fn test_page_batch_total_bytes() {
        let mut batch = PageBatch::new(false);
        batch.total_sectors = 16;
        assert_eq!(batch.total_bytes(), 8192); // 16 * 512
    }

    // ============================================================================
    // BatchCoalescer Tests
    // ============================================================================

    #[test]
    fn test_batch_coalescer_new() {
        let config = BatchConfig::default();
        let coalescer = BatchCoalescer::new(config);
        assert_eq!(coalescer.pending_reads(), 0);
        assert_eq!(coalescer.pending_writes(), 0);
    }

    #[test]
    fn test_batch_coalescer_add_single() {
        let mut config = BatchConfig::default();
        config.max_size = 10; // Small for testing
        let mut coalescer = BatchCoalescer::new(config);

        let req = PageRequest::new(0, 8, 0, 0, false);
        let result = coalescer.add(req);

        assert!(result.is_none()); // Not enough for batch
        assert_eq!(coalescer.pending_reads(), 1);
    }

    #[test]
    fn test_batch_coalescer_flush_on_size() {
        let mut config = BatchConfig::default();
        config.max_size = 5;
        let mut coalescer = BatchCoalescer::new(config);

        // Add 5 requests
        for i in 0..5 {
            let req = PageRequest::new(i * 8, 8, i as u16, 0, false);
            let result = coalescer.add(req);

            if i < 4 {
                assert!(result.is_none());
            } else {
                // 5th request should trigger flush
                assert!(result.is_some());
                let batch = result.unwrap();
                assert_eq!(batch.len(), 5);
                assert!(batch.is_sequential);
            }
        }

        assert_eq!(coalescer.stats().size_flushes, 1);
    }

    #[test]
    fn test_batch_coalescer_sequential_detection() {
        let mut config = BatchConfig::default();
        config.max_size = 10; // Large enough to not trigger flush during add
        config.sequential_gap_threshold = 0;
        let mut coalescer = BatchCoalescer::new(config);

        // Add sequential requests
        for i in 0..5 {
            let req = PageRequest::new(i * 8, 8, i as u16, 0, false);
            coalescer.add(req);
        }

        // Verify pending
        assert_eq!(coalescer.pending_reads(), 5);

        // All sequential
        let batches = coalescer.flush_all();
        assert_eq!(batches.len(), 1);
        assert!(batches[0].is_sequential);
        assert_eq!(batches[0].len(), 5);
    }

    #[test]
    fn test_batch_coalescer_non_sequential_detection() {
        let mut config = BatchConfig::default();
        config.max_size = 5;
        config.sequential_gap_threshold = 0;
        let mut coalescer = BatchCoalescer::new(config);

        // Add non-sequential requests (gap in middle)
        coalescer.add(PageRequest::new(0, 8, 0, 0, false));
        coalescer.add(PageRequest::new(8, 8, 1, 0, false));
        coalescer.add(PageRequest::new(100, 8, 2, 0, false)); // Gap!
        coalescer.add(PageRequest::new(108, 8, 3, 0, false));

        let batches = coalescer.flush_all();
        assert_eq!(batches.len(), 1);
        assert!(!batches[0].is_sequential);
    }

    #[test]
    fn test_batch_coalescer_sequential_with_gap_threshold() {
        let mut config = BatchConfig::default();
        config.max_size = 5;
        config.sequential_gap_threshold = 8; // Allow gap of 8 sectors
        let mut coalescer = BatchCoalescer::new(config);

        // Add requests with small gap
        coalescer.add(PageRequest::new(0, 8, 0, 0, false));
        coalescer.add(PageRequest::new(16, 8, 1, 0, false)); // Gap of 8

        let batches = coalescer.flush_all();
        assert!(batches[0].is_sequential); // Within threshold
    }

    #[test]
    fn test_batch_coalescer_separate_read_write() {
        let mut config = BatchConfig::default();
        config.max_size = 10;
        let mut coalescer = BatchCoalescer::new(config);

        // Add reads and writes
        coalescer.add(PageRequest::new(0, 8, 0, 0, false)); // read
        coalescer.add(PageRequest::new(8, 8, 1, 0, true)); // write
        coalescer.add(PageRequest::new(16, 8, 2, 0, false)); // read

        assert_eq!(coalescer.pending_reads(), 2);
        assert_eq!(coalescer.pending_writes(), 1);

        let batches = coalescer.flush_all();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_batch_coalescer_stats() {
        let mut config = BatchConfig::default();
        config.max_size = 5;
        let mut coalescer = BatchCoalescer::new(config);

        // Create two batches
        for i in 0..10 {
            coalescer.add(PageRequest::new(i * 8, 8, i as u16, 0, false));
        }

        assert_eq!(coalescer.stats().batches_created, 2);
        assert_eq!(coalescer.stats().pages_batched, 10);
        assert_eq!(coalescer.stats().size_flushes, 2);
    }

    #[test]
    fn test_batch_coalescer_timeout_check_no_pending() {
        let config = BatchConfig::default();
        let mut coalescer = BatchCoalescer::new(config);

        let result = coalescer.check_timeout();
        assert!(result.is_none());
    }

    #[test]
    fn test_batch_coalescer_timeout_check_not_expired() {
        let mut config = BatchConfig::default();
        config.timeout = Duration::from_secs(10); // Long timeout
        let mut coalescer = BatchCoalescer::new(config);

        coalescer.add(PageRequest::new(0, 8, 0, 0, false));

        let result = coalescer.check_timeout();
        assert!(result.is_none());
    }

    #[test]
    fn test_batch_stats_avg_batch_size() {
        let stats = BatchStats { batches_created: 4, pages_batched: 100, ..Default::default() };
        assert!((stats.avg_batch_size() - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_stats_avg_batch_size_zero() {
        let stats = BatchStats::default();
        assert_eq!(stats.avg_batch_size(), 0.0);
    }

    #[test]
    fn test_batch_stats_sequential_rate() {
        let stats =
            BatchStats { sequential_batches: 8, non_sequential_batches: 2, ..Default::default() };
        assert!((stats.sequential_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_batch_stats_sequential_rate_zero() {
        let stats = BatchStats::default();
        assert_eq!(stats.sequential_rate(), 0.0);
    }

    #[test]
    fn test_batch_coalescer_min_max_batch_tracking() {
        let mut config = BatchConfig::default();
        config.max_size = 10;
        let mut coalescer = BatchCoalescer::new(config);

        // First batch of 5
        for i in 0..5 {
            coalescer.add(PageRequest::new(i * 8, 8, i as u16, 0, false));
        }
        coalescer.flush_all();

        // Second batch of 10
        for i in 0..10 {
            coalescer.add(PageRequest::new(i * 8, 8, i as u16, 0, false));
        }

        assert_eq!(coalescer.stats().min_batch_size, 5);
        assert_eq!(coalescer.stats().max_batch_size, 10);
    }

    #[test]
    fn test_batch_total_sectors() {
        let mut config = BatchConfig::default();
        config.max_size = 5;
        let mut coalescer = BatchCoalescer::new(config);

        // Add requests with varying sector counts
        coalescer.add(PageRequest::new(0, 8, 0, 0, false));
        coalescer.add(PageRequest::new(8, 16, 1, 0, false));
        coalescer.add(PageRequest::new(24, 8, 2, 0, false));

        let batches = coalescer.flush_all();
        assert_eq!(batches[0].total_sectors, 8 + 16 + 8);
    }

    #[test]
    fn test_batch_start_sector() {
        let mut config = BatchConfig::default();
        config.max_size = 5;
        let mut coalescer = BatchCoalescer::new(config);

        coalescer.add(PageRequest::new(100, 8, 0, 0, false));
        coalescer.add(PageRequest::new(108, 8, 1, 0, false));

        let batches = coalescer.flush_all();
        assert_eq!(batches[0].start_sector, 100);
    }

    #[test]
    fn test_page_request_clone() {
        let req = PageRequest::new(100, 8, 5, 0, true);
        let cloned = req.clone();
        assert_eq!(req.sector, cloned.sector);
        assert_eq!(req.is_write, cloned.is_write);
    }

    #[test]
    fn test_page_batch_clone() {
        let batch = PageBatch::new(true);
        let cloned = batch.clone();
        assert_eq!(batch.is_write, cloned.is_write);
        assert_eq!(batch.is_sequential, cloned.is_sequential);
    }

    #[test]
    fn test_batch_config_clone() {
        let config = BatchConfig::high_throughput();
        let cloned = config.clone();
        assert_eq!(config.max_size, cloned.max_size);
    }
}
