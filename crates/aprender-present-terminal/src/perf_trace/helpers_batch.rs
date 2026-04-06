// ============================================================================
// BATCH PROCESSING & WORK QUEUE HELPERS (trueno-viz parity)
// ============================================================================

/// O(1) batch processor for fixed-size batch accumulation.
///
/// Accumulates items until batch is full, then signals ready for processing.
/// Common pattern for batching network writes, disk flushes, metric exports.
#[derive(Debug, Clone)]
pub struct BatchProcessor {
    /// Current batch count
    count: u64,
    /// Batch size threshold
    batch_size: u64,
    /// Total batches completed
    batches_completed: u64,
    /// Total items processed
    total_items: u64,
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new(100)
    }
}

impl BatchProcessor {
    /// Create with specified batch size
    #[must_use]
    pub fn new(batch_size: u64) -> Self {
        Self {
            count: 0,
            batch_size: batch_size.max(1),
            batches_completed: 0,
            total_items: 0,
        }
    }

    /// Create for network operations (batch size 1000)
    #[must_use]
    pub fn for_network() -> Self {
        Self::new(1000)
    }

    /// Create for disk operations (batch size 100)
    #[must_use]
    pub fn for_disk() -> Self {
        Self::new(100)
    }

    /// Create for metrics export (batch size 50)
    #[must_use]
    pub fn for_metrics() -> Self {
        Self::new(50)
    }

    /// Add item to batch, returns true if batch is now full
    pub fn add(&mut self) -> bool {
        self.count += 1;
        self.total_items += 1;
        if self.count >= self.batch_size {
            self.count = 0;
            self.batches_completed += 1;
            true
        } else {
            false
        }
    }

    /// Add multiple items, returns number of batches completed
    pub fn add_many(&mut self, n: u64) -> u64 {
        self.total_items += n;
        let new_count = self.count + n;
        let batches = new_count / self.batch_size;
        self.count = new_count % self.batch_size;
        self.batches_completed += batches;
        batches
    }

    /// Check if batch is ready (full)
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.count >= self.batch_size
    }

    /// Get current batch fill percentage
    #[must_use]
    pub fn fill_percentage(&self) -> f64 {
        (self.count as f64 / self.batch_size as f64) * 100.0
    }

    /// Get items remaining until full batch
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.batch_size.saturating_sub(self.count)
    }

    /// Get total batches completed
    #[must_use]
    pub fn batches_completed(&self) -> u64 {
        self.batches_completed
    }

    /// Get total items processed
    #[must_use]
    pub fn total_items(&self) -> u64 {
        self.total_items
    }

    /// Flush current batch (mark complete regardless of count)
    pub fn flush(&mut self) {
        if self.count > 0 {
            self.count = 0;
            self.batches_completed += 1;
        }
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        self.count = 0;
        self.batches_completed = 0;
        self.total_items = 0;
    }
}

/// O(1) pipeline stage latency and throughput tracker.
///
/// Tracks items entering and exiting a pipeline stage for monitoring
/// processing latency, queue depth, and throughput.
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Items currently in stage
    in_flight: u64,
    /// Peak in-flight items
    peak_in_flight: u64,
    /// Total items entered
    entered: u64,
    /// Total items exited
    exited: u64,
    /// Total latency (for average calculation)
    total_latency_us: u64,
}

impl Default for PipelineStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage {
    /// Create new pipeline stage tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            in_flight: 0,
            peak_in_flight: 0,
            entered: 0,
            exited: 0,
            total_latency_us: 0,
        }
    }

    /// Record item entering the stage
    pub fn enter(&mut self) {
        self.in_flight += 1;
        self.entered += 1;
        if self.in_flight > self.peak_in_flight {
            self.peak_in_flight = self.in_flight;
        }
    }

    /// Record item exiting the stage with latency in microseconds
    pub fn exit(&mut self, latency_us: u64) {
        self.in_flight = self.in_flight.saturating_sub(1);
        self.exited += 1;
        self.total_latency_us += latency_us;
    }

    /// Record item exiting without latency tracking
    pub fn exit_simple(&mut self) {
        self.in_flight = self.in_flight.saturating_sub(1);
        self.exited += 1;
    }

    /// Get current queue depth
    #[must_use]
    pub fn depth(&self) -> u64 {
        self.in_flight
    }

    /// Get peak queue depth
    #[must_use]
    pub fn peak_depth(&self) -> u64 {
        self.peak_in_flight
    }

    /// Get average latency in microseconds
    #[must_use]
    pub fn avg_latency_us(&self) -> f64 {
        if self.exited == 0 {
            0.0
        } else {
            self.total_latency_us as f64 / self.exited as f64
        }
    }

    /// Get average latency in milliseconds
    #[must_use]
    pub fn avg_latency_ms(&self) -> f64 {
        self.avg_latency_us() / 1000.0
    }

    /// Get throughput (items processed)
    #[must_use]
    pub fn throughput(&self) -> u64 {
        self.exited
    }

    /// Get total items that entered
    #[must_use]
    pub fn total_entered(&self) -> u64 {
        self.entered
    }

    /// Check if stage is idle (nothing in flight)
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.in_flight == 0
    }

    /// Check if stage is backlogged (depth > threshold)
    #[must_use]
    pub fn is_backlogged(&self, threshold: u64) -> bool {
        self.in_flight > threshold
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        self.in_flight = 0;
        self.peak_in_flight = 0;
        self.entered = 0;
        self.exited = 0;
        self.total_latency_us = 0;
    }
}

/// O(1) work queue metrics tracker.
///
/// Tracks enqueue/dequeue operations, wait times, and queue health.
#[derive(Debug, Clone)]
pub struct WorkQueue {
    /// Current queue size
    size: u64,
    /// Peak queue size
    peak_size: u64,
    /// Total enqueued
    enqueued: u64,
    /// Total dequeued
    dequeued: u64,
    /// Total wait time (for average)
    total_wait_us: u64,
    /// Capacity limit (0 = unlimited)
    capacity: u64,
}

impl Default for WorkQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkQueue {
    /// Create unbounded work queue tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            size: 0,
            peak_size: 0,
            enqueued: 0,
            dequeued: 0,
            total_wait_us: 0,
            capacity: 0,
        }
    }

    /// Create bounded work queue tracker
    #[must_use]
    pub fn with_capacity(capacity: u64) -> Self {
        Self {
            capacity,
            ..Self::new()
        }
    }

    /// Enqueue item
    pub fn enqueue(&mut self) -> bool {
        if self.capacity > 0 && self.size >= self.capacity {
            return false; // Would exceed capacity
        }
        self.size += 1;
        self.enqueued += 1;
        if self.size > self.peak_size {
            self.peak_size = self.size;
        }
        true
    }

    /// Dequeue item with wait time in microseconds
    pub fn dequeue(&mut self, wait_us: u64) -> bool {
        if self.size == 0 {
            return false;
        }
        self.size -= 1;
        self.dequeued += 1;
        self.total_wait_us += wait_us;
        true
    }

    /// Dequeue without wait time tracking
    pub fn dequeue_simple(&mut self) -> bool {
        if self.size == 0 {
            return false;
        }
        self.size -= 1;
        self.dequeued += 1;
        true
    }

    /// Get current queue size
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get peak queue size
    #[must_use]
    pub fn peak_size(&self) -> u64 {
        self.peak_size
    }

    /// Get average wait time in microseconds
    #[must_use]
    pub fn avg_wait_us(&self) -> f64 {
        if self.dequeued == 0 {
            0.0
        } else {
            self.total_wait_us as f64 / self.dequeued as f64
        }
    }

    /// Get queue utilization (current/capacity)
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.size as f64 / self.capacity as f64) * 100.0
        }
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Check if queue is full (bounded only)
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.capacity > 0 && self.size >= self.capacity
    }

    /// Get remaining capacity (0 if unbounded)
    #[must_use]
    pub fn remaining_capacity(&self) -> u64 {
        if self.capacity == 0 {
            u64::MAX
        } else {
            self.capacity.saturating_sub(self.size)
        }
    }

    /// Get total enqueued
    #[must_use]
    pub fn total_enqueued(&self) -> u64 {
        self.enqueued
    }

    /// Get total dequeued
    #[must_use]
    pub fn total_dequeued(&self) -> u64 {
        self.dequeued
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        self.size = 0;
        self.peak_size = 0;
        self.enqueued = 0;
        self.dequeued = 0;
        self.total_wait_us = 0;
    }
}

// ============================================================================
// RATE LIMITING HELPERS (trueno-viz parity)
// ============================================================================

/// O(1) leaky bucket rate limiter.
///
/// Classic leaky bucket algorithm: tokens leak at constant rate,
/// requests add tokens. Overflow = rate exceeded.
#[derive(Debug, Clone)]
pub struct LeakyBucket {
    /// Current bucket level
    level: f64,
    /// Bucket capacity
    capacity: f64,
    /// Leak rate (units per second)
    leak_rate: f64,
    /// Last update timestamp (microseconds)
    last_update_us: u64,
    /// Total overflows
    overflows: u64,
}

impl Default for LeakyBucket {
    fn default() -> Self {
        Self::new(100.0, 10.0)
    }
}

impl LeakyBucket {
    /// Create with capacity and leak rate
    #[must_use]
    pub fn new(capacity: f64, leak_rate: f64) -> Self {
        Self {
            level: 0.0,
            capacity: capacity.max(1.0),
            leak_rate: leak_rate.max(0.1),
            last_update_us: 0,
            overflows: 0,
        }
    }

    /// Create for API rate limiting (100 req/s, burst 200)
    #[must_use]
    pub fn for_api() -> Self {
        Self::new(200.0, 100.0)
    }

    /// Create for network throttling (1MB/s, burst 5MB)
    #[must_use]
    pub fn for_network() -> Self {
        Self::new(5_000_000.0, 1_000_000.0)
    }

    /// Add tokens, returns true if accepted (no overflow)
    pub fn add(&mut self, tokens: f64, now_us: u64) -> bool {
        self.leak(now_us);
        let new_level = self.level + tokens;
        if new_level > self.capacity {
            self.overflows += 1;
            false
        } else {
            self.level = new_level;
            true
        }
    }

    /// Leak tokens based on elapsed time
    fn leak(&mut self, now_us: u64) {
        if self.last_update_us == 0 {
            self.last_update_us = now_us;
            return;
        }
        let elapsed_s = (now_us.saturating_sub(self.last_update_us)) as f64 / 1_000_000.0;
        let leaked = elapsed_s * self.leak_rate;
        self.level = (self.level - leaked).max(0.0);
        self.last_update_us = now_us;
    }

    /// Get current bucket level
    #[must_use]
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Get fill percentage
    #[must_use]
    pub fn fill_percentage(&self) -> f64 {
        (self.level / self.capacity) * 100.0
    }

    /// Get overflow count
    #[must_use]
    pub fn overflows(&self) -> u64 {
        self.overflows
    }

    /// Check if bucket is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.level <= 0.0
    }

    /// Reset bucket
    pub fn reset(&mut self) {
        self.level = 0.0;
        self.overflows = 0;
        self.last_update_us = 0;
    }

    /// Update with current time (for testing)
    pub fn update_with_time(&mut self, now_us: u64) {
        self.leak(now_us);
    }
}

/// O(1) sliding window rate counter.
///
/// Counts events in a sliding time window using sub-windows.
/// More accurate than token bucket for bursty traffic.
#[derive(Debug, Clone)]
pub struct SlidingWindowRate {
    /// Sub-window counts (circular buffer)
    windows: [u64; 10],
    /// Current window index
    current: usize,
    /// Window duration in microseconds
    window_us: u64,
    /// Last window rotation timestamp
    last_rotate_us: u64,
    /// Rate limit
    limit: u64,
    /// Exceeded count
    exceeded: u64,
}

impl Default for SlidingWindowRate {
    fn default() -> Self {
        Self::new(1_000_000, 100)
    }
}

impl SlidingWindowRate {
    /// Create with window duration (us) and rate limit
    #[must_use]
    pub fn new(window_us: u64, limit: u64) -> Self {
        Self {
            windows: [0; 10],
            current: 0,
            window_us: window_us.max(10_000), // Min 10ms
            last_rotate_us: 0,
            limit,
            exceeded: 0,
        }
    }

    /// Create for 1 second window with limit
    #[must_use]
    pub fn per_second(limit: u64) -> Self {
        Self::new(1_000_000, limit)
    }

    /// Create for 1 minute window with limit
    #[must_use]
    pub fn per_minute(limit: u64) -> Self {
        Self::new(60_000_000, limit)
    }

    /// Record event, returns true if within limit
    pub fn record(&mut self, now_us: u64) -> bool {
        self.rotate(now_us);
        let count = self.count();
        if count >= self.limit {
            self.exceeded += 1;
            false
        } else {
            self.windows[self.current] += 1;
            true
        }
    }

    /// Rotate windows if needed
    fn rotate(&mut self, now_us: u64) {
        if self.last_rotate_us == 0 {
            self.last_rotate_us = now_us;
            return;
        }
        let sub_window_us = self.window_us / 10;
        let elapsed = now_us.saturating_sub(self.last_rotate_us);
        let rotations = (elapsed / sub_window_us).min(10) as usize;

        for _ in 0..rotations {
            self.current = (self.current + 1) % 10;
            self.windows[self.current] = 0;
        }
        if rotations > 0 {
            self.last_rotate_us = now_us;
        }
    }

    /// Get current count across all windows
    #[must_use]
    pub fn count(&self) -> u64 {
        self.windows.iter().sum()
    }

    /// Get current rate as percentage of limit
    #[must_use]
    pub fn rate_percentage(&self) -> f64 {
        if self.limit == 0 {
            0.0
        } else {
            (self.count() as f64 / self.limit as f64) * 100.0
        }
    }

    /// Check if rate limit would be exceeded
    #[must_use]
    pub fn would_exceed(&self) -> bool {
        self.count() >= self.limit
    }

    /// Get exceeded count
    #[must_use]
    pub fn exceeded(&self) -> u64 {
        self.exceeded
    }

    /// Reset all windows
    pub fn reset(&mut self) {
        self.windows = [0; 10];
        self.current = 0;
        self.exceeded = 0;
        self.last_rotate_us = 0;
    }

    /// Update with current time (for testing)
    pub fn update_with_time(&mut self, now_us: u64) {
        self.rotate(now_us);
    }
}

// ============================================================================
// RESOURCE POOL & SAMPLING HELPERS (trueno-viz parity)
// ============================================================================

/// O(1) resource pool tracker for connection/object pool monitoring.
///
/// Tracks pool utilization, wait times, and connection health.
#[derive(Debug, Clone)]
pub struct ResourcePool {
    /// Total pool size
    capacity: u64,
    /// Currently in use
    in_use: u64,
    /// Peak in use
    peak_in_use: u64,
    /// Total acquisitions
    acquisitions: u64,
    /// Total releases
    releases: u64,
    /// Total timeouts
    timeouts: u64,
    /// Total wait time (for average)
    total_wait_us: u64,
}

impl Default for ResourcePool {
    fn default() -> Self {
        Self::new(10)
    }
}

impl ResourcePool {
    /// Create pool with capacity
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity: capacity.max(1),
            in_use: 0,
            peak_in_use: 0,
            acquisitions: 0,
            releases: 0,
            timeouts: 0,
            total_wait_us: 0,
        }
    }

    /// Create for database connections (typical pool size 20)
    #[must_use]
    pub fn for_database() -> Self {
        Self::new(20)
    }

    /// Create for HTTP connections (typical pool size 100)
    #[must_use]
    pub fn for_http() -> Self {
        Self::new(100)
    }

    /// Acquire resource from pool
    pub fn acquire(&mut self, wait_us: u64) -> bool {
        if self.in_use >= self.capacity {
            self.timeouts += 1;
            return false;
        }
        self.in_use += 1;
        self.acquisitions += 1;
        self.total_wait_us += wait_us;
        if self.in_use > self.peak_in_use {
            self.peak_in_use = self.in_use;
        }
        true
    }

    /// Release resource back to pool
    pub fn release(&mut self) {
        if self.in_use > 0 {
            self.in_use -= 1;
            self.releases += 1;
        }
    }

    /// Get current utilization percentage
    #[must_use]
    pub fn utilization(&self) -> f64 {
        (self.in_use as f64 / self.capacity as f64) * 100.0
    }

    /// Get available resources
    #[must_use]
    pub fn available(&self) -> u64 {
        self.capacity.saturating_sub(self.in_use)
    }

    /// Get average wait time in microseconds
    #[must_use]
    pub fn avg_wait_us(&self) -> f64 {
        if self.acquisitions == 0 {
            0.0
        } else {
            self.total_wait_us as f64 / self.acquisitions as f64
        }
    }

    /// Get timeout rate
    #[must_use]
    pub fn timeout_rate(&self) -> f64 {
        let total = self.acquisitions + self.timeouts;
        if total == 0 {
            0.0
        } else {
            (self.timeouts as f64 / total as f64) * 100.0
        }
    }

    /// Check if pool is exhausted
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.in_use >= self.capacity
    }

    /// Check if pool is idle
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.in_use == 0
    }

    /// Get peak utilization percentage
    #[must_use]
    pub fn peak_utilization(&self) -> f64 {
        (self.peak_in_use as f64 / self.capacity as f64) * 100.0
    }

    /// Reset all counters (keep capacity)
    pub fn reset(&mut self) {
        self.in_use = 0;
        self.peak_in_use = 0;
        self.acquisitions = 0;
        self.releases = 0;
        self.timeouts = 0;
        self.total_wait_us = 0;
    }
}

/// O(1) 2D histogram for heatmap data accumulation.
///
/// Fixed-grid 2D histogram for accumulating values in x,y space.
#[derive(Debug, Clone)]
pub struct Histogram2D {
    /// Grid cells (10x10 = 100 cells)
    cells: [[u64; 10]; 10],
    /// X min
    x_min: f64,
    /// X max
    x_max: f64,
    /// Y min
    y_min: f64,
    /// Y max
    y_max: f64,
    /// Total samples
    count: u64,
}

impl Default for Histogram2D {
    fn default() -> Self {
        Self::new(0.0, 100.0, 0.0, 100.0)
    }
}

impl Histogram2D {
    /// Create with x and y ranges
    #[must_use]
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self {
            cells: [[0; 10]; 10],
            x_min,
            x_max: x_max.max(x_min + 1.0),
            y_min,
            y_max: y_max.max(y_min + 1.0),
            count: 0,
        }
    }

    /// Create for latency vs throughput (0-100ms, 0-1000 ops/s)
    #[must_use]
    pub fn for_latency_throughput() -> Self {
        Self::new(0.0, 100.0, 0.0, 1000.0)
    }

    /// Create for CPU vs Memory (0-100%)
    #[must_use]
    pub fn for_cpu_memory() -> Self {
        Self::new(0.0, 100.0, 0.0, 100.0)
    }

    /// Add sample
    pub fn add(&mut self, x: f64, y: f64) {
        let xi = self.x_to_index(x);
        let yi = self.y_to_index(y);
        self.cells[yi][xi] += 1;
        self.count += 1;
    }

    fn x_to_index(&self, x: f64) -> usize {
        let normalized = (x - self.x_min) / (self.x_max - self.x_min);
        (normalized * 10.0).clamp(0.0, 9.0) as usize
    }

    fn y_to_index(&self, y: f64) -> usize {
        let normalized = (y - self.y_min) / (self.y_max - self.y_min);
        (normalized * 10.0).clamp(0.0, 9.0) as usize
    }

    /// Get cell count
    #[must_use]
    pub fn get(&self, xi: usize, yi: usize) -> u64 {
        if xi < 10 && yi < 10 {
            self.cells[yi][xi]
        } else {
            0
        }
    }

    /// Get cell density (percentage of total)
    #[must_use]
    pub fn density(&self, xi: usize, yi: usize) -> f64 {
        if self.count == 0 || xi >= 10 || yi >= 10 {
            0.0
        } else {
            (self.cells[yi][xi] as f64 / self.count as f64) * 100.0
        }
    }

    /// Get max cell count
    #[must_use]
    pub fn max_count(&self) -> u64 {
        self.cells
            .iter()
            .flat_map(|r| r.iter())
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Get hotspot (cell with max count)
    #[must_use]
    pub fn hotspot(&self) -> (usize, usize) {
        let mut max_val = 0;
        let mut max_pos = (0, 0);
        for (yi, row) in self.cells.iter().enumerate() {
            for (xi, &val) in row.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_pos = (xi, yi);
                }
            }
        }
        max_pos
    }

    /// Get total sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset all cells
    pub fn reset(&mut self) {
        self.cells = [[0; 10]; 10];
        self.count = 0;
    }
}

/// O(1) reservoir sampler for uniform sampling of streams.
///
/// Maintains a fixed-size sample of items seen in a stream using
/// Algorithm R (reservoir sampling).
#[derive(Debug, Clone)]
pub struct ReservoirSampler {
    /// Sample values
    samples: [f64; 16],
    /// Number of valid samples
    size: usize,
    /// Capacity
    capacity: usize,
    /// Total items seen
    seen: u64,
    /// Simple LCG state for deterministic sampling
    rng_state: u64,
}

impl Default for ReservoirSampler {
    fn default() -> Self {
        Self::new(16)
    }
}

impl ReservoirSampler {
    /// Create with capacity (max 16)
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: [0.0; 16],
            size: 0,
            capacity: capacity.min(16),
            seen: 0,
            rng_state: 12345,
        }
    }

    /// Simple LCG random number generator
    fn next_random(&mut self) -> u64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.rng_state
    }

    /// Add item to reservoir
    pub fn add(&mut self, value: f64) {
        self.seen += 1;
        if self.size < self.capacity {
            self.samples[self.size] = value;
            self.size += 1;
        } else {
            // Reservoir sampling: replace with probability capacity/seen
            let r = (self.next_random() % self.seen) as usize;
            if r < self.capacity {
                self.samples[r] = value;
            }
        }
    }

    /// Get sample at index
    #[must_use]
    pub fn get(&self, index: usize) -> Option<f64> {
        if index < self.size {
            Some(self.samples[index])
        } else {
            None
        }
    }

    /// Get current sample size
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get total items seen
    #[must_use]
    pub fn total_seen(&self) -> u64 {
        self.seen
    }

    /// Get sample mean
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.size == 0 {
            0.0
        } else {
            self.samples[..self.size].iter().sum::<f64>() / self.size as f64
        }
    }

    /// Get sample min
    #[must_use]
    pub fn min(&self) -> f64 {
        if self.size == 0 {
            0.0
        } else {
            self.samples[..self.size]
                .iter()
                .fold(f64::MAX, |a, &b| a.min(b))
        }
    }

    /// Get sample max
    #[must_use]
    pub fn max(&self) -> f64 {
        if self.size == 0 {
            0.0
        } else {
            self.samples[..self.size]
                .iter()
                .fold(f64::MIN, |a, &b| a.max(b))
        }
    }

    /// Reset sampler
    pub fn reset(&mut self) {
        self.samples = [0.0; 16];
        self.size = 0;
        self.seen = 0;
        self.rng_state = 12345;
    }
}

/// O(1) exponential histogram for log-scale binning.
///
/// Bins values into exponential buckets for wide-range distributions.
#[derive(Debug, Clone)]
pub struct ExponentialHistogram {
    /// Bucket counts (8 buckets: 1, 2, 4, 8, 16, 32, 64, 128+)
    buckets: [u64; 8],
    /// Base value (bucket boundaries are base * 2^i)
    base: f64,
    /// Total count
    count: u64,
    /// Sum of all values
    sum: f64,
}

impl Default for ExponentialHistogram {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl ExponentialHistogram {
    /// Create with base value
    #[must_use]
    pub fn new(base: f64) -> Self {
        Self {
            buckets: [0; 8],
            base: base.max(0.001),
            count: 0,
            sum: 0.0,
        }
    }

    /// Create for latency (base 1ms: 1, 2, 4, 8, 16, 32, 64, 128+ ms)
    #[must_use]
    pub fn for_latency_ms() -> Self {
        Self::new(1.0)
    }

    /// Create for bytes (base 1KB: 1, 2, 4, 8, 16, 32, 64, 128+ KB)
    #[must_use]
    pub fn for_bytes_kb() -> Self {
        Self::new(1024.0)
    }

    /// Add value
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        let bucket = self.value_to_bucket(value);
        self.buckets[bucket] += 1;
    }

    fn value_to_bucket(&self, value: f64) -> usize {
        if value < self.base {
            return 0;
        }
        let ratio = value / self.base;
        let bucket = ratio.log2().floor() as usize;
        bucket.min(7)
    }

    /// Get bucket count
    #[must_use]
    pub fn bucket_count(&self, bucket: usize) -> u64 {
        if bucket < 8 {
            self.buckets[bucket]
        } else {
            0
        }
    }

    /// Get bucket upper bound
    #[must_use]
    pub fn bucket_upper_bound(&self, bucket: usize) -> f64 {
        if bucket >= 7 {
            f64::INFINITY
        } else {
            self.base * 2.0_f64.powi(bucket as i32 + 1)
        }
    }

    /// Get total count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get mean value
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Get bucket with most samples
    #[must_use]
    pub fn mode_bucket(&self) -> usize {
        self.buckets
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Reset histogram
    pub fn reset(&mut self) {
        self.buckets = [0; 8];
        self.count = 0;
        self.sum = 0.0;
    }
}
