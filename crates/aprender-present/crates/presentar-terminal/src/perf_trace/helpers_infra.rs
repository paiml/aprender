// ============================================================================
// CACHE & LOAD BALANCING HELPERS (trueno-viz parity)
// ============================================================================

/// O(1) cache statistics tracker.
///
/// Tracks cache hits, misses, evictions, and calculates hit rate.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total hits
    hits: u64,
    /// Total misses
    misses: u64,
    /// Total evictions
    evictions: u64,
    /// Total insertions
    insertions: u64,
    /// Bytes in cache
    bytes_cached: u64,
    /// Capacity in bytes
    capacity_bytes: u64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self::new(0)
    }
}

impl CacheStats {
    /// Create with capacity in bytes
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            insertions: 0,
            bytes_cached: 0,
            capacity_bytes,
        }
    }

    /// Create for L1 cache (32KB typical)
    #[must_use]
    pub fn for_l1_cache() -> Self {
        Self::new(32 * 1024)
    }

    /// Create for L2 cache (256KB typical)
    #[must_use]
    pub fn for_l2_cache() -> Self {
        Self::new(256 * 1024)
    }

    /// Create for application cache (16MB)
    #[must_use]
    pub fn for_app_cache() -> Self {
        Self::new(16 * 1024 * 1024)
    }

    /// Record a cache hit
    pub fn hit(&mut self) {
        self.hits += 1;
    }

    /// Record a cache miss
    pub fn miss(&mut self) {
        self.misses += 1;
    }

    /// Record an eviction
    pub fn evict(&mut self, bytes: u64) {
        self.evictions += 1;
        self.bytes_cached = self.bytes_cached.saturating_sub(bytes);
    }

    /// Record an insertion
    pub fn insert(&mut self, bytes: u64) {
        self.insertions += 1;
        self.bytes_cached += bytes;
    }

    /// Get hit rate as percentage
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Get miss rate as percentage
    #[must_use]
    pub fn miss_rate(&self) -> f64 {
        100.0 - self.hit_rate()
    }

    /// Get eviction rate (evictions per insertion)
    #[must_use]
    pub fn eviction_rate(&self) -> f64 {
        if self.insertions == 0 {
            0.0
        } else {
            self.evictions as f64 / self.insertions as f64
        }
    }

    /// Get fill percentage
    #[must_use]
    pub fn fill_percentage(&self) -> f64 {
        if self.capacity_bytes == 0 {
            0.0
        } else {
            (self.bytes_cached as f64 / self.capacity_bytes as f64) * 100.0
        }
    }

    /// Get total requests
    #[must_use]
    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }

    /// Check if cache is effective (hit rate > threshold)
    #[must_use]
    pub fn is_effective(&self, threshold: f64) -> bool {
        self.hit_rate() >= threshold
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.insertions = 0;
        self.bytes_cached = 0;
    }
}

/// O(1) Bloom filter for probabilistic membership testing.
///
/// Fixed-size bloom filter with configurable hash count.
/// False positives possible, false negatives impossible.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// Bit array (using u64 words)
    bits: [u64; 16], // 1024 bits
    /// Number of hash functions
    hash_count: u32,
    /// Items added
    items: u64,
}

impl Default for BloomFilter {
    fn default() -> Self {
        Self::new(3)
    }
}

impl BloomFilter {
    /// Create with number of hash functions
    #[must_use]
    pub fn new(hash_count: u32) -> Self {
        Self {
            bits: [0; 16],
            hash_count: hash_count.clamp(1, 10),
            items: 0,
        }
    }

    /// Create optimized for ~100 items (3 hashes)
    #[must_use]
    pub fn for_small() -> Self {
        Self::new(3)
    }

    /// Create optimized for ~500 items (5 hashes)
    #[must_use]
    pub fn for_medium() -> Self {
        Self::new(5)
    }

    /// Simple hash function (FNV-1a style)
    fn hash(&self, value: u64, seed: u32) -> usize {
        let mut h = value.wrapping_mul(0x517cc1b727220a95);
        h = h.wrapping_add(seed as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        (h as usize) % 1024
    }

    /// Add item to filter
    pub fn add(&mut self, value: u64) {
        for i in 0..self.hash_count {
            let bit_idx = self.hash(value, i);
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            self.bits[word_idx] |= 1 << bit_pos;
        }
        self.items += 1;
    }

    /// Check if item might be in filter
    #[must_use]
    pub fn might_contain(&self, value: u64) -> bool {
        for i in 0..self.hash_count {
            let bit_idx = self.hash(value, i);
            let word_idx = bit_idx / 64;
            let bit_pos = bit_idx % 64;
            if self.bits[word_idx] & (1 << bit_pos) == 0 {
                return false;
            }
        }
        true
    }

    /// Get number of items added
    #[must_use]
    pub fn len(&self) -> u64 {
        self.items
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items == 0
    }

    /// Get estimated false positive rate
    #[must_use]
    pub fn false_positive_rate(&self) -> f64 {
        let m = 1024.0; // bits
        let k = self.hash_count as f64;
        let n = self.items as f64;
        if n == 0.0 {
            return 0.0;
        }
        (1.0 - (-k * n / m).exp()).powf(k)
    }

    /// Get fill percentage (bits set / total bits)
    #[must_use]
    pub fn fill_percentage(&self) -> f64 {
        let set_bits: u32 = self.bits.iter().map(|w| w.count_ones()).sum();
        (set_bits as f64 / 1024.0) * 100.0
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.bits = [0; 16];
        self.items = 0;
    }
}

/// O(1) weighted round-robin load balancer.
///
/// Distributes load across backends with configurable weights.
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    /// Backend weights
    weights: [u32; 8],
    /// Current weights (for WRR algorithm)
    current: [i32; 8],
    /// Active backends count
    active: usize,
    /// Total requests dispatched
    dispatched: u64,
    /// Requests per backend
    per_backend: [u64; 8],
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer {
    /// Create empty load balancer
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: [0; 8],
            current: [0; 8],
            active: 0,
            dispatched: 0,
            per_backend: [0; 8],
        }
    }

    /// Create with equal weights for n backends
    #[must_use]
    pub fn equal_weights(n: usize) -> Self {
        let mut lb = Self::new();
        for _ in 0..n.min(8) {
            lb.add_backend(1);
        }
        lb
    }

    /// Add backend with weight
    pub fn add_backend(&mut self, weight: u32) {
        if self.active < 8 {
            self.weights[self.active] = weight.max(1);
            self.current[self.active] = 0;
            self.active += 1;
        }
    }

    /// Select next backend (weighted round-robin)
    #[must_use]
    pub fn select_backend(&mut self) -> Option<usize> {
        if self.active == 0 {
            return None;
        }

        // Weighted round-robin: select backend with highest current weight
        let total_weight: i32 = self.weights[..self.active].iter().map(|&w| w as i32).sum();

        // Add weights to current
        for i in 0..self.active {
            self.current[i] += self.weights[i] as i32;
        }

        // Find max current weight
        let mut max_idx = 0;
        let mut max_weight = self.current[0];
        for i in 1..self.active {
            if self.current[i] > max_weight {
                max_weight = self.current[i];
                max_idx = i;
            }
        }

        // Subtract total weight from selected
        self.current[max_idx] -= total_weight;
        self.dispatched += 1;
        self.per_backend[max_idx] += 1;

        Some(max_idx)
    }

    /// Get distribution percentage for backend
    #[must_use]
    pub fn distribution(&self, backend: usize) -> f64 {
        if self.dispatched == 0 || backend >= self.active {
            0.0
        } else {
            (self.per_backend[backend] as f64 / self.dispatched as f64) * 100.0
        }
    }

    /// Get total dispatched
    #[must_use]
    pub fn total_dispatched(&self) -> u64 {
        self.dispatched
    }

    /// Get active backend count
    #[must_use]
    pub fn backend_count(&self) -> usize {
        self.active
    }

    /// Check if load is balanced (within threshold)
    #[must_use]
    pub fn is_balanced(&self, threshold: f64) -> bool {
        if self.active <= 1 || self.dispatched < 10 {
            return true;
        }
        let avg = self.dispatched as f64 / self.active as f64;
        for i in 0..self.active {
            let deviation = ((self.per_backend[i] as f64 - avg) / avg).abs() * 100.0;
            if deviation > threshold {
                return false;
            }
        }
        true
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        self.current = [0; 8];
        self.dispatched = 0;
        self.per_backend = [0; 8];
    }
}

/// O(1) token bucket with burst tracking.
///
/// Enhanced token bucket that tracks burst patterns.
#[derive(Debug, Clone)]
pub struct BurstTracker {
    /// Current tokens
    tokens: f64,
    /// Bucket capacity
    capacity: f64,
    /// Refill rate (tokens per second)
    refill_rate: f64,
    /// Last update timestamp (us)
    last_update_us: u64,
    /// Current burst count
    burst_count: u64,
    /// Max burst seen
    max_burst: u64,
    /// Total bursts
    total_bursts: u64,
}

impl Default for BurstTracker {
    fn default() -> Self {
        Self::new(100.0, 10.0)
    }
}

impl BurstTracker {
    /// Create with capacity and refill rate
    #[must_use]
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            capacity: capacity.max(1.0),
            refill_rate: refill_rate.max(0.1),
            last_update_us: 0,
            burst_count: 0,
            max_burst: 0,
            total_bursts: 0,
        }
    }

    /// Create for API rate limiting
    #[must_use]
    pub fn for_api() -> Self {
        Self::new(100.0, 50.0)
    }

    /// Create for network throttling
    #[must_use]
    pub fn for_network() -> Self {
        Self::new(1000.0, 100.0)
    }

    /// Consume tokens, returns true if allowed
    pub fn consume(&mut self, tokens: f64, now_us: u64) -> bool {
        self.refill(now_us);

        if tokens <= self.tokens {
            self.tokens -= tokens;
            self.burst_count += 1;
            if self.burst_count > self.max_burst {
                self.max_burst = self.burst_count;
            }
            true
        } else {
            // End of burst
            if self.burst_count > 0 {
                self.total_bursts += 1;
            }
            self.burst_count = 0;
            false
        }
    }

    fn refill(&mut self, now_us: u64) {
        if self.last_update_us == 0 {
            self.last_update_us = now_us;
            return;
        }
        let elapsed_s = (now_us.saturating_sub(self.last_update_us)) as f64 / 1_000_000.0;
        let refill = elapsed_s * self.refill_rate;
        self.tokens = (self.tokens + refill).min(self.capacity);
        self.last_update_us = now_us;
    }

    /// Get current token count
    #[must_use]
    pub fn tokens(&self) -> f64 {
        self.tokens
    }

    /// Get fill percentage
    #[must_use]
    pub fn fill_percentage(&self) -> f64 {
        (self.tokens / self.capacity) * 100.0
    }

    /// Get max burst size seen
    #[must_use]
    pub fn max_burst(&self) -> u64 {
        self.max_burst
    }

    /// Get total bursts
    #[must_use]
    pub fn total_bursts(&self) -> u64 {
        self.total_bursts
    }

    /// Get average burst size
    #[must_use]
    pub fn avg_burst(&self) -> f64 {
        if self.total_bursts == 0 {
            0.0
        } else {
            self.max_burst as f64 // Approximation
        }
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.tokens = self.capacity;
        self.burst_count = 0;
        self.max_burst = 0;
        self.total_bursts = 0;
        self.last_update_us = 0;
    }
}

// ============================================================================
// TopKTracker - Fixed-size top-K value tracker (O(1) amortized insertion)
// ============================================================================

/// O(1) amortized top-K value tracker.
/// Uses a fixed-size array with insertion sort for small K values.
#[derive(Debug, Clone)]
pub struct TopKTracker {
    values: [f64; 32],
    count: usize,
    k: usize,
}

impl Default for TopKTracker {
    fn default() -> Self {
        Self::new(10)
    }
}

impl TopKTracker {
    /// Create new top-K tracker
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            values: [f64::NEG_INFINITY; 32],
            count: 0,
            k: k.min(32),
        }
    }

    /// Create for metrics (top 10)
    #[must_use]
    pub fn for_metrics() -> Self {
        Self::new(10)
    }

    /// Create for processes (top 20)
    #[must_use]
    pub fn for_processes() -> Self {
        Self::new(20)
    }

    /// Add value (O(k) insertion)
    pub fn add(&mut self, value: f64) {
        if self.count < self.k {
            // Not full yet, insert in sorted order
            let mut i = self.count;
            while i > 0 && self.values[i - 1] < value {
                self.values[i] = self.values[i - 1];
                i -= 1;
            }
            self.values[i] = value;
            self.count += 1;
        } else if value > self.values[self.k - 1] {
            // Replace minimum if value is larger
            let mut i = self.k - 1;
            while i > 0 && self.values[i - 1] < value {
                self.values[i] = self.values[i - 1];
                i -= 1;
            }
            self.values[i] = value;
        }
    }

    /// Get top-K values (sorted descending)
    #[must_use]
    pub fn top(&self) -> &[f64] {
        &self.values[..self.count]
    }

    /// Get K value
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get count of tracked values
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get minimum value in top-K
    #[must_use]
    pub fn minimum(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.values[self.count - 1])
        } else {
            None
        }
    }

    /// Get maximum value (always at index 0)
    #[must_use]
    pub fn maximum(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.values[0])
        } else {
            None
        }
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.values = [f64::NEG_INFINITY; 32];
        self.count = 0;
    }
}

// ============================================================================
// QuotaTracker - Resource quota tracking
// ============================================================================

/// O(1) resource quota tracker.
/// Tracks usage against a limit with percentage and exhaustion checks.
#[derive(Debug, Clone)]
pub struct QuotaTracker {
    limit: u64,
    used: u64,
    peak_usage: u64,
}

impl Default for QuotaTracker {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl QuotaTracker {
    /// Create with limit
    #[must_use]
    pub fn new(limit: u64) -> Self {
        Self {
            limit: limit.max(1),
            used: 0,
            peak_usage: 0,
        }
    }

    /// Create for API daily limit (10K requests)
    #[must_use]
    pub fn for_api_daily() -> Self {
        Self::new(10000)
    }

    /// Create for storage limit (100 GB)
    #[must_use]
    pub fn for_storage_gb() -> Self {
        Self::new(100)
    }

    /// Use quota, returns false if would exceed
    pub fn use_quota(&mut self, amount: u64) -> bool {
        if self.used + amount > self.limit {
            false
        } else {
            self.used += amount;
            if self.used > self.peak_usage {
                self.peak_usage = self.used;
            }
            true
        }
    }

    /// Release quota
    pub fn release(&mut self, amount: u64) {
        self.used = self.used.saturating_sub(amount);
    }

    /// Get limit
    #[must_use]
    pub fn limit(&self) -> u64 {
        self.limit
    }

    /// Get remaining quota
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.used)
    }

    /// Get usage percentage
    #[must_use]
    pub fn usage_percentage(&self) -> f64 {
        (self.used as f64 / self.limit as f64) * 100.0
    }

    /// Check if exhausted
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.used >= self.limit
    }

    /// Get peak usage
    #[must_use]
    pub fn peak_usage(&self) -> u64 {
        self.peak_usage
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.used = 0;
        self.peak_usage = 0;
    }
}

// ============================================================================
// FrequencyCounter - Categorical frequency tracking
// ============================================================================

/// O(1) categorical frequency counter.
/// Tracks occurrence counts and calculates frequencies for up to 16 categories.
#[derive(Debug, Clone)]
pub struct FrequencyCounter {
    counts: [u64; 16],
    total: u64,
}

impl Default for FrequencyCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl FrequencyCounter {
    /// Create new counter
    #[must_use]
    pub fn new() -> Self {
        Self {
            counts: [0; 16],
            total: 0,
        }
    }

    /// Increment category count
    pub fn increment(&mut self, category: usize) {
        if category < 16 {
            self.counts[category] += 1;
            self.total += 1;
        }
    }

    /// Add multiple to category
    pub fn add(&mut self, category: usize, count: u64) {
        if category < 16 {
            self.counts[category] += count;
            self.total += count;
        }
    }

    /// Get count for category
    #[must_use]
    pub fn count(&self, category: usize) -> u64 {
        if category < 16 {
            self.counts[category]
        } else {
            0
        }
    }

    /// Get frequency percentage for category
    #[must_use]
    pub fn frequency(&self, category: usize) -> f64 {
        if self.total == 0 || category >= 16 {
            0.0
        } else {
            (self.counts[category] as f64 / self.total as f64) * 100.0
        }
    }

    /// Get total count
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Get most frequent category
    #[must_use]
    pub fn most_frequent(&self) -> Option<usize> {
        if self.total == 0 {
            return None;
        }
        let mut max_idx = 0;
        let mut max_count = self.counts[0];
        for i in 1..16 {
            if self.counts[i] > max_count {
                max_count = self.counts[i];
                max_idx = i;
            }
        }
        Some(max_idx)
    }

    /// Get number of non-zero categories
    #[must_use]
    pub fn non_zero_count(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }

    /// Calculate Shannon entropy (normalized 0-1)
    #[must_use]
    pub fn entropy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let mut entropy = 0.0;
        for &count in &self.counts {
            if count > 0 {
                let p = count as f64 / self.total as f64;
                entropy -= p * p.log2();
            }
        }
        // Normalize by max entropy (log2(16) = 4)
        entropy / 4.0
    }

    /// Reset counter
    pub fn reset(&mut self) {
        self.counts = [0; 16];
        self.total = 0;
    }
}

// ============================================================================
// MovingRange - Moving min/max range tracking for volatility
// ============================================================================

/// O(1) moving range tracker for volatility analysis.
/// Maintains min/max over a sliding window for range and volatility metrics.
#[derive(Debug, Clone)]
pub struct MovingRange {
    values: [f64; 128],
    window_size: usize,
    head: usize,
    count: usize,
    current_min: f64,
    current_max: f64,
}

impl Default for MovingRange {
    fn default() -> Self {
        Self::new(10)
    }
}

impl MovingRange {
    /// Create with window size
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            values: [0.0; 128],
            window_size: window_size.min(128),
            head: 0,
            count: 0,
            current_min: f64::INFINITY,
            current_max: f64::NEG_INFINITY,
        }
    }

    /// Create for price volatility (20 samples)
    #[must_use]
    pub fn for_prices() -> Self {
        Self::new(20)
    }

    /// Create for latency volatility (100 samples)
    #[must_use]
    pub fn for_latency() -> Self {
        Self::new(100)
    }

    /// Add value to window
    pub fn add(&mut self, value: f64) {
        let idx = self.head;
        self.values[idx] = value;
        self.head = (self.head + 1) % self.window_size;
        if self.count < self.window_size {
            self.count += 1;
        }
        self.recalculate_minmax();
    }

    fn recalculate_minmax(&mut self) {
        self.current_min = f64::INFINITY;
        self.current_max = f64::NEG_INFINITY;
        for i in 0..self.count {
            let v = self.values[i];
            if v < self.current_min {
                self.current_min = v;
            }
            if v > self.current_max {
                self.current_max = v;
            }
        }
    }

    /// Get window size
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get current count
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get minimum value
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.current_min)
        } else {
            None
        }
    }

    /// Get maximum value
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.current_max)
        } else {
            None
        }
    }

    /// Get range (max - min)
    #[must_use]
    pub fn range(&self) -> f64 {
        if self.count > 0 {
            self.current_max - self.current_min
        } else {
            0.0
        }
    }

    /// Get mid-range ((max + min) / 2)
    #[must_use]
    pub fn midrange(&self) -> f64 {
        if self.count > 0 {
            (self.current_max + self.current_min) / 2.0
        } else {
            0.0
        }
    }

    /// Get volatility (range / midrange * 100)
    #[must_use]
    pub fn volatility(&self) -> f64 {
        let mid = self.midrange();
        if mid.abs() < 0.0001 {
            0.0
        } else {
            (self.range() / mid) * 100.0
        }
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.values = [0.0; 128];
        self.head = 0;
        self.count = 0;
        self.current_min = f64::INFINITY;
        self.current_max = f64::NEG_INFINITY;
    }
}

// ============================================================================
// TimeoutTracker - Operation timeout tracking
// ============================================================================

/// O(1) operation timeout tracker.
/// Tracks successful and timed-out operations with configurable timeout threshold.
#[derive(Debug, Clone)]
pub struct TimeoutTracker {
    timeout_us: u64,
    total: u64,
    timed_out: u64,
    last_duration_us: u64,
    max_duration_us: u64,
}

impl Default for TimeoutTracker {
    fn default() -> Self {
        Self::new(1_000_000) // 1 second default
    }
}

impl TimeoutTracker {
    /// Create with timeout threshold in microseconds
    #[must_use]
    pub fn new(timeout_us: u64) -> Self {
        Self {
            timeout_us: timeout_us.max(1),
            total: 0,
            timed_out: 0,
            last_duration_us: 0,
            max_duration_us: 0,
        }
    }

    /// Create for network operations (5s timeout)
    #[must_use]
    pub fn for_network() -> Self {
        Self::new(5_000_000)
    }

    /// Create for database operations (30s timeout)
    #[must_use]
    pub fn for_database() -> Self {
        Self::new(30_000_000)
    }

    /// Create for fast operations (100ms timeout)
    #[must_use]
    pub fn for_fast() -> Self {
        Self::new(100_000)
    }

    /// Record operation completion
    pub fn record(&mut self, duration_us: u64) {
        self.total += 1;
        self.last_duration_us = duration_us;
        if duration_us > self.max_duration_us {
            self.max_duration_us = duration_us;
        }
        if duration_us > self.timeout_us {
            self.timed_out += 1;
        }
    }

    /// Get total operations
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Get timed out count
    #[must_use]
    pub fn timed_out(&self) -> u64 {
        self.timed_out
    }

    /// Get timeout rate as percentage
    #[must_use]
    pub fn timeout_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.timed_out as f64 / self.total as f64) * 100.0
        }
    }

    /// Get success rate as percentage
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        100.0 - self.timeout_rate()
    }

    /// Check if timeout rate is acceptable
    #[must_use]
    pub fn is_healthy(&self, max_timeout_rate: f64) -> bool {
        self.timeout_rate() <= max_timeout_rate
    }

    /// Get max duration seen
    #[must_use]
    pub fn max_duration_us(&self) -> u64 {
        self.max_duration_us
    }

    /// Get timeout threshold
    #[must_use]
    pub fn timeout_threshold_us(&self) -> u64 {
        self.timeout_us
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.total = 0;
        self.timed_out = 0;
        self.last_duration_us = 0;
        self.max_duration_us = 0;
    }
}

// ============================================================================
// RetryTracker - Retry attempt tracking with backoff state
// ============================================================================

/// O(1) retry tracking with exponential backoff state.
/// Tracks retry attempts, success after retry, and calculates next retry delay.
#[derive(Debug, Clone)]
pub struct RetryTracker {
    max_retries: u32,
    base_delay_ms: u64,
    max_delay_ms: u64,
    total_attempts: u64,
    total_retries: u64,
    successful_retries: u64,
    current_retry: u32,
}

impl Default for RetryTracker {
    fn default() -> Self {
        Self::new(3, 100, 10000)
    }
}

impl RetryTracker {
    /// Create with max retries, base delay, and max delay in ms
    #[must_use]
    pub fn new(max_retries: u32, base_delay_ms: u64, max_delay_ms: u64) -> Self {
        Self {
            max_retries,
            base_delay_ms: base_delay_ms.max(1),
            max_delay_ms: max_delay_ms.max(base_delay_ms),
            total_attempts: 0,
            total_retries: 0,
            successful_retries: 0,
            current_retry: 0,
        }
    }

    /// Create for API retries (3 retries, 100ms base, 10s max)
    #[must_use]
    pub fn for_api() -> Self {
        Self::new(3, 100, 10000)
    }

    /// Create for network retries (5 retries, 1s base, 30s max)
    #[must_use]
    pub fn for_network() -> Self {
        Self::new(5, 1000, 30000)
    }

    /// Record attempt start
    pub fn attempt(&mut self) {
        self.total_attempts += 1;
    }

    /// Record retry (failed attempt, will retry)
    pub fn retry(&mut self) {
        self.total_retries += 1;
        if self.current_retry < self.max_retries {
            self.current_retry += 1;
        }
    }

    /// Record success (resets current retry count)
    pub fn success(&mut self) {
        if self.current_retry > 0 {
            self.successful_retries += 1;
        }
        self.current_retry = 0;
    }

    /// Get next retry delay in ms (exponential backoff)
    #[must_use]
    pub fn next_delay_ms(&self) -> u64 {
        let delay = self.base_delay_ms * (1 << self.current_retry);
        delay.min(self.max_delay_ms)
    }

    /// Check if retries exhausted
    #[must_use]
    pub fn retries_exhausted(&self) -> bool {
        self.current_retry >= self.max_retries
    }

    /// Get retry rate as percentage
    #[must_use]
    pub fn retry_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            0.0
        } else {
            (self.total_retries as f64 / self.total_attempts as f64) * 100.0
        }
    }

    /// Get successful retry rate
    #[must_use]
    pub fn successful_retry_rate(&self) -> f64 {
        if self.total_retries == 0 {
            0.0
        } else {
            (self.successful_retries as f64 / self.total_retries as f64) * 100.0
        }
    }

    /// Get current retry count
    #[must_use]
    pub fn current_retry(&self) -> u32 {
        self.current_retry
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.total_attempts = 0;
        self.total_retries = 0;
        self.successful_retries = 0;
        self.current_retry = 0;
    }
}

// ============================================================================
// ScheduleSlot - Time-based slot scheduling
// ============================================================================

/// O(1) time-based slot scheduler.
/// Divides time into slots and tracks which slot is currently active.
#[derive(Debug, Clone)]
pub struct ScheduleSlot {
    slot_duration_us: u64,
    num_slots: usize,
    current_slot: usize,
    slot_start_us: u64,
    executions_per_slot: [u64; 16],
}

impl Default for ScheduleSlot {
    fn default() -> Self {
        Self::new(1_000_000, 10) // 1 second slots, 10 slots
    }
}

impl ScheduleSlot {
    /// Create with slot duration in microseconds and number of slots
    #[must_use]
    pub fn new(slot_duration_us: u64, num_slots: usize) -> Self {
        Self {
            slot_duration_us: slot_duration_us.max(1),
            num_slots: num_slots.min(16).max(1),
            current_slot: 0,
            slot_start_us: 0,
            executions_per_slot: [0; 16],
        }
    }

    /// Create for round-robin scheduling (1 second slots, 10 slots)
    #[must_use]
    pub fn for_round_robin() -> Self {
        Self::new(1_000_000, 10)
    }

    /// Create for minute-based scheduling (1 minute slots, 5 slots)
    #[must_use]
    pub fn for_minute() -> Self {
        Self::new(60_000_000, 5)
    }

    /// Update slot based on current time
    pub fn update(&mut self, now_us: u64) {
        if self.slot_start_us == 0 {
            self.slot_start_us = now_us;
            return;
        }

        let elapsed = now_us.saturating_sub(self.slot_start_us);
        let slots_passed = (elapsed / self.slot_duration_us) as usize;

        if slots_passed > 0 {
            self.current_slot = (self.current_slot + slots_passed) % self.num_slots;
            self.slot_start_us = now_us;
        }
    }

    /// Record execution in current slot
    pub fn execute(&mut self, now_us: u64) {
        self.update(now_us);
        if self.current_slot < 16 {
            self.executions_per_slot[self.current_slot] += 1;
        }
    }

    /// Get current slot
    #[must_use]
    pub fn current_slot(&self) -> usize {
        self.current_slot
    }

    /// Get number of slots
    #[must_use]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Get executions for a slot
    #[must_use]
    pub fn executions(&self, slot: usize) -> u64 {
        if slot < 16 {
            self.executions_per_slot[slot]
        } else {
            0
        }
    }

    /// Get total executions across all slots
    #[must_use]
    pub fn total_executions(&self) -> u64 {
        self.executions_per_slot[..self.num_slots].iter().sum()
    }

    /// Check if slots are evenly distributed (within threshold %)
    #[must_use]
    pub fn is_balanced(&self, threshold: f64) -> bool {
        let total = self.total_executions();
        if total == 0 {
            return true;
        }
        let expected = total as f64 / self.num_slots as f64;
        for i in 0..self.num_slots {
            let diff = (self.executions_per_slot[i] as f64 - expected).abs();
            if diff / expected * 100.0 > threshold {
                return false;
            }
        }
        true
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.current_slot = 0;
        self.slot_start_us = 0;
        self.executions_per_slot = [0; 16];
    }
}

// ============================================================================
// CooldownTimer - Cooldown period tracking
// ============================================================================

/// O(1) cooldown timer for rate limiting actions.
/// Tracks when an action can next be performed based on cooldown period.
#[derive(Debug, Clone)]
pub struct CooldownTimer {
    cooldown_us: u64,
    last_action_us: u64,
    total_actions: u64,
    blocked_attempts: u64,
}

impl Default for CooldownTimer {
    fn default() -> Self {
        Self::new(1_000_000) // 1 second cooldown
    }
}

impl CooldownTimer {
    /// Create with cooldown period in microseconds
    #[must_use]
    pub fn new(cooldown_us: u64) -> Self {
        Self {
            cooldown_us: cooldown_us.max(1),
            last_action_us: 0,
            total_actions: 0,
            blocked_attempts: 0,
        }
    }

    /// Create for fast cooldown (100ms)
    #[must_use]
    pub fn for_fast() -> Self {
        Self::new(100_000)
    }

    /// Create for normal cooldown (1 second)
    #[must_use]
    pub fn for_normal() -> Self {
        Self::new(1_000_000)
    }

    /// Create for slow cooldown (10 seconds)
    #[must_use]
    pub fn for_slow() -> Self {
        Self::new(10_000_000)
    }

    /// Check if action is ready (cooldown expired)
    #[must_use]
    pub fn is_ready(&self, now_us: u64) -> bool {
        if self.last_action_us == 0 {
            return true;
        }
        now_us.saturating_sub(self.last_action_us) >= self.cooldown_us
    }

    /// Try to perform action, returns true if allowed
    pub fn try_action(&mut self, now_us: u64) -> bool {
        if self.is_ready(now_us) {
            self.last_action_us = now_us;
            self.total_actions += 1;
            true
        } else {
            self.blocked_attempts += 1;
            false
        }
    }

    /// Force action (ignores cooldown)
    pub fn force_action(&mut self, now_us: u64) {
        self.last_action_us = now_us;
        self.total_actions += 1;
    }

    /// Get remaining cooldown time in microseconds
    #[must_use]
    pub fn remaining_us(&self, now_us: u64) -> u64 {
        if self.is_ready(now_us) {
            0
        } else {
            self.cooldown_us
                .saturating_sub(now_us.saturating_sub(self.last_action_us))
        }
    }

    /// Get cooldown period
    #[must_use]
    pub fn cooldown_us(&self) -> u64 {
        self.cooldown_us
    }

    /// Get total actions performed
    #[must_use]
    pub fn total_actions(&self) -> u64 {
        self.total_actions
    }

    /// Get blocked attempts
    #[must_use]
    pub fn blocked_attempts(&self) -> u64 {
        self.blocked_attempts
    }

    /// Get block rate as percentage
    #[must_use]
    pub fn block_rate(&self) -> f64 {
        let total = self.total_actions + self.blocked_attempts;
        if total == 0 {
            0.0
        } else {
            (self.blocked_attempts as f64 / total as f64) * 100.0
        }
    }

    /// Reset timer
    pub fn reset(&mut self) {
        self.last_action_us = 0;
        self.total_actions = 0;
        self.blocked_attempts = 0;
    }
}

// ============================================================================
// BackpressureMonitor - Track backpressure signals
// ============================================================================

/// O(1) backpressure monitoring.
/// Tracks when downstream systems signal overload and calculates pressure rates.
#[derive(Debug, Clone)]
pub struct BackpressureMonitor {
    signals: u64,
    total_ops: u64,
    consecutive: u32,
    max_consecutive: u32,
    last_signal_us: u64,
}

impl Default for BackpressureMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl BackpressureMonitor {
    /// Create new monitor
    #[must_use]
    pub fn new() -> Self {
        Self {
            signals: 0,
            total_ops: 0,
            consecutive: 0,
            max_consecutive: 0,
            last_signal_us: 0,
        }
    }

    /// Record successful operation (no backpressure)
    pub fn success(&mut self) {
        self.total_ops += 1;
        self.consecutive = 0;
    }

    /// Record backpressure signal
    pub fn signal(&mut self, now_us: u64) {
        self.signals += 1;
        self.total_ops += 1;
        self.consecutive += 1;
        self.last_signal_us = now_us;
        if self.consecutive > self.max_consecutive {
            self.max_consecutive = self.consecutive;
        }
    }

    /// Get backpressure rate as percentage
    #[must_use]
    pub fn pressure_rate(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            (self.signals as f64 / self.total_ops as f64) * 100.0
        }
    }

    /// Check if currently under pressure (consecutive signals)
    #[must_use]
    pub fn is_under_pressure(&self, threshold: u32) -> bool {
        self.consecutive >= threshold
    }

    /// Get consecutive signal count
    #[must_use]
    pub fn consecutive(&self) -> u32 {
        self.consecutive
    }

    /// Get max consecutive signals
    #[must_use]
    pub fn max_consecutive(&self) -> u32 {
        self.max_consecutive
    }

    /// Get total signals
    #[must_use]
    pub fn total_signals(&self) -> u64 {
        self.signals
    }

    /// Check if healthy (below threshold)
    #[must_use]
    pub fn is_healthy(&self, max_rate: f64) -> bool {
        self.pressure_rate() <= max_rate
    }

    /// Reset monitor
    pub fn reset(&mut self) {
        self.signals = 0;
        self.total_ops = 0;
        self.consecutive = 0;
        self.max_consecutive = 0;
        self.last_signal_us = 0;
    }
}

// ============================================================================
// CapacityPlanner - Track capacity utilization for planning
// ============================================================================

/// O(1) capacity planning tracker.
/// Monitors utilization over time and predicts when capacity will be exhausted.
#[derive(Debug, Clone)]
pub struct CapacityPlanner {
    capacity: u64,
    current: u64,
    peak: u64,
    samples: u32,
    sum_utilization: f64,
    growth_rate: f64,
}

impl Default for CapacityPlanner {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl CapacityPlanner {
    /// Create with capacity
    #[must_use]
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity: capacity.max(1),
            current: 0,
            peak: 0,
            samples: 0,
            sum_utilization: 0.0,
            growth_rate: 0.0,
        }
    }

    /// Create for connections (1000)
    #[must_use]
    pub fn for_connections() -> Self {
        Self::new(1000)
    }

    /// Create for storage GB (100)
    #[must_use]
    pub fn for_storage() -> Self {
        Self::new(100)
    }

    /// Update current usage
    pub fn update(&mut self, current: u64) {
        let old = self.current;
        self.current = current;
        if current > self.peak {
            self.peak = current;
        }
        self.samples += 1;
        self.sum_utilization += self.utilization();

        // Calculate growth rate (simple difference)
        if old > 0 {
            self.growth_rate = (current as f64 - old as f64) / old as f64;
        }
    }

    /// Get current utilization as percentage
    #[must_use]
    pub fn utilization(&self) -> f64 {
        (self.current as f64 / self.capacity as f64) * 100.0
    }

    /// Get peak utilization as percentage
    #[must_use]
    pub fn peak_utilization(&self) -> f64 {
        (self.peak as f64 / self.capacity as f64) * 100.0
    }

    /// Get average utilization
    #[must_use]
    pub fn avg_utilization(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.sum_utilization / self.samples as f64
        }
    }

    /// Get remaining capacity
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.capacity.saturating_sub(self.current)
    }

    /// Check if at risk (above threshold)
    #[must_use]
    pub fn at_risk(&self, threshold: f64) -> bool {
        self.utilization() >= threshold
    }

    /// Get growth rate
    #[must_use]
    pub fn growth_rate(&self) -> f64 {
        self.growth_rate
    }

    /// Reset planner
    pub fn reset(&mut self) {
        self.current = 0;
        self.peak = 0;
        self.samples = 0;
        self.sum_utilization = 0.0;
        self.growth_rate = 0.0;
    }
}

// ============================================================================
// DriftTracker - Track clock/timing drift
// ============================================================================

/// O(1) drift tracking for timing synchronization.
/// Monitors deviation from expected intervals and detects clock drift.
#[derive(Debug, Clone)]
pub struct DriftTracker {
    expected_interval_us: u64,
    last_timestamp_us: u64,
    total_drift_us: i64,
    samples: u64,
    max_drift_us: i64,
    min_drift_us: i64,
}

impl Default for DriftTracker {
    fn default() -> Self {
        Self::new(1_000_000) // 1 second expected interval
    }
}

impl DriftTracker {
    /// Create with expected interval in microseconds
    #[must_use]
    pub fn new(expected_interval_us: u64) -> Self {
        Self {
            expected_interval_us: expected_interval_us.max(1),
            last_timestamp_us: 0,
            total_drift_us: 0,
            samples: 0,
            max_drift_us: i64::MIN,
            min_drift_us: i64::MAX,
        }
    }

    /// Create for 60fps (16.67ms interval)
    #[must_use]
    pub fn for_60fps() -> Self {
        Self::new(16_667)
    }

    /// Create for 1 second heartbeat
    #[must_use]
    pub fn for_heartbeat() -> Self {
        Self::new(1_000_000)
    }

    /// Record timestamp and calculate drift
    pub fn record(&mut self, now_us: u64) {
        if self.last_timestamp_us == 0 {
            self.last_timestamp_us = now_us;
            return;
        }

        let actual_interval = now_us.saturating_sub(self.last_timestamp_us);
        let drift = actual_interval as i64 - self.expected_interval_us as i64;

        self.total_drift_us += drift;
        self.samples += 1;

        if drift > self.max_drift_us {
            self.max_drift_us = drift;
        }
        if drift < self.min_drift_us {
            self.min_drift_us = drift;
        }

        self.last_timestamp_us = now_us;
    }

    /// Get average drift in microseconds
    #[must_use]
    pub fn avg_drift_us(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.total_drift_us as f64 / self.samples as f64
        }
    }

    /// Get max drift (positive = late, negative = early)
    #[must_use]
    pub fn max_drift_us(&self) -> i64 {
        if self.samples == 0 {
            0
        } else {
            self.max_drift_us
        }
    }

    /// Get min drift
    #[must_use]
    pub fn min_drift_us(&self) -> i64 {
        if self.samples == 0 {
            0
        } else {
            self.min_drift_us
        }
    }

    /// Check if drift is within tolerance
    #[must_use]
    pub fn is_stable(&self, tolerance_us: i64) -> bool {
        self.avg_drift_us().abs() < tolerance_us as f64
    }

    /// Get drift range
    #[must_use]
    pub fn drift_range_us(&self) -> i64 {
        if self.samples == 0 {
            0
        } else {
            self.max_drift_us - self.min_drift_us
        }
    }

    /// Get sample count
    #[must_use]
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.last_timestamp_us = 0;
        self.total_drift_us = 0;
        self.samples = 0;
        self.max_drift_us = i64::MIN;
        self.min_drift_us = i64::MAX;
    }
}

// ============================================================================
// SemaphoreTracker - Track semaphore/permit usage
// ============================================================================

/// O(1) semaphore usage tracker.
/// Monitors permit acquisition and release patterns.
#[derive(Debug, Clone)]
pub struct SemaphoreTracker {
    total_permits: u32,
    acquired: u32,
    peak_acquired: u32,
    acquisitions: u64,
    releases: u64,
    contentions: u64,
}

impl Default for SemaphoreTracker {
    fn default() -> Self {
        Self::new(10)
    }
}

impl SemaphoreTracker {
    /// Create with total permits
    #[must_use]
    pub fn new(total_permits: u32) -> Self {
        Self {
            total_permits: total_permits.max(1),
            acquired: 0,
            peak_acquired: 0,
            acquisitions: 0,
            releases: 0,
            contentions: 0,
        }
    }

    /// Create for database connections (20)
    #[must_use]
    pub fn for_database() -> Self {
        Self::new(20)
    }

    /// Create for worker threads (8)
    #[must_use]
    pub fn for_workers() -> Self {
        Self::new(8)
    }

    /// Try to acquire permit, returns true if successful
    pub fn try_acquire(&mut self) -> bool {
        if self.acquired < self.total_permits {
            self.acquired += 1;
            self.acquisitions += 1;
            if self.acquired > self.peak_acquired {
                self.peak_acquired = self.acquired;
            }
            true
        } else {
            self.contentions += 1;
            false
        }
    }

    /// Release permit
    pub fn release(&mut self) {
        if self.acquired > 0 {
            self.acquired -= 1;
            self.releases += 1;
        }
    }

    /// Get available permits
    #[must_use]
    pub fn available(&self) -> u32 {
        self.total_permits.saturating_sub(self.acquired)
    }

    /// Get utilization as percentage
    #[must_use]
    pub fn utilization(&self) -> f64 {
        (self.acquired as f64 / self.total_permits as f64) * 100.0
    }

    /// Get peak utilization as percentage
    #[must_use]
    pub fn peak_utilization(&self) -> f64 {
        (self.peak_acquired as f64 / self.total_permits as f64) * 100.0
    }

    /// Get contention rate
    #[must_use]
    pub fn contention_rate(&self) -> f64 {
        let total = self.acquisitions + self.contentions;
        if total == 0 {
            0.0
        } else {
            (self.contentions as f64 / total as f64) * 100.0
        }
    }

    /// Check if healthy (low contention)
    #[must_use]
    pub fn is_healthy(&self, max_contention: f64) -> bool {
        self.contention_rate() <= max_contention
    }

    /// Get total permits
    #[must_use]
    pub fn total_permits(&self) -> u32 {
        self.total_permits
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.acquired = 0;
        self.peak_acquired = 0;
        self.acquisitions = 0;
        self.releases = 0;
        self.contentions = 0;
    }
}
