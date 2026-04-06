// =============================================================================
// EMA TRACKER (trueno-viz O(1) smoothing pattern)
// =============================================================================

/// Exponential Moving Average tracker for O(1) smoothing (trueno-viz pattern)
///
/// Provides real-time smoothed values without storing history.
/// Commonly used for FPS counters, load averages, and trend detection.
#[derive(Debug, Clone)]
pub struct EmaTracker {
    /// Current smoothed value
    value: f64,
    /// Smoothing factor (0.0-1.0, higher = more responsive)
    alpha: f64,
    /// Whether we've received the first sample
    initialized: bool,
}

impl Default for EmaTracker {
    fn default() -> Self {
        Self::new(0.1) // Default 10% weight for new samples
    }
}

impl EmaTracker {
    /// Create a new EMA tracker with given smoothing factor
    ///
    /// Alpha should be between 0.0 and 1.0:
    /// - Higher alpha (e.g., 0.5) = more responsive, less smooth
    /// - Lower alpha (e.g., 0.05) = less responsive, more smooth
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha: alpha.clamp(0.0, 1.0),
            initialized: false,
        }
    }

    /// Create tracker optimized for FPS counting (fast response)
    #[must_use]
    pub fn for_fps() -> Self {
        Self::new(0.3)
    }

    /// Create tracker optimized for load average (slow response)
    #[must_use]
    pub fn for_load() -> Self {
        Self::new(0.05)
    }

    /// Update with a new sample (O(1))
    pub fn update(&mut self, sample: f64) {
        if self.initialized {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        } else {
            self.value = sample;
            self.initialized = true;
        }
    }

    /// Get current smoothed value (O(1))
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Check if tracker has been initialized
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get the alpha (smoothing factor)
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Reset tracker to uninitialized state
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }

    /// Set a new alpha value
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }
}

// =============================================================================
// RATE LIMITER (trueno-viz O(1) throttling pattern)
// =============================================================================

/// Token bucket rate limiter for O(1) throttling (trueno-viz pattern)
///
/// Used to limit update frequency without blocking.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    /// Last allowed time in microseconds
    last_allowed_us: u64,
    /// Minimum interval between allows in microseconds
    interval_us: u64,
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new_hz(60) // Default 60 Hz
    }
}

impl RateLimiter {
    /// Create rate limiter with interval in microseconds
    #[must_use]
    pub fn new(interval_us: u64) -> Self {
        Self {
            last_allowed_us: 0,
            interval_us,
        }
    }

    /// Create rate limiter for given frequency (Hz)
    #[must_use]
    pub fn new_hz(hz: u32) -> Self {
        let interval_us = if hz == 0 {
            1_000_000
        } else {
            1_000_000 / hz as u64
        };
        Self::new(interval_us)
    }

    /// Create rate limiter for given millisecond interval
    #[must_use]
    pub fn new_ms(ms: u64) -> Self {
        Self::new(ms * 1000)
    }

    /// Check if action is allowed now (O(1))
    ///
    /// Returns true and updates timestamp if allowed, false otherwise.
    pub fn check(&mut self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        if now >= self.last_allowed_us + self.interval_us {
            self.last_allowed_us = now;
            true
        } else {
            false
        }
    }

    /// Check without updating (peek)
    #[must_use]
    pub fn would_allow(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        now >= self.last_allowed_us + self.interval_us
    }

    /// Get interval in microseconds
    #[must_use]
    pub fn interval_us(&self) -> u64 {
        self.interval_us
    }

    /// Get interval in Hz
    #[must_use]
    pub fn hz(&self) -> f64 {
        if self.interval_us == 0 {
            0.0
        } else {
            1_000_000.0 / self.interval_us as f64
        }
    }

    /// Reset the limiter
    pub fn reset(&mut self) {
        self.last_allowed_us = 0;
    }
}

// =============================================================================
// THRESHOLD DETECTOR (trueno-viz O(1) level detection pattern)
// =============================================================================

/// Threshold-based level detector for O(1) state transitions (trueno-viz pattern)
///
/// Provides hysteresis to prevent rapid toggling at threshold boundaries.
/// Common use: CPU/memory warning levels, alert thresholds.
#[derive(Debug, Clone)]
pub struct ThresholdDetector {
    /// Low threshold (transition from High to Low when below this)
    low: f64,
    /// High threshold (transition from Low to High when above this)
    high: f64,
    /// Current state (true = high/alert, false = low/normal)
    is_high: bool,
}

impl ThresholdDetector {
    /// Create a new threshold detector with hysteresis
    ///
    /// # Arguments
    /// * `low` - Threshold to transition to normal state
    /// * `high` - Threshold to transition to alert state
    ///
    /// Hysteresis: low < high prevents rapid toggling
    #[must_use]
    pub fn new(low: f64, high: f64) -> Self {
        Self {
            low,
            high: high.max(low), // Ensure high >= low
            is_high: false,
        }
    }

    /// Create detector for percentage thresholds (0-100)
    #[must_use]
    pub fn percent(low: f64, high: f64) -> Self {
        Self::new(low.clamp(0.0, 100.0), high.clamp(0.0, 100.0))
    }

    /// Create detector for CPU/memory warnings (70/90)
    #[must_use]
    pub fn for_resource() -> Self {
        Self::new(70.0, 90.0)
    }

    /// Create detector for temperature warnings (60/80)
    #[must_use]
    pub fn for_temperature() -> Self {
        Self::new(60.0, 80.0)
    }

    /// Update with new value and return whether state changed (O(1))
    pub fn update(&mut self, value: f64) -> bool {
        let was_high = self.is_high;

        if self.is_high && value < self.low {
            self.is_high = false;
        } else if !self.is_high && value > self.high {
            self.is_high = true;
        }

        was_high != self.is_high
    }

    /// Check if currently in high/alert state
    #[must_use]
    pub fn is_high(&self) -> bool {
        self.is_high
    }

    /// Check if currently in low/normal state
    #[must_use]
    pub fn is_low(&self) -> bool {
        !self.is_high
    }

    /// Get the low threshold
    #[must_use]
    pub fn low_threshold(&self) -> f64 {
        self.low
    }

    /// Get the high threshold
    #[must_use]
    pub fn high_threshold(&self) -> f64 {
        self.high
    }

    /// Reset to low/normal state
    pub fn reset(&mut self) {
        self.is_high = false;
    }

    /// Force to high/alert state
    pub fn set_high(&mut self) {
        self.is_high = true;
    }
}

// =============================================================================
// SAMPLE COUNTER (trueno-viz O(1) counting pattern)
// =============================================================================

/// Sample counter with windowed rate calculation (trueno-viz pattern)
///
/// Tracks count and provides rate per second calculation.
#[derive(Debug, Clone)]
pub struct SampleCounter {
    /// Total count
    count: u64,
    /// Count at last rate calculation
    last_count: u64,
    /// Time of last rate calculation in microseconds
    last_time_us: u64,
    /// Calculated rate (samples per second)
    rate: f64,
}

impl Default for SampleCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl SampleCounter {
    /// Create a new counter
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: 0,
            last_count: 0,
            last_time_us: 0,
            rate: 0.0,
        }
    }

    /// Increment count by 1 (O(1))
    pub fn increment(&mut self) {
        self.count += 1;
    }

    /// Increment count by n (O(1))
    pub fn add(&mut self, n: u64) {
        self.count += n;
    }

    /// Get current count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Calculate and get rate (samples per second)
    ///
    /// Should be called periodically (e.g., once per second)
    pub fn calculate_rate(&mut self) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        if self.last_time_us > 0 {
            let elapsed_us = now.saturating_sub(self.last_time_us);
            if elapsed_us > 0 {
                let delta = self.count.saturating_sub(self.last_count);
                self.rate = (delta as f64 * 1_000_000.0) / elapsed_us as f64;
            }
        }

        self.last_count = self.count;
        self.last_time_us = now;
        self.rate
    }

    /// Get last calculated rate without recalculating
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Reset counter
    pub fn reset(&mut self) {
        self.count = 0;
        self.last_count = 0;
        self.last_time_us = 0;
        self.rate = 0.0;
    }
}

// =============================================================================
// BUDGET TRACKER (trueno-viz O(1) budget monitoring pattern)
// =============================================================================

/// Budget tracker for monitoring time/resource consumption (trueno-viz pattern)
///
/// Tracks usage against a budget and calculates utilization percentage.
#[derive(Debug, Clone)]
pub struct BudgetTracker {
    /// Budget limit
    budget: f64,
    /// Current usage
    usage: f64,
    /// Peak usage
    peak: f64,
}

impl BudgetTracker {
    /// Create a new budget tracker
    #[must_use]
    pub fn new(budget: f64) -> Self {
        Self {
            budget: budget.max(0.0),
            usage: 0.0,
            peak: 0.0,
        }
    }

    /// Create for 16ms render budget (60fps)
    #[must_use]
    pub fn for_render() -> Self {
        contract_pre_render!();
        Self::new(16_000.0) // 16ms in microseconds
    }

    /// Create for 1ms compute budget
    #[must_use]
    pub fn for_compute() -> Self {
        Self::new(1_000.0) // 1ms in microseconds
    }

    /// Record usage (O(1))
    pub fn record(&mut self, usage: f64) {
        self.usage = usage;
        self.peak = self.peak.max(usage);
    }

    /// Get current usage
    #[must_use]
    pub fn usage(&self) -> f64 {
        self.usage
    }

    /// Get peak usage
    #[must_use]
    pub fn peak(&self) -> f64 {
        self.peak
    }

    /// Get budget
    #[must_use]
    pub fn budget(&self) -> f64 {
        self.budget
    }

    /// Get utilization percentage (O(1))
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.budget <= 0.0 {
            0.0
        } else {
            (self.usage / self.budget) * 100.0
        }
    }

    /// Get peak utilization percentage (O(1))
    #[must_use]
    pub fn peak_utilization(&self) -> f64 {
        if self.budget <= 0.0 {
            0.0
        } else {
            (self.peak / self.budget) * 100.0
        }
    }

    /// Check if over budget
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        self.usage > self.budget
    }

    /// Get remaining budget
    #[must_use]
    pub fn remaining(&self) -> f64 {
        (self.budget - self.usage).max(0.0)
    }

    /// Reset usage and peak
    pub fn reset(&mut self) {
        self.usage = 0.0;
        self.peak = 0.0;
    }

    /// Set new budget
    pub fn set_budget(&mut self, budget: f64) {
        self.budget = budget.max(0.0);
    }
}

// =============================================================================
// MIN/MAX TRACKER (trueno-viz O(1) extrema tracking pattern)
// =============================================================================

/// Min/Max value tracker with timestamps (trueno-viz pattern)
///
/// Tracks minimum and maximum values along with when they occurred.
/// Useful for monitoring extreme values in metrics over time.
#[derive(Debug, Clone)]
pub struct MinMaxTracker {
    /// Minimum value seen
    min: f64,
    /// Maximum value seen
    max: f64,
    /// Time of minimum (microseconds since epoch)
    min_time_us: u64,
    /// Time of maximum (microseconds since epoch)
    max_time_us: u64,
    /// Sample count
    count: u64,
}

impl Default for MinMaxTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxTracker {
    /// Create a new min/max tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            min: f64::MAX,
            max: f64::MIN,
            min_time_us: 0,
            max_time_us: 0,
            count: 0,
        }
    }

    /// Record a value (O(1))
    pub fn record(&mut self, value: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        if value < self.min {
            self.min = value;
            self.min_time_us = now;
        }
        if value > self.max {
            self.max = value;
            self.max_time_us = now;
        }
        self.count += 1;
    }

    /// Get minimum value (O(1))
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.min)
        } else {
            None
        }
    }

    /// Get maximum value (O(1))
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.max)
        } else {
            None
        }
    }

    /// Get range (max - min) (O(1))
    #[must_use]
    pub fn range(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.max - self.min)
        } else {
            None
        }
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get time since minimum in microseconds
    #[must_use]
    pub fn time_since_min_us(&self) -> u64 {
        if self.min_time_us == 0 {
            return 0;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        now.saturating_sub(self.min_time_us)
    }

    /// Get time since maximum in microseconds
    #[must_use]
    pub fn time_since_max_us(&self) -> u64 {
        if self.max_time_us == 0 {
            return 0;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        now.saturating_sub(self.max_time_us)
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.min = f64::MAX;
        self.max = f64::MIN;
        self.min_time_us = 0;
        self.max_time_us = 0;
        self.count = 0;
    }
}

// =============================================================================
// MOVING WINDOW (trueno-viz O(1) time-windowed aggregation pattern)
// =============================================================================

/// Time-windowed aggregation tracker (trueno-viz pattern)
///
/// Tracks sum/count over a sliding time window for rate calculations.
/// Window expiration is O(1) using bucket rotation.
#[derive(Debug, Clone)]
pub struct MovingWindow {
    /// Current bucket sum
    current_sum: f64,
    /// Current bucket count
    current_count: u64,
    /// Previous bucket sum
    prev_sum: f64,
    /// Previous bucket count
    prev_count: u64,
    /// Window duration in microseconds
    window_us: u64,
    /// Current bucket start time
    bucket_start_us: u64,
}

impl MovingWindow {
    /// Create a new moving window
    #[must_use]
    pub fn new(window_ms: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        Self {
            current_sum: 0.0,
            current_count: 0,
            prev_sum: 0.0,
            prev_count: 0,
            window_us: window_ms * 1000,
            bucket_start_us: now,
        }
    }

    /// Create 1-second window
    #[must_use]
    pub fn one_second() -> Self {
        Self::new(1000)
    }

    /// Create 1-minute window
    #[must_use]
    pub fn one_minute() -> Self {
        Self::new(60_000)
    }

    /// Record a value (O(1) with potential bucket rotation)
    pub fn record(&mut self, value: f64) {
        self.maybe_rotate();
        self.current_sum += value;
        self.current_count += 1;
    }

    /// Increment count by 1 (O(1))
    pub fn increment(&mut self) {
        self.record(1.0);
    }

    /// Check and rotate buckets if needed (O(1))
    fn maybe_rotate(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let elapsed = now.saturating_sub(self.bucket_start_us);

        if elapsed >= self.window_us {
            // Rotate: current becomes previous
            self.prev_sum = self.current_sum;
            self.prev_count = self.current_count;
            self.current_sum = 0.0;
            self.current_count = 0;
            self.bucket_start_us = now;
        }
    }

    /// Get sum over window (O(1))
    #[must_use]
    pub fn sum(&mut self) -> f64 {
        self.maybe_rotate();
        self.current_sum + self.prev_sum
    }

    /// Get count over window (O(1))
    #[must_use]
    pub fn count(&mut self) -> u64 {
        self.maybe_rotate();
        self.current_count + self.prev_count
    }

    /// Get rate per second (O(1))
    #[must_use]
    pub fn rate_per_second(&mut self) -> f64 {
        self.maybe_rotate();
        let total = self.current_sum + self.prev_sum;
        let window_secs = (self.window_us as f64) / 1_000_000.0;
        if window_secs > 0.0 {
            total / window_secs
        } else {
            0.0
        }
    }

    /// Get count rate per second (O(1))
    #[must_use]
    pub fn count_rate(&mut self) -> f64 {
        self.maybe_rotate();
        let total = self.current_count + self.prev_count;
        let window_secs = (self.window_us as f64) / 1_000_000.0;
        if window_secs > 0.0 {
            total as f64 / window_secs
        } else {
            0.0
        }
    }

    /// Reset window
    pub fn reset(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        self.current_sum = 0.0;
        self.current_count = 0;
        self.prev_sum = 0.0;
        self.prev_count = 0;
        self.bucket_start_us = now;
    }
}

// =============================================================================
// PERCENTILE TRACKER (trueno-viz O(1) approximate percentile pattern)
// =============================================================================

/// Approximate percentile tracker using fixed buckets (trueno-viz pattern)
///
/// Provides O(1) approximate percentiles using histogram-based estimation.
/// Buckets: [0-1, 1-5, 5-10, 10-25, 25-50, 50-100, 100-250, 250-500, 500-1000, 1000+] ms
#[derive(Debug, Clone)]
pub struct PercentileTracker {
    /// Bucket counts
    buckets: [u64; 10],
    /// Total count
    count: u64,
    /// Bucket boundaries in microseconds
    boundaries: [u64; 10],
}

impl Default for PercentileTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PercentileTracker {
    /// Create a new percentile tracker with default latency buckets
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: [0; 10],
            count: 0,
            // Boundaries in microseconds: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1000ms, infinity
            boundaries: [
                1_000,     // 0-1ms
                5_000,     // 1-5ms
                10_000,    // 5-10ms
                25_000,    // 10-25ms
                50_000,    // 25-50ms
                100_000,   // 50-100ms
                250_000,   // 100-250ms
                500_000,   // 250-500ms
                1_000_000, // 500-1000ms
                u64::MAX,  // 1000ms+
            ],
        }
    }

    /// Create with custom bucket boundaries (in microseconds)
    #[must_use]
    pub fn with_boundaries(boundaries: [u64; 10]) -> Self {
        Self {
            buckets: [0; 10],
            count: 0,
            boundaries,
        }
    }

    /// Record a value in microseconds (O(1))
    pub fn record_us(&mut self, value_us: u64) {
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value_us < boundary {
                self.buckets[i] += 1;
                self.count += 1;
                return;
            }
        }
        // Shouldn't reach here due to u64::MAX boundary
        self.buckets[9] += 1;
        self.count += 1;
    }

    /// Record a value in milliseconds (O(1))
    pub fn record_ms(&mut self, value_ms: f64) {
        self.record_us((value_ms * 1000.0) as u64);
    }

    /// Get approximate percentile value in microseconds (O(1))
    #[must_use]
    pub fn percentile_us(&self, pct: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }

        let target = ((pct / 100.0) * self.count as f64) as u64;
        let mut cumulative = 0u64;

        for (i, &bucket_count) in self.buckets.iter().enumerate() {
            cumulative += bucket_count;
            if cumulative >= target {
                // Return midpoint of bucket
                let lower = if i == 0 { 0 } else { self.boundaries[i - 1] };
                let upper = self.boundaries[i];
                if upper == u64::MAX {
                    return lower + 500_000; // Estimate for last bucket
                }
                return (lower + upper) / 2;
            }
        }

        self.boundaries[8] // Return 1s as fallback
    }

    /// Get approximate percentile value in milliseconds (O(1))
    #[must_use]
    pub fn percentile_ms(&self, pct: f64) -> f64 {
        self.percentile_us(pct) as f64 / 1000.0
    }

    /// Get p50 (median) in milliseconds
    #[must_use]
    pub fn p50_ms(&self) -> f64 {
        self.percentile_ms(50.0)
    }

    /// Get p90 in milliseconds
    #[must_use]
    pub fn p90_ms(&self) -> f64 {
        self.percentile_ms(90.0)
    }

    /// Get p99 in milliseconds
    #[must_use]
    pub fn p99_ms(&self) -> f64 {
        self.percentile_ms(99.0)
    }

    /// Get total count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.buckets = [0; 10];
        self.count = 0;
    }
}

// =============================================================================
// STATE TRACKER (trueno-viz O(1) state machine pattern)
// =============================================================================

/// State transition tracker with history (trueno-viz pattern)
///
/// Tracks state transitions and time spent in each state.
/// Useful for monitoring component lifecycle or connection states.
#[derive(Debug, Clone)]
pub struct StateTracker<const N: usize> {
    /// Current state index (0..N-1)
    current: usize,
    /// Time entered current state (microseconds since epoch)
    entered_us: u64,
    /// Total time spent in each state (microseconds)
    durations: [u64; N],
    /// Transition count for each state
    transitions: [u64; N],
}

impl<const N: usize> Default for StateTracker<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> StateTracker<N> {
    /// Create a new state tracker starting in state 0
    #[must_use]
    pub fn new() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        let mut transitions = [0u64; N];
        if N > 0 {
            transitions[0] = 1; // Initial transition to state 0
        }

        Self {
            current: 0,
            entered_us: now,
            durations: [0u64; N],
            transitions,
        }
    }

    /// Transition to a new state (O(1))
    pub fn transition(&mut self, new_state: usize) {
        if new_state >= N {
            return; // Invalid state
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Record time in previous state
        let elapsed = now.saturating_sub(self.entered_us);
        self.durations[self.current] += elapsed;

        // Transition to new state
        self.current = new_state;
        self.entered_us = now;
        self.transitions[new_state] += 1;
    }

    /// Get current state (O(1))
    #[must_use]
    pub fn current(&self) -> usize {
        self.current
    }

    /// Get time in current state in microseconds (O(1))
    #[must_use]
    pub fn time_in_current_us(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        now.saturating_sub(self.entered_us)
    }

    /// Get total time in state in microseconds (O(1))
    #[must_use]
    pub fn total_time_in_state_us(&self, state: usize) -> u64 {
        if state >= N {
            return 0;
        }
        if state == self.current {
            self.durations[state] + self.time_in_current_us()
        } else {
            self.durations[state]
        }
    }

    /// Get transition count for state (O(1))
    #[must_use]
    pub fn transition_count(&self, state: usize) -> u64 {
        if state >= N {
            0
        } else {
            self.transitions[state]
        }
    }

    /// Get total transitions (O(N))
    #[must_use]
    pub fn total_transitions(&self) -> u64 {
        self.transitions.iter().sum()
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        self.current = 0;
        self.entered_us = now;
        self.durations = [0u64; N];
        self.transitions = [0u64; N];
        if N > 0 {
            self.transitions[0] = 1;
        }
    }
}

// =============================================================================
// CHANGE DETECTOR (trueno-viz O(1) significant change detection)
// =============================================================================

/// Significant change detector (trueno-viz pattern)
///
/// Detects when a value has changed significantly from a baseline.
/// Uses both absolute and relative thresholds.
#[derive(Debug, Clone)]
pub struct ChangeDetector {
    /// Baseline value
    baseline: f64,
    /// Absolute change threshold
    abs_threshold: f64,
    /// Relative change threshold (percentage)
    rel_threshold: f64,
    /// Last reported value
    last_value: f64,
    /// Number of changes detected
    change_count: u64,
}

impl Default for ChangeDetector {
    fn default() -> Self {
        Self::new(0.0, 1.0, 5.0)
    }
}

impl ChangeDetector {
    /// Create a new change detector
    ///
    /// # Arguments
    /// * `baseline` - Initial baseline value
    /// * `abs_threshold` - Absolute change required to trigger
    /// * `rel_threshold` - Relative change (%) required to trigger
    #[must_use]
    pub fn new(baseline: f64, abs_threshold: f64, rel_threshold: f64) -> Self {
        Self {
            baseline,
            abs_threshold: abs_threshold.abs(),
            rel_threshold: rel_threshold.abs(),
            last_value: baseline,
            change_count: 0,
        }
    }

    /// Create for percentage monitoring (1% absolute, 5% relative)
    #[must_use]
    pub fn for_percentage() -> Self {
        Self::new(0.0, 1.0, 5.0)
    }

    /// Create for latency monitoring (1ms absolute, 10% relative)
    #[must_use]
    pub fn for_latency() -> Self {
        Self::new(0.0, 1000.0, 10.0)
    }

    /// Check if value has changed significantly (O(1))
    #[must_use]
    pub fn has_changed(&self, value: f64) -> bool {
        let abs_diff = (value - self.last_value).abs();

        // Check absolute threshold
        if abs_diff >= self.abs_threshold {
            return true;
        }

        // Check relative threshold
        if self.last_value.abs() > f64::EPSILON {
            let rel_diff = (abs_diff / self.last_value.abs()) * 100.0;
            if rel_diff >= self.rel_threshold {
                return true;
            }
        }

        false
    }

    /// Update with new value and return whether it changed significantly (O(1))
    pub fn update(&mut self, value: f64) -> bool {
        let changed = self.has_changed(value);
        if changed {
            self.change_count += 1;
        }
        self.last_value = value;
        changed
    }

    /// Update baseline to current value (O(1))
    pub fn update_baseline(&mut self) {
        self.baseline = self.last_value;
    }

    /// Set baseline to specific value (O(1))
    pub fn set_baseline(&mut self, baseline: f64) {
        self.baseline = baseline;
    }

    /// Get current baseline
    #[must_use]
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// Get last value
    #[must_use]
    pub fn last_value(&self) -> f64 {
        self.last_value
    }

    /// Get change count
    #[must_use]
    pub fn change_count(&self) -> u64 {
        self.change_count
    }

    /// Get change from baseline
    #[must_use]
    pub fn change_from_baseline(&self) -> f64 {
        self.last_value - self.baseline
    }

    /// Get relative change from baseline (percentage)
    #[must_use]
    pub fn relative_change(&self) -> f64 {
        if self.baseline.abs() > f64::EPSILON {
            ((self.last_value - self.baseline) / self.baseline.abs()) * 100.0
        } else {
            0.0
        }
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.last_value = self.baseline;
        self.change_count = 0;
    }
}

// =============================================================================
// ACCUMULATOR (trueno-viz O(1) overflow-safe accumulation)
// =============================================================================

/// Overflow-safe accumulator (trueno-viz pattern)
///
/// Accumulates values with automatic overflow detection and handling.
/// Useful for counters that may wrap (network bytes, disk I/O).
#[derive(Debug, Clone)]
pub struct Accumulator {
    /// Current accumulated value
    value: u64,
    /// Previous raw value (for delta calculation)
    prev_raw: u64,
    /// Whether we've seen the first value
    initialized: bool,
    /// Overflow count
    overflows: u64,
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    /// Create a new accumulator
    #[must_use]
    pub fn new() -> Self {
        Self {
            value: 0,
            prev_raw: 0,
            initialized: false,
            overflows: 0,
        }
    }

    /// Update with a raw counter value (O(1))
    ///
    /// Handles counter wraps/overflows automatically.
    pub fn update(&mut self, raw: u64) {
        if !self.initialized {
            self.prev_raw = raw;
            self.initialized = true;
            return;
        }

        // Calculate delta, handling overflow
        let delta = if raw >= self.prev_raw {
            raw - self.prev_raw
        } else {
            // Counter wrapped
            self.overflows += 1;
            // Assume wrap to 0 from max
            (u64::MAX - self.prev_raw) + raw + 1
        };

        self.value += delta;
        self.prev_raw = raw;
    }

    /// Add a delta value directly (O(1))
    pub fn add(&mut self, delta: u64) {
        self.value += delta;
        self.initialized = true;
    }

    /// Get accumulated value (O(1))
    #[must_use]
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Get overflow count (O(1))
    #[must_use]
    pub fn overflows(&self) -> u64 {
        self.overflows
    }

    /// Check if initialized
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get last raw value
    #[must_use]
    pub fn last_raw(&self) -> u64 {
        self.prev_raw
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.value = 0;
        self.prev_raw = 0;
        self.initialized = false;
        self.overflows = 0;
    }
}

// =============================================================================
// EVENT COUNTER (trueno-viz O(1) categorized event counting)
// =============================================================================

/// Categorized event counter (trueno-viz pattern)
///
/// Counts events by category with O(1) increment and lookup.
/// Useful for error categorization, request types, etc.
#[derive(Debug, Clone)]
pub struct EventCounter<const N: usize> {
    /// Counts per category
    counts: [u64; N],
    /// Total count
    total: u64,
}

impl<const N: usize> Default for EventCounter<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> EventCounter<N> {
    /// Create a new event counter
    #[must_use]
    pub fn new() -> Self {
        Self {
            counts: [0u64; N],
            total: 0,
        }
    }

    /// Increment category count (O(1))
    pub fn increment(&mut self, category: usize) {
        if category < N {
            self.counts[category] += 1;
            self.total += 1;
        }
    }

    /// Add to category count (O(1))
    pub fn add(&mut self, category: usize, count: u64) {
        if category < N {
            self.counts[category] += count;
            self.total += count;
        }
    }

    /// Get category count (O(1))
    #[must_use]
    pub fn count(&self, category: usize) -> u64 {
        if category < N {
            self.counts[category]
        } else {
            0
        }
    }

    /// Get total count (O(1))
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Get category percentage (O(1))
    #[must_use]
    pub fn percentage(&self, category: usize) -> f64 {
        if self.total == 0 || category >= N {
            0.0
        } else {
            (self.counts[category] as f64 / self.total as f64) * 100.0
        }
    }

    /// Get dominant category (O(N))
    #[must_use]
    pub fn dominant(&self) -> Option<usize> {
        if self.total == 0 {
            return None;
        }
        self.counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
    }

    /// Reset all counts
    pub fn reset(&mut self) {
        self.counts = [0u64; N];
        self.total = 0;
    }
}

// =============================================================================
// TREND DETECTOR (trueno-viz O(1) trend analysis pattern)
// =============================================================================

/// Trend detection using linear regression slope (trueno-viz pattern)
///
/// Detects upward/downward/flat trends using a sliding window approach.
/// Uses simplified slope calculation for O(1) updates.
#[derive(Debug, Clone)]
pub struct TrendDetector {
    /// Sum of values
    sum: f64,
    /// Sum of index * value (for slope)
    sum_xy: f64,
    /// Current index
    index: u64,
    /// Number of samples
    count: u64,
    /// Threshold for trend detection (slope magnitude)
    threshold: f64,
}

impl Default for TrendDetector {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    /// Values increasing
    Up,
    /// Values decreasing
    Down,
    /// Values stable
    Flat,
    /// Not enough data
    Unknown,
}

impl TrendDetector {
    /// Create a new trend detector
    ///
    /// # Arguments
    /// * `threshold` - Minimum slope magnitude to detect trend
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            sum: 0.0,
            sum_xy: 0.0,
            index: 0,
            count: 0,
            threshold: threshold.abs(),
        }
    }

    /// Create for percentage changes (0.5% threshold)
    #[must_use]
    pub fn for_percentage() -> Self {
        Self::new(0.5)
    }

    /// Create for latency changes (1ms threshold)
    #[must_use]
    pub fn for_latency() -> Self {
        Self::new(1.0)
    }

    /// Update with a new value (O(1))
    pub fn update(&mut self, value: f64) {
        self.sum += value;
        self.sum_xy += (self.index as f64) * value;
        self.index += 1;
        self.count += 1;
    }

    /// Calculate slope (O(1))
    ///
    /// Uses least squares linear regression formula:
    /// slope = (n * Σxy - Σx * Σy) / (n * Σx² - (Σx)²)
    #[must_use]
    pub fn slope(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }

        let n = self.count as f64;
        let sum_x = (self.count * (self.count - 1)) as f64 / 2.0; // Sum of 0..n-1
        let sum_x2 = (self.count * (self.count - 1) * (2 * self.count - 1)) as f64 / 6.0;

        // Linear regression denominator: n * Σx² - (Σx)²
        let sum_x_squared = sum_x.powi(2);
        let denominator = n * sum_x2 - sum_x_squared;
        if denominator.abs() < f64::EPSILON {
            return 0.0;
        }

        (n * self.sum_xy - sum_x * self.sum) / denominator
    }

    /// Get current trend (O(1))
    #[must_use]
    pub fn trend(&self) -> Trend {
        if self.count < 3 {
            return Trend::Unknown;
        }

        let slope = self.slope();
        if slope > self.threshold {
            Trend::Up
        } else if slope < -self.threshold {
            Trend::Down
        } else {
            Trend::Flat
        }
    }

    /// Check if trending up
    #[must_use]
    pub fn is_trending_up(&self) -> bool {
        self.trend() == Trend::Up
    }

    /// Check if trending down
    #[must_use]
    pub fn is_trending_down(&self) -> bool {
        self.trend() == Trend::Down
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.sum_xy = 0.0;
        self.index = 0;
        self.count = 0;
    }
}

// =============================================================================
// ANOMALY DETECTOR (trueno-viz O(1) z-score anomaly detection)
// =============================================================================

/// Z-score based anomaly detector (trueno-viz pattern)
///
/// Detects anomalies using running mean/variance and z-score threshold.
/// Useful for alerting on unusual values.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Running mean
    mean: f64,
    /// Running M2 (sum of squared differences)
    m2: f64,
    /// Sample count
    count: u64,
    /// Z-score threshold for anomaly
    threshold: f64,
    /// Last value
    last_value: f64,
    /// Anomaly count
    anomaly_count: u64,
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new(3.0)
    }
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    ///
    /// # Arguments
    /// * `threshold` - Z-score threshold (typically 2.0-3.0)
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
            threshold: threshold.abs(),
            last_value: 0.0,
            anomaly_count: 0,
        }
    }

    /// Create with 2-sigma threshold (95% confidence)
    #[must_use]
    pub fn two_sigma() -> Self {
        Self::new(2.0)
    }

    /// Create with 3-sigma threshold (99.7% confidence)
    #[must_use]
    pub fn three_sigma() -> Self {
        Self::new(3.0)
    }

    /// Update with a new value and return whether it's an anomaly (O(1))
    ///
    /// Uses Welford's online algorithm for running variance.
    pub fn update(&mut self, value: f64) -> bool {
        self.last_value = value;
        self.count += 1;

        // First value can't be an anomaly
        if self.count == 1 {
            self.mean = value;
            return false;
        }

        // Calculate z-score before updating stats
        let is_anomaly = self.is_anomaly(value);
        if is_anomaly {
            self.anomaly_count += 1;
        }

        // Update running stats (Welford's algorithm)
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        is_anomaly
    }

    /// Check if a value would be an anomaly (O(1))
    #[must_use]
    pub fn is_anomaly(&self, value: f64) -> bool {
        if self.count < 10 {
            return false; // Need enough samples
        }

        let z = self.z_score(value);
        z.abs() > self.threshold
    }

    /// Calculate z-score for a value (O(1))
    #[must_use]
    pub fn z_score(&self, value: f64) -> f64 {
        let std_dev = self.std_dev();
        if std_dev < f64::EPSILON {
            return 0.0;
        }
        (value - self.mean) / std_dev
    }

    /// Get running mean (O(1))
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get running variance (O(1))
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get running standard deviation (O(1))
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get anomaly count
    #[must_use]
    pub fn anomaly_count(&self) -> u64 {
        self.anomaly_count
    }

    /// Get anomaly rate (percentage)
    #[must_use]
    pub fn anomaly_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.anomaly_count as f64 / self.count as f64) * 100.0
        }
    }

    /// Get threshold
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
        self.last_value = 0.0;
        self.anomaly_count = 0;
    }
}
