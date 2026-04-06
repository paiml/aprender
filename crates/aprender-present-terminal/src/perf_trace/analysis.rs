// =============================================================================
// THROUGHPUT TRACKER (trueno-viz O(1) rate calculation)
// =============================================================================

/// Throughput tracker for bytes/ops per second (trueno-viz pattern)
///
/// Tracks cumulative totals and calculates rates over time intervals.
#[derive(Debug, Clone)]
pub struct ThroughputTracker {
    /// Total bytes/ops
    total: u64,
    /// Previous total (for delta)
    prev_total: u64,
    /// Last calculation time (microseconds)
    last_time_us: u64,
    /// Calculated rate (units per second)
    rate: f64,
    /// Peak rate
    peak_rate: f64,
}

impl Default for ThroughputTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ThroughputTracker {
    /// Create a new throughput tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            total: 0,
            prev_total: 0,
            last_time_us: 0,
            rate: 0.0,
            peak_rate: 0.0,
        }
    }

    /// Add bytes/ops (O(1))
    pub fn add(&mut self, count: u64) {
        self.total += count;
    }

    /// Calculate rate (should be called periodically) (O(1))
    pub fn calculate_rate(&mut self) -> f64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        if self.last_time_us > 0 {
            let elapsed_us = now.saturating_sub(self.last_time_us);
            if elapsed_us > 0 {
                let delta = self.total.saturating_sub(self.prev_total);
                self.rate = (delta as f64 * 1_000_000.0) / elapsed_us as f64;
                self.peak_rate = self.peak_rate.max(self.rate);
            }
        }

        self.prev_total = self.total;
        self.last_time_us = now;
        self.rate
    }

    /// Get current rate (O(1))
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Get peak rate (O(1))
    #[must_use]
    pub fn peak_rate(&self) -> f64 {
        self.peak_rate
    }

    /// Get total (O(1))
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Format rate as human-readable string (O(1))
    #[must_use]
    pub fn format_rate(&self) -> String {
        let rate = self.rate;
        if rate >= 1_000_000_000.0 {
            format!("{:.1}G/s", rate / 1_000_000_000.0)
        } else if rate >= 1_000_000.0 {
            format!("{:.1}M/s", rate / 1_000_000.0)
        } else if rate >= 1_000.0 {
            format!("{:.1}K/s", rate / 1_000.0)
        } else {
            format!("{:.0}/s", rate)
        }
    }

    /// Format rate as bytes/second (O(1))
    #[must_use]
    pub fn format_bytes_rate(&self) -> String {
        let rate = self.rate;
        if rate >= 1_073_741_824.0 {
            format!("{:.1}GB/s", rate / 1_073_741_824.0)
        } else if rate >= 1_048_576.0 {
            format!("{:.1}MB/s", rate / 1_048_576.0)
        } else if rate >= 1_024.0 {
            format!("{:.1}KB/s", rate / 1_024.0)
        } else {
            format!("{:.0}B/s", rate)
        }
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.total = 0;
        self.prev_total = 0;
        self.last_time_us = 0;
        self.rate = 0.0;
        self.peak_rate = 0.0;
    }
}

// =============================================================================
// JITTER TRACKER (trueno-viz O(1) latency jitter analysis)
// =============================================================================

/// Jitter tracker for latency variation (trueno-viz pattern)
///
/// Tracks inter-arrival time variation (jitter) common in network/audio.
#[derive(Debug, Clone)]
pub struct JitterTracker {
    /// Previous value
    prev: f64,
    /// Running jitter (smoothed)
    jitter: f64,
    /// Peak jitter
    peak_jitter: f64,
    /// Sample count
    count: u64,
    /// Smoothing factor (like RFC 3550)
    alpha: f64,
}

impl Default for JitterTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl JitterTracker {
    /// Create a new jitter tracker with RFC 3550 smoothing
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev: 0.0,
            jitter: 0.0,
            peak_jitter: 0.0,
            count: 0,
            alpha: 1.0 / 16.0, // RFC 3550 default
        }
    }

    /// Create with custom smoothing factor
    #[must_use]
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            prev: 0.0,
            jitter: 0.0,
            peak_jitter: 0.0,
            count: 0,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Update with new inter-arrival time (O(1))
    ///
    /// Uses RFC 3550 jitter calculation: J = J + (|D| - J) / 16
    pub fn update(&mut self, value: f64) {
        self.count += 1;

        if self.count == 1 {
            self.prev = value;
            return;
        }

        // Calculate difference from previous
        let diff = (value - self.prev).abs();
        self.prev = value;

        // Exponential smoothing (RFC 3550 style)
        self.jitter += self.alpha * (diff - self.jitter);
        self.peak_jitter = self.peak_jitter.max(self.jitter);
    }

    /// Get current jitter (O(1))
    #[must_use]
    pub fn jitter(&self) -> f64 {
        self.jitter
    }

    /// Get peak jitter (O(1))
    #[must_use]
    pub fn peak_jitter(&self) -> f64 {
        self.peak_jitter
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Check if jitter exceeds threshold
    #[must_use]
    pub fn exceeds(&self, threshold: f64) -> bool {
        self.jitter > threshold
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.prev = 0.0;
        self.jitter = 0.0;
        self.peak_jitter = 0.0;
        self.count = 0;
    }
}

// =============================================================================
// DERIVATIVE TRACKER (trueno-viz O(1) rate-of-change pattern)
// =============================================================================

/// First derivative (rate of change) tracker (trueno-viz pattern)
///
/// Tracks instantaneous and smoothed rate of change for metrics.
/// Useful for detecting acceleration/deceleration in CPU, memory, etc.
#[derive(Debug, Clone)]
pub struct DerivativeTracker {
    /// Previous value
    prev: f64,
    /// Previous time (microseconds)
    prev_time_us: u64,
    /// Instantaneous derivative
    derivative: f64,
    /// Smoothed derivative (EMA)
    smoothed: f64,
    /// Smoothing factor
    alpha: f64,
    /// Sample count
    count: u64,
}

impl Default for DerivativeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl DerivativeTracker {
    /// Create a new derivative tracker with default smoothing (0.3)
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev: 0.0,
            prev_time_us: 0,
            derivative: 0.0,
            smoothed: 0.0,
            alpha: 0.3,
            count: 0,
        }
    }

    /// Create with custom smoothing factor
    #[must_use]
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            prev: 0.0,
            prev_time_us: 0,
            derivative: 0.0,
            smoothed: 0.0,
            alpha: alpha.clamp(0.0, 1.0),
            count: 0,
        }
    }

    /// Update with new value (O(1))
    pub fn update(&mut self, value: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        self.count += 1;

        if self.count == 1 {
            self.prev = value;
            self.prev_time_us = now;
            return;
        }

        let dt = (now.saturating_sub(self.prev_time_us)) as f64 / 1_000_000.0; // seconds
        if dt > 0.0 {
            self.derivative = (value - self.prev) / dt;
            self.smoothed = self.alpha * self.derivative + (1.0 - self.alpha) * self.smoothed;
        }

        self.prev = value;
        self.prev_time_us = now;
    }

    /// Update with explicit delta time (for testing)
    pub fn update_with_dt(&mut self, value: f64, dt_secs: f64) {
        self.count += 1;

        if self.count == 1 {
            self.prev = value;
            return;
        }

        if dt_secs > 0.0 {
            self.derivative = (value - self.prev) / dt_secs;
            self.smoothed = self.alpha * self.derivative + (1.0 - self.alpha) * self.smoothed;
        }

        self.prev = value;
    }

    /// Get instantaneous derivative (O(1))
    #[must_use]
    pub fn derivative(&self) -> f64 {
        self.derivative
    }

    /// Get smoothed derivative (O(1))
    #[must_use]
    pub fn smoothed(&self) -> f64 {
        self.smoothed
    }

    /// Check if accelerating (positive derivative)
    #[must_use]
    pub fn is_accelerating(&self) -> bool {
        self.smoothed > 0.0
    }

    /// Check if decelerating (negative derivative)
    #[must_use]
    pub fn is_decelerating(&self) -> bool {
        self.smoothed < 0.0
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.prev = 0.0;
        self.prev_time_us = 0;
        self.derivative = 0.0;
        self.smoothed = 0.0;
        self.count = 0;
    }
}

// =============================================================================
// INTEGRAL TRACKER (trueno-viz O(1) cumulative area pattern)
// =============================================================================

/// Integral (cumulative area) tracker (trueno-viz pattern)
///
/// Tracks cumulative area under the curve using trapezoidal rule.
/// Useful for energy consumption, work done, cumulative load.
#[derive(Debug, Clone)]
pub struct IntegralTracker {
    /// Previous value
    prev: f64,
    /// Previous time (microseconds)
    prev_time_us: u64,
    /// Cumulative integral
    integral: f64,
    /// Sample count
    count: u64,
}

impl Default for IntegralTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegralTracker {
    /// Create a new integral tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev: 0.0,
            prev_time_us: 0,
            integral: 0.0,
            count: 0,
        }
    }

    /// Update with new value (O(1))
    ///
    /// Uses trapezoidal rule: area = (v1 + v2) / 2 * dt
    pub fn update(&mut self, value: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        self.count += 1;

        if self.count == 1 {
            self.prev = value;
            self.prev_time_us = now;
            return;
        }

        let dt = (now.saturating_sub(self.prev_time_us)) as f64 / 1_000_000.0; // seconds
                                                                               // Trapezoidal rule
        self.integral += (self.prev + value) / 2.0 * dt;

        self.prev = value;
        self.prev_time_us = now;
    }

    /// Update with explicit delta time (for testing)
    pub fn update_with_dt(&mut self, value: f64, dt_secs: f64) {
        self.count += 1;

        if self.count == 1 {
            self.prev = value;
            return;
        }

        // Trapezoidal rule
        self.integral += (self.prev + value) / 2.0 * dt_secs;
        self.prev = value;
    }

    /// Get cumulative integral (O(1))
    #[must_use]
    pub fn integral(&self) -> f64 {
        self.integral
    }

    /// Get average value (integral / time) (O(1))
    #[must_use]
    pub fn average(&self) -> f64 {
        if self.count < 2 {
            return self.prev;
        }
        // Would need total time tracking for true average
        self.prev
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.prev = 0.0;
        self.prev_time_us = 0;
        self.integral = 0.0;
        self.count = 0;
    }
}

// =============================================================================
// CORRELATION TRACKER (trueno-viz O(1) running correlation)
// =============================================================================

/// Running correlation coefficient tracker (trueno-viz pattern)
///
/// Tracks Pearson correlation between two variables using online algorithm.
/// Useful for finding related metrics (CPU vs memory, network vs disk).
#[derive(Debug, Clone)]
pub struct CorrelationTracker {
    /// Mean of X
    mean_x: f64,
    /// Mean of Y
    mean_y: f64,
    /// Sum of (xi - mean_x) * (yi - mean_y)
    cov_sum: f64,
    /// Sum of (xi - mean_x)^2
    var_x_sum: f64,
    /// Sum of (yi - mean_y)^2
    var_y_sum: f64,
    /// Sample count
    count: u64,
}

impl Default for CorrelationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CorrelationTracker {
    /// Create a new correlation tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            mean_x: 0.0,
            mean_y: 0.0,
            cov_sum: 0.0,
            var_x_sum: 0.0,
            var_y_sum: 0.0,
            count: 0,
        }
    }

    /// Update with new (x, y) pair (O(1))
    ///
    /// Uses Welford's online algorithm for covariance.
    pub fn update(&mut self, x: f64, y: f64) {
        self.count += 1;
        let n = self.count as f64;

        // Update means
        let delta_x = x - self.mean_x;
        let delta_y = y - self.mean_y;

        self.mean_x += delta_x / n;
        self.mean_y += delta_y / n;

        // Update covariance and variance sums
        let delta_x2 = x - self.mean_x;
        let delta_y2 = y - self.mean_y;

        self.cov_sum += delta_x * delta_y2;
        self.var_x_sum += delta_x * delta_x2;
        self.var_y_sum += delta_y * delta_y2;
    }

    /// Get correlation coefficient (O(1))
    ///
    /// Returns value in [-1, 1] or 0 if insufficient data.
    #[must_use]
    pub fn correlation(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }

        let denominator = (self.var_x_sum * self.var_y_sum).sqrt();
        if denominator < f64::EPSILON {
            return 0.0;
        }

        (self.cov_sum / denominator).clamp(-1.0, 1.0)
    }

    /// Check if positively correlated (r > 0.5)
    #[must_use]
    pub fn is_positive(&self) -> bool {
        self.correlation() > 0.5
    }

    /// Check if negatively correlated (r < -0.5)
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.correlation() < -0.5
    }

    /// Check if strongly correlated (|r| > 0.7)
    #[must_use]
    pub fn is_strong(&self) -> bool {
        self.correlation().abs() > 0.7
    }

    /// Get covariance (O(1))
    #[must_use]
    pub fn covariance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.cov_sum / (self.count - 1) as f64
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.mean_x = 0.0;
        self.mean_y = 0.0;
        self.cov_sum = 0.0;
        self.var_x_sum = 0.0;
        self.var_y_sum = 0.0;
        self.count = 0;
    }
}

// =============================================================================
// CIRCUIT BREAKER (trueno-viz O(1) resilience pattern)
// =============================================================================

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation, requests allowed
    Closed,
    /// Too many failures, requests blocked
    Open,
    /// Testing if service recovered
    HalfOpen,
}

/// Circuit breaker for failure handling (trueno-viz pattern)
///
/// Prevents cascading failures by temporarily blocking requests
/// after repeated failures.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Current state
    state: CircuitState,
    /// Failure count
    failures: u64,
    /// Success count (in half-open state)
    successes: u64,
    /// Failure threshold to open circuit
    failure_threshold: u64,
    /// Success threshold to close circuit
    success_threshold: u64,
    /// Time circuit was opened (microseconds)
    opened_at: u64,
    /// Timeout before trying half-open (microseconds)
    timeout_us: u64,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, 3, 30_000_000) // 5 failures, 3 successes, 30s timeout
    }
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    ///
    /// # Arguments
    /// * `failure_threshold` - Failures before opening
    /// * `success_threshold` - Successes in half-open before closing
    /// * `timeout_us` - Microseconds before trying half-open
    #[must_use]
    pub fn new(failure_threshold: u64, success_threshold: u64, timeout_us: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            failures: 0,
            successes: 0,
            failure_threshold,
            success_threshold,
            opened_at: 0,
            timeout_us,
        }
    }

    /// Create for network operations (5 failures, 30s timeout)
    #[must_use]
    pub fn for_network() -> Self {
        Self::new(5, 3, 30_000_000)
    }

    /// Create for fast-fail (3 failures, 5s timeout)
    #[must_use]
    pub fn for_fast_fail() -> Self {
        Self::new(3, 2, 5_000_000)
    }

    /// Check if request is allowed (O(1))
    #[must_use]
    pub fn is_allowed(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;

                if now.saturating_sub(self.opened_at) >= self.timeout_us {
                    self.state = CircuitState::HalfOpen;
                    self.successes = 0;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a success (O(1))
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failures = 0;
            }
            CircuitState::HalfOpen => {
                self.successes += 1;
                if self.successes >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failures = 0;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failure (O(1))
    pub fn record_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failures += 1;
                if self.failures >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    self.opened_at = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_micros() as u64;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.opened_at = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
            }
            CircuitState::Open => {}
        }
    }

    /// Get current state (O(1))
    #[must_use]
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Get failure count
    #[must_use]
    pub fn failures(&self) -> u64 {
        self.failures
    }

    /// Check if circuit is open
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.state == CircuitState::Open
    }

    /// Check if circuit is closed
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.state == CircuitState::Closed
    }

    /// Force reset to closed state
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failures = 0;
        self.successes = 0;
    }
}

// =============================================================================
// EXPONENTIAL BACKOFF (trueno-viz O(1) retry timing pattern)
// =============================================================================

/// Exponential backoff calculator (trueno-viz pattern)
///
/// Calculates retry delays with exponential growth and optional jitter.
/// Useful for retry logic in network operations.
#[derive(Debug, Clone)]
pub struct ExponentialBackoff {
    /// Base delay (microseconds)
    base_us: u64,
    /// Maximum delay (microseconds)
    max_us: u64,
    /// Current attempt
    attempt: u64,
    /// Multiplier for each attempt
    multiplier: f64,
    /// Add jitter (randomness)
    jitter: bool,
}

impl Default for ExponentialBackoff {
    fn default() -> Self {
        Self::new(100_000, 30_000_000) // 100ms base, 30s max
    }
}

impl ExponentialBackoff {
    /// Create a new exponential backoff
    ///
    /// # Arguments
    /// * `base_us` - Base delay in microseconds
    /// * `max_us` - Maximum delay in microseconds
    #[must_use]
    pub fn new(base_us: u64, max_us: u64) -> Self {
        Self {
            base_us,
            max_us,
            attempt: 0,
            multiplier: 2.0,
            jitter: false,
        }
    }

    /// Create with jitter enabled
    #[must_use]
    pub fn with_jitter(mut self) -> Self {
        self.jitter = true;
        self
    }

    /// Create with custom multiplier
    #[must_use]
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.multiplier = multiplier.max(1.0);
        self
    }

    /// Create for network retries (100ms base, 30s max, with jitter)
    #[must_use]
    pub fn for_network() -> Self {
        Self::new(100_000, 30_000_000).with_jitter()
    }

    /// Create for fast retries (10ms base, 1s max)
    #[must_use]
    pub fn for_fast() -> Self {
        Self::new(10_000, 1_000_000)
    }

    /// Get next delay and increment attempt (O(1))
    pub fn next_delay(&mut self) -> u64 {
        let delay = self.current_delay();
        self.attempt += 1;
        delay
    }

    /// Get current delay without incrementing (O(1))
    #[must_use]
    pub fn current_delay(&self) -> u64 {
        let delay = (self.base_us as f64 * self.multiplier.powi(self.attempt as i32)) as u64;
        let capped = delay.min(self.max_us);

        if self.jitter {
            // Simple deterministic jitter based on attempt
            let jitter_factor = 0.5 + (self.attempt % 10) as f64 * 0.05;
            ((capped as f64) * jitter_factor) as u64
        } else {
            capped
        }
    }

    /// Get current delay in milliseconds
    #[must_use]
    pub fn current_delay_ms(&self) -> u64 {
        self.current_delay() / 1000
    }

    /// Get attempt count
    #[must_use]
    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    /// Check if at max delay
    #[must_use]
    pub fn is_at_max(&self) -> bool {
        self.current_delay() >= self.max_us
    }

    /// Reset to first attempt
    pub fn reset(&mut self) {
        self.attempt = 0;
    }
}

// =============================================================================
// SLIDING MEDIAN (trueno-viz O(1) approximate median pattern)
// =============================================================================

/// Approximate sliding median using histogram buckets (trueno-viz pattern)
///
/// Uses fixed-size histogram for O(1) median approximation.
/// Good for latency percentiles where exact values aren't critical.
#[derive(Debug, Clone)]
pub struct SlidingMedian {
    /// Histogram buckets
    buckets: [u64; 10],
    /// Bucket boundaries (upper bounds)
    boundaries: [f64; 10],
    /// Total count
    count: u64,
    /// Min value seen
    min: f64,
    /// Max value seen
    max: f64,
}

impl Default for SlidingMedian {
    fn default() -> Self {
        Self::new()
    }
}

impl SlidingMedian {
    /// Create with default boundaries (0-1000 linear)
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: [0; 10],
            boundaries: [
                100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
            ],
            count: 0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Create for latency (0-100ms, exponential)
    #[must_use]
    pub fn for_latency() -> Self {
        Self {
            buckets: [0; 10],
            boundaries: [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
            count: 0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Create for percentage (0-100%)
    #[must_use]
    pub fn for_percentage() -> Self {
        Self {
            buckets: [0; 10],
            boundaries: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            count: 0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Update with new value (O(1))
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Find bucket
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value <= boundary {
                self.buckets[i] += 1;
                return;
            }
        }
        // Above all boundaries, put in last bucket
        self.buckets[9] += 1;
    }

    /// Get approximate median (O(1))
    #[must_use]
    pub fn median(&self) -> f64 {
        self.percentile(50)
    }

    /// Get approximate percentile (O(1))
    #[must_use]
    pub fn percentile(&self, p: u8) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        let target = (self.count as f64 * p as f64 / 100.0) as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                // Return bucket midpoint
                let lower = if i == 0 { 0.0 } else { self.boundaries[i - 1] };
                return (lower + self.boundaries[i]) / 2.0;
            }
        }

        self.boundaries[9]
    }

    /// Get count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get min value
    #[must_use]
    pub fn min(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.min
        }
    }

    /// Get max value
    #[must_use]
    pub fn max(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max
        }
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.buckets = [0; 10];
        self.count = 0;
        self.min = f64::MAX;
        self.max = f64::MIN;
    }
}

// =============================================================================
// HYSTERESIS FILTER (trueno-viz O(1) noise filtering pattern)
// =============================================================================

/// Hysteresis filter for noise reduction (trueno-viz pattern)
///
/// Only changes output when input crosses threshold by dead band amount.
/// Prevents rapid toggling from noisy inputs.
#[derive(Debug, Clone)]
pub struct HysteresisFilter {
    /// Current output value
    output: f64,
    /// Dead band (minimum change to update)
    dead_band: f64,
    /// Sample count
    count: u64,
}

impl Default for HysteresisFilter {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl HysteresisFilter {
    /// Create with specified dead band
    #[must_use]
    pub fn new(dead_band: f64) -> Self {
        Self {
            output: 0.0,
            dead_band: dead_band.abs(),
            count: 0,
        }
    }

    /// Create for percentage values (1% dead band)
    #[must_use]
    pub fn for_percentage() -> Self {
        Self::new(1.0)
    }

    /// Create for latency values (0.5ms dead band)
    #[must_use]
    pub fn for_latency() -> Self {
        Self::new(0.5)
    }

    /// Create for temperature (0.5 degree dead band)
    #[must_use]
    pub fn for_temperature() -> Self {
        Self::new(0.5)
    }

    /// Update with new value (O(1))
    ///
    /// Returns true if output changed.
    pub fn update(&mut self, value: f64) -> bool {
        self.count += 1;

        if self.count == 1 {
            self.output = value;
            return true;
        }

        if (value - self.output).abs() >= self.dead_band {
            self.output = value;
            return true;
        }

        false
    }

    /// Get filtered output (O(1))
    #[must_use]
    pub fn output(&self) -> f64 {
        self.output
    }

    /// Get dead band
    #[must_use]
    pub fn dead_band(&self) -> f64 {
        self.dead_band
    }

    /// Set dead band
    pub fn set_dead_band(&mut self, dead_band: f64) {
        self.dead_band = dead_band.abs();
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.output = 0.0;
        self.count = 0;
    }
}

// =============================================================================
// SPIKE FILTER (trueno-viz O(1) outlier rejection pattern)
// =============================================================================

/// Spike filter for outlier rejection (trueno-viz pattern)
///
/// Rejects values that differ too much from recent average.
/// Good for sensor readings with occasional bad values.
#[derive(Debug, Clone)]
pub struct SpikeFilter {
    /// Running average
    avg: f64,
    /// Spike threshold (max deviation from avg)
    threshold: f64,
    /// Smoothing factor for avg
    alpha: f64,
    /// Spike count
    spikes: u64,
    /// Sample count
    count: u64,
    /// Last accepted value
    last_accepted: f64,
}

impl Default for SpikeFilter {
    fn default() -> Self {
        Self::new(3.0)
    }
}

impl SpikeFilter {
    /// Create with specified threshold (multiples of running avg)
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            avg: 0.0,
            threshold: threshold.abs(),
            alpha: 0.1,
            spikes: 0,
            count: 0,
            last_accepted: 0.0,
        }
    }

    /// Create for percentage values
    #[must_use]
    pub fn for_percentage() -> Self {
        Self::new(50.0) // 50% deviation threshold
    }

    /// Create for latency values
    #[must_use]
    pub fn for_latency() -> Self {
        Self::new(100.0) // 100ms deviation threshold
    }

    /// Update with new value (O(1))
    ///
    /// Returns the filtered value (original if accepted, last accepted if spike).
    pub fn update(&mut self, value: f64) -> f64 {
        self.count += 1;

        if self.count == 1 {
            self.avg = value;
            self.last_accepted = value;
            return value;
        }

        // Check if spike
        let deviation = (value - self.avg).abs();
        if deviation > self.threshold {
            self.spikes += 1;
            return self.last_accepted;
        }

        // Accept and update average
        self.avg = self.alpha * value + (1.0 - self.alpha) * self.avg;
        self.last_accepted = value;
        value
    }

    /// Get running average (O(1))
    #[must_use]
    pub fn average(&self) -> f64 {
        self.avg
    }

    /// Get spike count
    #[must_use]
    pub fn spikes(&self) -> u64 {
        self.spikes
    }

    /// Get spike rate (percentage)
    #[must_use]
    pub fn spike_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            (self.spikes as f64 / self.count as f64) * 100.0
        }
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get last accepted value
    #[must_use]
    pub fn last_accepted(&self) -> f64 {
        self.last_accepted
    }

    /// Reset filter
    pub fn reset(&mut self) {
        self.avg = 0.0;
        self.spikes = 0;
        self.count = 0;
        self.last_accepted = 0.0;
    }
}

// =============================================================================
// GAUGE TRACKER (trueno-viz O(1) current value tracking pattern)
// =============================================================================

/// Gauge tracker for current values (trueno-viz pattern)
///
/// Tracks current value with min/max/avg statistics.
/// Useful for memory, connections, queue depth.
#[derive(Debug, Clone)]
pub struct GaugeTracker {
    /// Current value
    current: f64,
    /// Minimum value
    min: f64,
    /// Maximum value
    max: f64,
    /// Running sum for average
    sum: f64,
    /// Sample count
    count: u64,
}

impl Default for GaugeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl GaugeTracker {
    /// Create a new gauge tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: 0.0,
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
            count: 0,
        }
    }

    /// Set current value (O(1))
    pub fn set(&mut self, value: f64) {
        self.current = value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
        self.count += 1;
    }

    /// Increment current value
    pub fn inc(&mut self) {
        self.set(self.current + 1.0);
    }

    /// Decrement current value
    pub fn dec(&mut self) {
        self.set(self.current - 1.0);
    }

    /// Add to current value
    pub fn add(&mut self, delta: f64) {
        self.set(self.current + delta);
    }

    /// Get current value (O(1))
    #[must_use]
    pub fn current(&self) -> f64 {
        self.current
    }

    /// Get minimum value (O(1))
    #[must_use]
    pub fn min(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.min
        }
    }

    /// Get maximum value (O(1))
    #[must_use]
    pub fn max(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max
        }
    }

    /// Get average value (O(1))
    #[must_use]
    pub fn average(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Get range (max - min) (O(1))
    #[must_use]
    pub fn range(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.max - self.min
        }
    }

    /// Get sample count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.current = 0.0;
        self.min = f64::MAX;
        self.max = f64::MIN;
        self.sum = 0.0;
        self.count = 0;
    }
}

// =============================================================================
// COUNTER PAIR (trueno-viz O(1) success/failure tracking pattern)
// =============================================================================

/// Counter pair for success/failure tracking (trueno-viz pattern)
///
/// Tracks success and failure counts with ratio calculation.
/// Useful for request success rates, error rates.
#[derive(Debug, Clone)]
pub struct CounterPair {
    /// Success count
    successes: u64,
    /// Failure count
    failures: u64,
}

impl Default for CounterPair {
    fn default() -> Self {
        Self::new()
    }
}

impl CounterPair {
    /// Create a new counter pair
    #[must_use]
    pub fn new() -> Self {
        Self {
            successes: 0,
            failures: 0,
        }
    }

    /// Record a success (O(1))
    pub fn success(&mut self) {
        self.successes += 1;
    }

    /// Record a failure (O(1))
    pub fn failure(&mut self) {
        self.failures += 1;
    }

    /// Record multiple successes
    pub fn add_successes(&mut self, count: u64) {
        self.successes += count;
    }

    /// Record multiple failures
    pub fn add_failures(&mut self, count: u64) {
        self.failures += count;
    }

    /// Get success count (O(1))
    #[must_use]
    pub fn successes(&self) -> u64 {
        self.successes
    }

    /// Get failure count (O(1))
    #[must_use]
    pub fn failures(&self) -> u64 {
        self.failures
    }

    /// Get total count (O(1))
    #[must_use]
    pub fn total(&self) -> u64 {
        self.successes + self.failures
    }

    /// Get success rate (percentage) (O(1))
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            100.0
        } else {
            (self.successes as f64 / total as f64) * 100.0
        }
    }

    /// Get failure rate (percentage) (O(1))
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        100.0 - self.success_rate()
    }

    /// Check if healthy (success rate > threshold)
    #[must_use]
    pub fn is_healthy(&self, threshold: f64) -> bool {
        self.success_rate() >= threshold
    }

    /// Reset counters
    pub fn reset(&mut self) {
        self.successes = 0;
        self.failures = 0;
    }
}

// =============================================================================
// HEALTH SCORE (trueno-viz O(1) composite health pattern)
// =============================================================================

/// Health score calculator (trueno-viz pattern)
///
/// Combines multiple metrics into a single health score (0-100).
/// Useful for system health dashboards.
#[derive(Debug, Clone)]
pub struct HealthScore {
    /// Component scores (up to 8)
    scores: [f64; 8],
    /// Component weights
    weights: [f64; 8],
    /// Number of active components
    active: usize,
}

impl Default for HealthScore {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthScore {
    /// Create a new health score calculator
    #[must_use]
    pub fn new() -> Self {
        Self {
            scores: [100.0; 8],
            weights: [1.0; 8],
            active: 0,
        }
    }

    /// Set a component score (O(1))
    ///
    /// Index 0-7, score 0-100.
    pub fn set(&mut self, index: usize, score: f64) {
        if index < 8 {
            self.scores[index] = score.clamp(0.0, 100.0);
            if index >= self.active {
                self.active = index + 1;
            }
        }
    }

    /// Set a component weight (O(1))
    pub fn set_weight(&mut self, index: usize, weight: f64) {
        if index < 8 {
            self.weights[index] = weight.max(0.0);
        }
    }

    /// Get overall health score (O(1))
    #[must_use]
    pub fn score(&self) -> f64 {
        if self.active == 0 {
            return 100.0;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for i in 0..self.active {
            weighted_sum += self.scores[i] * self.weights[i];
            weight_sum += self.weights[i];
        }

        if weight_sum < f64::EPSILON {
            100.0
        } else {
            (weighted_sum / weight_sum).clamp(0.0, 100.0)
        }
    }

    /// Get health status (O(1))
    #[must_use]
    pub fn status(&self) -> HealthStatus {
        let score = self.score();
        if score >= 90.0 {
            HealthStatus::Healthy
        } else if score >= 70.0 {
            HealthStatus::Degraded
        } else if score >= 50.0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        }
    }

    /// Check if healthy (score >= 90)
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.score() >= 90.0
    }

    /// Get minimum component score (O(1))
    #[must_use]
    pub fn min_score(&self) -> f64 {
        if self.active == 0 {
            return 100.0;
        }
        self.scores[..self.active]
            .iter()
            .fold(f64::MAX, |a, &b| a.min(b))
    }

    /// Get number of active components
    #[must_use]
    pub fn active_components(&self) -> usize {
        self.active
    }

    /// Reset all scores to 100
    pub fn reset(&mut self) {
        self.scores = [100.0; 8];
        self.active = 0;
    }
}

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Score >= 90
    Healthy,
    /// Score >= 70
    Degraded,
    /// Score >= 50
    Warning,
    /// Score < 50
    Critical,
}
