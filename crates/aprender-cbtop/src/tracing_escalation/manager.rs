//! Tracing escalation manager and rate limiting.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::types::{EscalationReason, EscalationThresholds, SyscallBreakdown, TraceResult};

/// Rate limiter for trace storm prevention
#[derive(Debug)]
struct RateLimiter {
    /// Trace count in current interval
    count: u32,
    /// Start of current interval
    interval_start: Instant,
    /// Maximum traces per interval
    max_count: u32,
    /// Interval duration
    interval: Duration,
}

impl RateLimiter {
    fn new(max_count: u32, interval: Duration) -> Self {
        Self { count: 0, interval_start: Instant::now(), max_count, interval }
    }

    fn should_allow(&mut self) -> bool {
        let now = Instant::now();

        // Reset if interval elapsed
        if now.duration_since(self.interval_start) >= self.interval {
            self.count = 0;
            self.interval_start = now;
        }

        if self.count < self.max_count {
            self.count += 1;
            true
        } else {
            false
        }
    }

    fn current_count(&self) -> u32 {
        self.count
    }
}

/// Tracing escalation manager
#[derive(Debug)]
pub struct TracingEscalation {
    /// Thresholds configuration
    thresholds: EscalationThresholds,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// OTLP endpoint (if configured)
    otlp_endpoint: Option<String>,
    /// Trace history for analysis
    trace_history: Vec<TraceResult>,
    /// Maximum history size
    max_history: usize,
}

impl Default for TracingEscalation {
    fn default() -> Self {
        Self::new(EscalationThresholds::default())
    }
}

impl TracingEscalation {
    /// Create a new tracing escalation manager
    pub fn new(thresholds: EscalationThresholds) -> Self {
        let rate_limiter = RateLimiter::new(thresholds.rate_limit, thresholds.rate_interval);
        Self {
            thresholds,
            rate_limiter,
            otlp_endpoint: None,
            trace_history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Set OTLP endpoint
    pub fn with_otlp_endpoint(mut self, endpoint: &str) -> Self {
        self.otlp_endpoint = Some(endpoint.to_string());
        self
    }

    /// Check if tracing should be escalated based on metrics
    pub fn should_trace(&self, cv_percent: f64, efficiency_percent: f64) -> bool {
        cv_percent > self.thresholds.cv_threshold
            || efficiency_percent < self.thresholds.efficiency_threshold
    }

    /// Check if GPU transfer overhead should trigger escalation
    pub fn should_trace_gpu_transfer(&self, transfer_overhead_percent: f64) -> bool {
        transfer_overhead_percent > self.thresholds.gpu_transfer_threshold
    }

    /// Determine escalation reason
    pub fn escalation_reason(
        &self,
        cv_percent: f64,
        efficiency_percent: f64,
    ) -> Option<EscalationReason> {
        let cv_exceeded = cv_percent > self.thresholds.cv_threshold;
        let efficiency_low = efficiency_percent < self.thresholds.efficiency_threshold;

        match (cv_exceeded, efficiency_low) {
            (true, true) => Some(EscalationReason::Both),
            (true, false) => Some(EscalationReason::CvExceeded),
            (false, true) => Some(EscalationReason::EfficiencyLow),
            (false, false) => None,
        }
    }

    /// Check rate limit and record trace if allowed
    pub fn try_trace(
        &mut self,
        brick_name: &str,
        budget_us: u64,
        actual_us: u64,
        reason: EscalationReason,
        breakdown: SyscallBreakdown,
    ) -> Option<TraceResult> {
        if !self.rate_limiter.should_allow() {
            return None;
        }

        let result = TraceResult {
            brick_name: brick_name.to_string(),
            budget_us,
            actual_us,
            reason,
            syscall_breakdown: breakdown,
            timestamp: Instant::now(),
        };

        // Add to history
        if self.trace_history.len() >= self.max_history {
            self.trace_history.remove(0);
        }
        self.trace_history.push(result.clone());

        Some(result)
    }

    /// Get current rate limit count
    pub fn trace_count(&self) -> u32 {
        self.rate_limiter.current_count()
    }

    /// Get trace history
    pub fn history(&self) -> &[TraceResult] {
        &self.trace_history
    }

    /// Get thresholds
    pub fn thresholds(&self) -> &EscalationThresholds {
        &self.thresholds
    }

    /// Update thresholds
    pub fn set_thresholds(&mut self, thresholds: EscalationThresholds) {
        self.thresholds = thresholds;
        self.rate_limiter =
            RateLimiter::new(self.thresholds.rate_limit, self.thresholds.rate_interval);
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.trace_history.clear();
    }
}

/// OTLP span attributes for brick tracing
#[derive(Debug, Clone)]
pub struct OtlpSpanAttributes {
    /// All attributes as key-value pairs
    pub attributes: HashMap<String, String>,
}

impl OtlpSpanAttributes {
    /// Create from trace result
    pub fn from_trace_result(result: &TraceResult) -> Self {
        Self { attributes: result.as_otlp_attributes() }
    }

    /// Add custom attribute
    pub fn with_attribute(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    /// Check if all required attributes are present
    pub fn has_required_attributes(&self) -> bool {
        let required = [
            "brick.name",
            "brick.budget_us",
            "brick.actual_us",
            "brick.efficiency",
            "brick.over_budget",
            "escalation.reason",
            "syscall.overhead_percent",
            "syscall.dominant",
        ];
        required.iter().all(|key| self.attributes.contains_key(*key))
    }
}
