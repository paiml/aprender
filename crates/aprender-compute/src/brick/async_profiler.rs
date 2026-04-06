//! Async task profiler for measuring poll efficiency.

use super::exec_graph::ExecutionNode;
use super::profiling::{cached_nanos_or_now, cpu_cycles};

// ============================================================================
// Async Task Profiler (Pattern 3 from actix-web)
// ============================================================================

/// Async task profiler for measuring poll efficiency (Phase 11, E.9.4).
///
/// Tracks how many times a future is polled before completion.
/// High poll counts indicate inefficient async code or spurious wakeups.
///
/// # Example
/// ```rust,ignore
/// let mut profiler = AsyncTaskProfiler::new("inference_request");
///
/// profiler.on_poll_start();
/// // ... poll the future ...
/// profiler.on_poll_end(is_ready);
///
/// println!("Poll efficiency: {:.1}%", profiler.efficiency() * 100.0);
/// ```
#[derive(Debug, Clone)]
pub struct AsyncTaskProfiler {
    /// Task name for identification
    pub name: String,
    /// Number of times poll() was called
    pub poll_count: u64,
    /// Number of times poll() returned Pending
    pub yield_count: u64,
    /// Total time spent in poll() (nanoseconds)
    pub total_poll_ns: u64,
    /// Start time of current poll
    last_poll_start: u64,
    /// CPU cycles at poll start
    last_poll_cycles: u64,
    /// Total CPU cycles in poll()
    pub total_poll_cycles: u64,
}

impl AsyncTaskProfiler {
    /// Create a new async task profiler.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            poll_count: 0,
            yield_count: 0,
            total_poll_ns: 0,
            last_poll_start: 0,
            last_poll_cycles: 0,
            total_poll_cycles: 0,
        }
    }

    /// Call at the start of each poll() invocation.
    #[inline]
    pub fn on_poll_start(&mut self) {
        self.poll_count += 1;
        self.last_poll_start = cached_nanos_or_now();
        self.last_poll_cycles = cpu_cycles();
    }

    /// Call at the end of each poll() invocation.
    ///
    /// # Arguments
    /// - `is_ready`: true if the future returned Poll::Ready
    #[inline]
    pub fn on_poll_end(&mut self, is_ready: bool) {
        let now = cached_nanos_or_now();
        let cycles = cpu_cycles();

        self.total_poll_ns += now.saturating_sub(self.last_poll_start);
        self.total_poll_cycles += cycles.saturating_sub(self.last_poll_cycles);

        if !is_ready {
            self.yield_count += 1;
        }
    }

    /// Poll efficiency ratio (0.0 to 1.0).
    ///
    /// - 1.0 = Perfect (ready on first poll)
    /// - 0.5 = 2 polls required
    /// - Lower = more wakeups/polls needed
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        if self.poll_count == 0 {
            0.0
        } else {
            1.0 / self.poll_count as f64
        }
    }

    /// Average time per poll in microseconds.
    #[must_use]
    pub fn avg_poll_us(&self) -> f64 {
        if self.poll_count == 0 {
            0.0
        } else {
            self.total_poll_ns as f64 / self.poll_count as f64 / 1000.0
        }
    }

    /// Yield ratio (Pending / total polls).
    ///
    /// High yield ratio indicates the task is often not ready when polled.
    #[must_use]
    pub fn yield_ratio(&self) -> f64 {
        if self.poll_count == 0 {
            0.0
        } else {
            self.yield_count as f64 / self.poll_count as f64
        }
    }

    /// Convert to ExecutionNode for graph integration.
    pub fn to_execution_node(&self) -> ExecutionNode {
        ExecutionNode::AsyncTask {
            name: self.name.clone(),
            poll_count: self.poll_count,
            yield_count: self.yield_count,
            total_poll_ns: self.total_poll_ns,
        }
    }
}

impl Default for AsyncTaskProfiler {
    fn default() -> Self {
        Self::new("unnamed")
    }
}
