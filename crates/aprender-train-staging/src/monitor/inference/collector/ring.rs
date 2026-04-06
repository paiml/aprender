//! RingCollector - Vec-based ring buffer for real-time

use super::super::path::DecisionPath;
use super::super::trace::DecisionTrace;
use super::traits::TraceCollector;

/// Ring buffer collector with fixed capacity
///
/// Target: <100ns per trace
///
/// # Features
/// - O(1) push operation
/// - Overwrites oldest entries when full
/// - No unsafe code
///
/// # Example
///
/// ```ignore
/// use entrenar::monitor::inference::{RingCollector, LinearPath};
///
/// let mut collector = RingCollector::<LinearPath, 64>::new();
/// collector.record(trace);
/// let recent = collector.recent(10);
/// ```
pub struct RingCollector<P: DecisionPath, const N: usize> {
    buffer: Vec<DecisionTrace<P>>,
    head: usize,
}

impl<P: DecisionPath, const N: usize> RingCollector<P, N> {
    /// Create a new ring collector
    pub fn new() -> Self {
        Self { buffer: Vec::with_capacity(N), head: 0 }
    }

    /// Get the most recent n traces (or all if n > count)
    pub fn recent(&self, n: usize) -> Vec<&DecisionTrace<P>> {
        let take = n.min(self.buffer.len());
        let mut result = Vec::with_capacity(take);

        for i in 0..take {
            let idx = if self.buffer.len() < N {
                // Not yet wrapped
                self.buffer.len() - 1 - i
            } else {
                // Wrapped: head points to next write, so head-1 is most recent
                (self.head + N - 1 - i) % N
            };
            result.push(&self.buffer[idx]);
        }

        result
    }

    /// Get all traces in order (oldest first)
    pub fn all(&self) -> Vec<&DecisionTrace<P>> {
        let mut result = Vec::with_capacity(self.buffer.len());

        if self.buffer.is_empty() {
            return result;
        }

        if self.buffer.len() < N {
            // Not yet wrapped - just iterate in order
            for trace in &self.buffer {
                result.push(trace);
            }
        } else {
            // Wrapped: head is the oldest
            for i in 0..N {
                let idx = (self.head + i) % N;
                result.push(&self.buffer[idx]);
            }
        }

        result
    }

    /// Get the last trace if any
    pub fn last(&self) -> Option<&DecisionTrace<P>> {
        if self.buffer.is_empty() {
            return None;
        }
        if self.buffer.len() < N {
            self.buffer.last()
        } else {
            let idx = (self.head + N - 1) % N;
            Some(&self.buffer[idx])
        }
    }

    /// Clear all traces
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.head = 0;
    }

    /// Capacity of the ring buffer
    pub const fn capacity(&self) -> usize {
        N
    }
}

impl<P: DecisionPath, const N: usize> TraceCollector<P> for RingCollector<P, N> {
    fn record(&mut self, trace: DecisionTrace<P>) {
        if self.buffer.len() < N {
            // Buffer not yet full, just push
            self.buffer.push(trace);
        } else {
            // Buffer full, overwrite oldest
            self.buffer[self.head] = trace;
            self.head = (self.head + 1) % N;
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Ring buffer doesn't need flushing
        Ok(())
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl<P: DecisionPath, const N: usize> Default for RingCollector<P, N> {
    fn default() -> Self {
        Self::new()
    }
}
