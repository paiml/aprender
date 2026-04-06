//! Trace collector trait definition

use super::super::path::DecisionPath;
use super::super::trace::DecisionTrace;

/// Strategy for collecting decision traces
pub trait TraceCollector<P: DecisionPath>: Send + Sync {
    /// Record a decision trace
    fn record(&mut self, trace: DecisionTrace<P>);

    /// Flush any buffered traces
    fn flush(&mut self) -> std::io::Result<()>;

    /// Number of traces recorded
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
