//! Performance Tracing Spans
//!
//! Hierarchical span-based performance measurement.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static NEXT_SPAN_ID: AtomicU64 = AtomicU64::new(1);

/// Unique span identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanId(u64);

impl SpanId {
    /// Create a new unique span ID
    #[must_use]
    pub fn new() -> Self {
        Self(NEXT_SPAN_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

/// A performance measurement span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Unique identifier
    pub id: SpanId,
    /// Span name
    pub name: String,
    /// Start timestamp (nanoseconds from trace start)
    pub start_ns: u64,
    /// End timestamp (nanoseconds from trace start)
    pub end_ns: Option<u64>,
    /// Parent span ID
    pub parent: Option<SpanId>,
    /// Span category
    pub category: Option<String>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl Span {
    /// Create a new span
    #[must_use]
    pub fn new(name: impl Into<String>, start_ns: u64) -> Self {
        Self {
            id: SpanId::new(),
            name: name.into(),
            start_ns,
            end_ns: None,
            parent: None,
            category: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Create span with parent
    #[must_use]
    pub fn with_parent(mut self, parent: SpanId) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Set category
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Close the span
    pub fn close(&mut self, end_ns: u64) {
        self.end_ns = Some(end_ns);
    }

    /// Get duration in nanoseconds
    #[must_use]
    pub fn duration_ns(&self) -> Option<u64> {
        self.end_ns.map(|end| end.saturating_sub(self.start_ns))
    }

    /// Get duration as Duration
    #[must_use]
    pub fn duration(&self) -> Option<Duration> {
        self.duration_ns().map(Duration::from_nanos)
    }

    /// Check if span is closed
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.end_ns.is_some()
    }
}

/// Internal tracer state for RefCell-based interior mutability
pub(crate) struct TracerState {
    pub active_spans: std::collections::HashMap<SpanId, ActiveSpan>,
    pub completed_spans: Vec<Span>,
    pub current_span: Option<SpanId>,
    pub trace_start: Option<Instant>,
}

impl TracerState {
    pub fn new() -> Self {
        Self {
            active_spans: std::collections::HashMap::new(),
            completed_spans: Vec::new(),
            current_span: None,
            trace_start: None,
        }
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.trace_start
            .map(|start| start.elapsed().as_nanos() as u64)
            .unwrap_or(0)
    }
}

/// Shared reference to tracer state
pub(crate) type SharedTracerState = Rc<RefCell<TracerState>>;

/// RAII guard for automatic span closure
pub struct SpanGuard {
    state: SharedTracerState,
    span_id: SpanId,
    max_spans: usize,
}

impl std::fmt::Debug for SpanGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpanGuard")
            .field("span_id", &self.span_id)
            .field("max_spans", &self.max_spans)
            .finish_non_exhaustive()
    }
}

impl SpanGuard {
    /// Create a new span guard
    pub(crate) fn new(state: SharedTracerState, span_id: SpanId, max_spans: usize) -> Self {
        Self {
            state,
            span_id,
            max_spans,
        }
    }

    /// Get the span ID
    #[must_use]
    pub fn id(&self) -> SpanId {
        self.span_id
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        if let Some(mut active) = state.active_spans.remove(&self.span_id) {
            let end_ns = state.elapsed_ns();
            active.span.close(end_ns);

            // Update current span to parent
            state.current_span = active.span.parent;

            // Store completed span
            if state.completed_spans.len() < self.max_spans {
                state.completed_spans.push(active.span);
            }
        }
    }
}

/// Internal span with timing information
#[derive(Debug)]
pub(crate) struct ActiveSpan {
    pub span: Span,
    #[allow(dead_code)]
    pub start_instant: Instant,
}

impl ActiveSpan {
    pub fn new(name: impl Into<String>, start_ns: u64, start_instant: Instant) -> Self {
        Self {
            span: Span::new(name, start_ns),
            start_instant,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_span_id_unique() {
        let id1 = SpanId::new();
        let id2 = SpanId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_span_id_as_u64() {
        let id = SpanId::new();
        assert!(id.as_u64() > 0);
    }

    #[test]
    fn test_span_id_default() {
        let id1 = SpanId::default();
        let id2 = SpanId::default();
        // Default should call new(), producing unique IDs
        assert_ne!(id1, id2);
        assert!(id1.as_u64() > 0);
    }

    #[test]
    fn test_span_id_equality() {
        let id = SpanId::new();
        let id_copy = id;
        assert_eq!(id, id_copy);
    }

    #[test]
    fn test_span_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let id1 = SpanId::new();
        let id2 = SpanId::new();
        set.insert(id1);
        set.insert(id2);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&id1));
        assert!(set.contains(&id2));
    }

    #[test]
    fn test_span_new() {
        let span = Span::new("test", 1000);
        assert_eq!(span.name, "test");
        assert_eq!(span.start_ns, 1000);
        assert!(!span.is_closed());
    }

    #[test]
    fn test_span_with_parent() {
        let parent_id = SpanId::new();
        let span = Span::new("child", 2000).with_parent(parent_id);
        assert_eq!(span.parent, Some(parent_id));
    }

    #[test]
    fn test_span_with_category() {
        let span = Span::new("test", 0).with_category("render");
        assert_eq!(span.category, Some("render".to_string()));
    }

    #[test]
    fn test_span_close() {
        let mut span = Span::new("test", 1000);
        span.close(2000);
        assert!(span.is_closed());
        assert_eq!(span.duration_ns(), Some(1000));
    }

    #[test]
    fn test_span_duration() {
        let mut span = Span::new("test", 0);
        span.close(1_000_000); // 1ms in ns
        let duration = span.duration().unwrap();
        assert_eq!(duration, Duration::from_nanos(1_000_000));
    }

    #[test]
    fn test_span_metadata() {
        let mut span = Span::new("test", 0);
        span.add_metadata("key", "value");
        assert_eq!(span.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_span_duration_ns_unclosed() {
        let span = Span::new("test", 1000);
        // Unclosed span should return None for duration
        assert_eq!(span.duration_ns(), None);
    }

    #[test]
    fn test_span_duration_unclosed() {
        let span = Span::new("test", 1000);
        // Unclosed span should return None for duration
        assert!(span.duration().is_none());
    }

    #[test]
    fn test_span_duration_saturating_sub() {
        // Edge case: end_ns < start_ns (shouldn't happen normally, but we handle it)
        let mut span = Span::new("test", 5000);
        span.close(1000); // End before start
                          // saturating_sub should return 0 instead of underflowing
        assert_eq!(span.duration_ns(), Some(0));
        assert_eq!(span.duration(), Some(Duration::from_nanos(0)));
    }

    #[test]
    fn test_span_multiple_metadata() {
        let mut span = Span::new("test", 0);
        span.add_metadata("key1", "value1");
        span.add_metadata("key2", "value2");
        span.add_metadata("key1", "updated"); // Overwrite existing key
        assert_eq!(span.metadata.get("key1"), Some(&"updated".to_string()));
        assert_eq!(span.metadata.get("key2"), Some(&"value2".to_string()));
        assert_eq!(span.metadata.len(), 2);
    }

    #[test]
    fn test_span_is_closed_initially_false() {
        let span = Span::new("test", 0);
        assert!(!span.is_closed());
        assert!(span.end_ns.is_none());
    }

    // TracerState tests
    #[test]
    fn test_tracer_state_new() {
        let state = TracerState::new();
        assert!(state.active_spans.is_empty());
        assert!(state.completed_spans.is_empty());
        assert!(state.current_span.is_none());
        assert!(state.trace_start.is_none());
    }

    #[test]
    fn test_tracer_state_elapsed_ns_no_start() {
        let state = TracerState::new();
        // When trace_start is None, elapsed_ns returns 0
        assert_eq!(state.elapsed_ns(), 0);
    }

    #[test]
    fn test_tracer_state_elapsed_ns_with_start() {
        let mut state = TracerState::new();
        state.trace_start = Some(Instant::now());
        // Sleep briefly to ensure elapsed time > 0
        std::thread::sleep(Duration::from_micros(100));
        let elapsed = state.elapsed_ns();
        assert!(elapsed > 0);
    }

    // ActiveSpan tests
    #[test]
    fn test_active_span_new() {
        let instant = Instant::now();
        let active = ActiveSpan::new("test_span", 12345, instant);
        assert_eq!(active.span.name, "test_span");
        assert_eq!(active.span.start_ns, 12345);
        assert!(!active.span.is_closed());
    }

    #[test]
    fn test_active_span_debug() {
        let instant = Instant::now();
        let active = ActiveSpan::new("debug_test", 0, instant);
        let debug_str = format!("{:?}", active);
        assert!(debug_str.contains("ActiveSpan"));
        assert!(debug_str.contains("span"));
    }

    // SpanGuard tests
    #[test]
    fn test_span_guard_new_and_id() {
        let state = Rc::new(RefCell::new(TracerState::new()));
        let span_id = SpanId::new();
        let guard = SpanGuard::new(Rc::clone(&state), span_id, 100);
        assert_eq!(guard.id(), span_id);
    }

    #[test]
    fn test_span_guard_debug() {
        let state = Rc::new(RefCell::new(TracerState::new()));
        let span_id = SpanId::new();
        let guard = SpanGuard::new(Rc::clone(&state), span_id, 100);
        let debug_str = format!("{:?}", guard);
        assert!(debug_str.contains("SpanGuard"));
        assert!(debug_str.contains("span_id"));
        assert!(debug_str.contains("max_spans"));
    }

    #[test]
    fn test_span_guard_drop_closes_span() {
        let state = Rc::new(RefCell::new(TracerState::new()));
        {
            let mut s = state.borrow_mut();
            s.trace_start = Some(Instant::now());
        }

        let span_id = SpanId::new();
        let active_span = ActiveSpan::new("test", 0, Instant::now());

        {
            let mut s = state.borrow_mut();
            s.active_spans.insert(span_id, active_span);
            s.current_span = Some(span_id);
        }

        // Create guard and let it drop
        {
            let _guard = SpanGuard::new(Rc::clone(&state), span_id, 100);
        } // guard drops here

        let s = state.borrow();
        // Span should be removed from active_spans
        assert!(!s.active_spans.contains_key(&span_id));
        // Span should be in completed_spans
        assert_eq!(s.completed_spans.len(), 1);
        assert!(s.completed_spans[0].is_closed());
        // current_span should be updated to parent (None in this case)
        assert!(s.current_span.is_none());
    }

    #[test]
    fn test_span_guard_drop_respects_max_spans() {
        let state = Rc::new(RefCell::new(TracerState::new()));
        {
            let mut s = state.borrow_mut();
            s.trace_start = Some(Instant::now());
            // Pre-fill completed_spans to max capacity
            for i in 0..5 {
                let mut span = Span::new(format!("existing_{}", i), i as u64 * 100);
                span.close(i as u64 * 100 + 50);
                s.completed_spans.push(span);
            }
        }

        let span_id = SpanId::new();
        let active_span = ActiveSpan::new("overflow_test", 500, Instant::now());

        {
            let mut s = state.borrow_mut();
            s.active_spans.insert(span_id, active_span);
        }

        // Create guard with max_spans = 5 (already at capacity)
        {
            let _guard = SpanGuard::new(Rc::clone(&state), span_id, 5);
        } // guard drops here

        let s = state.borrow();
        // Span should be removed from active_spans
        assert!(!s.active_spans.contains_key(&span_id));
        // completed_spans should NOT grow beyond max (5 items)
        assert_eq!(s.completed_spans.len(), 5);
    }

    #[test]
    fn test_span_guard_drop_updates_parent() {
        let state = Rc::new(RefCell::new(TracerState::new()));
        {
            let mut s = state.borrow_mut();
            s.trace_start = Some(Instant::now());
        }

        let parent_id = SpanId::new();
        let child_id = SpanId::new();

        let mut child_span = ActiveSpan::new("child", 100, Instant::now());
        child_span.span.parent = Some(parent_id);

        {
            let mut s = state.borrow_mut();
            s.active_spans.insert(child_id, child_span);
            s.current_span = Some(child_id);
        }

        // Drop the child guard
        {
            let _guard = SpanGuard::new(Rc::clone(&state), child_id, 100);
        }

        let s = state.borrow();
        // current_span should now be the parent
        assert_eq!(s.current_span, Some(parent_id));
    }

    #[test]
    fn test_span_guard_drop_missing_span() {
        // Test dropping a guard when the span has already been removed
        let state = Rc::new(RefCell::new(TracerState::new()));
        {
            let mut s = state.borrow_mut();
            s.trace_start = Some(Instant::now());
        }

        let span_id = SpanId::new();
        // Note: we don't add the span to active_spans

        // Create and drop guard for non-existent span
        {
            let _guard = SpanGuard::new(Rc::clone(&state), span_id, 100);
        } // guard drops here - should not panic

        let s = state.borrow();
        // Nothing should have been added
        assert!(s.completed_spans.is_empty());
    }

    // Serialization tests
    #[test]
    fn test_span_id_serde() {
        let id = SpanId::new();
        let serialized = serde_json::to_string(&id).unwrap();
        let deserialized: SpanId = serde_json::from_str(&serialized).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn test_span_serde() {
        let mut span = Span::new("serde_test", 1000);
        span.close(2000);
        span.category = Some("test_category".to_string());
        span.add_metadata("key", "value");

        let serialized = serde_json::to_string(&span).unwrap();
        let deserialized: Span = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.name, "serde_test");
        assert_eq!(deserialized.start_ns, 1000);
        assert_eq!(deserialized.end_ns, Some(2000));
        assert_eq!(deserialized.category, Some("test_category".to_string()));
        assert_eq!(deserialized.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_span_serde_empty_metadata() {
        let span = Span::new("minimal", 500);
        let serialized = serde_json::to_string(&span).unwrap();
        let deserialized: Span = serde_json::from_str(&serialized).unwrap();
        assert!(deserialized.metadata.is_empty());
    }

    #[test]
    fn test_span_clone() {
        let mut original = Span::new("original", 100);
        original.close(200);
        original.category = Some("cat".to_string());
        original.add_metadata("k", "v");

        let cloned = original.clone();
        assert_eq!(cloned.name, original.name);
        assert_eq!(cloned.start_ns, original.start_ns);
        assert_eq!(cloned.end_ns, original.end_ns);
        assert_eq!(cloned.category, original.category);
        assert_eq!(cloned.metadata, original.metadata);
        // IDs should be the same (clone, not new)
        assert_eq!(cloned.id, original.id);
    }

    #[test]
    fn test_span_id_clone_copy() {
        let id = SpanId::new();
        let cloned = id;
        let copied = id;
        assert_eq!(id, cloned);
        assert_eq!(id, copied);
    }

    #[test]
    fn test_span_zero_duration() {
        let mut span = Span::new("instant", 1000);
        span.close(1000); // Same start and end
        assert_eq!(span.duration_ns(), Some(0));
        assert_eq!(span.duration(), Some(Duration::ZERO));
    }

    #[test]
    fn test_span_large_timestamps() {
        let start = u64::MAX - 1000;
        let end = u64::MAX;
        let mut span = Span::new("large", start);
        span.close(end);
        assert_eq!(span.duration_ns(), Some(1000));
    }

    #[test]
    fn test_tracer_state_active_spans_operations() {
        let mut state = TracerState::new();
        let id1 = SpanId::new();
        let id2 = SpanId::new();

        let span1 = ActiveSpan::new("span1", 0, Instant::now());
        let span2 = ActiveSpan::new("span2", 100, Instant::now());

        state.active_spans.insert(id1, span1);
        state.active_spans.insert(id2, span2);

        assert_eq!(state.active_spans.len(), 2);
        assert!(state.active_spans.contains_key(&id1));
        assert!(state.active_spans.contains_key(&id2));

        state.active_spans.remove(&id1);
        assert_eq!(state.active_spans.len(), 1);
        assert!(!state.active_spans.contains_key(&id1));
    }
}
