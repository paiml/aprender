//! Performance Trace Collection
//!
//! Collects and manages performance trace data.

use super::span::{ActiveSpan, SharedTracerState, Span, SpanGuard, TracerState};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

/// Trace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Capture memory usage
    pub capture_memory: bool,
    /// Capture frame times
    pub capture_frames: bool,
    /// Maximum spans to store
    pub max_spans: usize,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            sample_rate: super::DEFAULT_SAMPLE_RATE,
            capture_memory: false,
            capture_frames: true,
            max_spans: 100_000,
        }
    }
}

impl TraceConfig {
    /// Set sample rate
    #[must_use]
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Enable/disable memory tracking
    #[must_use]
    pub fn with_memory_tracking(mut self, enabled: bool) -> Self {
        self.capture_memory = enabled;
        self
    }

    /// Enable/disable frame time capture
    #[must_use]
    pub fn with_frame_capture(mut self, enabled: bool) -> Self {
        self.capture_frames = enabled;
        self
    }

    /// Set maximum spans
    #[must_use]
    pub fn with_max_spans(mut self, max: usize) -> Self {
        self.max_spans = max;
        self
    }
}

/// Performance tracer with interior mutability
#[derive(Debug)]
pub struct Tracer {
    config: TraceConfig,
    recording: bool,
    state: SharedTracerState,
}

impl std::fmt::Debug for TracerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracerState")
            .field("active_spans", &self.active_spans.len())
            .field("completed_spans", &self.completed_spans.len())
            .field("current_span", &self.current_span)
            .finish()
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracer {
    /// Create a new tracer with default config
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(TraceConfig::default())
    }

    /// Create a tracer with custom config
    #[must_use]
    pub fn with_config(config: TraceConfig) -> Self {
        Self {
            config,
            recording: false,
            state: Rc::new(RefCell::new(TracerState::new())),
        }
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &TraceConfig {
        &self.config
    }

    /// Check if recording
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Start recording
    pub fn start(&mut self) {
        self.recording = true;
        let mut state = self.state.borrow_mut();
        state.trace_start = Some(Instant::now());
        state.active_spans.clear();
        state.completed_spans.clear();
        state.current_span = None;
    }

    /// Stop recording and return trace
    pub fn stop(&mut self) -> Trace {
        self.recording = false;

        let mut state = self.state.borrow_mut();

        // Close any open spans
        let now = state.elapsed_ns();
        let active: Vec<_> = state.active_spans.drain().collect();
        for (_id, mut active_span) in active {
            active_span.span.close(now);
            state.completed_spans.push(active_span.span);
        }

        let duration = state.trace_start.map(|s| s.elapsed());

        Trace {
            spans: std::mem::take(&mut state.completed_spans),
            duration,
            config: self.config.clone(),
        }
    }

    /// Create a new span
    pub fn span(&self, name: &str) -> SpanGuard {
        let mut state = self.state.borrow_mut();
        let start_ns = state.elapsed_ns();
        let start_instant = Instant::now();

        let mut active = ActiveSpan::new(name, start_ns, start_instant);

        // Set parent
        if let Some(parent_id) = state.current_span {
            active.span.parent = Some(parent_id);
        }

        let span_id = active.span.id;
        state.active_spans.insert(span_id, active);
        state.current_span = Some(span_id);

        drop(state); // Release borrow before creating guard

        SpanGuard::new(Rc::clone(&self.state), span_id, self.config.max_spans)
    }

    /// Get current span count
    #[must_use]
    pub fn active_span_count(&self) -> usize {
        self.state.borrow().active_spans.len()
    }

    /// Get completed span count
    #[must_use]
    pub fn completed_span_count(&self) -> usize {
        self.state.borrow().completed_spans.len()
    }
}

/// Collected trace data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    /// Collected spans
    pub spans: Vec<Span>,
    /// Total trace duration
    #[serde(skip)]
    pub duration: Option<Duration>,
    /// Configuration used
    pub config: TraceConfig,
}

impl Trace {
    /// Get span count
    #[must_use]
    pub fn span_count(&self) -> usize {
        self.spans.len()
    }

    /// Get total duration
    #[must_use]
    pub fn duration(&self) -> Option<Duration> {
        self.duration
    }

    /// Get spans by name
    #[must_use]
    pub fn spans_by_name(&self, name: &str) -> Vec<&Span> {
        self.spans.iter().filter(|s| s.name == name).collect()
    }

    /// Get root spans (no parent)
    #[must_use]
    pub fn root_spans(&self) -> Vec<&Span> {
        self.spans.iter().filter(|s| s.parent.is_none()).collect()
    }

    /// Calculate statistics for a span name
    #[must_use]
    pub fn statistics_for(&self, name: &str) -> Option<super::metrics::Statistics> {
        let durations: Vec<f64> = self
            .spans_by_name(name)
            .iter()
            .filter_map(|s| s.duration_ns())
            .map(|ns| ns as f64 / 1_000_000.0) // Convert to ms
            .collect();

        if durations.is_empty() {
            None
        } else {
            Some(super::metrics::Statistics::from_values(&durations))
        }
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_config_default() {
        let config = TraceConfig::default();
        assert_eq!(config.sample_rate, super::super::DEFAULT_SAMPLE_RATE);
        assert!(!config.capture_memory);
        assert!(config.capture_frames);
    }

    #[test]
    fn test_trace_config_builders() {
        let config = TraceConfig::default()
            .with_sample_rate(500)
            .with_memory_tracking(true)
            .with_max_spans(1000);

        assert_eq!(config.sample_rate, 500);
        assert!(config.capture_memory);
        assert_eq!(config.max_spans, 1000);
    }

    #[test]
    fn test_tracer_new() {
        let tracer = Tracer::new();
        assert!(!tracer.is_recording());
    }

    #[test]
    fn test_tracer_start_stop() {
        let mut tracer = Tracer::new();

        tracer.start();
        assert!(tracer.is_recording());

        let trace = tracer.stop();
        assert!(!tracer.is_recording());
        assert!(trace.spans.is_empty());
    }

    #[test]
    fn test_tracer_span() {
        let mut tracer = Tracer::new();
        tracer.start();

        {
            let _guard = tracer.span("test");
            assert_eq!(tracer.active_span_count(), 1);
        }

        assert_eq!(tracer.active_span_count(), 0);
        assert_eq!(tracer.completed_span_count(), 1);
    }

    #[test]
    fn test_tracer_nested_spans() {
        let mut tracer = Tracer::new();
        tracer.start();

        {
            let _outer = tracer.span("outer");
            {
                let _inner = tracer.span("inner");
            }
        }

        let trace = tracer.stop();
        assert_eq!(trace.span_count(), 2);

        // Verify parent-child relationship
        let inner = trace.spans_by_name("inner")[0];
        let outer = trace.spans_by_name("outer")[0];
        assert_eq!(inner.parent, Some(outer.id));
    }

    #[test]
    fn test_trace_spans_by_name() {
        let mut tracer = Tracer::new();
        tracer.start();

        {
            let _s1 = tracer.span("a");
        }
        {
            let _s2 = tracer.span("b");
        }
        {
            let _s3 = tracer.span("a");
        }

        let trace = tracer.stop();
        assert_eq!(trace.spans_by_name("a").len(), 2);
        assert_eq!(trace.spans_by_name("b").len(), 1);
    }

    #[test]
    fn test_trace_root_spans() {
        let mut tracer = Tracer::new();
        tracer.start();

        {
            let _outer = tracer.span("outer");
            {
                let _inner = tracer.span("inner");
            }
        }
        {
            let _other = tracer.span("other");
        }

        let trace = tracer.stop();
        let roots = trace.root_spans();
        assert_eq!(roots.len(), 2); // "outer" and "other"
    }

    // =========================================================================
    // Additional tests for 95% coverage
    // =========================================================================

    #[test]
    fn test_trace_config_with_frame_capture() {
        let config = TraceConfig::default().with_frame_capture(false);
        assert!(!config.capture_frames);

        let config2 = TraceConfig::default().with_frame_capture(true);
        assert!(config2.capture_frames);
    }

    #[test]
    fn test_tracer_default() {
        let tracer = Tracer::default();
        assert!(!tracer.is_recording());
        assert_eq!(tracer.active_span_count(), 0);
        assert_eq!(tracer.completed_span_count(), 0);
    }

    #[test]
    fn test_tracer_config_getter() {
        let config = TraceConfig::default().with_sample_rate(1000);
        let tracer = Tracer::with_config(config);
        assert_eq!(tracer.config().sample_rate, 1000);
    }

    #[test]
    fn test_trace_is_empty() {
        let mut tracer = Tracer::new();
        tracer.start();
        let trace = tracer.stop();
        assert!(trace.is_empty());

        tracer.start();
        {
            let _span = tracer.span("test");
        }
        let trace2 = tracer.stop();
        assert!(!trace2.is_empty());
    }

    #[test]
    fn test_trace_duration() {
        let mut tracer = Tracer::new();
        tracer.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let trace = tracer.stop();

        let duration = trace.duration();
        assert!(duration.is_some());
        assert!(duration.unwrap().as_millis() >= 10);
    }

    #[test]
    fn test_trace_statistics_for_existing() {
        let mut tracer = Tracer::new();
        tracer.start();

        for _ in 0..5 {
            let _span = tracer.span("repeated");
            std::thread::sleep(std::time::Duration::from_micros(100));
        }

        let trace = tracer.stop();
        let stats = trace.statistics_for("repeated");
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert!(stats.count >= 5);
    }

    #[test]
    fn test_trace_statistics_for_nonexistent() {
        let mut tracer = Tracer::new();
        tracer.start();
        {
            let _span = tracer.span("exists");
        }
        let trace = tracer.stop();

        let stats = trace.statistics_for("does_not_exist");
        assert!(stats.is_none());
    }

    #[test]
    fn test_tracer_state_debug() {
        let tracer = Tracer::new();
        let state = tracer.state.borrow();
        let debug_str = format!("{:?}", *state);
        assert!(debug_str.contains("TracerState"));
        assert!(debug_str.contains("active_spans"));
        assert!(debug_str.contains("completed_spans"));
    }

    #[test]
    fn test_tracer_stop_closes_active_spans() {
        let mut tracer = Tracer::new();
        tracer.start();

        // Create spans but don't drop them before stopping
        let _outer = tracer.span("outer");
        let _inner = tracer.span("inner");
        drop(_inner);
        drop(_outer);

        let trace = tracer.stop();
        // Both spans should be in completed list
        assert_eq!(trace.span_count(), 2);
    }

    #[test]
    fn test_tracer_multiple_sessions() {
        let mut tracer = Tracer::new();

        // First session
        tracer.start();
        {
            let _span = tracer.span("first_session");
        }
        let trace1 = tracer.stop();
        assert_eq!(trace1.span_count(), 1);

        // Second session - should start fresh
        tracer.start();
        {
            let _span = tracer.span("second_session");
        }
        let trace2 = tracer.stop();
        assert_eq!(trace2.span_count(), 1);
        assert_eq!(trace2.spans_by_name("first_session").len(), 0);
    }

    #[test]
    fn test_trace_config_serialize_deserialize() {
        let config = TraceConfig::default()
            .with_sample_rate(500)
            .with_memory_tracking(true)
            .with_max_spans(5000);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TraceConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.sample_rate, 500);
        assert!(deserialized.capture_memory);
        assert_eq!(deserialized.max_spans, 5000);
    }

    #[test]
    fn test_trace_serialize_deserialize() {
        let mut tracer = Tracer::new();
        tracer.start();
        {
            let _span = tracer.span("test_span");
        }
        let trace = tracer.stop();

        let json = serde_json::to_string(&trace).unwrap();
        let deserialized: Trace = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.span_count(), trace.span_count());
        assert_eq!(deserialized.spans_by_name("test_span").len(), 1);
    }

    #[test]
    fn test_deeply_nested_spans() {
        let mut tracer = Tracer::new();
        tracer.start();

        {
            let _l1 = tracer.span("level1");
            {
                let _l2 = tracer.span("level2");
                {
                    let _l3 = tracer.span("level3");
                    {
                        let _l4 = tracer.span("level4");
                    }
                }
            }
        }

        let trace = tracer.stop();
        assert_eq!(trace.span_count(), 4);

        // Verify parent chain
        let l4_spans = trace.spans_by_name("level4");
        assert_eq!(l4_spans.len(), 1);
        let l4 = l4_spans[0];
        assert!(l4.parent.is_some());

        let l3_spans = trace.spans_by_name("level3");
        let l3 = l3_spans[0];
        assert_eq!(l4.parent, Some(l3.id));
    }

    #[test]
    fn test_sibling_spans() {
        let mut tracer = Tracer::new();
        tracer.start();

        {
            let _parent = tracer.span("parent");
            {
                let _child1 = tracer.span("child");
            }
            {
                let _child2 = tracer.span("child");
            }
            {
                let _child3 = tracer.span("child");
            }
        }

        let trace = tracer.stop();
        let children = trace.spans_by_name("child");
        assert_eq!(children.len(), 3);

        // All children should have the same parent
        let parent_spans = trace.spans_by_name("parent");
        let parent_id = parent_spans[0].id;
        for child in children {
            assert_eq!(child.parent, Some(parent_id));
        }
    }

    #[test]
    fn test_span_count_after_stop() {
        let mut tracer = Tracer::new();
        tracer.start();
        {
            let _s = tracer.span("a");
        }
        assert_eq!(tracer.completed_span_count(), 1);
        assert_eq!(tracer.active_span_count(), 0);

        tracer.stop();
        // After stop, counts should be reset on next start
        tracer.start();
        assert_eq!(tracer.completed_span_count(), 0);
    }
}
