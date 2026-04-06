//! Performance Trace Export Formats
//!
//! Export traces to Chrome Trace JSON, Flame Graphs, and CI metrics.

use super::trace::Trace;
use serde::{Deserialize, Serialize};

/// Chrome Trace format for chrome://tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeTrace {
    /// Trace events
    #[serde(rename = "traceEvents")]
    pub trace_events: Vec<ChromeTraceEvent>,
}

/// Single event in Chrome Trace format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeTraceEvent {
    /// Event name
    pub name: String,
    /// Category
    pub cat: String,
    /// Phase (B=begin, E=end, X=complete)
    pub ph: String,
    /// Timestamp in microseconds
    pub ts: u64,
    /// Duration in microseconds (for X events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<u64>,
    /// Process ID
    pub pid: u32,
    /// Thread ID
    pub tid: u32,
    /// Arguments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

impl ChromeTrace {
    /// Create Chrome Trace from a trace
    #[must_use]
    pub fn from_trace(trace: &Trace) -> Self {
        let mut events = Vec::new();

        for span in &trace.spans {
            if let Some(dur_ns) = span.duration_ns() {
                events.push(ChromeTraceEvent {
                    name: span.name.clone(),
                    cat: span
                        .category
                        .clone()
                        .unwrap_or_else(|| "default".to_string()),
                    ph: "X".to_string(),      // Complete event
                    ts: span.start_ns / 1000, // Convert to microseconds
                    dur: Some(dur_ns / 1000),
                    pid: 1,
                    tid: 1,
                    args: if span.metadata.is_empty() {
                        None
                    } else {
                        Some(serde_json::json!(span.metadata))
                    },
                });
            }
        }

        Self {
            trace_events: events,
        }
    }

    /// Export to JSON string
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export to compact JSON
    #[must_use]
    pub fn to_json_compact(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Flame graph data structure
#[derive(Debug, Clone)]
pub struct FlameGraph {
    /// Stacks with counts
    stacks: Vec<FlameStack>,
}

/// A stack in the flame graph
#[derive(Debug, Clone)]
pub struct FlameStack {
    /// Stack frames (from root to leaf)
    pub frames: Vec<String>,
    /// Total time in this stack (ms)
    pub value: f64,
}

impl FlameGraph {
    /// Create flame graph from trace
    #[must_use]
    pub fn from_trace(trace: &Trace) -> Self {
        // Build stack traces from spans
        let mut stacks = Vec::new();

        // Simple approach: each span is a stack
        for span in &trace.spans {
            if let Some(dur_ns) = span.duration_ns() {
                let mut frames = Vec::new();

                // Build frame stack by following parents
                frames.push(span.name.clone());

                // Find parent names
                if let Some(parent_id) = span.parent {
                    if let Some(parent) = trace.spans.iter().find(|s| s.id == parent_id) {
                        frames.insert(0, parent.name.clone());
                    }
                }

                stacks.push(FlameStack {
                    frames,
                    value: dur_ns as f64 / 1_000_000.0,
                });
            }
        }

        Self { stacks }
    }

    /// Export to collapsed stack format (for FlameGraph tools)
    #[must_use]
    pub fn to_collapsed(&self) -> String {
        let mut output = String::new();

        for stack in &self.stacks {
            let stack_str = stack.frames.join(";");
            output.push_str(&format!("{} {}\n", stack_str, stack.value as u64));
        }

        output
    }

    /// Generate simple SVG flame graph
    #[must_use]
    pub fn to_svg(&self, width: u32, height: u32) -> String {
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            width, height, width, height
        );

        svg.push_str(
            r#"
  <style>
    .frame { stroke: #333; stroke-width: 0.5; }
    .frame:hover { stroke: #000; stroke-width: 1; }
    text { font-family: monospace; font-size: 10px; fill: #333; }
  </style>
"#,
        );

        // Simple rendering: each span as a rectangle
        let total_value: f64 = self.stacks.iter().map(|s| s.value).sum();
        if total_value > 0.0 {
            let mut y = 0.0;
            let bar_height = 20.0;

            for stack in &self.stacks {
                let w = (stack.value / total_value) * width as f64;
                if w > 1.0 {
                    let color = random_color(stack.frames.last().unwrap_or(&String::new()));
                    svg.push_str(&format!(
                        r#"  <rect class="frame" x="0" y="{}" width="{}" height="{}" fill="{}"><title>{}: {:.2}ms</title></rect>"#,
                        y, w, bar_height, color,
                        stack.frames.join(" → "), stack.value
                    ));
                    svg.push('\n');
                    y += bar_height;
                }
            }
        }

        svg.push_str("</svg>");
        svg
    }
}

/// Generate a deterministic color from a string
fn random_color(s: &str) -> String {
    let hash: u32 = s
        .bytes()
        .fold(0, |acc, b| acc.wrapping_add(b as u32).wrapping_mul(31));
    let hue = hash % 360;
    format!("hsl({}, 70%, 60%)", hue)
}

/// CI-friendly metrics export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiMetrics {
    /// Number of spans recorded
    pub span_count: usize,
    /// Total trace duration in ms
    pub duration_ms: f64,
    /// Function timing summaries
    pub functions: Vec<FunctionMetric>,
    /// Pass/fail status
    pub passed: bool,
    /// Failure reasons
    pub failures: Vec<String>,
}

/// Metric for a single function/span type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetric {
    /// Function/span name
    pub name: String,
    /// Call count
    pub count: usize,
    /// Mean duration (ms)
    pub mean_ms: f64,
    /// P99 duration (ms)
    pub p99_ms: f64,
    /// Total time (ms)
    pub total_ms: f64,
}

impl CiMetrics {
    /// Create CI metrics from trace
    #[must_use]
    pub fn from_trace(trace: &Trace) -> Self {
        let perf = super::metrics::PerformanceMetrics::from_trace(trace);

        let functions: Vec<FunctionMetric> = perf
            .function_times
            .iter()
            .map(|(name, stats)| FunctionMetric {
                name: name.clone(),
                count: stats.count,
                mean_ms: stats.mean,
                p99_ms: stats.p99,
                total_ms: stats.mean * stats.count as f64,
            })
            .collect();

        Self {
            span_count: trace.span_count(),
            duration_ms: trace
                .duration
                .map(|d| d.as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
            functions,
            passed: true,
            failures: Vec::new(),
        }
    }

    /// Check against thresholds
    #[must_use]
    pub fn check_thresholds(&self, max_p99_ms: f64) -> Self {
        let mut result = self.clone();
        result.failures.clear();
        result.passed = true;

        for func in &self.functions {
            if func.p99_ms > max_p99_ms {
                result.failures.push(format!(
                    "{}: p99 {:.2}ms exceeds threshold {:.2}ms",
                    func.name, func.p99_ms, max_p99_ms
                ));
                result.passed = false;
            }
        }

        result
    }

    /// Export to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::perf::trace::Tracer;

    fn create_test_trace() -> Trace {
        let mut tracer = Tracer::new();
        tracer.start();

        {
            let _outer = tracer.span("render");
            std::thread::sleep(std::time::Duration::from_micros(100));
            {
                let _inner = tracer.span("draw");
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }

        tracer.stop()
    }

    #[test]
    fn test_chrome_trace_from_trace() {
        let trace = create_test_trace();
        let chrome = ChromeTrace::from_trace(&trace);

        assert_eq!(chrome.trace_events.len(), 2);
    }

    #[test]
    fn test_chrome_trace_to_json() {
        let trace = create_test_trace();
        let chrome = ChromeTrace::from_trace(&trace);
        let json = chrome.to_json();

        assert!(json.contains("traceEvents"));
        assert!(json.contains("render"));
        assert!(json.contains("draw"));
    }

    #[test]
    fn test_chrome_trace_event_fields() {
        let trace = create_test_trace();
        let chrome = ChromeTrace::from_trace(&trace);

        let event = &chrome.trace_events[0];
        assert!(!event.name.is_empty());
        assert_eq!(event.ph, "X");
        assert!(event.dur.is_some());
    }

    #[test]
    fn test_flame_graph_from_trace() {
        let trace = create_test_trace();
        let flame = FlameGraph::from_trace(&trace);

        assert!(!flame.stacks.is_empty());
    }

    #[test]
    fn test_flame_graph_to_collapsed() {
        let trace = create_test_trace();
        let flame = FlameGraph::from_trace(&trace);
        let collapsed = flame.to_collapsed();

        assert!(!collapsed.is_empty());
        assert!(collapsed.contains("render") || collapsed.contains("draw"));
    }

    #[test]
    fn test_flame_graph_to_svg() {
        let trace = create_test_trace();
        let flame = FlameGraph::from_trace(&trace);
        let svg = flame.to_svg(800, 400);

        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("rect"));
    }

    #[test]
    fn test_ci_metrics_from_trace() {
        let trace = create_test_trace();
        let metrics = CiMetrics::from_trace(&trace);

        assert_eq!(metrics.span_count, 2);
        assert!(metrics.passed);
    }

    #[test]
    fn test_ci_metrics_to_json() {
        let trace = create_test_trace();
        let metrics = CiMetrics::from_trace(&trace);
        let json = metrics.to_json();

        assert!(json.contains("span_count"));
        assert!(json.contains("functions"));
    }

    #[test]
    fn test_ci_metrics_check_thresholds() {
        let trace = create_test_trace();
        let metrics = CiMetrics::from_trace(&trace);

        // Very tight threshold may or may not fail depending on timing
        let _checked = metrics.check_thresholds(0.001);

        // Very loose threshold should pass
        let checked = metrics.check_thresholds(10000.0);
        assert!(checked.passed);
    }

    #[test]
    fn test_random_color() {
        let c1 = random_color("test");
        let c2 = random_color("test");
        let c3 = random_color("other");

        // Same input = same output
        assert_eq!(c1, c2);
        // Different input = different output (usually)
        assert_ne!(c1, c3);
        // Valid HSL format
        assert!(c1.starts_with("hsl("));
    }
}
