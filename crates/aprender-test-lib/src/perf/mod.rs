//! Performance Benchmarking with Renacer Integration (Advanced Feature C)
//!
//! Unified performance tracing for WASM and TUI applications with
//! Chrome Trace export, flame graph generation, and CI metrics.

#![allow(clippy::redundant_pub_crate)]

mod export;
mod metrics;
mod span;
mod trace;

pub use export::{ChromeTrace, CiMetrics, FlameGraph};
pub use metrics::{FrameMetrics, MemoryMetrics, PerformanceMetrics, Statistics};
pub use span::{Span, SpanGuard, SpanId};
pub use trace::{Trace, TraceConfig, Tracer};

/// Default sample rate (Hz)
pub const DEFAULT_SAMPLE_RATE: u32 = 1000;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-PERF-01: Tracer creation
    // =========================================================================

    #[test]
    fn h0_perf_01_tracer_creation() {
        let tracer = Tracer::new();
        assert!(!tracer.is_recording());
    }

    #[test]
    fn h0_perf_02_tracer_with_config() {
        let config = TraceConfig::default()
            .with_sample_rate(500)
            .with_memory_tracking(true);

        let tracer = Tracer::with_config(config);
        assert_eq!(tracer.config().sample_rate, 500);
        assert!(tracer.config().capture_memory);
    }

    // =========================================================================
    // H₀-PERF-03: Span recording
    // =========================================================================

    #[test]
    fn h0_perf_03_span_creation() {
        let mut tracer = Tracer::new();
        tracer.start();

        {
            let _guard = tracer.span("test_span");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let trace = tracer.stop();
        assert!(trace.span_count() > 0);
    }

    #[test]
    fn h0_perf_04_nested_spans() {
        let mut tracer = Tracer::new();
        tracer.start();

        {
            let _outer = tracer.span("outer");
            {
                let _inner = tracer.span("inner");
            }
        }

        let trace = tracer.stop();
        assert!(trace.span_count() >= 2);
    }

    // =========================================================================
    // H₀-PERF-05: Performance metrics
    // =========================================================================

    #[test]
    fn h0_perf_05_statistics() {
        let stats = Statistics::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert!((stats.mean - 3.0).abs() < f64::EPSILON);
        assert!((stats.min - 1.0).abs() < f64::EPSILON);
        assert!((stats.max - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn h0_perf_06_frame_metrics() {
        let metrics = FrameMetrics::new(16.67); // ~60 FPS
        assert!((metrics.frame_time_ms - 16.67).abs() < 0.01);
        assert!(metrics.fps() > 59.0 && metrics.fps() < 61.0);
    }

    // =========================================================================
    // H₀-PERF-07: Export formats
    // =========================================================================

    #[test]
    fn h0_perf_07_chrome_trace_export() {
        let mut tracer = Tracer::new();
        tracer.start();
        {
            let _guard = tracer.span("test");
        }
        let trace = tracer.stop();

        let chrome_trace = ChromeTrace::from_trace(&trace);
        let json = chrome_trace.to_json();

        assert!(json.contains("traceEvents"));
    }

    #[test]
    fn h0_perf_08_ci_metrics_export() {
        let mut tracer = Tracer::new();
        tracer.start();
        {
            let _guard = tracer.span("test");
        }
        let trace = tracer.stop();

        let ci_metrics = CiMetrics::from_trace(&trace);
        let json = ci_metrics.to_json();

        assert!(json.contains("span_count"));
    }

    // =========================================================================
    // H₀-PERF-09: Memory tracking
    // =========================================================================

    #[test]
    fn h0_perf_09_memory_metrics() {
        let metrics = MemoryMetrics {
            heap_used: 1024 * 1024,
            heap_total: 2 * 1024 * 1024,
            peak_usage: 1536 * 1024,
        };

        assert_eq!(metrics.usage_percent(), 50.0);
    }
}
