//! Real-Time Terminal Monitoring and Visualization (ENT-054 through ENT-067)
//!
//! Terminal-based training visualization using trueno-viz exclusively.
//!
//! # Features
//!
//! - `MetricsBuffer`: O(1) ring buffer for streaming metrics (ENT-055)
//! - `Sparkline`: Unicode sparklines for inline metrics (ENT-057)
//! - `ProgressBar`: Progress bar with Kalman-filtered ETA (ENT-058)
//! - `RefreshPolicy`: Adaptive refresh rate control (ENT-060)
//! - `AndonSystem`: Health monitoring with NaN/Inf detection (ENT-066)
//! - `TerminalMonitorCallback`: Unified callback for training loop (ENT-054)
//!
//! # References
//!
//! - Tufte, E. R. (2006). *Beautiful Evidence*. Graphics Press. (Sparklines)
//! - Welch, G., & Bishop, G. (1995). "An Introduction to the Kalman Filter." (ETA)

mod andon;
mod buffer;
mod callback;
mod capability;
mod charts;
mod config;
mod progress;
mod reference;
mod refresh;
mod sparkline;

// Re-export all public types
pub use andon::{Alert, AlertLevel, AndonSystem};
pub use buffer::MetricsBuffer;
pub use callback::TerminalMonitorCallback;
pub use capability::{DashboardLayout, TerminalCapabilities, TerminalMode};
pub use charts::{
    FeatureImportanceChart, GradientFlowHeatmap, LossCurveDisplay, SeriesSummaryTuple,
};
pub use config::MonitorConfig;
pub use progress::{format_duration, KalmanEta, ProgressBar};
pub use reference::ReferenceCurve;
pub use refresh::RefreshPolicy;
pub use sparkline::{sparkline, sparkline_range, SPARK_CHARS};

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// MetricsBuffer length never exceeds capacity
        #[test]
        fn metrics_buffer_bounded(values in prop::collection::vec(-1000.0f32..1000.0, 0..1000)) {
            let mut buf = MetricsBuffer::new(100);
            for v in &values {
                buf.push(*v);
            }
            prop_assert!(buf.len() <= buf.capacity());
        }

        /// MetricsBuffer values are in chronological order
        #[test]
        fn metrics_buffer_order(values in prop::collection::vec(0.0f32..1000.0, 1..100)) {
            let mut buf = MetricsBuffer::new(values.len());
            for v in &values {
                buf.push(*v);
            }
            prop_assert_eq!(buf.values(), values);
        }

        /// Sparkline length matches input (or width if subsampled)
        #[test]
        fn sparkline_length(
            values in prop::collection::vec(-100.0f32..100.0, 1..100),
            width in 1usize..50
        ) {
            let result = sparkline(&values, width);
            let expected_len = values.len().min(width);
            prop_assert_eq!(result.chars().count(), expected_len);
        }

        /// Sparkline chars are valid
        #[test]
        fn sparkline_valid_chars(values in prop::collection::vec(-100.0f32..100.0, 1..100)) {
            let result = sparkline(&values, values.len());
            for c in result.chars() {
                prop_assert!(SPARK_CHARS.contains(&c));
            }
        }

        /// Progress percentage is bounded [0, 100]
        #[test]
        fn progress_bar_bounded(current in 0usize..1000, total in 1usize..1000) {
            let mut bar = ProgressBar::new(total, 20);
            // Use the update method to set current
            for _ in 0..current {
                bar.update(current);
            }
            let pct = bar.percent();
            prop_assert!(pct >= 0.0);
            // Can exceed 100% if current > total
            prop_assert!(pct <= 100.0 || current > total);
        }

        /// Kalman ETA is non-negative
        #[test]
        fn kalman_eta_nonnegative(
            durations in prop::collection::vec(0.001f64..10.0, 1..100),
            remaining in 0usize..1000
        ) {
            let mut kalman = KalmanEta::new();
            for d in durations {
                kalman.update(d);
            }
            prop_assert!(kalman.eta_seconds(remaining) >= 0.0);
        }

        /// Andon doesn't false positive on normal losses
        #[test]
        fn andon_no_false_positive(values in prop::collection::vec(0.0f32..100.0, 1..100)) {
            let mut andon = AndonSystem::new().with_stop_on_critical(false);
            for v in values {
                andon.check_loss(v);
            }
            // Normal losses should not trigger critical
            prop_assert!(!andon.has_critical());
        }
    }
}
