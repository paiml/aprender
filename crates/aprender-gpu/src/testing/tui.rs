//! TUI Monitoring Mode for Stress Testing
//!
//! Real-time terminal UI for monitoring stress test progress.
//! Uses presentar/crossterm (via simular) for rendering.
//!
//! # Feature Flag
//!
//! Requires `tui-monitor` feature:
//! ```toml
//! trueno-gpu = { version = "0.1", features = ["tui-monitor"] }
//! ```

use super::stress::{PerformanceResult, StressReport};

/// TUI configuration
#[derive(Debug, Clone)]
pub struct TuiConfig {
    /// Refresh rate in milliseconds
    pub refresh_rate_ms: u64,
    /// Show frame time sparkline
    pub show_frame_times: bool,
    /// Show memory usage
    pub show_memory_usage: bool,
    /// Show anomaly alerts
    pub show_anomaly_alerts: bool,
    /// Title for the TUI window
    pub title: String,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            refresh_rate_ms: 100,
            show_frame_times: true,
            show_memory_usage: true,
            show_anomaly_alerts: true,
            title: "trueno-gpu Stress Test Monitor".to_string(),
        }
    }
}

/// TUI state for rendering
#[derive(Debug, Clone, Default)]
pub struct TuiState {
    /// Current cycle
    pub current_cycle: u32,
    /// Total cycles to run
    pub total_cycles: u32,
    /// Current FPS
    pub current_fps: f64,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Recent frame times (for sparkline)
    pub frame_times: Vec<u64>,
    /// Test results per test name
    pub test_results: Vec<(String, u64, bool)>, // (name, duration_ms, passed)
    /// Number of anomalies
    pub anomaly_count: usize,
    /// Number of regressions
    pub regression_count: usize,
    /// Pass rate (0.0 to 1.0)
    pub pass_rate: f64,
    /// Is running
    pub running: bool,
    /// Is paused
    pub paused: bool,
}

impl TuiState {
    /// Create new TUI state
    #[must_use]
    pub fn new(total_cycles: u32) -> Self {
        Self { total_cycles, running: true, ..Default::default() }
    }

    /// Update state from stress report
    pub fn update_from_report(&mut self, report: &StressReport) {
        self.current_cycle = report.cycles_completed;
        self.anomaly_count = report.anomalies.len();
        self.pass_rate = report.pass_rate();

        // Update frame times (keep last 50)
        self.frame_times = report.frames.iter().rev().take(50).map(|f| f.duration_ms).collect();
        self.frame_times.reverse();

        // Calculate FPS from mean frame time
        let mean_ms = report.mean_frame_time_ms();
        self.current_fps = if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 };

        // Memory from last frame
        if let Some(last) = report.frames.last() {
            self.memory_bytes = last.memory_bytes;
        }
    }

    /// Format memory as human-readable string
    #[must_use]
    pub fn format_memory(&self) -> String {
        let bytes = self.memory_bytes as f64;
        if bytes < 1024.0 {
            format!("{:.0} B", bytes)
        } else if bytes < 1024.0 * 1024.0 {
            format!("{:.1} KB", bytes / 1024.0)
        } else {
            format!("{:.1} MB", bytes / (1024.0 * 1024.0))
        }
    }

    /// Generate sparkline data (normalized 0-7 for block characters)
    #[must_use]
    pub fn sparkline_data(&self) -> Vec<u8> {
        if self.frame_times.is_empty() {
            return vec![];
        }

        let max = *self.frame_times.iter().max().unwrap_or(&1) as f64;
        let min = *self.frame_times.iter().min().unwrap_or(&0) as f64;
        let range = (max - min).max(1.0);

        self.frame_times
            .iter()
            .map(|&t| {
                let normalized = (t as f64 - min) / range;
                // normalized is in [0, 1], so result is in [0, 7]
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let level = (normalized * 7.0).round() as u8;
                level
            })
            .collect()
    }

    /// Generate sparkline string using Unicode block characters
    #[must_use]
    pub fn sparkline_string(&self) -> String {
        const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        self.sparkline_data().iter().map(|&v| BLOCKS[v.min(7) as usize]).collect()
    }
}

/// Render TUI state to string (for non-interactive output)
#[must_use]
pub fn render_to_string(
    state: &TuiState,
    report: &StressReport,
    perf: &PerformanceResult,
) -> String {
    let mut output = String::new();

    // Header
    output.push_str("╔══════════════════════════════════════════════════════════════╗\n");
    output.push_str("║  trueno-gpu Stress Test Monitor (simular TUI)                ║\n");
    output.push_str("╠══════════════════════════════════════════════════════════════╣\n");

    // Status line
    output.push_str(&format!(
        "║  Cycle: {}/{}    FPS: {:.1}    Memory: {:<10}       ║\n",
        state.current_cycle,
        state.total_cycles,
        state.current_fps,
        state.format_memory()
    ));
    output.push_str("║                                                               ║\n");

    // Sparkline
    let sparkline = state.sparkline_string();
    if !sparkline.is_empty() {
        output.push_str(&format!("║  Frame Times (ms):  {:<40} ║\n", sparkline));
    }

    // Stats
    output.push_str(&format!(
        "║  Mean: {:.0}ms  Max: {}ms  Variance: {:.2}                     ║\n",
        perf.mean_frame_ms, perf.max_frame_ms, perf.variance
    ));
    output.push_str("║                                                               ║\n");

    // Test results
    output.push_str("║  Test Results:                                                ║\n");
    let passed = report.total_passed;
    let failed = report.total_failed;
    output.push_str(&format!(
        "║  ✓ Passed: {:<6}  ✗ Failed: {:<6}                         ║\n",
        passed, failed
    ));
    output.push_str("║                                                               ║\n");

    // Summary
    let status = if perf.passed { "PASS" } else { "FAIL" };
    output.push_str(&format!(
        "║  Anomalies: {}    Regressions: {}    Status: {:<4}            ║\n",
        state.anomaly_count, state.regression_count, status
    ));

    // Footer
    output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
    output.push_str("║  [q] Quit  [p] Pause  [r] Reset  [s] Save Report             ║\n");
    output.push_str("╚══════════════════════════════════════════════════════════════╝\n");

    output
}

/// Simple ASCII progress bar
#[must_use]
pub fn progress_bar(current: u32, total: u32, width: usize) -> String {
    if total == 0 {
        return format!("[{}]", " ".repeat(width));
    }

    let progress = (current as f64 / total as f64).min(1.0);
    // progress is in [0, 1], so filled is in [0, width]
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let filled = (progress * width as f64).round() as usize;
    let empty = width - filled;

    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

#[cfg(feature = "tui-monitor")]
pub mod interactive {
    //! Interactive TUI using presentar/crossterm
    //!
    //! Only available with `tui-monitor` feature.

    #[allow(clippy::wildcard_imports)]
    use super::*;

    /// Run interactive TUI (requires tui-monitor feature)
    ///
    /// # Errors
    ///
    /// Returns error if terminal initialization fails
    pub fn run_interactive(
        _config: TuiConfig,
        _state: &mut TuiState,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Full implementation requires presentar/crossterm
        // This is a placeholder for the feature-gated implementation
        Err("Interactive TUI requires tui-monitor feature with presentar".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tui_config_default() {
        let config = TuiConfig::default();
        assert_eq!(config.refresh_rate_ms, 100);
        assert!(config.show_frame_times);
        assert!(config.show_memory_usage);
        assert!(config.show_anomaly_alerts);
    }

    #[test]
    fn test_tui_state_new() {
        let state = TuiState::new(100);
        assert_eq!(state.total_cycles, 100);
        assert!(state.running);
        assert!(!state.paused);
    }

    #[test]
    fn test_format_memory() {
        let mut state = TuiState::default();

        state.memory_bytes = 512;
        assert_eq!(state.format_memory(), "512 B");

        state.memory_bytes = 2048;
        assert_eq!(state.format_memory(), "2.0 KB");

        state.memory_bytes = 5 * 1024 * 1024;
        assert_eq!(state.format_memory(), "5.0 MB");
    }

    #[test]
    fn test_sparkline_data() {
        let mut state = TuiState::default();
        state.frame_times = vec![10, 20, 30, 40, 50];

        let data = state.sparkline_data();
        assert_eq!(data.len(), 5);
        assert_eq!(data[0], 0); // min
        assert_eq!(data[4], 7); // max
    }

    #[test]
    fn test_sparkline_string() {
        let mut state = TuiState::default();
        state.frame_times = vec![10, 20, 30, 40, 50];

        let sparkline = state.sparkline_string();
        assert_eq!(sparkline.chars().count(), 5);
        assert!(sparkline.starts_with('▁'));
        assert!(sparkline.ends_with('█'));
    }

    #[test]
    fn test_progress_bar() {
        assert_eq!(progress_bar(0, 100, 10), "[░░░░░░░░░░]");
        assert_eq!(progress_bar(50, 100, 10), "[█████░░░░░]");
        assert_eq!(progress_bar(100, 100, 10), "[██████████]");
        assert_eq!(progress_bar(0, 0, 10), "[          ]"); // Edge case
    }

    #[test]
    fn test_render_to_string() {
        let state = TuiState::new(100);
        let report = StressReport::default();
        let perf = PerformanceResult {
            passed: true,
            max_frame_ms: 50,
            mean_frame_ms: 40.0,
            variance: 0.1,
            pass_rate: 1.0,
            violations: vec![],
        };

        let output = render_to_string(&state, &report, &perf);
        assert!(output.contains("trueno-gpu Stress Test Monitor"));
        assert!(output.contains("Cycle: 0/100"));
        assert!(output.contains("PASS"));
    }

    #[test]
    fn test_update_from_report() {
        use super::super::stress::{FrameProfile, StressReport};

        let mut state = TuiState::new(10);
        let mut report = StressReport::default();

        for i in 0..5 {
            report.add_frame(FrameProfile {
                cycle: i,
                duration_ms: 50 + i as u64 * 10,
                memory_bytes: 1024,
                tests_passed: 5,
                tests_failed: 0,
                ..Default::default()
            });
        }

        state.update_from_report(&report);

        assert_eq!(state.current_cycle, 5);
        assert_eq!(state.frame_times.len(), 5);
        assert!(state.current_fps > 0.0);
        assert!((state.pass_rate - 1.0).abs() < 0.001);
    }
}
