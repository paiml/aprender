//! State management for the TUI monitor.

use crate::monitor::ring_buffer::RingBuffer;
use crate::monitor::types::Metrics;
use std::collections::HashMap;

/// Shared state for the monitoring application.
#[derive(Debug)]
pub struct State {
    /// Metrics history per collector.
    pub history: HashMap<String, RingBuffer<Metrics>>,
    /// Whether the application should quit.
    pub should_quit: bool,
    /// Currently selected panel index.
    pub selected_panel: usize,
    /// Whether help is visible.
    pub show_help: bool,
}

impl State {
    /// Creates a new state with default values.
    ///
    /// Note: `history_size` is stored for documentation but histories are
    /// created lazily with their own size when `record()` is called.
    #[must_use]
    pub fn new(_history_size: usize) -> Self {
        Self { history: HashMap::new(), should_quit: false, selected_panel: 0, show_help: false }
    }

    /// Records metrics from a collector.
    pub fn record(&mut self, collector_id: &str, metrics: Metrics, history_size: usize) {
        self.history
            .entry(collector_id.to_string())
            .or_insert_with(|| RingBuffer::new(history_size))
            .push(metrics);
    }

    /// Gets the latest metrics for a collector.
    #[must_use]
    pub fn latest(&self, collector_id: &str) -> Option<&Metrics> {
        self.history.get(collector_id).and_then(|h| h.latest())
    }

    /// Signals that the application should quit.
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// Toggles help visibility.
    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    /// Selects the next panel.
    pub fn next_panel(&mut self, panel_count: usize) {
        if panel_count > 0 {
            self.selected_panel = (self.selected_panel + 1) % panel_count;
        }
    }

    /// Selects the previous panel.
    pub fn prev_panel(&mut self, panel_count: usize) {
        if panel_count > 0 {
            self.selected_panel = self.selected_panel.checked_sub(1).unwrap_or(panel_count - 1);
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new(300)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_new() {
        let state = State::new(100);
        assert!(!state.should_quit);
        assert_eq!(state.selected_panel, 0);
    }

    #[test]
    fn test_state_quit() {
        let mut state = State::new(100);
        state.quit();
        assert!(state.should_quit);
    }

    #[test]
    fn test_state_record_and_latest() {
        let mut state = State::new(100);
        let metrics = Metrics::new();

        state.record("cpu", metrics.clone(), 100);

        let latest = state.latest("cpu");
        assert!(latest.is_some());
    }

    #[test]
    fn test_state_panel_navigation() {
        let mut state = State::new(100);

        state.next_panel(3);
        assert_eq!(state.selected_panel, 1);

        state.next_panel(3);
        assert_eq!(state.selected_panel, 2);

        state.next_panel(3);
        assert_eq!(state.selected_panel, 0); // Wraps around

        state.prev_panel(3);
        assert_eq!(state.selected_panel, 2);
    }
}
