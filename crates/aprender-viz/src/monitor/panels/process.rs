//! Process panel component.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::Widget;

use crate::monitor::collectors::ProcessCollector;

/// Process monitoring panel.
#[derive(Debug)]
pub struct ProcessPanel {
    /// Process collector.
    pub collector: ProcessCollector,
}

impl ProcessPanel {
    /// Creates a new process panel.
    #[must_use]
    pub fn new() -> Self {
        Self { collector: ProcessCollector::new() }
    }
}

impl Default for ProcessPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for &ProcessPanel {
    /// Renders the process panel as a widget.
    ///
    /// Note: This is a stub implementation. Full rendering is done in
    /// the ttop crate's `panels::draw_process()` function which has access
    /// to the full application state for btop-style layout.
    fn render(self, _area: Rect, _buf: &mut Buffer) {
        // Intentionally minimal - see ttop::panels::draw_process() for full rendering
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_panel_new() {
        let _panel = ProcessPanel::new();
    }

    #[test]
    fn test_process_panel_default() {
        let _panel = ProcessPanel::default();
    }

    #[test]
    fn test_process_panel_render() {
        let panel = ProcessPanel::new();
        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        (&panel).render(Rect::new(0, 0, 40, 10), &mut buf);
    }
}
