//! Memory panel component.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::Widget;

use crate::monitor::collectors::MemoryCollector;

/// Memory monitoring panel.
#[derive(Debug)]
pub struct MemoryPanel {
    /// Memory collector.
    pub collector: MemoryCollector,
}

impl MemoryPanel {
    /// Creates a new memory panel.
    #[must_use]
    pub fn new() -> Self {
        Self { collector: MemoryCollector::new() }
    }
}

impl Default for MemoryPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for &MemoryPanel {
    /// Renders the memory panel as a widget.
    ///
    /// Note: This is a stub implementation. Full rendering is done in
    /// the ttop crate's `panels::draw_memory()` function which has access
    /// to the full application state for btop-style layout.
    fn render(self, _area: Rect, _buf: &mut Buffer) {
        // Intentionally minimal - see ttop::panels::draw_memory() for full rendering
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_panel_new() {
        let _panel = MemoryPanel::new();
    }

    #[test]
    fn test_memory_panel_default() {
        let _panel = MemoryPanel::default();
    }

    #[test]
    fn test_memory_panel_render() {
        let panel = MemoryPanel::new();
        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        (&panel).render(Rect::new(0, 0, 40, 10), &mut buf);
    }
}
