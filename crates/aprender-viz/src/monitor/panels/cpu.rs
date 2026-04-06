//! CPU panel component.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::Widget;

use crate::monitor::collectors::CpuCollector;

/// CPU monitoring panel.
#[derive(Debug)]
pub struct CpuPanel {
    /// CPU collector.
    pub collector: CpuCollector,
}

impl CpuPanel {
    /// Creates a new CPU panel.
    #[must_use]
    pub fn new() -> Self {
        Self { collector: CpuCollector::new() }
    }
}

impl Default for CpuPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for &CpuPanel {
    /// Renders the CPU panel as a widget.
    ///
    /// Note: This is a stub implementation. Full rendering is done in
    /// the ttop crate's `panels::draw_cpu()` function which has access
    /// to the full application state for btop-style layout.
    fn render(self, _area: Rect, _buf: &mut Buffer) {
        // Intentionally minimal - see ttop::panels::draw_cpu() for full rendering
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_panel_new() {
        let panel = CpuPanel::new();
        assert!(panel.collector.core_count() >= 1);
    }

    #[test]
    fn test_cpu_panel_default() {
        let panel = CpuPanel::default();
        assert!(panel.collector.core_count() >= 1);
    }

    #[test]
    fn test_cpu_panel_render() {
        let panel = CpuPanel::new();
        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        (&panel).render(Rect::new(0, 0, 40, 10), &mut buf);
        // Stub render - just ensure no panic
    }
}
