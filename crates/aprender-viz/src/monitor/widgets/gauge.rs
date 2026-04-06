//! Gauge widget for displaying percentage values.
//!
//! Supports arc/circular display modes for compact metric visualization.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

/// Display mode for the gauge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GaugeMode {
    /// Full circle (360°).
    #[default]
    Full,
    /// Half circle (180°).
    Half,
    /// Quarter circle (90°).
    Quarter,
    /// Compact single-line mode.
    Compact,
}

/// A gauge widget for displaying percentage values.
#[derive(Debug, Clone)]
pub struct Gauge<'a> {
    /// Value to display (0.0 - 1.0).
    value: f64,
    /// Display mode.
    mode: GaugeMode,
    /// Label text.
    label: &'a str,
    /// Fill color (or gradient start).
    fill_color: Color,
    /// Background color.
    bg_color: Color,
    /// Threshold for warning color (0.0 - 1.0).
    warn_threshold: f64,
    /// Warning color.
    warn_color: Color,
    /// Threshold for critical color (0.0 - 1.0).
    crit_threshold: f64,
    /// Critical color.
    crit_color: Color,
    /// Show percentage text.
    show_percent: bool,
}

impl<'a> Gauge<'a> {
    /// Creates a new gauge with the given value.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            mode: GaugeMode::default(),
            label: "",
            fill_color: Color::Cyan,
            bg_color: Color::DarkGray,
            warn_threshold: 0.7,
            warn_color: Color::Yellow,
            crit_threshold: 0.9,
            crit_color: Color::Red,
            show_percent: true,
        }
    }

    /// Sets the display mode.
    #[must_use]
    pub fn mode(mut self, mode: GaugeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the label.
    #[must_use]
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = label;
        self
    }

    /// Sets the fill color.
    #[must_use]
    pub fn fill_color(mut self, color: Color) -> Self {
        self.fill_color = color;
        self
    }

    /// Sets the background color.
    #[must_use]
    pub fn bg_color(mut self, color: Color) -> Self {
        self.bg_color = color;
        self
    }

    /// Sets the warning threshold and color.
    #[must_use]
    pub fn warn(mut self, threshold: f64, color: Color) -> Self {
        self.warn_threshold = threshold.clamp(0.0, 1.0);
        self.warn_color = color;
        self
    }

    /// Sets the critical threshold and color.
    #[must_use]
    pub fn critical(mut self, threshold: f64, color: Color) -> Self {
        self.crit_threshold = threshold.clamp(0.0, 1.0);
        self.crit_color = color;
        self
    }

    /// Sets whether to show percentage text.
    #[must_use]
    pub fn show_percent(mut self, show: bool) -> Self {
        self.show_percent = show;
        self
    }

    /// Returns the color based on current value and thresholds.
    fn current_color(&self) -> Color {
        if self.value >= self.crit_threshold {
            self.crit_color
        } else if self.value >= self.warn_threshold {
            self.warn_color
        } else {
            self.fill_color
        }
    }

    /// Renders a compact single-line gauge.
    fn render_compact(&self, area: Rect, buf: &mut Buffer) {
        if area.width < 5 || area.height < 1 {
            return;
        }

        let color = self.current_color();

        // Format: [████░░░░] 75%
        let bar_width = if self.show_percent {
            area.width.saturating_sub(6) // Reserve space for " XX%"
        } else {
            area.width.saturating_sub(2) // Just brackets
        };

        let filled = ((self.value * f64::from(bar_width)).round() as u16).min(bar_width);

        // Opening bracket
        buf.set_string(area.x, area.y, "[", Style::default().fg(Color::White));

        // Filled portion
        for i in 0..filled {
            buf.set_string(area.x + 1 + i, area.y, "█", Style::default().fg(color));
        }

        // Unfilled portion
        for i in filled..bar_width {
            buf.set_string(area.x + 1 + i, area.y, "░", Style::default().fg(self.bg_color));
        }

        // Closing bracket
        buf.set_string(area.x + 1 + bar_width, area.y, "]", Style::default().fg(Color::White));

        // Percentage
        if self.show_percent && area.width > bar_width + 4 {
            let percent = format!("{:3.0}%", self.value * 100.0);
            buf.set_string(area.x + 2 + bar_width, area.y, percent, Style::default().fg(color));
        }
    }

    /// Renders a half-circle gauge using Unicode box drawing.
    fn render_half(&self, area: Rect, buf: &mut Buffer) {
        if area.width < 7 || area.height < 3 {
            self.render_compact(area, buf);
            return;
        }

        let color = self.current_color();
        let center_x = area.x + area.width / 2;
        let y = area.y;

        // Draw arc: ╭───────╮
        //           ████████
        //           Label XX%

        // Top arc
        let arc_width = area.width.min(15);
        let arc_start = center_x.saturating_sub(arc_width / 2);

        buf.set_string(arc_start, y, "╭", Style::default().fg(Color::White));
        for i in 1..arc_width - 1 {
            buf.set_string(arc_start + i, y, "─", Style::default().fg(Color::White));
        }
        buf.set_string(arc_start + arc_width - 1, y, "╮", Style::default().fg(Color::White));

        // Fill bar
        let bar_width = arc_width.saturating_sub(2);
        let filled = ((self.value * f64::from(bar_width)).round() as u16).min(bar_width);

        for i in 0..bar_width {
            let char = if i < filled { "█" } else { "░" };
            let char_color = if i < filled { color } else { self.bg_color };
            buf.set_string(arc_start + 1 + i, y + 1, char, Style::default().fg(char_color));
        }

        // Label and percentage
        if area.height >= 3 {
            let text = if self.show_percent {
                if self.label.is_empty() {
                    format!("{:3.0}%", self.value * 100.0)
                } else {
                    format!("{} {:3.0}%", self.label, self.value * 100.0)
                }
            } else {
                self.label.to_string()
            };

            let text_x = center_x.saturating_sub(text.len() as u16 / 2);
            buf.set_string(text_x, y + 2, text, Style::default().fg(color));
        }
    }

    /// Renders a full circle gauge (simplified as box with fill).
    fn render_full(&self, area: Rect, buf: &mut Buffer) {
        if area.width < 9 || area.height < 5 {
            self.render_half(area, buf);
            return;
        }

        let color = self.current_color();
        let center_x = area.x + area.width / 2;
        let center_y = area.y + area.height / 2;

        // Draw a box representing the gauge
        // ╭───────╮
        // │  XX%  │
        // │ ████  │
        // │ Label │
        // ╰───────╯

        let box_width = area.width.min(11);
        let box_height = area.height.min(5);
        let box_x = center_x.saturating_sub(box_width / 2);
        let box_y = center_y.saturating_sub(box_height / 2);

        // Top border
        buf.set_string(box_x, box_y, "╭", Style::default().fg(Color::White));
        for i in 1..box_width - 1 {
            buf.set_string(box_x + i, box_y, "─", Style::default().fg(Color::White));
        }
        buf.set_string(box_x + box_width - 1, box_y, "╮", Style::default().fg(Color::White));

        // Side borders and content
        for row in 1..box_height - 1 {
            buf.set_string(box_x, box_y + row, "│", Style::default().fg(Color::White));
            buf.set_string(
                box_x + box_width - 1,
                box_y + row,
                "│",
                Style::default().fg(Color::White),
            );
        }

        // Bottom border
        buf.set_string(box_x, box_y + box_height - 1, "╰", Style::default().fg(Color::White));
        for i in 1..box_width - 1 {
            buf.set_string(
                box_x + i,
                box_y + box_height - 1,
                "─",
                Style::default().fg(Color::White),
            );
        }
        buf.set_string(
            box_x + box_width - 1,
            box_y + box_height - 1,
            "╯",
            Style::default().fg(Color::White),
        );

        // Percentage in center
        if self.show_percent && box_height >= 3 {
            let percent = format!("{:3.0}%", self.value * 100.0);
            let text_x = center_x.saturating_sub(percent.len() as u16 / 2);
            buf.set_string(text_x, box_y + 1, percent, Style::default().fg(color));
        }

        // Fill bar
        if box_height >= 4 {
            let bar_width = box_width.saturating_sub(4);
            let filled = ((self.value * f64::from(bar_width)).round() as u16).min(bar_width);
            let bar_x = box_x + 2;
            let bar_y = box_y + 2;

            for i in 0..bar_width {
                let char = if i < filled { "█" } else { "░" };
                let char_color = if i < filled { color } else { self.bg_color };
                buf.set_string(bar_x + i, bar_y, char, Style::default().fg(char_color));
            }
        }

        // Label
        if !self.label.is_empty() && box_height >= 5 {
            let label_x = center_x.saturating_sub(self.label.len() as u16 / 2);
            buf.set_string(
                label_x.max(box_x + 1),
                box_y + box_height - 2,
                self.label,
                Style::default().fg(Color::White),
            );
        }
    }
}

impl Widget for Gauge<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        match self.mode {
            GaugeMode::Compact => self.render_compact(area, buf),
            GaugeMode::Quarter | GaugeMode::Half => self.render_half(area, buf),
            GaugeMode::Full => self.render_full(area, buf),
        }
    }
}

// ============================================================================
// Tests (TDD - Written First)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn create_test_terminal() -> Terminal<TestBackend> {
        let backend = TestBackend::new(40, 10);
        Terminal::new(backend).expect("Failed to create terminal")
    }

    #[test]
    fn test_gauge_new() {
        let gauge = Gauge::new(0.5);
        assert!((gauge.value - 0.5).abs() < 0.01);
        assert_eq!(gauge.mode, GaugeMode::Full);
        assert!(gauge.show_percent);
    }

    #[test]
    fn test_gauge_value_clamping() {
        let gauge_low = Gauge::new(-0.5);
        assert!((gauge_low.value - 0.0).abs() < 0.01);

        let gauge_high = Gauge::new(1.5);
        assert!((gauge_high.value - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gauge_builder() {
        let gauge = Gauge::new(0.75)
            .mode(GaugeMode::Compact)
            .label("CPU")
            .fill_color(Color::Green)
            .warn(0.8, Color::Yellow)
            .critical(0.95, Color::Red)
            .show_percent(false);

        assert_eq!(gauge.mode, GaugeMode::Compact);
        assert_eq!(gauge.label, "CPU");
        assert_eq!(gauge.fill_color, Color::Green);
        assert!((gauge.warn_threshold - 0.8).abs() < 0.01);
        assert!(!gauge.show_percent);
    }

    #[test]
    fn test_gauge_current_color() {
        let gauge = Gauge::new(0.5)
            .fill_color(Color::Green)
            .warn(0.7, Color::Yellow)
            .critical(0.9, Color::Red);

        assert_eq!(gauge.current_color(), Color::Green);

        let gauge_warn = Gauge::new(0.75)
            .fill_color(Color::Green)
            .warn(0.7, Color::Yellow)
            .critical(0.9, Color::Red);

        assert_eq!(gauge_warn.current_color(), Color::Yellow);

        let gauge_crit = Gauge::new(0.95)
            .fill_color(Color::Green)
            .warn(0.7, Color::Yellow)
            .critical(0.9, Color::Red);

        assert_eq!(gauge_crit.current_color(), Color::Red);
    }

    #[test]
    fn test_gauge_mode_default() {
        assert_eq!(GaugeMode::default(), GaugeMode::Full);
    }

    #[test]
    fn test_gauge_render_compact() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.5).mode(GaugeMode::Compact);
                frame.render_widget(gauge, frame.area());
            })
            .expect("Failed to draw");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // Should contain brackets and fill characters
        assert!(content.contains('['), "Should contain opening bracket");
        assert!(content.contains(']'), "Should contain closing bracket");
    }

    #[test]
    fn test_gauge_render_half() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.75).mode(GaugeMode::Half).label("GPU");
                frame.render_widget(gauge, frame.area());
            })
            .expect("Failed to draw");

        // Just verify it doesn't panic
    }

    #[test]
    fn test_gauge_render_full() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.6).mode(GaugeMode::Full).label("MEM");
                frame.render_widget(gauge, frame.area());
            })
            .expect("Failed to draw");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // Should contain box corners
        assert!(content.contains('╭') || content.contains('['), "Should contain gauge elements");
    }

    #[test]
    fn test_gauge_render_zero() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.0).mode(GaugeMode::Compact);
                frame.render_widget(gauge, frame.area());
            })
            .expect("Should handle zero value");
    }

    #[test]
    fn test_gauge_render_full_value() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(1.0).mode(GaugeMode::Compact);
                frame.render_widget(gauge, frame.area());
            })
            .expect("Should handle full value");
    }

    #[test]
    fn test_gauge_small_area() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let small_area = Rect::new(0, 0, 3, 1);
                let gauge = Gauge::new(0.5).mode(GaugeMode::Full);
                frame.render_widget(gauge, small_area);
            })
            .expect("Should handle small area gracefully");
    }

    #[test]
    fn test_gauge_bg_color() {
        let gauge = Gauge::new(0.5).bg_color(Color::DarkGray);
        assert_eq!(gauge.bg_color, Color::DarkGray);
    }

    #[test]
    fn test_gauge_compact_no_percent() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.5).mode(GaugeMode::Compact).show_percent(false);
                frame.render_widget(gauge, frame.area());
            })
            .expect("Should render compact without percent");
    }

    #[test]
    fn test_gauge_half_no_percent_no_label() {
        let backend = TestBackend::new(20, 5);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.75).mode(GaugeMode::Half).show_percent(false);
                frame.render_widget(gauge, frame.area());
            })
            .expect("Should render half without percent or label");
    }

    #[test]
    fn test_gauge_half_no_percent_with_label() {
        let backend = TestBackend::new(20, 5);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.75).mode(GaugeMode::Half).label("CPU").show_percent(false);
                frame.render_widget(gauge, frame.area());
            })
            .expect("Should render half with label but no percent");
    }

    #[test]
    fn test_gauge_half_with_percent_no_label() {
        let backend = TestBackend::new(20, 5);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.25).mode(GaugeMode::Half).show_percent(true);
                frame.render_widget(gauge, frame.area());
            })
            .expect("Should render half with percent only");
    }

    #[test]
    fn test_gauge_full_with_bar_and_label() {
        let backend = TestBackend::new(20, 8);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        terminal
            .draw(|frame| {
                let gauge =
                    Gauge::new(0.8).mode(GaugeMode::Full).label("DISK").bg_color(Color::DarkGray);
                frame.render_widget(gauge, frame.area());
            })
            .expect("Should render full gauge with bar");
    }

    #[test]
    fn test_gauge_full_with_bg_color_used() {
        let backend = TestBackend::new(15, 6);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.3).mode(GaugeMode::Full).bg_color(Color::Blue);
                frame.render_widget(gauge, frame.area());
            })
            .expect("Should use bg_color for unfilled portion");
    }

    #[test]
    fn test_gauge_quarter_mode() {
        let backend = TestBackend::new(15, 5);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        terminal
            .draw(|frame| {
                let gauge = Gauge::new(0.5).mode(GaugeMode::Quarter).label("Q");
                frame.render_widget(gauge, frame.area());
            })
            .expect("Quarter mode should render as half");
    }
}
