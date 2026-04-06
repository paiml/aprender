//! Percentage meter widget with gradient coloring.
//!
//! Displays a horizontal bar showing a percentage value with optional
//! gradient coloring based on the value.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

/// A horizontal percentage meter.
#[derive(Debug, Clone)]
pub struct Meter {
    /// Value between 0.0 and 1.0.
    value: f64,
    /// Optional label to display.
    label: Option<String>,
    /// Color for the filled portion.
    color: Color,
    /// Whether to show the percentage text.
    show_percentage: bool,
}

impl Meter {
    /// Creates a new meter with the given value (0.0 - 1.0).
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
            label: None,
            color: Color::Green,
            show_percentage: true,
        }
    }

    /// Sets the label.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the color.
    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Sets whether to show the percentage.
    #[must_use]
    pub fn show_percentage(mut self, show: bool) -> Self {
        self.show_percentage = show;
        self
    }
}

impl Widget for Meter {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }

        // Calculate bar width
        let label_width = self.label.as_ref().map_or(0, |l| l.len() + 1) as u16;
        let percent_width = if self.show_percentage { 5 } else { 0 }; // " 100%"
        let bar_width = area.width.saturating_sub(label_width + percent_width);

        let mut x = area.x;

        // Render label
        if let Some(label) = &self.label {
            buf.set_string(x, area.y, label, Style::default());
            x += label.len() as u16 + 1;
        }

        // Render bar
        let filled = ((self.value * f64::from(bar_width)) as u16).min(bar_width);
        for i in 0..bar_width {
            let char = if i < filled { '█' } else { '░' };
            let style = if i < filled {
                Style::default().fg(self.color)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            buf.set_string(x + i, area.y, char.to_string(), style);
        }

        // Render percentage
        if self.show_percentage {
            let percent = format!("{:3.0}%", self.value * 100.0);
            buf.set_string(x + bar_width + 1, area.y, percent, Style::default());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn test_meter_new() {
        let meter = Meter::new(0.5);
        assert!((meter.value - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_meter_clamps_value() {
        let meter = Meter::new(1.5);
        assert!((meter.value - 1.0).abs() < f64::EPSILON);

        let meter = Meter::new(-0.5);
        assert!((meter.value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_meter_builder() {
        let meter = Meter::new(0.75).label("CPU").color(Color::Red).show_percentage(false);

        assert_eq!(meter.label, Some("CPU".to_string()));
        assert_eq!(meter.color, Color::Red);
        assert!(!meter.show_percentage);
    }

    #[test]
    fn test_meter_renders() {
        let backend = TestBackend::new(40, 1);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        terminal
            .draw(|frame| {
                let meter = Meter::new(0.5).label("Test");
                frame.render_widget(meter, frame.area());
            })
            .expect("operation should succeed");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        assert!(content.contains("Test"), "Should contain label");
        assert!(content.contains("50%") || content.contains(" 50%"), "Should contain percentage");
    }
}
