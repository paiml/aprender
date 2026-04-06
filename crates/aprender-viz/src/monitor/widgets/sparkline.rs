//! Inline sparkline widget for compact trend display.
//!
//! Uses 8-level Unicode block characters (▁▂▃▄▅▆▇█) to show trends
//! in a compact format suitable for table cells.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

/// A compact inline sparkline.
#[derive(Debug, Clone)]
pub struct MonitorSparkline<'a> {
    /// Data points to display.
    data: &'a [f64],
    /// Color for the sparkline.
    color: Color,
    /// Whether to show a trend indicator suffix (↑↓→).
    show_trend: bool,
}

impl<'a> MonitorSparkline<'a> {
    /// Creates a new sparkline with the given data.
    #[must_use]
    pub fn new(data: &'a [f64]) -> Self {
        Self { data, color: Color::Cyan, show_trend: true }
    }

    /// Sets the color.
    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Sets whether to show trend indicator.
    #[must_use]
    pub fn show_trend(mut self, show: bool) -> Self {
        self.show_trend = show;
        self
    }

    /// Calculates the trend based on recent values.
    fn trend(&self) -> char {
        if self.data.len() < 2 {
            return '→';
        }

        let recent: Vec<_> = self.data.iter().rev().take(5).collect();
        if recent.len() < 2 {
            return '→';
        }

        // Safe: we've verified recent.len() >= 2 above
        let (Some(first), Some(last)) = (recent.last(), recent.first()) else {
            return '→';
        };
        let first = **first;
        let last = **last;
        let diff = last - first;
        let threshold = 0.05; // 5% threshold

        if diff > threshold {
            '↑'
        } else if diff < -threshold {
            '↓'
        } else {
            '→'
        }
    }
}

impl Widget for MonitorSparkline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || self.data.is_empty() {
            return;
        }

        // 8-level block characters
        let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        // Find min/max for scaling
        let min = self.data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        // Reserve space for trend indicator
        let chart_width =
            if self.show_trend { area.width.saturating_sub(1) } else { area.width } as usize;

        // Sample data to fit width
        for i in 0..chart_width.min(self.data.len()) {
            let data_idx = if chart_width >= self.data.len() {
                i
            } else {
                (i * self.data.len()) / chart_width
            };

            let value = self.data.get(data_idx).copied().unwrap_or(0.0);
            let normalized =
                if range > 0.0 { ((value - min) / range).clamp(0.0, 1.0) } else { 0.5 };

            let block_idx = ((normalized * 7.0) as usize).min(7);
            let block = blocks[block_idx];

            buf.set_string(
                area.x + i as u16,
                area.y,
                block.to_string(),
                Style::default().fg(self.color),
            );
        }

        // Add trend indicator
        if self.show_trend && area.width > 1 {
            let trend = self.trend();
            buf.set_string(
                area.x + area.width - 1,
                area.y,
                trend.to_string(),
                Style::default().fg(self.color),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn test_sparkline_new() {
        let data = vec![0.5; 10];
        let sparkline = MonitorSparkline::new(&data);

        assert_eq!(sparkline.color, Color::Cyan);
        assert!(sparkline.show_trend);
    }

    #[test]
    fn test_sparkline_color() {
        let data = vec![0.5];
        let sparkline = MonitorSparkline::new(&data).color(Color::Red);

        assert_eq!(sparkline.color, Color::Red);
    }

    #[test]
    fn test_sparkline_show_trend_false() {
        let data = vec![0.5];
        let sparkline = MonitorSparkline::new(&data).show_trend(false);

        assert!(!sparkline.show_trend);
    }

    #[test]
    fn test_sparkline_trend_up() {
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let sparkline = MonitorSparkline::new(&data);

        assert_eq!(sparkline.trend(), '↑');
    }

    #[test]
    fn test_sparkline_trend_down() {
        let data = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let sparkline = MonitorSparkline::new(&data);

        assert_eq!(sparkline.trend(), '↓');
    }

    #[test]
    fn test_sparkline_trend_stable() {
        let data = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let sparkline = MonitorSparkline::new(&data);

        assert_eq!(sparkline.trend(), '→');
    }

    #[test]
    fn test_sparkline_trend_single_point() {
        let data = vec![0.5];
        let sparkline = MonitorSparkline::new(&data);

        // Single point should return stable
        assert_eq!(sparkline.trend(), '→');
    }

    #[test]
    fn test_sparkline_trend_empty() {
        let data: Vec<f64> = vec![];
        let sparkline = MonitorSparkline::new(&data);

        // Empty should return stable
        assert_eq!(sparkline.trend(), '→');
    }

    #[test]
    fn test_sparkline_renders() {
        let backend = TestBackend::new(15, 1);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        let data = vec![0.1, 0.3, 0.5, 0.7, 0.9];

        terminal
            .draw(|frame| {
                let sparkline = MonitorSparkline::new(&data);
                frame.render_widget(sparkline, frame.area());
            })
            .expect("operation should succeed");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // Should contain block characters
        assert!(content.chars().any(|c| "▁▂▃▄▅▆▇█".contains(c)), "Should contain block characters");
    }

    #[test]
    fn test_sparkline_render_empty_data() {
        let backend = TestBackend::new(10, 1);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        let data: Vec<f64> = vec![];

        terminal
            .draw(|frame| {
                let sparkline = MonitorSparkline::new(&data);
                frame.render_widget(sparkline, frame.area());
            })
            .expect("operation should succeed");

        // Should render without panic
        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // Empty data should produce spaces only
        assert!(content.trim().is_empty() || content.chars().all(|c| c == ' '));
    }

    #[test]
    fn test_sparkline_render_zero_width() {
        let backend = TestBackend::new(10, 1);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        let data = vec![0.5, 0.6, 0.7];

        terminal
            .draw(|frame| {
                let sparkline = MonitorSparkline::new(&data);
                // Use zero-width area
                let area = Rect::new(0, 0, 0, 1);
                frame.render_widget(sparkline, area);
            })
            .expect("operation should succeed");

        // Should render without panic
    }

    #[test]
    fn test_sparkline_render_no_trend() {
        let backend = TestBackend::new(10, 1);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        let data = vec![0.1, 0.3, 0.5, 0.7, 0.9];

        terminal
            .draw(|frame| {
                let sparkline = MonitorSparkline::new(&data).show_trend(false);
                frame.render_widget(sparkline, frame.area());
            })
            .expect("operation should succeed");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // Should NOT contain trend arrows
        assert!(!content.contains('↑'));
        assert!(!content.contains('↓'));
        assert!(!content.contains('→'));
    }

    #[test]
    fn test_sparkline_render_constant_values() {
        let backend = TestBackend::new(10, 1);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        // All same values -> range = 0
        let data = vec![0.5, 0.5, 0.5, 0.5, 0.5];

        terminal
            .draw(|frame| {
                let sparkline = MonitorSparkline::new(&data);
                frame.render_widget(sparkline, frame.area());
            })
            .expect("operation should succeed");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // Should render middle block for constant values
        assert!(content.contains('▄')); // Middle block (normalized = 0.5)
    }

    #[test]
    fn test_sparkline_render_data_longer_than_width() {
        let backend = TestBackend::new(5, 1);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        // 20 data points but only 5 chars width (minus trend = 4)
        let data: Vec<f64> = (0..20).map(|i| f64::from(i) / 20.0).collect();

        terminal
            .draw(|frame| {
                let sparkline = MonitorSparkline::new(&data);
                frame.render_widget(sparkline, frame.area());
            })
            .expect("operation should succeed");

        // Should render without panic, sampling the data
        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        assert!(content.chars().any(|c| "▁▂▃▄▅▆▇█↑↓→".contains(c)));
    }

    #[test]
    fn test_sparkline_render_width_larger_than_data() {
        let backend = TestBackend::new(20, 1);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        // Only 3 data points but 20 chars width
        let data = vec![0.0, 0.5, 1.0];

        terminal
            .draw(|frame| {
                let sparkline = MonitorSparkline::new(&data);
                frame.render_widget(sparkline, frame.area());
            })
            .expect("operation should succeed");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // Should contain blocks for each data point
        assert!(content.contains('▁')); // Low
        assert!(content.contains('█')); // High
    }

    #[test]
    fn test_sparkline_render_with_color() {
        let backend = TestBackend::new(10, 1);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        let data = vec![0.1, 0.5, 0.9];

        terminal
            .draw(|frame| {
                let sparkline = MonitorSparkline::new(&data).color(Color::Green);
                frame.render_widget(sparkline, frame.area());
            })
            .expect("operation should succeed");

        // Check that cells have the correct color
        let buffer = terminal.backend().buffer();
        let cell = buffer.cell((0, 0)).expect("operation should succeed");
        assert_eq!(cell.fg, Color::Green);
    }

    #[test]
    fn test_sparkline_trend_within_threshold() {
        // Just under threshold should be stable
        let data = vec![0.5, 0.5, 0.5, 0.5, 0.54]; // diff = 0.04 < 0.05
        let sparkline = MonitorSparkline::new(&data);

        assert_eq!(sparkline.trend(), '→');
    }

    #[test]
    fn test_sparkline_trend_at_threshold() {
        // At threshold should be up
        let data = vec![0.5, 0.5, 0.5, 0.5, 0.56]; // diff = 0.06 > 0.05
        let sparkline = MonitorSparkline::new(&data);

        assert_eq!(sparkline.trend(), '↑');
    }
}
