//! Custom TUI widgets for trueno-ublk dashboard

#![allow(dead_code)]

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::Widget,
};

/// Horizontal bar chart for entropy distribution
pub struct EntropyBar {
    gpu_pct: u16,
    simd_pct: u16,
    scalar_pct: u16,
}

impl EntropyBar {
    pub fn new(gpu: u64, simd: u64, scalar: u64) -> Self {
        let total = gpu + simd + scalar;
        if total == 0 {
            return Self { gpu_pct: 0, simd_pct: 0, scalar_pct: 0 };
        }

        Self {
            gpu_pct: ((gpu * 100) / total) as u16,
            simd_pct: ((simd * 100) / total) as u16,
            scalar_pct: ((scalar * 100) / total) as u16,
        }
    }
}

impl Widget for EntropyBar {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let width = area.width as usize;
        let gpu_width = (width * self.gpu_pct as usize) / 100;
        let simd_width = (width * self.simd_pct as usize) / 100;
        let scalar_width = width.saturating_sub(gpu_width + simd_width);

        let mut x = area.x;

        // GPU portion (magenta)
        for _ in 0..gpu_width {
            if x < area.x + area.width {
                buf[(x, area.y)].set_char('█').set_style(Style::default().fg(Color::Magenta));
                x += 1;
            }
        }

        // SIMD portion (cyan)
        for _ in 0..simd_width {
            if x < area.x + area.width {
                buf[(x, area.y)].set_char('█').set_style(Style::default().fg(Color::Cyan));
                x += 1;
            }
        }

        // Scalar portion (yellow)
        for _ in 0..scalar_width {
            if x < area.x + area.width {
                buf[(x, area.y)].set_char('█').set_style(Style::default().fg(Color::Yellow));
                x += 1;
            }
        }
    }
}

/// Sparkline for throughput history
pub struct ThroughputSparkline<'a> {
    data: &'a [f64],
    max: f64,
}

impl<'a> ThroughputSparkline<'a> {
    pub fn new(data: &'a [f64]) -> Self {
        let max = data.iter().cloned().fold(0.0f64, f64::max);
        Self { data, max: max.max(1.0) }
    }
}

impl Widget for ThroughputSparkline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        let width = area.width as usize;
        let data_start = self.data.len().saturating_sub(width);

        for (i, &value) in self.data[data_start..].iter().enumerate() {
            let x = area.x + i as u16;
            if x >= area.x + area.width {
                break;
            }

            let normalized = (value / self.max * 7.0).round() as usize;
            let bar_char = BARS[normalized.min(7)];

            buf[(x, area.y)].set_char(bar_char).set_style(Style::default().fg(Color::Green));
        }
    }
}

/// Compression ratio indicator
pub struct RatioIndicator {
    ratio: f64,
}

impl RatioIndicator {
    pub fn new(ratio: f64) -> Self {
        Self { ratio }
    }
}

impl Widget for RatioIndicator {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let color = if self.ratio >= 3.0 {
            Color::Green
        } else if self.ratio >= 2.0 {
            Color::Yellow
        } else if self.ratio >= 1.5 {
            Color::LightYellow
        } else {
            Color::Red
        };

        let text = format!("{:.2}:1", self.ratio);
        let x = area.x;
        for (i, ch) in text.chars().enumerate() {
            if x + (i as u16) < area.x + area.width {
                buf[(x + (i as u16), area.y)].set_char(ch).set_style(Style::default().fg(color));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Helper functions for widget testing
    // ========================================================================

    fn create_test_buffer(width: u16, height: u16) -> (Buffer, Rect) {
        let area = Rect::new(0, 0, width, height);
        let buffer = Buffer::empty(area);
        (buffer, area)
    }

    // ========================================================================
    // Section E: TUI & Observability Tests (from Renacer Verification Matrix)
    // ========================================================================

    // E.51: Dashboard renders throughput sparkline correctly
    #[test]
    fn test_throughput_sparkline_renders() {
        let (mut buf, area) = create_test_buffer(10, 1);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let sparkline = ThroughputSparkline::new(&data);
        sparkline.render(area, &mut buf);

        // Verify that characters are rendered (sparkline bars)
        let chars: String =
            (0..10).map(|x| buf[(x, 0)].symbol().chars().next().unwrap_or(' ')).collect();
        assert!(!chars.trim().is_empty(), "Sparkline should render characters");
    }

    #[test]
    fn test_throughput_sparkline_bar_heights() {
        let (mut buf, area) = create_test_buffer(8, 1);
        // Data that should produce increasing bar heights
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let sparkline = ThroughputSparkline::new(&data);
        sparkline.render(area, &mut buf);

        // First bar should be lowest (▁), last should be highest (█)
        let first_char = buf[(0, 0)].symbol().chars().next().unwrap();
        let last_char = buf[(7, 0)].symbol().chars().next().unwrap();
        assert_eq!(first_char, '▁', "First bar should be lowest");
        assert_eq!(last_char, '█', "Last bar should be highest");
    }

    #[test]
    fn test_throughput_sparkline_empty_data() {
        let (mut buf, area) = create_test_buffer(10, 1);
        let data: Vec<f64> = vec![];

        let sparkline = ThroughputSparkline::new(&data);
        sparkline.render(area, &mut buf);

        // Should not panic with empty data
    }

    #[test]
    fn test_throughput_sparkline_truncates_to_width() {
        let (mut buf, area) = create_test_buffer(5, 1);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let sparkline = ThroughputSparkline::new(&data);
        sparkline.render(area, &mut buf);

        // Should only render last 5 values
        // Position 4 should have highest value (from 10.0)
        let last_char = buf[(4, 0)].symbol().chars().next().unwrap();
        assert_eq!(last_char, '█', "Last value should be max height");
    }

    // E.52: Compression ratio widget updates in real-time
    #[test]
    fn test_ratio_indicator_color_green() {
        let (mut buf, area) = create_test_buffer(10, 1);
        let indicator = RatioIndicator::new(3.5);
        indicator.render(area, &mut buf);

        // Check color is green for ratio >= 3.0
        let style = buf[(0, 0)].style();
        assert_eq!(style.fg, Some(Color::Green), "Ratio >= 3.0 should be green");
    }

    #[test]
    fn test_ratio_indicator_color_yellow() {
        let (mut buf, area) = create_test_buffer(10, 1);
        let indicator = RatioIndicator::new(2.5);
        indicator.render(area, &mut buf);

        // Check color is yellow for ratio >= 2.0 and < 3.0
        let style = buf[(0, 0)].style();
        assert_eq!(style.fg, Some(Color::Yellow), "Ratio 2.0-3.0 should be yellow");
    }

    #[test]
    fn test_ratio_indicator_color_light_yellow() {
        let (mut buf, area) = create_test_buffer(10, 1);
        let indicator = RatioIndicator::new(1.7);
        indicator.render(area, &mut buf);

        // Check color is light yellow for ratio >= 1.5 and < 2.0
        let style = buf[(0, 0)].style();
        assert_eq!(style.fg, Some(Color::LightYellow), "Ratio 1.5-2.0 should be light yellow");
    }

    #[test]
    fn test_ratio_indicator_color_red() {
        let (mut buf, area) = create_test_buffer(10, 1);
        let indicator = RatioIndicator::new(1.2);
        indicator.render(area, &mut buf);

        // Check color is red for ratio < 1.5
        let style = buf[(0, 0)].style();
        assert_eq!(style.fg, Some(Color::Red), "Ratio < 1.5 should be red");
    }

    #[test]
    fn test_ratio_indicator_text_format() {
        let (mut buf, area) = create_test_buffer(10, 1);
        let indicator = RatioIndicator::new(2.45);
        indicator.render(area, &mut buf);

        // Collect rendered text
        let text: String =
            (0..area.width).map(|x| buf[(x, 0)].symbol().chars().next().unwrap_or(' ')).collect();

        assert!(text.starts_with("2.45:1"), "Should render as '2.45:1', got '{}'", text);
    }

    // E.53: Entropy distribution bar chart matches data input
    #[test]
    fn test_entropy_bar_percentages() {
        let bar = EntropyBar::new(50, 30, 20);
        assert_eq!(bar.gpu_pct, 50);
        assert_eq!(bar.simd_pct, 30);
        assert_eq!(bar.scalar_pct, 20);
    }

    #[test]
    fn test_entropy_bar_renders_correct_proportions() {
        let (mut buf, area) = create_test_buffer(100, 1);
        // GPU: 50%, SIMD: 30%, Scalar: 20%
        let bar = EntropyBar::new(50, 30, 20);
        bar.render(area, &mut buf);

        // Count colors
        let mut gpu_count = 0;
        let mut simd_count = 0;
        let mut scalar_count = 0;

        for x in 0..100 {
            let style = buf[(x, 0)].style();
            match style.fg {
                Some(Color::Magenta) => gpu_count += 1,
                Some(Color::Cyan) => simd_count += 1,
                Some(Color::Yellow) => scalar_count += 1,
                _ => {}
            }
        }

        // Allow ±2 tolerance due to rounding
        assert!((gpu_count as i32 - 50).abs() <= 2, "GPU should be ~50%, got {}", gpu_count);
        assert!((simd_count as i32 - 30).abs() <= 2, "SIMD should be ~30%, got {}", simd_count);
        assert!(
            (scalar_count as i32 - 20).abs() <= 2,
            "Scalar should be ~20%, got {}",
            scalar_count
        );
    }

    #[test]
    fn test_entropy_bar_empty() {
        let bar = EntropyBar::new(0, 0, 0);
        assert_eq!(bar.gpu_pct, 0);
        assert_eq!(bar.simd_pct, 0);
        assert_eq!(bar.scalar_pct, 0);
    }

    #[test]
    fn test_entropy_bar_single_category() {
        // All GPU
        let bar = EntropyBar::new(100, 0, 0);
        assert_eq!(bar.gpu_pct, 100);
        assert_eq!(bar.simd_pct, 0);
        assert_eq!(bar.scalar_pct, 0);

        // All SIMD
        let bar = EntropyBar::new(0, 100, 0);
        assert_eq!(bar.gpu_pct, 0);
        assert_eq!(bar.simd_pct, 100);
        assert_eq!(bar.scalar_pct, 0);

        // All Scalar
        let bar = EntropyBar::new(0, 0, 100);
        assert_eq!(bar.gpu_pct, 0);
        assert_eq!(bar.simd_pct, 0);
        assert_eq!(bar.scalar_pct, 100);
    }

    // E.55: Window resizing does not cause panic or layout corruption
    #[test]
    fn test_entropy_bar_various_sizes() {
        // Test various window sizes
        for width in [1, 5, 10, 50, 100, 200] {
            let (mut buf, area) = create_test_buffer(width, 1);
            let bar = EntropyBar::new(33, 33, 34);
            bar.render(area, &mut buf); // Should not panic
        }
    }

    #[test]
    fn test_sparkline_various_sizes() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for width in [1, 3, 5, 10, 50, 100] {
            let (mut buf, area) = create_test_buffer(width, 1);
            let sparkline = ThroughputSparkline::new(&data);
            sparkline.render(area, &mut buf); // Should not panic
        }
    }

    #[test]
    fn test_ratio_indicator_various_sizes() {
        for width in [1, 5, 10, 20] {
            let (mut buf, area) = create_test_buffer(width, 1);
            let indicator = RatioIndicator::new(2.5);
            indicator.render(area, &mut buf); // Should not panic
        }
    }

    #[test]
    fn test_widgets_zero_size_area() {
        // Zero width - should not panic
        let (mut buf, area) = create_test_buffer(0, 0);

        let bar = EntropyBar::new(50, 30, 20);
        bar.render(area, &mut buf);

        let data = vec![1.0, 2.0, 3.0];
        let sparkline = ThroughputSparkline::new(&data);
        sparkline.render(area, &mut buf);

        let indicator = RatioIndicator::new(2.0);
        indicator.render(area, &mut buf);
    }

    // Additional widget tests for edge cases
    #[test]
    fn test_ratio_indicator_extreme_values() {
        // Very high ratio
        let (mut buf, area) = create_test_buffer(20, 1);
        let indicator = RatioIndicator::new(100.0);
        indicator.render(area, &mut buf);
        let style = buf[(0, 0)].style();
        assert_eq!(style.fg, Some(Color::Green), "High ratio should be green");

        // Very low ratio
        let indicator = RatioIndicator::new(0.5);
        indicator.render(area, &mut buf);
        let style = buf[(0, 0)].style();
        assert_eq!(style.fg, Some(Color::Red), "Low ratio should be red");
    }

    #[test]
    fn test_sparkline_single_value() {
        let (mut buf, area) = create_test_buffer(10, 1);
        let data = vec![5.0];

        let sparkline = ThroughputSparkline::new(&data);
        sparkline.render(area, &mut buf);

        // Single value should render at max height
        let char0 = buf[(0, 0)].symbol().chars().next().unwrap();
        assert_eq!(char0, '█', "Single value should be max height");
    }

    #[test]
    fn test_sparkline_all_zeros() {
        let (mut buf, area) = create_test_buffer(5, 1);
        let data = vec![0.0, 0.0, 0.0, 0.0, 0.0];

        let sparkline = ThroughputSparkline::new(&data);
        sparkline.render(area, &mut buf);

        // All zeros should render as lowest bar
        for x in 0..5 {
            let ch = buf[(x, 0)].symbol().chars().next().unwrap();
            assert_eq!(ch, '▁', "Zero value should be lowest bar");
        }
    }

    #[test]
    fn test_entropy_bar_fill_character() {
        let (mut buf, area) = create_test_buffer(10, 1);
        let bar = EntropyBar::new(100, 0, 0);
        bar.render(area, &mut buf);

        // Check that solid blocks are used
        for x in 0..10 {
            let ch = buf[(x, 0)].symbol().chars().next().unwrap();
            assert_eq!(ch, '█', "Should use solid block character");
        }
    }
}
