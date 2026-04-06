//! `MicroHeatBar` - Tufte-inspired proportional breakdown visualization
//!
//! A showcase widget demonstrating presentar's data science capabilities:
//! - Heatmap color intensity encoding
//! - Proportional area encoding (Tufte data-ink ratio)
//! - Multi-category display in minimal space
//! - Optional trend indicators
//!
//! # Design Principles (Tufte, 1983)
//! 1. **Data-Ink Ratio**: Every pixel conveys information
//! 2. **Layering**: Color intensity + width = two dimensions in one row
//! 3. **Small Multiples**: Consistent encoding across all instances
//!
//! # Example
//! ```
//! use presentar_terminal::{MicroHeatBar, HeatScheme};
//!
//! // CPU breakdown: usr=54%, sys=19%, io=4%, idle=23%
//! let bar = MicroHeatBar::new(&[54.0, 19.0, 4.0, 23.0])
//!     .with_labels(&["U", "S", "I", "Id"])
//!     .with_scheme(HeatScheme::Thermal);
//! ```

use presentar_core::{Canvas, Color, Point, TextStyle};

/// Color scheme for heat intensity encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeatScheme {
    /// Thermal: green → yellow → orange → red (for CPU/load)
    #[default]
    Thermal,
    /// Cool: light blue → dark blue (for memory)
    Cool,
    /// Warm: yellow → orange → red (for temperature)
    Warm,
    /// Mono: grayscale (for accessibility)
    Mono,
}

impl HeatScheme {
    /// Map a percentage (0-100) to a color
    pub fn color_for_percent(&self, pct: f64) -> Color {
        let p = pct.clamp(0.0, 100.0) / 100.0;

        match self {
            Self::Thermal => {
                // Green → Yellow → Orange → Red
                if p < 0.5 {
                    // Green to Yellow
                    let t = p * 2.0;
                    Color::new(t as f32, 0.8, 0.2, 1.0)
                } else {
                    // Yellow to Red
                    let t = (p - 0.5) * 2.0;
                    Color::new(1.0, (0.8 - t * 0.6) as f32, 0.2, 1.0)
                }
            }
            Self::Cool => {
                // Light blue to dark blue
                Color::new(0.2, 0.4 + (p * 0.4) as f32, 0.9, 1.0)
            }
            Self::Warm => {
                // Yellow to red
                Color::new(1.0, (0.9 - p * 0.7) as f32, 0.1, 1.0)
            }
            Self::Mono => {
                // White to dark gray
                let v = (0.9 - p * 0.7) as f32;
                Color::new(v, v, v, 1.0)
            }
        }
    }
}

/// Style for the micro heat bar rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BarStyle {
    /// Solid blocks: ████▓▓░░
    #[default]
    Blocks,
    /// Gradient shading: uses 8-level Unicode blocks
    Gradient,
    /// Dots/circles: ●●●○○○
    Dots,
    /// Segments with gaps: █ █ █ ░ ░
    Segments,
}

/// A micro heatmap-style proportional bar for category breakdowns
///
/// Renders categories as colored segments where:
/// - Width ∝ percentage (proportional encoding)
/// - Color intensity ∝ "heat" of that category (heatmap encoding)
#[derive(Debug, Clone)]
pub struct MicroHeatBar {
    /// Percentages for each category (should sum to ~100)
    values: Vec<f64>,
    /// Optional short labels for each category
    labels: Vec<String>,
    /// Color scheme
    scheme: HeatScheme,
    /// Rendering style
    style: BarStyle,
    /// Total width in characters
    width: usize,
    /// Show numeric values
    show_values: bool,
}

impl MicroHeatBar {
    /// Create a new MicroHeatBar with the given percentages
    pub fn new(values: &[f64]) -> Self {
        Self {
            values: values.to_vec(),
            labels: Vec::new(),
            scheme: HeatScheme::Thermal,
            style: BarStyle::Blocks,
            width: 20,
            show_values: false,
        }
    }

    /// Set category labels (short, 1-2 chars recommended)
    pub fn with_labels(mut self, labels: &[&str]) -> Self {
        self.labels = labels.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Set the color scheme
    pub fn with_scheme(mut self, scheme: HeatScheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// Set the rendering style
    pub fn with_style(mut self, style: BarStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the total width in characters
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Show numeric values inline
    pub fn with_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Render the bar to a string (for simple display)
    pub fn render_string(&self) -> String {
        let total: f64 = self.values.iter().sum();
        if total <= 0.0 || self.width == 0 {
            return "░".repeat(self.width);
        }
        contract_pre_render!();

        let mut result = String::new();
        let mut remaining_width = self.width;

        for &val in self.values.iter() {
            let proportion = val / total;
            let char_count =
                ((proportion * self.width as f64).round() as usize).min(remaining_width);

            if char_count == 0 {
                continue;
            }

            let ch = match self.style {
                BarStyle::Blocks => {
                    if val > 70.0 {
                        '█'
                    } else if val > 40.0 {
                        '▓'
                    } else if val > 20.0 {
                        '▒'
                    } else if val > 5.0 {
                        '░'
                    } else {
                        ' '
                    }
                }
                BarStyle::Gradient => {
                    // 8-level gradient based on value intensity
                    let level = ((val / 100.0) * 7.0).round() as usize;
                    ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][level.min(7)]
                }
                BarStyle::Dots => {
                    if val > 50.0 {
                        '●'
                    } else {
                        '○'
                    }
                }
                BarStyle::Segments => '█',
            };

            for _ in 0..char_count {
                result.push(ch);
            }
            remaining_width = remaining_width.saturating_sub(char_count);
        }

        // Fill remaining with empty
        while result.chars().count() < self.width {
            result.push('░');
        }

        result
    }

    /// Paint the bar to a canvas at the given position
    pub fn paint(&self, canvas: &mut dyn Canvas, pos: Point) {
        let total: f64 = self.values.iter().sum();
        if total <= 0.0 || self.width == 0 {
            return;
        }

        let mut x = pos.x;

        for &val in self.values.iter() {
            let proportion = val / total;
            let char_count = (proportion * self.width as f64).round() as usize;

            if char_count == 0 {
                continue;
            }

            // Color intensity based on the value itself (heatmap principle)
            let color = self.scheme.color_for_percent(val);

            let ch = match self.style {
                BarStyle::Blocks => '█',
                BarStyle::Gradient => {
                    let level = ((val / 100.0) * 7.0).round() as usize;
                    ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][level.min(7)]
                }
                BarStyle::Dots => '●',
                BarStyle::Segments => '█',
            };

            let segment: String = std::iter::repeat(ch).take(char_count).collect();
            canvas.draw_text(
                &segment,
                Point::new(x, pos.y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );

            x += char_count as f32;
        }

        // Fill remaining width with dim background
        let remaining = self.width.saturating_sub((x - pos.x) as usize);
        if remaining > 0 {
            let bg: String = std::iter::repeat('░').take(remaining).collect();
            canvas.draw_text(
                &bg,
                Point::new(x, pos.y),
                &TextStyle {
                    color: Color::new(0.2, 0.2, 0.2, 1.0),
                    ..Default::default()
                },
            );
        }
    }
}

/// Compact breakdown showing label + micro bar + percentage
/// Example: "U:54 S:19 I:4 ████▓▓░░"
pub struct CompactBreakdown {
    /// Category values
    values: Vec<f64>,
    /// Category labels
    labels: Vec<String>,
    /// Color scheme
    scheme: HeatScheme,
}

impl CompactBreakdown {
    pub fn new(labels: &[&str], values: &[f64]) -> Self {
        Self {
            values: values.to_vec(),
            labels: labels.iter().map(|s| (*s).to_string()).collect(),
            scheme: HeatScheme::Thermal,
        }
    }

    pub fn with_scheme(mut self, scheme: HeatScheme) -> Self {
        self.scheme = scheme;
        self
    }

    /// Render as a compact string: "U:54 S:19 I:4 Id:23"
    pub fn render_text(&self, _width: usize) -> String {
        contract_pre_render!();
        let parts: Vec<String> = self
            .labels
            .iter()
            .zip(self.values.iter())
            .map(|(l, v)| format!("{}:{:.0}", l, v))
            .collect();

        parts.join(" ")
    }

    /// Paint with colors to canvas
    pub fn paint(&self, canvas: &mut dyn Canvas, pos: Point) {
        let mut x = pos.x;

        for (label, &val) in self.labels.iter().zip(self.values.iter()) {
            let color = self.scheme.color_for_percent(val);
            let text = format!("{}:{:.0} ", label, val);

            canvas.draw_text(
                &text,
                Point::new(x, pos.y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );

            x += text.chars().count() as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    // =========================================================================
    // HEAT SCHEME TESTS
    // =========================================================================

    #[test]
    fn test_heat_scheme_default() {
        assert_eq!(HeatScheme::default(), HeatScheme::Thermal);
    }

    #[test]
    fn test_heat_scheme_thermal() {
        let scheme = HeatScheme::Thermal;

        let low = scheme.color_for_percent(10.0);
        let high = scheme.color_for_percent(90.0);

        // Low should be more green, high should be more red
        assert!(low.g > low.r);
        assert!(high.r > high.g);
    }

    #[test]
    fn test_heat_scheme_thermal_midpoint() {
        let scheme = HeatScheme::Thermal;
        let mid = scheme.color_for_percent(50.0);
        // At 50%, should be transitioning (yellow-ish)
        assert!(mid.r > 0.5);
        assert!(mid.g > 0.5);
    }

    #[test]
    fn test_heat_scheme_cool() {
        let scheme = HeatScheme::Cool;
        let low = scheme.color_for_percent(10.0);
        let high = scheme.color_for_percent(90.0);

        // Cool scheme should be blue-ish
        assert!(low.b > low.r);
        assert!(high.b > high.r);
        // Higher values should have more green component
        assert!(high.g > low.g);
    }

    #[test]
    fn test_heat_scheme_warm() {
        let scheme = HeatScheme::Warm;
        let low = scheme.color_for_percent(10.0);
        let high = scheme.color_for_percent(90.0);

        // Low should be yellow-ish (high green), high should be red
        assert!(low.g > high.g);
        assert_eq!(low.r, 1.0);
        assert_eq!(high.r, 1.0);
    }

    #[test]
    fn test_heat_scheme_mono() {
        let scheme = HeatScheme::Mono;
        let low = scheme.color_for_percent(10.0);
        let high = scheme.color_for_percent(90.0);

        // Mono should have equal RGB components (grayscale)
        assert_eq!(low.r, low.g);
        assert_eq!(low.g, low.b);
        assert_eq!(high.r, high.g);
        assert_eq!(high.g, high.b);

        // Lower values should be brighter (closer to white)
        assert!(low.r > high.r);
    }

    #[test]
    fn test_heat_scheme_clamps_values() {
        let scheme = HeatScheme::Thermal;

        // Values outside 0-100 should be clamped
        let neg = scheme.color_for_percent(-50.0);
        let over = scheme.color_for_percent(150.0);

        // Same as 0% and 100%
        let zero = scheme.color_for_percent(0.0);
        let hundred = scheme.color_for_percent(100.0);

        assert_eq!(neg.r, zero.r);
        assert_eq!(over.r, hundred.r);
    }

    // =========================================================================
    // BAR STYLE TESTS
    // =========================================================================

    #[test]
    fn test_bar_style_default() {
        assert_eq!(BarStyle::default(), BarStyle::Blocks);
    }

    // =========================================================================
    // MICRO HEAT BAR TESTS
    // =========================================================================

    #[test]
    fn test_micro_heat_bar_new() {
        let bar = MicroHeatBar::new(&[50.0, 30.0, 20.0]);
        assert_eq!(bar.values.len(), 3);
        assert_eq!(bar.width, 20);
        assert!(!bar.show_values);
    }

    #[test]
    fn test_micro_heat_bar_with_labels() {
        let bar = MicroHeatBar::new(&[50.0, 50.0]).with_labels(&["A", "B"]);
        assert_eq!(bar.labels.len(), 2);
        assert_eq!(bar.labels[0], "A");
    }

    #[test]
    fn test_micro_heat_bar_with_scheme() {
        let bar = MicroHeatBar::new(&[50.0]).with_scheme(HeatScheme::Cool);
        assert_eq!(bar.scheme, HeatScheme::Cool);
    }

    #[test]
    fn test_micro_heat_bar_with_style() {
        let bar = MicroHeatBar::new(&[50.0]).with_style(BarStyle::Gradient);
        assert_eq!(bar.style, BarStyle::Gradient);
    }

    #[test]
    fn test_micro_heat_bar_with_width() {
        let bar = MicroHeatBar::new(&[50.0]).with_width(40);
        assert_eq!(bar.width, 40);
    }

    #[test]
    fn test_micro_heat_bar_with_values() {
        let bar = MicroHeatBar::new(&[50.0]).with_values(true);
        assert!(bar.show_values);
    }

    #[test]
    fn test_micro_heat_bar_render() {
        let bar = MicroHeatBar::new(&[54.0, 19.0, 4.0, 23.0]).with_width(20);

        let rendered = bar.render_string();
        assert_eq!(rendered.chars().count(), 20);
    }

    #[test]
    fn test_micro_heat_bar_render_empty() {
        let bar = MicroHeatBar::new(&[]).with_width(10);
        let rendered = bar.render_string();
        assert_eq!(rendered, "░░░░░░░░░░");
    }

    #[test]
    fn test_micro_heat_bar_render_zero_width() {
        let bar = MicroHeatBar::new(&[50.0]).with_width(0);
        let rendered = bar.render_string();
        assert_eq!(rendered, "");
    }

    #[test]
    fn test_micro_heat_bar_render_all_zeros() {
        let bar = MicroHeatBar::new(&[0.0, 0.0, 0.0]).with_width(10);
        let rendered = bar.render_string();
        assert_eq!(rendered, "░░░░░░░░░░");
    }

    #[test]
    fn test_micro_heat_bar_render_blocks_style() {
        let bar = MicroHeatBar::new(&[80.0, 50.0, 30.0, 10.0, 2.0])
            .with_style(BarStyle::Blocks)
            .with_width(10);
        let rendered = bar.render_string();
        // High values should use denser blocks
        assert!(rendered.contains('█') || rendered.contains('▓') || rendered.contains('▒'));
    }

    #[test]
    fn test_micro_heat_bar_render_gradient_style() {
        let bar = MicroHeatBar::new(&[50.0, 50.0])
            .with_style(BarStyle::Gradient)
            .with_width(10);
        let rendered = bar.render_string();
        // Gradient uses different block heights
        assert!(rendered
            .chars()
            .any(|c| matches!(c, '▁' | '▂' | '▃' | '▄' | '▅' | '▆' | '▇' | '█')));
    }

    #[test]
    fn test_micro_heat_bar_render_dots_style() {
        let bar = MicroHeatBar::new(&[60.0, 40.0])
            .with_style(BarStyle::Dots)
            .with_width(10);
        let rendered = bar.render_string();
        // Dots style uses filled or empty circles
        assert!(rendered.contains('●') || rendered.contains('○'));
    }

    #[test]
    fn test_micro_heat_bar_render_segments_style() {
        let bar = MicroHeatBar::new(&[50.0, 50.0])
            .with_style(BarStyle::Segments)
            .with_width(10);
        let rendered = bar.render_string();
        assert!(rendered.contains('█'));
    }

    #[test]
    fn test_micro_heat_bar_paint() {
        let mut buffer = CellBuffer::new(30, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let bar = MicroHeatBar::new(&[54.0, 19.0, 4.0, 23.0]).with_width(20);
        bar.paint(&mut canvas, Point::new(0.0, 0.0));
    }

    #[test]
    fn test_micro_heat_bar_paint_empty() {
        let mut buffer = CellBuffer::new(30, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let bar = MicroHeatBar::new(&[]).with_width(10);
        bar.paint(&mut canvas, Point::new(0.0, 0.0));
        // Should return early without error
    }

    #[test]
    fn test_micro_heat_bar_paint_gradient() {
        let mut buffer = CellBuffer::new(30, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let bar = MicroHeatBar::new(&[50.0, 50.0])
            .with_style(BarStyle::Gradient)
            .with_width(20);
        bar.paint(&mut canvas, Point::new(0.0, 0.0));
    }

    #[test]
    fn test_micro_heat_bar_paint_dots() {
        let mut buffer = CellBuffer::new(30, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let bar = MicroHeatBar::new(&[60.0, 40.0])
            .with_style(BarStyle::Dots)
            .with_width(20);
        bar.paint(&mut canvas, Point::new(0.0, 0.0));
    }

    #[test]
    fn test_micro_heat_bar_paint_with_remaining() {
        let mut buffer = CellBuffer::new(50, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Small values that won't fill the whole width
        let bar = MicroHeatBar::new(&[10.0]).with_width(30);
        bar.paint(&mut canvas, Point::new(0.0, 0.0));
        // Should fill remaining with dim background
    }

    // =========================================================================
    // COMPACT BREAKDOWN TESTS
    // =========================================================================

    #[test]
    fn test_compact_breakdown_new() {
        let breakdown = CompactBreakdown::new(&["U", "S", "I"], &[50.0, 30.0, 20.0]);
        assert_eq!(breakdown.labels.len(), 3);
        assert_eq!(breakdown.values.len(), 3);
    }

    #[test]
    fn test_compact_breakdown_with_scheme() {
        let breakdown = CompactBreakdown::new(&["A"], &[50.0]).with_scheme(HeatScheme::Cool);
        assert_eq!(breakdown.scheme, HeatScheme::Cool);
    }

    #[test]
    fn test_compact_breakdown_render_text() {
        let breakdown = CompactBreakdown::new(&["U", "S", "I", "Id"], &[54.0, 19.0, 4.0, 23.0]);

        let text = breakdown.render_text(40);
        assert!(text.contains("U:54"));
        assert!(text.contains("S:19"));
        assert!(text.contains("I:4"));
        assert!(text.contains("Id:23"));
    }

    #[test]
    fn test_compact_breakdown_paint() {
        let mut buffer = CellBuffer::new(50, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let breakdown = CompactBreakdown::new(&["U", "S"], &[60.0, 40.0]);
        breakdown.paint(&mut canvas, Point::new(0.0, 0.0));
    }

    #[test]
    fn test_compact_breakdown_paint_with_scheme() {
        let mut buffer = CellBuffer::new(50, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let breakdown =
            CompactBreakdown::new(&["A", "B"], &[80.0, 20.0]).with_scheme(HeatScheme::Warm);
        breakdown.paint(&mut canvas, Point::new(0.0, 0.0));
    }
}
