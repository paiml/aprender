//! Gantt chart widget for OTLP span waterfall visualization
//!
//! Renders spans as horizontal bars proportional to duration,
//! with nested spans indented by parent relationship.
//! Critical path spans are highlighted in bold.

use crate::visualize::collectors::span::{SpanKind, SpanRecord};
use crate::visualize::theme::graph;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Widget},
};

/// Characters for rendering Gantt bars
const BAR_START: char = '▐';
const BAR_MIDDLE: char = '█';
const BAR_END: char = '▌';
const BAR_THIN: char = '▬';

/// Gantt chart configuration
#[derive(Debug, Clone)]
pub struct GanttConfig {
    /// Minimum bar width (in characters)
    pub min_bar_width: u16,
    /// Maximum nesting depth to display
    pub max_depth: usize,
    /// Indent per nesting level
    pub indent: u16,
    /// Show span names
    pub show_names: bool,
    /// Show durations
    pub show_durations: bool,
}

impl Default for GanttConfig {
    fn default() -> Self {
        Self { min_bar_width: 1, max_depth: 10, indent: 2, show_names: true, show_durations: true }
    }
}

/// Gantt chart widget for span visualization
pub struct GanttChart<'a> {
    /// Spans to render
    spans: &'a [SpanRecord],
    /// Time range (start_ns, end_ns)
    time_range: (u64, u64),
    /// Selected span index
    selected: Option<usize>,
    /// Configuration
    config: GanttConfig,
    /// Block for borders
    block: Option<Block<'a>>,
}

impl<'a> GanttChart<'a> {
    /// Create new Gantt chart with spans
    pub fn new(spans: &'a [SpanRecord]) -> Self {
        let time_range = Self::calculate_time_range(spans);
        Self { spans, time_range, selected: None, config: GanttConfig::default(), block: None }
    }

    /// Calculate time range from spans
    fn calculate_time_range(spans: &[SpanRecord]) -> (u64, u64) {
        if spans.is_empty() {
            return (0, 1);
        }

        let min = spans.iter().map(|s| s.start_time_ns).min().unwrap_or(0);
        let max = spans.iter().map(|s| s.end_time_ns).max().unwrap_or(1);

        (min, max.max(min + 1))
    }

    /// Set time range manually
    pub fn time_range(mut self, start: u64, end: u64) -> Self {
        self.time_range = (start, end.max(start + 1));
        self
    }

    /// Set selected span
    pub fn selected(mut self, idx: Option<usize>) -> Self {
        self.selected = idx;
        self
    }

    /// Set configuration
    pub fn config(mut self, config: GanttConfig) -> Self {
        self.config = config;
        self
    }

    /// Set block for borders
    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    /// Get color for span kind
    fn span_color(kind: SpanKind, is_error: bool, is_critical: bool) -> Color {
        if is_error {
            return graph::OUTLIER;
        }

        if is_critical {
            return Color::Rgb(255, 220, 100); // Gold for critical path
        }

        match kind {
            SpanKind::Server => graph::SYSCALL_NET,
            SpanKind::Client => graph::SYSCALL_FILE,
            SpanKind::Internal => graph::SYSCALL_PROC,
            SpanKind::Producer => graph::SYSCALL_MEM,
            SpanKind::Consumer => graph::CLUSTER_1,
        }
    }

    /// Render a single span bar
    fn render_bar(buf: &mut Buffer, x: u16, y: u16, width: u16, color: Color, is_selected: bool) {
        if width == 0 {
            return;
        }

        let style = if is_selected {
            Style::default().fg(color).add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(color)
        };

        if width == 1 {
            buf[(x, y)].set_char(BAR_THIN).set_style(style);
        } else if width == 2 {
            buf[(x, y)].set_char(BAR_START).set_style(style);
            buf[(x + 1, y)].set_char(BAR_END).set_style(style);
        } else {
            buf[(x, y)].set_char(BAR_START).set_style(style);
            for i in 1..width - 1 {
                buf[(x + i, y)].set_char(BAR_MIDDLE).set_style(style);
            }
            buf[(x + width - 1, y)].set_char(BAR_END).set_style(style);
        }
    }

    /// Format duration for display
    fn format_duration(ns: u64) -> String {
        if ns < 1_000 {
            format!("{}ns", ns)
        } else if ns < 1_000_000 {
            format!("{:.1}μs", ns as f64 / 1_000.0)
        } else if ns < 1_000_000_000 {
            format!("{:.2}ms", ns as f64 / 1_000_000.0)
        } else {
            format!("{:.2}s", ns as f64 / 1_000_000_000.0)
        }
    }
}

impl Widget for GanttChart<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Handle block
        let chart_area = if let Some(block) = &self.block {
            let inner = block.inner(area);
            block.clone().render(area, buf);
            inner
        } else {
            area
        };

        if chart_area.width < 10 || chart_area.height < 1 {
            return;
        }

        let (time_start, time_end) = self.time_range;
        let time_range = time_end - time_start;

        // Calculate bar area (leave room for labels)
        let label_width = if self.config.show_names { 15 } else { 0 };
        let duration_width = if self.config.show_durations { 10 } else { 0 };
        let bar_area_start = chart_area.x + label_width;
        let bar_area_width = chart_area.width.saturating_sub(label_width + duration_width);

        if bar_area_width < 2 {
            return;
        }

        // Render each span
        for (idx, span) in self.spans.iter().enumerate() {
            if idx >= chart_area.height as usize {
                break;
            }

            let y = chart_area.y + idx as u16;
            let is_selected = self.selected == Some(idx);

            // Calculate indent
            let indent = (span.depth.min(self.config.max_depth) as u16) * self.config.indent;

            // Render name label
            if self.config.show_names && label_width > indent {
                let name_area = label_width - indent - 1;
                let name: String = span.name.chars().take(name_area as usize).collect();
                let style = Style::default().fg(Color::DarkGray);

                // Write indent spaces
                for i in 0..indent {
                    buf[(chart_area.x + i, y)].set_char(' ').set_style(style);
                }

                // Write name
                for (i, ch) in name.chars().enumerate() {
                    let x = chart_area.x + indent + i as u16;
                    if x < bar_area_start {
                        buf[(x, y)].set_char(ch).set_style(style);
                    }
                }
            }

            // Calculate bar position and width
            let span_start = span.start_time_ns.saturating_sub(time_start);
            let span_duration = span.duration_ns();

            let bar_x = ((span_start as f64 / time_range as f64) * bar_area_width as f64) as u16;
            let bar_width = ((span_duration as f64 / time_range as f64) * bar_area_width as f64)
                .max(self.config.min_bar_width as f64) as u16;

            let bar_x = bar_area_start + bar_x.min(bar_area_width.saturating_sub(1));
            let bar_width = bar_width.min(bar_area_width.saturating_sub(bar_x - bar_area_start));

            // Get bar color
            let color = Self::span_color(span.kind, span.is_error(), span.is_critical_path);

            // Render bar
            Self::render_bar(buf, bar_x, y, bar_width, color, is_selected);

            // Render duration
            if self.config.show_durations {
                let duration_str = Self::format_duration(span.duration_ns());
                let duration_x = chart_area.x + chart_area.width - duration_width;
                let style = Style::default().fg(Color::DarkGray);

                for (i, ch) in duration_str.chars().enumerate() {
                    let x = duration_x + i as u16;
                    if x < chart_area.x + chart_area.width {
                        buf[(x, y)].set_char(ch).set_style(style);
                    }
                }
            }
        }
    }
}

/// Render Gantt chart to frame
pub fn render(f: &mut ratatui::Frame, spans: &[SpanRecord], area: Rect, selected: Option<usize>) {
    let chart =
        GanttChart::new(spans).selected(selected).block(Block::default().borders(Borders::ALL));

    f.render_widget(chart, area);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_span(name: &str, start: u64, duration: u64) -> SpanRecord {
        SpanRecord::new("trace1", &format!("span_{}", name), name, start, start + duration)
    }

    #[test]
    fn test_gantt_config_default() {
        let config = GanttConfig::default();
        assert_eq!(config.min_bar_width, 1);
        assert_eq!(config.max_depth, 10);
        assert!(config.show_names);
        assert!(config.show_durations);
    }

    #[test]
    fn test_gantt_time_range_calculation() {
        let spans = vec![make_span("a", 100, 50), make_span("b", 50, 100), make_span("c", 200, 25)];

        let chart = GanttChart::new(&spans);
        assert_eq!(chart.time_range, (50, 225)); // min start to max end
    }

    #[test]
    fn test_gantt_time_range_empty() {
        let spans: Vec<SpanRecord> = vec![];
        let chart = GanttChart::new(&spans);
        assert_eq!(chart.time_range, (0, 1));
    }

    #[test]
    fn test_gantt_selected() {
        let spans = vec![make_span("a", 0, 100)];
        let chart = GanttChart::new(&spans).selected(Some(0));
        assert_eq!(chart.selected, Some(0));
    }

    #[test]
    fn test_format_duration_ns() {
        assert_eq!(GanttChart::format_duration(500), "500ns");
        assert_eq!(GanttChart::format_duration(999), "999ns");
    }

    #[test]
    fn test_format_duration_us() {
        assert_eq!(GanttChart::format_duration(1_000), "1.0μs");
        assert_eq!(GanttChart::format_duration(1_500), "1.5μs");
        assert_eq!(GanttChart::format_duration(999_000), "999.0μs");
    }

    #[test]
    fn test_format_duration_ms() {
        assert_eq!(GanttChart::format_duration(1_000_000), "1.00ms");
        assert_eq!(GanttChart::format_duration(15_500_000), "15.50ms");
    }

    #[test]
    fn test_format_duration_s() {
        assert_eq!(GanttChart::format_duration(1_000_000_000), "1.00s");
        assert_eq!(GanttChart::format_duration(2_500_000_000), "2.50s");
    }

    #[test]
    fn test_span_color_error() {
        let color = GanttChart::span_color(SpanKind::Server, true, false);
        assert_eq!(color, graph::OUTLIER);
    }

    #[test]
    fn test_span_color_critical() {
        let color = GanttChart::span_color(SpanKind::Internal, false, true);
        assert_eq!(color, Color::Rgb(255, 220, 100));
    }

    #[test]
    fn test_span_color_by_kind() {
        assert_eq!(GanttChart::span_color(SpanKind::Server, false, false), graph::SYSCALL_NET);
        assert_eq!(GanttChart::span_color(SpanKind::Client, false, false), graph::SYSCALL_FILE);
        assert_eq!(GanttChart::span_color(SpanKind::Internal, false, false), graph::SYSCALL_PROC);
    }

    #[test]
    fn test_gantt_render_empty() {
        let spans: Vec<SpanRecord> = vec![];
        let mut buf = Buffer::empty(Rect::new(0, 0, 80, 10));
        let chart = GanttChart::new(&spans);
        chart.render(Rect::new(0, 0, 80, 10), &mut buf);
        // Should not panic
    }

    #[test]
    fn test_gantt_render_single_span() {
        let spans = vec![make_span("test", 0, 1000)];
        let mut buf = Buffer::empty(Rect::new(0, 0, 80, 5));
        let chart = GanttChart::new(&spans);
        chart.render(Rect::new(0, 0, 80, 5), &mut buf);

        // Should have some content
        let content: String =
            (0..80).map(|x| buf[(x, 0)].symbol().chars().next().unwrap_or(' ')).collect();
        assert!(!content.trim().is_empty());
    }

    #[test]
    fn test_gantt_render_with_depth() {
        let mut spans = vec![make_span("root", 0, 1000), make_span("child", 100, 500)];
        spans[1].depth = 1;

        let mut buf = Buffer::empty(Rect::new(0, 0, 80, 5));
        let chart = GanttChart::new(&spans);
        chart.render(Rect::new(0, 0, 80, 5), &mut buf);
        // Should render without panic
    }

    #[test]
    fn test_gantt_small_area() {
        let spans = vec![make_span("test", 0, 1000)];
        let mut buf = Buffer::empty(Rect::new(0, 0, 5, 1));
        let chart = GanttChart::new(&spans);
        chart.render(Rect::new(0, 0, 5, 1), &mut buf);
        // Should handle small area gracefully
    }

    #[test]
    fn test_span_color_producer_consumer() {
        assert_eq!(GanttChart::span_color(SpanKind::Producer, false, false), graph::SYSCALL_MEM);
        assert_eq!(GanttChart::span_color(SpanKind::Consumer, false, false), graph::CLUSTER_1);
    }

    #[test]
    fn test_render_bar_various_widths() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 20, 1));

        // Width 0 - should do nothing
        GanttChart::render_bar(&mut buf, 0, 0, 0, Color::Red, false);

        // Width 1 - thin bar
        GanttChart::render_bar(&mut buf, 1, 0, 1, Color::Green, false);

        // Width 2 - start and end
        GanttChart::render_bar(&mut buf, 3, 0, 2, Color::Blue, false);

        // Width 3+ - start, middle, end
        GanttChart::render_bar(&mut buf, 6, 0, 5, Color::Yellow, false);

        // Selected bar
        GanttChart::render_bar(&mut buf, 12, 0, 3, Color::Cyan, true);
    }

    #[test]
    fn test_gantt_with_block() {
        let spans = vec![make_span("test", 0, 1000)];
        let mut buf = Buffer::empty(Rect::new(0, 0, 80, 10));
        let chart = GanttChart::new(&spans).block(Block::default().borders(Borders::ALL));
        chart.render(Rect::new(0, 0, 80, 10), &mut buf);
    }

    #[test]
    fn test_gantt_with_custom_config() {
        let spans = vec![make_span("test", 0, 1000)];
        let mut buf = Buffer::empty(Rect::new(0, 0, 80, 10));
        let config = GanttConfig {
            min_bar_width: 2,
            max_depth: 5,
            indent: 4,
            show_names: false,
            show_durations: false,
        };
        let chart = GanttChart::new(&spans).config(config);
        chart.render(Rect::new(0, 0, 80, 10), &mut buf);
    }

    #[test]
    fn test_gantt_config_debug() {
        let config = GanttConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("min_bar_width"));
    }

    #[test]
    fn test_gantt_config_clone() {
        let config = GanttConfig {
            min_bar_width: 5,
            max_depth: 8,
            indent: 3,
            show_names: false,
            show_durations: true,
        };
        let cloned = config.clone();
        assert_eq!(cloned.min_bar_width, 5);
        assert_eq!(cloned.max_depth, 8);
    }
}
