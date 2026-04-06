//! TUI Dashboard for RAG Oracle
//!
//! Interactive terminal UI for visualizing index health, query results,
//! and system metrics. Implements Toyota Way Principle 7: Visual Control.
//!
//! ## Architecture (PROBAR-SPEC-009)
//!
//! Migrated from ratatui to presentar-terminal for stack consistency.
//! Uses Brick Architecture with Jidoka verification gates.

#[cfg(feature = "presentar-terminal")]
use std::collections::VecDeque;
#[cfg(feature = "presentar-terminal")]
use std::io::{self, Write};
#[cfg(feature = "presentar-terminal")]
use std::time::Duration;

#[cfg(feature = "presentar-terminal")]
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

#[cfg(feature = "presentar-terminal")]
use presentar_terminal::{CellBuffer, Color, DiffRenderer, Modifiers};

#[cfg(feature = "presentar-terminal")]
use super::types::{IndexHealthMetrics, RelevanceMetrics};

/// CYAN color constant (not in presentar-terminal)
#[cfg(feature = "presentar-terminal")]
const CYAN: Color = Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };

/// Query record for history display
#[derive(Debug, Clone)]
pub struct QueryRecord {
    /// Query timestamp (Unix epoch ms)
    pub timestamp_ms: u64,
    /// Query text (truncated)
    pub query: String,
    /// Primary component matched
    pub component: String,
    /// Query latency (ms)
    pub latency_ms: u64,
    /// Success flag
    pub success: bool,
}

/// TUI Dashboard state
#[cfg(feature = "presentar-terminal")]
pub struct OracleDashboard {
    /// Index health metrics
    pub index_health: IndexHealthMetrics,
    /// Query history (most recent first)
    pub query_history: VecDeque<QueryRecord>,
    /// Latency samples for sparkline
    pub latency_samples: Vec<u64>,
    /// Retrieval quality metrics
    pub retrieval_metrics: RelevanceMetrics,
    /// Selected component index
    selected_component: usize,
    /// Max history size
    max_history: usize,
    /// Refresh interval
    refresh_interval: Duration,
    /// Cell buffer for rendering
    buffer: CellBuffer,
    /// Diff renderer for efficient updates
    renderer: DiffRenderer,
    /// Terminal width
    width: u16,
    /// Terminal height
    height: u16,
}

#[cfg(feature = "presentar-terminal")]
impl OracleDashboard {
    /// Create a new dashboard
    pub fn new() -> Self {
        let (width, height) = crossterm::terminal::size().unwrap_or((100, 30));
        Self {
            index_health: IndexHealthMetrics::default(),
            query_history: VecDeque::new(),
            latency_samples: Vec::new(),
            retrieval_metrics: RelevanceMetrics::default(),
            selected_component: 0,
            max_history: 100,
            refresh_interval: Duration::from_millis(100),
            buffer: CellBuffer::new(width, height),
            renderer: DiffRenderer::new(),
            width,
            height,
        }
    }

    /// Add a query to history
    pub fn record_query(&mut self, record: QueryRecord) {
        self.latency_samples.push(record.latency_ms);
        if self.latency_samples.len() > 50 {
            self.latency_samples.remove(0);
        }
        self.query_history.push_front(record);
        if self.query_history.len() > self.max_history {
            self.query_history.pop_back();
        }
    }

    /// Update index health metrics
    pub fn update_health(&mut self, health: IndexHealthMetrics) {
        self.index_health = health;
    }

    /// Run the TUI dashboard
    pub fn run(&mut self) -> anyhow::Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, cursor::Hide)?;

        let result = self.run_loop(&mut stdout);

        disable_raw_mode()?;
        execute!(stdout, LeaveAlternateScreen, cursor::Show)?;

        result
    }

    /// Main event loop
    fn run_loop(&mut self, stdout: &mut io::Stdout) -> anyhow::Result<()> {
        loop {
            // Update terminal size
            let (w, h) = crossterm::terminal::size().unwrap_or((100, 30));
            if w != self.width || h != self.height {
                self.width = w;
                self.height = h;
                self.buffer.resize(w, h);
                self.renderer.reset();
            }

            // Clear and render
            self.buffer.clear();
            self.render();

            // Flush to terminal
            self.renderer.flush(&mut self.buffer, stdout)?;
            stdout.flush()?;

            if event::poll(self.refresh_interval)? {
                let event = event::read()?;
                let Event::Key(key) = event else { continue };
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Up | KeyCode::Char('k') => {
                        if self.selected_component > 0 {
                            self.selected_component -= 1;
                        }
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        let max = self.index_health.docs_per_component.len().saturating_sub(1);
                        if self.selected_component < max {
                            self.selected_component += 1;
                        }
                    }
                    KeyCode::Char('r') => {
                        // Trigger refresh - placeholder
                    }
                    _ => {}
                }
            }
        }
    }

    /// Render the dashboard
    fn render(&mut self) {
        let w = self.width;
        let h = self.height;

        // Layout: Header(3) | Panels(12+) | History(8) | Help(1)
        let header_h: u16 = 3;
        let help_h: u16 = 1;
        let history_h: u16 = 8;
        let panels_h = h.saturating_sub(header_h + history_h + help_h);

        self.render_header(0, 0, w, header_h);
        self.render_panels(0, header_h, w, panels_h);
        self.render_history(0, header_h + panels_h, w, history_h);
        self.render_help(0, h.saturating_sub(help_h), w, help_h);
    }

    /// Write a string with color
    fn write_str(&mut self, x: u16, y: u16, s: &str, fg: Color) {
        let mut cx = x;
        for ch in s.chars() {
            if cx >= self.width {
                break;
            }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer.update(cx, y, s, fg, Color::TRANSPARENT, Modifiers::NONE);
            cx = cx.saturating_add(1);
        }
    }

    /// Write a string clipped to panel content width (panel_w minus 2 for borders)
    fn write_str_clipped(&mut self, x: u16, y: u16, s: &str, panel_w: u16, fg: Color) {
        let max_len = panel_w.saturating_sub(2) as usize;
        self.write_str(x, y, &s[..s.len().min(max_len)], fg);
    }

    /// Set a single character with color
    fn set_char(&mut self, x: u16, y: u16, ch: char, fg: Color) {
        if x < self.width && y < self.height {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buffer.update(x, y, s, fg, Color::TRANSPARENT, Modifiers::NONE);
        }
    }

    /// Render header with overall health
    fn render_header(&mut self, x: u16, y: u16, w: u16, _h: u16) {
        let coverage = self.index_health.coverage_percent;
        let total_docs: usize = self.index_health.docs_per_component.iter().map(|(_, c)| c).sum();

        // Draw border
        self.draw_box(x, y, w, 3, " Oracle RAG Dashboard ");

        // Draw gauge bar inside
        let bar_width = w.saturating_sub(4) as usize;
        let filled = ((coverage as usize) * bar_width / 100).min(bar_width);
        let color = self.health_color(coverage);

        let label = format!("Index Health: {}%  |  Docs: {}", coverage, total_docs);

        // Draw progress bar
        let bar = format_bar_segments(filled, bar_width);
        self.write_str(x + 2, y + 1, &bar[..bar_width.min(bar.len())], color);

        // Center the label
        let label_x = x + 2 + ((bar_width.saturating_sub(label.len())) / 2) as u16;
        self.write_str(label_x, y + 1, &label, color);
    }

    /// Render main panels (index status, latency, quality)
    fn render_panels(&mut self, x: u16, y: u16, w: u16, h: u16) {
        let panel_w = w / 3;

        self.render_index_status(x, y, panel_w, h);
        self.render_latency(x + panel_w, y, panel_w, h);
        self.render_quality(x + 2 * panel_w, y, w.saturating_sub(2 * panel_w), h);
    }

    /// Render index status by component
    fn render_index_status(&mut self, x: u16, y: u16, w: u16, h: u16) {
        self.draw_box(x, y, w, h, " Index Status ");

        let content_y = y + 1;
        let content_h = h.saturating_sub(2) as usize;

        // Collect data first to avoid borrow conflict
        let rows: Vec<_> = self
            .index_health
            .docs_per_component
            .iter()
            .take(content_h)
            .enumerate()
            .map(|(i, (name, count))| {
                let bar = render_bar(*count, 500, 15);
                let (marker, color) = if i == self.selected_component {
                    (">", Color::YELLOW)
                } else {
                    (" ", Color::WHITE)
                };
                let line = format!("{} {:12} {} {}", marker, name, bar, count);
                (i, line, color)
            })
            .collect();

        for (i, line, color) in rows {
            self.write_str_clipped(x + 1, content_y + i as u16, &line, w, color);
        }
    }

    /// Render latency sparkline
    fn render_latency(&mut self, x: u16, y: u16, w: u16, h: u16) {
        self.draw_box(x, y, w, h, " Query Latency ");

        // Draw sparkline - collect points first to avoid borrow conflict
        let sparkline_h = h.saturating_sub(4) as usize;
        let spark_w = w.saturating_sub(2) as usize;

        let points: Vec<(u16, u16)> = if !self.latency_samples.is_empty() && sparkline_h > 0 {
            let max_val = *self.latency_samples.iter().max().unwrap_or(&1);
            self.latency_samples
                .iter()
                .rev()
                .take(spark_w)
                .enumerate()
                .flat_map(|(i, &val)| {
                    let bar_h = if max_val > 0 {
                        ((val as usize) * sparkline_h / (max_val as usize)).min(sparkline_h)
                    } else {
                        0
                    };
                    (0..bar_h).filter_map(move |j| {
                        let cy = y + 1 + (sparkline_h - 1 - j) as u16;
                        let cx = x + 1 + i as u16;
                        if cx < x + w - 1 {
                            Some((cx, cy))
                        } else {
                            None
                        }
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        for (cx, cy) in points {
            self.set_char(cx, cy, '▄', CYAN);
        }

        // Stats
        let (avg, p99) = if !self.latency_samples.is_empty() {
            let sum: u64 = self.latency_samples.iter().sum();
            let avg = sum / self.latency_samples.len() as u64;
            let mut sorted = self.latency_samples.clone();
            sorted.sort();
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;
            let p99 = sorted.get(p99_idx.min(sorted.len() - 1)).copied().unwrap_or(0);
            (avg, p99)
        } else {
            (0, 0)
        };

        let stats = format!("avg: {}ms  p99: {}ms", avg, p99);
        self.write_str(x + 1, y + h - 2, &stats, Color::WHITE);
    }

    /// Render quality metrics
    fn render_quality(&mut self, x: u16, y: u16, w: u16, h: u16) {
        self.draw_box(x, y, w, h, " Retrieval Quality ");

        let metrics = &self.retrieval_metrics;
        let content_y = y + 1;

        let rows =
            [("MRR", metrics.mrr), ("NDCG", metrics.ndcg_at_k), ("R@10", metrics.recall_at_k)];

        for (i, (label, value)) in rows.iter().enumerate() {
            let bar = render_bar((*value * 100.0) as usize, 100, 12);
            let line = format!("{:5} {:.3} {}", label, value, bar);
            self.write_str_clipped(x + 1, content_y + i as u16, &line, w, Color::WHITE);
        }
    }

    /// Render query history
    fn render_history(&mut self, x: u16, y: u16, w: u16, h: u16) {
        self.draw_box(x, y, w, h, " Recent Queries ");

        // Header
        let header = "Time       Query                          Component    Latency";
        self.write_str_clipped(x + 1, y + 1, header, w, Color::YELLOW);

        // Collect data first to avoid borrow conflict
        let rows: Vec<_> = self
            .query_history
            .iter()
            .take(h.saturating_sub(3) as usize)
            .enumerate()
            .map(|(i, record)| {
                let time = format_timestamp(record.timestamp_ms);
                let (status_char, color) =
                    if record.success { ('+', Color::GREEN) } else { ('x', Color::RED) };
                let line = format!(
                    "{} {:30} {:12} {:>6}ms {}",
                    time,
                    truncate_query(&record.query, 30),
                    record.component,
                    record.latency_ms,
                    status_char
                );
                (i, line, color)
            })
            .collect();

        let content_y = y + 2;
        for (i, line, color) in rows {
            self.write_str_clipped(x + 1, content_y + i as u16, &line, w, color);
        }
    }

    /// Render help bar
    fn render_help(&mut self, x: u16, y: u16, w: u16, _h: u16) {
        let help = " [q]uit  [r]efresh  [↑/↓]navigate ";
        let gray = Color::new(0.5, 0.5, 0.5, 1.0);
        self.write_str(x, y, &help[..help.len().min(w as usize)], gray);
    }

    /// Draw a box with border and title
    fn draw_box(&mut self, x: u16, y: u16, w: u16, h: u16, title: &str) {
        if w < 2 || h < 2 {
            return;
        }

        // Top border
        self.set_char(x, y, '┌', Color::WHITE);
        for i in 1..w - 1 {
            self.set_char(x + i, y, '─', Color::WHITE);
        }
        self.set_char(x + w - 1, y, '┐', Color::WHITE);

        // Title
        if !title.is_empty() && w > title.len() as u16 + 2 {
            let title_x = x + 2;
            self.write_str(title_x, y, title, CYAN);
        }

        // Sides
        for i in 1..h - 1 {
            self.set_char(x, y + i, '│', Color::WHITE);
            self.set_char(x + w - 1, y + i, '│', Color::WHITE);
        }

        // Bottom border
        self.set_char(x, y + h - 1, '└', Color::WHITE);
        for i in 1..w - 1 {
            self.set_char(x + i, y + h - 1, '─', Color::WHITE);
        }
        self.set_char(x + w - 1, y + h - 1, '┘', Color::WHITE);
    }

    /// Get color based on health percentage
    fn health_color(&self, percent: u16) -> Color {
        match percent {
            0..=60 => Color::RED,
            61..=80 => Color::YELLOW,
            _ => Color::GREEN,
        }
    }
}

#[cfg(feature = "presentar-terminal")]
impl Default for OracleDashboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a bar string from pre-computed filled count and total width.
fn format_bar_segments(filled: usize, width: usize) -> String {
    let clamped = filled.min(width);
    let empty = width.saturating_sub(clamped);
    format!("{}{}", "\u{2588}".repeat(clamped), "\u{2591}".repeat(empty))
}

/// Render a horizontal bar
fn render_bar(value: usize, max: usize, width: usize) -> String {
    let filled = if max > 0 { (value * width / max).min(width) } else { 0 };
    format_bar_segments(filled, width)
}

/// Format timestamp for display
fn format_timestamp(timestamp_ms: u64) -> String {
    let secs = timestamp_ms / 1000;
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let secs = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}

/// Truncate query for display
fn truncate_query(query: &str, max_len: usize) -> String {
    if query.len() <= max_len {
        query.to_string()
    } else {
        format!("{}...", &query[..max_len - 3])
    }
}

/// Inline visualizations for CLI output
pub mod inline {
    use super::format_bar_segments;

    /// Render a horizontal bar chart
    pub fn bar(value: f64, max: f64, width: usize) -> String {
        let filled = if max > 0.0 { ((value / max) * width as f64) as usize } else { 0 };
        format_bar_segments(filled, width)
    }

    /// Render a sparkline from values
    pub fn sparkline(values: &[f64]) -> String {
        const BARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if values.is_empty() {
            return String::new();
        }

        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let range = max - min;

        values
            .iter()
            .map(|v| {
                let idx = if range == 0.0 { 0 } else { ((v - min) / range * 7.0) as usize };
                BARS[idx.min(7)]
            })
            .collect()
    }

    /// Format a score as a bar with percentage
    pub fn score_bar(score: f64, width: usize) -> String {
        let pct = (score * 100.0) as usize;
        format!("{} {:3}%", bar(score, 1.0, width), pct)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a `QueryRecord` with defaults, overriding specific fields.
    fn make_record(query: &str, latency_ms: u64, success: bool) -> QueryRecord {
        QueryRecord {
            timestamp_ms: 0,
            query: query.to_string(),
            component: "test".to_string(),
            latency_ms,
            success,
        }
    }

    /// Helper: count occurrences of a character in a bar string.
    fn count_bar_char(bar: &str, ch: char) -> usize {
        bar.chars().filter(|c| *c == ch).count()
    }

    #[test]
    fn test_render_bar() {
        let bar = render_bar(50, 100, 10);
        assert_eq!(count_bar_char(&bar, '\u{2588}'), 5);
        assert_eq!(count_bar_char(&bar, '\u{2591}'), 5);
    }

    #[test]
    fn test_render_bar_full() {
        let bar = render_bar(100, 100, 10);
        assert_eq!(count_bar_char(&bar, '\u{2588}'), 10);
    }

    #[test]
    fn test_render_bar_empty() {
        let bar = render_bar(0, 100, 10);
        assert_eq!(count_bar_char(&bar, '\u{2591}'), 10);
    }

    #[test]
    fn test_format_timestamp() {
        let ts = format_timestamp(45296000); // 12:34:56
        assert_eq!(ts, "12:34:56");
    }

    #[test]
    fn test_truncate_query_short() {
        let q = truncate_query("short", 10);
        assert_eq!(q, "short");
    }

    #[test]
    fn test_truncate_query_long() {
        let q = truncate_query("this is a very long query", 15);
        assert!(q.ends_with("..."));
        assert!(q.len() <= 15);
    }

    #[test]
    fn test_inline_bar() {
        let bar = inline::bar(0.5, 1.0, 10);
        assert_eq!(count_bar_char(&bar, '\u{2588}'), 5);
    }

    #[test]
    fn test_inline_sparkline() {
        let spark = inline::sparkline(&[0.0, 0.5, 1.0, 0.5, 0.0]);
        assert_eq!(spark.chars().count(), 5);
        assert!(spark.contains('\u{2581}'));
        assert!(spark.contains('\u{2588}'));
    }

    #[test]
    fn test_inline_sparkline_empty() {
        let spark = inline::sparkline(&[]);
        assert!(spark.is_empty());
    }

    #[test]
    fn test_inline_score_bar() {
        let bar = inline::score_bar(0.85, 10);
        assert!(bar.contains("85%"));
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_dashboard_creation() {
        let dashboard = OracleDashboard::new();
        assert!(dashboard.query_history.is_empty());
        assert!(dashboard.latency_samples.is_empty());
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_dashboard_record_query() {
        let mut dashboard = OracleDashboard::new();
        let mut record = make_record("test query", 50, true);
        record.timestamp_ms = 1234567890;
        record.component = "trueno".to_string();
        dashboard.record_query(record);

        assert_eq!(dashboard.query_history.len(), 1);
        assert_eq!(dashboard.latency_samples.len(), 1);
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_dashboard_default() {
        let dashboard = OracleDashboard::default();
        assert!(dashboard.query_history.is_empty());
        assert_eq!(dashboard.selected_component, 0);
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_dashboard_update_health() {
        let mut dashboard = OracleDashboard::new();
        let health = IndexHealthMetrics {
            coverage_percent: 85,
            docs_per_component: vec![("trueno".to_string(), 100)],
            component_names: vec!["trueno".to_string()],
            latency_samples: vec![10, 20, 30],
            mrr_history: vec![0.8, 0.85],
            ndcg_history: vec![0.9, 0.92],
            freshness_score: 95.0,
        };

        dashboard.update_health(health);
        assert_eq!(dashboard.index_health.coverage_percent, 85);
        assert_eq!(dashboard.index_health.docs_per_component.len(), 1);
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_dashboard_latency_samples_bounded() {
        let mut dashboard = OracleDashboard::new();

        for i in 0..60 {
            let mut record = make_record(&format!("query {}", i), i as u64 * 10, true);
            record.timestamp_ms = i as u64;
            dashboard.record_query(record);
        }

        assert_eq!(dashboard.latency_samples.len(), 50);
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_dashboard_query_history_bounded() {
        let mut dashboard = OracleDashboard::new();

        for i in 0..110 {
            let mut record = make_record(&format!("query {}", i), 10, i % 2 == 0);
            record.timestamp_ms = i as u64;
            dashboard.record_query(record);
        }

        assert_eq!(dashboard.query_history.len(), 100);
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_dashboard_query_order() {
        let mut dashboard = OracleDashboard::new();

        let mut first = make_record("first", 10, true);
        first.timestamp_ms = 100;
        dashboard.record_query(first);

        let mut second = make_record("second", 20, true);
        second.timestamp_ms = 200;
        dashboard.record_query(second);

        assert_eq!(dashboard.query_history.front().expect("unexpected failure").query, "second");
        assert_eq!(dashboard.query_history.back().expect("unexpected failure").query, "first");
    }

    #[test]
    fn test_render_bar_overflow() {
        let bar = render_bar(200, 100, 10);
        assert_eq!(count_bar_char(&bar, '\u{2588}'), 10);
    }

    #[test]
    fn test_render_bar_zero_max() {
        let bar = render_bar(50, 0, 10);
        assert_eq!(count_bar_char(&bar, '\u{2591}'), 10);
    }

    #[test]
    fn test_format_timestamp_edge() {
        assert_eq!(format_timestamp(0), "00:00:00");
        assert_eq!(format_timestamp(86399000), "23:59:59");
    }

    #[test]
    fn test_truncate_query_exact() {
        let q = truncate_query("exactly_ten", 10);
        assert!(q.len() <= 10);
    }

    #[test]
    fn test_truncate_query_unicode() {
        let q = truncate_query("hello world test", 10);
        assert!(q.len() <= 10);
    }

    #[test]
    fn test_inline_bar_zero() {
        let bar = inline::bar(0.0, 1.0, 10);
        assert_eq!(count_bar_char(&bar, '\u{2591}'), 10);
    }

    #[test]
    fn test_inline_bar_full() {
        let bar = inline::bar(1.0, 1.0, 10);
        assert_eq!(count_bar_char(&bar, '\u{2588}'), 10);
    }

    #[test]
    fn test_inline_bar_zero_max() {
        let bar = inline::bar(0.5, 0.0, 10);
        assert_eq!(count_bar_char(&bar, '\u{2591}'), 10);
    }

    #[test]
    fn test_inline_sparkline_constant() {
        let spark = inline::sparkline(&[5.0, 5.0, 5.0]);
        assert_eq!(spark.chars().count(), 3);
        let chars: Vec<char> = spark.chars().collect();
        assert_eq!(chars[0], chars[1]);
        assert_eq!(chars[1], chars[2]);
    }

    #[test]
    fn test_inline_sparkline_single() {
        let spark = inline::sparkline(&[1.0]);
        assert_eq!(spark.chars().count(), 1);
    }

    #[test]
    fn test_inline_score_bar_zero() {
        let bar = inline::score_bar(0.0, 10);
        assert!(bar.contains("0%"));
    }

    #[test]
    fn test_inline_score_bar_full() {
        let bar = inline::score_bar(1.0, 10);
        assert!(bar.contains("100%"));
    }

    #[test]
    fn test_query_record_fields() {
        let mut record = make_record("test", 50, false);
        record.timestamp_ms = 1000;
        record.component = "comp".to_string();

        assert_eq!(record.timestamp_ms, 1000);
        assert_eq!(record.query, "test");
        assert_eq!(record.component, "comp");
        assert_eq!(record.latency_ms, 50);
        assert!(!record.success);
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_health_color_red() {
        let dashboard = OracleDashboard::new();
        let color = dashboard.health_color(50);
        assert_eq!(color, Color::RED);
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_health_color_yellow() {
        let dashboard = OracleDashboard::new();
        let color = dashboard.health_color(75);
        assert_eq!(color, Color::YELLOW);
    }

    #[cfg(feature = "presentar-terminal")]
    #[test]
    fn test_health_color_green() {
        let dashboard = OracleDashboard::new();
        let color = dashboard.health_color(90);
        assert_eq!(color, Color::GREEN);
    }
}
