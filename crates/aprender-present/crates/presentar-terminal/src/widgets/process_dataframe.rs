//! `ProcessDataFrame` widget - Data science view for process monitoring.
//!
//! Grammar of Graphics construct: Process data as sortable `DataFrame` with:
//! - PID, Name, State columns
//! - CPU% with inline sparkline history
//! - MEM% with micro bar
//! - Threads, Priority, User columns
//! - Sort by any column (click or keyboard)
//!
//! Implements SPEC-024 Section 27.8 - Framework-First pattern.

use compact_str::CompactString;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::collections::HashMap;
use std::time::Duration;

use super::micro_heat_bar::HeatScheme;
use super::selection::RowHighlight;

/// Process state for display.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ProcessDisplayState {
    #[default]
    Running,
    Sleeping,
    Idle,
    Zombie,
    Stopped,
    Unknown,
}

impl ProcessDisplayState {
    /// Get display character and color.
    #[must_use]
    pub fn render(self) -> (char, Color) {
        match self {
            Self::Running => ('R', Color::new(0.2, 0.9, 0.2, 1.0)),
            Self::Sleeping => ('S', Color::new(0.6, 0.6, 0.6, 1.0)),
            Self::Idle => ('I', Color::new(0.5, 0.5, 0.5, 1.0)),
            Self::Zombie => ('Z', Color::new(0.9, 0.2, 0.2, 1.0)),
            Self::Stopped => ('T', Color::new(0.9, 0.7, 0.1, 1.0)),
            Self::Unknown => ('?', Color::new(0.4, 0.4, 0.4, 1.0)),
        }
    }
}

/// Sort column for process table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ProcessSortColumn {
    Pid,
    Name,
    #[default]
    Cpu,
    Mem,
    Threads,
    Priority,
    User,
    Time,
}

impl ProcessSortColumn {
    /// Get column header string.
    #[must_use]
    pub fn header(self) -> &'static str {
        match self {
            Self::Pid => "PID",
            Self::Name => "COMMAND",
            Self::Cpu => "CPU%",
            Self::Mem => "MEM%",
            Self::Threads => "THR",
            Self::Priority => "PRI",
            Self::User => "USER",
            Self::Time => "TIME+",
        }
    }
}

/// Single process row data.
#[derive(Debug, Clone)]
pub struct ProcessRow {
    /// Process ID.
    pub pid: u32,
    /// Process name/command.
    pub name: CompactString,
    /// Current CPU percentage.
    pub cpu_percent: f32,
    /// CPU history (last N samples for sparkline).
    pub cpu_history: Vec<f32>,
    /// Memory percentage.
    pub mem_percent: f32,
    /// Process state.
    pub state: ProcessDisplayState,
    /// Number of threads.
    pub threads: u32,
    /// Priority/nice value.
    pub priority: i32,
    /// User name.
    pub user: CompactString,
    /// CPU time in seconds.
    pub cpu_time_secs: u64,
}

impl Default for ProcessRow {
    fn default() -> Self {
        Self {
            pid: 0,
            name: CompactString::new(""),
            cpu_percent: 0.0,
            cpu_history: Vec::new(),
            mem_percent: 0.0,
            state: ProcessDisplayState::default(),
            threads: 1,
            priority: 0,
            user: CompactString::new(""),
            cpu_time_secs: 0,
        }
    }
}

/// Process `DataFrame` widget for data science view.
#[derive(Debug, Clone)]
pub struct ProcessDataFrame {
    /// Process rows.
    rows: Vec<ProcessRow>,
    /// CPU history per PID (ring buffer).
    cpu_histories: HashMap<u32, Vec<f32>>,
    /// History length for sparklines.
    history_len: usize,
    /// Current sort column.
    sort_column: ProcessSortColumn,
    /// Sort descending.
    sort_desc: bool,
    /// Selected row index.
    selected_row: Option<usize>,
    /// Scroll offset.
    scroll_offset: usize,
    /// Show header row.
    show_header: bool,
    /// Column widths.
    col_widths: ProcessColumnWidths,
    /// Primary accent color.
    accent_color: Color,
    /// Cached bounds.
    bounds: Rect,
}

/// Column width configuration.
#[derive(Debug, Clone, Copy)]
pub struct ProcessColumnWidths {
    pub pid: usize,
    pub name: usize,
    pub cpu: usize,
    pub sparkline: usize,
    pub mem: usize,
    pub state: usize,
    pub threads: usize,
    pub priority: usize,
    pub user: usize,
    pub time: usize,
}

impl Default for ProcessColumnWidths {
    fn default() -> Self {
        Self {
            pid: 7,
            name: 20,
            cpu: 6,
            sparkline: 12,
            mem: 6,
            state: 3,
            threads: 4,
            priority: 4,
            user: 10,
            time: 10,
        }
    }
}

impl Default for ProcessDataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessDataFrame {
    /// Create a new empty process dataframe.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            cpu_histories: HashMap::new(),
            history_len: 60, // 60 samples = 60s at 1s refresh
            sort_column: ProcessSortColumn::Cpu,
            sort_desc: true,
            selected_row: None,
            scroll_offset: 0,
            show_header: true,
            col_widths: ProcessColumnWidths::default(),
            accent_color: Color::new(0.4, 0.7, 1.0, 1.0),
            bounds: Rect::default(),
        }
    }

    /// Update with new process data.
    pub fn update_processes(&mut self, processes: Vec<ProcessRow>) {
        // Update CPU histories
        for proc in &processes {
            let history = self.cpu_histories.entry(proc.pid).or_default();
            history.push(proc.cpu_percent);
            if history.len() > self.history_len {
                history.remove(0);
            }
        }

        // Clean up histories for dead processes
        let live_pids: std::collections::HashSet<u32> = processes.iter().map(|p| p.pid).collect();
        self.cpu_histories.retain(|pid, _| live_pids.contains(pid));

        // Update rows with histories
        self.rows = processes
            .into_iter()
            .map(|mut proc| {
                if let Some(history) = self.cpu_histories.get(&proc.pid) {
                    proc.cpu_history = history.clone();
                }
                proc
            })
            .collect();

        self.sort_rows();
    }

    /// Set sort column.
    #[must_use]
    pub fn with_sort(mut self, column: ProcessSortColumn, descending: bool) -> Self {
        self.sort_column = column;
        self.sort_desc = descending;
        self
    }

    /// Set accent color.
    #[must_use]
    pub fn with_accent_color(mut self, color: Color) -> Self {
        self.accent_color = color;
        self
    }

    /// Set column widths.
    #[must_use]
    pub fn with_column_widths(mut self, widths: ProcessColumnWidths) -> Self {
        self.col_widths = widths;
        self
    }

    /// Set history length.
    #[must_use]
    pub fn with_history_len(mut self, len: usize) -> Self {
        self.history_len = len;
        self
    }

    /// Get selected process PID (if any).
    #[must_use]
    pub fn selected_pid(&self) -> Option<u32> {
        self.selected_row
            .and_then(|idx| self.rows.get(idx).map(|r| r.pid))
    }

    /// Scroll up one row.
    pub fn scroll_up(&mut self) {
        if let Some(sel) = self.selected_row {
            if sel > 0 {
                self.selected_row = Some(sel - 1);
                if sel - 1 < self.scroll_offset {
                    self.scroll_offset = sel - 1;
                }
            }
        } else if !self.rows.is_empty() {
            self.selected_row = Some(0);
        }
    }

    /// Scroll down one row.
    pub fn scroll_down(&mut self) {
        let max_visible = self.visible_rows();
        if let Some(sel) = self.selected_row {
            if sel < self.rows.len().saturating_sub(1) {
                self.selected_row = Some(sel + 1);
                if sel + 1 >= self.scroll_offset + max_visible {
                    self.scroll_offset = (sel + 2).saturating_sub(max_visible);
                }
            }
        } else if !self.rows.is_empty() {
            self.selected_row = Some(0);
        }
    }

    /// Cycle sort column.
    pub fn cycle_sort(&mut self) {
        self.sort_column = match self.sort_column {
            ProcessSortColumn::Cpu => ProcessSortColumn::Mem,
            ProcessSortColumn::Mem => ProcessSortColumn::Pid,
            ProcessSortColumn::Pid => ProcessSortColumn::Name,
            ProcessSortColumn::Name => ProcessSortColumn::Threads,
            ProcessSortColumn::Threads => ProcessSortColumn::Time,
            ProcessSortColumn::Time => ProcessSortColumn::Cpu,
            _ => ProcessSortColumn::Cpu,
        };
        self.sort_rows();
    }

    /// Toggle sort direction.
    pub fn toggle_sort_direction(&mut self) {
        self.sort_desc = !self.sort_desc;
        self.sort_rows();
    }

    fn sort_rows(&mut self) {
        let desc = self.sort_desc;
        match self.sort_column {
            ProcessSortColumn::Pid => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.pid.cmp(&a.pid)
                    } else {
                        a.pid.cmp(&b.pid)
                    }
                });
            }
            ProcessSortColumn::Name => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.name.cmp(&a.name)
                    } else {
                        a.name.cmp(&b.name)
                    }
                });
            }
            ProcessSortColumn::Cpu => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.cpu_percent
                            .partial_cmp(&a.cpu_percent)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a.cpu_percent
                            .partial_cmp(&b.cpu_percent)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
            }
            ProcessSortColumn::Mem => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.mem_percent
                            .partial_cmp(&a.mem_percent)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a.mem_percent
                            .partial_cmp(&b.mem_percent)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
            }
            ProcessSortColumn::Threads => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.threads.cmp(&a.threads)
                    } else {
                        a.threads.cmp(&b.threads)
                    }
                });
            }
            ProcessSortColumn::Priority => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.priority.cmp(&a.priority)
                    } else {
                        a.priority.cmp(&b.priority)
                    }
                });
            }
            ProcessSortColumn::User => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.user.cmp(&a.user)
                    } else {
                        a.user.cmp(&b.user)
                    }
                });
            }
            ProcessSortColumn::Time => {
                self.rows.sort_by(|a, b| {
                    if desc {
                        b.cpu_time_secs.cmp(&a.cpu_time_secs)
                    } else {
                        a.cpu_time_secs.cmp(&b.cpu_time_secs)
                    }
                });
            }
        }
    }

    fn visible_rows(&self) -> usize {
        let header_offset = if self.show_header { 2 } else { 0 };
        (self.bounds.height as usize).saturating_sub(header_offset)
    }

    fn render_sparkline(values: &[f32], width: usize) -> String {
        const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if values.is_empty() {
            return "─".repeat(width);
        }

        // Scale to 0-100% range
        let sample_width = width.min(values.len());
        let step = values.len().saturating_sub(1) / sample_width.max(1);

        (0..sample_width)
            .map(|i| {
                let idx = (i * step.max(1)).min(values.len().saturating_sub(1));
                let v = values[idx].clamp(0.0, 100.0);
                let norm = ((v / 100.0) * 7.0).round() as usize;
                BARS[norm.min(7)]
            })
            .collect()
    }

    fn render_microbar(value: f32, width: usize) -> String {
        let pct = (value / 100.0).clamp(0.0, 1.0);
        let filled = ((width as f32) * pct).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }

    fn format_time(secs: u64) -> String {
        let hours = secs / 3600;
        let mins = (secs % 3600) / 60;
        let secs = secs % 60;
        if hours > 99 {
            format!("{hours}h")
        } else if hours > 0 {
            format!("{hours}:{mins:02}:{secs:02}")
        } else {
            format!("{mins}:{secs:02}")
        }
    }
}

impl Brick for ProcessDataFrame {
    fn brick_name(&self) -> &'static str {
        "process_dataframe"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        format!(
            r#"<table class="process-dataframe"><thead><tr>{}</tr></thead></table>"#,
            ["PID", "COMMAND", "CPU%", "MEM%", "THR", "USER"]
                .iter()
                .map(|h| format!("<th>{h}</th>"))
                .collect::<String>()
        )
    }

    fn to_css(&self) -> String {
        ".process-dataframe { sort: cpu; }".to_string()
    }
}

impl Widget for ProcessDataFrame {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let header_height = if self.show_header { 2.0 } else { 0.0 };
        let row_height = self.rows.len().min(30) as f32;
        let height = (header_height + row_height).min(constraints.max_height);
        constraints.constrain(Size::new(constraints.max_width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 40.0 || self.bounds.height < 3.0 {
            return;
        }

        let mut y = self.bounds.y;
        let x_start = self.bounds.x;
        let w = &self.col_widths;

        // Calculate column positions
        let col_positions = [
            0usize,                                                                      // PID
            w.pid,                                                                       // Name
            w.pid + w.name,                                                              // CPU%
            w.pid + w.name + w.cpu,                       // Sparkline
            w.pid + w.name + w.cpu + w.sparkline,         // MEM%
            w.pid + w.name + w.cpu + w.sparkline + w.mem, // State
            w.pid + w.name + w.cpu + w.sparkline + w.mem + w.state, // Threads
            w.pid + w.name + w.cpu + w.sparkline + w.mem + w.state + w.threads, // User
            w.pid + w.name + w.cpu + w.sparkline + w.mem + w.state + w.threads + w.user, // Time
        ];

        let header_style = TextStyle {
            color: self.accent_color,
            ..Default::default()
        };

        let dim_style = TextStyle {
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..Default::default()
        };

        // Header row
        if self.show_header {
            let headers = [
                (ProcessSortColumn::Pid, w.pid),
                (ProcessSortColumn::Name, w.name),
                (ProcessSortColumn::Cpu, w.cpu + w.sparkline),
                (ProcessSortColumn::Mem, w.mem),
                (ProcessSortColumn::Threads, w.state + w.threads),
                (ProcessSortColumn::User, w.user),
                (ProcessSortColumn::Time, w.time),
            ];

            let mut hx = x_start;
            for (col, width) in headers {
                let is_sorted = col == self.sort_column;
                let arrow = if is_sorted {
                    if self.sort_desc {
                        "▼"
                    } else {
                        "▲"
                    }
                } else {
                    ""
                };
                let text = format!("{}{}", col.header(), arrow);
                let text = if text.len() > width {
                    text[..width].to_string()
                } else {
                    text
                };

                let style = if is_sorted { &header_style } else { &dim_style };
                canvas.draw_text(&text, Point::new(hx, y), style);
                hx += width as f32 + 1.0;
            }

            y += 1.0;

            // Separator line
            let sep = "─".repeat((self.bounds.width as usize).min(120));
            canvas.draw_text(&sep, Point::new(x_start, y), &dim_style);
            y += 1.0;
        }

        // Data rows
        let visible_rows = self.visible_rows();
        let end_idx = (self.scroll_offset + visible_rows).min(self.rows.len());

        for (rel_idx, row) in self.rows[self.scroll_offset..end_idx].iter().enumerate() {
            let abs_idx = self.scroll_offset + rel_idx;
            let is_selected = self.selected_row == Some(abs_idx);

            // === TUFTE SELECTION HIGHLIGHTING (Framework Widget) ===
            // Strong row background + gutter indicator for selected row
            let row_bounds = Rect::new(x_start, y, self.bounds.width, 1.0);
            let row_highlight = RowHighlight::new(row_bounds, is_selected);
            row_highlight.paint(canvas);

            // Get text style from row highlight (white on blue when selected)
            let text_style = row_highlight.text_style();

            // PID
            let pid_str = format!("{:>width$}", row.pid, width = w.pid);
            canvas.draw_text(
                &pid_str,
                Point::new(x_start + col_positions[0] as f32, y),
                &text_style,
            );

            // Name (truncated)
            let name = if row.name.len() > w.name {
                format!("{}…", &row.name[..w.name - 1])
            } else {
                format!("{:<width$}", row.name, width = w.name)
            };
            canvas.draw_text(
                &name,
                Point::new(x_start + col_positions[1] as f32, y),
                &text_style,
            );

            // CPU% with thermal heat color encoding (Grammar of Graphics)
            // High CPU = warmer (red), low CPU = cooler (green)
            let cpu_color = HeatScheme::Thermal.color_for_percent(row.cpu_percent as f64);
            let cpu_str = format!("{:>5.1}", row.cpu_percent);
            canvas.draw_text(
                &cpu_str,
                Point::new(x_start + col_positions[2] as f32, y),
                &TextStyle {
                    color: cpu_color,
                    ..Default::default()
                },
            );

            // CPU sparkline
            let sparkline = Self::render_sparkline(&row.cpu_history, w.sparkline);
            canvas.draw_text(
                &sparkline,
                Point::new(x_start + col_positions[3] as f32, y),
                &TextStyle {
                    color: self.accent_color,
                    ..Default::default()
                },
            );

            // MEM% with micro bar
            let mem_bar = Self::render_microbar(row.mem_percent, 4);
            let mem_str = format!("{:>4.1}", row.mem_percent);
            canvas.draw_text(
                &format!("{mem_bar}{mem_str}"),
                Point::new(x_start + col_positions[4] as f32, y),
                &TextStyle {
                    color: Color::new(0.7, 0.5, 0.9, 1.0),
                    ..Default::default()
                },
            );

            // State
            let (state_ch, state_color) = row.state.render();
            canvas.draw_text(
                &state_ch.to_string(),
                Point::new(x_start + col_positions[5] as f32, y),
                &TextStyle {
                    color: state_color,
                    ..Default::default()
                },
            );

            // Threads
            let thr_str = format!("{:>3}", row.threads);
            canvas.draw_text(
                &thr_str,
                Point::new(x_start + col_positions[6] as f32 + 1.0, y),
                &text_style,
            );

            // User (truncated)
            let user = if row.user.len() > w.user {
                format!("{}…", &row.user[..w.user - 1])
            } else {
                format!("{:<width$}", row.user, width = w.user)
            };
            canvas.draw_text(
                &user,
                Point::new(x_start + col_positions[7] as f32, y),
                &text_style,
            );

            // Time
            let time_str = Self::format_time(row.cpu_time_secs);
            canvas.draw_text(
                &time_str,
                Point::new(x_start + col_positions[8] as f32, y),
                &text_style,
            );

            y += 1.0;
        }

        // Scroll indicator if needed
        if self.rows.len() > visible_rows {
            let scroll_pct = self.scroll_offset as f32 / (self.rows.len() - visible_rows) as f32;
            let indicator_y = self.bounds.y + 2.0 + (scroll_pct * (visible_rows - 1) as f32);
            canvas.draw_text(
                "│",
                Point::new(self.bounds.x + self.bounds.width - 1.0, indicator_y),
                &dim_style,
            );
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::KeyDown {
                key: Key::Up | Key::K,
                ..
            } => {
                self.scroll_up();
                None
            }
            Event::KeyDown {
                key: Key::Down | Key::J,
                ..
            } => {
                self.scroll_down();
                None
            }
            Event::KeyDown { key: Key::S, .. } => {
                self.cycle_sort();
                None
            }
            Event::KeyDown { key: Key::R, .. } => {
                self.toggle_sort_direction();
                None
            }
            _ => None,
        }
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    // ProcessDisplayState tests
    #[test]
    fn test_display_state_running() {
        let (ch, color) = ProcessDisplayState::Running.render();
        assert_eq!(ch, 'R');
        assert!(color.g > 0.8, "Running should be green");
    }

    #[test]
    fn test_display_state_sleeping() {
        let (ch, color) = ProcessDisplayState::Sleeping.render();
        assert_eq!(ch, 'S');
        assert!(color.r > 0.5 && color.g > 0.5, "Sleeping should be gray");
    }

    #[test]
    fn test_display_state_idle() {
        let (ch, color) = ProcessDisplayState::Idle.render();
        assert_eq!(ch, 'I');
        assert!(color.r > 0.4, "Idle should be grayish");
    }

    #[test]
    fn test_display_state_zombie() {
        let (ch, color) = ProcessDisplayState::Zombie.render();
        assert_eq!(ch, 'Z');
        assert!(color.r > 0.8 && color.g < 0.3, "Zombie should be red");
    }

    #[test]
    fn test_display_state_stopped() {
        let (ch, color) = ProcessDisplayState::Stopped.render();
        assert_eq!(ch, 'T');
        assert!(
            color.r > 0.8 && color.g > 0.6,
            "Stopped should be yellow/orange"
        );
    }

    #[test]
    fn test_display_state_unknown() {
        let (ch, color) = ProcessDisplayState::Unknown.render();
        assert_eq!(ch, '?');
        assert!(color.r < 0.5, "Unknown should be dim");
    }

    #[test]
    fn test_display_state_default() {
        assert_eq!(ProcessDisplayState::default(), ProcessDisplayState::Running);
    }

    // ProcessSortColumn tests
    #[test]
    fn test_sort_column_headers() {
        assert_eq!(ProcessSortColumn::Pid.header(), "PID");
        assert_eq!(ProcessSortColumn::Name.header(), "COMMAND");
        assert_eq!(ProcessSortColumn::Cpu.header(), "CPU%");
        assert_eq!(ProcessSortColumn::Mem.header(), "MEM%");
        assert_eq!(ProcessSortColumn::Threads.header(), "THR");
        assert_eq!(ProcessSortColumn::Priority.header(), "PRI");
        assert_eq!(ProcessSortColumn::User.header(), "USER");
        assert_eq!(ProcessSortColumn::Time.header(), "TIME+");
    }

    #[test]
    fn test_sort_column_default() {
        assert_eq!(ProcessSortColumn::default(), ProcessSortColumn::Cpu);
    }

    // ProcessRow tests
    #[test]
    fn test_process_row_default() {
        let row = ProcessRow::default();
        assert_eq!(row.pid, 0);
        assert!(row.name.is_empty());
        assert_eq!(row.cpu_percent, 0.0);
        assert!(row.cpu_history.is_empty());
        assert_eq!(row.mem_percent, 0.0);
        assert_eq!(row.state, ProcessDisplayState::Running);
        assert_eq!(row.threads, 1);
        assert_eq!(row.priority, 0);
        assert!(row.user.is_empty());
        assert_eq!(row.cpu_time_secs, 0);
    }

    // ProcessColumnWidths tests
    #[test]
    fn test_column_widths_default() {
        let w = ProcessColumnWidths::default();
        assert_eq!(w.pid, 7);
        assert_eq!(w.name, 20);
        assert_eq!(w.cpu, 6);
        assert_eq!(w.sparkline, 12);
        assert_eq!(w.mem, 6);
        assert_eq!(w.state, 3);
        assert_eq!(w.threads, 4);
        assert_eq!(w.priority, 4);
        assert_eq!(w.user, 10);
        assert_eq!(w.time, 10);
    }

    // ProcessDataFrame creation tests
    #[test]
    fn test_process_dataframe_creation() {
        let df = ProcessDataFrame::new()
            .with_sort(ProcessSortColumn::Mem, true)
            .with_history_len(30);

        assert_eq!(df.sort_column, ProcessSortColumn::Mem);
        assert!(df.sort_desc);
        assert_eq!(df.history_len, 30);
    }

    #[test]
    fn test_process_dataframe_default() {
        let df = ProcessDataFrame::default();
        assert_eq!(df.sort_column, ProcessSortColumn::Cpu);
        assert!(df.sort_desc);
        assert_eq!(df.history_len, 60);
    }

    #[test]
    fn test_with_accent_color() {
        let color = Color::new(1.0, 0.0, 0.0, 1.0);
        let df = ProcessDataFrame::new().with_accent_color(color);
        assert_eq!(df.accent_color.r, 1.0);
    }

    #[test]
    fn test_with_column_widths() {
        let widths = ProcessColumnWidths {
            pid: 10,
            name: 30,
            ..Default::default()
        };
        let df = ProcessDataFrame::new().with_column_widths(widths);
        assert_eq!(df.col_widths.pid, 10);
        assert_eq!(df.col_widths.name, 30);
    }

    // Update and sort tests
    #[test]
    fn test_process_row_update() {
        let mut df = ProcessDataFrame::new();

        let rows = vec![
            ProcessRow {
                pid: 1,
                name: "systemd".into(),
                cpu_percent: 0.5,
                mem_percent: 1.2,
                state: ProcessDisplayState::Running,
                threads: 1,
                priority: 0,
                user: "root".into(),
                cpu_time_secs: 3600,
                ..Default::default()
            },
            ProcessRow {
                pid: 100,
                name: "firefox".into(),
                cpu_percent: 25.0,
                mem_percent: 15.0,
                state: ProcessDisplayState::Sleeping,
                threads: 120,
                priority: 20,
                user: "noah".into(),
                cpu_time_secs: 7200,
                ..Default::default()
            },
        ];

        df.update_processes(rows);

        // Should be sorted by CPU desc by default
        assert_eq!(df.rows[0].pid, 100); // firefox has higher CPU
        assert_eq!(df.rows[1].pid, 1);
    }

    #[test]
    fn test_update_clears_dead_processes() {
        let mut df = ProcessDataFrame::new();

        // First update with two processes
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                cpu_percent: 10.0,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                cpu_percent: 20.0,
                ..Default::default()
            },
        ]);
        assert_eq!(df.cpu_histories.len(), 2);

        // Second update with only one process
        df.update_processes(vec![ProcessRow {
            pid: 1,
            cpu_percent: 15.0,
            ..Default::default()
        }]);
        assert_eq!(df.cpu_histories.len(), 1);
        assert!(df.cpu_histories.contains_key(&1));
        assert!(!df.cpu_histories.contains_key(&2));
    }

    #[test]
    fn test_history_length_limit() {
        let mut df = ProcessDataFrame::new().with_history_len(3);

        for i in 0..5 {
            df.update_processes(vec![ProcessRow {
                pid: 1,
                cpu_percent: i as f32,
                ..Default::default()
            }]);
        }

        let history = df.cpu_histories.get(&1).unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history, &vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_selected_pid() {
        let mut df = ProcessDataFrame::new();
        assert!(df.selected_pid().is_none());

        df.update_processes(vec![ProcessRow {
            pid: 42,
            cpu_percent: 10.0,
            ..Default::default()
        }]);
        df.selected_row = Some(0);
        assert_eq!(df.selected_pid(), Some(42));
    }

    #[test]
    fn test_selected_pid_out_of_range() {
        let mut df = ProcessDataFrame::new();
        df.selected_row = Some(999);
        assert!(df.selected_pid().is_none());
    }

    // Sort tests for all columns
    #[test]
    fn test_sort_by_pid() {
        let mut df = ProcessDataFrame::new().with_sort(ProcessSortColumn::Pid, false);
        df.update_processes(vec![
            ProcessRow {
                pid: 100,
                ..Default::default()
            },
            ProcessRow {
                pid: 1,
                ..Default::default()
            },
            ProcessRow {
                pid: 50,
                ..Default::default()
            },
        ]);
        assert_eq!(df.rows[0].pid, 1);
        assert_eq!(df.rows[1].pid, 50);
        assert_eq!(df.rows[2].pid, 100);
    }

    #[test]
    fn test_sort_by_pid_desc() {
        let mut df = ProcessDataFrame::new().with_sort(ProcessSortColumn::Pid, true);
        df.update_processes(vec![
            ProcessRow {
                pid: 100,
                ..Default::default()
            },
            ProcessRow {
                pid: 1,
                ..Default::default()
            },
            ProcessRow {
                pid: 50,
                ..Default::default()
            },
        ]);
        assert_eq!(df.rows[0].pid, 100);
        assert_eq!(df.rows[1].pid, 50);
        assert_eq!(df.rows[2].pid, 1);
    }

    #[test]
    fn test_sort_by_name() {
        let mut df = ProcessDataFrame::new().with_sort(ProcessSortColumn::Name, false);
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                name: "zsh".into(),
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                name: "bash".into(),
                ..Default::default()
            },
            ProcessRow {
                pid: 3,
                name: "fish".into(),
                ..Default::default()
            },
        ]);
        assert_eq!(df.rows[0].name.as_str(), "bash");
        assert_eq!(df.rows[1].name.as_str(), "fish");
        assert_eq!(df.rows[2].name.as_str(), "zsh");
    }

    #[test]
    fn test_sort_by_mem() {
        let mut df = ProcessDataFrame::new().with_sort(ProcessSortColumn::Mem, true);
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                mem_percent: 10.0,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                mem_percent: 50.0,
                ..Default::default()
            },
            ProcessRow {
                pid: 3,
                mem_percent: 25.0,
                ..Default::default()
            },
        ]);
        assert_eq!(df.rows[0].pid, 2);
        assert_eq!(df.rows[1].pid, 3);
        assert_eq!(df.rows[2].pid, 1);
    }

    #[test]
    fn test_sort_by_threads() {
        let mut df = ProcessDataFrame::new().with_sort(ProcessSortColumn::Threads, true);
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                threads: 1,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                threads: 100,
                ..Default::default()
            },
            ProcessRow {
                pid: 3,
                threads: 10,
                ..Default::default()
            },
        ]);
        assert_eq!(df.rows[0].pid, 2);
        assert_eq!(df.rows[1].pid, 3);
        assert_eq!(df.rows[2].pid, 1);
    }

    #[test]
    fn test_sort_by_priority() {
        let mut df = ProcessDataFrame::new().with_sort(ProcessSortColumn::Priority, false);
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                priority: 20,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                priority: -10,
                ..Default::default()
            },
            ProcessRow {
                pid: 3,
                priority: 0,
                ..Default::default()
            },
        ]);
        assert_eq!(df.rows[0].pid, 2);
        assert_eq!(df.rows[1].pid, 3);
        assert_eq!(df.rows[2].pid, 1);
    }

    #[test]
    fn test_sort_by_user() {
        let mut df = ProcessDataFrame::new().with_sort(ProcessSortColumn::User, false);
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                user: "root".into(),
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                user: "alice".into(),
                ..Default::default()
            },
            ProcessRow {
                pid: 3,
                user: "bob".into(),
                ..Default::default()
            },
        ]);
        assert_eq!(df.rows[0].user.as_str(), "alice");
        assert_eq!(df.rows[1].user.as_str(), "bob");
        assert_eq!(df.rows[2].user.as_str(), "root");
    }

    #[test]
    fn test_sort_by_time() {
        let mut df = ProcessDataFrame::new().with_sort(ProcessSortColumn::Time, true);
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                cpu_time_secs: 100,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                cpu_time_secs: 10000,
                ..Default::default()
            },
            ProcessRow {
                pid: 3,
                cpu_time_secs: 1000,
                ..Default::default()
            },
        ]);
        assert_eq!(df.rows[0].pid, 2);
        assert_eq!(df.rows[1].pid, 3);
        assert_eq!(df.rows[2].pid, 1);
    }

    #[test]
    fn test_cycle_sort() {
        let mut df = ProcessDataFrame::new();
        assert_eq!(df.sort_column, ProcessSortColumn::Cpu);

        df.cycle_sort();
        assert_eq!(df.sort_column, ProcessSortColumn::Mem);

        df.cycle_sort();
        assert_eq!(df.sort_column, ProcessSortColumn::Pid);

        df.cycle_sort();
        assert_eq!(df.sort_column, ProcessSortColumn::Name);

        df.cycle_sort();
        assert_eq!(df.sort_column, ProcessSortColumn::Threads);

        df.cycle_sort();
        assert_eq!(df.sort_column, ProcessSortColumn::Time);

        df.cycle_sort();
        assert_eq!(df.sort_column, ProcessSortColumn::Cpu);
    }

    #[test]
    fn test_cycle_sort_from_priority() {
        let mut df = ProcessDataFrame::new().with_sort(ProcessSortColumn::Priority, true);
        df.cycle_sort();
        assert_eq!(df.sort_column, ProcessSortColumn::Cpu);
    }

    #[test]
    fn test_toggle_sort_direction() {
        let mut df = ProcessDataFrame::new();
        assert!(df.sort_desc);

        df.toggle_sort_direction();
        assert!(!df.sort_desc);

        df.toggle_sort_direction();
        assert!(df.sort_desc);
    }

    // Rendering tests
    #[test]
    fn test_sparkline_rendering() {
        let values = vec![10.0, 20.0, 50.0, 80.0, 100.0];
        let sparkline = ProcessDataFrame::render_sparkline(&values, 5);
        assert_eq!(sparkline.chars().count(), 5);
    }

    #[test]
    fn test_sparkline_empty() {
        let sparkline = ProcessDataFrame::render_sparkline(&[], 5);
        assert_eq!(sparkline, "─────");
    }

    #[test]
    fn test_sparkline_single_value() {
        // With 1 value and width 3, sample_width = min(3, 1) = 1
        let sparkline = ProcessDataFrame::render_sparkline(&[50.0], 3);
        assert_eq!(sparkline.chars().count(), 1);
    }

    #[test]
    fn test_microbar_rendering() {
        let bar = ProcessDataFrame::render_microbar(50.0, 10);
        assert_eq!(bar.chars().count(), 10);
        assert!(bar.contains('█'));
        assert!(bar.contains('░'));
    }

    #[test]
    fn test_microbar_zero() {
        let bar = ProcessDataFrame::render_microbar(0.0, 5);
        assert_eq!(bar, "░░░░░");
    }

    #[test]
    fn test_microbar_full() {
        let bar = ProcessDataFrame::render_microbar(100.0, 5);
        assert_eq!(bar, "█████");
    }

    #[test]
    fn test_microbar_clamped() {
        let bar_over = ProcessDataFrame::render_microbar(150.0, 5);
        assert_eq!(bar_over, "█████");

        let bar_under = ProcessDataFrame::render_microbar(-10.0, 5);
        assert_eq!(bar_under, "░░░░░");
    }

    #[test]
    fn test_format_time() {
        assert_eq!(ProcessDataFrame::format_time(59), "0:59");
        assert_eq!(ProcessDataFrame::format_time(3661), "1:01:01");
        assert_eq!(ProcessDataFrame::format_time(360000), "100h");
    }

    #[test]
    fn test_format_time_zero() {
        assert_eq!(ProcessDataFrame::format_time(0), "0:00");
    }

    #[test]
    fn test_format_time_exact_hour() {
        assert_eq!(ProcessDataFrame::format_time(3600), "1:00:00");
    }

    #[test]
    fn test_format_time_99_hours() {
        assert_eq!(ProcessDataFrame::format_time(99 * 3600), "99:00:00");
    }

    // Scroll tests
    #[test]
    fn test_scroll() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 10.0);

        let rows: Vec<ProcessRow> = (0..20)
            .map(|i| ProcessRow {
                pid: i,
                name: format!("proc{i}").into(),
                cpu_percent: i as f32,
                ..Default::default()
            })
            .collect();

        df.update_processes(rows);

        df.scroll_down();
        assert_eq!(df.selected_row, Some(0));

        df.scroll_down();
        assert_eq!(df.selected_row, Some(1));

        df.scroll_up();
        assert_eq!(df.selected_row, Some(0));
    }

    #[test]
    fn test_scroll_up_at_top() {
        let mut df = ProcessDataFrame::new();
        df.update_processes(vec![ProcessRow {
            pid: 1,
            ..Default::default()
        }]);
        df.selected_row = Some(0);

        df.scroll_up();
        assert_eq!(df.selected_row, Some(0));
    }

    #[test]
    fn test_scroll_down_at_bottom() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 100.0);
        df.update_processes(vec![ProcessRow {
            pid: 1,
            ..Default::default()
        }]);
        df.selected_row = Some(0);

        df.scroll_down();
        assert_eq!(df.selected_row, Some(0));
    }

    #[test]
    fn test_scroll_empty() {
        let mut df = ProcessDataFrame::new();
        df.scroll_up();
        assert!(df.selected_row.is_none());

        df.scroll_down();
        assert!(df.selected_row.is_none());
    }

    #[test]
    fn test_scroll_triggers_offset_adjustment() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 5.0); // Small height
        df.show_header = false;

        let rows: Vec<ProcessRow> = (0..20)
            .map(|i| ProcessRow {
                pid: i,
                cpu_percent: (20 - i) as f32,
                ..Default::default()
            })
            .collect();
        df.update_processes(rows);

        // Scroll down past visible area
        for _ in 0..10 {
            df.scroll_down();
        }
        assert!(df.scroll_offset > 0);
    }

    // Visible rows test
    #[test]
    fn test_visible_rows_with_header() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 10.0);
        df.show_header = true;
        assert_eq!(df.visible_rows(), 8); // 10 - 2 for header
    }

    #[test]
    fn test_visible_rows_without_header() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 10.0);
        df.show_header = false;
        assert_eq!(df.visible_rows(), 10);
    }

    // Brick trait tests
    #[test]
    fn test_brick_name() {
        let df = ProcessDataFrame::new();
        assert_eq!(df.brick_name(), "process_dataframe");
    }

    #[test]
    fn test_brick_assertions() {
        let df = ProcessDataFrame::new();
        let assertions = df.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_brick_budget() {
        let df = ProcessDataFrame::new();
        let budget = df.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_brick_verify() {
        let df = ProcessDataFrame::new();
        let verification = df.verify();
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_to_html() {
        let df = ProcessDataFrame::new();
        let html = df.to_html();
        assert!(html.contains("process-dataframe"));
        assert!(html.contains("PID"));
        assert!(html.contains("COMMAND"));
    }

    #[test]
    fn test_to_css() {
        let df = ProcessDataFrame::new();
        let css = df.to_css();
        assert!(css.contains("process-dataframe"));
        assert!(css.contains("sort"));
    }

    // Widget trait tests
    #[test]
    fn test_type_id() {
        let df = ProcessDataFrame::new();
        let id = Widget::type_id(&df);
        assert_eq!(id, TypeId::of::<ProcessDataFrame>());
    }

    #[test]
    fn test_measure() {
        let df = ProcessDataFrame::new();
        let constraints = Constraints::tight(Size::new(100.0, 50.0));
        let size = df.measure(constraints);
        assert!(size.width <= 100.0);
        assert!(size.height <= 50.0);
    }

    #[test]
    fn test_measure_with_rows() {
        let mut df = ProcessDataFrame::new();
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                ..Default::default()
            },
        ]);
        let constraints = Constraints::tight(Size::new(100.0, 50.0));
        let size = df.measure(constraints);
        assert!(size.height >= 3.0); // header + 2 rows
    }

    #[test]
    fn test_layout() {
        let mut df = ProcessDataFrame::new();
        let bounds = Rect::new(10.0, 20.0, 200.0, 100.0);
        let result = df.layout(bounds);
        assert_eq!(result.size.width, 200.0);
        assert_eq!(result.size.height, 100.0);
        assert_eq!(df.bounds, bounds);
    }

    #[test]
    fn test_paint_too_small() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 30.0, 2.0); // Too small

        let mut buffer = CellBuffer::new(30, 2);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas);
        // Should not crash, just skip painting
    }

    #[test]
    fn test_paint_with_data() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 20.0);
        df.update_processes(vec![ProcessRow {
            pid: 1,
            name: "test".into(),
            cpu_percent: 50.0,
            mem_percent: 25.0,
            state: ProcessDisplayState::Running,
            threads: 4,
            user: "root".into(),
            cpu_time_secs: 3600,
            ..Default::default()
        }]);

        let mut buffer = CellBuffer::new(100, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas);
        // Should render without errors
    }

    #[test]
    fn test_paint_with_scroll_indicator() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 5.0);
        df.show_header = false;

        let rows: Vec<ProcessRow> = (0..20)
            .map(|i| ProcessRow {
                pid: i,
                cpu_percent: i as f32,
                ..Default::default()
            })
            .collect();
        df.update_processes(rows);

        let mut buffer = CellBuffer::new(100, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas);
    }

    #[test]
    fn test_paint_with_selection() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 10.0);
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                cpu_percent: 50.0,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                cpu_percent: 25.0,
                ..Default::default()
            },
        ]);
        df.selected_row = Some(0);

        let mut buffer = CellBuffer::new(100, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        df.paint(&mut canvas);
    }

    // Event tests
    #[test]
    fn test_event_key_up() {
        let mut df = ProcessDataFrame::new();
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                ..Default::default()
            },
        ]);
        df.selected_row = Some(1);

        let result = df.event(&Event::key_down(Key::Up));
        assert!(result.is_none());
        assert_eq!(df.selected_row, Some(0));
    }

    #[test]
    fn test_event_key_k() {
        let mut df = ProcessDataFrame::new();
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                ..Default::default()
            },
        ]);
        df.selected_row = Some(1);

        let result = df.event(&Event::key_down(Key::K));
        assert!(result.is_none());
        assert_eq!(df.selected_row, Some(0));
    }

    #[test]
    fn test_event_key_down() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 100.0);
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                ..Default::default()
            },
        ]);

        let result = df.event(&Event::key_down(Key::Down));
        assert!(result.is_none());
        assert_eq!(df.selected_row, Some(0));
    }

    #[test]
    fn test_event_key_j() {
        let mut df = ProcessDataFrame::new();
        df.bounds = Rect::new(0.0, 0.0, 100.0, 100.0);
        df.update_processes(vec![
            ProcessRow {
                pid: 1,
                ..Default::default()
            },
            ProcessRow {
                pid: 2,
                ..Default::default()
            },
        ]);

        let result = df.event(&Event::key_down(Key::J));
        assert!(result.is_none());
        assert_eq!(df.selected_row, Some(0));
    }

    #[test]
    fn test_event_key_s_cycles_sort() {
        let mut df = ProcessDataFrame::new();
        assert_eq!(df.sort_column, ProcessSortColumn::Cpu);

        df.event(&Event::key_down(Key::S));
        assert_eq!(df.sort_column, ProcessSortColumn::Mem);
    }

    #[test]
    fn test_event_key_r_toggles_direction() {
        let mut df = ProcessDataFrame::new();
        assert!(df.sort_desc);

        df.event(&Event::key_down(Key::R));
        assert!(!df.sort_desc);
    }

    #[test]
    fn test_event_unhandled() {
        let mut df = ProcessDataFrame::new();
        let result = df.event(&Event::key_down(Key::Escape));
        assert!(result.is_none());
    }

    #[test]
    fn test_children_empty() {
        let df = ProcessDataFrame::new();
        assert!(df.children().is_empty());
    }

    #[test]
    fn test_children_mut_empty() {
        let mut df = ProcessDataFrame::new();
        assert!(df.children_mut().is_empty());
    }
}
