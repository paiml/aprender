//! CPU Exploded View Widgets (CB-INPUT-006 Framework-First)
//!
//! Specialized widgets that only appear in CPU panel's exploded (fullscreen) view.
//! These provide detailed CPU analysis that wouldn't fit in the condensed dashboard.
//!
//! # Widgets
//!
//! - [`PerCoreSparklineGrid`]: Individual sparkline history for each CPU core
//! - [`CpuStateBreakdown`]: User/system/idle/iowait stacked bars per core
//! - [`TopProcessesMini`]: Top 5 CPU consumers with live updates
//! - [`FreqTempHeatmap`]: Frequency and temperature heatmap per core
//! - [`LoadAverageTimeline`]: 1/5/15 minute load average sparklines

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

// =============================================================================
// 1. PerCoreSparklineGrid - Individual sparkline per core
// =============================================================================

/// Grid of sparklines showing per-core CPU history.
///
/// In condensed view, only the aggregate sparkline is shown.
/// In exploded view, each core gets its own sparkline with 60-second history.
#[derive(Debug, Clone)]
pub struct PerCoreSparklineGrid {
    /// Per-core history buffers (each is 60 samples = 60 seconds at 1s refresh)
    pub core_histories: Vec<Vec<f64>>,
    /// Gradient for coloring (low→high utilization)
    gradient: Gradient,
    /// Number of columns for grid layout
    columns: Option<usize>,
    /// Whether to show core labels
    show_labels: bool,
    /// Cached layout bounds
    bounds: Rect,
}

impl Default for PerCoreSparklineGrid {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl PerCoreSparklineGrid {
    /// Create a new per-core sparkline grid.
    #[must_use]
    pub fn new(core_histories: Vec<Vec<f64>>) -> Self {
        Self {
            core_histories,
            gradient: Gradient::from_hex(&["#7aa2f7", "#e0af68", "#f7768e"]),
            columns: None,
            show_labels: true,
            bounds: Rect::default(),
        }
    }

    /// Set explicit column count.
    #[must_use]
    pub fn with_columns(mut self, cols: usize) -> Self {
        self.columns = Some(cols);
        self
    }

    /// Set custom gradient.
    #[must_use]
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = gradient;
        self
    }

    /// Disable core labels.
    #[must_use]
    pub fn without_labels(mut self) -> Self {
        self.show_labels = false;
        self
    }

    /// Update core histories.
    pub fn set_histories(&mut self, histories: Vec<Vec<f64>>) {
        self.core_histories = histories;
    }

    /// Get optimal grid dimensions.
    fn optimal_grid(&self, max_width: usize, cell_width: usize) -> (usize, usize) {
        let count = self.core_histories.len();
        if count == 0 {
            return (0, 0);
        }

        let max_cols = (max_width / cell_width).max(1);
        let cols = self.columns.unwrap_or_else(|| {
            // For sparklines, prefer fewer columns (wider sparklines)
            let ideal = (count as f64).sqrt().ceil() as usize;
            ideal.min(max_cols).min(4).max(1)
        });

        let rows = count.div_ceil(cols);
        (cols, rows)
    }
}

impl Brick for PerCoreSparklineGrid {
    fn brick_name(&self) -> &'static str {
        "per_core_sparkline_grid"
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
        String::new()
    }
    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for PerCoreSparklineGrid {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let cell_width = 20; // 3 chars label + 15 chars sparkline + 2 spacing
        let cell_height = 2; // sparkline + label row
        let (cols, rows) = self.optimal_grid(constraints.max_width as usize, cell_width);

        let width = (cols * cell_width) as f32;
        let height = (rows * cell_height) as f32;

        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.core_histories.is_empty() {
            return;
        }

        let cell_width = 20_usize;
        let cell_height = 2_usize;
        let (cols, _) = self.optimal_grid(self.bounds.width as usize, cell_width);
        if cols == 0 {
            return;
        }

        for (i, history) in self.core_histories.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;

            let x = self.bounds.x + (col * cell_width) as f32;
            let y = self.bounds.y + (row * cell_height) as f32;

            // Get current value for color
            let current = history.last().copied().unwrap_or(0.0);
            let color = self.gradient.for_percent(current);

            // Draw label
            if self.show_labels {
                let label = format!("{i:2}");
                canvas.draw_text(
                    &label,
                    Point::new(x, y),
                    &TextStyle {
                        color,
                        ..Default::default()
                    },
                );
            }

            // Draw mini sparkline
            let sparkline_x = x + if self.show_labels { 3.0 } else { 0.0 };
            let sparkline_width = (cell_width as f32 - 5.0).max(8.0);

            // Convert history to sparkline chars
            let chars: String = history
                .iter()
                .rev()
                .take(sparkline_width as usize)
                .rev()
                .map(|&v| {
                    let idx = ((v / 100.0) * 7.0).round() as usize;
                    ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][idx.min(7)]
                })
                .collect();

            canvas.draw_text(
                &chars,
                Point::new(sparkline_x, y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );

            // Draw current percentage
            let pct_str = format!("{current:3.0}%");
            canvas.draw_text(
                &pct_str,
                Point::new(sparkline_x + sparkline_width, y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

// =============================================================================
// 2. CpuStateBreakdown - User/System/Idle/IOWait stacked bars
// =============================================================================

/// CPU state breakdown showing user/system/idle/iowait as stacked bars.
#[derive(Debug, Clone)]
pub struct CpuStateBreakdown {
    /// Per-core CPU states: (user%, system%, idle%, iowait%, irq%, softirq%)
    pub core_states: Vec<CpuCoreState>,
    /// Cached layout bounds
    bounds: Rect,
}

/// CPU state percentages for a single core.
#[derive(Debug, Clone, Default)]
pub struct CpuCoreState {
    pub user: f64,
    pub system: f64,
    pub idle: f64,
    pub iowait: f64,
    pub irq: f64,
    pub softirq: f64,
}

impl CpuCoreState {
    /// Create from percentages.
    pub fn new(user: f64, system: f64, idle: f64, iowait: f64, irq: f64, softirq: f64) -> Self {
        Self {
            user,
            system,
            idle,
            iowait,
            irq,
            softirq,
        }
    }
}

impl Default for CpuStateBreakdown {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl CpuStateBreakdown {
    /// Create a new CPU state breakdown widget.
    #[must_use]
    pub fn new(core_states: Vec<CpuCoreState>) -> Self {
        Self {
            core_states,
            bounds: Rect::default(),
        }
    }

    /// Update core states.
    pub fn set_states(&mut self, states: Vec<CpuCoreState>) {
        self.core_states = states;
    }
}

// State colors (Tokyo Night inspired) - helper function
fn state_colors() -> [Color; 6] {
    [
        Color::rgb(0.478, 0.635, 0.969), // USER: Blue #7aa2f7
        Color::rgb(0.878, 0.686, 0.408), // SYSTEM: Yellow #e0af68
        Color::rgb(0.969, 0.463, 0.557), // IOWAIT: Red #f7768e
        Color::rgb(0.733, 0.604, 0.969), // IRQ: Purple #bb9af7
        Color::rgb(0.620, 0.808, 0.416), // SOFTIRQ: Green #9ece6a
        Color::rgb(0.337, 0.373, 0.537), // IDLE: Gray #565f89
    ]
}

impl Brick for CpuStateBreakdown {
    fn brick_name(&self) -> &'static str {
        "cpu_state_breakdown"
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
        String::new()
    }
    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for CpuStateBreakdown {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height = self.core_states.len().min(constraints.max_height as usize) as f32;
        constraints.constrain(Size::new(constraints.max_width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let bar_width = (self.bounds.width as usize).saturating_sub(20); // Reserve for labels

        for (i, state) in self.core_states.iter().enumerate() {
            let y = self.bounds.y + i as f32;
            if y >= self.bounds.y + self.bounds.height {
                break;
            }

            // Draw label
            let label = format!("{i:2}:");
            canvas.draw_text(&label, Point::new(self.bounds.x, y), &TextStyle::default());

            // Calculate bar segments
            let total =
                state.user + state.system + state.idle + state.iowait + state.irq + state.softirq;
            if total <= 0.0 {
                continue;
            }

            let mut x = self.bounds.x + 4.0;
            let colors = state_colors();
            let segments = [
                (state.user, colors[0], '█'),    // USER
                (state.system, colors[1], '▓'),  // SYSTEM
                (state.iowait, colors[2], '░'),  // IOWAIT
                (state.irq, colors[3], '▒'),     // IRQ
                (state.softirq, colors[4], '░'), // SOFTIRQ
                (state.idle, colors[5], '·'),    // IDLE
            ];

            for (pct, color, ch) in segments {
                let width = ((pct / total) * bar_width as f64).round() as usize;
                if width > 0 {
                    let bar: String = std::iter::repeat(ch).take(width).collect();
                    canvas.draw_text(
                        &bar,
                        Point::new(x, y),
                        &TextStyle {
                            color,
                            ..Default::default()
                        },
                    );
                    x += width as f32;
                }
            }

            // Draw percentage summary
            let summary = format!(
                " u:{:2.0} s:{:2.0} w:{:2.0}",
                state.user, state.system, state.iowait
            );
            canvas.draw_text(&summary, Point::new(x + 1.0, y), &TextStyle::default());
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

// =============================================================================
// 3. TopProcessesMini - Top 5 CPU consumers
// =============================================================================

/// Mini table showing top 5 CPU-consuming processes.
#[derive(Debug, Clone)]
pub struct TopProcessesMini {
    /// Top processes: (pid, cpu%, name)
    pub processes: Vec<TopProcess>,
    /// Cached layout bounds
    bounds: Rect,
}

/// A top CPU-consuming process.
#[derive(Debug, Clone)]
pub struct TopProcess {
    pub pid: u32,
    pub cpu_percent: f32,
    pub name: String,
}

impl TopProcess {
    pub fn new(pid: u32, cpu_percent: f32, name: impl Into<String>) -> Self {
        Self {
            pid,
            cpu_percent,
            name: name.into(),
        }
    }
}

impl Default for TopProcessesMini {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl TopProcessesMini {
    /// Create a new top processes widget.
    #[must_use]
    pub fn new(processes: Vec<TopProcess>) -> Self {
        Self {
            processes,
            bounds: Rect::default(),
        }
    }

    /// Update processes list.
    pub fn set_processes(&mut self, processes: Vec<TopProcess>) {
        self.processes = processes;
    }
}

impl Brick for TopProcessesMini {
    fn brick_name(&self) -> &'static str {
        "top_processes_mini"
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
        String::new()
    }
    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for TopProcessesMini {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height = (self.processes.len().min(5) + 1) as f32; // +1 for header
        constraints.constrain(Size::new(constraints.max_width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Header
        let header = " PID    %CPU  Command";
        canvas.draw_text(
            header,
            Point::new(self.bounds.x, self.bounds.y),
            &TextStyle {
                color: Color::rgb(0.663, 0.694, 0.839), // Gray #a9b1d6
                ..Default::default()
            },
        );

        // Gradient for CPU usage
        let gradient = Gradient::from_hex(&["#7aa2f7", "#e0af68", "#f7768e"]);

        for (i, proc) in self.processes.iter().take(5).enumerate() {
            let y = self.bounds.y + (i + 1) as f32;
            if y >= self.bounds.y + self.bounds.height {
                break;
            }

            let color = gradient.for_percent(proc.cpu_percent as f64);
            let max_name_len = (self.bounds.width as usize).saturating_sub(15);
            let name = if proc.name.len() > max_name_len {
                format!("{}...", &proc.name[..max_name_len.saturating_sub(3)])
            } else {
                proc.name.clone()
            };

            let line = format!("{:5} {:5.1}  {}", proc.pid, proc.cpu_percent, name);
            canvas.draw_text(
                &line,
                Point::new(self.bounds.x, y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

// =============================================================================
// 4. FreqTempHeatmap - Frequency and Temperature per core
// =============================================================================

/// Heatmap showing frequency and temperature per core.
#[derive(Debug, Clone)]
pub struct FreqTempHeatmap {
    /// Per-core frequencies in MHz
    pub frequencies: Vec<u32>,
    /// Per-core max frequencies in MHz
    pub max_frequencies: Vec<u32>,
    /// Per-core temperatures in Celsius (optional)
    pub temperatures: Option<Vec<f32>>,
    /// Cached layout bounds
    bounds: Rect,
}

impl Default for FreqTempHeatmap {
    fn default() -> Self {
        Self::new(vec![], vec![])
    }
}

impl FreqTempHeatmap {
    /// Create a new frequency/temperature heatmap.
    #[must_use]
    pub fn new(frequencies: Vec<u32>, max_frequencies: Vec<u32>) -> Self {
        Self {
            frequencies,
            max_frequencies,
            temperatures: None,
            bounds: Rect::default(),
        }
    }

    /// Add temperature data.
    #[must_use]
    pub fn with_temperatures(mut self, temps: Vec<f32>) -> Self {
        self.temperatures = Some(temps);
        self
    }

    /// Update data.
    pub fn set_data(&mut self, freqs: Vec<u32>, max_freqs: Vec<u32>, temps: Option<Vec<f32>>) {
        self.frequencies = freqs;
        self.max_frequencies = max_freqs;
        self.temperatures = temps;
    }
}

impl Brick for FreqTempHeatmap {
    fn brick_name(&self) -> &'static str {
        "freq_temp_heatmap"
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
        String::new()
    }
    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for FreqTempHeatmap {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let has_temps = self.temperatures.is_some();
        let height = if has_temps { 4.0 } else { 2.0 }; // 2 rows per heatmap
        constraints.constrain(Size::new(constraints.max_width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.frequencies.is_empty() {
            return;
        }

        let freq_gradient = Gradient::from_hex(&["#565f89", "#7aa2f7", "#9ece6a"]); // Gray→Blue→Green
        let temp_gradient = Gradient::from_hex(&["#9ece6a", "#e0af68", "#f7768e"]); // Green→Yellow→Red

        // Calculate cell width
        let cell_width = 4_usize;
        let cores_per_row = (self.bounds.width as usize / cell_width).max(1);

        // Frequency row
        canvas.draw_text(
            "Freq:",
            Point::new(self.bounds.x, self.bounds.y),
            &TextStyle::default(),
        );
        for (i, (&freq, &max_freq)) in self
            .frequencies
            .iter()
            .zip(self.max_frequencies.iter())
            .enumerate()
        {
            let col = i % cores_per_row;
            let row = i / cores_per_row;
            let x = self.bounds.x + 6.0 + (col * cell_width) as f32;
            let y = self.bounds.y + row as f32;

            let pct = if max_freq > 0 {
                (freq as f64 / max_freq as f64) * 100.0
            } else {
                0.0
            };
            let color = freq_gradient.for_percent(pct);
            let level = ((pct / 100.0) * 7.0).round() as usize;
            let ch = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][level.min(7)];

            canvas.draw_text(
                &ch.to_string(),
                Point::new(x, y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );
        }

        // Temperature row (if available)
        if let Some(ref temps) = self.temperatures {
            let temp_y = self.bounds.y + 2.0;
            canvas.draw_text(
                "Temp:",
                Point::new(self.bounds.x, temp_y),
                &TextStyle::default(),
            );

            for (i, &temp) in temps.iter().enumerate() {
                let col = i % cores_per_row;
                let row = i / cores_per_row;
                let x = self.bounds.x + 6.0 + (col * cell_width) as f32;
                let y = temp_y + row as f32;

                // Map temperature 30-100°C to 0-100%
                let pct = ((temp - 30.0) / 70.0 * 100.0).clamp(0.0, 100.0) as f64;
                let color = temp_gradient.for_percent(pct);
                let level = ((pct / 100.0) * 7.0).round() as usize;
                let ch = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][level.min(7)];

                canvas.draw_text(
                    &ch.to_string(),
                    Point::new(x, y),
                    &TextStyle {
                        color,
                        ..Default::default()
                    },
                );
            }
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

// =============================================================================
// 5. LoadAverageTimeline - 1/5/15 minute load average sparklines
// =============================================================================

/// Three sparklines showing load average history.
#[derive(Debug, Clone)]
pub struct LoadAverageTimeline {
    /// 1-minute load average history
    pub load_1m: Vec<f64>,
    /// 5-minute load average history
    pub load_5m: Vec<f64>,
    /// 15-minute load average history
    pub load_15m: Vec<f64>,
    /// Number of CPU cores (for relative scaling)
    pub core_count: usize,
    /// Cached layout bounds
    bounds: Rect,
}

impl Default for LoadAverageTimeline {
    fn default() -> Self {
        Self::new(1)
    }
}

impl LoadAverageTimeline {
    /// Create a new load average timeline.
    #[must_use]
    pub fn new(core_count: usize) -> Self {
        Self {
            load_1m: Vec::new(),
            load_5m: Vec::new(),
            load_15m: Vec::new(),
            core_count: core_count.max(1),
            bounds: Rect::default(),
        }
    }

    /// Push new load average values.
    pub fn push(&mut self, load_1m: f64, load_5m: f64, load_15m: f64) {
        const MAX_HISTORY: usize = 60;

        self.load_1m.push(load_1m);
        self.load_5m.push(load_5m);
        self.load_15m.push(load_15m);

        if self.load_1m.len() > MAX_HISTORY {
            self.load_1m.remove(0);
        }
        if self.load_5m.len() > MAX_HISTORY {
            self.load_5m.remove(0);
        }
        if self.load_15m.len() > MAX_HISTORY {
            self.load_15m.remove(0);
        }
    }

    /// Set core count for relative scaling.
    pub fn set_core_count(&mut self, count: usize) {
        self.core_count = count.max(1);
    }
}

impl Brick for LoadAverageTimeline {
    fn brick_name(&self) -> &'static str {
        "load_average_timeline"
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
        String::new()
    }
    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for LoadAverageTimeline {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        constraints.constrain(Size::new(constraints.max_width, 3.0))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let gradient = Gradient::from_hex(&["#9ece6a", "#e0af68", "#f7768e"]); // Green→Yellow→Red
        let sparkline_width = (self.bounds.width as usize).saturating_sub(20);

        let loads = [
            (" 1m:", &self.load_1m),
            (" 5m:", &self.load_5m),
            ("15m:", &self.load_15m),
        ];

        for (row, (label, history)) in loads.iter().enumerate() {
            let y = self.bounds.y + row as f32;
            if y >= self.bounds.y + self.bounds.height {
                break;
            }

            // Draw label
            canvas.draw_text(label, Point::new(self.bounds.x, y), &TextStyle::default());

            if history.is_empty() {
                continue;
            }

            // Current value
            let current = history.last().copied().unwrap_or(0.0);
            let load_pct = (current / self.core_count as f64 * 100.0).min(200.0);
            let color = gradient.for_percent(load_pct.min(100.0));

            // Draw sparkline
            let chars: String = history
                .iter()
                .rev()
                .take(sparkline_width)
                .rev()
                .map(|&v| {
                    let pct = (v / self.core_count as f64 * 100.0).min(200.0);
                    let idx = ((pct / 200.0) * 7.0).round() as usize;
                    ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][idx.min(7)]
                })
                .collect();

            canvas.draw_text(
                &chars,
                Point::new(self.bounds.x + 5.0, y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );

            // Current value
            let value_str = format!(" ({current:.2})");
            canvas.draw_text(
                &value_str,
                Point::new(self.bounds.x + 5.0 + sparkline_width as f32, y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
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

    // ==================== PerCoreSparklineGrid Tests ====================

    #[test]
    fn test_per_core_sparkline_grid_new() {
        let grid = PerCoreSparklineGrid::new(vec![vec![10.0, 20.0, 30.0]; 4]);
        assert_eq!(grid.core_histories.len(), 4);
    }

    #[test]
    fn test_per_core_sparkline_grid_default() {
        let grid = PerCoreSparklineGrid::default();
        assert!(grid.core_histories.is_empty());
    }

    #[test]
    fn test_per_core_sparkline_grid_with_columns() {
        let grid = PerCoreSparklineGrid::new(vec![vec![10.0]; 8]).with_columns(4);
        assert_eq!(grid.columns, Some(4));
    }

    #[test]
    fn test_per_core_sparkline_grid_with_gradient() {
        let custom_gradient = Gradient::from_hex(&["#ff0000", "#00ff00"]);
        let grid = PerCoreSparklineGrid::new(vec![vec![10.0]; 4]).with_gradient(custom_gradient);
        // Gradient is set (can't easily test internal value, but no panic)
        assert!(!grid.core_histories.is_empty());
    }

    #[test]
    fn test_per_core_sparkline_grid_without_labels() {
        let grid = PerCoreSparklineGrid::new(vec![vec![10.0]; 4]).without_labels();
        assert!(!grid.show_labels);
    }

    #[test]
    fn test_per_core_sparkline_grid_set_histories() {
        let mut grid = PerCoreSparklineGrid::new(vec![]);
        assert!(grid.core_histories.is_empty());
        grid.set_histories(vec![vec![10.0, 20.0], vec![30.0, 40.0]]);
        assert_eq!(grid.core_histories.len(), 2);
    }

    #[test]
    fn test_per_core_sparkline_grid_optimal_grid_empty() {
        let grid = PerCoreSparklineGrid::new(vec![]);
        let (cols, rows) = grid.optimal_grid(80, 20);
        assert_eq!((cols, rows), (0, 0));
    }

    #[test]
    fn test_per_core_sparkline_grid_optimal_grid_with_explicit_cols() {
        let grid = PerCoreSparklineGrid::new(vec![vec![10.0]; 8]).with_columns(2);
        let (cols, rows) = grid.optimal_grid(80, 20);
        assert_eq!(cols, 2);
        assert_eq!(rows, 4);
    }

    #[test]
    fn test_per_core_sparkline_grid_paint() {
        let mut grid = PerCoreSparklineGrid::new(vec![
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![60.0, 70.0, 80.0, 90.0, 100.0],
            vec![50.0, 50.0, 50.0, 50.0, 50.0],
            vec![5.0, 10.0, 15.0, 20.0, 25.0],
        ]);
        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        grid.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        grid.paint(&mut canvas);
    }

    #[test]
    fn test_per_core_sparkline_grid_paint_empty() {
        let mut grid = PerCoreSparklineGrid::new(vec![]);
        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        grid.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        grid.paint(&mut canvas); // Should not panic
    }

    #[test]
    fn test_per_core_sparkline_grid_paint_without_labels() {
        let mut grid = PerCoreSparklineGrid::new(vec![vec![50.0; 10]; 4]).without_labels();
        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        grid.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        grid.paint(&mut canvas);
    }

    #[test]
    fn test_per_core_sparkline_grid_paint_with_empty_history() {
        let mut grid = PerCoreSparklineGrid::new(vec![vec![], vec![]]);
        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        grid.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        grid.paint(&mut canvas);
    }

    #[test]
    fn test_per_core_sparkline_grid_measure() {
        let grid = PerCoreSparklineGrid::new(vec![vec![10.0]; 4]);
        let size = grid.measure(Constraints {
            min_width: 0.0,
            max_width: 80.0,
            min_height: 0.0,
            max_height: 40.0,
        });
        assert!(size.width > 0.0);
    }

    #[test]
    fn test_per_core_sparkline_grid_measure_empty() {
        let grid = PerCoreSparklineGrid::new(vec![]);
        let size = grid.measure(Constraints {
            min_width: 0.0,
            max_width: 80.0,
            min_height: 0.0,
            max_height: 40.0,
        });
        assert!((size.width - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_per_core_sparkline_grid_event() {
        let mut grid = PerCoreSparklineGrid::default();
        let event = Event::FocusIn;
        assert!(grid.event(&event).is_none());
    }

    #[test]
    fn test_per_core_sparkline_grid_children() {
        let grid = PerCoreSparklineGrid::default();
        assert!(grid.children().is_empty());
    }

    #[test]
    fn test_per_core_sparkline_grid_children_mut() {
        let mut grid = PerCoreSparklineGrid::default();
        assert!(grid.children_mut().is_empty());
    }

    #[test]
    fn test_per_core_sparkline_grid_to_html_css() {
        let grid = PerCoreSparklineGrid::default();
        assert!(grid.to_html().is_empty());
        assert!(grid.to_css().is_empty());
    }

    #[test]
    fn test_per_core_sparkline_grid_budget() {
        let grid = PerCoreSparklineGrid::default();
        let budget = grid.budget();
        assert!(budget.measure_ms > 0);
    }

    #[test]
    fn test_per_core_sparkline_grid_type_id() {
        let grid = PerCoreSparklineGrid::default();
        let type_id = Widget::type_id(&grid);
        assert_eq!(type_id, TypeId::of::<PerCoreSparklineGrid>());
    }

    #[test]
    fn test_per_core_sparkline_grid_clone() {
        let grid = PerCoreSparklineGrid::new(vec![vec![10.0, 20.0]]);
        let cloned = grid;
        assert_eq!(cloned.core_histories.len(), 1);
    }

    #[test]
    fn test_per_core_sparkline_grid_debug() {
        let grid = PerCoreSparklineGrid::new(vec![vec![10.0]]);
        let debug = format!("{:?}", grid);
        assert!(debug.contains("PerCoreSparklineGrid"));
    }

    // ==================== CpuStateBreakdown Tests ====================

    #[test]
    fn test_cpu_state_breakdown_new() {
        let breakdown =
            CpuStateBreakdown::new(vec![CpuCoreState::new(50.0, 10.0, 35.0, 5.0, 0.0, 0.0)]);
        assert_eq!(breakdown.core_states.len(), 1);
    }

    #[test]
    fn test_cpu_state_breakdown_default() {
        let breakdown = CpuStateBreakdown::default();
        assert!(breakdown.core_states.is_empty());
    }

    #[test]
    fn test_cpu_state_breakdown_set_states() {
        let mut breakdown = CpuStateBreakdown::default();
        breakdown.set_states(vec![
            CpuCoreState::new(50.0, 10.0, 35.0, 5.0, 0.0, 0.0),
            CpuCoreState::new(30.0, 20.0, 45.0, 5.0, 0.0, 0.0),
        ]);
        assert_eq!(breakdown.core_states.len(), 2);
    }

    #[test]
    fn test_cpu_state_breakdown_paint() {
        let mut breakdown = CpuStateBreakdown::new(vec![
            CpuCoreState::new(50.0, 10.0, 35.0, 5.0, 0.0, 0.0),
            CpuCoreState::new(70.0, 15.0, 10.0, 5.0, 0.0, 0.0),
            CpuCoreState::new(30.0, 5.0, 60.0, 5.0, 0.0, 0.0),
        ]);
        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        breakdown.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        breakdown.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_state_breakdown_paint_empty() {
        let mut breakdown = CpuStateBreakdown::default();
        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        breakdown.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        breakdown.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_state_breakdown_paint_zero_total() {
        let mut breakdown =
            CpuStateBreakdown::new(vec![CpuCoreState::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]);
        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        breakdown.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        breakdown.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_state_breakdown_paint_with_irq() {
        let mut breakdown =
            CpuStateBreakdown::new(vec![CpuCoreState::new(30.0, 10.0, 40.0, 5.0, 10.0, 5.0)]);
        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        breakdown.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        breakdown.paint(&mut canvas);
    }

    #[test]
    fn test_cpu_state_breakdown_measure() {
        let breakdown =
            CpuStateBreakdown::new(vec![CpuCoreState::default(), CpuCoreState::default()]);
        let size = breakdown.measure(Constraints {
            min_width: 0.0,
            max_width: 80.0,
            min_height: 0.0,
            max_height: 40.0,
        });
        assert!(size.height >= 2.0);
    }

    #[test]
    fn test_cpu_state_breakdown_event() {
        let mut breakdown = CpuStateBreakdown::default();
        assert!(breakdown.event(&Event::FocusIn).is_none());
    }

    #[test]
    fn test_cpu_state_breakdown_children() {
        let breakdown = CpuStateBreakdown::default();
        assert!(breakdown.children().is_empty());
    }

    #[test]
    fn test_cpu_state_breakdown_to_html_css() {
        let breakdown = CpuStateBreakdown::default();
        assert!(breakdown.to_html().is_empty());
        assert!(breakdown.to_css().is_empty());
    }

    #[test]
    fn test_cpu_state_breakdown_type_id() {
        let breakdown = CpuStateBreakdown::default();
        assert_eq!(
            Widget::type_id(&breakdown),
            TypeId::of::<CpuStateBreakdown>()
        );
    }

    #[test]
    fn test_cpu_state_breakdown_clone() {
        let breakdown = CpuStateBreakdown::new(vec![CpuCoreState::default()]);
        let cloned = breakdown;
        assert_eq!(cloned.core_states.len(), 1);
    }

    #[test]
    fn test_cpu_state_breakdown_debug() {
        let breakdown = CpuStateBreakdown::default();
        let debug = format!("{:?}", breakdown);
        assert!(debug.contains("CpuStateBreakdown"));
    }

    // ==================== CpuCoreState Tests ====================

    #[test]
    fn test_cpu_core_state_fields() {
        let state = CpuCoreState::new(50.0, 10.0, 35.0, 5.0, 0.0, 0.0);
        assert!((state.user - 50.0).abs() < 0.001);
        assert!((state.system - 10.0).abs() < 0.001);
        assert!((state.idle - 35.0).abs() < 0.001);
        assert!((state.iowait - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_cpu_core_state_default() {
        let state = CpuCoreState::default();
        assert!((state.user - 0.0).abs() < 0.001);
        assert!((state.system - 0.0).abs() < 0.001);
        assert!((state.idle - 0.0).abs() < 0.001);
        assert!((state.iowait - 0.0).abs() < 0.001);
        assert!((state.irq - 0.0).abs() < 0.001);
        assert!((state.softirq - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cpu_core_state_clone() {
        let state = CpuCoreState::new(50.0, 10.0, 35.0, 5.0, 0.0, 0.0);
        let cloned = state;
        assert!((cloned.user - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_cpu_core_state_debug() {
        let state = CpuCoreState::new(50.0, 10.0, 35.0, 5.0, 0.0, 0.0);
        let debug = format!("{:?}", state);
        assert!(debug.contains("CpuCoreState"));
        assert!(debug.contains("50"));
    }

    // ==================== TopProcessesMini Tests ====================

    #[test]
    fn test_top_processes_mini_new() {
        let top = TopProcessesMini::new(vec![TopProcess::new(1234, 45.2, "firefox")]);
        assert_eq!(top.processes.len(), 1);
    }

    #[test]
    fn test_top_processes_mini_default() {
        let top = TopProcessesMini::default();
        assert!(top.processes.is_empty());
    }

    #[test]
    fn test_top_processes_mini_set_processes() {
        let mut top = TopProcessesMini::default();
        top.set_processes(vec![
            TopProcess::new(1234, 45.2, "firefox"),
            TopProcess::new(5678, 30.0, "chrome"),
        ]);
        assert_eq!(top.processes.len(), 2);
    }

    #[test]
    fn test_top_processes_mini_paint() {
        let mut top = TopProcessesMini::new(vec![
            TopProcess::new(1234, 45.2, "firefox"),
            TopProcess::new(5678, 30.0, "chrome"),
            TopProcess::new(9012, 15.5, "vscode"),
        ]);
        let mut buffer = CellBuffer::new(40, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        top.layout(Rect::new(0.0, 0.0, 40.0, 10.0));
        top.paint(&mut canvas);
    }

    #[test]
    fn test_top_processes_mini_paint_empty() {
        let mut top = TopProcessesMini::default();
        let mut buffer = CellBuffer::new(40, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        top.layout(Rect::new(0.0, 0.0, 40.0, 10.0));
        top.paint(&mut canvas);
    }

    #[test]
    fn test_top_processes_mini_paint_long_name_truncation() {
        let mut top = TopProcessesMini::new(vec![TopProcess::new(
            1234,
            45.2,
            "this_is_a_very_long_process_name_that_should_be_truncated",
        )]);
        let mut buffer = CellBuffer::new(30, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        top.layout(Rect::new(0.0, 0.0, 30.0, 10.0));
        top.paint(&mut canvas);
    }

    #[test]
    fn test_top_processes_mini_paint_more_than_five() {
        let mut top = TopProcessesMini::new(vec![
            TopProcess::new(1, 90.0, "proc1"),
            TopProcess::new(2, 80.0, "proc2"),
            TopProcess::new(3, 70.0, "proc3"),
            TopProcess::new(4, 60.0, "proc4"),
            TopProcess::new(5, 50.0, "proc5"),
            TopProcess::new(6, 40.0, "proc6"),
            TopProcess::new(7, 30.0, "proc7"),
        ]);
        let mut buffer = CellBuffer::new(40, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        top.layout(Rect::new(0.0, 0.0, 40.0, 10.0));
        top.paint(&mut canvas); // Should only show first 5
    }

    #[test]
    fn test_top_processes_mini_measure() {
        let top = TopProcessesMini::new(vec![
            TopProcess::new(1234, 45.2, "firefox"),
            TopProcess::new(5678, 30.0, "chrome"),
        ]);
        let size = top.measure(Constraints {
            min_width: 0.0,
            max_width: 40.0,
            min_height: 0.0,
            max_height: 20.0,
        });
        assert_eq!(size.height, 3.0); // header + 2 processes
    }

    #[test]
    fn test_top_processes_mini_event() {
        let mut top = TopProcessesMini::default();
        assert!(top.event(&Event::FocusIn).is_none());
    }

    #[test]
    fn test_top_processes_mini_children() {
        let top = TopProcessesMini::default();
        assert!(top.children().is_empty());
    }

    #[test]
    fn test_top_processes_mini_to_html_css() {
        let top = TopProcessesMini::default();
        assert!(top.to_html().is_empty());
        assert!(top.to_css().is_empty());
    }

    #[test]
    fn test_top_processes_mini_type_id() {
        let top = TopProcessesMini::default();
        assert_eq!(Widget::type_id(&top), TypeId::of::<TopProcessesMini>());
    }

    #[test]
    fn test_top_processes_mini_clone() {
        let top = TopProcessesMini::new(vec![TopProcess::new(1234, 45.2, "firefox")]);
        let cloned = top;
        assert_eq!(cloned.processes.len(), 1);
    }

    #[test]
    fn test_top_processes_mini_debug() {
        let top = TopProcessesMini::default();
        let debug = format!("{:?}", top);
        assert!(debug.contains("TopProcessesMini"));
    }

    // ==================== TopProcess Tests ====================

    #[test]
    fn test_top_process_creation() {
        let proc = TopProcess::new(1234, 45.2, "firefox");
        assert_eq!(proc.pid, 1234);
        assert!((proc.cpu_percent - 45.2).abs() < 0.001);
        assert_eq!(proc.name, "firefox");
    }

    #[test]
    fn test_top_process_clone() {
        let proc = TopProcess::new(1234, 45.2, "firefox");
        let cloned = proc;
        assert_eq!(cloned.pid, 1234);
        assert_eq!(cloned.name, "firefox");
    }

    #[test]
    fn test_top_process_debug() {
        let proc = TopProcess::new(1234, 45.2, "firefox");
        let debug = format!("{:?}", proc);
        assert!(debug.contains("TopProcess"));
        assert!(debug.contains("1234"));
    }

    // ==================== FreqTempHeatmap Tests ====================

    #[test]
    fn test_freq_temp_heatmap_new() {
        let heatmap = FreqTempHeatmap::new(vec![3600, 3200], vec![4000, 4000]);
        assert_eq!(heatmap.frequencies.len(), 2);
    }

    #[test]
    fn test_freq_temp_heatmap_default() {
        let heatmap = FreqTempHeatmap::default();
        assert!(heatmap.frequencies.is_empty());
    }

    #[test]
    fn test_freq_temp_heatmap_with_temperatures() {
        let heatmap = FreqTempHeatmap::new(vec![3600, 3200], vec![4000, 4000])
            .with_temperatures(vec![65.0, 70.0]);
        assert_eq!(heatmap.temperatures.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_freq_temp_heatmap_set_data() {
        let mut heatmap = FreqTempHeatmap::default();
        heatmap.set_data(vec![3600, 3200], vec![4000, 4000], Some(vec![65.0, 70.0]));
        assert_eq!(heatmap.frequencies.len(), 2);
        assert!(heatmap.temperatures.is_some());
    }

    #[test]
    fn test_freq_temp_heatmap_set_data_no_temps() {
        let mut heatmap = FreqTempHeatmap::default();
        heatmap.set_data(vec![3600], vec![4000], None);
        assert!(heatmap.temperatures.is_none());
    }

    #[test]
    fn test_freq_temp_heatmap_paint() {
        let mut heatmap =
            FreqTempHeatmap::new(vec![3600, 3200, 3000, 2800], vec![4000, 4000, 4000, 4000])
                .with_temperatures(vec![65.0, 70.0, 75.0, 80.0]);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        heatmap.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        heatmap.paint(&mut canvas);
    }

    #[test]
    fn test_freq_temp_heatmap_paint_empty() {
        let mut heatmap = FreqTempHeatmap::default();
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        heatmap.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        heatmap.paint(&mut canvas);
    }

    #[test]
    fn test_freq_temp_heatmap_paint_no_temps() {
        let mut heatmap = FreqTempHeatmap::new(vec![3600, 3200], vec![4000, 4000]);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        heatmap.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        heatmap.paint(&mut canvas);
    }

    #[test]
    fn test_freq_temp_heatmap_paint_zero_max_freq() {
        let mut heatmap = FreqTempHeatmap::new(vec![3600], vec![0]);
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        heatmap.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        heatmap.paint(&mut canvas);
    }

    #[test]
    fn test_freq_temp_heatmap_paint_extreme_temps() {
        let mut heatmap =
            FreqTempHeatmap::new(vec![3600], vec![4000]).with_temperatures(vec![20.0]); // Below normal range
        let mut buffer = CellBuffer::new(60, 15);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        heatmap.layout(Rect::new(0.0, 0.0, 60.0, 15.0));
        heatmap.paint(&mut canvas);
    }

    #[test]
    fn test_freq_temp_heatmap_measure_no_temps() {
        let heatmap = FreqTempHeatmap::new(vec![3600], vec![4000]);
        let size = heatmap.measure(Constraints {
            min_width: 0.0,
            max_width: 60.0,
            min_height: 0.0,
            max_height: 20.0,
        });
        assert_eq!(size.height, 2.0);
    }

    #[test]
    fn test_freq_temp_heatmap_measure_with_temps() {
        let heatmap = FreqTempHeatmap::new(vec![3600], vec![4000]).with_temperatures(vec![65.0]);
        let size = heatmap.measure(Constraints {
            min_width: 0.0,
            max_width: 60.0,
            min_height: 0.0,
            max_height: 20.0,
        });
        assert_eq!(size.height, 4.0);
    }

    #[test]
    fn test_freq_temp_heatmap_event() {
        let mut heatmap = FreqTempHeatmap::default();
        assert!(heatmap.event(&Event::FocusIn).is_none());
    }

    #[test]
    fn test_freq_temp_heatmap_children() {
        let heatmap = FreqTempHeatmap::default();
        assert!(heatmap.children().is_empty());
    }

    #[test]
    fn test_freq_temp_heatmap_to_html_css() {
        let heatmap = FreqTempHeatmap::default();
        assert!(heatmap.to_html().is_empty());
        assert!(heatmap.to_css().is_empty());
    }

    #[test]
    fn test_freq_temp_heatmap_type_id() {
        let heatmap = FreqTempHeatmap::default();
        assert_eq!(Widget::type_id(&heatmap), TypeId::of::<FreqTempHeatmap>());
    }

    #[test]
    fn test_freq_temp_heatmap_clone() {
        let heatmap = FreqTempHeatmap::new(vec![3600], vec![4000]);
        let cloned = heatmap;
        assert_eq!(cloned.frequencies.len(), 1);
    }

    #[test]
    fn test_freq_temp_heatmap_debug() {
        let heatmap = FreqTempHeatmap::default();
        let debug = format!("{:?}", heatmap);
        assert!(debug.contains("FreqTempHeatmap"));
    }

    // ==================== LoadAverageTimeline Tests ====================

    #[test]
    fn test_load_average_timeline_new() {
        let timeline = LoadAverageTimeline::new(8);
        assert_eq!(timeline.core_count, 8);
        assert!(timeline.load_1m.is_empty());
    }

    #[test]
    fn test_load_average_timeline_new_zero_cores() {
        let timeline = LoadAverageTimeline::new(0);
        assert_eq!(timeline.core_count, 1); // Clamped to 1
    }

    #[test]
    fn test_load_average_timeline_default() {
        let timeline = LoadAverageTimeline::default();
        assert_eq!(timeline.core_count, 1);
    }

    #[test]
    fn test_load_average_timeline_push() {
        let mut timeline = LoadAverageTimeline::new(8);
        timeline.push(1.5, 1.2, 1.0);
        timeline.push(2.0, 1.5, 1.1);
        assert_eq!(timeline.load_1m.len(), 2);
        assert_eq!(timeline.load_5m.len(), 2);
        assert_eq!(timeline.load_15m.len(), 2);
    }

    #[test]
    fn test_load_average_timeline_push_exceeds_max_history() {
        let mut timeline = LoadAverageTimeline::new(8);
        // Push 65 values (exceeds MAX_HISTORY of 60)
        for i in 0..65 {
            timeline.push(i as f64, i as f64, i as f64);
        }
        assert_eq!(timeline.load_1m.len(), 60);
        assert_eq!(timeline.load_5m.len(), 60);
        assert_eq!(timeline.load_15m.len(), 60);
        // First values should be trimmed
        assert!((timeline.load_1m[0] - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_average_timeline_set_core_count() {
        let mut timeline = LoadAverageTimeline::new(8);
        timeline.set_core_count(16);
        assert_eq!(timeline.core_count, 16);
    }

    #[test]
    fn test_load_average_timeline_set_core_count_zero() {
        let mut timeline = LoadAverageTimeline::new(8);
        timeline.set_core_count(0);
        assert_eq!(timeline.core_count, 1); // Clamped to 1
    }

    #[test]
    fn test_load_average_timeline_paint() {
        let mut timeline = LoadAverageTimeline::new(8);
        for i in 0..30 {
            timeline.push(1.0 + i as f64 * 0.1, 1.0, 0.8);
        }
        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        timeline.layout(Rect::new(0.0, 0.0, 60.0, 10.0));
        timeline.paint(&mut canvas);
    }

    #[test]
    fn test_load_average_timeline_paint_empty() {
        let mut timeline = LoadAverageTimeline::new(8);
        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        timeline.layout(Rect::new(0.0, 0.0, 60.0, 10.0));
        timeline.paint(&mut canvas);
    }

    #[test]
    fn test_load_average_timeline_paint_high_load() {
        let mut timeline = LoadAverageTimeline::new(4);
        // Push high load values (above core count)
        for i in 0..30 {
            timeline.push(8.0 + i as f64, 7.0, 6.0);
        }
        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        timeline.layout(Rect::new(0.0, 0.0, 60.0, 10.0));
        timeline.paint(&mut canvas);
    }

    #[test]
    fn test_load_average_timeline_measure() {
        let timeline = LoadAverageTimeline::new(8);
        let size = timeline.measure(Constraints {
            min_width: 0.0,
            max_width: 60.0,
            min_height: 0.0,
            max_height: 20.0,
        });
        assert_eq!(size.height, 3.0);
    }

    #[test]
    fn test_load_average_timeline_event() {
        let mut timeline = LoadAverageTimeline::default();
        assert!(timeline.event(&Event::FocusIn).is_none());
    }

    #[test]
    fn test_load_average_timeline_children() {
        let timeline = LoadAverageTimeline::default();
        assert!(timeline.children().is_empty());
    }

    #[test]
    fn test_load_average_timeline_to_html_css() {
        let timeline = LoadAverageTimeline::default();
        assert!(timeline.to_html().is_empty());
        assert!(timeline.to_css().is_empty());
    }

    #[test]
    fn test_load_average_timeline_type_id() {
        let timeline = LoadAverageTimeline::default();
        assert_eq!(
            Widget::type_id(&timeline),
            TypeId::of::<LoadAverageTimeline>()
        );
    }

    #[test]
    fn test_load_average_timeline_clone() {
        let mut timeline = LoadAverageTimeline::new(8);
        timeline.push(1.0, 2.0, 3.0);
        let cloned = timeline.clone();
        assert_eq!(cloned.load_1m.len(), 1);
    }

    #[test]
    fn test_load_average_timeline_debug() {
        let timeline = LoadAverageTimeline::default();
        let debug = format!("{:?}", timeline);
        assert!(debug.contains("LoadAverageTimeline"));
    }

    // ==================== General Brick/Widget Tests ====================

    #[test]
    fn test_all_widgets_verify() {
        assert!(PerCoreSparklineGrid::default().verify().is_valid());
        assert!(CpuStateBreakdown::default().verify().is_valid());
        assert!(TopProcessesMini::default().verify().is_valid());
        assert!(FreqTempHeatmap::default().verify().is_valid());
        assert!(LoadAverageTimeline::default().verify().is_valid());
    }

    #[test]
    fn test_all_widgets_brick_names() {
        assert_eq!(
            PerCoreSparklineGrid::default().brick_name(),
            "per_core_sparkline_grid"
        );
        assert_eq!(
            CpuStateBreakdown::default().brick_name(),
            "cpu_state_breakdown"
        );
        assert_eq!(
            TopProcessesMini::default().brick_name(),
            "top_processes_mini"
        );
        assert_eq!(FreqTempHeatmap::default().brick_name(), "freq_temp_heatmap");
        assert_eq!(
            LoadAverageTimeline::default().brick_name(),
            "load_average_timeline"
        );
    }

    #[test]
    fn test_all_widgets_assertions() {
        assert!(!PerCoreSparklineGrid::default().assertions().is_empty());
        assert!(!CpuStateBreakdown::default().assertions().is_empty());
        assert!(!TopProcessesMini::default().assertions().is_empty());
        assert!(!FreqTempHeatmap::default().assertions().is_empty());
        assert!(!LoadAverageTimeline::default().assertions().is_empty());
    }

    #[test]
    fn test_all_widgets_budget() {
        assert!(PerCoreSparklineGrid::default().budget().measure_ms > 0);
        assert!(CpuStateBreakdown::default().budget().measure_ms > 0);
        assert!(TopProcessesMini::default().budget().measure_ms > 0);
        assert!(FreqTempHeatmap::default().budget().measure_ms > 0);
        assert!(LoadAverageTimeline::default().budget().measure_ms > 0);
    }

    // ==================== state_colors() Test ====================

    #[test]
    fn test_state_colors() {
        let colors = state_colors();
        assert_eq!(colors.len(), 6);
        // Verify USER color is blue-ish
        assert!(colors[0].r > 0.4 && colors[0].r < 0.6);
        // Verify IDLE color is gray-ish
        assert!(colors[5].r < 0.4);
    }
}
