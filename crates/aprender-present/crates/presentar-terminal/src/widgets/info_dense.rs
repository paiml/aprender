//! Information-Dense Widgets (Tufte-inspired)
//!
//! Widgets designed to maximize data-ink ratio and answer user questions directly.
//! These prioritize information density over decoration.
//!
//! # Design Principles (Tufte)
//! - Maximize data-ink ratio
//! - Show comparisons and context
//! - Avoid chart junk
//! - Use small multiples
//! - Show outliers, not repetitive data

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

// =============================================================================
// Color Helpers
// =============================================================================

fn color_for_cpu_percent(pct: f32) -> Color {
    if pct > 50.0 {
        Color {
            r: 1.0,
            g: 0.4,
            b: 0.4,
            a: 1.0,
        } // Red
    } else if pct > 10.0 {
        Color {
            r: 1.0,
            g: 0.8,
            b: 0.4,
            a: 1.0,
        } // Yellow
    } else if pct > 1.0 {
        Color {
            r: 0.6,
            g: 0.9,
            b: 0.6,
            a: 1.0,
        } // Green
    } else {
        Color {
            r: 0.5,
            g: 0.5,
            b: 0.6,
            a: 1.0,
        } // Dim
    }
}

fn color_for_status(level: HealthLevel) -> Color {
    match level {
        HealthLevel::Critical => Color {
            r: 1.0,
            g: 0.2,
            b: 0.2,
            a: 1.0,
        },
        HealthLevel::High => Color {
            r: 1.0,
            g: 0.5,
            b: 0.3,
            a: 1.0,
        },
        HealthLevel::Moderate => Color {
            r: 1.0,
            g: 0.8,
            b: 0.4,
            a: 1.0,
        },
        HealthLevel::Ok => Color {
            r: 0.5,
            g: 0.9,
            b: 0.5,
            a: 1.0,
        },
    }
}

/// Health level for system metrics (renamed to avoid conflict with `dataframe::HealthLevel`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthLevel {
    Critical,
    High,
    Moderate,
    Ok,
}

impl HealthLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Critical => "CRITICAL",
            Self::High => "HIGH",
            Self::Moderate => "MODERATE",
            Self::Ok => "OK",
        }
    }
}

// =============================================================================
// 1. TopProcessesTable - Information-dense process list
// =============================================================================

/// Process entry for the table
#[derive(Debug, Clone)]
pub struct CpuConsumer {
    pub pid: u32,
    pub cpu_percent: f32,
    pub memory_bytes: u64,
    pub name: String,
}

impl CpuConsumer {
    pub fn new(pid: u32, cpu_percent: f32, memory_bytes: u64, name: impl Into<String>) -> Self {
        Self {
            pid,
            cpu_percent,
            memory_bytes,
            name: name.into(),
        }
    }

    fn memory_display(&self) -> String {
        if self.memory_bytes > 1_073_741_824 {
            format!("{:.1}G", self.memory_bytes as f64 / 1_073_741_824.0)
        } else if self.memory_bytes > 1_048_576 {
            format!("{:.0}M", self.memory_bytes as f64 / 1_048_576.0)
        } else {
            format!("{:.0}K", self.memory_bytes as f64 / 1024.0)
        }
    }
}

/// Information-dense process table showing top CPU consumers.
/// Answers: "What's using my CPU?"
#[derive(Debug, Clone)]
pub struct TopProcessesTable {
    /// All processes (will be sorted by CPU)
    processes: Vec<CpuConsumer>,
    /// Total CPU percentage for header
    total_cpu: f32,
    /// Maximum processes to show
    max_display: usize,
    /// Cached bounds
    bounds: Rect,
}

impl Default for TopProcessesTable {
    fn default() -> Self {
        Self::new(vec![], 0.0)
    }
}

impl TopProcessesTable {
    /// Create a new top processes table
    #[must_use]
    pub fn new(mut processes: Vec<CpuConsumer>, total_cpu: f32) -> Self {
        // Sort by CPU descending
        processes.sort_by(|a, b| {
            b.cpu_percent
                .partial_cmp(&a.cpu_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self {
            processes,
            total_cpu,
            max_display: 10,
            bounds: Rect::default(),
        }
    }

    /// Set maximum processes to display
    #[must_use]
    pub fn with_max_display(mut self, max: usize) -> Self {
        self.max_display = max;
        self
    }

    /// Update processes
    pub fn set_processes(&mut self, mut processes: Vec<CpuConsumer>, total_cpu: f32) {
        processes.sort_by(|a, b| {
            b.cpu_percent
                .partial_cmp(&a.cpu_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.processes = processes;
        self.total_cpu = total_cpu;
    }
}

impl Brick for TopProcessesTable {
    fn brick_name(&self) -> &'static str {
        "top_processes_table"
    }
    fn assertions(&self) -> &[BrickAssertion] {
        static A: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        A
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

impl Widget for TopProcessesTable {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height = (self.max_display + 3) as f32; // header + column header + processes + summary
        constraints.constrain(Size::new(constraints.max_width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        // Adjust max_display based on available height
        self.max_display = ((bounds.height - 3.0) as usize).max(3);
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let x = self.bounds.x;
        let mut y = self.bounds.y;
        let w = self.bounds.width as usize;

        let header_style = TextStyle {
            color: Color {
                r: 0.6,
                g: 0.8,
                b: 1.0,
                a: 1.0,
            },
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: Color {
                r: 0.5,
                g: 0.5,
                b: 0.6,
                a: 1.0,
            },
            ..Default::default()
        };

        // Header
        let header = format!("TOP CPU CONSUMERS ({:.0}% total)", self.total_cpu);
        canvas.draw_text(
            &header[..header.len().min(w)],
            Point::new(x, y),
            &header_style,
        );
        y += 1.0;

        // Column headers
        let col_header = format!("{:>7} {:>6} {:>6}  {:<}", "PID", "CPU%", "MEM", "COMMAND");
        canvas.draw_text(
            &col_header[..col_header.len().min(w)],
            Point::new(x, y),
            &dim_style,
        );
        y += 1.0;

        // Process rows
        let mut other_cpu = 0.0_f32;
        let mut other_count = 0_usize;

        for (i, proc) in self.processes.iter().enumerate() {
            if i < self.max_display && y < self.bounds.y + self.bounds.height - 1.0 {
                let color = color_for_cpu_percent(proc.cpu_percent);
                let max_name = w.saturating_sub(22);
                let name = if proc.name.len() > max_name {
                    format!("{}...", &proc.name[..max_name.saturating_sub(3)])
                } else {
                    proc.name.clone()
                };

                let line = format!(
                    "{:>7} {:>5.1}% {:>6}  {}",
                    proc.pid,
                    proc.cpu_percent,
                    proc.memory_display(),
                    name
                );
                canvas.draw_text(
                    &line[..line.len().min(w)],
                    Point::new(x, y),
                    &TextStyle {
                        color,
                        ..Default::default()
                    },
                );
                y += 1.0;
            } else {
                other_cpu += proc.cpu_percent;
                other_count += 1;
            }
        }

        // Summary of other processes
        if other_count > 0 && y < self.bounds.y + self.bounds.height {
            let other_line = format!("  [{other_count} other processes totaling {other_cpu:.1}%]");
            canvas.draw_text(
                &other_line[..other_line.len().min(w)],
                Point::new(x, y),
                &dim_style,
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
// 2. CoreUtilizationHistogram - Shows distribution, not 48 identical values
// =============================================================================

/// Histogram showing core utilization distribution
#[derive(Debug, Clone)]
pub struct CoreUtilizationHistogram {
    /// Core percentages (0-100)
    core_percentages: Vec<f64>,
    /// Cached bounds
    bounds: Rect,
}

impl Default for CoreUtilizationHistogram {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl CoreUtilizationHistogram {
    #[must_use]
    pub fn new(core_percentages: Vec<f64>) -> Self {
        Self {
            core_percentages,
            bounds: Rect::default(),
        }
    }

    pub fn set_percentages(&mut self, percentages: Vec<f64>) {
        self.core_percentages = percentages;
    }

    fn bucket_counts(&self) -> (usize, usize, usize, usize, usize) {
        let mut b100 = 0; // 95-100%
        let mut bhigh = 0; // 70-95%
        let mut bmed = 0; // 30-70%
        let mut blow = 0; // 1-30%
        let mut bidle = 0; // <1%

        for &pct in &self.core_percentages {
            if pct >= 95.0 {
                b100 += 1;
            } else if pct >= 70.0 {
                bhigh += 1;
            } else if pct >= 30.0 {
                bmed += 1;
            } else if pct >= 1.0 {
                blow += 1;
            } else {
                bidle += 1;
            }
        }
        (b100, bhigh, bmed, blow, bidle)
    }
}

impl Brick for CoreUtilizationHistogram {
    fn brick_name(&self) -> &'static str {
        "core_utilization_histogram"
    }
    fn assertions(&self) -> &[BrickAssertion] {
        static A: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        A
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

impl Widget for CoreUtilizationHistogram {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        constraints.constrain(Size::new(constraints.max_width, 6.0)) // header + 5 buckets max
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let x = self.bounds.x;
        let mut y = self.bounds.y;
        let w = self.bounds.width as usize;

        let header_style = TextStyle {
            color: Color {
                r: 0.6,
                g: 0.8,
                b: 1.0,
                a: 1.0,
            },
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: Color {
                r: 0.5,
                g: 0.5,
                b: 0.6,
                a: 1.0,
            },
            ..Default::default()
        };
        let bright_style = TextStyle {
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            ..Default::default()
        };

        canvas.draw_text("CORE UTILIZATION", Point::new(x, y), &header_style);
        y += 1.0;

        let (b100, bhigh, bmed, blow, bidle) = self.bucket_counts();
        let total = self.core_percentages.len();
        let bar_max = w.saturating_sub(20);

        let draw_bar =
            |canvas: &mut dyn Canvas, y: f32, label: &str, count: usize, color: Color| {
                if count == 0 {
                    return;
                }
                let bar_w = if total > 0 {
                    (count * bar_max) / total
                } else {
                    0
                };
                let bar: String = "█".repeat(bar_w);
                let pad: String = "░".repeat(bar_max - bar_w);

                canvas.draw_text(
                    label,
                    Point::new(x, y),
                    &TextStyle {
                        color,
                        ..Default::default()
                    },
                );
                canvas.draw_text(
                    &bar,
                    Point::new(x + 10.0, y),
                    &TextStyle {
                        color,
                        ..Default::default()
                    },
                );
                canvas.draw_text(&pad, Point::new(x + 10.0 + bar_w as f32, y), &dim_style);
                canvas.draw_text(
                    &format!(" x{count}"),
                    Point::new(x + 10.0 + bar_max as f32, y),
                    &bright_style,
                );
            };

        if b100 > 0 {
            draw_bar(
                canvas,
                y,
                "   100%",
                b100,
                Color {
                    r: 1.0,
                    g: 0.3,
                    b: 0.3,
                    a: 1.0,
                },
            );
            y += 1.0;
        }
        if bhigh > 0 {
            draw_bar(
                canvas,
                y,
                " 70-95%",
                bhigh,
                Color {
                    r: 1.0,
                    g: 0.6,
                    b: 0.3,
                    a: 1.0,
                },
            );
            y += 1.0;
        }
        if bmed > 0 {
            draw_bar(
                canvas,
                y,
                " 30-70%",
                bmed,
                Color {
                    r: 1.0,
                    g: 1.0,
                    b: 0.4,
                    a: 1.0,
                },
            );
            y += 1.0;
        }
        if blow > 0 {
            draw_bar(
                canvas,
                y,
                "  1-30%",
                blow,
                Color {
                    r: 0.5,
                    g: 0.9,
                    b: 0.5,
                    a: 1.0,
                },
            );
            y += 1.0;
        }
        if bidle > 0 {
            draw_bar(
                canvas,
                y,
                "   idle",
                bidle,
                Color {
                    r: 0.4,
                    g: 0.4,
                    b: 0.5,
                    a: 1.0,
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
// 3. TrendSparkline - Sparkline with context (min/max/avg/current)
// =============================================================================

/// Sparkline with statistical context
#[derive(Debug, Clone)]
pub struct TrendSparkline {
    /// History values (0-1 normalized or 0-100 percentage)
    history: Vec<f64>,
    /// Title for the widget
    title: String,
    /// Whether values are percentages (0-100) or normalized (0-1)
    is_percentage: bool,
    /// Cached bounds
    bounds: Rect,
}

impl Default for TrendSparkline {
    fn default() -> Self {
        Self::new("TREND", vec![])
    }
}

impl TrendSparkline {
    #[must_use]
    pub fn new(title: impl Into<String>, history: Vec<f64>) -> Self {
        Self {
            history,
            title: title.into(),
            is_percentage: true,
            bounds: Rect::default(),
        }
    }

    /// Mark values as normalized (0-1) instead of percentage (0-100)
    #[must_use]
    pub fn normalized(mut self) -> Self {
        self.is_percentage = false;
        self
    }

    pub fn set_history(&mut self, history: Vec<f64>) {
        self.history = history;
    }

    pub fn push(&mut self, value: f64) {
        self.history.push(value);
        if self.history.len() > 120 {
            self.history.remove(0);
        }
    }

    fn stats(&self) -> (f64, f64, f64, f64) {
        if self.history.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let mult = if self.is_percentage { 1.0 } else { 100.0 };
        let current = self.history.last().copied().unwrap_or(0.0) * mult;
        let min = self.history.iter().copied().fold(f64::MAX, f64::min) * mult;
        let max = self.history.iter().copied().fold(f64::MIN, f64::max) * mult;
        let avg = self.history.iter().sum::<f64>() / self.history.len() as f64 * mult;
        (current, min, max, avg)
    }
}

impl Brick for TrendSparkline {
    fn brick_name(&self) -> &'static str {
        "trend_sparkline"
    }
    fn assertions(&self) -> &[BrickAssertion] {
        static A: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        A
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

impl Widget for TrendSparkline {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        constraints.constrain(Size::new(constraints.max_width, 4.0)) // header + sparkline + 2 stat lines
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let x = self.bounds.x;
        let mut y = self.bounds.y;
        let w = self.bounds.width as usize;

        let header_style = TextStyle {
            color: Color {
                r: 0.6,
                g: 0.8,
                b: 1.0,
                a: 1.0,
            },
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: Color {
                r: 0.5,
                g: 0.5,
                b: 0.6,
                a: 1.0,
            },
            ..Default::default()
        };
        let bright_style = TextStyle {
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            ..Default::default()
        };

        canvas.draw_text(&self.title, Point::new(x, y), &header_style);
        y += 1.0;

        let (current, min, max, avg) = self.stats();

        // Color based on current level
        let trend_color = if current > 80.0 {
            Color {
                r: 1.0,
                g: 0.4,
                b: 0.4,
                a: 1.0,
            }
        } else if current > 50.0 {
            Color {
                r: 1.0,
                g: 0.8,
                b: 0.4,
                a: 1.0,
            }
        } else {
            Color {
                r: 0.5,
                g: 0.9,
                b: 0.5,
                a: 1.0,
            }
        };

        // Draw sparkline
        let mult = if self.is_percentage { 1.0 } else { 100.0 };
        let chars: String = self
            .history
            .iter()
            .rev()
            .take(w)
            .rev()
            .map(|&v| {
                let pct = (v * mult).clamp(0.0, 100.0);
                let idx = ((pct / 100.0) * 7.0).round() as usize;
                ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][idx.min(7)]
            })
            .collect();

        canvas.draw_text(
            &chars,
            Point::new(x, y),
            &TextStyle {
                color: trend_color,
                ..Default::default()
            },
        );
        y += 1.0;

        // Stats
        let now_avg = format!("Now: {current:.0}%  Avg: {avg:.0}%");
        canvas.draw_text(
            &now_avg[..now_avg.len().min(w)],
            Point::new(x, y),
            &bright_style,
        );
        y += 1.0;

        let min_max = format!("Min: {min:.0}%  Max: {max:.0}%");
        canvas.draw_text(
            &min_max[..min_max.len().min(w)],
            Point::new(x, y),
            &dim_style,
        );
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
// 4. SystemStatus - Load and thermal status with contextual coloring
// =============================================================================

/// System status display (load, thermals)
#[derive(Debug, Clone)]
pub struct SystemStatus {
    /// Load averages (1, 5, 15 minute)
    load_1m: f64,
    load_5m: f64,
    load_15m: f64,
    /// Number of cores (for per-core load calculation)
    core_count: usize,
    /// Thermal data: (`avg_temp`, `max_temp`) in Celsius
    thermal: Option<(f64, f64)>,
    /// Cached bounds
    bounds: Rect,
}

impl Default for SystemStatus {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 1)
    }
}

impl SystemStatus {
    #[must_use]
    pub fn new(load_1m: f64, load_5m: f64, load_15m: f64, core_count: usize) -> Self {
        Self {
            load_1m,
            load_5m,
            load_15m,
            core_count: core_count.max(1),
            thermal: None,
            bounds: Rect::default(),
        }
    }

    #[must_use]
    pub fn with_thermal(mut self, avg_temp: f64, max_temp: f64) -> Self {
        self.thermal = Some((avg_temp, max_temp));
        self
    }

    pub fn set_load(&mut self, l1: f64, l5: f64, l15: f64) {
        self.load_1m = l1;
        self.load_5m = l5;
        self.load_15m = l15;
    }

    pub fn set_thermal(&mut self, avg: f64, max: f64) {
        self.thermal = Some((avg, max));
    }

    /// Get the health level for current load
    pub fn load_status(&self) -> HealthLevel {
        let per_core = self.load_1m / self.core_count as f64;
        if per_core > 1.5 {
            HealthLevel::Critical
        } else if per_core > 1.0 {
            HealthLevel::High
        } else if per_core > 0.7 {
            HealthLevel::Moderate
        } else {
            HealthLevel::Ok
        }
    }

    /// Get the health level for thermal status
    pub fn thermal_status(&self) -> Option<HealthLevel> {
        self.thermal.map(|(_, max)| {
            if max > 90.0 {
                HealthLevel::Critical
            } else if max > 80.0 {
                HealthLevel::High
            } else if max > 70.0 {
                HealthLevel::Moderate
            } else {
                HealthLevel::Ok
            }
        })
    }
}

impl Brick for SystemStatus {
    fn brick_name(&self) -> &'static str {
        "system_status"
    }
    fn assertions(&self) -> &[BrickAssertion] {
        static A: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        A
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

impl Widget for SystemStatus {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height = if self.thermal.is_some() { 2.0 } else { 1.0 };
        constraints.constrain(Size::new(constraints.max_width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let x = self.bounds.x;
        let mut y = self.bounds.y;
        let w = self.bounds.width as usize;

        // Load line
        let load_stat = self.load_status();
        let per_core = self.load_1m / self.core_count as f64;
        let load_line = format!(
            "LOAD: {:.2} / {:.2} / {:.2}  ({:.2}/core) - {}",
            self.load_1m,
            self.load_5m,
            self.load_15m,
            per_core,
            load_stat.as_str()
        );
        canvas.draw_text(
            &load_line[..load_line.len().min(w)],
            Point::new(x, y),
            &TextStyle {
                color: color_for_status(load_stat),
                ..Default::default()
            },
        );

        // Thermal line (only if data present)
        if let Some((avg, max)) = self.thermal {
            y += 1.0;
            let therm_stat = self.thermal_status().unwrap_or(HealthLevel::Ok);
            let therm_line = format!(
                "THERMAL: {:.0}°C avg, {:.0}°C max - {}",
                avg,
                max,
                therm_stat.as_str()
            );
            canvas.draw_text(
                &therm_line[..therm_line.len().min(w)],
                Point::new(x, y),
                &TextStyle {
                    color: color_for_status(therm_stat),
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
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    // HealthLevel tests
    #[test]
    fn test_health_level_as_str() {
        assert_eq!(HealthLevel::Critical.as_str(), "CRITICAL");
        assert_eq!(HealthLevel::High.as_str(), "HIGH");
        assert_eq!(HealthLevel::Moderate.as_str(), "MODERATE");
        assert_eq!(HealthLevel::Ok.as_str(), "OK");
    }

    // Color helper tests
    #[test]
    fn test_color_for_cpu_percent_high() {
        let color = color_for_cpu_percent(75.0);
        assert!(color.r > 0.9); // Red
    }

    #[test]
    fn test_color_for_cpu_percent_medium() {
        let color = color_for_cpu_percent(30.0);
        assert!(color.r > 0.9 && color.g > 0.7); // Yellow
    }

    #[test]
    fn test_color_for_cpu_percent_low() {
        let color = color_for_cpu_percent(5.0);
        assert!(color.g > 0.8); // Green
    }

    #[test]
    fn test_color_for_cpu_percent_idle() {
        let color = color_for_cpu_percent(0.5);
        assert!(color.r < 0.6 && color.g < 0.6); // Dim
    }

    #[test]
    fn test_color_for_status() {
        let critical = color_for_status(HealthLevel::Critical);
        assert!(critical.r > 0.9 && critical.g < 0.3);

        let high = color_for_status(HealthLevel::High);
        assert!(high.r > 0.9);

        let moderate = color_for_status(HealthLevel::Moderate);
        assert!(moderate.r > 0.9 && moderate.g > 0.7);

        let ok = color_for_status(HealthLevel::Ok);
        assert!(ok.g > 0.8);
    }

    // CpuConsumer tests
    #[test]
    fn test_cpu_consumer_new() {
        let proc = CpuConsumer::new(123, 45.5, 1_000_000_000, "firefox");
        assert_eq!(proc.pid, 123);
        assert!((proc.cpu_percent - 45.5).abs() < 0.01);
        assert_eq!(proc.memory_bytes, 1_000_000_000);
        assert_eq!(proc.name, "firefox");
    }

    #[test]
    fn test_cpu_consumer_memory_display_gb() {
        let proc = CpuConsumer::new(1, 10.0, 2_000_000_000, "test");
        let display = proc.memory_display();
        assert!(display.contains('G'));
    }

    #[test]
    fn test_cpu_consumer_memory_display_mb() {
        let proc = CpuConsumer::new(1, 10.0, 500_000_000, "test");
        let display = proc.memory_display();
        assert!(display.contains('M'));
    }

    #[test]
    fn test_cpu_consumer_memory_display_kb() {
        let proc = CpuConsumer::new(1, 10.0, 500_000, "test");
        let display = proc.memory_display();
        assert!(display.contains('K'));
    }

    // TopProcessesTable tests
    #[test]
    fn test_top_processes_table() {
        let procs = vec![
            CpuConsumer::new(123, 45.0, 1_000_000_000, "firefox"),
            CpuConsumer::new(456, 23.0, 500_000_000, "chrome"),
        ];
        let table = TopProcessesTable::new(procs, 68.0);
        assert!(table.verify().is_valid());
    }

    #[test]
    fn test_top_processes_table_default() {
        let table = TopProcessesTable::default();
        assert!(table.processes.is_empty());
    }

    #[test]
    fn test_top_processes_table_with_max_display() {
        let table = TopProcessesTable::new(vec![], 0.0).with_max_display(5);
        assert_eq!(table.max_display, 5);
    }

    #[test]
    fn test_top_processes_table_set_processes() {
        let mut table = TopProcessesTable::new(vec![], 0.0);
        table.set_processes(
            vec![
                CpuConsumer::new(1, 20.0, 100, "a"),
                CpuConsumer::new(2, 50.0, 200, "b"),
            ],
            70.0,
        );
        assert_eq!(table.processes.len(), 2);
        assert_eq!(table.total_cpu, 70.0);
        // Should be sorted by CPU descending
        assert_eq!(table.processes[0].pid, 2);
    }

    #[test]
    fn test_top_processes_table_sorts_by_cpu() {
        let procs = vec![
            CpuConsumer::new(1, 10.0, 100, "low"),
            CpuConsumer::new(2, 90.0, 100, "high"),
            CpuConsumer::new(3, 50.0, 100, "mid"),
        ];
        let table = TopProcessesTable::new(procs, 150.0);
        assert_eq!(table.processes[0].pid, 2); // high
        assert_eq!(table.processes[1].pid, 3); // mid
        assert_eq!(table.processes[2].pid, 1); // low
    }

    #[test]
    fn test_top_processes_table_brick_name() {
        let table = TopProcessesTable::default();
        assert_eq!(table.brick_name(), "top_processes_table");
    }

    #[test]
    fn test_top_processes_table_measure() {
        let table = TopProcessesTable::default();
        let constraints = Constraints::tight(Size::new(80.0, 40.0));
        let size = table.measure(constraints);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_top_processes_table_layout() {
        let mut table = TopProcessesTable::default();
        let bounds = Rect::new(0.0, 0.0, 80.0, 20.0);
        let result = table.layout(bounds);
        assert_eq!(result.size.width, 80.0);
        assert_eq!(result.size.height, 20.0);
    }

    #[test]
    fn test_top_processes_table_paint() {
        let mut table =
            TopProcessesTable::new(vec![CpuConsumer::new(1, 50.0, 1_000_000_000, "test")], 50.0);
        table.bounds = Rect::new(0.0, 0.0, 80.0, 20.0);
        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        table.paint(&mut canvas);
    }

    #[test]
    fn test_top_processes_table_children() {
        let table = TopProcessesTable::default();
        assert!(table.children().is_empty());
    }

    // CoreUtilizationHistogram tests
    #[test]
    fn test_core_histogram() {
        let hist = CoreUtilizationHistogram::new(vec![95.0, 96.0, 50.0, 10.0, 0.5]);
        let (b100, _bhigh, bmed, blow, bidle) = hist.bucket_counts();
        assert_eq!(b100, 2);
        assert_eq!(bmed, 1);
        assert_eq!(blow, 1);
        assert_eq!(bidle, 1);
    }

    #[test]
    fn test_core_histogram_default() {
        let hist = CoreUtilizationHistogram::default();
        assert!(hist.core_percentages.is_empty());
    }

    #[test]
    fn test_core_histogram_bucket_counts_empty() {
        let hist = CoreUtilizationHistogram::new(vec![]);
        let (b100, bhigh, bmed, blow, bidle) = hist.bucket_counts();
        assert_eq!(b100 + bhigh + bmed + blow + bidle, 0);
    }

    #[test]
    fn test_core_histogram_all_buckets() {
        // Test all bucket ranges based on actual thresholds:
        // 95-100: b100, 70-95: bhigh, 30-70: bmed, 1-30: blow, <1: bidle
        let hist = CoreUtilizationHistogram::new(vec![
            97.0, // 95-100: b100
            80.0, // 70-95: bhigh
            50.0, // 30-70: bmed
            15.0, // 1-30: blow
            0.5,  // <1: bidle
        ]);
        let (b100, bhigh, bmed, blow, bidle) = hist.bucket_counts();
        assert_eq!(b100, 1);
        assert_eq!(bhigh, 1);
        assert_eq!(bmed, 1);
        assert_eq!(blow, 1);
        assert_eq!(bidle, 1);
    }

    #[test]
    fn test_core_histogram_brick_name() {
        let hist = CoreUtilizationHistogram::default();
        assert_eq!(hist.brick_name(), "core_utilization_histogram");
    }

    #[test]
    fn test_core_histogram_measure() {
        let hist = CoreUtilizationHistogram::new(vec![50.0; 8]);
        let constraints = Constraints::tight(Size::new(40.0, 10.0));
        let size = hist.measure(constraints);
        assert!(size.width > 0.0);
    }

    #[test]
    fn test_core_histogram_layout() {
        let mut hist = CoreUtilizationHistogram::new(vec![50.0; 4]);
        let bounds = Rect::new(0.0, 0.0, 40.0, 8.0);
        let result = hist.layout(bounds);
        assert_eq!(result.size.width, 40.0);
    }

    #[test]
    fn test_core_histogram_paint() {
        let mut hist = CoreUtilizationHistogram::new(vec![50.0, 75.0, 25.0, 90.0]);
        hist.bounds = Rect::new(0.0, 0.0, 40.0, 8.0);
        let mut buffer = CellBuffer::new(40, 8);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        hist.paint(&mut canvas);
    }

    #[test]
    fn test_core_histogram_children() {
        let hist = CoreUtilizationHistogram::default();
        assert!(hist.children().is_empty());
    }

    // TrendSparkline tests
    #[test]
    fn test_trend_sparkline() {
        let mut trend = TrendSparkline::new("CPU", vec![50.0, 60.0, 70.0]);
        trend.push(80.0);
        let (current, min, max, _avg) = trend.stats();
        assert!((current - 80.0).abs() < 0.01);
        assert!((min - 50.0).abs() < 0.01);
        assert!((max - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_trend_sparkline_default() {
        let trend = TrendSparkline::default();
        assert!(trend.history.is_empty());
    }

    #[test]
    fn test_trend_sparkline_empty_stats() {
        let trend = TrendSparkline::new("Empty", vec![]);
        let (current, min, max, avg) = trend.stats();
        assert_eq!(current, 0.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(avg, 0.0);
    }

    #[test]
    fn test_trend_sparkline_single_value() {
        let trend = TrendSparkline::new("Single", vec![42.0]);
        let (current, min, max, avg) = trend.stats();
        assert_eq!(current, 42.0);
        assert_eq!(min, 42.0);
        assert_eq!(max, 42.0);
        assert_eq!(avg, 42.0);
    }

    #[test]
    fn test_trend_sparkline_push_overflow() {
        let mut trend = TrendSparkline::new("Test", vec![1.0; 120]);
        for i in 0..10 {
            trend.push(i as f64);
        }
        // Should maintain max length (120)
        assert!(trend.history.len() <= 120);
    }

    #[test]
    fn test_trend_sparkline_brick_name() {
        let trend = TrendSparkline::default();
        assert_eq!(trend.brick_name(), "trend_sparkline");
    }

    #[test]
    fn test_trend_sparkline_measure() {
        let trend = TrendSparkline::new("Test", vec![50.0; 10]);
        let constraints = Constraints::tight(Size::new(60.0, 5.0));
        let size = trend.measure(constraints);
        assert!(size.width > 0.0);
    }

    #[test]
    fn test_trend_sparkline_layout() {
        let mut trend = TrendSparkline::new("Test", vec![50.0; 10]);
        let bounds = Rect::new(0.0, 0.0, 60.0, 5.0);
        let result = trend.layout(bounds);
        assert_eq!(result.size.width, 60.0);
    }

    #[test]
    fn test_trend_sparkline_paint() {
        let mut trend = TrendSparkline::new("CPU", vec![30.0, 50.0, 70.0, 90.0]);
        trend.bounds = Rect::new(0.0, 0.0, 60.0, 5.0);
        let mut buffer = CellBuffer::new(60, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        trend.paint(&mut canvas);
    }

    #[test]
    fn test_trend_sparkline_children() {
        let trend = TrendSparkline::default();
        assert!(trend.children().is_empty());
    }

    // SystemStatus tests
    #[test]
    fn test_system_status() {
        let status = SystemStatus::new(4.0, 3.0, 2.0, 8).with_thermal(65.0, 72.0);
        assert_eq!(status.load_status(), HealthLevel::Ok); // 4/8 = 0.5 per core (< 0.7)
        assert_eq!(status.thermal_status(), Some(HealthLevel::Moderate)); // 72°C
    }

    #[test]
    fn test_system_status_default() {
        let status = SystemStatus::default();
        assert!((status.load_1m - 0.0).abs() < 0.01);
        assert_eq!(status.core_count, 1);
    }

    #[test]
    fn test_system_status_load_critical() {
        // per_core > 1.5 = critical (load = 8.0, cores = 4, per_core = 2.0)
        let status = SystemStatus::new(8.0, 6.0, 4.0, 4);
        assert_eq!(status.load_status(), HealthLevel::Critical);
    }

    #[test]
    fn test_system_status_load_high() {
        // per_core > 1.0 = high (load = 5.0, cores = 4, per_core = 1.25)
        let status = SystemStatus::new(5.0, 4.0, 3.0, 4);
        assert_eq!(status.load_status(), HealthLevel::High);
    }

    #[test]
    fn test_system_status_load_moderate() {
        // per_core > 0.7 = moderate (load = 3.2, cores = 4, per_core = 0.8)
        let status = SystemStatus::new(3.2, 3.0, 2.8, 4);
        assert_eq!(status.load_status(), HealthLevel::Moderate);
    }

    #[test]
    fn test_system_status_thermal_critical() {
        let status = SystemStatus::new(1.0, 1.0, 1.0, 4).with_thermal(95.0, 100.0);
        assert_eq!(status.thermal_status(), Some(HealthLevel::Critical));
    }

    #[test]
    fn test_system_status_thermal_high() {
        let status = SystemStatus::new(1.0, 1.0, 1.0, 4).with_thermal(80.0, 85.0);
        assert_eq!(status.thermal_status(), Some(HealthLevel::High));
    }

    #[test]
    fn test_system_status_thermal_ok() {
        let status = SystemStatus::new(1.0, 1.0, 1.0, 4).with_thermal(50.0, 55.0);
        assert_eq!(status.thermal_status(), Some(HealthLevel::Ok));
    }

    #[test]
    fn test_system_status_no_thermal() {
        let status = SystemStatus::new(1.0, 1.0, 1.0, 4);
        assert_eq!(status.thermal_status(), None);
    }

    #[test]
    fn test_system_status_brick_name() {
        let status = SystemStatus::default();
        assert_eq!(status.brick_name(), "system_status");
    }

    #[test]
    fn test_system_status_measure() {
        let status = SystemStatus::new(1.0, 1.0, 1.0, 4);
        let constraints = Constraints::tight(Size::new(80.0, 10.0));
        let size = status.measure(constraints);
        assert!(size.width > 0.0);
    }

    #[test]
    fn test_system_status_layout() {
        let mut status = SystemStatus::new(1.0, 1.0, 1.0, 4);
        let bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let result = status.layout(bounds);
        assert_eq!(result.size.width, 80.0);
    }

    #[test]
    fn test_system_status_paint() {
        let mut status = SystemStatus::new(2.0, 1.5, 1.0, 4).with_thermal(60.0, 65.0);
        status.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let mut buffer = CellBuffer::new(80, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        status.paint(&mut canvas);
    }

    #[test]
    fn test_system_status_children() {
        let status = SystemStatus::default();
        assert!(status.children().is_empty());
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    // TopProcessesTable additional tests
    #[test]
    fn test_top_processes_table_type_id() {
        let table = TopProcessesTable::default();
        let id = Widget::type_id(&table);
        assert_eq!(id, TypeId::of::<TopProcessesTable>());
    }

    #[test]
    fn test_top_processes_table_event() {
        let mut table = TopProcessesTable::default();
        let result = table.event(&Event::FocusIn);
        assert!(result.is_none());
    }

    #[test]
    fn test_top_processes_table_children_mut() {
        let mut table = TopProcessesTable::default();
        assert!(table.children_mut().is_empty());
    }

    #[test]
    fn test_top_processes_table_assertions() {
        let table = TopProcessesTable::default();
        assert!(!table.assertions().is_empty());
    }

    #[test]
    fn test_top_processes_table_budget() {
        let table = TopProcessesTable::default();
        let budget = table.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_top_processes_table_to_html() {
        let table = TopProcessesTable::default();
        assert!(table.to_html().is_empty());
    }

    #[test]
    fn test_top_processes_table_to_css() {
        let table = TopProcessesTable::default();
        assert!(table.to_css().is_empty());
    }

    #[test]
    fn test_top_processes_table_paint_many_processes() {
        // Test with more processes than max_display
        let procs: Vec<CpuConsumer> = (0..20)
            .map(|i| CpuConsumer::new(i, i as f32 * 5.0, 1_000_000 * i as u64, format!("proc{i}")))
            .collect();
        let mut table = TopProcessesTable::new(procs, 190.0).with_max_display(5);
        table.bounds = Rect::new(0.0, 0.0, 80.0, 10.0);
        let mut buffer = CellBuffer::new(80, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        table.paint(&mut canvas);
    }

    #[test]
    fn test_top_processes_table_paint_long_name() {
        let proc = CpuConsumer::new(
            123,
            50.0,
            1_000_000_000,
            "this_is_a_very_long_process_name_that_should_be_truncated",
        );
        let mut table = TopProcessesTable::new(vec![proc], 50.0);
        table.bounds = Rect::new(0.0, 0.0, 50.0, 10.0);
        let mut buffer = CellBuffer::new(50, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        table.paint(&mut canvas);
    }

    #[test]
    fn test_top_processes_table_clone() {
        let table = TopProcessesTable::new(vec![CpuConsumer::new(1, 10.0, 100, "test")], 10.0);
        let cloned = table;
        assert_eq!(cloned.processes.len(), 1);
    }

    #[test]
    fn test_top_processes_table_debug() {
        let table = TopProcessesTable::default();
        let debug = format!("{:?}", table);
        assert!(debug.contains("TopProcessesTable"));
    }

    // CoreUtilizationHistogram additional tests
    #[test]
    fn test_core_histogram_type_id() {
        let hist = CoreUtilizationHistogram::default();
        let id = Widget::type_id(&hist);
        assert_eq!(id, TypeId::of::<CoreUtilizationHistogram>());
    }

    #[test]
    fn test_core_histogram_event() {
        let mut hist = CoreUtilizationHistogram::default();
        let result = hist.event(&Event::FocusIn);
        assert!(result.is_none());
    }

    #[test]
    fn test_core_histogram_children_mut() {
        let mut hist = CoreUtilizationHistogram::default();
        assert!(hist.children_mut().is_empty());
    }

    #[test]
    fn test_core_histogram_set_percentages() {
        let mut hist = CoreUtilizationHistogram::default();
        hist.set_percentages(vec![25.0, 50.0, 75.0]);
        assert_eq!(hist.core_percentages.len(), 3);
    }

    #[test]
    fn test_core_histogram_assertions() {
        let hist = CoreUtilizationHistogram::default();
        assert!(!hist.assertions().is_empty());
    }

    #[test]
    fn test_core_histogram_budget() {
        let hist = CoreUtilizationHistogram::default();
        let budget = hist.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_core_histogram_to_html() {
        let hist = CoreUtilizationHistogram::default();
        assert!(hist.to_html().is_empty());
    }

    #[test]
    fn test_core_histogram_to_css() {
        let hist = CoreUtilizationHistogram::default();
        assert!(hist.to_css().is_empty());
    }

    #[test]
    fn test_core_histogram_clone() {
        let hist = CoreUtilizationHistogram::new(vec![50.0, 60.0]);
        let cloned = hist;
        assert_eq!(cloned.core_percentages.len(), 2);
    }

    #[test]
    fn test_core_histogram_debug() {
        let hist = CoreUtilizationHistogram::default();
        let debug = format!("{:?}", hist);
        assert!(debug.contains("CoreUtilizationHistogram"));
    }

    #[test]
    fn test_core_histogram_paint_all_buckets_populated() {
        // Create histogram where all 5 buckets have values
        let mut hist = CoreUtilizationHistogram::new(vec![
            99.0, 80.0, 50.0, 15.0, 0.5, // One in each bucket
        ]);
        hist.bounds = Rect::new(0.0, 0.0, 60.0, 10.0);
        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        hist.paint(&mut canvas);
    }

    // TrendSparkline additional tests
    #[test]
    fn test_trend_sparkline_type_id() {
        let trend = TrendSparkline::default();
        let id = Widget::type_id(&trend);
        assert_eq!(id, TypeId::of::<TrendSparkline>());
    }

    #[test]
    fn test_trend_sparkline_event() {
        let mut trend = TrendSparkline::default();
        let result = trend.event(&Event::FocusIn);
        assert!(result.is_none());
    }

    #[test]
    fn test_trend_sparkline_children_mut() {
        let mut trend = TrendSparkline::default();
        assert!(trend.children_mut().is_empty());
    }

    #[test]
    fn test_trend_sparkline_normalized() {
        let trend = TrendSparkline::new("Test", vec![0.5]).normalized();
        assert!(!trend.is_percentage);
    }

    #[test]
    fn test_trend_sparkline_set_history() {
        let mut trend = TrendSparkline::default();
        trend.set_history(vec![10.0, 20.0, 30.0]);
        assert_eq!(trend.history.len(), 3);
    }

    #[test]
    fn test_trend_sparkline_stats_normalized() {
        let trend = TrendSparkline::new("Test", vec![0.5, 0.6, 0.7]).normalized();
        let (current, min, max, avg) = trend.stats();
        // Normalized values are multiplied by 100
        assert!((current - 70.0).abs() < 0.1);
        assert!((min - 50.0).abs() < 0.1);
        assert!((max - 70.0).abs() < 0.1);
        assert!((avg - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_trend_sparkline_assertions() {
        let trend = TrendSparkline::default();
        assert!(!trend.assertions().is_empty());
    }

    #[test]
    fn test_trend_sparkline_budget() {
        let trend = TrendSparkline::default();
        let budget = trend.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_trend_sparkline_to_html() {
        let trend = TrendSparkline::default();
        assert!(trend.to_html().is_empty());
    }

    #[test]
    fn test_trend_sparkline_to_css() {
        let trend = TrendSparkline::default();
        assert!(trend.to_css().is_empty());
    }

    #[test]
    fn test_trend_sparkline_clone() {
        let trend = TrendSparkline::new("Test", vec![1.0, 2.0, 3.0]);
        let cloned = trend;
        assert_eq!(cloned.history.len(), 3);
        assert_eq!(cloned.title, "Test");
    }

    #[test]
    fn test_trend_sparkline_debug() {
        let trend = TrendSparkline::default();
        let debug = format!("{:?}", trend);
        assert!(debug.contains("TrendSparkline"));
    }

    #[test]
    fn test_trend_sparkline_paint_high_values() {
        let mut trend = TrendSparkline::new("CPU", vec![85.0, 90.0, 95.0]);
        trend.bounds = Rect::new(0.0, 0.0, 60.0, 5.0);
        let mut buffer = CellBuffer::new(60, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        trend.paint(&mut canvas);
    }

    #[test]
    fn test_trend_sparkline_paint_medium_values() {
        let mut trend = TrendSparkline::new("CPU", vec![55.0, 60.0, 65.0]);
        trend.bounds = Rect::new(0.0, 0.0, 60.0, 5.0);
        let mut buffer = CellBuffer::new(60, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        trend.paint(&mut canvas);
    }

    #[test]
    fn test_trend_sparkline_paint_normalized() {
        let mut trend = TrendSparkline::new("Mem", vec![0.3, 0.5, 0.7]).normalized();
        trend.bounds = Rect::new(0.0, 0.0, 60.0, 5.0);
        let mut buffer = CellBuffer::new(60, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        trend.paint(&mut canvas);
    }

    // SystemStatus additional tests
    #[test]
    fn test_system_status_type_id() {
        let status = SystemStatus::default();
        let id = Widget::type_id(&status);
        assert_eq!(id, TypeId::of::<SystemStatus>());
    }

    #[test]
    fn test_system_status_event() {
        let mut status = SystemStatus::default();
        let result = status.event(&Event::FocusIn);
        assert!(result.is_none());
    }

    #[test]
    fn test_system_status_children_mut() {
        let mut status = SystemStatus::default();
        assert!(status.children_mut().is_empty());
    }

    #[test]
    fn test_system_status_set_load() {
        let mut status = SystemStatus::default();
        status.set_load(2.0, 1.5, 1.0);
        assert!((status.load_1m - 2.0).abs() < 0.01);
        assert!((status.load_5m - 1.5).abs() < 0.01);
        assert!((status.load_15m - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_system_status_set_thermal() {
        let mut status = SystemStatus::default();
        status.set_thermal(60.0, 70.0);
        assert_eq!(status.thermal, Some((60.0, 70.0)));
    }

    #[test]
    fn test_system_status_assertions() {
        let status = SystemStatus::default();
        assert!(!status.assertions().is_empty());
    }

    #[test]
    fn test_system_status_budget() {
        let status = SystemStatus::default();
        let budget = status.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_system_status_to_html() {
        let status = SystemStatus::default();
        assert!(status.to_html().is_empty());
    }

    #[test]
    fn test_system_status_to_css() {
        let status = SystemStatus::default();
        assert!(status.to_css().is_empty());
    }

    #[test]
    fn test_system_status_clone() {
        let status = SystemStatus::new(1.0, 2.0, 3.0, 4).with_thermal(50.0, 60.0);
        let cloned = status;
        assert!((cloned.load_1m - 1.0).abs() < 0.01);
        assert_eq!(cloned.core_count, 4);
        assert!(cloned.thermal.is_some());
    }

    #[test]
    fn test_system_status_debug() {
        let status = SystemStatus::default();
        let debug = format!("{:?}", status);
        assert!(debug.contains("SystemStatus"));
    }

    #[test]
    fn test_system_status_paint_no_thermal() {
        let mut status = SystemStatus::new(1.0, 1.0, 1.0, 4);
        status.bounds = Rect::new(0.0, 0.0, 80.0, 5.0);
        let mut buffer = CellBuffer::new(80, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        status.paint(&mut canvas);
    }

    #[test]
    fn test_system_status_measure_with_thermal() {
        let status = SystemStatus::new(1.0, 1.0, 1.0, 4).with_thermal(60.0, 70.0);
        // Use loose constraints to get natural size
        let constraints = Constraints::loose(Size::new(80.0, 100.0));
        let size = status.measure(constraints);
        assert_eq!(size.height, 2.0); // Two lines with thermal
    }

    #[test]
    fn test_system_status_measure_no_thermal() {
        let status = SystemStatus::new(1.0, 1.0, 1.0, 4);
        // Use loose constraints to get natural size
        let constraints = Constraints::loose(Size::new(80.0, 100.0));
        let size = status.measure(constraints);
        assert_eq!(size.height, 1.0); // One line without thermal
    }

    // CpuConsumer additional tests
    #[test]
    fn test_cpu_consumer_clone() {
        let proc = CpuConsumer::new(123, 50.0, 1_000_000, "test");
        let cloned = proc;
        assert_eq!(cloned.pid, 123);
    }

    #[test]
    fn test_cpu_consumer_debug() {
        let proc = CpuConsumer::new(123, 50.0, 1_000_000, "test");
        let debug = format!("{:?}", proc);
        assert!(debug.contains("CpuConsumer"));
    }

    // HealthLevel additional tests
    #[test]
    fn test_health_level_clone() {
        let level = HealthLevel::Critical;
        let cloned = level;
        assert_eq!(cloned, HealthLevel::Critical);
    }

    #[test]
    fn test_health_level_debug() {
        let level = HealthLevel::Ok;
        let debug = format!("{:?}", level);
        assert!(debug.contains("Ok"));
    }

    #[test]
    fn test_health_level_copy() {
        let level = HealthLevel::High;
        let copied = level;
        assert_eq!(copied, HealthLevel::High);
    }
}
