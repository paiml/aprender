//! TUI widget types for compute monitoring.

use std::collections::VecDeque;

// ============================================================================
// Widget Types
// ============================================================================

/// TUI widget
#[derive(Debug, Clone)]
pub enum Widget {
    /// Progress gauge (0-100%)
    Gauge(GaugeWidget),
    /// Sparkline (history graph)
    Sparkline(SparklineWidget),
    /// Progress bar with label
    ProgressBar(ProgressBarWidget),
    /// Table with rows
    Table(TableWidget),
    /// Text label
    Text(TextWidget),
}

/// Gauge widget for showing percentages
#[derive(Debug, Clone)]
pub struct GaugeWidget {
    /// Widget label
    pub label: String,
    /// Current value (0.0-100.0)
    pub value_pct: f64,
    /// Warning threshold
    pub warning_threshold: f64,
    /// Critical threshold
    pub critical_threshold: f64,
    /// Maximum value (usually 100)
    pub max_value: f64,
    /// Suffix text (e.g., "%" or "°C")
    pub suffix: String,
}

impl GaugeWidget {
    /// Create a new gauge
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value_pct: 0.0,
            warning_threshold: 70.0,
            critical_threshold: 90.0,
            max_value: 100.0,
            suffix: "%".to_string(),
        }
    }

    /// Set value
    #[must_use]
    pub fn with_value(mut self, value: f64) -> Self {
        self.value_pct = value;
        self
    }

    /// Set thresholds
    #[must_use]
    pub fn with_thresholds(mut self, warning: f64, critical: f64) -> Self {
        self.warning_threshold = warning;
        self.critical_threshold = critical;
        self
    }

    /// Get color based on value
    #[must_use]
    pub fn color(&self) -> GaugeColor {
        if self.value_pct >= self.critical_threshold {
            GaugeColor::Critical
        } else if self.value_pct >= self.warning_threshold {
            GaugeColor::Warning
        } else {
            GaugeColor::Ok
        }
    }

    /// Render as ASCII bar
    #[must_use]
    pub fn render_bar(&self, width: usize) -> String {
        let ratio = (self.value_pct / self.max_value).min(1.0);
        let filled = (ratio * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);

        format!(
            "{}: [{}{}] {:.1}{}",
            self.label,
            "\u{2588}".repeat(filled),
            "\u{2591}".repeat(empty),
            self.value_pct,
            self.suffix
        )
    }
}

/// Gauge color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaugeColor {
    /// Normal (green)
    Ok,
    /// Warning (yellow/orange)
    Warning,
    /// Critical (red)
    Critical,
}

/// Sparkline widget for showing history
#[derive(Debug, Clone)]
pub struct SparklineWidget {
    /// Data points (0.0-1.0 normalized or raw values)
    pub data: VecDeque<f64>,
    /// Widget label
    pub label: String,
    /// Optional baseline to show
    pub baseline: Option<f64>,
    /// Auto-scale to data range
    pub auto_scale: bool,
}

impl SparklineWidget {
    /// Create a new sparkline
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            data: VecDeque::with_capacity(60),
            label: label.into(),
            baseline: None,
            auto_scale: true,
        }
    }

    /// Set data
    #[must_use]
    pub fn with_data(mut self, data: VecDeque<f64>) -> Self {
        self.data = data;
        self
    }

    /// Render as Unicode sparkline
    #[must_use]
    pub fn render(&self) -> String {
        if self.data.is_empty() {
            return String::new();
        }

        let blocks = [
            '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
            '\u{2588}',
        ];

        let (min, max) = if self.auto_scale {
            let min = self.data.iter().copied().fold(f64::INFINITY, f64::min);
            let max = self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        } else {
            (0.0, 100.0)
        };

        let range = (max - min).max(0.001);

        self.data
            .iter()
            .map(|&v| {
                let normalized = ((v - min) / range).clamp(0.0, 1.0);
                let idx = (normalized * 7.0).round() as usize;
                blocks[idx.min(7)]
            })
            .collect()
    }
}

/// Progress bar widget
#[derive(Debug, Clone)]
pub struct ProgressBarWidget {
    /// Widget label
    pub label: String,
    /// Progress (0.0-1.0)
    pub progress: f64,
    /// Total description (e.g., "14.0 / 24.0 GB")
    pub total_desc: String,
}

impl ProgressBarWidget {
    /// Create a new progress bar
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self { label: label.into(), progress: 0.0, total_desc: String::new() }
    }

    /// Set progress
    #[must_use]
    pub fn with_progress(mut self, progress: f64) -> Self {
        self.progress = progress.clamp(0.0, 1.0);
        self
    }

    /// Set total description
    #[must_use]
    pub fn with_total(mut self, desc: impl Into<String>) -> Self {
        self.total_desc = desc.into();
        self
    }

    /// Render as ASCII bar
    #[must_use]
    pub fn render(&self, width: usize) -> String {
        let filled = (self.progress * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);

        format!(
            "{}: [{}{}] {}",
            self.label,
            "\u{2588}".repeat(filled),
            "\u{2591}".repeat(empty),
            self.total_desc
        )
    }
}

/// Table widget
#[derive(Debug, Clone)]
pub struct TableWidget {
    /// Column headers
    pub headers: Vec<String>,
    /// Row data
    pub rows: Vec<Vec<String>>,
    /// Currently highlighted row
    pub highlight_row: Option<usize>,
    /// Column widths (auto-calculated if empty)
    pub column_widths: Vec<usize>,
}

impl TableWidget {
    /// Create a new table
    #[must_use]
    pub fn new(headers: Vec<String>) -> Self {
        Self { headers, rows: Vec::new(), highlight_row: None, column_widths: Vec::new() }
    }

    /// Add a row
    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    /// Set highlighted row
    pub fn highlight(&mut self, row: usize) {
        self.highlight_row = Some(row);
    }

    /// Calculate column widths
    #[must_use]
    pub fn calculate_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();

        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        widths
    }
}

/// Text widget
#[derive(Debug, Clone)]
pub struct TextWidget {
    /// Text content
    pub content: String,
    /// Text style
    pub style: TextStyle,
}

impl TextWidget {
    /// Create a new text widget
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self { content: content.into(), style: TextStyle::Normal }
    }

    /// Set style
    #[must_use]
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }
}

/// Text style
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextStyle {
    /// Normal text
    Normal,
    /// Bold text
    Bold,
    /// Dimmed text
    Dim,
    /// Italic text
    Italic,
    /// Header text
    Header,
    /// Error text
    Error,
    /// Warning text
    Warning,
    /// Success text
    Success,
}
