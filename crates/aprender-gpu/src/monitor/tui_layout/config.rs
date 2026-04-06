//! TUI configuration and section definitions.

use super::widgets::Widget;

// ============================================================================
// TUI Configuration
// ============================================================================

/// TUI layout configuration
#[derive(Debug, Clone)]
pub struct TuiLayout {
    /// Minimum terminal width
    pub min_width: u16,
    /// Minimum terminal height
    pub min_height: u16,
    /// Recommended terminal width
    pub rec_width: u16,
    /// Recommended terminal height
    pub rec_height: u16,
    /// Section definitions
    pub sections: Vec<Section>,
    /// Refresh rate in milliseconds
    pub refresh_rate_ms: u64,
    /// Number of sparkline data points
    pub sparkline_points: usize,
}

impl Default for TuiLayout {
    fn default() -> Self {
        Self {
            min_width: 80,
            min_height: 24,
            rec_width: 160,
            rec_height: 48,
            sections: vec![
                Section::new("compute", "COMPUTE", 0.25),
                Section::new("memory", "MEMORY", 0.20),
                Section::new("dataflow", "DATA FLOW", 0.20),
                Section::new("kernels", "KERNELS", 0.20),
            ],
            refresh_rate_ms: 100,
            sparkline_points: 60,
        }
    }
}

impl TuiLayout {
    /// Create a new TUI layout
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set refresh rate
    #[must_use]
    pub fn with_refresh_rate(mut self, ms: u64) -> Self {
        self.refresh_rate_ms = ms;
        self
    }

    /// Check if terminal size meets minimum requirements
    #[must_use]
    pub fn check_size(&self, width: u16, height: u16) -> SizeCheck {
        if width >= self.rec_width && height >= self.rec_height {
            SizeCheck::Recommended
        } else if width >= self.min_width && height >= self.min_height {
            SizeCheck::Minimum
        } else {
            SizeCheck::TooSmall
        }
    }
}

/// Terminal size check result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeCheck {
    /// Terminal meets recommended size
    Recommended,
    /// Terminal meets minimum size
    Minimum,
    /// Terminal too small
    TooSmall,
}

// ============================================================================
// Section Definition
// ============================================================================

/// TUI section definition
#[derive(Debug, Clone)]
pub struct Section {
    /// Section identifier
    pub id: String,
    /// Section title
    pub title: String,
    /// Height as percentage of total (0.0-1.0)
    pub height_pct: f32,
    /// Widgets in this section
    pub widgets: Vec<Widget>,
    /// Is section collapsed
    pub collapsed: bool,
    /// Is section focused
    pub focused: bool,
}

impl Section {
    /// Create a new section
    #[must_use]
    pub fn new(id: impl Into<String>, title: impl Into<String>, height_pct: f32) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            height_pct,
            widgets: Vec::new(),
            collapsed: false,
            focused: false,
        }
    }

    /// Add a widget to the section
    pub fn add_widget(&mut self, widget: Widget) {
        self.widgets.push(widget);
    }

    /// Toggle collapsed state
    pub fn toggle_collapsed(&mut self) {
        self.collapsed = !self.collapsed;
    }
}
