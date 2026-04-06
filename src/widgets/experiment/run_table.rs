//! Run table widget for displaying experiment run status.
//!
//! A sortable table showing experiment runs with their status, duration, and metrics.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;

/// Status of an experiment run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RunStatus {
    /// Run is queued but not started.
    Pending,
    /// Run is currently executing.
    Running,
    /// Run finished successfully.
    Completed,
    /// Run terminated with an error.
    Failed,
}

impl RunStatus {
    /// Get a display character/emoji for the status.
    #[must_use]
    pub fn indicator(&self) -> &'static str {
        match self {
            Self::Pending => "\u{23F3}",   // ⏳
            Self::Running => "\u{25B6}",   // ▶
            Self::Completed => "\u{2705}", // ✅
            Self::Failed => "\u{274C}",    // ❌
        }
    }

    /// Check if the run is in a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed)
    }

    /// Check if the run is active (pending or running).
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Pending | Self::Running)
    }
}

impl fmt::Display for RunStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Pending => "Pending",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
        };
        write!(f, "{s}")
    }
}

/// A single row in the run table.
#[derive(Debug, Clone)]
pub struct RunRow {
    /// Unique identifier for the run.
    pub id: String,
    /// Current status.
    pub status: RunStatus,
    /// Duration in seconds (None if not started or still running).
    pub duration: Option<f64>,
    /// Arbitrary metrics (e.g., "loss" -> 0.05, "accuracy" -> 0.95).
    pub metrics: HashMap<String, f64>,
}

impl RunRow {
    /// Create a new run row.
    #[must_use]
    pub fn new(id: impl Into<String>, status: RunStatus) -> Self {
        Self { id: id.into(), status, duration: None, metrics: HashMap::new() }
    }

    /// Set the duration.
    #[must_use]
    pub fn with_duration(mut self, seconds: f64) -> Self {
        self.duration = Some(seconds);
        self
    }

    /// Add a metric.
    #[must_use]
    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }

    /// Get a metric value.
    #[must_use]
    pub fn metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// Format duration as human-readable string.
    #[must_use]
    pub fn duration_display(&self) -> String {
        match self.duration {
            Some(secs) if secs >= 3600.0 => {
                let hours = secs / 3600.0;
                format!("{hours:.1}h")
            }
            Some(secs) if secs >= 60.0 => {
                let mins = secs / 60.0;
                format!("{mins:.1}m")
            }
            Some(secs) => format!("{secs:.1}s"),
            None => "-".to_string(),
        }
    }
}

/// Column to sort the run table by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortColumn {
    /// Sort by run ID.
    #[default]
    Id,
    /// Sort by status.
    Status,
    /// Sort by duration.
    Duration,
    /// Sort by a specific metric (index into metric names).
    Metric(usize),
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortDirection {
    /// Ascending order (A-Z, 0-9).
    #[default]
    Ascending,
    /// Descending order (Z-A, 9-0).
    Descending,
}

/// A sortable table of experiment runs.
#[derive(Debug, Clone)]
pub struct RunTable {
    /// The run rows.
    runs: Vec<RunRow>,
    /// Metric column names (for rendering headers).
    metric_columns: Vec<String>,
    /// Current sort column.
    sort_column: SortColumn,
    /// Current sort direction.
    sort_direction: SortDirection,
}

impl Default for RunTable {
    fn default() -> Self {
        Self::new()
    }
}

impl RunTable {
    /// Create a new empty run table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            metric_columns: Vec::new(),
            sort_column: SortColumn::Id,
            sort_direction: SortDirection::Ascending,
        }
    }

    /// Create a run table from a list of runs.
    #[must_use]
    pub fn from_runs(runs: Vec<RunRow>) -> Self {
        // Extract all unique metric names
        let mut metric_names: Vec<String> =
            runs.iter().flat_map(|r| r.metrics.keys().cloned()).collect();
        metric_names.sort();
        metric_names.dedup();

        let mut table = Self {
            runs,
            metric_columns: metric_names,
            sort_column: SortColumn::Id,
            sort_direction: SortDirection::Ascending,
        };
        table.apply_sort();
        table
    }

    /// Add a run to the table.
    pub fn add_run(&mut self, run: RunRow) {
        // Update metric columns
        for key in run.metrics.keys() {
            if !self.metric_columns.contains(key) {
                self.metric_columns.push(key.clone());
                self.metric_columns.sort();
            }
        }
        self.runs.push(run);
    }

    /// Get the runs (in current sort order).
    #[must_use]
    pub fn runs(&self) -> &[RunRow] {
        &self.runs
    }

    /// Get the metric column names.
    #[must_use]
    pub fn metric_columns(&self) -> &[String] {
        &self.metric_columns
    }

    /// Get the number of runs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.runs.len()
    }

    /// Check if the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Sort the table by the given column.
    pub fn sort_by(&mut self, column: SortColumn) {
        // If clicking the same column, toggle direction
        if self.sort_column == column {
            self.sort_direction = match self.sort_direction {
                SortDirection::Ascending => SortDirection::Descending,
                SortDirection::Descending => SortDirection::Ascending,
            };
        } else {
            self.sort_column = column;
            self.sort_direction = SortDirection::Ascending;
        }

        self.apply_sort();
    }

    /// Sort by a metric column name.
    pub fn sort_by_metric(&mut self, metric_name: &str) {
        if let Some(idx) = self.metric_columns.iter().position(|n| n == metric_name) {
            self.sort_by(SortColumn::Metric(idx));
        }
    }

    /// Apply the current sort settings.
    fn apply_sort(&mut self) {
        let metric_columns = &self.metric_columns;
        let sort_column = self.sort_column;
        let ascending = self.sort_direction == SortDirection::Ascending;

        self.runs.sort_by(|a, b| {
            let cmp = match sort_column {
                SortColumn::Id => a.id.cmp(&b.id),
                SortColumn::Status => status_order(a.status).cmp(&status_order(b.status)),
                SortColumn::Duration => {
                    let a_dur = a.duration.unwrap_or(f64::MAX);
                    let b_dur = b.duration.unwrap_or(f64::MAX);
                    a_dur.partial_cmp(&b_dur).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortColumn::Metric(idx) => {
                    let metric_name = metric_columns.get(idx).map(String::as_str);
                    let a_val = metric_name.and_then(|n| a.metrics.get(n)).unwrap_or(&f64::MAX);
                    let b_val = metric_name.and_then(|n| b.metrics.get(n)).unwrap_or(&f64::MAX);
                    a_val.partial_cmp(b_val).unwrap_or(std::cmp::Ordering::Equal)
                }
            };

            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });
    }

    /// Get the current sort column.
    #[must_use]
    pub fn sort_column(&self) -> SortColumn {
        self.sort_column
    }

    /// Get the current sort direction.
    #[must_use]
    pub fn sort_direction(&self) -> SortDirection {
        self.sort_direction
    }

    /// Count runs by status.
    #[must_use]
    pub fn status_counts(&self) -> HashMap<RunStatus, usize> {
        let mut counts = HashMap::new();
        for run in &self.runs {
            *counts.entry(run.status).or_insert(0) += 1;
        }
        counts
    }

    /// Get runs filtered by status.
    #[must_use]
    pub fn filter_by_status(&self, status: RunStatus) -> Vec<&RunRow> {
        self.runs.iter().filter(|r| r.status == status).collect()
    }

    /// Render the table as a formatted string (for terminal display).
    #[must_use]
    pub fn render(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str("| ID | Status | Duration |");
        for col in &self.metric_columns {
            let _ = write!(output, " {col} |");
        }
        output.push('\n');

        // Separator
        output.push_str("|----|---------|---------");
        for _ in &self.metric_columns {
            output.push_str("|---------");
        }
        output.push_str("|\n");

        // Rows
        for run in &self.runs {
            let _ = write!(
                output,
                "| {} | {} {} | {} |",
                run.id,
                run.status.indicator(),
                run.status,
                run.duration_display()
            );
            for col in &self.metric_columns {
                let value = run.metrics.get(col).map_or("-".to_string(), |v| format!("{v:.4}"));
                let _ = write!(output, " {value} |");
            }
            output.push('\n');
        }

        output
    }
}

/// Convert status to numeric order for sorting.
fn status_order(status: RunStatus) -> u8 {
    match status {
        RunStatus::Running => 0,
        RunStatus::Pending => 1,
        RunStatus::Completed => 2,
        RunStatus::Failed => 3,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_status_display() {
        assert_eq!(RunStatus::Pending.to_string(), "Pending");
        assert_eq!(RunStatus::Running.to_string(), "Running");
        assert_eq!(RunStatus::Completed.to_string(), "Completed");
        assert_eq!(RunStatus::Failed.to_string(), "Failed");
    }

    #[test]
    fn test_run_status_indicator() {
        // Just verify they return valid UTF-8
        assert!(!RunStatus::Pending.indicator().is_empty());
        assert!(!RunStatus::Running.indicator().is_empty());
        assert!(!RunStatus::Completed.indicator().is_empty());
        assert!(!RunStatus::Failed.indicator().is_empty());
    }

    #[test]
    fn test_run_status_terminal() {
        assert!(!RunStatus::Pending.is_terminal());
        assert!(!RunStatus::Running.is_terminal());
        assert!(RunStatus::Completed.is_terminal());
        assert!(RunStatus::Failed.is_terminal());
    }

    #[test]
    fn test_run_row_creation() {
        let row = RunRow::new("run-001", RunStatus::Running)
            .with_duration(3600.0)
            .with_metric("loss", 0.05)
            .with_metric("accuracy", 0.95);

        assert_eq!(row.id, "run-001");
        assert_eq!(row.status, RunStatus::Running);
        assert_eq!(row.duration, Some(3600.0));
        assert_eq!(row.metric("loss"), Some(0.05));
        assert_eq!(row.metric("accuracy"), Some(0.95));
        assert_eq!(row.metric("unknown"), None);
    }

    #[test]
    fn test_run_row_duration_display() {
        let row = RunRow::new("r1", RunStatus::Completed);
        assert_eq!(row.duration_display(), "-");

        let row = RunRow::new("r2", RunStatus::Completed).with_duration(30.0);
        assert_eq!(row.duration_display(), "30.0s");

        let row = RunRow::new("r3", RunStatus::Completed).with_duration(120.0);
        assert_eq!(row.duration_display(), "2.0m");

        let row = RunRow::new("r4", RunStatus::Completed).with_duration(7200.0);
        assert_eq!(row.duration_display(), "2.0h");
    }

    #[test]
    fn test_run_table_sorting() {
        let runs = vec![
            RunRow::new("c", RunStatus::Completed).with_duration(100.0),
            RunRow::new("a", RunStatus::Running).with_duration(50.0),
            RunRow::new("b", RunStatus::Pending).with_duration(200.0),
        ];

        let mut table = RunTable::from_runs(runs);

        // from_runs initializes with Id/Ascending sort already applied
        assert_eq!(table.runs()[0].id, "a");
        assert_eq!(table.runs()[1].id, "b");
        assert_eq!(table.runs()[2].id, "c");

        // Sort by ID again (toggles to descending)
        table.sort_by(SortColumn::Id);
        assert_eq!(table.runs()[0].id, "c");
        assert_eq!(table.runs()[1].id, "b");
        assert_eq!(table.runs()[2].id, "a");

        // Sort by duration (new column, starts ascending)
        table.sort_by(SortColumn::Duration);
        assert_eq!(table.runs()[0].duration, Some(50.0));
        assert_eq!(table.runs()[1].duration, Some(100.0));
        assert_eq!(table.runs()[2].duration, Some(200.0));
    }

    #[test]
    fn test_run_table_metric_sorting() {
        let runs = vec![
            RunRow::new("r1", RunStatus::Completed).with_metric("loss", 0.5),
            RunRow::new("r2", RunStatus::Completed).with_metric("loss", 0.1),
            RunRow::new("r3", RunStatus::Completed).with_metric("loss", 0.3),
        ];

        let mut table = RunTable::from_runs(runs);
        table.sort_by_metric("loss");

        assert_eq!(table.runs()[0].id, "r2"); // 0.1
        assert_eq!(table.runs()[1].id, "r3"); // 0.3
        assert_eq!(table.runs()[2].id, "r1"); // 0.5
    }

    #[test]
    fn test_run_table_status_counts() {
        let runs = vec![
            RunRow::new("r1", RunStatus::Running),
            RunRow::new("r2", RunStatus::Completed),
            RunRow::new("r3", RunStatus::Completed),
            RunRow::new("r4", RunStatus::Failed),
        ];

        let table = RunTable::from_runs(runs);
        let counts = table.status_counts();

        assert_eq!(counts.get(&RunStatus::Running), Some(&1));
        assert_eq!(counts.get(&RunStatus::Completed), Some(&2));
        assert_eq!(counts.get(&RunStatus::Failed), Some(&1));
        assert_eq!(counts.get(&RunStatus::Pending), None);
    }

    #[test]
    fn test_run_table_render() {
        let runs = vec![
            RunRow::new("run-001", RunStatus::Completed)
                .with_duration(3600.0)
                .with_metric("loss", 0.05),
            RunRow::new("run-002", RunStatus::Running)
                .with_duration(1800.0)
                .with_metric("loss", 0.15),
        ];

        let table = RunTable::from_runs(runs);
        let rendered = table.render();

        assert!(rendered.contains("run-001"));
        assert!(rendered.contains("run-002"));
        assert!(rendered.contains("loss"));
    }

    #[test]
    fn test_run_status_is_active() {
        assert!(RunStatus::Pending.is_active());
        assert!(RunStatus::Running.is_active());
        assert!(!RunStatus::Completed.is_active());
        assert!(!RunStatus::Failed.is_active());
    }

    #[test]
    fn test_run_table_default() {
        let table = RunTable::default();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.metric_columns().is_empty());
    }

    #[test]
    fn test_run_table_add_run() {
        let mut table = RunTable::new();
        assert!(table.is_empty());

        table.add_run(RunRow::new("r1", RunStatus::Running).with_metric("loss", 0.5));
        assert_eq!(table.len(), 1);
        assert!(table.metric_columns().contains(&"loss".to_string()));

        // Add another run with a new metric
        table.add_run(RunRow::new("r2", RunStatus::Pending).with_metric("accuracy", 0.9));
        assert_eq!(table.len(), 2);
        assert!(table.metric_columns().contains(&"accuracy".to_string()));
    }

    #[test]
    fn test_run_table_sort_accessors() {
        let table = RunTable::new();
        assert_eq!(table.sort_column(), SortColumn::Id);
        assert_eq!(table.sort_direction(), SortDirection::Ascending);
    }

    #[test]
    fn test_run_table_sort_by_status() {
        let runs = vec![
            RunRow::new("r1", RunStatus::Completed),
            RunRow::new("r2", RunStatus::Running),
            RunRow::new("r3", RunStatus::Pending),
            RunRow::new("r4", RunStatus::Failed),
        ];

        let mut table = RunTable::from_runs(runs);
        table.sort_by(SortColumn::Status);

        // Order: Running=0, Pending=1, Completed=2, Failed=3
        assert_eq!(table.runs()[0].status, RunStatus::Running);
        assert_eq!(table.runs()[1].status, RunStatus::Pending);
        assert_eq!(table.runs()[2].status, RunStatus::Completed);
        assert_eq!(table.runs()[3].status, RunStatus::Failed);
    }

    #[test]
    fn test_run_table_filter_by_status() {
        let runs = vec![
            RunRow::new("r1", RunStatus::Running),
            RunRow::new("r2", RunStatus::Completed),
            RunRow::new("r3", RunStatus::Completed),
            RunRow::new("r4", RunStatus::Failed),
        ];

        let table = RunTable::from_runs(runs);

        let completed = table.filter_by_status(RunStatus::Completed);
        assert_eq!(completed.len(), 2);

        let pending = table.filter_by_status(RunStatus::Pending);
        assert!(pending.is_empty());
    }

    #[test]
    fn test_run_table_sort_by_nonexistent_metric() {
        let runs = vec![
            RunRow::new("r1", RunStatus::Completed).with_metric("loss", 0.5),
            RunRow::new("r2", RunStatus::Completed).with_metric("loss", 0.1),
        ];

        let mut table = RunTable::from_runs(runs);
        // Sort by a metric that doesn't exist - should do nothing
        table.sort_by_metric("nonexistent");
        // Order should remain as initialized (by id ascending)
        assert_eq!(table.runs()[0].id, "r1");
    }

    #[test]
    fn test_run_table_sort_with_none_durations() {
        let runs = vec![
            RunRow::new("r1", RunStatus::Completed).with_duration(100.0),
            RunRow::new("r2", RunStatus::Pending), // No duration
            RunRow::new("r3", RunStatus::Running).with_duration(50.0),
        ];

        let mut table = RunTable::from_runs(runs);
        table.sort_by(SortColumn::Duration);

        // r3 (50.0) < r1 (100.0) < r2 (MAX/None)
        assert_eq!(table.runs()[0].id, "r3");
        assert_eq!(table.runs()[1].id, "r1");
        assert_eq!(table.runs()[2].id, "r2");
    }

    #[test]
    fn test_run_table_sort_with_missing_metrics() {
        let runs = vec![
            RunRow::new("r1", RunStatus::Completed).with_metric("loss", 0.5),
            RunRow::new("r2", RunStatus::Completed), // No loss metric
            RunRow::new("r3", RunStatus::Completed).with_metric("loss", 0.1),
        ];

        let mut table = RunTable::from_runs(runs);
        table.sort_by_metric("loss");

        // r3 (0.1) < r1 (0.5) < r2 (MAX/missing)
        assert_eq!(table.runs()[0].id, "r3");
        assert_eq!(table.runs()[1].id, "r1");
        assert_eq!(table.runs()[2].id, "r2");
    }

    #[test]
    fn test_run_table_render_empty() {
        let table = RunTable::new();
        let rendered = table.render();
        // Should have header but no data rows
        assert!(rendered.contains("ID"));
        assert!(rendered.contains("Status"));
        assert!(rendered.contains("Duration"));
    }

    #[test]
    fn test_run_table_render_with_missing_metric() {
        let runs = vec![
            RunRow::new("r1", RunStatus::Completed).with_metric("loss", 0.05),
            RunRow::new("r2", RunStatus::Running), // No metrics
        ];

        let table = RunTable::from_runs(runs);
        let rendered = table.render();

        // r2 should show "-" for missing metric
        assert!(rendered.contains('-'));
        assert!(rendered.contains("0.0500"));
    }
}
