//! REPL Session State Management (ALIM-REPL-001)
//!
//! Stateful session that holds loaded datasets in memory, eliminating
//! the "Muda of Processing" (re-loading large datasets for every minor query).

use std::{collections::HashMap, fmt::Write, path::Path, sync::Arc};

use super::commands::ReplCommand;
use crate::{
    dataset::Dataset,
    quality::{LetterGrade, QualityChecker, QualityScore},
    ArrowDataset, Error, Result,
};

/// Display configuration for REPL output (Mieruka - Visual Control)
#[derive(Debug, Clone)]
pub struct DisplayConfig {
    /// Maximum rows to display in head/tail
    pub max_rows: usize,
    /// Maximum column width before truncation
    pub max_column_width: usize,
    /// Enable color output (Andon)
    pub color_output: bool,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            max_rows: 10,
            max_column_width: 50,
            color_output: true,
        }
    }
}

impl DisplayConfig {
    /// Set maximum rows to display
    #[must_use]
    pub fn with_max_rows(mut self, rows: usize) -> Self {
        self.max_rows = rows;
        self
    }

    /// Set maximum column width
    #[must_use]
    pub fn with_max_column_width(mut self, width: usize) -> Self {
        self.max_column_width = width;
        self
    }

    /// Enable/disable color output
    #[must_use]
    pub fn with_color(mut self, enabled: bool) -> Self {
        self.color_output = enabled;
        self
    }
}

/// Cached quality score for the active dataset
#[derive(Debug, Clone)]
pub struct QualityCache {
    /// The computed quality score
    pub score: QualityScore,
    /// When the cache was last updated
    pub timestamp: std::time::Instant,
}

/// Stateful REPL session (ALIM-REPL-001)
///
/// Prevents reload waste by keeping datasets in memory.
/// Maintains quality cache for instant Andon display.
#[derive(Debug)]
pub struct ReplSession {
    /// Loaded datasets keyed by name
    datasets: HashMap<String, Arc<ArrowDataset>>,
    /// Currently active dataset name
    active_name: Option<String>,
    /// Command history for reproducibility (ALIM-REPL-006)
    history: Vec<String>,
    /// Display configuration
    pub config: DisplayConfig,
    /// Quality score cache for Andon prompt
    quality_cache: Option<QualityCache>,
}

impl Default for ReplSession {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplSession {
    /// Create a new empty session
    #[must_use]
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
            active_name: None,
            history: Vec::new(),
            config: DisplayConfig::default(),
            quality_cache: None,
        }
    }

    /// Load a dataset into the session
    ///
    /// The dataset becomes the active dataset and quality is computed.
    pub fn load_dataset(&mut self, name: &str, dataset: ArrowDataset) {
        // Compute quality score for Andon display
        let score = self.compute_quality(&dataset);

        let arc_dataset = Arc::new(dataset);
        self.datasets.insert(name.to_string(), arc_dataset);
        self.active_name = Some(name.to_string());

        if let Some(score) = score {
            self.quality_cache = Some(QualityCache {
                score,
                timestamp: std::time::Instant::now(),
            });
        }
    }

    /// Compute quality score for a dataset
    fn compute_quality(&self, dataset: &ArrowDataset) -> Option<QualityScore> {
        let checker = QualityChecker::new();
        match checker.check(dataset) {
            Ok(report) => {
                // Build basic checklist from report
                let checklist = self.build_basic_checklist(&report);
                Some(QualityScore::from_checklist(checklist))
            }
            Err(_) => None,
        }
    }

    /// Build a basic checklist from quality report
    #[allow(clippy::unused_self)]
    fn build_basic_checklist(
        &self,
        report: &crate::quality::QualityReport,
    ) -> Vec<crate::quality::ChecklistItem> {
        use crate::quality::{ChecklistItem, Severity};

        let mut items = Vec::new();

        // Basic schema check (always passes if we got here)
        items.push(ChecklistItem::new(
            1,
            "Schema is readable",
            Severity::Critical,
            true,
        ));

        // Check for null columns
        let has_excessive_nulls = report.columns.values().any(|c| c.null_ratio > 0.5);
        items.push(ChecklistItem::new(
            2,
            "No excessive null ratios (>50%)",
            Severity::High,
            !has_excessive_nulls,
        ));

        // Check for duplicates
        let has_high_duplicates = report.columns.values().any(|c| c.duplicate_ratio > 0.3);
        items.push(ChecklistItem::new(
            3,
            "No high duplicate ratios (>30%)",
            Severity::Medium,
            !has_high_duplicates,
        ));

        // Check row count
        items.push(ChecklistItem::new(
            4,
            "Dataset has rows",
            Severity::Critical,
            report.row_count > 0,
        ));

        // Check column count
        items.push(ChecklistItem::new(
            5,
            "Dataset has columns",
            Severity::Critical,
            report.column_count > 0,
        ));

        items
    }

    /// Switch to a different loaded dataset
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset name is not found.
    pub fn use_dataset(&mut self, name: &str) -> Result<()> {
        if self.datasets.contains_key(name) {
            self.active_name = Some(name.to_string());

            // Recompute quality for the newly active dataset
            if let Some(dataset) = self.datasets.get(name) {
                if let Some(score) = self.compute_quality(dataset) {
                    self.quality_cache = Some(QualityCache {
                        score,
                        timestamp: std::time::Instant::now(),
                    });
                }
            }

            Ok(())
        } else {
            Err(Error::NotFound(format!("Dataset '{}' not found", name)))
        }
    }

    /// Get the currently active dataset
    #[must_use]
    pub fn active_dataset(&self) -> Option<&Arc<ArrowDataset>> {
        self.active_name.as_ref().and_then(|n| self.datasets.get(n))
    }

    /// Get the name of the active dataset
    #[must_use]
    pub fn active_name(&self) -> Option<String> {
        self.active_name.clone()
    }

    /// Get the quality grade for the active dataset
    #[must_use]
    pub fn active_grade(&self) -> Option<LetterGrade> {
        self.quality_cache.as_ref().map(|c| c.score.grade)
    }

    /// Get the row count of the active dataset
    #[must_use]
    pub fn active_row_count(&self) -> Option<usize> {
        self.active_dataset().map(|d| d.len())
    }

    /// List all loaded dataset names
    #[must_use]
    pub fn datasets(&self) -> Vec<String> {
        self.datasets.keys().cloned().collect()
    }

    /// Add a command to history
    pub fn add_history(&mut self, command: &str) {
        self.history.push(command.to_string());
    }

    /// Get command history
    #[must_use]
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Get the quality cache
    #[must_use]
    pub fn quality_cache(&self) -> Option<&QualityCache> {
        self.quality_cache.as_ref()
    }

    /// Export history as a reproducible shell script (ALIM-REPL-006)
    #[must_use]
    pub fn export_history(&self) -> String {
        let mut script = String::new();
        script.push_str("#!/usr/bin/env bash\n");
        script.push_str("# alimentar session export\n");
        let _ = writeln!(script, "# Generated: {}", chrono_now());
        script.push_str("# Reproducible session for Batuta pipeline integration\n\n");

        for cmd in &self.history {
            // Convert REPL commands to batch CLI equivalents
            let batch_cmd = self.repl_to_batch(cmd);
            let _ = writeln!(script, "alimentar {}", batch_cmd);
        }

        script
    }

    /// Convert REPL command to batch CLI equivalent
    fn repl_to_batch(&self, cmd: &str) -> String {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() {
            return String::new();
        }

        match parts[0] {
            "load" => {
                // Keep load command as-is for reproducibility
                cmd.to_string()
            }
            "info" | "schema" | "head" => {
                // These need the active file path
                if let Some(name) = &self.active_name {
                    format!("{} {}", cmd, name)
                } else {
                    cmd.to_string()
                }
            }
            "quality" => {
                if parts.len() > 1 {
                    if let Some(name) = &self.active_name {
                        format!("{} {} {}", parts[0], parts[1], name)
                    } else {
                        cmd.to_string()
                    }
                } else {
                    cmd.to_string()
                }
            }
            _ => cmd.to_string(),
        }
    }

    /// Get schema column names for autocomplete
    #[must_use]
    pub fn column_names(&self) -> Vec<String> {
        self.active_dataset()
            .map(|d| {
                d.schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Execute a REPL command
    ///
    /// # Errors
    ///
    /// Returns an error if the command execution fails.
    pub fn execute(&mut self, cmd: ReplCommand) -> Result<()> {
        match cmd {
            ReplCommand::Load { path } => self.cmd_load(&path),
            ReplCommand::Info => self.cmd_info(),
            ReplCommand::Head { n } => self.cmd_head(n),
            ReplCommand::Schema => self.cmd_schema(),
            ReplCommand::QualityCheck => self.cmd_quality_check(),
            ReplCommand::QualityScore {
                suggest,
                json,
                badge,
            } => self.cmd_quality_score(suggest, json, badge),
            ReplCommand::DriftDetect { reference } => self.cmd_drift_detect(&reference),
            ReplCommand::Convert { format } => self.cmd_convert(&format),
            ReplCommand::Datasets => self.cmd_datasets(),
            ReplCommand::Use { name } => self.use_dataset(&name),
            ReplCommand::History { export } => self.cmd_history(export),
            ReplCommand::Help { topic } => self.cmd_help(topic.as_deref()),
            ReplCommand::Export { what, json } => self.cmd_export(&what, json),
            ReplCommand::Validate { schema } => self.cmd_validate(&schema),
            ReplCommand::Quit => Ok(()), // Handled in main loop
        }
    }

    fn cmd_load(&mut self, path: &str) -> Result<()> {
        let dataset = load_dataset_from_path(path)?;
        let name = Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("data")
            .to_string();

        self.load_dataset(&name, dataset);
        println!(
            "Loaded '{}' ({} rows)",
            name,
            self.active_row_count().unwrap_or(0)
        );
        Ok(())
    }

    fn cmd_info(&self) -> Result<()> {
        let dataset = self.require_active()?;
        println!(
            "Dataset: {}",
            self.active_name.as_deref().unwrap_or("unnamed")
        );
        println!("Rows: {}", dataset.len());
        println!("Columns: {}", dataset.schema().fields().len());

        if let Some(cache) = &self.quality_cache {
            println!("Quality: {} ({:.1})", cache.score.grade, cache.score.score);
        }

        Ok(())
    }

    fn cmd_head(&self, n: usize) -> Result<()> {
        use crate::transform::Transform;

        let dataset = self.require_active()?;
        let mut total_rows = 0;
        let mut output_batches = Vec::new();

        for batch in dataset.iter() {
            if total_rows >= n {
                break;
            }
            let rows_needed = n - total_rows;
            let rows_to_take = rows_needed.min(batch.num_rows());

            if rows_to_take > 0 {
                let take_transform = crate::transform::Take::new(rows_to_take);
                let limited = take_transform.apply(batch)?;
                total_rows += limited.num_rows();
                output_batches.push(limited);
            }
        }

        println!(
            "{}",
            arrow::util::pretty::pretty_format_batches(&output_batches)?
        );
        Ok(())
    }

    fn cmd_schema(&self) -> Result<()> {
        let dataset = self.require_active()?;
        let schema = dataset.schema();

        println!("Schema ({} columns):", schema.fields().len());
        for field in schema.fields() {
            let nullable = if field.is_nullable() {
                "nullable"
            } else {
                "not null"
            };
            println!("  {}: {} ({})", field.name(), field.data_type(), nullable);
        }

        Ok(())
    }

    fn cmd_quality_check(&self) -> Result<()> {
        let dataset = self.require_active()?;
        let checker = QualityChecker::new();
        let report = checker.check(dataset)?;

        println!("Quality Check Results:");
        println!("  Total rows: {}", report.row_count);
        println!("  Total columns: {}", report.column_count);

        for (name, col) in &report.columns {
            println!("\n  Column: {}", name);
            println!("    Null ratio: {:.2}%", col.null_ratio * 100.0);
            println!("    Duplicate ratio: {:.2}%", col.duplicate_ratio * 100.0);
        }

        Ok(())
    }

    fn cmd_quality_score(&self, suggest: bool, json: bool, badge: bool) -> Result<()> {
        if let Some(cache) = &self.quality_cache {
            if json {
                println!("{}", cache.score.to_json());
            } else if badge {
                println!("{}", cache.score.badge_url());
            } else {
                println!(
                    "Quality Score: {} ({:.1}/100)",
                    cache.score.grade, cache.score.score
                );
                println!("Decision: {}", cache.score.grade.publication_decision());

                if suggest {
                    let failed = cache.score.failed_items();
                    if !failed.is_empty() {
                        println!("\nSuggestions:");
                        for item in failed {
                            println!("  [{:?}] {}", item.severity, item.description);
                            if let Some(s) = &item.suggestion {
                                println!("    → {}", s);
                            }
                        }
                    }
                }
            }
            Ok(())
        } else {
            Err(Error::NotFound("No quality data available".to_string()))
        }
    }

    fn cmd_drift_detect(&self, reference: &str) -> Result<()> {
        let dataset = self.require_active()?;
        let ref_dataset = load_dataset_from_path(reference)?;

        let detector = crate::DriftDetector::new(ref_dataset);
        let report = detector.detect(dataset)?;

        println!("Drift Detection Report:");
        println!("  Columns analyzed: {}", report.column_scores.len());
        println!(
            "  Drifted columns: {}",
            report
                .column_scores
                .values()
                .filter(|d| d.drift_detected)
                .count()
        );

        for (name, drift) in &report.column_scores {
            if drift.drift_detected {
                println!("  {} [{:?}]: {:.2?}", name, drift.severity, drift.p_value);
            }
        }

        Ok(())
    }

    fn cmd_convert(&self, format: &str) -> Result<()> {
        let dataset = self.require_active()?;
        let name = self.active_name.as_deref().unwrap_or("data");
        let output = format!("{}.{}", name, format);

        match format {
            "csv" => dataset.to_csv(&output)?,
            "parquet" => dataset.to_parquet(&output)?,
            "json" => dataset.to_json(&output)?,
            _ => return Err(Error::InvalidFormat(format!("Unknown format: {}", format))),
        }

        println!("Converted to {}", output);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn cmd_datasets(&self) -> Result<()> {
        if self.datasets.is_empty() {
            println!("No datasets loaded. Use 'load <file>' to load one.");
        } else {
            println!("Loaded datasets:");
            for name in self.datasets.keys() {
                let marker = if Some(name) == self.active_name.as_ref() {
                    "* "
                } else {
                    "  "
                };
                if let Some(ds) = self.datasets.get(name) {
                    println!("{}{} ({} rows)", marker, name, ds.len());
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn cmd_history(&self, export: bool) -> Result<()> {
        if export {
            print!("{}", self.export_history());
        } else {
            for (i, cmd) in self.history.iter().enumerate() {
                println!("{:4}  {}", i + 1, cmd);
            }
        }
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn cmd_help(&self, topic: Option<&str>) -> Result<()> {
        match topic {
            None => {
                println!("alimentar REPL Commands:");
                println!();
                println!("Data Loading:");
                println!("  load <file>          Load a dataset (parquet, csv, json)");
                println!("  datasets             List loaded datasets");
                println!("  use <name>           Switch active dataset");
                println!();
                println!("Data Inspection:");
                println!("  info                 Show dataset metadata");
                println!("  head [n]             Show first n rows (default: 10)");
                println!("  schema               Display column schema");
                println!();
                println!("Quality (Andon):");
                println!("  quality check        Run quality checks");
                println!("  quality score        100-point quality score");
                println!("    --suggest          Show improvement suggestions");
                println!("    --json             Output as JSON");
                println!("    --badge            Output shields.io badge URL");
                println!();
                println!("Analysis:");
                println!("  drift detect <ref>   Compare with reference dataset");
                println!("  convert <format>     Export to format (csv, parquet, json)");
                println!();
                println!("Pipeline (Batuta):");
                println!("  export quality --json  Export quality for PMAT");
                println!("  validate --schema <f>  Validate against schema spec");
                println!();
                println!("Session:");
                println!("  history              Show command history");
                println!("  history --export     Export as reproducible script");
                println!("  help [topic]         Show help (topics: quality, drift, export)");
                println!("  quit, exit           Exit REPL");
            }
            Some("quality") => {
                println!("Quality Commands (Jidoka - Built-in Quality):");
                println!();
                println!("The quality system provides a 100-point scoring based on:");
                println!("  - Critical (2x weight): Blocks publication");
                println!("  - High (1.5x weight): Needs immediate attention");
                println!("  - Medium (1x weight): Should fix before publish");
                println!("  - Low (0.5x weight): Minor/informational");
                println!();
                println!("Letter Grades:");
                println!("  A (95+): Publish immediately");
                println!("  B (85-94): Publish with caveats");
                println!("  C (70-84): Remediation required");
                println!("  D (50-69): Major rework needed");
                println!("  F (<50): Do not publish");
            }
            Some("drift") => {
                println!("Drift Detection:");
                println!();
                println!("Compare your active dataset against a reference to detect:");
                println!("  - Statistical distribution changes");
                println!("  - Schema differences");
                println!("  - Value range shifts");
                println!();
                println!("Usage: drift detect <reference.parquet>");
            }
            Some("export") => {
                println!("Export Commands (Batuta Integration):");
                println!();
                println!("  export quality --json   Quality metrics for PMAT");
                println!("  validate --schema <f>   Pre-transpilation validation");
                println!("  history --export        Reproducible session script");
            }
            Some(t) => {
                println!("Unknown help topic: '{}'. Try: quality, drift, export", t);
            }
        }
        Ok(())
    }

    fn cmd_export(&self, what: &str, json: bool) -> Result<()> {
        match what {
            "quality" => {
                if let Some(cache) = &self.quality_cache {
                    if json {
                        println!("{}", cache.score.to_json());
                    } else {
                        println!("Quality: {} ({:.1})", cache.score.grade, cache.score.score);
                    }
                    Ok(())
                } else {
                    Err(Error::NotFound("No quality data available".to_string()))
                }
            }
            _ => Err(Error::InvalidFormat(format!(
                "Unknown export type: {}",
                what
            ))),
        }
    }

    fn cmd_validate(&self, schema_path: &str) -> Result<()> {
        let _ = self.require_active()?;
        // Basic validation - check if schema file exists
        if std::path::Path::new(schema_path).exists() {
            println!(
                "Validation against {} - Feature in development",
                schema_path
            );
            Ok(())
        } else {
            Err(Error::NotFound(format!(
                "Schema file not found: {}",
                schema_path
            )))
        }
    }

    fn require_active(&self) -> Result<&Arc<ArrowDataset>> {
        self.active_dataset().ok_or_else(|| {
            Error::NotFound("No active dataset. Use 'load <file>' first.".to_string())
        })
    }
}

/// Get current time as string (avoids chrono dependency in core)
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    format!("{}s since epoch", duration.as_secs())
}

/// Load dataset from path, auto-detecting format
fn load_dataset_from_path(path: &str) -> Result<ArrowDataset> {
    let path = Path::new(path);
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "parquet" | "pq" => ArrowDataset::from_parquet(path),
        "csv" => ArrowDataset::from_csv(path),
        "json" | "jsonl" => ArrowDataset::from_json(path),
        _ => Err(Error::unsupported_format(format!(
            "Unknown file extension: .{}. Supported: parquet, csv, json",
            extension
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Float64Array, Int32Array, StringArray},
        datatypes::{DataType, Field, Schema as ArrowSchema},
        record_batch::RecordBatch,
    };

    use super::*;

    fn create_test_dataset() -> ArrowDataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("value", DataType::Float64, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    None,
                    Some("d"),
                    Some("e"),
                ])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
            ],
        )
        .unwrap();

        ArrowDataset::new(vec![batch]).unwrap()
    }

    // DisplayConfig tests
    #[test]
    fn test_display_config_default() {
        let config = DisplayConfig::default();
        assert_eq!(config.max_rows, 10);
        assert_eq!(config.max_column_width, 50);
        assert!(config.color_output);
    }

    #[test]
    fn test_display_config_with_max_rows() {
        let config = DisplayConfig::default().with_max_rows(20);
        assert_eq!(config.max_rows, 20);
    }

    #[test]
    fn test_display_config_with_max_column_width() {
        let config = DisplayConfig::default().with_max_column_width(100);
        assert_eq!(config.max_column_width, 100);
    }

    #[test]
    fn test_display_config_with_color() {
        let config = DisplayConfig::default().with_color(false);
        assert!(!config.color_output);
    }

    #[test]
    fn test_display_config_chained() {
        let config = DisplayConfig::default()
            .with_max_rows(25)
            .with_max_column_width(75)
            .with_color(false);
        assert_eq!(config.max_rows, 25);
        assert_eq!(config.max_column_width, 75);
        assert!(!config.color_output);
    }

    // ReplSession tests
    #[test]
    fn test_session_new() {
        let session = ReplSession::new();
        assert!(session.datasets().is_empty());
        assert!(session.active_name().is_none());
        assert!(session.active_dataset().is_none());
        assert!(session.history().is_empty());
    }

    #[test]
    fn test_session_default() {
        let session = ReplSession::default();
        assert!(session.datasets().is_empty());
    }

    #[test]
    fn test_session_load_dataset() {
        let mut session = ReplSession::new();
        let dataset = create_test_dataset();

        session.load_dataset("test", dataset);

        assert_eq!(session.datasets().len(), 1);
        assert!(session.datasets().contains(&"test".to_string()));
        assert_eq!(session.active_name(), Some("test".to_string()));
        assert!(session.active_dataset().is_some());
    }

    #[test]
    fn test_session_use_dataset_success() {
        let mut session = ReplSession::new();
        session.load_dataset("ds1", create_test_dataset());
        session.load_dataset("ds2", create_test_dataset());

        assert_eq!(session.active_name(), Some("ds2".to_string()));

        session.use_dataset("ds1").unwrap();
        assert_eq!(session.active_name(), Some("ds1".to_string()));
    }

    #[test]
    fn test_session_use_dataset_not_found() {
        let mut session = ReplSession::new();
        let result = session.use_dataset("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_session_active_row_count() {
        let mut session = ReplSession::new();
        assert!(session.active_row_count().is_none());

        session.load_dataset("test", create_test_dataset());
        assert_eq!(session.active_row_count(), Some(5));
    }

    #[test]
    fn test_session_active_grade() {
        let mut session = ReplSession::new();
        assert!(session.active_grade().is_none());

        session.load_dataset("test", create_test_dataset());
        // After loading, quality cache should be populated
        assert!(session.active_grade().is_some());
    }

    #[test]
    fn test_session_column_names() {
        let mut session = ReplSession::new();
        assert!(session.column_names().is_empty());

        session.load_dataset("test", create_test_dataset());
        let cols = session.column_names();
        assert_eq!(cols.len(), 3);
        assert!(cols.contains(&"id".to_string()));
        assert!(cols.contains(&"name".to_string()));
        assert!(cols.contains(&"value".to_string()));
    }

    #[test]
    fn test_session_history() {
        let mut session = ReplSession::new();

        session.add_history("load test.parquet");
        session.add_history("head 5");
        session.add_history("schema");

        assert_eq!(session.history().len(), 3);
        assert_eq!(session.history()[0], "load test.parquet");
        assert_eq!(session.history()[1], "head 5");
        assert_eq!(session.history()[2], "schema");
    }

    #[test]
    fn test_session_quality_cache() {
        let mut session = ReplSession::new();
        assert!(session.quality_cache().is_none());

        session.load_dataset("test", create_test_dataset());
        assert!(session.quality_cache().is_some());
    }

    #[test]
    fn test_session_export_history() {
        let mut session = ReplSession::new();
        session.add_history("load test.parquet");
        session.add_history("head 5");

        let script = session.export_history();
        assert!(script.starts_with("#!/usr/bin/env bash"));
        assert!(script.contains("alimentar load test.parquet"));
        assert!(script.contains("alimentar head 5"));
    }

    #[test]
    fn test_session_export_history_with_active_dataset() {
        let mut session = ReplSession::new();
        session.load_dataset("test.parquet", create_test_dataset());
        session.add_history("info");
        session.add_history("schema");
        session.add_history("head");

        let script = session.export_history();
        assert!(script.contains("alimentar info test.parquet"));
        assert!(script.contains("alimentar schema test.parquet"));
        assert!(script.contains("alimentar head test.parquet"));
    }

    #[test]
    fn test_session_repl_to_batch_empty() {
        let session = ReplSession::new();
        assert_eq!(session.repl_to_batch(""), "");
    }

    #[test]
    fn test_session_repl_to_batch_load() {
        let session = ReplSession::new();
        assert_eq!(
            session.repl_to_batch("load test.parquet"),
            "load test.parquet"
        );
    }

    #[test]
    fn test_session_repl_to_batch_quality_with_args() {
        let mut session = ReplSession::new();
        session.load_dataset("data.parquet", create_test_dataset());

        assert_eq!(
            session.repl_to_batch("quality check"),
            "quality check data.parquet"
        );
    }

    #[test]
    fn test_session_repl_to_batch_quality_no_active() {
        let session = ReplSession::new();
        assert_eq!(session.repl_to_batch("quality check"), "quality check");
    }

    #[test]
    fn test_session_repl_to_batch_unknown_command() {
        let session = ReplSession::new();
        assert_eq!(session.repl_to_batch("custom cmd"), "custom cmd");
    }

    // chrono_now test
    #[test]
    fn test_chrono_now() {
        let now = chrono_now();
        assert!(now.contains("since epoch"));
        // Should contain a number
        assert!(now.chars().any(|c| c.is_ascii_digit()));
    }

    // load_dataset_from_path tests
    #[test]
    fn test_load_dataset_unknown_extension() {
        let result = load_dataset_from_path("test.xyz");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown file extension"));
    }

    #[test]
    fn test_load_dataset_no_extension() {
        let result = load_dataset_from_path("test");
        assert!(result.is_err());
    }

    // Execute command tests with actual dataset
    #[test]
    fn test_execute_info() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::Info);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_info_no_dataset() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Info);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_schema() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::Schema);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_head() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::Head { n: 3 });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_head_no_dataset() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Head { n: 3 });
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_quality_check() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::QualityCheck);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_quality_score() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::QualityScore {
            suggest: false,
            json: false,
            badge: false,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_quality_score_suggest() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::QualityScore {
            suggest: true,
            json: false,
            badge: false,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_quality_score_json() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::QualityScore {
            suggest: false,
            json: true,
            badge: false,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_quality_score_badge() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::QualityScore {
            suggest: false,
            json: false,
            badge: true,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_quality_score_no_cache() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::QualityScore {
            suggest: false,
            json: false,
            badge: false,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_datasets_empty() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Datasets);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_datasets_with_data() {
        let mut session = ReplSession::new();
        session.load_dataset("ds1", create_test_dataset());
        session.load_dataset("ds2", create_test_dataset());

        let result = session.execute(ReplCommand::Datasets);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_history() {
        let mut session = ReplSession::new();
        session.add_history("test cmd");

        let result = session.execute(ReplCommand::History { export: false });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_history_export() {
        let mut session = ReplSession::new();
        session.add_history("test cmd");

        let result = session.execute(ReplCommand::History { export: true });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_help_none() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Help { topic: None });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_help_quality() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Help {
            topic: Some("quality".to_string()),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_help_drift() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Help {
            topic: Some("drift".to_string()),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_help_export() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Help {
            topic: Some("export".to_string()),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_help_unknown() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Help {
            topic: Some("unknown".to_string()),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_export_quality() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::Export {
            what: "quality".to_string(),
            json: false,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_export_quality_json() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::Export {
            what: "quality".to_string(),
            json: true,
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_export_quality_no_cache() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Export {
            what: "quality".to_string(),
            json: false,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_export_unknown() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::Export {
            what: "unknown".to_string(),
            json: false,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_validate_file_not_found() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::Validate {
            schema: "nonexistent.json".to_string(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_convert_invalid_format() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::Convert {
            format: "invalid".to_string(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_quit() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Quit);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_use() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.execute(ReplCommand::Use {
            name: "test".to_string(),
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_use_not_found() {
        let mut session = ReplSession::new();
        let result = session.execute(ReplCommand::Use {
            name: "nonexistent".to_string(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_require_active_none() {
        let session = ReplSession::new();
        let result = session.require_active();
        assert!(result.is_err());
    }

    #[test]
    fn test_require_active_some() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let result = session.require_active();
        assert!(result.is_ok());
    }

    // Multiple datasets test
    #[test]
    fn test_multiple_datasets() {
        let mut session = ReplSession::new();
        session.load_dataset("first", create_test_dataset());
        session.load_dataset("second", create_test_dataset());
        session.load_dataset("third", create_test_dataset());

        assert_eq!(session.datasets().len(), 3);
        assert_eq!(session.active_name(), Some("third".to_string()));

        session.use_dataset("first").unwrap();
        assert_eq!(session.active_name(), Some("first".to_string()));
    }

    // Quality checklist tests
    #[test]
    fn test_build_basic_checklist() {
        let session = ReplSession::new();
        let dataset = create_test_dataset();
        let checker = QualityChecker::new();
        let report = checker.check(&dataset).unwrap();

        let checklist = session.build_basic_checklist(&report);
        assert_eq!(checklist.len(), 5);
    }
}
