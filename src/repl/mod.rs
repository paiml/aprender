//! Interactive REPL for alimentar (ALIM-SPEC-006)
//!
//! Implements the Interactive Andon concept from the CLI & REPL Quality
//! Specification.
//!
//! # Design Principles (Toyota Way)
//!
//! - **Genchi Genbutsu** (Go and See): Interactive data inspection without
//!   compilation
//! - **Jikotei Kanketsu** (Self-completion): Complete data quality tasks in one
//!   environment
//! - **Poka-Yoke** (Mistake Proofing): Schema-aware autocomplete prevents
//!   invalid input
//! - **Andon** (Visual Control): Color-coded prompts show dataset health status
//!
//! # Requirements Implemented
//!
//! - ALIM-REPL-001: Stateful session with dataset caching
//! - ALIM-REPL-002: <100ms response time for metadata queries
//! - ALIM-REPL-003: Schema-aware autocomplete with reedline
//! - ALIM-REPL-004: Contextual help system
//! - ALIM-REPL-005: Batuta pipeline integration hooks
//! - ALIM-REPL-006: Reproducible session export
//! - ALIM-REPL-007: Progressive disclosure commands
//!
//! # References
//! - [5] Nielsen (1993). Usability Engineering - 100ms response threshold
//! - [16] Perez & Granger (2007). IPython - reproducible sessions

mod commands;
mod completer;
mod prompt;
mod session;

use std::io::IsTerminal;

pub use commands::{CommandParser, ReplCommand};
pub use completer::SchemaAwareCompleter;
pub use prompt::{AndonPrompt, HealthStatus};
#[cfg(feature = "repl")]
use reedline::{Reedline, Signal};
pub use session::{DisplayConfig, ReplSession};

use crate::Result;

/// Run the interactive REPL
///
/// # Errors
///
/// Returns an error if REPL initialization fails
#[cfg(feature = "repl")]
pub fn run() -> Result<()> {
    // Check if stdin is a terminal - use simple mode for piped input (testing)
    if std::io::stdin().is_terminal() {
        run_interactive()
    } else {
        run_non_interactive()
    }
}

/// Run REPL in interactive mode with reedline (full features)
#[cfg(feature = "repl")]
fn run_interactive() -> Result<()> {
    let mut session = ReplSession::new();
    let mut line_editor = create_editor(&session)?;
    let prompt = AndonPrompt::new();

    println!(
        "alimentar {} - Interactive Data Explorer",
        env!("CARGO_PKG_VERSION")
    );
    println!("Type 'help' for commands, 'quit' to exit\n");

    loop {
        let sig = line_editor.read_line(&prompt);
        match sig {
            Ok(Signal::Success(line)) => {
                if dispatch_line(&line, &mut session, &line_editor) {
                    break;
                }
            }
            Ok(Signal::CtrlC) => {
                println!("^C");
            }
            Ok(Signal::CtrlD) => {
                println!("\nGoodbye!");
                break;
            }
            Err(e) => {
                eprintln!("Input error: {e}");
                break;
            }
        }
    }

    Ok(())
}

/// Parse and execute a single REPL line. Returns true if the REPL should exit.
#[cfg(feature = "repl")]
fn dispatch_line(line: &str, session: &mut ReplSession, editor: &Reedline) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    session.add_history(trimmed);

    match CommandParser::parse(trimmed) {
        Ok(cmd) => {
            if matches!(cmd, ReplCommand::Quit) {
                println!("Goodbye!");
                return true;
            }
            if let Err(e) = session.execute(cmd) {
                eprintln!("Error: {e}");
            }
        }
        Err(e) => eprintln!("Parse error: {e}"),
    }

    update_completer(editor, session);
    false
}

/// Run REPL in non-interactive mode (for testing and piped input)
#[cfg(feature = "repl")]
#[allow(clippy::unnecessary_wraps)] // Consistent API with run_interactive()
fn run_non_interactive() -> Result<()> {
    use std::io::BufRead;

    let mut session = ReplSession::new();

    println!(
        "alimentar {} - Interactive Data Explorer",
        env!("CARGO_PKG_VERSION")
    );
    println!("Type 'help' for commands, 'quit' to exit\n");

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else {
            println!("Goodbye!");
            break;
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        session.add_history(trimmed);

        match CommandParser::parse(trimmed) {
            Ok(cmd) => {
                if matches!(cmd, ReplCommand::Quit) {
                    println!("Goodbye!");
                    break;
                }
                if let Err(e) = session.execute(cmd) {
                    eprintln!("Error: {e}");
                }
            }
            Err(e) => eprintln!("Parse error: {e}"),
        }
    }

    // Handle EOF gracefully
    if std::io::stdin().lock().lines().next().is_none() {
        println!("Goodbye!");
    }

    Ok(())
}

#[cfg(feature = "repl")]
fn create_editor(session: &ReplSession) -> Result<Reedline> {
    use reedline::{FileBackedHistory, Reedline};

    let history_path = dirs_home().join(".alimentar_history");
    let history = FileBackedHistory::with_file(1000, history_path)
        .map_err(|e| crate::Error::io_no_path(std::io::Error::other(e.to_string())))?;

    let completer = Box::new(SchemaAwareCompleter::new(session));

    let editor = Reedline::create()
        .with_history(Box::new(history))
        .with_completer(completer);

    Ok(editor)
}

#[cfg(feature = "repl")]
fn update_completer(editor: &Reedline, session: &ReplSession) {
    let completer = Box::new(SchemaAwareCompleter::new(session));
    // Note: reedline doesn't support runtime completer updates easily
    // This is a placeholder for future enhancement
    let _ = (editor, completer);
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS - EXTREME TDD: Tests written first per specification
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArrowDataset;

    // ─────────────────────────────────────────────────────────────────────────────
    // ALIM-REPL-001: Stateful Session Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_session_creation() {
        let session = ReplSession::new();
        assert!(session.active_dataset().is_none());
        assert!(session.datasets().is_empty());
        assert!(session.history().is_empty());
    }

    #[test]
    fn test_session_load_dataset() {
        let mut session = ReplSession::new();

        // Create a test dataset
        let dataset = create_test_dataset();
        session.load_dataset("test", dataset);

        assert!(session.datasets().contains(&"test".to_string()));
        assert!(session.active_dataset().is_some());
    }

    #[test]
    fn test_session_switch_dataset() {
        let mut session = ReplSession::new();

        // Load two datasets
        session.load_dataset("data1", create_test_dataset());
        session.load_dataset("data2", create_test_dataset());

        assert_eq!(session.active_name(), Some("data2".to_string()));

        session.use_dataset("data1").unwrap();
        assert_eq!(session.active_name(), Some("data1".to_string()));
    }

    #[test]
    fn test_session_history_tracking() {
        let mut session = ReplSession::new();

        session.add_history("load data.parquet");
        session.add_history("info");
        session.add_history("head 10");

        assert_eq!(session.history().len(), 3);
        assert_eq!(session.history()[0], "load data.parquet");
    }

    #[test]
    fn test_session_history_export() {
        let mut session = ReplSession::new();

        session.add_history("load data.parquet");
        session.add_history("quality check");
        session.add_history("convert csv");

        let script = session.export_history();

        assert!(script.contains("# alimentar session export"));
        assert!(script.contains("alimentar"));
        assert!(script.contains("load data.parquet"));
    }

    #[test]
    fn test_session_quality_cache() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        // Quality cache should be populated on load
        assert!(session.quality_cache().is_some());
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // ALIM-REPL-003: Command Parser Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_load_command() {
        let cmd = CommandParser::parse("load data.parquet").unwrap();
        assert!(matches!(cmd, ReplCommand::Load { path } if path == "data.parquet"));
    }

    #[test]
    fn test_parse_info_command() {
        let cmd = CommandParser::parse("info").unwrap();
        assert!(matches!(cmd, ReplCommand::Info));
    }

    #[test]
    fn test_parse_head_command_default() {
        let cmd = CommandParser::parse("head").unwrap();
        assert!(matches!(cmd, ReplCommand::Head { n: 10 }));
    }

    #[test]
    fn test_parse_head_command_with_count() {
        let cmd = CommandParser::parse("head 25").unwrap();
        assert!(matches!(cmd, ReplCommand::Head { n: 25 }));
    }

    #[test]
    fn test_parse_schema_command() {
        let cmd = CommandParser::parse("schema").unwrap();
        assert!(matches!(cmd, ReplCommand::Schema));
    }

    #[test]
    fn test_parse_quality_check_command() {
        let cmd = CommandParser::parse("quality check").unwrap();
        assert!(matches!(cmd, ReplCommand::QualityCheck));
    }

    #[test]
    fn test_parse_quality_score_command() {
        let cmd = CommandParser::parse("quality score").unwrap();
        assert!(matches!(
            cmd,
            ReplCommand::QualityScore {
                suggest: false,
                json: false,
                badge: false
            }
        ));
    }

    #[test]
    fn test_parse_quality_score_with_flags() {
        let cmd = CommandParser::parse("quality score --suggest --json").unwrap();
        assert!(matches!(
            cmd,
            ReplCommand::QualityScore {
                suggest: true,
                json: true,
                badge: false
            }
        ));
    }

    #[test]
    fn test_parse_drift_detect_command() {
        let cmd = CommandParser::parse("drift detect baseline.parquet").unwrap();
        assert!(
            matches!(cmd, ReplCommand::DriftDetect { reference } if reference == "baseline.parquet")
        );
    }

    #[test]
    fn test_parse_convert_command() {
        let cmd = CommandParser::parse("convert csv").unwrap();
        assert!(matches!(cmd, ReplCommand::Convert { format } if format == "csv"));
    }

    #[test]
    fn test_parse_datasets_command() {
        let cmd = CommandParser::parse("datasets").unwrap();
        assert!(matches!(cmd, ReplCommand::Datasets));
    }

    #[test]
    fn test_parse_use_command() {
        let cmd = CommandParser::parse("use my_data").unwrap();
        assert!(matches!(cmd, ReplCommand::Use { name } if name == "my_data"));
    }

    #[test]
    fn test_parse_history_command() {
        let cmd = CommandParser::parse("history").unwrap();
        assert!(matches!(cmd, ReplCommand::History { export: false }));
    }

    #[test]
    fn test_parse_history_export_command() {
        let cmd = CommandParser::parse("history --export").unwrap();
        assert!(matches!(cmd, ReplCommand::History { export: true }));
    }

    #[test]
    fn test_parse_help_command() {
        let cmd = CommandParser::parse("help").unwrap();
        assert!(matches!(cmd, ReplCommand::Help { topic: None }));
    }

    #[test]
    fn test_parse_help_with_topic() {
        let cmd = CommandParser::parse("help quality").unwrap();
        assert!(matches!(cmd, ReplCommand::Help { topic: Some(t) } if t == "quality"));
    }

    #[test]
    fn test_parse_question_mark_help() {
        let cmd = CommandParser::parse("?").unwrap();
        assert!(matches!(cmd, ReplCommand::Help { topic: None }));
    }

    #[test]
    fn test_parse_quit_command() {
        let cmd = CommandParser::parse("quit").unwrap();
        assert!(matches!(cmd, ReplCommand::Quit));
    }

    #[test]
    fn test_parse_exit_command() {
        let cmd = CommandParser::parse("exit").unwrap();
        assert!(matches!(cmd, ReplCommand::Quit));
    }

    #[test]
    fn test_parse_invalid_command() {
        let result = CommandParser::parse("invalid_command_xyz");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_command() {
        let result = CommandParser::parse("");
        assert!(result.is_err());
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // ALIM-REPL-004: Schema-Aware Completer Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_completer_command_suggestions() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("lo");
        assert!(suggestions.iter().any(|s| s == "load"));
    }

    #[test]
    fn test_completer_subcommand_suggestions() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("quality ");
        assert!(suggestions.iter().any(|s| s == "check"));
        assert!(suggestions.iter().any(|s| s == "score"));
    }

    #[test]
    fn test_completer_column_suggestions() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let completer = SchemaAwareCompleter::new(&session);

        // Column suggestions for commands that accept columns
        let suggestions = completer.complete("select ");
        // Should suggest column names from schema
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_completer_dataset_suggestions() {
        let mut session = ReplSession::new();
        session.load_dataset("train", create_test_dataset());
        session.load_dataset("test", create_test_dataset());

        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("use ");
        assert!(suggestions.iter().any(|s| s == "train"));
        assert!(suggestions.iter().any(|s| s == "test"));
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // ALIM-REPL-005: Andon Prompt Tests (Visual Control)
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_prompt_no_dataset() {
        let session = ReplSession::new();
        let prompt_str = AndonPrompt::render(&session);

        assert!(prompt_str.contains("alimentar"));
        assert!(!prompt_str.contains('[')); // No dataset indicator
    }

    #[test]
    fn test_prompt_healthy_dataset() {
        let mut session = ReplSession::new();
        session.load_dataset("data", create_test_dataset());
        // Assume quality score A

        let prompt_str = AndonPrompt::render(&session);

        assert!(prompt_str.contains("data"));
        assert!(prompt_str.contains('A') || prompt_str.contains("rows"));
    }

    #[test]
    fn test_health_status_from_grade() {
        assert_eq!(HealthStatus::from_grade('A'), HealthStatus::Healthy);
        assert_eq!(HealthStatus::from_grade('B'), HealthStatus::Healthy);
        assert_eq!(HealthStatus::from_grade('C'), HealthStatus::Warning);
        assert_eq!(HealthStatus::from_grade('D'), HealthStatus::Warning);
        assert_eq!(HealthStatus::from_grade('F'), HealthStatus::Critical);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // ALIM-REPL-006: Export Command Tests (Batuta Integration)
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_export_quality_json() {
        let cmd = CommandParser::parse("export quality --json").unwrap();
        assert!(matches!(cmd, ReplCommand::Export { what, json: true } if what == "quality"));
    }

    #[test]
    fn test_parse_validate_schema() {
        let cmd = CommandParser::parse("validate --schema spec.yaml").unwrap();
        assert!(matches!(cmd, ReplCommand::Validate { schema } if schema == "spec.yaml"));
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // DisplayConfig Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_display_config_defaults() {
        let config = DisplayConfig::default();

        assert_eq!(config.max_rows, 10);
        assert_eq!(config.max_column_width, 50);
        assert!(config.color_output);
    }

    #[test]
    fn test_display_config_builder() {
        let config = DisplayConfig::default()
            .with_max_rows(20)
            .with_max_column_width(100)
            .with_color(false);

        assert_eq!(config.max_rows, 20);
        assert_eq!(config.max_column_width, 100);
        assert!(!config.color_output);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Helper Functions for Tests
    // ─────────────────────────────────────────────────────────────────────────────

    fn create_test_dataset() -> ArrowDataset {
        use std::sync::Arc;

        use arrow::{
            array::{Int32Array, StringArray},
            datatypes::{DataType, Field, Schema},
            record_batch::RecordBatch,
        };

        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Int32, true),
        ]);

        let id_array = Int32Array::from(vec![1, 2, 3]);
        let name_array = StringArray::from(vec![Some("Alice"), Some("Bob"), Some("Charlie")]);
        let score_array = Int32Array::from(vec![Some(85), Some(92), Some(78)]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(id_array),
                Arc::new(name_array),
                Arc::new(score_array),
            ],
        )
        .unwrap();

        ArrowDataset::from_batch(batch).unwrap()
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // COVERAGE BOOST: HealthStatus Tests (prompt.rs)
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_health_status_indicator_all_variants() {
        assert_eq!(HealthStatus::Healthy.indicator(), "");
        assert_eq!(HealthStatus::Warning.indicator(), "!");
        assert_eq!(HealthStatus::Critical.indicator(), "!!");
        assert_eq!(HealthStatus::None.indicator(), "");
    }

    #[test]
    fn test_health_status_from_grade_unknown() {
        assert_eq!(HealthStatus::from_grade('X'), HealthStatus::None);
        assert_eq!(HealthStatus::from_grade('Z'), HealthStatus::None);
        assert_eq!(HealthStatus::from_grade(' '), HealthStatus::None);
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_health_status_color_all_variants() {
        use nu_ansi_term::Color;

        assert_eq!(HealthStatus::Healthy.color(), Color::Green);
        assert_eq!(HealthStatus::Warning.color(), Color::Yellow);
        assert_eq!(HealthStatus::Critical.color(), Color::Red);
        assert_eq!(HealthStatus::None.color(), Color::Default);
    }

    #[test]
    fn test_andon_prompt_new_and_default() {
        let prompt1 = AndonPrompt::new();
        let prompt2 = AndonPrompt::default();
        // Both should create valid prompts
        let session = ReplSession::new();
        let render1 = AndonPrompt::render(&session);
        let _ = (prompt1, prompt2); // Use them
        assert!(render1.contains("alimentar"));
    }

    #[test]
    fn test_andon_prompt_render_with_grade_indicator() {
        let mut session = ReplSession::new();
        session.load_dataset("test_data", create_test_dataset());

        let prompt_str = AndonPrompt::render(&session);
        // Should contain dataset name and row info
        assert!(prompt_str.contains("test_data"));
        assert!(prompt_str.contains("rows"));
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_render_colored() {
        let mut session = ReplSession::new();
        session.load_dataset("colored_test", create_test_dataset());

        let prompt_str = AndonPrompt::render_colored(&session);
        // Should contain ANSI escape codes and data
        assert!(prompt_str.contains("alimentar"));
        assert!(prompt_str.contains("colored_test"));
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_render_colored_no_dataset() {
        let session = ReplSession::new();
        let prompt_str = AndonPrompt::render_colored(&session);
        assert!(prompt_str.contains("alimentar"));
        // Prompt may contain ANSI escape codes including '[' for colors
        // Just verify it renders without error
    }

    #[cfg(feature = "repl")]
    #[test]
    fn test_andon_prompt_trait_methods() {
        use reedline::Prompt;

        let prompt = AndonPrompt::new();

        assert_eq!(prompt.render_prompt_left().as_ref(), "alimentar > ");
        assert_eq!(prompt.render_prompt_right().as_ref(), "");
        assert_eq!(prompt.render_prompt_multiline_indicator().as_ref(), "... ");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // COVERAGE BOOST: Session Tests (session.rs)
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_session_use_dataset_not_found() {
        let mut session = ReplSession::new();
        session.load_dataset("data1", create_test_dataset());

        let result = session.use_dataset("nonexistent");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_session_active_row_count() {
        let mut session = ReplSession::new();
        assert!(session.active_row_count().is_none());

        session.load_dataset("test", create_test_dataset());
        assert_eq!(session.active_row_count(), Some(3));
    }

    #[test]
    fn test_session_active_grade() {
        let mut session = ReplSession::new();
        assert!(session.active_grade().is_none());

        session.load_dataset("test", create_test_dataset());
        assert!(session.active_grade().is_some());
    }

    #[test]
    fn test_session_column_names() {
        let mut session = ReplSession::new();
        assert!(session.column_names().is_empty());

        session.load_dataset("test", create_test_dataset());
        let columns = session.column_names();
        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"name".to_string()));
        assert!(columns.contains(&"score".to_string()));
    }

    #[test]
    fn test_session_export_history_with_active_dataset() {
        let mut session = ReplSession::new();
        session.load_dataset("data.parquet", create_test_dataset());

        session.add_history("info");
        session.add_history("head 5");
        session.add_history("quality check");

        let script = session.export_history();
        assert!(script.contains("#!/usr/bin/env bash"));
        assert!(script.contains("alimentar session export"));
        // Commands should include file path
        assert!(script.contains("info"));
    }

    #[test]
    fn test_session_config_access() {
        let session = ReplSession::new();
        assert_eq!(session.config.max_rows, 10);
        assert_eq!(session.config.max_column_width, 50);
        assert!(session.config.color_output);
    }

    #[test]
    fn test_session_default_implementation() {
        let session1 = ReplSession::new();
        let session2 = ReplSession::default();

        assert_eq!(session1.datasets().len(), session2.datasets().len());
        assert_eq!(session1.history().len(), session2.history().len());
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // COVERAGE BOOST: Completer Tests (completer.rs)
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_completer_empty_input() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("");
        // Should return all commands
        assert!(suggestions.len() > 5);
        assert!(suggestions.contains(&"load".to_string()));
        assert!(suggestions.contains(&"info".to_string()));
        assert!(suggestions.contains(&"quit".to_string()));
    }

    #[test]
    fn test_completer_convert_suggestions() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("convert ");
        assert!(suggestions.contains(&"csv".to_string()));
        assert!(suggestions.contains(&"parquet".to_string()));
        assert!(suggestions.contains(&"json".to_string()));
    }

    #[test]
    fn test_completer_convert_partial() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("convert p");
        assert!(suggestions.contains(&"parquet".to_string()));
        assert!(!suggestions.contains(&"csv".to_string()));
    }

    #[test]
    fn test_completer_help_suggestions() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("help ");
        assert!(suggestions.contains(&"quality".to_string()));
        assert!(suggestions.contains(&"drift".to_string()));
        assert!(suggestions.contains(&"export".to_string()));
    }

    #[test]
    fn test_completer_help_partial() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("help q");
        assert!(suggestions.contains(&"quality".to_string()));
        assert!(!suggestions.contains(&"drift".to_string()));
    }

    #[test]
    fn test_completer_drift_subcommands() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("drift ");
        assert!(suggestions.contains(&"detect".to_string()));
    }

    #[test]
    fn test_completer_load_no_suggestions() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        // File path completion is not implemented (OS level)
        let suggestions = completer.complete("load ");
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_completer_select_column_suggestions() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("select ");
        assert!(suggestions.contains(&"id".to_string()));
        assert!(suggestions.contains(&"name".to_string()));
        assert!(suggestions.contains(&"score".to_string()));
    }

    #[test]
    fn test_completer_drop_column_suggestions() {
        let mut session = ReplSession::new();
        session.load_dataset("test", create_test_dataset());

        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("drop ");
        assert!(suggestions.contains(&"id".to_string()));
    }

    #[test]
    fn test_completer_update_columns() {
        let mut session = ReplSession::new();
        let mut completer = SchemaAwareCompleter::new(&session);

        // Initially no columns
        let suggestions = completer.complete("select ");
        assert!(suggestions.is_empty());

        // Load dataset and update
        session.load_dataset("test", create_test_dataset());
        completer.update_columns(&session);

        let suggestions = completer.complete("select ");
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_completer_update_datasets() {
        let mut session = ReplSession::new();
        let mut completer = SchemaAwareCompleter::new(&session);

        // Initially no datasets
        let suggestions = completer.complete("use ");
        assert!(suggestions.is_empty());

        // Load dataset and update
        session.load_dataset("mydata", create_test_dataset());
        completer.update_datasets(&session);

        let suggestions = completer.complete("use ");
        assert!(suggestions.contains(&"mydata".to_string()));
    }

    #[test]
    fn test_completer_unknown_command() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        // Unknown command should return empty
        let suggestions = completer.complete("unknowncmd ");
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_completer_partial_command_filtering() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("he");
        assert!(suggestions.contains(&"head".to_string()));
        assert!(suggestions.contains(&"help".to_string()));
        assert!(!suggestions.contains(&"load".to_string()));
    }

    #[test]
    fn test_completer_quality_partial_subcommand() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("quality c");
        assert!(suggestions.contains(&"check".to_string()));
        assert!(!suggestions.contains(&"score".to_string()));
    }

    #[test]
    fn test_completer_use_with_partial_name() {
        let mut session = ReplSession::new();
        session.load_dataset("training_data", create_test_dataset());
        session.load_dataset("test_data", create_test_dataset());

        let completer = SchemaAwareCompleter::new(&session);

        let suggestions = completer.complete("use tr");
        assert!(suggestions.contains(&"training_data".to_string()));
        assert!(!suggestions.contains(&"test_data".to_string()));
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // COVERAGE BOOST: Command Parser Tests (commands.rs)
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_command_names_returns_all() {
        let names = CommandParser::command_names();
        assert!(names.contains(&"load"));
        assert!(names.contains(&"info"));
        assert!(names.contains(&"head"));
        assert!(names.contains(&"schema"));
        assert!(names.contains(&"quality"));
        assert!(names.contains(&"drift"));
        assert!(names.contains(&"convert"));
        assert!(names.contains(&"datasets"));
        assert!(names.contains(&"use"));
        assert!(names.contains(&"history"));
        assert!(names.contains(&"help"));
        assert!(names.contains(&"export"));
        assert!(names.contains(&"validate"));
        assert!(names.contains(&"quit"));
        assert!(names.contains(&"exit"));
    }

    #[test]
    fn test_subcommands_quality() {
        let subs = CommandParser::subcommands("quality");
        assert!(subs.contains(&"check"));
        assert!(subs.contains(&"score"));
    }

    #[test]
    fn test_subcommands_drift() {
        let subs = CommandParser::subcommands("drift");
        assert!(subs.contains(&"detect"));
    }

    #[test]
    fn test_subcommands_unknown() {
        let subs = CommandParser::subcommands("unknown");
        assert!(subs.is_empty());
    }

    #[test]
    fn test_flags_quality_score() {
        let flags = CommandParser::flags("quality", Some("score"));
        assert!(flags.contains(&"--suggest"));
        assert!(flags.contains(&"--json"));
        assert!(flags.contains(&"--badge"));
    }

    #[test]
    fn test_flags_export() {
        let flags = CommandParser::flags("export", None);
        assert!(flags.contains(&"--json"));
    }

    #[test]
    fn test_flags_validate() {
        let flags = CommandParser::flags("validate", None);
        assert!(flags.contains(&"--schema"));
    }

    #[test]
    fn test_flags_history() {
        let flags = CommandParser::flags("history", None);
        assert!(flags.contains(&"--export"));
    }

    #[test]
    fn test_flags_unknown() {
        let flags = CommandParser::flags("unknown", None);
        assert!(flags.is_empty());
    }

    #[test]
    fn test_parse_quality_score_badge() {
        let cmd = CommandParser::parse("quality score --badge").unwrap();
        assert!(matches!(cmd, ReplCommand::QualityScore { badge: true, .. }));
    }

    #[test]
    fn test_parse_history_shorthand() {
        let cmd = CommandParser::parse("history -e").unwrap();
        assert!(matches!(cmd, ReplCommand::History { export: true }));
    }

    #[test]
    fn test_parse_validate_shorthand() {
        let cmd = CommandParser::parse("validate -s schema.yaml").unwrap();
        assert!(matches!(cmd, ReplCommand::Validate { schema } if schema == "schema.yaml"));
    }

    #[test]
    fn test_parse_case_insensitive() {
        let cmd1 = CommandParser::parse("LOAD file.csv").unwrap();
        let cmd2 = CommandParser::parse("Load FILE.CSV").unwrap();
        assert!(matches!(cmd1, ReplCommand::Load { .. }));
        assert!(matches!(cmd2, ReplCommand::Load { .. }));
    }

    #[test]
    fn test_parse_head_invalid_number() {
        // Parser returns error for invalid numbers
        let result = CommandParser::parse("head abc");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid number"));
    }

    #[test]
    fn test_parse_load_missing_path() {
        let result = CommandParser::parse("load");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_use_missing_name() {
        let result = CommandParser::parse("use");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_convert_missing_format() {
        let result = CommandParser::parse("convert");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_drift_missing_reference() {
        let result = CommandParser::parse("drift detect");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_quality_invalid_subcommand() {
        let result = CommandParser::parse("quality invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_quality_missing_subcommand() {
        let result = CommandParser::parse("quality");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_drift_invalid_subcommand() {
        let result = CommandParser::parse("drift invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_validate_missing_schema() {
        let result = CommandParser::parse("validate --schema");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_export_missing_what() {
        let result = CommandParser::parse("export");
        assert!(result.is_err());
    }
}
