//! Schema-Aware Autocomplete (ALIM-REPL-003)
//!
//! Implements Poka-Yoke (mistake proofing) through context-aware completion.
//! Users cannot easily enter invalid commands or column names.
//!
//! # References
//! - [6] Myers (1990). Taxonomies of Visual Programming
//! - [13] Ko et al. (2004). Six Learning Barriers

use super::{commands::CommandParser, session::ReplSession};

/// Schema-aware completer for Poka-Yoke input validation
#[derive(Debug)]
pub struct SchemaAwareCompleter {
    /// Known command names
    commands: Vec<String>,
    /// Known subcommands by command
    subcommands: Vec<(String, Vec<String>)>,
    /// Column names from active dataset schema
    columns: Vec<String>,
    /// Loaded dataset names
    datasets: Vec<String>,
}

impl SchemaAwareCompleter {
    /// Create a new completer from session state
    #[must_use]
    pub fn new(session: &ReplSession) -> Self {
        Self {
            commands: CommandParser::command_names()
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            subcommands: vec![
                (
                    "quality".to_string(),
                    vec!["check".to_string(), "score".to_string()],
                ),
                ("drift".to_string(), vec!["detect".to_string()]),
            ],
            columns: session.column_names(),
            datasets: session.datasets(),
        }
    }

    /// Get completions for the given input
    ///
    /// Returns a list of possible completions.
    #[must_use]
    pub fn complete(&self, input: &str) -> Vec<String> {
        let trimmed = input.trim();
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        let ends_with_space = input.ends_with(' ') && !trimmed.is_empty();

        match parts.len() {
            0 => {
                // Empty input - suggest all commands
                self.commands.clone()
            }
            1 if !ends_with_space => {
                // Partial command - filter matching commands
                let prefix = parts[0].to_lowercase();
                self.commands
                    .iter()
                    .filter(|cmd| cmd.starts_with(&prefix))
                    .cloned()
                    .collect()
            }
            1 if ends_with_space => {
                // Command complete, start context completion
                let cmd = parts[0].to_lowercase();
                self.context_complete(&cmd, &[], input)
            }
            _ => {
                // Command entered - context-aware completion
                let cmd = parts[0].to_lowercase();
                self.context_complete(&cmd, &parts[1..], input)
            }
        }
    }

    /// Context-aware completion based on command
    fn context_complete(&self, cmd: &str, args: &[&str], full_input: &str) -> Vec<String> {
        match cmd {
            "load" => {
                // File path completion - not implemented (OS level)
                vec![]
            }
            "use" => {
                // Suggest loaded dataset names
                if args.is_empty() || !full_input.ends_with(' ') {
                    let prefix = args.first().map_or("", |s| *s);
                    self.datasets
                        .iter()
                        .filter(|d| d.starts_with(prefix))
                        .cloned()
                        .collect()
                } else {
                    self.datasets.clone()
                }
            }
            "quality" | "drift" => {
                // Suggest subcommands
                self.complete_subcommand(cmd, args)
            }
            "convert" => {
                // Suggest formats
                vec!["csv".to_string(), "parquet".to_string(), "json".to_string()]
                    .into_iter()
                    .filter(|f| args.first().map_or(true, |prefix| f.starts_with(*prefix)))
                    .collect()
            }
            "select" | "drop" => {
                // Suggest column names for column operations
                if args.is_empty() || !full_input.ends_with(' ') {
                    let prefix = args.last().map_or("", |s| *s);
                    self.columns
                        .iter()
                        .filter(|c| c.starts_with(prefix))
                        .cloned()
                        .collect()
                } else {
                    self.columns.clone()
                }
            }
            "help" => {
                // Suggest help topics
                vec![
                    "quality".to_string(),
                    "drift".to_string(),
                    "export".to_string(),
                ]
                .into_iter()
                .filter(|t| args.first().map_or(true, |prefix| t.starts_with(*prefix)))
                .collect()
            }
            _ => vec![],
        }
    }

    /// Complete subcommands for commands with children
    fn complete_subcommand(&self, cmd: &str, args: &[&str]) -> Vec<String> {
        let subcommands: Vec<String> = self
            .subcommands
            .iter()
            .find(|(c, _)| c == cmd)
            .map(|(_, subs)| subs.clone())
            .unwrap_or_default();

        if args.is_empty() {
            subcommands
        } else {
            let prefix = args[0].to_lowercase();
            subcommands
                .into_iter()
                .filter(|sub| sub.starts_with(&prefix))
                .collect()
        }
    }

    /// Update column names from session
    pub fn update_columns(&mut self, session: &ReplSession) {
        self.columns = session.column_names();
    }

    /// Update dataset names from session
    pub fn update_datasets(&mut self, session: &ReplSession) {
        self.datasets = session.datasets();
    }
}

#[cfg(feature = "repl")]
use reedline::{Completer, Span, Suggestion};

#[cfg(feature = "repl")]
impl Completer for SchemaAwareCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        // Get the part of the line up to cursor
        let input = &line[..pos];

        // Get completions
        let completions = Self::complete(self, input);

        // Find the word being completed
        let word_start = input.rfind(' ').map_or(0, |i| i + 1);
        let span = Span::new(word_start, pos);

        completions
            .into_iter()
            .map(|value| Suggestion {
                value,
                description: None,
                style: None,
                extra: None,
                span,
                append_whitespace: true,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema as ArrowSchema},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::ArrowDataset;

    fn create_test_dataset() -> ArrowDataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();
        ArrowDataset::new(vec![batch]).unwrap()
    }

    #[test]
    fn test_completer_new() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        assert!(!completer.commands.is_empty());
    }

    #[test]
    fn test_completer_empty_input() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("");
        assert!(!completions.is_empty());
        assert!(completions.contains(&"help".to_string()));
    }

    #[test]
    fn test_completer_partial_command() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("he");
        assert!(
            completions.contains(&"help".to_string()) || completions.contains(&"head".to_string())
        );
    }

    #[test]
    fn test_completer_command_with_space() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("quality ");
        assert!(
            completions.contains(&"check".to_string())
                || completions.contains(&"score".to_string())
        );
    }

    #[test]
    fn test_completer_use_with_datasets() {
        let mut session = ReplSession::new();
        session.load_dataset("test.parquet", create_test_dataset());
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("use ");
        assert!(completions.contains(&"test.parquet".to_string()));
    }

    #[test]
    fn test_completer_select_with_columns() {
        let mut session = ReplSession::new();
        session.load_dataset("test.parquet", create_test_dataset());
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("select ");
        assert!(completions.contains(&"id".to_string()));
        assert!(completions.contains(&"name".to_string()));
    }

    #[test]
    fn test_completer_drop_with_columns() {
        let mut session = ReplSession::new();
        session.load_dataset("test.parquet", create_test_dataset());
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("drop ");
        assert!(completions.contains(&"id".to_string()));
        assert!(completions.contains(&"name".to_string()));
    }

    #[test]
    fn test_completer_convert_formats() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("convert ");
        assert!(completions.contains(&"csv".to_string()));
        assert!(completions.contains(&"parquet".to_string()));
        assert!(completions.contains(&"json".to_string()));
    }

    #[test]
    fn test_completer_convert_partial_format() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("convert cs");
        assert!(completions.contains(&"csv".to_string()));
        assert!(!completions.contains(&"parquet".to_string()));
    }

    #[test]
    fn test_completer_quality_subcommands() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("quality ");
        assert!(completions.contains(&"check".to_string()));
        assert!(completions.contains(&"score".to_string()));
    }

    #[test]
    fn test_completer_quality_partial_subcommand() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("quality ch");
        assert!(completions.contains(&"check".to_string()));
        assert!(!completions.contains(&"score".to_string()));
    }

    #[test]
    fn test_completer_drift_subcommands() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("drift ");
        assert!(completions.contains(&"detect".to_string()));
    }

    #[test]
    fn test_completer_help_topics() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("help ");
        assert!(completions.contains(&"quality".to_string()));
        assert!(completions.contains(&"drift".to_string()));
        assert!(completions.contains(&"export".to_string()));
    }

    #[test]
    fn test_completer_help_partial_topic() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("help qu");
        assert!(completions.contains(&"quality".to_string()));
        assert!(!completions.contains(&"drift".to_string()));
    }

    #[test]
    fn test_completer_load_returns_empty() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("load ");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_completer_unknown_command_returns_empty() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("unknowncommand ");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_completer_update_columns() {
        let session = ReplSession::new();
        let mut completer = SchemaAwareCompleter::new(&session);
        assert!(completer.columns.is_empty());

        let mut session_with_data = ReplSession::new();
        session_with_data.load_dataset("test.parquet", create_test_dataset());
        completer.update_columns(&session_with_data);
        assert!(!completer.columns.is_empty());
    }

    #[test]
    fn test_completer_update_datasets() {
        let session = ReplSession::new();
        let mut completer = SchemaAwareCompleter::new(&session);
        assert!(completer.datasets.is_empty());

        let mut session_with_data = ReplSession::new();
        session_with_data.load_dataset("test.parquet", create_test_dataset());
        completer.update_datasets(&session_with_data);
        assert!(completer.datasets.contains(&"test.parquet".to_string()));
    }

    #[test]
    fn test_completer_select_partial_column() {
        let mut session = ReplSession::new();
        session.load_dataset("test.parquet", create_test_dataset());
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("select i");
        assert!(completions.contains(&"id".to_string()));
        assert!(!completions.contains(&"name".to_string()));
    }

    #[test]
    fn test_completer_use_partial_dataset() {
        let mut session = ReplSession::new();
        session.load_dataset("test.parquet", create_test_dataset());
        session.load_dataset("other.csv", create_test_dataset());
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("use te");
        assert!(completions.contains(&"test.parquet".to_string()));
        assert!(!completions.contains(&"other.csv".to_string()));
    }

    #[test]
    fn test_completer_multiple_args() {
        let mut session = ReplSession::new();
        session.load_dataset("test.parquet", create_test_dataset());
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete("select id ");
        assert!(completions.contains(&"id".to_string()));
        assert!(completions.contains(&"name".to_string()));
    }

    #[test]
    fn test_completer_debug() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let debug = format!("{:?}", completer);
        assert!(debug.contains("SchemaAwareCompleter"));
    }

    #[test]
    fn test_complete_subcommand_unknown_command() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete_subcommand("unknown", &[]);
        assert!(completions.is_empty());
    }

    #[test]
    fn test_complete_subcommand_with_partial() {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete_subcommand("quality", &["sc"]);
        assert!(completions.contains(&"score".to_string()));
        assert!(!completions.contains(&"check".to_string()));
    }
}
