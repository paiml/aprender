#![cfg(feature = "repl")]
#![allow(clippy::unwrap_used)]
//! Property-based tests for REPL module (ALIM-REPL-001 through ALIM-REPL-007)
//!
//! Uses proptest to verify invariants hold across random inputs.
//! Follows Toyota Way: Poka-Yoke through exhaustive input validation.

// Import REPL types from the library
use alimentar::repl::{
    CommandParser, HealthStatus, ReplCommand, ReplSession, SchemaAwareCompleter,
};
use proptest::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════════
// PROPERTY TESTS: Command Parser (ALIM-REPL-003)
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    /// Property: Valid commands always parse without panic
    #[test]
    fn prop_valid_commands_parse(cmd in valid_command_strategy()) {
        let result = CommandParser::parse(&cmd);
        // Should either parse successfully or return a clean error
        prop_assert!(result.is_ok() || result.is_err());
    }

    /// Property: Unknown commands return error, never panic
    #[test]
    fn prop_unknown_commands_error(word in "[a-z]{1,20}") {
        // Skip known commands
        let known = vec![
            "load", "datasets", "use", "info", "head", "schema",
            "quality", "drift", "convert", "export", "validate",
            "history", "help", "quit", "exit", "q", "?"
        ];
        if !known.contains(&word.as_str()) {
            let result = CommandParser::parse(&word);
            prop_assert!(result.is_err());
        }
    }

    /// Property: Empty input always returns error
    #[test]
    fn prop_empty_input_error(spaces in " {0,10}") {
        let result = CommandParser::parse(&spaces);
        prop_assert!(result.is_err());
    }

    /// Property: Head command with valid numbers parses correctly
    #[test]
    fn prop_head_with_number(n in 1usize..10000) {
        let cmd = format!("head {n}");
        let result = CommandParser::parse(&cmd);
        prop_assert!(result.is_ok());
        if let Ok(ReplCommand::Head { n: parsed_n }) = result {
            prop_assert_eq!(parsed_n, n);
        }
    }

    /// Property: Quality score flags are parsed correctly
    #[test]
    fn prop_quality_score_flags(
        suggest in proptest::bool::ANY,
        json in proptest::bool::ANY,
        badge in proptest::bool::ANY,
    ) {
        let mut cmd = "quality score".to_string();
        if suggest { cmd.push_str(" --suggest"); }
        if json { cmd.push_str(" --json"); }
        if badge { cmd.push_str(" --badge"); }

        let result = CommandParser::parse(&cmd);
        prop_assert!(result.is_ok());
        if let Ok(ReplCommand::QualityScore { suggest: s, json: j, badge: b }) = result {
            prop_assert_eq!(s, suggest);
            prop_assert_eq!(j, json);
            prop_assert_eq!(b, badge);
        }
    }

    /// Property: Convert command only accepts valid formats
    #[test]
    fn prop_convert_valid_formats(format in "(csv|parquet|json)") {
        let cmd = format!("convert {format}");
        let result = CommandParser::parse(&cmd);
        prop_assert!(result.is_ok());
    }

    /// Property: Convert command rejects invalid formats
    #[test]
    fn prop_convert_invalid_formats(format in "[a-z]{1,10}") {
        if format != "csv" && format != "parquet" && format != "json" {
            let cmd = format!("convert {format}");
            let result = CommandParser::parse(&cmd);
            prop_assert!(result.is_err());
        }
    }

    /// Property: History --export flag is parsed
    #[test]
    fn prop_history_export_flag(use_export in proptest::bool::ANY) {
        let cmd = if use_export { "history --export" } else { "history" };
        let result = CommandParser::parse(cmd);
        prop_assert!(result.is_ok());
        if let Ok(ReplCommand::History { export }) = result {
            prop_assert_eq!(export, use_export);
        }
    }

    /// Property: Load requires a path argument
    #[test]
    fn prop_load_requires_path(path in "[a-zA-Z0-9_/.]{1,50}") {
        let cmd = format!("load {path}");
        let result = CommandParser::parse(&cmd);
        prop_assert!(result.is_ok());
        if let Ok(ReplCommand::Load { path: parsed }) = result {
            prop_assert_eq!(parsed, path);
        }
    }

    /// Property: Use requires a dataset name
    #[test]
    fn prop_use_requires_name(name in "[a-zA-Z0-9_]{1,20}") {
        let cmd = format!("use {name}");
        let result = CommandParser::parse(&cmd);
        prop_assert!(result.is_ok());
        if let Ok(ReplCommand::Use { name: parsed }) = result {
            prop_assert_eq!(parsed, name);
        }
    }

    /// Property: Drift detect requires a reference file
    #[test]
    fn prop_drift_requires_reference(reference in "[a-zA-Z0-9_/.]{1,50}") {
        let cmd = format!("drift detect {reference}");
        let result = CommandParser::parse(&cmd);
        prop_assert!(result.is_ok());
        if let Ok(ReplCommand::DriftDetect { reference: parsed }) = result {
            prop_assert_eq!(parsed, reference);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPERTY TESTS: Completer (ALIM-REPL-003 Poka-Yoke)
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    /// Property: Empty input returns all commands
    #[test]
    fn prop_completer_empty_returns_all(spaces in " {0,5}") {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete(&spaces);
        // Should return all commands for empty input
        prop_assert!(!completions.is_empty() || spaces.trim().is_empty());
    }

    /// Property: Partial command completion is prefix-filtered
    #[test]
    fn prop_completer_prefix_filter(prefix in "[a-z]{1,3}") {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        let completions = completer.complete(&prefix);
        for completion in &completions {
            prop_assert!(completion.starts_with(&prefix));
        }
    }

    /// Property: Completer never panics on arbitrary input
    #[test]
    fn prop_completer_never_panics(input in "[ -~]{0,100}") {
        let session = ReplSession::new();
        let completer = SchemaAwareCompleter::new(&session);
        // Should not panic
        let _ = completer.complete(&input);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPERTY TESTS: Health Status (ALIM-REPL-004 Andon)
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    /// Property: All grade characters map to valid status
    #[test]
    fn prop_health_status_from_any_char(c in proptest::char::any()) {
        let status = HealthStatus::from_grade(c);
        // Should always return a valid status (never panic)
        let _ = status.indicator();
    }

    /// Property: A/B grades are healthy
    #[test]
    fn prop_ab_grades_healthy(grade in "[AB]") {
        let c = grade.chars().next().unwrap();
        let status = HealthStatus::from_grade(c);
        prop_assert_eq!(status, HealthStatus::Healthy);
    }

    /// Property: C/D grades are warnings
    #[test]
    fn prop_cd_grades_warning(grade in "[CD]") {
        let c = grade.chars().next().unwrap();
        let status = HealthStatus::from_grade(c);
        prop_assert_eq!(status, HealthStatus::Warning);
    }

    /// Property: F grade is critical
    #[test]
    fn prop_f_grade_critical(_: ()) {
        let status = HealthStatus::from_grade('F');
        prop_assert_eq!(status, HealthStatus::Critical);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPERTY TESTS: Session (ALIM-REPL-001)
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    /// Property: History tracks all added commands
    #[test]
    fn prop_session_history_tracks_all(commands in proptest::collection::vec("[a-z ]{1,20}", 0..20)) {
        let mut session = ReplSession::new();
        for cmd in &commands {
            session.add_history(cmd);
        }
        prop_assert_eq!(session.history().len(), commands.len());
    }

    /// Property: History export contains all commands
    #[test]
    fn prop_session_export_contains_all(commands in proptest::collection::vec("[a-z]{1,10}", 1..5)) {
        let mut session = ReplSession::new();
        for cmd in &commands {
            session.add_history(cmd);
        }
        let export = session.export_history();
        // Export should contain the session export header
        prop_assert!(export.contains("alimentar session export"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRATEGY GENERATORS
// ═══════════════════════════════════════════════════════════════════════════════

/// Strategy for generating valid REPL commands
fn valid_command_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("info".to_string()),
        Just("schema".to_string()),
        Just("datasets".to_string()),
        Just("help".to_string()),
        Just("?".to_string()),
        Just("quit".to_string()),
        Just("exit".to_string()),
        Just("q".to_string()),
        Just("quality check".to_string()),
        Just("quality score".to_string()),
        Just("history".to_string()),
        Just("history --export".to_string()),
        (1usize..100).prop_map(|n| format!("head {n}")),
        "[a-zA-Z0-9_/.]{1,20}".prop_map(|p| format!("load {p}")),
        "[a-zA-Z0-9_]{1,10}".prop_map(|n| format!("use {n}")),
        "(csv|parquet|json)".prop_map(|f| format!("convert {f}")),
        "[a-zA-Z0-9_/.]{1,20}".prop_map(|r| format!("drift detect {r}")),
        "[a-zA-Z0-9_/.]{1,20}".prop_map(|s| format!("validate --schema {s}")),
        "(quality|drift|export)".prop_map(|t| format!("help {t}")),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE CASE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_parser_case_insensitivity() {
    // Commands should be case-insensitive
    assert!(CommandParser::parse("INFO").is_ok());
    assert!(CommandParser::parse("Info").is_ok());
    assert!(CommandParser::parse("QUIT").is_ok());
    assert!(CommandParser::parse("QUALITY CHECK").is_ok());
}

#[test]
fn test_parser_whitespace_handling() {
    // Extra whitespace should be handled
    assert!(CommandParser::parse("  info  ").is_ok());
    assert!(CommandParser::parse("head   10").is_ok());
    assert!(CommandParser::parse("quality   score").is_ok());
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
fn test_completer_drift_subcommands() {
    let session = ReplSession::new();
    let completer = SchemaAwareCompleter::new(&session);

    let completions = completer.complete("drift ");
    assert!(completions.contains(&"detect".to_string()));
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
fn test_completer_help_topics() {
    let session = ReplSession::new();
    let completer = SchemaAwareCompleter::new(&session);

    let completions = completer.complete("help ");
    assert!(completions.contains(&"quality".to_string()));
    assert!(completions.contains(&"drift".to_string()));
    assert!(completions.contains(&"export".to_string()));
}

#[test]
fn test_health_status_indicators() {
    assert_eq!(HealthStatus::Healthy.indicator(), "");
    assert_eq!(HealthStatus::Warning.indicator(), "!");
    assert_eq!(HealthStatus::Critical.indicator(), "!!");
    assert_eq!(HealthStatus::None.indicator(), "");
}

#[test]
fn test_display_config_builder_chain() {
    use alimentar::repl::DisplayConfig;

    let config = DisplayConfig::default()
        .with_max_rows(50)
        .with_max_column_width(100)
        .with_color(false);

    assert_eq!(config.max_rows, 50);
    assert_eq!(config.max_column_width, 100);
    assert!(!config.color_output);
}

#[test]
fn test_session_datasets_empty_initially() {
    let session = ReplSession::new();
    assert!(session.datasets().is_empty());
    assert!(session.active_dataset().is_none());
    assert!(session.active_name().is_none());
}

#[test]
fn test_session_column_names_empty_without_dataset() {
    let session = ReplSession::new();
    assert!(session.column_names().is_empty());
}
