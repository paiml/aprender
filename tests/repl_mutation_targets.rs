#![cfg(feature = "repl")]
#![allow(clippy::unwrap_used)]
//! Targeted mutation testing for REPL module
//!
//! These tests are designed to catch common mutations:
//! - Boundary conditions (off-by-one errors)
//! - Boolean inversions
//! - Return value changes
//! - Comparison operator changes

use alimentar::repl::{
    CommandParser, DisplayConfig, HealthStatus, ReplCommand, ReplSession, SchemaAwareCompleter,
};

// ═══════════════════════════════════════════════════════════════════════════════
// BOUNDARY CONDITION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn mutation_head_boundary_zero() {
    // Test head with 0 - should parse as 0
    let result = CommandParser::parse("head 0");
    assert!(result.is_ok());
    if let Ok(ReplCommand::Head { n }) = result {
        assert_eq!(n, 0);
    }
}

#[test]
fn mutation_head_boundary_one() {
    let result = CommandParser::parse("head 1");
    assert!(result.is_ok());
    if let Ok(ReplCommand::Head { n }) = result {
        assert_eq!(n, 1);
    }
}

#[test]
fn mutation_head_default_exactly_10() {
    // Default should be exactly 10, not 9 or 11
    let result = CommandParser::parse("head");
    assert!(result.is_ok());
    if let Ok(ReplCommand::Head { n }) = result {
        assert_eq!(n, 10);
    }
}

#[test]
fn mutation_display_config_defaults() {
    let config = DisplayConfig::default();
    // Exact values matter for mutations
    assert_eq!(config.max_rows, 10);
    assert_eq!(config.max_column_width, 50);
    assert!(config.color_output);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BOOLEAN INVERSION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn mutation_quality_score_suggest_true() {
    let result = CommandParser::parse("quality score --suggest");
    assert!(result.is_ok());
    if let Ok(ReplCommand::QualityScore {
        suggest,
        json,
        badge,
    }) = result
    {
        assert!(suggest, "suggest should be true");
        assert!(!json, "json should be false");
        assert!(!badge, "badge should be false");
    }
}

#[test]
fn mutation_quality_score_json_true() {
    let result = CommandParser::parse("quality score --json");
    assert!(result.is_ok());
    if let Ok(ReplCommand::QualityScore {
        suggest,
        json,
        badge,
    }) = result
    {
        assert!(!suggest, "suggest should be false");
        assert!(json, "json should be true");
        assert!(!badge, "badge should be false");
    }
}

#[test]
fn mutation_quality_score_badge_true() {
    let result = CommandParser::parse("quality score --badge");
    assert!(result.is_ok());
    if let Ok(ReplCommand::QualityScore {
        suggest,
        json,
        badge,
    }) = result
    {
        assert!(!suggest, "suggest should be false");
        assert!(!json, "json should be false");
        assert!(badge, "badge should be true");
    }
}

#[test]
fn mutation_quality_score_all_false() {
    let result = CommandParser::parse("quality score");
    assert!(result.is_ok());
    if let Ok(ReplCommand::QualityScore {
        suggest,
        json,
        badge,
    }) = result
    {
        assert!(!suggest, "suggest should be false");
        assert!(!json, "json should be false");
        assert!(!badge, "badge should be false");
    }
}

#[test]
fn mutation_history_export_true() {
    let result = CommandParser::parse("history --export");
    assert!(result.is_ok());
    if let Ok(ReplCommand::History { export }) = result {
        assert!(export, "export should be true");
    }
}

#[test]
fn mutation_history_export_false() {
    let result = CommandParser::parse("history");
    assert!(result.is_ok());
    if let Ok(ReplCommand::History { export }) = result {
        assert!(!export, "export should be false");
    }
}

#[test]
fn mutation_export_json_true() {
    let result = CommandParser::parse("export quality --json");
    assert!(result.is_ok());
    if let Ok(ReplCommand::Export { what: _, json }) = result {
        assert!(json, "json should be true");
    }
}

#[test]
fn mutation_export_json_false() {
    let result = CommandParser::parse("export quality");
    assert!(result.is_ok());
    if let Ok(ReplCommand::Export { what: _, json }) = result {
        assert!(!json, "json should be false");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HEALTH STATUS GRADE BOUNDARY TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn mutation_grade_a_is_healthy() {
    assert_eq!(HealthStatus::from_grade('A'), HealthStatus::Healthy);
}

#[test]
fn mutation_grade_b_is_healthy() {
    assert_eq!(HealthStatus::from_grade('B'), HealthStatus::Healthy);
}

#[test]
fn mutation_grade_c_is_warning() {
    assert_eq!(HealthStatus::from_grade('C'), HealthStatus::Warning);
}

#[test]
fn mutation_grade_d_is_warning() {
    assert_eq!(HealthStatus::from_grade('D'), HealthStatus::Warning);
}

#[test]
fn mutation_grade_f_is_critical() {
    assert_eq!(HealthStatus::from_grade('F'), HealthStatus::Critical);
}

#[test]
fn mutation_grade_unknown_is_none() {
    assert_eq!(HealthStatus::from_grade('X'), HealthStatus::None);
    assert_eq!(HealthStatus::from_grade('E'), HealthStatus::None);
    assert_eq!(HealthStatus::from_grade('1'), HealthStatus::None);
}

#[test]
fn mutation_indicator_warning_has_single_bang() {
    assert_eq!(HealthStatus::Warning.indicator(), "!");
}

#[test]
fn mutation_indicator_critical_has_double_bang() {
    assert_eq!(HealthStatus::Critical.indicator(), "!!");
}

#[test]
fn mutation_indicator_healthy_is_empty() {
    assert_eq!(HealthStatus::Healthy.indicator(), "");
}

#[test]
fn mutation_indicator_none_is_empty() {
    assert_eq!(HealthStatus::None.indicator(), "");
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMAND PARSER EXACT MATCHES
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn mutation_quit_variants_all_work() {
    assert!(matches!(
        CommandParser::parse("quit").unwrap(),
        ReplCommand::Quit
    ));
    assert!(matches!(
        CommandParser::parse("exit").unwrap(),
        ReplCommand::Quit
    ));
    assert!(matches!(
        CommandParser::parse("q").unwrap(),
        ReplCommand::Quit
    ));
}

#[test]
fn mutation_help_variants_all_work() {
    assert!(matches!(
        CommandParser::parse("help").unwrap(),
        ReplCommand::Help { topic: None }
    ));
    assert!(matches!(
        CommandParser::parse("?").unwrap(),
        ReplCommand::Help { topic: None }
    ));
}

#[test]
fn mutation_info_exact_match() {
    assert!(matches!(
        CommandParser::parse("info").unwrap(),
        ReplCommand::Info
    ));
}

#[test]
fn mutation_schema_exact_match() {
    assert!(matches!(
        CommandParser::parse("schema").unwrap(),
        ReplCommand::Schema
    ));
}

#[test]
fn mutation_datasets_exact_match() {
    assert!(matches!(
        CommandParser::parse("datasets").unwrap(),
        ReplCommand::Datasets
    ));
}

#[test]
fn mutation_quality_check_exact_match() {
    assert!(matches!(
        CommandParser::parse("quality check").unwrap(),
        ReplCommand::QualityCheck
    ));
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPLETER CONTEXT COMPLETIONS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn mutation_completer_load_returns_empty() {
    // Load path completion is OS-level, should return empty
    let session = ReplSession::new();
    let completer = SchemaAwareCompleter::new(&session);
    let completions = completer.complete("load ");
    assert!(completions.is_empty());
}

#[test]
fn mutation_completer_unknown_returns_empty() {
    let session = ReplSession::new();
    let completer = SchemaAwareCompleter::new(&session);
    let completions = completer.complete("unknowncmd ");
    assert!(completions.is_empty());
}

#[test]
fn mutation_completer_convert_exact_three_formats() {
    let session = ReplSession::new();
    let completer = SchemaAwareCompleter::new(&session);
    let completions = completer.complete("convert ");
    assert_eq!(completions.len(), 3);
}

#[test]
fn mutation_completer_help_exact_three_topics() {
    let session = ReplSession::new();
    let completer = SchemaAwareCompleter::new(&session);
    let completions = completer.complete("help ");
    assert_eq!(completions.len(), 3);
}

#[test]
fn mutation_completer_quality_exact_two_subcommands() {
    let session = ReplSession::new();
    let completer = SchemaAwareCompleter::new(&session);
    let completions = completer.complete("quality ");
    assert_eq!(completions.len(), 2);
}

#[test]
fn mutation_completer_drift_exact_one_subcommand() {
    let session = ReplSession::new();
    let completer = SchemaAwareCompleter::new(&session);
    let completions = completer.complete("drift ");
    assert_eq!(completions.len(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION STATE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn mutation_session_history_preserves_order() {
    let mut session = ReplSession::new();
    session.add_history("first");
    session.add_history("second");
    session.add_history("third");

    let history = session.history();
    assert_eq!(history[0], "first");
    assert_eq!(history[1], "second");
    assert_eq!(history[2], "third");
}

#[test]
fn mutation_session_export_has_shebang() {
    let mut session = ReplSession::new();
    session.add_history("info");
    let export = session.export_history();
    assert!(export.starts_with("#!/usr/bin/env bash"));
}

#[test]
fn mutation_session_export_has_comment_header() {
    let mut session = ReplSession::new();
    session.add_history("info");
    let export = session.export_history();
    assert!(export.contains("# alimentar session export"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// ERROR CONDITION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn mutation_load_without_path_errors() {
    assert!(CommandParser::parse("load").is_err());
}

#[test]
fn mutation_use_without_name_errors() {
    assert!(CommandParser::parse("use").is_err());
}

#[test]
fn mutation_quality_without_subcommand_errors() {
    assert!(CommandParser::parse("quality").is_err());
}

#[test]
fn mutation_drift_without_subcommand_errors() {
    assert!(CommandParser::parse("drift").is_err());
}

#[test]
fn mutation_drift_detect_without_reference_errors() {
    assert!(CommandParser::parse("drift detect").is_err());
}

#[test]
fn mutation_convert_without_format_errors() {
    assert!(CommandParser::parse("convert").is_err());
}

#[test]
fn mutation_validate_without_schema_errors() {
    assert!(CommandParser::parse("validate").is_err());
}

#[test]
fn mutation_head_with_invalid_number_errors() {
    assert!(CommandParser::parse("head abc").is_err());
    assert!(CommandParser::parse("head -5").is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
// TARGETED MUTATION KILLERS - Catches specific mutant patterns
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn mutation_subcommands_quality_exact() {
    // Catches: delete match arm "quality" in CommandParser::subcommands
    // Catches: replace CommandParser::subcommands with vec![] or vec![""]
    let subs = CommandParser::subcommands("quality");
    assert_eq!(subs.len(), 2, "quality must have exactly 2 subcommands");
    assert!(
        subs.contains(&"check"),
        "quality must have 'check' subcommand"
    );
    assert!(
        subs.contains(&"score"),
        "quality must have 'score' subcommand"
    );
}

#[test]
fn mutation_subcommands_drift_exact() {
    let subs = CommandParser::subcommands("drift");
    assert_eq!(subs.len(), 1, "drift must have exactly 1 subcommand");
    assert_eq!(subs[0], "detect", "drift must have 'detect' subcommand");
}

#[test]
fn mutation_subcommands_unknown_empty() {
    let subs = CommandParser::subcommands("unknown");
    assert!(subs.is_empty(), "unknown command must have no subcommands");
}

#[test]
fn mutation_flags_quality_score_exact() {
    // Catches: replace CommandParser::flags with vec!["xyzzy"]
    let flags = CommandParser::flags("quality", Some("score"));
    assert_eq!(flags.len(), 3, "quality score must have exactly 3 flags");
    assert!(flags.contains(&"--suggest"), "must have --suggest");
    assert!(flags.contains(&"--json"), "must have --json");
    assert!(flags.contains(&"--badge"), "must have --badge");
}

#[test]
fn mutation_flags_export_exact() {
    // Catches: delete match arm ("export", _) in CommandParser::flags
    let flags = CommandParser::flags("export", None);
    assert_eq!(flags.len(), 1, "export must have exactly 1 flag");
    assert_eq!(flags[0], "--json", "export must have --json flag");
}

#[test]
fn mutation_flags_validate_exact() {
    let flags = CommandParser::flags("validate", None);
    assert_eq!(flags.len(), 1, "validate must have exactly 1 flag");
    assert_eq!(flags[0], "--schema", "validate must have --schema flag");
}

#[test]
fn mutation_flags_history_exact() {
    let flags = CommandParser::flags("history", None);
    assert_eq!(flags.len(), 1, "history must have exactly 1 flag");
    assert_eq!(flags[0], "--export", "history must have --export flag");
}

#[test]
fn mutation_flags_unknown_empty() {
    let flags = CommandParser::flags("unknown", None);
    assert!(flags.is_empty(), "unknown command must have no flags");
}

#[test]
fn mutation_validate_shorthand_s_flag() {
    // Catches: replace != with == in CommandParser::parse_validate
    // Catches: replace || with && in CommandParser::parse_validate
    let result = CommandParser::parse("validate -s schema.json");
    assert!(result.is_ok(), "-s shorthand must work");
    if let Ok(ReplCommand::Validate { schema }) = result {
        assert_eq!(schema, "schema.json");
    }
}

#[test]
fn mutation_validate_long_schema_flag() {
    let result = CommandParser::parse("validate --schema schema.json");
    assert!(result.is_ok(), "--schema flag must work");
    if let Ok(ReplCommand::Validate { schema }) = result {
        assert_eq!(schema, "schema.json");
    }
}

#[test]
fn mutation_validate_schema_at_boundary() {
    // Catches: replace < with <= in CommandParser::parse_validate
    // Test when --schema is last arg (no value after it)
    let result = CommandParser::parse("validate --schema");
    assert!(result.is_err(), "--schema without value must error");
}

#[test]
fn mutation_validate_s_at_boundary() {
    // Same boundary test for shorthand
    let result = CommandParser::parse("validate -s");
    assert!(result.is_err(), "-s without value must error");
}

#[test]
fn mutation_history_shorthand_e_flag() {
    // Catches: replace == with != in CommandParser::parse_history
    // Tests that -e shorthand works correctly
    let result = CommandParser::parse("history -e");
    assert!(result.is_ok(), "-e shorthand must work");
    if let Ok(ReplCommand::History { export }) = result {
        assert!(export, "history -e must have export=true");
    }
}

#[test]
fn mutation_history_without_flag_is_false() {
    // Tests that without flag export is definitely false
    let result = CommandParser::parse("history");
    assert!(result.is_ok());
    if let Ok(ReplCommand::History { export }) = result {
        assert!(!export, "history without flags must have export=false");
    }
}

#[test]
fn mutation_history_other_arg_not_export() {
    // If we pass some other arg, export should still be false
    // This catches mutation from == to !=
    let result = CommandParser::parse("history somethingelse");
    assert!(result.is_ok());
    if let Ok(ReplCommand::History { export }) = result {
        assert!(!export, "random arg should not trigger export");
    }
}
