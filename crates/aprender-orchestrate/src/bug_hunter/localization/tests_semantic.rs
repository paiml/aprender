//! Tests for semantic scoring, static findings, and localize integration.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::bug_hunter::types::{ChannelWeights, LocalizationStrategy};

use super::*;

// =========================================================================
// Coverage gap: compute_semantic_score()
// =========================================================================

#[test]
fn test_compute_semantic_score_no_error_message() {
    let localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    // No error message set
    let score = localizer.compute_semantic_score(
        Path::new("test.rs"),
        1,
        "fn main() { println!(\"hello\"); }",
    );
    assert_eq!(score, 0.0, "Should return 0.0 when no error message is set");
}

#[test]
fn test_compute_semantic_score_matching_keywords() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    localizer.set_error_message("index overflow detected in array");

    // Content with matching keywords on line 1
    let content = "fn process_array() { index overflow check }";
    let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
    assert!(score > 0.0, "Should match keywords from error message");
    assert!(score <= 1.0);
}

#[test]
fn test_compute_semantic_score_no_matching_keywords() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    localizer.set_error_message("index overflow detected in array");

    // Content with no matching keywords
    let content = "fn hello_world() { println!(\"hi\"); }";
    let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
    assert_eq!(score, 0.0, "Should return 0.0 when no keywords match");
}

#[test]
fn test_compute_semantic_score_partial_match() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    localizer.set_error_message("buffer overflow detected in memory allocation");

    // Only some keywords match (overflow matches, but not buffer/detected/memory/allocation)
    let content = "fn check() { handle overflow here }";
    let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
    assert!(score > 0.0, "Should have partial match");
    assert!(score < 1.0, "Should not be perfect match");
}

#[test]
fn test_compute_semantic_score_all_keywords_match() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    // Only words > 3 chars are used as keywords
    localizer.set_error_message("buffer overflow detected");

    // Line with all keywords matching
    let content = "buffer overflow detected here";
    let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "Should return 1.0 when all keywords match, got {}",
        score
    );
}

#[test]
fn test_compute_semantic_score_short_words_filtered() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    // All words <= 3 chars are filtered out
    localizer.set_error_message("a be on if");

    let content = "a be on if match";
    let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
    assert_eq!(score, 0.0, "Should return 0.0 when all error words are <= 3 chars");
}

#[test]
fn test_compute_semantic_score_multiline_content() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    localizer.set_error_message("panic detected crash");

    // Multi-line content, target line 2
    let content = "fn safe() {}\nfn dangerous() { panic detected crash }\nfn other() {}";
    let score = localizer.compute_semantic_score(Path::new("test.rs"), 2, content);
    assert!(score > 0.0, "Should match keywords on the target line");
}

#[test]
fn test_compute_semantic_score_line_beyond_content() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    localizer.set_error_message("panic detected");

    // Only 1 line, but asking for line 100
    let content = "fn main() {}";
    let score = localizer.compute_semantic_score(Path::new("test.rs"), 100, content);
    assert_eq!(score, 0.0, "Should return 0.0 for line beyond content length");
}

#[test]
fn test_compute_semantic_score_case_insensitive() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());
    localizer.set_error_message("BUFFER OVERFLOW");

    let content = "fn check() { buffer overflow here }";
    let score = localizer.compute_semantic_score(Path::new("test.rs"), 1, content);
    assert!(score > 0.0, "Should be case-insensitive when matching keywords");
}

// =========================================================================
// Coverage gap: add_static_finding() and set_error_message()
// =========================================================================

#[test]
fn test_add_static_finding() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

    localizer.add_static_finding(Path::new("src/lib.rs"), 42, 0.75);
    localizer.add_static_finding(Path::new("src/main.rs"), 10, 0.3);

    assert_eq!(*localizer.static_findings.get(&(PathBuf::from("src/lib.rs"), 42)).unwrap(), 0.75);
    assert_eq!(*localizer.static_findings.get(&(PathBuf::from("src/main.rs"), 10)).unwrap(), 0.3);
}

#[test]
fn test_set_error_message() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

    assert!(localizer.error_message.is_none());
    localizer.set_error_message("test error");
    assert_eq!(localizer.error_message.as_deref(), Some("test error"));
}

// =========================================================================
// Coverage gap: localize() with error message (triggers semantic scoring)
// =========================================================================

#[test]
fn test_localize_with_error_message() {
    let temp = std::env::temp_dir().join("test_localize_semantic");
    let _ = std::fs::create_dir_all(&temp);

    // Create a file with content matching the error message
    let file_name = "semantic_test.rs";
    std::fs::write(
        temp.join(file_name),
        "fn process() {\n    panic!(\"buffer overflow detected\");\n}\n",
    )
    .unwrap();

    let mut localizer = MultiChannelLocalizer::new(
        LocalizationStrategy::MultiChannel,
        ChannelWeights {
            spectrum: 0.0,
            mutation: 0.0,
            static_analysis: 0.0,
            semantic: 1.0,
            quality: 0.0,
        },
    );

    // Set error message and add location at the file
    localizer.set_error_message("buffer overflow detected");
    localizer.static_findings.insert((PathBuf::from(file_name), 2), 0.5);

    let results = localizer.localize(&temp);
    assert_eq!(results.len(), 1);
    // Semantic score should be computed because error_message is set
    // The file exists so read will succeed and compute_semantic_score runs
    // The final_score uses semantic weight=1.0, so final_score should reflect semantic_score
}

#[test]
fn test_localize_with_error_message_file_not_found() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::MultiChannel, ChannelWeights::default());

    localizer.set_error_message("some error");
    localizer.static_findings.insert((PathBuf::from("nonexistent_file.rs"), 10), 0.5);

    // /nonexistent as project_path -- file won't be found, semantic_score stays 0.0
    let results = localizer.localize(Path::new("/nonexistent_project"));
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].semantic_score, 0.0);
}

// =========================================================================
// Coverage gap: localize with add_coverage integration
// =========================================================================

#[test]
fn test_localize_after_add_coverage() {
    let mut localizer =
        MultiChannelLocalizer::new(LocalizationStrategy::Sbfl, ChannelWeights::default());

    let mut pass_lines = HashMap::new();
    pass_lines.insert((PathBuf::from("lib.rs"), 10), 1);

    let mut fail_lines = HashMap::new();
    fail_lines.insert((PathBuf::from("lib.rs"), 10), 1);
    fail_lines.insert((PathBuf::from("lib.rs"), 20), 1);

    localizer.add_coverage(&[
        TestCoverage { test_name: "pass".to_string(), passed: true, executed_lines: pass_lines },
        TestCoverage { test_name: "fail".to_string(), passed: false, executed_lines: fail_lines },
    ]);

    let results = localizer.localize(Path::new("/tmp"));
    // Line 20 only in failed tests => higher score than line 10
    assert_eq!(results.len(), 2);
    let line20 = results.iter().find(|r| r.line == 20).unwrap();
    let line10 = results.iter().find(|r| r.line == 10).unwrap();
    assert!(
        line20.final_score >= line10.final_score,
        "Line only in failing tests should have higher score"
    );
}
