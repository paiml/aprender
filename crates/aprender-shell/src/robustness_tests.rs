//! Robustness tests salvaged from the deleted `aprender-shell` CLI test suites.
//!
//! The `aprender-shell` binary was removed in f5db50ae0 ("enforce Rule 1 --
//! delete 7 unauthorized bins") per `contracts/apr-mono-binary-rule-v1.yaml`.
//! Its `tests/cli_integration.rs`, `tests/real_world_tests.rs` and
//! `tests/performance_tests.rs` kept driving `Command::cargo_bin("aprender-shell")`
//! and had been failing (or silently `#[ignore]`d) ever since.
//!
//! The assertions that were about *library* behaviour rather than argv plumbing
//! are re-expressed here against the public API, so they run under `--lib` — the
//! only aprender-shell target CI actually executes.

use crate::config::{suggest_with_fallback, ShellConfig};
use crate::error::ShellError;
use crate::model::MarkovModel;
use crate::paged_model::PagedMarkovModel;
use crate::validation::load_model_graceful;
use std::io::Write;
use tempfile::NamedTempFile;

// Benchmark fixtures, previously loaded by tests/real_world_tests.rs.
const SMALL_HISTORY: &str = include_str!("../benches/fixtures/small_history.txt");
const MEDIUM_HISTORY: &str = include_str!("../benches/fixtures/medium_history.txt");
const LARGE_HISTORY: &str = include_str!("../benches/fixtures/large_history.txt");

/// Strip comments and blank lines from a fixture, as the CLI's history parser did.
fn fixture_commands(content: &str) -> Vec<String> {
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToString::to_string)
        .collect()
}

/// Train an in-memory model on a fixture corpus.
fn train_on(content: &str) -> MarkovModel {
    let mut model = MarkovModel::new(3);
    model.train(&fixture_commands(content));
    model
}

// =========================================================================
// Chaos: malformed model files must degrade, never panic (was CLI_021)
// =========================================================================

/// A file holding only the APR magic bytes is a *corrupt* model, not a missing one.
#[test]
fn test_truncated_model_is_corrupt_not_missing() {
    let mut tmp = NamedTempFile::new().expect("create temp file");
    tmp.write_all(b"APRN").expect("write magic");
    tmp.flush().expect("flush");

    let result = load_model_graceful(tmp.path());

    // The file exists, so ModelNotFound would be a misdiagnosis.
    assert!(
        matches!(
            result,
            Err(ShellError::ModelCorrupted { .. }) | Err(ShellError::ModelLoadFailed { .. })
        ),
        "truncated model must report corruption, got {:?}",
        result.map(|_| "Ok(model)")
    );
}

/// Wrong magic bytes must be rejected rather than parsed as a body.
#[test]
fn test_wrong_magic_bytes_rejected() {
    let mut tmp = NamedTempFile::new().expect("create temp file");
    tmp.write_all(b"XXXX12345678901234567890")
        .expect("write bad magic");
    tmp.flush().expect("flush");

    let result = load_model_graceful(tmp.path());

    assert!(
        matches!(
            result,
            Err(ShellError::ModelCorrupted { .. }) | Err(ShellError::ModelLoadFailed { .. })
        ),
        "wrong magic must report corruption, got {:?}",
        result.map(|_| "Ok(model)")
    );
}

// =========================================================================
// Chaos: adversarial prefixes (was CLI_021)
// =========================================================================

/// A 10 KB prefix must actually be truncated to the configured bound.
#[test]
fn test_oversized_prefix_is_truncated_to_bound() {
    let config = ShellConfig::default();
    let long_prefix = "g".repeat(10_000);

    let truncated = config.truncate_prefix(&long_prefix);

    assert_eq!(
        truncated.len(),
        config.max_prefix_length,
        "oversized prefix was not truncated to max_prefix_length"
    );

    // And the full pipeline still honours the suggestion cap.
    let model = train_on(SMALL_HISTORY);
    let suggestions = suggest_with_fallback(&long_prefix, Some(&model), &config);
    assert!(suggestions.len() <= config.max_suggestions);
}

/// Truncation must land on a UTF-8 char boundary, not slice through a codepoint.
#[test]
fn test_truncation_backs_off_to_char_boundary() {
    // 100 x "é" = 200 bytes; byte 101 is mid-codepoint.
    let prefix = "é".repeat(100);
    let config = ShellConfig::default().with_max_prefix_length(101);

    let truncated = config.truncate_prefix(&prefix);

    assert_eq!(
        truncated.len(),
        100,
        "truncation must back off from the mid-codepoint boundary at 101"
    );
    assert_eq!(truncated.chars().count(), 50);
}

/// Unicode edge cases must flow through the suggestion pipeline without panicking.
///
/// The bound is deliberately 7 bytes so that several of these prefixes are cut
/// mid-codepoint — a naive `&prefix[..max]` slice panics here.
#[test]
fn test_unicode_prefixes_handled_gracefully() {
    let model = train_on(SMALL_HISTORY);
    let config = ShellConfig::default().with_max_prefix_length(7);

    let cases = [
        "🚀".to_string(),                // emoji
        "日本語".to_string(),            // CJK (9 bytes, cut at 7)
        "مرحبا".to_string(),             // RTL (10 bytes, cut at 7)
        "\u{FEFF}git".to_string(),       // BOM
        "git\u{200B}status".to_string(), // zero-width space
        "git\u{202E}status".to_string(), // RTL override
        "é".repeat(100),                 // many multi-byte chars (200 bytes)
    ];

    for prefix in &cases {
        let truncated = config.truncate_prefix(prefix);
        assert!(
            truncated.len() <= config.max_prefix_length,
            "prefix {prefix:?} exceeded the byte bound after truncation"
        );
        assert!(
            prefix.starts_with(truncated),
            "prefix {prefix:?} truncated to a non-prefix {truncated:?}"
        );

        let suggestions = suggest_with_fallback(prefix, Some(&model), &config);
        assert!(
            suggestions.len() <= config.max_suggestions,
            "prefix {prefix:?} exceeded the suggestion cap"
        );
        assert!(
            suggestions.iter().all(|(s, _)| !s.is_empty()),
            "prefix {prefix:?} produced an empty suggestion"
        );
    }
}

// =========================================================================
// Chaos: concurrent readers of one model file (was CLI_021)
// =========================================================================

/// Five threads loading the same model file must all see identical suggestions.
#[test]
fn test_concurrent_readers_agree() {
    let model = train_on(MEDIUM_HISTORY);
    let path = NamedTempFile::new().expect("create model file");
    model.save(path.path()).expect("save model");

    // Ask for far more than exist so score ties cannot change the set.
    let reference: std::collections::BTreeSet<String> = MarkovModel::load(path.path())
        .expect("load model")
        .suggest("git ", 1000)
        .into_iter()
        .map(|(s, _)| s)
        .collect();
    assert!(
        !reference.is_empty(),
        "medium fixture must yield git suggestions"
    );

    let model_path = path.path().to_path_buf();
    let handles: Vec<_> = (0..5)
        .map(|_| {
            let model_path = model_path.clone();
            let reference = reference.clone();
            std::thread::spawn(move || {
                for _ in 0..10 {
                    let loaded = MarkovModel::load(&model_path).expect("concurrent load");
                    let got: std::collections::BTreeSet<String> = loaded
                        .suggest("git ", 1000)
                        .into_iter()
                        .map(|(s, _)| s)
                        .collect();
                    assert_eq!(got, reference, "concurrent reader diverged");
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("reader thread panicked");
    }
}

// =========================================================================
// Corpus scale: real fixtures round-trip through .apr (was REAL_001..003)
// =========================================================================

/// Suggestions for a command family must stay inside that family.
fn assert_family(model: &MarkovModel, prefix: &str, family: &str) {
    let suggestions = model.suggest(prefix, 1000);
    assert!(
        !suggestions.is_empty(),
        "expected suggestions for {prefix:?}"
    );
    for (suggestion, _) in &suggestions {
        assert!(
            suggestion.starts_with(family),
            "{prefix:?} leaked a non-{family} suggestion: {suggestion:?}"
        );
    }
}

#[test]
fn test_small_fixture_round_trip_keeps_families_separate() {
    let model = train_on(SMALL_HISTORY);
    let path = NamedTempFile::new().expect("create model file");
    model.save(path.path()).expect("save model");

    let loaded = MarkovModel::load(path.path()).expect("load model");
    assert_eq!(loaded.total_commands(), model.total_commands());

    assert_family(&loaded, "git ", "git");
    assert_family(&loaded, "cargo ", "cargo");
}

#[test]
fn test_medium_fixture_covers_container_tooling() {
    let model = train_on(MEDIUM_HISTORY);

    assert_family(&model, "docker ", "docker");
    assert_family(&model, "kubectl ", "kubectl");
}

#[test]
fn test_large_fixture_completes_partial_token() {
    let model = train_on(LARGE_HISTORY);
    let path = NamedTempFile::new().expect("create model file");
    model.save(path.path()).expect("save large model");

    let loaded = MarkovModel::load(path.path()).expect("load large model");
    // "git co" is a partial token: every completion must extend it, never
    // fall back to the whole "git" family.
    assert_family(&loaded, "git co", "git co");
}

// =========================================================================
// Incremental update (was REAL_009)
// =========================================================================

/// `train_incremental` must add the new commands, not silently no-op.
#[test]
fn test_incremental_update_adds_new_commands() {
    let mut model = train_on(SMALL_HISTORY);
    let baseline = model.total_commands();

    assert!(
        model.suggest("new-special-command", 10).is_empty(),
        "fixture must not already contain the probe command"
    );

    model.train_incremental(&[
        "new-special-command arg1".to_string(),
        "new-special-command arg2".to_string(),
    ]);

    assert_eq!(model.total_commands(), baseline + 2);
    assert_eq!(model.last_trained_position(), baseline + 2);
    assert_family(&model, "new-special-command", "new-special-command");
}

// =========================================================================
// Paged model at a tight memory limit (was REAL_008)
// =========================================================================

/// A 1 MB-limited paged model must train, persist and reload the large fixture.
#[test]
fn test_paged_model_round_trip_under_tight_limit() {
    let commands = fixture_commands(LARGE_HISTORY);
    let mut paged = PagedMarkovModel::new(3, 1);
    paged.train(&commands);

    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("paged.model");
    paged.save(&path).expect("save paged model");

    let mut loaded = PagedMarkovModel::load(&path, 1).expect("load paged model");
    assert_eq!(loaded.total_commands(), commands.len());

    let suggestions = loaded.suggest("git ", 1000);
    assert!(
        !suggestions.is_empty(),
        "paged model must still suggest git commands"
    );
    for (suggestion, _) in &suggestions {
        assert!(
            suggestion.starts_with("git"),
            "paged model leaked a non-git suggestion: {suggestion:?}"
        );
    }
}
