//! Pixel Regression Tests for apr-cli visualization commands (GH-122)
//!
//! These tests compare command output against golden snapshots to detect
//! unintended visual regressions in TUI output.
//!
//! Toyota Way: Jidoka - Build quality in, stop on defects.

#![allow(clippy::unwrap_used)]

use assert_cmd::Command;
use std::fs;
use std::path::PathBuf;

// ============================================================================
// Helper Functions
// ============================================================================

fn apr() -> Command {
    Command::cargo_bin("apr").expect("Failed to find apr binary")
}

fn snapshots_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("playbooks")
        .join("snapshots")
}

fn test_apr_file() -> PathBuf {
    snapshots_dir().join("test.apr")
}

/// Compare output against golden snapshot, stripping ANSI color codes
fn assert_matches_snapshot(output: &str, snapshot_name: &str) {
    let snapshot_path = snapshots_dir().join(snapshot_name);
    let expected = fs::read_to_string(&snapshot_path)
        .unwrap_or_else(|_| panic!("Failed to read snapshot: {}", snapshot_path.display()));

    // Strip ANSI codes for comparison
    let actual_clean = strip_ansi_codes(output);
    let expected_clean = strip_ansi_codes(&expected);

    // Compare line by line for better error messages
    let actual_lines: Vec<&str> = actual_clean.lines().collect();
    let expected_lines: Vec<&str> = expected_clean.lines().collect();

    for (i, (actual, expected)) in actual_lines.iter().zip(expected_lines.iter()).enumerate() {
        if actual != expected {
            panic!(
                "Snapshot mismatch at line {}:\n  expected: {:?}\n  actual:   {:?}\n\nFull diff:\n{}",
                i + 1,
                expected,
                actual,
                create_diff(&expected_clean, &actual_clean)
            );
        }
    }

    // Check for length mismatch
    if actual_lines.len() != expected_lines.len() {
        panic!(
            "Snapshot line count mismatch: expected {} lines, got {} lines\n\nSnapshot: {}\n\nActual:\n{}",
            expected_lines.len(),
            actual_lines.len(),
            snapshot_name,
            actual_clean
        );
    }
}

/// Strip ANSI escape codes from string
fn strip_ansi_codes(s: &str) -> String {
    let re = regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    re.replace_all(s, "").to_string()
}

/// Create a simple diff for debugging
fn create_diff(expected: &str, actual: &str) -> String {
    let mut diff = String::new();
    let expected_lines: Vec<&str> = expected.lines().collect();
    let actual_lines: Vec<&str> = actual.lines().collect();

    let max_lines = expected_lines.len().max(actual_lines.len());

    for i in 0..max_lines {
        let exp = expected_lines.get(i).unwrap_or(&"<missing>");
        let act = actual_lines.get(i).unwrap_or(&"<missing>");

        if exp != act {
            diff.push_str(&format!("Line {}: DIFF\n", i + 1));
            diff.push_str(&format!("  - {}\n", exp));
            diff.push_str(&format!("  + {}\n", act));
        }
    }

    if diff.is_empty() {
        diff = "No differences found".to_string();
    }

    diff
}

// ============================================================================
// Hex Dump Pixel Tests
// ============================================================================

#[test]
fn test_pixel_hex_dump() {
    let output = apr()
        .args([
            "hex",
            test_apr_file().to_str().unwrap(),
            "--tensor",
            "encoder.layers.0",
            "--limit",
            "16",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_matches_snapshot(&stdout, "hex_dump.txt");
}

// ============================================================================
// Tree Pixel Tests
// ============================================================================

#[test]
fn test_pixel_tree_ascii() {
    let output = apr()
        .args(["tree", test_apr_file().to_str().unwrap(), "--sizes"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_matches_snapshot(&stdout, "tree_ascii.txt");
}

#[test]
fn test_pixel_tree_mermaid() {
    let output = apr()
        .args([
            "tree",
            test_apr_file().to_str().unwrap(),
            "--format",
            "mermaid",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_matches_snapshot(&stdout, "tree_mermaid.md");
}

// ============================================================================
// Flow Pixel Tests
// ============================================================================

#[test]
fn test_pixel_flow_cross_attn() {
    let output = apr()
        .args([
            "flow",
            test_apr_file().to_str().unwrap(),
            "--component",
            "cross_attn",
        ])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_matches_snapshot(&stdout, "flow_cross_attn.txt");
}

#[test]
fn test_pixel_flow_full() {
    let output = apr()
        .args(["flow", test_apr_file().to_str().unwrap()])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_matches_snapshot(&stdout, "flow_full.txt");
}

// ============================================================================
// Snapshot Update Mode (for regenerating golden files)
// ============================================================================

/// Run with UPDATE_SNAPSHOTS=1 to regenerate golden files
#[test]
#[ignore = "Run manually with UPDATE_SNAPSHOTS=1 to regenerate golden files"]
fn update_all_snapshots() {
    if std::env::var("UPDATE_SNAPSHOTS").is_err() {
        return;
    }

    // Hex dump
    let output = apr()
        .args([
            "hex",
            test_apr_file().to_str().unwrap(),
            "--tensor",
            "encoder.layers.0",
            "--limit",
            "16",
        ])
        .output()
        .expect("Failed");
    fs::write(
        snapshots_dir().join("hex_dump.txt"),
        String::from_utf8_lossy(&output.stdout).as_ref(),
    )
    .unwrap();

    // Tree ASCII
    let output = apr()
        .args(["tree", test_apr_file().to_str().unwrap(), "--sizes"])
        .output()
        .expect("Failed");
    fs::write(
        snapshots_dir().join("tree_ascii.txt"),
        String::from_utf8_lossy(&output.stdout).as_ref(),
    )
    .unwrap();

    // Tree Mermaid
    let output = apr()
        .args([
            "tree",
            test_apr_file().to_str().unwrap(),
            "--format",
            "mermaid",
        ])
        .output()
        .expect("Failed");
    fs::write(
        snapshots_dir().join("tree_mermaid.md"),
        String::from_utf8_lossy(&output.stdout).as_ref(),
    )
    .unwrap();

    // Flow cross_attn
    let output = apr()
        .args([
            "flow",
            test_apr_file().to_str().unwrap(),
            "--component",
            "cross_attn",
        ])
        .output()
        .expect("Failed");
    fs::write(
        snapshots_dir().join("flow_cross_attn.txt"),
        String::from_utf8_lossy(&output.stdout).as_ref(),
    )
    .unwrap();

    // Flow full
    let output = apr()
        .args(["flow", test_apr_file().to_str().unwrap()])
        .output()
        .expect("Failed");
    fs::write(
        snapshots_dir().join("flow_full.txt"),
        String::from_utf8_lossy(&output.stdout).as_ref(),
    )
    .unwrap();

    println!("All snapshots updated!");
}
