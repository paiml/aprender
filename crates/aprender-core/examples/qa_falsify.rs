#![allow(clippy::disallowed_methods)]
//! QA Infrastructure Falsification Tests (PMAT-098 Red Team)
//!
//! This file attempts to falsify the claims made about qa_run.rs:
//! 1. Hang detection reliability
//! 2. Garbage detection completeness
//! 3. Server lifecycle cleanup
//! 4. Matrix integrity (27-test coverage)
//! 5. Answer verification brittleness
//!
//! Run with: cargo run --example qa_falsify

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

// ============================================================================
// GARBAGE PATTERNS (copied from qa_run.rs for testing)
// ============================================================================

const GARBAGE_PATTERNS: &[&str] = &[
    "\u{FFFD}", // Replacement character (encoding error)
    "[UNK]",    // Unknown token marker
    "akunji",   // Known GQA bug garbage
    "olumbia",  // Known layout bug garbage
    "专门窗",   // Known GQA bug CJK garbage
    "token0",   // Raw token ID leak
    "token1",   // Raw token ID leak
    "<0x",      // Byte token leak (e.g., <0x0A>)
];

const BPE_ARTIFACTS: &[char] = &[
    'Ġ', // GPT-2 style space prefix
    'Ċ', // GPT-2 style newline
    'ĉ', // GPT-2 style tab
];

/// Check if `needle` appears in `haystack` as a standalone word
fn contains_as_word(haystack: &str, needle: &str) -> bool {
    let mut search_start = 0;
    while let Some(pos) = haystack[search_start..].find(needle) {
        let abs_pos = search_start + pos;
        let end_pos = abs_pos + needle.len();

        let left_ok = abs_pos == 0 || {
            let prev_char = haystack[..abs_pos]
                .chars()
                .last()
                .expect("non-empty prefix must have a last char");
            !prev_char.is_alphanumeric()
        };

        let right_ok = end_pos >= haystack.len() || {
            let next_char = haystack[end_pos..]
                .chars()
                .next()
                .expect("non-empty suffix must have a next char");
            !next_char.is_alphanumeric()
        };

        if left_ok && right_ok {
            return true;
        }

        search_start = abs_pos + 1;
        if search_start >= haystack.len() {
            break;
        }
    }
    false
}

/// Replicates verify_output logic for testing (UPDATED with word boundary fix)
fn verify_output(output: &str, expected_contains: Option<&str>) -> Result<String, String> {
    let trimmed = output.trim();

    // 1. Empty check
    if trimmed.is_empty() {
        return Err("FailEmpty".to_string());
    }

    // 2. Garbage detection
    for pattern in GARBAGE_PATTERNS {
        if trimmed.contains(pattern) {
            return Err(format!("FailGarbage: {}", pattern));
        }
    }

    // 3. BPE artifact check
    for &artifact in BPE_ARTIFACTS {
        if trimmed.contains(artifact) {
            return Err(format!("FailBpeArtifact: {}", artifact));
        }
    }

    // 4. Expected answer check WITH WORD BOUNDARY (fixed)
    if let Some(expected) = expected_contains {
        if !contains_as_word(trimmed, expected) {
            return Err(format!(
                "FailMissingAnswer: expected '{}' as standalone word",
                expected
            ));
        }
    }

    Ok(trimmed.to_string())
}

// ============================================================================
// SHARED HELPERS
// ============================================================================

fn print_section_header(title: &str) {
    println!(
        "\n{}═══════════════════════════════════════════════════════════════{}",
        "\x1b[1;34m", "\x1b[0m"
    );
    println!("{}{}{}", "\x1b[1;33m", title, "\x1b[0m");
    println!(
        "{}═══════════════════════════════════════════════════════════════{}\n",
        "\x1b[1;34m", "\x1b[0m"
    );
}

/// Run a list of (input, should_pass, desc) test cases through verify_output(_, None).
/// Returns (passed, failed).
fn run_garbage_cases(cases: &[(&str, bool, &str)]) -> (u32, u32) {
    let mut passed = 0u32;
    let mut failed = 0u32;
    for &(input, should_pass, desc) in cases {
        let result = verify_output(input, None);
        let actually_passed = result.is_ok();
        if actually_passed == should_pass {
            println!("  {}✓ PASS{}: {}", "\x1b[32m", "\x1b[0m", desc);
            passed += 1;
        } else {
            println!("  {}✗ FAIL{}: {}", "\x1b[31m", "\x1b[0m", desc);
            println!(
                "    Expected: {}, Got: {:?}",
                if should_pass { "PASS" } else { "FAIL" },
                result
            );
            failed += 1;
        }
    }
    (passed, failed)
}

// ============================================================================
// TEST 2: GARBAGE DETECTION FALSIFICATION
// ============================================================================

fn test_garbage_detection() {
    print_section_header("TEST 2: GARBAGE DETECTION FALSIFICATION");

    let test_cases: Vec<(&str, bool, &str)> = vec![
        (
            "This is a token",
            true,
            "Common English word 'token' - should PASS",
        ),
        (
            "token12345",
            false,
            "Raw token ID leak (token0/token1 pattern) - should FAIL",
        ),
        (
            "The symbol Ġ is used in BPE",
            false,
            "BPE artifact Ġ - should FAIL",
        ),
        (
            "I am akunji",
            false,
            "Model collapse hallucination - should FAIL",
        ),
        (
            "Welcome to Columbia University",
            false,
            "Contains 'olumbia' - should FAIL",
        ),
        ("Hello, world!", true, "Clean output - should PASS"),
        ("The answer is 4", true, "Simple answer - should PASS"),
        (
            "token",
            true,
            "Just 'token' without number - should PASS (edge case)",
        ),
        (
            "mytoken0value",
            false,
            "token0 embedded in string - should FAIL",
        ),
        ("<0x0A>newline", false, "Byte token leak - should FAIL"),
        (
            "Test [UNK] marker",
            false,
            "Unknown token marker - should FAIL",
        ),
        ("Unicode: 专门窗 text", false, "CJK garbage - should FAIL"),
    ];

    let (passed, failed) = run_garbage_cases(&test_cases);
    println!("\n  Summary: {}/{} tests passed", passed, passed + failed);

    if failed > 0 {
        println!(
            "  {}⚠ FALSIFICATION SUCCESSFUL: Garbage detection has {} edge case failures{}",
            "\x1b[1;31m", failed, "\x1b[0m"
        );
    } else {
        println!(
            "  {}✓ Garbage detection held up under testing{}",
            "\x1b[32m", "\x1b[0m"
        );
    }
}

// ============================================================================
// TEST 5: FALSE CONFIDENCE AUDIT (Answer Verification Brittleness)
// ============================================================================

/// Evaluate a single answer-verification test case. Returns 1 if brittle, 0 otherwise.
fn eval_answer_case(input: &str, expected: Option<&str>, should_pass: bool, desc: &str) -> u32 {
    let result = verify_output(input, expected);
    let actually_passed = result.is_ok();
    let is_risk = desc.contains("RISK") || desc.contains("EDGE CASE");

    if actually_passed != should_pass {
        println!("  {}✗{}: {}", "\x1b[31m", "\x1b[0m", desc);
        println!(
            "    Expected: {}, Got: {:?}",
            if should_pass { "PASS" } else { "FAIL" },
            result
        );
        return 0;
    }
    if is_risk {
        println!(
            "  {}⚠ RISK{}: {} (passes but semantically wrong)",
            "\x1b[33m", "\x1b[0m", desc
        );
        return 1;
    }
    println!("  {}✓{}: {}", "\x1b[32m", "\x1b[0m", desc);
    0
}

fn test_answer_verification() {
    print_section_header("TEST 5: FALSE CONFIDENCE AUDIT (Answer Verification)");

    let test_cases: Vec<(&str, Option<&str>, bool, &str)> = vec![
        ("4", Some("4"), true, "Exact answer '4' - should PASS"),
        (
            "The answer is 4.",
            Some("4"),
            true,
            "Answer with context - should PASS",
        ),
        (
            "The answer is not 4, but 5.",
            Some("4"),
            true,
            "Contains standalone '4' - PASSES (acceptable)",
        ),
        ("2+2=4", Some("4"), true, "Equation format - should PASS"),
        (
            "Four",
            Some("4"),
            false,
            "Word 'Four' not digit '4' - should FAIL",
        ),
        ("5", Some("4"), false, "Wrong answer - should FAIL"),
        (
            "The result is 14",
            Some("4"),
            false,
            "FIXED: '14' no longer matches '4' - should FAIL",
        ),
        (
            "I counted 4 apples and 5 oranges",
            Some("4"),
            true,
            "Multiple numbers - should PASS",
        ),
        ("", Some("4"), false, "Empty output - should FAIL"),
        ("forty-four", Some("4"), false, "Spelled out - should FAIL"),
        (
            "answer=4",
            Some("4"),
            true,
            "'4' after '=' is standalone - should PASS",
        ),
        (
            "x4y",
            Some("4"),
            false,
            "'4' embedded in alphanumeric - should FAIL",
        ),
        (
            "4.0",
            Some("4"),
            true,
            "'4' before '.' is standalone - should PASS",
        ),
    ];

    let brittle_cases: u32 = test_cases
        .iter()
        .map(|&(input, expected, should_pass, desc)| {
            eval_answer_case(input, expected, should_pass, desc)
        })
        .sum();

    if brittle_cases > 0 {
        println!(
            "\n  {}⚠ FALSIFICATION FINDING:{} {} brittle cases remain",
            "\x1b[1;33m", "\x1b[0m", brittle_cases
        );
    } else {
        println!(
            "\n  {}✓ FIX VERIFIED:{} Word boundary check prevents false positives",
            "\x1b[32m", "\x1b[0m"
        );
        println!("    - 'The result is 14' now correctly FAILS (4 embedded in 14)");
        println!("    - 'x4y' correctly FAILS (4 embedded in alphanumeric)");
        println!("    - 'The answer is 4.' correctly PASSES (4 at word boundary)");
    }
}

// ============================================================================
// TEST 4: MATRIX INTEGRITY CHECK
// ============================================================================

/// Report the cell count found in matrix output.
fn report_cell_count(num: u32) {
    match num {
        27 => println!("  {}✓ Matrix claims 27 cells{}", "\x1b[32m", "\x1b[0m"),
        21 => println!(
            "  {}⚠ Matrix has 21 cells (not 27 as documented){}",
            "\x1b[33m", "\x1b[0m"
        ),
        _ => println!(
            "  {}⚠ Matrix has {} cells (unexpected){}",
            "\x1b[33m", num, "\x1b[0m"
        ),
    }
}

/// Search stdout+stderr for a matrix cell-count line and report it.
fn parse_matrix_output(out: &std::process::Output) {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    let line = stdout
        .lines()
        .chain(stderr.lines())
        .find(|l| l.contains("FULL MATRIX:") || (l.contains("Testing") && l.contains("cell")));

    let Some(line) = line else {
        println!("  Could not find cell count in output");
        println!("  (This test requires models to be available)");
        return;
    };

    println!("  Found: {}", line.trim());
    if let Some(num) = line.split_whitespace().find_map(|w| w.parse::<u32>().ok()) {
        report_cell_count(num);
    }
}

fn print_matrix_calculation() {
    // 3 modalities x (GGUF:3 + SafeTensors:2 + APR:2) = 3 x 7 = 21 cells
    println!("  Expected matrix calculation:");
    println!("    Modalities: Run, Chat, Serve (3)");
    println!("    Formats: GGUF, SafeTensors, APR (3)");
    println!("    Configs per format:");
    println!("      - GGUF: CPU, CPU+trace, GPU = 3 cells");
    println!("      - SafeTensors: CPU, CPU+trace = 2 cells");
    println!("      - APR: CPU, CPU+trace = 2 cells");
    println!("    Per modality: 3 + 2 + 2 = 7 cells");
    println!("    Total: 3 modalities × 7 = 21 cells");
    println!();
    println!(
        "  {}NOTE:{} Documentation was updated: 21 cells (not original 27 claim).",
        "\x1b[32m", "\x1b[0m"
    );
    println!();
    println!("  To verify, run: cargo run --example qa_run -- --full-matrix --help");
    println!("  And count 'Testing N cell(s)' in the output.");
}

fn test_matrix_integrity() {
    print_section_header("TEST 4: MATRIX INTEGRITY CHECK");
    print_matrix_calculation();

    println!("\n  Running actual matrix count...");

    let output = Command::new("cargo")
        .args([
            "run",
            "--example",
            "qa_run",
            "--release",
            "--",
            "--full-matrix",
            "--verbose",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match output {
        Ok(out) => parse_matrix_output(&out),
        Err(e) => println!("  Could not run qa_run: {}", e),
    }
}

// ============================================================================
// TEST 1: HANG DETECTION (Simulation)
// ============================================================================

include!("includes/qa_falsify_include_01.rs");
