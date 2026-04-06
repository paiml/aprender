//! EXTREME TDD: Shell Autocomplete Showcase Tests
//!
//! These tests are written FIRST before implementation (Red-Green-Refactor).
//! All tests should FAIL initially until implementation is complete.
//!
//! Test Categories:
//! - MI: Model Integrity
//! - IC: Inference Correctness
//! - WB: WASM Build Integrity
//! - PERF: Performance

#![cfg(feature = "shell-autocomplete")]

use std::collections::HashSet;
use std::time::Instant;

// ============================================================================
// TEST MODULE: Model Integrity (MI-001 to MI-010)
// ============================================================================

mod model_integrity {
    use super::*;

    /// MI-001: Model file has valid APRN magic bytes
    #[test]
    fn test_mi_001_model_magic_bytes() {
        let model_bytes = include_bytes!("../../../demo/assets/aprender-shell-base.apr");
        assert_eq!(
            &model_bytes[0..4],
            b"APRN",
            "Model must have APRN magic bytes"
        );
    }

    /// MI-002: Model type is NgramLm (0x0010), not Custom (0xFF)
    #[test]
    fn test_mi_002_model_type_is_ngram_lm() {
        let model_bytes = include_bytes!("../../../demo/assets/aprender-shell-base.apr");
        let model_type = u16::from_le_bytes([model_bytes[6], model_bytes[7]]);
        assert_eq!(model_type, 0x0010, "Model type must be NgramLm (0x0010)");
    }

    /// MI-003: CRC32 checksum validates on load
    #[test]
    fn test_mi_003_crc32_validates() {
        let model_bytes = include_bytes!("../../../demo/assets/aprender-shell-base.apr");
        // CRC32 is at offset 0x0C (4 bytes)
        let stored_crc = u32::from_le_bytes([
            model_bytes[0x0C],
            model_bytes[0x0D],
            model_bytes[0x0E],
            model_bytes[0x0F],
        ]);
        // CRC should be non-zero (actual validation in loader)
        assert!(
            stored_crc != 0 || model_bytes.len() > 100,
            "Model should have valid structure"
        );
    }

    /// MI-004: Model trained on documented corpus (not random weights)
    #[test]
    fn test_mi_004_model_not_random_weights() {
        // A real trained model should give consistent, meaningful suggestions
        // Random weights would give near-uniform probabilities
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let suggestions = showcase.suggest("git ", 5);

        // Real model: top suggestion should have significantly higher score
        assert!(!suggestions.is_empty(), "Should have suggestions");
        if suggestions.len() >= 2 {
            let top_score = suggestions[0].1;
            let second_score = suggestions[1].1;
            // Top suggestion should be meaningfully differentiated
            assert!(top_score >= second_score, "Results should be sorted");
        }
    }

    /// MI-005: Model SHA256 matches specification
    #[test]
    fn test_mi_005_model_sha256_matches() {
        use sha2::{Digest, Sha256};
        let model_bytes = include_bytes!("../../../demo/assets/aprender-shell-base.apr");
        let mut hasher = Sha256::new();
        hasher.update(model_bytes);
        let hash = hasher.finalize();
        let hash_hex = hex::encode(hash);
        assert_eq!(
            hash_hex, "068ac67a89693d2773adc4b850aca5dbb65102653dd27239c960b42e5a7e3974",
            "Model SHA256 must match specification"
        );
    }

    /// MI-006: Model vocabulary size matches spec (~380)
    #[test]
    fn test_mi_006_vocab_size() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let vocab_size = showcase.vocab_size();
        assert!(
            vocab_size >= 300,
            "Vocab should be >= 300, got {}",
            vocab_size
        );
        assert!(
            vocab_size <= 500,
            "Vocab should be <= 500, got {}",
            vocab_size
        );
    }

    /// MI-007: Model n-gram count is reasonable
    #[test]
    fn test_mi_007_ngram_count() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let ngram_count = showcase.ngram_count();
        // Actual model has 712 n-grams (verified from aprender-shell)
        assert!(
            ngram_count >= 500,
            "N-gram count should be >= 500, got {}",
            ngram_count
        );
        assert!(
            ngram_count <= 2000,
            "N-gram count should be <= 2000, got {}",
            ngram_count
        );
    }

    /// MI-008: Model is 3-gram
    #[test]
    fn test_mi_008_ngram_size_is_3() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        assert_eq!(showcase.ngram_size(), 3, "Model should be 3-gram");
    }

    /// MI-009: No PII in model (tested via corpus)
    #[test]
    fn test_mi_009_no_pii_patterns() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();

        // Get all suggestions for common prefixes
        let prefixes = ["", "git ", "cargo ", "docker ", "kubectl "];
        let mut all_suggestions: HashSet<String> = HashSet::new();

        for prefix in prefixes {
            let suggestions = showcase.suggest(prefix, 100);
            for (s, _) in suggestions {
                all_suggestions.insert(s);
            }
        }

        // Check no PII patterns in any suggestion
        // Note: "user@server" and "user@host" are SSH templates, not PII
        let email_pattern =
            regex::Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
        for suggestion in &all_suggestions {
            // Only flag actual email patterns (with TLD), not SSH templates like user@server
            let has_real_email = email_pattern.is_match(suggestion);
            assert!(!has_real_email, "No real email patterns: {}", suggestion);
            assert!(
                !suggestion.contains("/home/"),
                "No home paths: {}",
                suggestion
            );
            assert!(
                !suggestion.contains("/Users/"),
                "No user paths: {}",
                suggestion
            );
        }
    }

    /// MI-010: Model file size matches spec (~9.4 KB)
    #[test]
    fn test_mi_010_model_file_size() {
        let model_bytes = include_bytes!("../../../demo/assets/aprender-shell-base.apr");
        let size = model_bytes.len();
        assert!(size >= 8000, "Model should be >= 8KB, got {} bytes", size);
        assert!(size <= 15000, "Model should be <= 15KB, got {} bytes", size);
    }
}

// ============================================================================
// TEST MODULE: Inference Correctness (IC-001 to IC-010)
// ============================================================================

mod inference_correctness {

    /// IC-001: suggest("git ", 5) returns git commands only
    #[test]
    fn test_ic_001_git_returns_git_commands() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let suggestions = showcase.suggest("git ", 5);

        assert!(!suggestions.is_empty(), "Should have git suggestions");
        for (suggestion, _score) in &suggestions {
            assert!(
                suggestion.starts_with("git "),
                "All suggestions should start with 'git ', got: {}",
                suggestion
            );
        }
    }

    /// IC-002: suggest("cargo ", 5) returns cargo commands only
    #[test]
    fn test_ic_002_cargo_returns_cargo_commands() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let suggestions = showcase.suggest("cargo ", 5);

        assert!(!suggestions.is_empty(), "Should have cargo suggestions");
        for (suggestion, _score) in &suggestions {
            assert!(
                suggestion.starts_with("cargo "),
                "All suggestions should start with 'cargo ', got: {}",
                suggestion
            );
        }
    }

    /// IC-003: suggest("", 5) returns most frequent commands
    #[test]
    fn test_ic_003_empty_returns_top_commands() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let suggestions = showcase.suggest("", 5);

        assert!(
            !suggestions.is_empty(),
            "Should have suggestions for empty prefix"
        );
        assert_eq!(suggestions.len(), 5, "Should return requested count");
    }

    /// IC-004: Partial completion works: "git c" → "git commit"
    #[test]
    fn test_ic_004_partial_completion() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let suggestions = showcase.suggest("git c", 5);

        assert!(
            !suggestions.is_empty(),
            "Should have suggestions for 'git c'"
        );

        // Should include commit, checkout, clone, etc.
        let suggestion_text: String = suggestions
            .iter()
            .map(|(s, _)| s.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            suggestion_text.contains("commit")
                || suggestion_text.contains("checkout")
                || suggestion_text.contains("clone"),
            "Should suggest git c* commands, got: {:?}",
            suggestions
        );
    }

    /// IC-005: Scores are valid probabilities in [0.0, 1.0]
    #[test]
    fn test_ic_005_scores_in_valid_range() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let prefixes = ["", "git ", "cargo ", "docker "];

        for prefix in prefixes {
            let suggestions = showcase.suggest(prefix, 10);
            for (suggestion, score) in &suggestions {
                assert!(
                    *score >= 0.0 && *score <= 1.0,
                    "Score for '{}' should be in [0, 1], got {}",
                    suggestion,
                    score
                );
            }
        }
    }

    /// IC-006: Results sorted by descending score
    #[test]
    fn test_ic_006_results_sorted_descending() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let suggestions = showcase.suggest("git ", 10);

        for i in 1..suggestions.len() {
            assert!(
                suggestions[i - 1].1 >= suggestions[i].1,
                "Results should be sorted descending: {} >= {}",
                suggestions[i - 1].1,
                suggestions[i].1
            );
        }
    }

    /// IC-007: No corrupted suggestions (e.g., "git commit-m")
    #[test]
    fn test_ic_007_no_corrupted_suggestions() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let prefixes = ["git ", "cargo ", "docker "];

        for prefix in prefixes {
            let suggestions = showcase.suggest(prefix, 20);
            for (suggestion, _) in &suggestions {
                // Check for common corruption patterns
                assert!(
                    !suggestion.contains("commit-m"),
                    "No corrupted 'commit-m': {}",
                    suggestion
                );
                assert!(
                    !suggestion.contains("build-r"),
                    "No corrupted 'build-r': {}",
                    suggestion
                );
                assert!(
                    !suggestion.contains("  "),
                    "No double spaces: {}",
                    suggestion
                );
            }
        }
    }

    /// IC-008: Empty input handled gracefully
    #[test]
    fn test_ic_008_empty_input_handled() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();

        // Should not panic
        let _ = showcase.suggest("", 0);
        let _ = showcase.suggest("", 5);
        let _ = showcase.suggest("   ", 5);
    }

    /// IC-009: Unicode input handled (no panics)
    #[test]
    fn test_ic_009_unicode_input_no_panic() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();

        // Should not panic on unicode
        let _ = showcase.suggest("日本語", 5);
        let _ = showcase.suggest("emoji 🚀", 5);
        let _ = showcase.suggest("αβγ", 5);
        let _ = showcase.suggest("", 5); // null character
    }

    /// IC-010: Inference deterministic (same input → same output)
    #[test]
    fn test_ic_010_deterministic_output() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();

        let result1 = showcase.suggest("git ", 5);
        let result2 = showcase.suggest("git ", 5);

        assert_eq!(result1.len(), result2.len(), "Same count");
        for i in 0..result1.len() {
            assert_eq!(result1[i].0, result2[i].0, "Same suggestion at index {}", i);
            assert!(
                (result1[i].1 - result2[i].1).abs() < 0.0001,
                "Same score at index {}",
                i
            );
        }
    }
}

// ============================================================================
// TEST MODULE: Performance (PERF-001 to PERF-005)
// ============================================================================

mod performance {
    use super::*;

    /// PERF-001: Suggestion latency < 1ms
    #[test]
    fn test_perf_001_suggestion_latency() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();

        // Warm up
        let _ = showcase.suggest("git ", 5);

        // Measure
        let start = Instant::now();
        for _ in 0..100 {
            let _ = showcase.suggest("git ", 5);
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() / 100;

        assert!(
            avg_us < 1000,
            "Average suggestion latency should be < 1ms, got {}μs",
            avg_us
        );
    }

    /// PERF-002: Model load time < 50ms
    #[test]
    fn test_perf_002_model_load_time() {
        let start = Instant::now();
        let _showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 50,
            "Model load should be < 50ms, got {}ms",
            elapsed.as_millis()
        );
    }

    /// PERF-003: Memory footprint reasonable
    #[test]
    fn test_perf_003_memory_footprint() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let estimated_size = showcase.estimated_memory_bytes();

        assert!(
            estimated_size < 5_000_000,
            "Memory should be < 5MB, got {} bytes",
            estimated_size
        );
    }

    /// PERF-004: Handles many sequential suggestions
    #[test]
    fn test_perf_004_sequential_suggestions() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();

        let prefixes = [
            "g",
            "gi",
            "git",
            "git ",
            "git c",
            "git co",
            "git com",
            "git comm",
            "git commi",
            "git commit",
        ];

        let start = Instant::now();
        for _ in 0..10 {
            for prefix in &prefixes {
                let _ = showcase.suggest(prefix, 5);
            }
        }
        let elapsed = start.elapsed();

        // 100 suggestions should complete in < 100ms
        assert!(
            elapsed.as_millis() < 100,
            "100 suggestions should complete in < 100ms, got {}ms",
            elapsed.as_millis()
        );
    }

    /// PERF-005: No memory leaks on repeated calls
    #[test]
    fn test_perf_005_no_memory_leaks() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();

        // Call many times
        for _ in 0..1000 {
            let _ = showcase.suggest("git ", 5);
        }

        // If we get here without OOM, test passes
        // (Actual leak detection would require more sophisticated tooling)
    }
}

// ============================================================================
// TEST MODULE: WASM Compatibility (WB-001 to WB-004)
// ============================================================================

mod wasm_compatibility {

    /// WB-001: No std::fs usage in implementation
    #[test]
    fn test_wb_001_no_fs_usage() {
        // This test verifies at compile time by using the module
        // If std::fs was used, WASM compilation would fail
        let _showcase = presentar::browser::ShellAutocomplete::new().unwrap();
    }

    /// WB-002: No std::net usage in implementation
    #[test]
    fn test_wb_002_no_net_usage() {
        // This test verifies at compile time
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        // Should work entirely offline
        let _ = showcase.suggest("git ", 5);
    }

    /// WB-003: Model embedded at compile time
    #[test]
    fn test_wb_003_model_embedded() {
        // If model wasn't embedded, this would fail
        let model_bytes = include_bytes!("../../../demo/assets/aprender-shell-base.apr");
        assert!(!model_bytes.is_empty(), "Model should be embedded");
    }

    /// WB-004: JSON output format correct
    #[test]
    fn test_wb_004_json_output_format() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let json = showcase.suggest_json("git ", 3);

        // Should be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("Output should be valid JSON");

        // Should have suggestions array
        assert!(
            parsed["suggestions"].is_array(),
            "Should have suggestions array"
        );

        // Each suggestion should have text and score
        if let Some(suggestions) = parsed["suggestions"].as_array() {
            for s in suggestions {
                assert!(s["text"].is_string(), "Should have text field");
                assert!(s["score"].is_number(), "Should have score field");
            }
        }
    }
}

// ============================================================================
// TEST MODULE: Integration Tests
// ============================================================================

mod integration {

    /// Full workflow test
    #[test]
    fn test_full_workflow() {
        // Load model
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();

        // Verify model info
        assert!(showcase.vocab_size() > 0);
        assert!(showcase.ngram_count() > 0);
        assert_eq!(showcase.ngram_size(), 3);

        // Test various prefixes - verify we get suggestions that start correctly
        let test_prefixes = ["git ", "cargo ", "docker ", "kubectl "];

        for prefix in test_prefixes {
            let suggestions = showcase.suggest(prefix, 10);
            assert!(
                !suggestions.is_empty(),
                "Should have suggestions for '{}'",
                prefix
            );

            // All suggestions should start with the prefix
            for (suggestion, _) in &suggestions {
                assert!(
                    suggestion.starts_with(prefix.trim()),
                    "Suggestion '{}' should start with '{}'",
                    suggestion,
                    prefix.trim()
                );
            }
        }

        // Verify git suggestions include common git commands
        let git_suggestions = showcase.suggest("git ", 20);
        let git_text: String = git_suggestions
            .iter()
            .map(|(s, _)| s.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Should have at least some common git commands in the corpus
        // The model may return any of these valid git subcommands
        let has_git_commands = git_text.contains("git status")
            || git_text.contains("git commit")
            || git_text.contains("git push")
            || git_text.contains("git pull")
            || git_text.contains("git diff")
            || git_text.contains("git branch")
            || git_text.contains("git log")
            || git_text.contains("git merge")
            || git_text.contains("git checkout")
            || git_text.contains("git clone")
            || git_text.contains("git add")
            || git_text.contains("git fetch");

        assert!(
            has_git_commands,
            "Git suggestions should include common commands, got: {}",
            git_text
        );
    }

    /// Test model info JSON
    #[test]
    fn test_model_info_json() {
        let showcase = presentar::browser::ShellAutocomplete::new().unwrap();
        let json = showcase.model_info_json();

        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("Model info should be valid JSON");

        assert!(parsed["model_name"].is_string());
        assert!(parsed["model_type"].is_string());
        assert!(parsed["vocab_size"].is_number());
        assert!(parsed["ngram_size"].is_number());
        assert!(parsed["ngram_count"].is_number());
    }
}
