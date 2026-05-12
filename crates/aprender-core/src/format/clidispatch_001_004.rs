// `cli-dispatch-v1` algorithm-level PARTIAL discharge for the 4 CLI-
// dispatch falsifiers (dispatch completeness, exit-code injectivity,
// JSON output parseability, inspection idempotency).
//
// Contract: `contracts/cli-dispatch-v1.yaml`.
// Refs: POSIX.1-2017 Utility Conventions, GNU Coding Standards (Exit
// Status), apr-cli/src/error.rs::CliError.
//
// ## Disambiguation
//
// `apr-cli-commands-v1.yaml` (task #278) is a sibling contract covering
// 4 different FALSIFY-CLI-001..004 gates on command registration. This
// contract — cli-dispatch-v1 — covers dispatch correctness, exit codes,
// and output format. Module suffix `clidispatch_` disambiguates from
// any `cli_*` registration modules.

use std::collections::HashSet;

/// Canonical exit codes per `cli_config.exit_codes`.
pub const AC_CLIDISPATCH_EXIT_CODES: [(&str, u8); 11] = [
    ("success", 0),
    ("general_error", 1),
    ("file_not_found", 3),
    ("invalid_format", 4),
    ("validation_failed", 5),
    ("model_load_failed", 6),
    ("io_error", 7),
    ("inference_failed", 8),
    ("feature_disabled", 9),
    ("network_error", 10),
    ("http_not_found", 11),
];

/// Output format identifiers per `cli_config.output_formats`.
pub const AC_CLIDISPATCH_OUTPUT_FORMATS: [&str; 6] =
    ["text", "json", "yaml", "csv", "srt", "vtt"];

/// Read-only inspection commands per equation `idempotent_inspection`.
pub const AC_CLIDISPATCH_INSPECTION_CMDS: [&str; 7] =
    ["check", "inspect", "debug", "validate", "lint", "explain", "list"];

// =============================================================================
// FALSIFY-CLI-001 — every Commands variant has a dispatch handler
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchCompletenessVerdict {
    /// Every subcommand in the registry has a dispatch handler that
    /// returned exit 0 on `--help` (clap-recognized).
    Pass,
    /// At least one subcommand failed `--help` (no handler).
    Fail,
}

/// `(subcommand_name, --help_exit_code)` per registered subcommand.
#[must_use]
pub fn verdict_from_dispatch_completeness(help_results: &[(&str, i32)]) -> DispatchCompletenessVerdict {
    if help_results.is_empty() {
        return DispatchCompletenessVerdict::Fail;
    }
    for (_cmd, exit) in help_results {
        if *exit != 0 {
            return DispatchCompletenessVerdict::Fail;
        }
    }
    DispatchCompletenessVerdict::Pass
}

// =============================================================================
// FALSIFY-CLI-002 — exit codes are injective (no collisions)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitCodeInjectivityVerdict {
    /// Distinct CliError variants map to distinct exit codes.
    Pass,
    /// Two error variants share the same code.
    Fail,
}

#[must_use]
pub fn verdict_from_exit_code_injectivity(error_code_pairs: &[(&str, u8)]) -> ExitCodeInjectivityVerdict {
    if error_code_pairs.is_empty() {
        return ExitCodeInjectivityVerdict::Fail;
    }
    let mut seen: HashSet<u8> = HashSet::new();
    for (_name, code) in error_code_pairs {
        if !seen.insert(*code) {
            return ExitCodeInjectivityVerdict::Fail;
        }
    }
    ExitCodeInjectivityVerdict::Pass
}

// =============================================================================
// FALSIFY-CLI-003 — JSON output is always parseable
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonParseableVerdict {
    /// `--json` output starts with valid JSON sentinel ('{' or '['),
    /// has balanced braces/brackets, and contains no obvious malformed
    /// markers.
    Pass,
    /// Output not JSON-shaped — unescaped string, trailing comma, etc.
    Fail,
}

#[must_use]
pub fn verdict_from_json_parseable(output: &str) -> JsonParseableVerdict {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return JsonParseableVerdict::Fail;
    }
    // Safety: `trimmed.is_empty()` was checked above, so both .next() and .last() yield Some.
    let first = trimmed.chars().next().expect("non-empty trimmed");
    let last = trimmed.chars().last().expect("non-empty trimmed");
    let valid_start = first == '{' || first == '[';
    let valid_end = last == '}' || last == ']';
    if !valid_start || !valid_end {
        return JsonParseableVerdict::Fail;
    }
    // Match braces and brackets pairwise — outer wrapper only.
    if first == '{' && last != '}' {
        return JsonParseableVerdict::Fail;
    }
    if first == '[' && last != ']' {
        return JsonParseableVerdict::Fail;
    }
    // Catch obvious trailing-comma regression: ",}" or ",]" sequence.
    if trimmed.contains(",}") || trimmed.contains(",]") {
        return JsonParseableVerdict::Fail;
    }
    JsonParseableVerdict::Pass
}

// =============================================================================
// FALSIFY-CLI-004 — inspection commands are idempotent
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectionIdempotencyVerdict {
    /// Two runs of an inspection command produce identical stdout AND exit code.
    Pass,
    /// Diverged — hidden state mutation or non-determinism.
    Fail,
}

#[must_use]
pub fn verdict_from_inspection_idempotency(
    stdout_a: &[u8],
    stdout_b: &[u8],
    exit_code_a: i32,
    exit_code_b: i32,
) -> InspectionIdempotencyVerdict {
    if stdout_a.is_empty() && stdout_b.is_empty() {
        return InspectionIdempotencyVerdict::Fail;
    }
    if stdout_a != stdout_b {
        return InspectionIdempotencyVerdict::Fail;
    }
    if exit_code_a != exit_code_b {
        return InspectionIdempotencyVerdict::Fail;
    }
    InspectionIdempotencyVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_exit_codes_are_injective_in_constants() {
        let mut seen: HashSet<u8> = HashSet::new();
        for (_, code) in AC_CLIDISPATCH_EXIT_CODES {
            assert!(seen.insert(code), "exit code {code} duplicated in constants");
        }
    }

    #[test]
    fn provenance_success_is_zero() {
        assert_eq!(AC_CLIDISPATCH_EXIT_CODES[0].1, 0);
    }

    #[test]
    fn provenance_file_not_found_is_3() {
        let entry = AC_CLIDISPATCH_EXIT_CODES.iter().find(|(n, _)| *n == "file_not_found").unwrap();
        assert_eq!(entry.1, 3);
    }

    #[test]
    fn provenance_output_formats_count_6() {
        assert_eq!(AC_CLIDISPATCH_OUTPUT_FORMATS.len(), 6);
    }

    #[test]
    fn provenance_inspection_cmds_count_7() {
        assert_eq!(AC_CLIDISPATCH_INSPECTION_CMDS.len(), 7);
    }

    // -------------------------------------------------------------------------
    // Section 2: CLI-001 dispatch completeness.
    // -------------------------------------------------------------------------
    #[test]
    fn fcli001_pass_all_dispatched() {
        let r = [
            ("check", 0), ("run", 0), ("inspect", 0), ("debug", 0),
            ("validate", 0), ("lint", 0), ("explain", 0),
        ];
        assert_eq!(
            verdict_from_dispatch_completeness(&r),
            DispatchCompletenessVerdict::Pass
        );
    }

    #[test]
    fn fcli001_fail_one_command_failed() {
        let r = [("check", 0), ("run", 1)];
        assert_eq!(
            verdict_from_dispatch_completeness(&r),
            DispatchCompletenessVerdict::Fail
        );
    }

    #[test]
    fn fcli001_fail_empty() {
        let r: [(&str, i32); 0] = [];
        assert_eq!(
            verdict_from_dispatch_completeness(&r),
            DispatchCompletenessVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: CLI-002 exit-code injectivity.
    // -------------------------------------------------------------------------
    #[test]
    fn fcli002_pass_all_distinct() {
        let pairs = [("Ok", 0u8), ("Generic", 1), ("FileNotFound", 3)];
        assert_eq!(
            verdict_from_exit_code_injectivity(&pairs),
            ExitCodeInjectivityVerdict::Pass
        );
    }

    #[test]
    fn fcli002_pass_canonical_constants() {
        let v: Vec<(&str, u8)> = AC_CLIDISPATCH_EXIT_CODES.to_vec();
        assert_eq!(
            verdict_from_exit_code_injectivity(&v),
            ExitCodeInjectivityVerdict::Pass
        );
    }

    #[test]
    fn fcli002_fail_collision() {
        let pairs = [("FileNotFound", 3u8), ("InvalidFormat", 3)];
        assert_eq!(
            verdict_from_exit_code_injectivity(&pairs),
            ExitCodeInjectivityVerdict::Fail
        );
    }

    #[test]
    fn fcli002_fail_empty() {
        let pairs: [(&str, u8); 0] = [];
        assert_eq!(
            verdict_from_exit_code_injectivity(&pairs),
            ExitCodeInjectivityVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: CLI-003 JSON parseability.
    // -------------------------------------------------------------------------
    #[test]
    fn fcli003_pass_object() {
        assert_eq!(
            verdict_from_json_parseable(r#"{"key": "value"}"#),
            JsonParseableVerdict::Pass
        );
    }

    #[test]
    fn fcli003_pass_array() {
        assert_eq!(
            verdict_from_json_parseable("[1, 2, 3]"),
            JsonParseableVerdict::Pass
        );
    }

    #[test]
    fn fcli003_pass_object_with_whitespace() {
        assert_eq!(
            verdict_from_json_parseable("\n  {\"k\": 1}  \n"),
            JsonParseableVerdict::Pass
        );
    }

    #[test]
    fn fcli003_fail_empty() {
        assert_eq!(verdict_from_json_parseable(""), JsonParseableVerdict::Fail);
    }

    #[test]
    fn fcli003_fail_plain_text() {
        assert_eq!(
            verdict_from_json_parseable("not json"),
            JsonParseableVerdict::Fail
        );
    }

    #[test]
    fn fcli003_fail_unmatched_brace() {
        assert_eq!(
            verdict_from_json_parseable("{not closed"),
            JsonParseableVerdict::Fail
        );
    }

    #[test]
    fn fcli003_fail_trailing_comma_object() {
        assert_eq!(
            verdict_from_json_parseable(r#"{"k": 1,}"#),
            JsonParseableVerdict::Fail
        );
    }

    #[test]
    fn fcli003_fail_trailing_comma_array() {
        assert_eq!(
            verdict_from_json_parseable("[1, 2, 3,]"),
            JsonParseableVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: CLI-004 inspection idempotency.
    // -------------------------------------------------------------------------
    #[test]
    fn fcli004_pass_identical_runs() {
        let stdout = b"model OK\n";
        assert_eq!(
            verdict_from_inspection_idempotency(stdout, stdout, 0, 0),
            InspectionIdempotencyVerdict::Pass
        );
    }

    #[test]
    fn fcli004_fail_stdout_diverges() {
        let a = b"version 1\n";
        let b = b"version 2\n";
        assert_eq!(
            verdict_from_inspection_idempotency(a, b, 0, 0),
            InspectionIdempotencyVerdict::Fail
        );
    }

    #[test]
    fn fcli004_fail_exit_diverges() {
        let stdout = b"output\n";
        assert_eq!(
            verdict_from_inspection_idempotency(stdout, stdout, 0, 1),
            InspectionIdempotencyVerdict::Fail
        );
    }

    #[test]
    fn fcli004_fail_both_empty() {
        // No output captured = harness defect.
        assert_eq!(
            verdict_from_inspection_idempotency(b"", b"", 0, 0),
            InspectionIdempotencyVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — full healthy CLI passes all 4.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_cli_passes_all_4() {
        // 001
        let cmds = [
            ("check", 0), ("run", 0), ("serve", 0), ("inspect", 0),
        ];
        assert_eq!(
            verdict_from_dispatch_completeness(&cmds),
            DispatchCompletenessVerdict::Pass
        );
        // 002 (canonical exit-code table).
        let pairs: Vec<(&str, u8)> = AC_CLIDISPATCH_EXIT_CODES.to_vec();
        assert_eq!(
            verdict_from_exit_code_injectivity(&pairs),
            ExitCodeInjectivityVerdict::Pass
        );
        // 003
        let json = r#"{"model": "qwen2.5", "tensors": 339}"#;
        assert_eq!(verdict_from_json_parseable(json), JsonParseableVerdict::Pass);
        // 004
        let stdout = b"validation OK\n";
        assert_eq!(
            verdict_from_inspection_idempotency(stdout, stdout, 0, 0),
            InspectionIdempotencyVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_4_failures() {
        // 001: subcommand not wired.
        let bad_cmds = [("rm", 127)];
        assert_eq!(
            verdict_from_dispatch_completeness(&bad_cmds),
            DispatchCompletenessVerdict::Fail
        );
        // 002: typo gave two errors the same code.
        let bad_pairs = [("FileNotFound", 3u8), ("InvalidFormat", 3)];
        assert_eq!(
            verdict_from_exit_code_injectivity(&bad_pairs),
            ExitCodeInjectivityVerdict::Fail
        );
        // 003: JSON serializer dropped close-brace.
        assert_eq!(
            verdict_from_json_parseable(r#"{"key": "value""#),
            JsonParseableVerdict::Fail
        );
        // 004: inspection mutated state.
        assert_eq!(
            verdict_from_inspection_idempotency(b"a\n", b"b\n", 0, 0),
            InspectionIdempotencyVerdict::Fail
        );
    }
}
