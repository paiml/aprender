//! Pattern detection utilities for bug-hunter.
//!
//! This module contains functions for detecting code patterns and determining
//! whether they represent real technical debt or false positives.
//!
//! # Safety:
//!
//! This module contains the string literals "unsafe {" and "transmute" as
//! pattern matchers for detecting unsafe code in scanned files. These are
//! string constants used for pattern matching, not actual unsafe code.

use super::types::Finding;
use std::collections::HashSet;

/// Check if a finding should be suppressed (BH-15).
/// Wired into analyze_common_patterns per issue #17.
pub fn should_suppress_finding(finding: &Finding, line_content: &str) -> bool {
    // Issue #17: Suppress identical-blocks warnings for mapper functions
    if finding.title.contains("identical blocks") || finding.title.contains("if_same_then_else") {
        // Check if this looks like a mapper function (returns enum variants)
        if line_content.contains("=>") || line_content.contains("PartitionSpec::") {
            return true;
        }
        // Check for intentional comment
        if line_content.contains("INTENTIONAL") || line_content.contains("intentional") {
            return true;
        }
    }

    // Suppress warnings about code that detects patterns (meta-level)
    if (line_content.contains("PATTERN_MARKERS") || line_content.contains("pattern"))
        && (finding.title.contains("FIXME") || finding.title.contains("TODO"))
    {
        return true;
    }

    false
}

/// Determine which lines are inside test code (after #[cfg(test)] or #[test]).
pub fn compute_test_lines(content: &str) -> HashSet<usize> {
    let mut test_lines = HashSet::new();
    let mut in_test_module = false;
    let mut test_module_start_depth: i32 = 0;
    let mut brace_depth: i32 = 0;
    let mut waiting_for_brace = false;

    for (line_num, line) in content.lines().enumerate() {
        let line_num = line_num + 1;
        let trimmed = line.trim();

        // Track brace depth changes on this line
        let open_braces = line.matches('{').count() as i32;
        let close_braces = line.matches('}').count() as i32;

        // Check for test module entry: #[cfg(test)]
        if trimmed == "#[cfg(test)]" {
            waiting_for_brace = true;
            test_lines.insert(line_num); // The attribute itself is test code
        }

        // Check for individual test function: #[test]
        if trimmed == "#[test]" || trimmed.starts_with("#[test]") {
            waiting_for_brace = true;
            test_lines.insert(line_num); // The attribute itself is test code
        }

        // If we're waiting for the opening brace of a test block
        if waiting_for_brace && open_braces > 0 {
            in_test_module = true;
            test_module_start_depth = brace_depth; // Remember depth BEFORE this line's braces
            waiting_for_brace = false;
        }

        // Update brace depth
        brace_depth += open_braces - close_braces;

        // Mark lines inside test modules
        if in_test_module {
            test_lines.insert(line_num);
            // Check if we've exited the test module (brace depth returned to start level)
            if brace_depth <= test_module_start_depth {
                in_test_module = false;
            }
        }
    }

    test_lines
}

/// Check tech debt markers (TODO/FIXME/HACK/XXX) for real vs false positive.
fn check_tech_debt_real(line: &str, before: &str, trimmed: &str) -> bool {
    let is_doc_comment = trimmed.starts_with("///") || trimmed.starts_with("//!");
    if is_doc_comment {
        return false;
    }
    let pattern_count =
        ["TODO", "FIXME", "HACK", "XXX"].iter().filter(|p| line.contains(*p)).count();
    if pattern_count >= 2 {
        return false;
    }
    let has_comment = before.contains("//") || before.contains("/*");
    let quotes_before = before.matches('"').count();
    let in_string = quotes_before % 2 == 1;
    let char_before = before.chars().last();
    let has_space_before = matches!(char_before, Some(' ' | '\t' | '/' | '*') | None);
    has_comment && !in_string && has_space_before
}

/// Check comment-based patterns (test debt, GPU errors) for real vs false positive.
fn check_comment_pattern_real(line: &str, before: &str, trimmed: &str) -> bool {
    let is_comment = trimmed.starts_with("//");
    let quotes_before = before.matches('"').count();
    let in_string = quotes_before % 2 == 1;
    let is_doc_comment = trimmed.starts_with("///") || trimmed.starts_with("//!");
    if is_doc_comment {
        return false;
    }
    let line_lower = line.to_lowercase();
    if line_lower.contains("debug:")
        || line_lower.contains("for debugging")
        || line_lower.contains("diagnostic")
    {
        return false;
    }
    if line_lower.contains("returns cuda_error")
        || line_lower.contains("fix:")
        || line_lower.contains("via ")
        || line_lower.contains("sentinel")
        || line_lower.contains("recreates")
    {
        return false;
    }
    is_comment && !in_string
}

/// Check "unimplemented" pattern for intentional design choices vs real debt.
fn check_unimplemented_exclusions(line: &str, trimmed: &str) -> bool {
    let line_lower = line.to_lowercase();
    if line_lower.contains("does not support")
        || line_lower.contains("not supported")
        || line_lower.contains("use minimize")
        || line_lower.contains("by design")
    {
        return true;
    }
    let trimmed_lower = trimmed.to_lowercase();
    if trimmed_lower == "unimplemented!("
        || (trimmed_lower.starts_with("unimplemented!(") && !trimmed_lower.contains(')'))
    {
        return true;
    }
    if line_lower.contains("_unimplemented")
        || line_lower.contains("should_panic")
        || line_lower.contains("// test unimplemented")
    {
        return true;
    }
    false
}

/// Check if "not implemented" appears in a test-assertion context.
fn is_not_implemented_test_context(line_lower: &str) -> bool {
    line_lower.contains("assert")
        || line_lower.contains("expect")
        || line_lower.contains("returns error")
        || line_lower.contains("should fail")
        || line_lower.contains("should panic")
        || line_lower.contains("test_")
        || line_lower.contains("_test")
        || line_lower.contains("is_err")
}

/// Check if "not implemented" is inside a format string or string literal.
fn is_not_implemented_in_string(line: &str, trimmed: &str) -> bool {
    let trimmed_end = trimmed.trim_end();
    trimmed_end.ends_with("\",")
        || trimmed_end.ends_with('"')
        || line.contains("{}")
        || line.contains("{:")
}

/// Check if a "not implemented" comment is benign (short or describes failures).
fn is_not_implemented_benign_comment(line_lower: &str, trimmed: &str) -> bool {
    if !trimmed.starts_with("//") {
        return false;
    }
    line_lower.contains("fails")
        || line_lower.contains("error")
        || line_lower.contains("but not implemented")
        || trimmed.len() < 50
}

/// Check "not implemented" pattern for test context vs real debt.
fn check_not_implemented_exclusions(line: &str, trimmed: &str) -> bool {
    let line_lower = line.to_lowercase();
    is_not_implemented_test_context(&line_lower)
        || is_not_implemented_in_string(line, trimmed)
        || is_not_implemented_benign_comment(&line_lower, trimmed)
}

/// Check if a single-word euphemism is mid-identifier (false positive).
fn is_mid_identifier_euphemism(pattern: &str, before: &str) -> bool {
    const SINGLE_WORD_EUPHEMISMS: [&str; 7] =
        ["placeholder", "stub", "dummy", "fake", "mock", "temporary", "hardcoded"];
    if !SINGLE_WORD_EUPHEMISMS.contains(&pattern) {
        return false;
    }
    before.chars().last().is_some_and(|c| c == '_' || c.is_alphanumeric())
}

/// Check if "hardcoded"/"hard-coded" is used descriptively (not as debt).
fn is_hardcoded_descriptive(line: &str, pattern: &str, trimmed: &str) -> bool {
    if pattern != "hardcoded" && pattern != "hard-coded" {
        return false;
    }
    let line_lower = line.to_lowercase();
    line_lower.contains("from the hardcoded")
        || line_lower.contains("uses hardcoded")
        || line_lower.contains("using hardcoded")
        || (trimmed.starts_with("//") && line_lower.contains("should"))
}

/// Check euphemism patterns (placeholder, stub, dummy, etc.) for real vs false positive.
fn check_euphemism_real(line: &str, pattern: &str, before: &str, trimmed: &str) -> bool {
    let is_doc_comment = trimmed.starts_with("///") || trimmed.starts_with("//!");
    if is_doc_comment {
        return false;
    }
    if before.matches('"').count() % 2 == 1 {
        return false;
    }
    if pattern == "unimplemented" && check_unimplemented_exclusions(line, trimmed) {
        return false;
    }
    if pattern == "not implemented" && check_not_implemented_exclusions(line, trimmed) {
        return false;
    }
    if is_mid_identifier_euphemism(pattern, before) {
        return false;
    }
    if is_hardcoded_descriptive(line, pattern, trimmed) {
        return false;
    }
    true
}

/// Check code patterns (unwrap, unsafe, etc.) for real vs false positive.
fn check_code_pattern_real(before: &str, pattern: &str, trimmed: &str) -> bool {
    let quotes_before = before.matches('"').count();
    let in_string = quotes_before % 2 == 1;
    let is_doc_comment = trimmed.starts_with("///") || trimmed.starts_with("//!");
    let is_comment = trimmed.starts_with("//");
    // SAFETY: no actual unsafe code -- string literals for pattern matching against scanned code
    let keyword_patterns = ["unsafe {", "transmute", "panic!"];
    if keyword_patterns
        .iter()
        .any(|kw| pattern.starts_with(kw.split_whitespace().next().unwrap_or(kw)))
    {
        if let Some(c) = before.chars().last() {
            if c.is_alphanumeric() || c == '_' {
                return false;
            }
        }
    }
    !in_string && !is_doc_comment && !is_comment
}

/// Check if pattern appears in a "real" code context, not inside a string literal.
pub fn is_real_pattern(line: &str, pattern: &str) -> bool {
    let Some(pos) = line.find(pattern) else {
        return false;
    };
    let trimmed = line.trim();
    let before = &line[..pos];

    if matches!(pattern, "TODO" | "FIXME" | "HACK" | "XXX") {
        return check_tech_debt_real(line, before, trimmed);
    }

    let is_comment_pattern = matches!(
        pattern,
        "were removed"
            | "tests hang"
            | "hang during"
            | "compilation hang"
            | "// skip"
            | "// skipped"
            | "// broken"
            | "// fails"
            | "// disabled"
            | "// fallback"
            | "// degraded"
            | "CUDA_ERROR"
            | "INVALID_PTX"
            | "PTX error"
            | "kernel fail"
    );
    if is_comment_pattern {
        return check_comment_pattern_real(line, before, trimmed);
    }

    let is_euphemism_pattern = matches!(
        pattern,
        "placeholder"
            | "stub"
            | "dummy"
            | "fake"
            | "mock"
            | "simplified"
            | "for demonstration"
            | "demo only"
            | "not implemented"
            | "unimplemented"
            | "temporary"
            | "hardcoded"
            | "hard-coded"
            | "magic number"
            | "workaround"
            | "quick fix"
            | "quick-fix"
            | "bandaid"
            | "band-aid"
            | "kludge"
            | "tech debt"
            | "technical debt"
    );
    if is_euphemism_pattern {
        return check_euphemism_real(line, pattern, before, trimmed);
    }

    check_code_pattern_real(before, pattern, trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_real_pattern_todo_in_comment() {
        assert!(is_real_pattern("// TODO: fix this", "TODO"));
    }

    #[test]
    fn test_is_real_pattern_todo_in_string() {
        assert!(!is_real_pattern(r#"let msg = "TODO: implement";"#, "TODO"));
    }

    #[test]
    fn test_is_real_pattern_todo_in_doc_comment() {
        assert!(!is_real_pattern("/// TODO: document this", "TODO"));
    }

    #[test]
    fn test_is_real_pattern_multiple_patterns() {
        // Line mentions multiple SATD patterns - probably explaining them
        assert!(!is_real_pattern("// For TODO/FIXME/HACK/XXX patterns", "TODO"));
    }

    #[test]
    fn test_compute_test_lines_basic() {
        let content = "fn normal() {}\n\n#[cfg(test)]\nmod tests {\n    fn test_foo() {}\n}\n";
        let test_lines = compute_test_lines(content);
        // Line 3 is #[cfg(test)], lines 4-6 are inside test module
        assert!(test_lines.contains(&3)); // #[cfg(test)]
        assert!(test_lines.contains(&4)); // mod tests {
        assert!(test_lines.contains(&5)); // fn test_foo() {}
        assert!(test_lines.contains(&6)); // }
                                          // Line 1 is normal function, not in test
        assert!(!test_lines.contains(&1));
    }

    // =========================================================================
    // is_real_pattern: comment_pattern branch (lines 138-189)
    // =========================================================================

    #[test]
    fn test_is_real_pattern_comment_pattern_in_comment() {
        // "were removed" in a regular comment → real
        assert!(is_real_pattern("// tests were removed from suite", "were removed"));
        assert!(is_real_pattern("// tests hang during CI", "tests hang"));
    }

    #[test]
    fn test_is_real_pattern_comment_pattern_in_doc_comment() {
        // "were removed" in a doc comment → excluded
        assert!(!is_real_pattern("/// tests were removed from suite", "were removed"));
        assert!(!is_real_pattern("//! tests hang during CI", "tests hang"));
    }

    #[test]
    fn test_is_real_pattern_comment_pattern_in_code() {
        // "were removed" in actual code (not a comment) → excluded
        assert!(!is_real_pattern("let msg = were_removed();", "were removed"));
    }

    #[test]
    fn test_is_real_pattern_comment_pattern_in_string() {
        // Inside a string literal → excluded
        assert!(!is_real_pattern(r#"let msg = "tests hang";"#, "tests hang"));
    }

    #[test]
    fn test_is_real_pattern_comment_pattern_debug_excluded() {
        // Debug/diagnostic comments → excluded
        assert!(!is_real_pattern("// Debug: hang during test", "hang during"));
        assert!(!is_real_pattern("// for debugging: compilation hang", "compilation hang"));
        assert!(!is_real_pattern("// diagnostic: kernel fail info", "kernel fail"));
    }

    #[test]
    fn test_is_real_pattern_comment_pattern_arch_excluded() {
        // Architectural documentation → excluded
        assert!(!is_real_pattern("// returns CUDA_ERROR_UNKNOWN in this case", "CUDA_ERROR"));
        assert!(!is_real_pattern("// Fix: INVALID_PTX via recompilation", "INVALID_PTX"));
        assert!(!is_real_pattern("// sentinel: PTX error code", "PTX error"));
    }

    #[test]
    fn test_is_real_pattern_gpu_patterns() {
        // These are in the is_comment_pattern allowlist — match in comments
        assert!(is_real_pattern("// CUDA_ERROR observed in production", "CUDA_ERROR"));
        assert!(is_real_pattern("// INVALID_PTX found in kernel", "INVALID_PTX"));
        assert!(is_real_pattern("// kernel fail during batch", "kernel fail"));
        // "cuBLAS fallback" is NOT in comment_pattern list → excluded in comments
        assert!(!is_real_pattern("// cuBLAS fallback triggered", "cuBLAS fallback"));
    }

    // =========================================================================
    // is_real_pattern: euphemism_pattern branch (lines 191-341)
    // =========================================================================

    #[test]
    fn test_is_real_pattern_euphemism_in_code() {
        // "placeholder" in code → real
        assert!(is_real_pattern("let placeholder = vec![0.0; 10];", "placeholder"));
        assert!(is_real_pattern("fn stub_impl() { }", "stub"));
    }

    #[test]
    fn test_is_real_pattern_euphemism_in_doc_comment() {
        // Euphemism in doc comment → excluded
        assert!(!is_real_pattern("/// This is a placeholder for later", "placeholder"));
        assert!(!is_real_pattern("//! stub implementation", "stub"));
    }

    #[test]
    fn test_is_real_pattern_euphemism_in_string() {
        // Euphemism in string literal → excluded
        assert!(!is_real_pattern(r#"let msg = "placeholder value";"#, "placeholder"));
    }

    #[test]
    fn test_is_real_pattern_euphemism_mid_identifier() {
        // Euphemism as part of a larger identifier (preceded by _ or alphanumeric) → excluded
        assert!(!is_real_pattern("let foo_placeholder = 1;", "placeholder"));
        assert!(!is_real_pattern("fn my_stub() {}", "stub"));
    }

    #[test]
    fn test_is_real_pattern_unimplemented_with_explanation() {
        // unimplemented!() with design explanation → excluded
        assert!(!is_real_pattern(
            r#"unimplemented!("does not support stochastic updates")"#,
            "unimplemented"
        ));
        assert!(!is_real_pattern(r#"unimplemented!("not supported by design")"#, "unimplemented"));
    }

    #[test]
    fn test_is_real_pattern_unimplemented_bare() {
        // Bare unimplemented!( without closing paren → excluded (msg on next line)
        assert!(!is_real_pattern("        unimplemented!(", "unimplemented"));
    }

    #[test]
    fn test_is_real_pattern_unimplemented_in_test() {
        // unimplemented in test context → excluded
        assert!(!is_real_pattern("fn test_foo_unimplemented() {", "unimplemented"));
        assert!(!is_real_pattern("#[should_panic] fn unimplemented_test() {}", "unimplemented"));
    }

    #[test]
    fn test_is_real_pattern_not_implemented_in_test_assertion() {
        // "not implemented" in test assertion context → excluded
        assert!(!is_real_pattern(
            r#"assert!(result.is_err()); // not implemented"#,
            "not implemented"
        ));
        assert!(!is_real_pattern("assert_eq!(err, \"not implemented\");", "not implemented"));
    }

    #[test]
    fn test_is_real_pattern_not_implemented_format_string() {
        // "not implemented" in format string → excluded
        assert!(!is_real_pattern(r#"format!("{} not implemented", name)"#, "not implemented"));
    }

    #[test]
    fn test_is_real_pattern_not_implemented_comment_short() {
        // Short comment about "not implemented" → excluded (len < 50)
        assert!(!is_real_pattern("// not implemented yet", "not implemented"));
        // Describing failure → excluded
        assert!(!is_real_pattern("// Still fails because not implemented", "not implemented"));
    }

    #[test]
    fn test_is_real_pattern_hardcoded_exclusions() {
        // hardcoded in test explanation → excluded
        assert!(!is_real_pattern("// from the hardcoded test data", "hardcoded"));
        assert!(!is_real_pattern("// uses hardcoded values for testing", "hardcoded"));
    }

    #[test]
    fn test_is_real_pattern_tech_debt_markers() {
        assert!(is_real_pattern("let x = 1; // tech debt from v1", "tech debt"));
        assert!(is_real_pattern("// This is a kludge that needs fixing", "kludge"));
        assert!(is_real_pattern("let workaround = compute();", "workaround"));
    }

    // =========================================================================
    // is_real_pattern: code_pattern branch (lines 343-370)
    // =========================================================================

    #[test]
    fn test_is_real_pattern_code_pattern_in_doc_comment() {
        // Code patterns in doc comments → excluded
        assert!(!is_real_pattern("/// Use unwrap() only in tests", "unwrap()"));
        // SAFETY: no actual unsafe code -- testing pattern detection for doc comment exclusion
        assert!(!is_real_pattern("//! unsafe blocks require safety docs", "unsafe {"));
    }

    #[test]
    fn test_is_real_pattern_code_pattern_in_regular_comment() {
        // Code patterns in regular comments → excluded
        assert!(!is_real_pattern("// be careful with unwrap()", "unwrap()"));
        assert!(!is_real_pattern("// avoid panic! in production", "panic!"));
    }

    #[test]
    fn test_is_real_pattern_keyword_in_identifier() {
        // SAFETY: no actual unsafe code -- testing identifier-embedded keyword exclusion
        assert!(!is_real_pattern("if in_unsafe {", "unsafe {"));
        assert!(!is_real_pattern("let foo_unsafe = true;", "unsafe {"));
    }

    #[test]
    fn test_is_real_pattern_code_pattern_real() {
        // SAFETY: no actual unsafe code -- testing that real unsafe patterns ARE detected
        assert!(is_real_pattern("    unsafe { ptr::read(p) }", "unsafe {"));
        assert!(is_real_pattern("let x = opt.unwrap();", "unwrap()"));
        assert!(is_real_pattern("    transmute::<u32, f32>(bits)", "transmute"));
    }

    #[test]
    fn test_is_real_pattern_pattern_not_found() {
        // Pattern not in line at all
        assert!(!is_real_pattern("fn main() {}", "TODO"));
    }

    // =========================================================================
    // Existing tests below
    // =========================================================================

    #[test]
    fn test_should_suppress_identical_blocks_mapper() {
        let finding = Finding::new("BH-001", std::path::PathBuf::new(), 1, "identical blocks");
        assert!(should_suppress_finding(&finding, "Foo => Bar"));
    }

    #[test]
    fn test_should_suppress_intentional() {
        let finding = Finding::new("BH-001", std::path::PathBuf::new(), 1, "identical blocks");
        assert!(should_suppress_finding(&finding, "// INTENTIONAL duplicate"));
    }
}
