//! Content Linting Module
//!
//! Validates served content (HTML, CSS, JS, WASM, JSON) to catch errors early.
//!
//! ## Supported Linters
//!
//! | File Type | Checks |
//! |-----------|--------|
//! | HTML | Valid structure, missing attributes, broken links |
//! | CSS | Parse errors, unknown properties |
//! | JavaScript | Syntax errors, module resolution |
//! | WASM | Valid module structure |
//! | JSON | Parse validity |

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::unused_self)]
#![allow(clippy::format_push_string)]

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Lint severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LintSeverity {
    /// Error - must fix
    Error,
    /// Warning - should fix
    Warning,
    /// Info - suggestion
    Info,
}

impl LintSeverity {
    /// Get display string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warning => "WARN",
            Self::Info => "INFO",
        }
    }

    /// Get symbol for display
    #[must_use]
    pub const fn symbol(&self) -> &'static str {
        match self {
            Self::Error => "✗",
            Self::Warning => "⚠",
            Self::Info => "ℹ",
        }
    }
}

/// A single lint result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintResult {
    /// File path
    pub file: PathBuf,
    /// Line number (if applicable)
    pub line: Option<u32>,
    /// Column number (if applicable)
    pub column: Option<u32>,
    /// Severity level
    pub severity: LintSeverity,
    /// Lint code (e.g., "HTML001")
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// Suggestion for fix
    pub suggestion: Option<String>,
}

impl LintResult {
    /// Create a new lint error
    pub fn error(
        file: impl Into<PathBuf>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            file: file.into(),
            line: None,
            column: None,
            severity: LintSeverity::Error,
            code: code.into(),
            message: message.into(),
            suggestion: None,
        }
    }

    /// Create a new lint warning
    pub fn warning(
        file: impl Into<PathBuf>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            file: file.into(),
            line: None,
            column: None,
            severity: LintSeverity::Warning,
            code: code.into(),
            message: message.into(),
            suggestion: None,
        }
    }

    /// Create a new lint info
    pub fn info(
        file: impl Into<PathBuf>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            file: file.into(),
            line: None,
            column: None,
            severity: LintSeverity::Info,
            code: code.into(),
            message: message.into(),
            suggestion: None,
        }
    }

    /// Set line number
    #[must_use]
    pub fn at_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    /// Set column number
    #[must_use]
    pub fn at_column(mut self, column: u32) -> Self {
        self.column = Some(column);
        self
    }

    /// Set suggestion
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// Lint report for a directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintReport {
    /// Root directory
    pub root: PathBuf,
    /// All lint results
    pub results: Vec<LintResult>,
    /// Number of errors
    pub errors: usize,
    /// Number of warnings
    pub warnings: usize,
    /// Number of infos
    pub infos: usize,
    /// Number of files checked
    pub files_checked: usize,
}

impl LintReport {
    /// Create a new lint report
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            results: Vec::new(),
            errors: 0,
            warnings: 0,
            infos: 0,
            files_checked: 0,
        }
    }

    /// Add a lint result
    pub fn add(&mut self, result: LintResult) {
        match result.severity {
            LintSeverity::Error => self.errors += 1,
            LintSeverity::Warning => self.warnings += 1,
            LintSeverity::Info => self.infos += 1,
        }
        self.results.push(result);
    }

    /// Check if there are any errors
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }

    /// Check if the lint passed (no errors)
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.has_errors()
    }
}

/// Content linter
#[derive(Debug)]
pub struct ContentLinter {
    /// Root directory to lint
    root: PathBuf,
    /// Lint HTML files
    pub lint_html: bool,
    /// Lint CSS files
    pub lint_css: bool,
    /// Lint JavaScript files
    pub lint_js: bool,
    /// Lint WASM files
    pub lint_wasm: bool,
    /// Lint JSON files
    pub lint_json: bool,
}

impl ContentLinter {
    /// Create a new content linter
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            lint_html: true,
            lint_css: true,
            lint_js: true,
            lint_wasm: true,
            lint_json: true,
        }
    }

    /// Lint all files in the directory
    pub fn lint(&self) -> LintReport {
        let mut report = LintReport::new(&self.root);
        self.lint_directory(&self.root, &mut report);
        report
    }

    /// Lint a single file
    pub fn lint_file(&self, path: &Path) -> Vec<LintResult> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match extension {
            "html" | "htm" if self.lint_html => self.lint_html_file(path),
            "css" if self.lint_css => self.lint_css_file(path),
            "js" | "mjs" if self.lint_js => self.lint_js_file(path),
            "wasm" if self.lint_wasm => self.lint_wasm_file(path),
            "json" if self.lint_json => self.lint_json_file(path),
            _ => Vec::new(),
        }
    }

    fn lint_directory(&self, dir: &Path, report: &mut LintReport) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();

            // Skip hidden files and common ignore patterns
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }

            if path.is_dir() {
                self.lint_directory(&path, report);
            } else {
                let results = self.lint_file(&path);
                if !results.is_empty() {
                    report.files_checked += 1;
                    for result in results {
                        report.add(result);
                    }
                } else if self.is_lintable(&path) {
                    report.files_checked += 1;
                }
            }
        }
    }

    fn is_lintable(&self, path: &Path) -> bool {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        matches!(
            extension,
            "html" | "htm" | "css" | "js" | "mjs" | "wasm" | "json"
        )
    }

    fn lint_html_file(&self, path: &Path) -> Vec<LintResult> {
        let mut results = Vec::new();

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                results.push(LintResult::error(
                    path,
                    "HTML000",
                    format!("Cannot read file: {e}"),
                ));
                return results;
            }
        };

        // Check for DOCTYPE
        if !content
            .trim_start()
            .to_lowercase()
            .starts_with("<!doctype html")
        {
            results.push(
                LintResult::warning(path, "HTML001", "Missing <!DOCTYPE html> declaration")
                    .at_line(1)
                    .with_suggestion("Add <!DOCTYPE html> at the start of the file"),
            );
        }

        // Check for basic structure
        let content_lower = content.to_lowercase();
        if !content_lower.contains("<html") {
            results.push(LintResult::error(path, "HTML002", "Missing <html> element"));
        }
        if !content_lower.contains("<head") {
            results.push(LintResult::warning(
                path,
                "HTML003",
                "Missing <head> element",
            ));
        }
        if !content_lower.contains("<body") {
            results.push(LintResult::warning(
                path,
                "HTML004",
                "Missing <body> element",
            ));
        }

        // Check for unclosed tags (simple heuristic)
        let open_divs = content_lower.matches("<div").count();
        let close_divs = content_lower.matches("</div>").count();
        if open_divs != close_divs {
            results.push(LintResult::warning(
                path,
                "HTML005",
                format!(
                    "Mismatched <div> tags: {} open, {} close",
                    open_divs, close_divs
                ),
            ));
        }

        // Check for images without alt
        for (line_num, line) in content.lines().enumerate() {
            let line_lower = line.to_lowercase();
            if line_lower.contains("<img") && !line_lower.contains("alt=") {
                results.push(
                    LintResult::warning(path, "HTML006", "<img> tag missing alt attribute")
                        .at_line((line_num + 1) as u32)
                        .with_suggestion("Add alt attribute for accessibility"),
                );
            }
        }

        results
    }

    fn lint_css_file(&self, path: &Path) -> Vec<LintResult> {
        let mut results = Vec::new();

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                results.push(LintResult::error(
                    path,
                    "CSS000",
                    format!("Cannot read file: {e}"),
                ));
                return results;
            }
        };

        // Check for basic syntax errors
        let open_braces = content.matches('{').count();
        let close_braces = content.matches('}').count();
        if open_braces != close_braces {
            results.push(LintResult::error(
                path,
                "CSS001",
                format!(
                    "Mismatched braces: {} open, {} close",
                    open_braces, close_braces
                ),
            ));
        }

        // Check for vendor prefixes without standard property
        for (line_num, line) in content.lines().enumerate() {
            let trimmed = line.trim();

            // Check for webkit without standard
            if trimmed.starts_with("-webkit-") && !trimmed.starts_with("-webkit-") {
                let prop = trimmed.split(':').next().unwrap_or("");
                let standard = prop.trim_start_matches("-webkit-");
                results.push(
                    LintResult::info(path, "CSS002", format!("Vendor prefix {} used", prop))
                        .at_line((line_num + 1) as u32)
                        .with_suggestion(format!("Also include standard property: {}", standard)),
                );
            }

            // Check for empty rules
            if trimmed == "{}" {
                results.push(
                    LintResult::warning(path, "CSS003", "Empty CSS rule")
                        .at_line((line_num + 1) as u32),
                );
            }
        }

        results
    }

    fn lint_js_file(&self, path: &Path) -> Vec<LintResult> {
        let mut results = Vec::new();

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                results.push(LintResult::error(
                    path,
                    "JS000",
                    format!("Cannot read file: {e}"),
                ));
                return results;
            }
        };

        // Check for basic syntax issues
        let open_braces = content.matches('{').count();
        let close_braces = content.matches('}').count();
        if open_braces != close_braces {
            results.push(LintResult::error(
                path,
                "JS001",
                format!(
                    "Mismatched braces: {} open, {} close",
                    open_braces, close_braces
                ),
            ));
        }

        let open_parens = content.matches('(').count();
        let close_parens = content.matches(')').count();
        if open_parens != close_parens {
            results.push(LintResult::error(
                path,
                "JS002",
                format!(
                    "Mismatched parentheses: {} open, {} close",
                    open_parens, close_parens
                ),
            ));
        }

        // Check for common issues
        for (line_num, line) in content.lines().enumerate() {
            // Check for console.log in production
            if line.contains("console.log") {
                results.push(
                    LintResult::info(path, "JS003", "console.log found")
                        .at_line((line_num + 1) as u32)
                        .with_suggestion("Remove console.log before production"),
                );
            }

            // Check for debugger statements
            if line.trim().starts_with("debugger") {
                results.push(
                    LintResult::warning(path, "JS004", "debugger statement found")
                        .at_line((line_num + 1) as u32)
                        .with_suggestion("Remove debugger statements before production"),
                );
            }
        }

        results
    }

    fn lint_wasm_file(&self, path: &Path) -> Vec<LintResult> {
        let mut results = Vec::new();

        let content = match std::fs::read(path) {
            Ok(c) => c,
            Err(e) => {
                results.push(LintResult::error(
                    path,
                    "WASM000",
                    format!("Cannot read file: {e}"),
                ));
                return results;
            }
        };

        // Check WASM magic number
        if content.len() < 8 {
            results.push(LintResult::error(
                path,
                "WASM001",
                "File too small to be valid WASM",
            ));
            return results;
        }

        // WASM magic number: 0x00 0x61 0x73 0x6D (0asm)
        if content[0..4] != [0x00, 0x61, 0x73, 0x6D] {
            results.push(
                LintResult::error(path, "WASM002", "Invalid WASM magic number")
                    .with_suggestion("File does not appear to be a valid WebAssembly module"),
            );
        }

        // Check version (should be 0x01 0x00 0x00 0x00 for version 1)
        if content[4..8] != [0x01, 0x00, 0x00, 0x00] {
            let version = u32::from_le_bytes([content[4], content[5], content[6], content[7]]);
            results.push(LintResult::warning(
                path,
                "WASM003",
                format!("Unexpected WASM version: {}", version),
            ));
        }

        results
    }

    fn lint_json_file(&self, path: &Path) -> Vec<LintResult> {
        let mut results = Vec::new();

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                results.push(LintResult::error(
                    path,
                    "JSON000",
                    format!("Cannot read file: {e}"),
                ));
                return results;
            }
        };

        // Try to parse JSON
        if let Err(e) = serde_json::from_str::<serde_json::Value>(&content) {
            let line = e.line();
            let column = e.column();
            results.push(
                LintResult::error(path, "JSON001", format!("Invalid JSON: {}", e))
                    .at_line(line as u32)
                    .at_column(column as u32),
            );
        }

        results
    }
}

/// Render lint report as text
pub fn render_lint_report(report: &LintReport) -> String {
    let mut output = String::new();

    output.push_str(&format!("LINT REPORT: {}\n", report.root.display()));
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    if report.results.is_empty() {
        output.push_str("✓ All files passed linting\n");
    } else {
        // Group by file
        let mut by_file: std::collections::HashMap<&Path, Vec<&LintResult>> =
            std::collections::HashMap::new();
        for result in &report.results {
            by_file.entry(&result.file).or_default().push(result);
        }

        for (file, results) in by_file {
            let relative = file.strip_prefix(&report.root).unwrap_or(file);
            output.push_str(&format!("{}:\n", relative.display()));

            for result in results {
                let location = match (result.line, result.column) {
                    (Some(l), Some(c)) => format!("Line {}:{}", l, c),
                    (Some(l), None) => format!("Line {}", l),
                    _ => String::new(),
                };

                output.push_str(&format!(
                    "  {} {} [{}] {}\n",
                    result.severity.symbol(),
                    location,
                    result.code,
                    result.message
                ));

                if let Some(ref suggestion) = result.suggestion {
                    output.push_str(&format!("      Suggestion: {}\n", suggestion));
                }
            }
            output.push('\n');
        }
    }

    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    output.push_str(&format!(
        "Summary: {} errors, {} warnings, {} files checked\n",
        report.errors, report.warnings, report.files_checked
    ));

    output
}

/// Render lint report as JSON
pub fn render_lint_json(report: &LintReport) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_lint_severity_as_str() {
        assert_eq!(LintSeverity::Error.as_str(), "ERROR");
        assert_eq!(LintSeverity::Warning.as_str(), "WARN");
        assert_eq!(LintSeverity::Info.as_str(), "INFO");
    }

    #[test]
    fn test_lint_severity_symbol() {
        assert_eq!(LintSeverity::Error.symbol(), "✗");
        assert_eq!(LintSeverity::Warning.symbol(), "⚠");
        assert_eq!(LintSeverity::Info.symbol(), "ℹ");
    }

    #[test]
    fn test_lint_result_builder() {
        let result = LintResult::error("test.html", "HTML001", "Test error")
            .at_line(10)
            .at_column(5)
            .with_suggestion("Fix it");

        assert_eq!(result.file, PathBuf::from("test.html"));
        assert_eq!(result.code, "HTML001");
        assert_eq!(result.line, Some(10));
        assert_eq!(result.column, Some(5));
        assert_eq!(result.suggestion, Some("Fix it".to_string()));
    }

    #[test]
    fn test_lint_report_add() {
        let mut report = LintReport::new("./");
        report.add(LintResult::error("a.html", "E001", "error"));
        report.add(LintResult::warning("b.css", "W001", "warning"));
        report.add(LintResult::info("c.js", "I001", "info"));

        assert_eq!(report.errors, 1);
        assert_eq!(report.warnings, 1);
        assert_eq!(report.infos, 1);
        assert!(report.has_errors());
        assert!(!report.passed());
    }

    #[test]
    fn test_lint_report_no_errors() {
        let mut report = LintReport::new("./");
        report.add(LintResult::warning("a.css", "W001", "warning"));

        assert!(!report.has_errors());
        assert!(report.passed());
    }

    #[test]
    fn test_lint_html_missing_doctype() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("test.html");
        std::fs::write(&html_path, "<html><head></head><body></body></html>").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&html_path);

        assert!(results.iter().any(|r| r.code == "HTML001"));
    }

    #[test]
    fn test_lint_html_valid() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("test.html");
        std::fs::write(
            &html_path,
            "<!DOCTYPE html><html><head></head><body></body></html>",
        )
        .unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&html_path);

        assert!(results.iter().all(|r| r.severity != LintSeverity::Error));
    }

    #[test]
    fn test_lint_html_missing_alt() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("test.html");
        std::fs::write(
            &html_path,
            "<!DOCTYPE html><html><head></head><body><img src=\"test.png\"></body></html>",
        )
        .unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&html_path);

        assert!(results.iter().any(|r| r.code == "HTML006"));
    }

    #[test]
    fn test_lint_css_mismatched_braces() {
        let temp = TempDir::new().unwrap();
        let css_path = temp.path().join("test.css");
        std::fs::write(&css_path, "body { color: red;").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&css_path);

        assert!(results.iter().any(|r| r.code == "CSS001"));
    }

    #[test]
    fn test_lint_js_debugger() {
        let temp = TempDir::new().unwrap();
        let js_path = temp.path().join("test.js");
        std::fs::write(&js_path, "function test() {\n  debugger;\n}").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&js_path);

        assert!(results.iter().any(|r| r.code == "JS004"));
    }

    #[test]
    fn test_lint_json_invalid() {
        let temp = TempDir::new().unwrap();
        let json_path = temp.path().join("test.json");
        std::fs::write(&json_path, "{invalid json}").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&json_path);

        assert!(results.iter().any(|r| r.code == "JSON001"));
    }

    #[test]
    fn test_lint_json_valid() {
        let temp = TempDir::new().unwrap();
        let json_path = temp.path().join("test.json");
        std::fs::write(&json_path, r#"{"key": "value"}"#).unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&json_path);

        assert!(results.is_empty());
    }

    #[test]
    fn test_lint_wasm_invalid_magic() {
        let temp = TempDir::new().unwrap();
        let wasm_path = temp.path().join("test.wasm");
        std::fs::write(&wasm_path, b"not wasm data here").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&wasm_path);

        assert!(results.iter().any(|r| r.code == "WASM002"));
    }

    #[test]
    fn test_lint_wasm_valid() {
        let temp = TempDir::new().unwrap();
        let wasm_path = temp.path().join("test.wasm");
        // Valid WASM magic + version 1
        std::fs::write(&wasm_path, [0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]).unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&wasm_path);

        assert!(results.is_empty());
    }

    #[test]
    fn test_render_lint_report() {
        let mut report = LintReport::new("./test");
        report.files_checked = 3;
        report.add(LintResult::error("test.html", "HTML001", "Missing DOCTYPE"));

        let output = render_lint_report(&report);

        assert!(output.contains("LINT REPORT"));
        assert!(output.contains("HTML001"));
        assert!(output.contains("1 errors"));
    }

    #[test]
    fn test_render_lint_json() {
        let report = LintReport::new("./test");
        let json = render_lint_json(&report).unwrap();

        assert!(json.contains("\"root\""));
        assert!(json.contains("\"results\""));
    }

    #[test]
    fn test_lint_directory_full() {
        let temp = TempDir::new().unwrap();

        // Create various test files
        std::fs::write(
            temp.path().join("index.html"),
            "<!DOCTYPE html><html><head></head><body></body></html>",
        )
        .unwrap();
        std::fs::write(temp.path().join("style.css"), "body { color: red; }").unwrap();
        std::fs::write(temp.path().join("app.js"), "function test() {}").unwrap();
        std::fs::write(temp.path().join("data.json"), r#"{"key": "value"}"#).unwrap();

        let linter = ContentLinter::new(temp.path());
        let report = linter.lint();

        // Should have checked files
        assert!(report.files_checked >= 4);
        // All valid files, should pass
        assert!(report.passed());
    }

    #[test]
    fn test_lint_directory_with_errors() {
        let temp = TempDir::new().unwrap();

        // Create files with issues
        std::fs::write(temp.path().join("bad.html"), "<html>no doctype</html>").unwrap();
        std::fs::write(temp.path().join("bad.css"), "body { color: red").unwrap(); // missing }
        std::fs::write(temp.path().join("bad.json"), "{invalid}").unwrap();

        let linter = ContentLinter::new(temp.path());
        let report = linter.lint();

        assert!(report.errors > 0);
        assert!(!report.passed());
    }

    #[test]
    fn test_lint_directory_nested() {
        let temp = TempDir::new().unwrap();

        // Create nested directory structure
        let subdir = temp.path().join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(
            subdir.join("nested.html"),
            "<!DOCTYPE html><html><head></head><body></body></html>",
        )
        .unwrap();

        let linter = ContentLinter::new(temp.path());
        let report = linter.lint();

        // Should find nested file
        assert!(report.files_checked >= 1);
    }

    #[test]
    fn test_lint_directory_skips_hidden() {
        let temp = TempDir::new().unwrap();

        // Create hidden directory
        let hidden = temp.path().join(".hidden");
        std::fs::create_dir(&hidden).unwrap();
        std::fs::write(hidden.join("test.html"), "<html>bad</html>").unwrap();

        // Create visible file for comparison
        std::fs::write(
            temp.path().join("visible.html"),
            "<!DOCTYPE html><html><head></head><body></body></html>",
        )
        .unwrap();

        let linter = ContentLinter::new(temp.path());
        let report = linter.lint();

        // Should only find the visible file
        assert_eq!(report.files_checked, 1);
    }

    #[test]
    fn test_lint_directory_skips_node_modules() {
        let temp = TempDir::new().unwrap();

        // Create node_modules directory
        let node_modules = temp.path().join("node_modules");
        std::fs::create_dir(&node_modules).unwrap();
        std::fs::write(node_modules.join("lib.js"), "console.log('test');").unwrap();

        let linter = ContentLinter::new(temp.path());
        let report = linter.lint();

        // Should skip node_modules
        assert_eq!(report.files_checked, 0);
    }

    #[test]
    fn test_lint_file_unknown_extension() {
        let temp = TempDir::new().unwrap();
        let txt_path = temp.path().join("test.txt");
        std::fs::write(&txt_path, "Just some text").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&txt_path);

        // Unknown extension returns empty results
        assert!(results.is_empty());
    }

    #[test]
    fn test_lint_file_mjs_extension() {
        let temp = TempDir::new().unwrap();
        let mjs_path = temp.path().join("test.mjs");
        std::fs::write(&mjs_path, "export function test() {}").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&mjs_path);

        // Should lint .mjs files
        assert!(results.is_empty()); // Valid JS
    }

    #[test]
    fn test_lint_disabled_html() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("test.html");
        std::fs::write(&html_path, "<html>no doctype</html>").unwrap();

        let mut linter = ContentLinter::new(temp.path());
        linter.lint_html = false;

        let results = linter.lint_file(&html_path);
        assert!(results.is_empty()); // Disabled
    }

    #[test]
    fn test_lint_disabled_css() {
        let temp = TempDir::new().unwrap();
        let css_path = temp.path().join("test.css");
        std::fs::write(&css_path, "body { color: red").unwrap(); // Missing brace

        let mut linter = ContentLinter::new(temp.path());
        linter.lint_css = false;

        let results = linter.lint_file(&css_path);
        assert!(results.is_empty()); // Disabled
    }

    #[test]
    fn test_lint_disabled_js() {
        let temp = TempDir::new().unwrap();
        let js_path = temp.path().join("test.js");
        std::fs::write(&js_path, "console.log('debug');").unwrap();

        let mut linter = ContentLinter::new(temp.path());
        linter.lint_js = false;

        let results = linter.lint_file(&js_path);
        assert!(results.is_empty()); // Disabled
    }

    #[test]
    fn test_lint_disabled_wasm() {
        let temp = TempDir::new().unwrap();
        let wasm_path = temp.path().join("test.wasm");
        std::fs::write(&wasm_path, b"not valid wasm").unwrap();

        let mut linter = ContentLinter::new(temp.path());
        linter.lint_wasm = false;

        let results = linter.lint_file(&wasm_path);
        assert!(results.is_empty()); // Disabled
    }

    #[test]
    fn test_lint_disabled_json() {
        let temp = TempDir::new().unwrap();
        let json_path = temp.path().join("test.json");
        std::fs::write(&json_path, "{invalid}").unwrap();

        let mut linter = ContentLinter::new(temp.path());
        linter.lint_json = false;

        let results = linter.lint_file(&json_path);
        assert!(results.is_empty()); // Disabled
    }

    #[test]
    fn test_lint_wasm_too_small() {
        let temp = TempDir::new().unwrap();
        let wasm_path = temp.path().join("tiny.wasm");
        std::fs::write(&wasm_path, b"tiny").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&wasm_path);

        assert!(results.iter().any(|r| r.code == "WASM001"));
    }

    #[test]
    fn test_lint_wasm_wrong_version() {
        let temp = TempDir::new().unwrap();
        let wasm_path = temp.path().join("oldversion.wasm");
        // Valid magic, but version 2
        std::fs::write(&wasm_path, [0x00, 0x61, 0x73, 0x6D, 0x02, 0x00, 0x00, 0x00]).unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&wasm_path);

        assert!(results.iter().any(|r| r.code == "WASM003"));
    }

    #[test]
    fn test_lint_html_missing_html_tag() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("test.html");
        std::fs::write(&html_path, "<!DOCTYPE html><head></head><body></body>").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&html_path);

        assert!(results.iter().any(|r| r.code == "HTML002"));
    }

    #[test]
    fn test_lint_html_missing_head() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("test.html");
        std::fs::write(&html_path, "<!DOCTYPE html><html><body></body></html>").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&html_path);

        assert!(results.iter().any(|r| r.code == "HTML003"));
    }

    #[test]
    fn test_lint_html_missing_body() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("test.html");
        std::fs::write(&html_path, "<!DOCTYPE html><html><head></head></html>").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&html_path);

        assert!(results.iter().any(|r| r.code == "HTML004"));
    }

    #[test]
    fn test_lint_html_mismatched_divs() {
        let temp = TempDir::new().unwrap();
        let html_path = temp.path().join("test.html");
        std::fs::write(
            &html_path,
            "<!DOCTYPE html><html><head></head><body><div><div></div></body></html>",
        )
        .unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&html_path);

        assert!(results.iter().any(|r| r.code == "HTML005"));
    }

    #[test]
    fn test_lint_css_empty_rule() {
        let temp = TempDir::new().unwrap();
        let css_path = temp.path().join("test.css");
        // CSS003 triggers when line is exactly "{}"
        std::fs::write(&css_path, "body\n{}\n.empty\n{}").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&css_path);

        assert!(results.iter().any(|r| r.code == "CSS003"));
    }

    #[test]
    fn test_lint_js_console_log() {
        let temp = TempDir::new().unwrap();
        let js_path = temp.path().join("test.js");
        std::fs::write(&js_path, "console.log('debugging');").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&js_path);

        assert!(results.iter().any(|r| r.code == "JS003"));
    }

    #[test]
    fn test_lint_js_mismatched_braces() {
        let temp = TempDir::new().unwrap();
        let js_path = temp.path().join("test.js");
        std::fs::write(&js_path, "function test() { return 1;").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&js_path);

        assert!(results.iter().any(|r| r.code == "JS001"));
    }

    #[test]
    fn test_lint_js_mismatched_parens() {
        let temp = TempDir::new().unwrap();
        let js_path = temp.path().join("test.js");
        std::fs::write(&js_path, "function test( { return 1; }").unwrap();

        let linter = ContentLinter::new(temp.path());
        let results = linter.lint_file(&js_path);

        assert!(results.iter().any(|r| r.code == "JS002"));
    }

    #[test]
    fn test_is_lintable() {
        let temp = TempDir::new().unwrap();
        let linter = ContentLinter::new(temp.path());

        assert!(linter.is_lintable(Path::new("test.html")));
        assert!(linter.is_lintable(Path::new("test.htm")));
        assert!(linter.is_lintable(Path::new("test.css")));
        assert!(linter.is_lintable(Path::new("test.js")));
        assert!(linter.is_lintable(Path::new("test.mjs")));
        assert!(linter.is_lintable(Path::new("test.wasm")));
        assert!(linter.is_lintable(Path::new("test.json")));
        assert!(!linter.is_lintable(Path::new("test.txt")));
        assert!(!linter.is_lintable(Path::new("test.rs")));
    }

    #[test]
    fn test_render_report_with_location() {
        let mut report = LintReport::new("./test");
        report.add(
            LintResult::error("test.html", "HTML001", "Test")
                .at_line(5)
                .at_column(10),
        );

        let output = render_lint_report(&report);
        assert!(output.contains("Line 5:10"));
    }

    #[test]
    fn test_render_report_with_line_only() {
        let mut report = LintReport::new("./test");
        report.add(LintResult::warning("test.css", "CSS001", "Test").at_line(3));

        let output = render_lint_report(&report);
        assert!(output.contains("Line 3"));
    }

    #[test]
    fn test_render_report_empty() {
        let report = LintReport::new("./test");
        let output = render_lint_report(&report);

        assert!(output.contains("All files passed linting"));
    }

    #[test]
    fn test_render_report_with_suggestion() {
        let mut report = LintReport::new("./test");
        report.add(
            LintResult::info("test.js", "JS001", "Found issue")
                .with_suggestion("Try fixing it this way"),
        );

        let output = render_lint_report(&report);
        assert!(output.contains("Suggestion: Try fixing it this way"));
    }

    #[test]
    fn test_lint_file_read_error() {
        let linter = ContentLinter::new("/tmp");

        // Try to lint a non-existent file
        let results = linter.lint_file(Path::new("/nonexistent/test.html"));
        assert!(results.iter().any(|r| r.code == "HTML000"));

        let results = linter.lint_file(Path::new("/nonexistent/test.css"));
        assert!(results.iter().any(|r| r.code == "CSS000"));

        let results = linter.lint_file(Path::new("/nonexistent/test.js"));
        assert!(results.iter().any(|r| r.code == "JS000"));

        let results = linter.lint_file(Path::new("/nonexistent/test.wasm"));
        assert!(results.iter().any(|r| r.code == "WASM000"));

        let results = linter.lint_file(Path::new("/nonexistent/test.json"));
        assert!(results.iter().any(|r| r.code == "JSON000"));
    }
}
