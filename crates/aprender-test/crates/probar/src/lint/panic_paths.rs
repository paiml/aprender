//! Panic Path Detection Linter (PROBAR-WASM-006)
//!
//! Static analysis to detect panic-inducing patterns in WASM code.
//!
//! ## Motivation
//!
//! In WASM, panic paths cause `wasm_bindgen::throw_str` which terminates
//! the entire WASM instance. Unlike native Rust where panics can be caught,
//! WASM panics are unrecoverable and break the user experience.
//!
//! ## Detection Rules
//!
//! | Rule ID | Description | Severity |
//! |---------|-------------|----------|
//! | WASM-PANIC-001 | `unwrap()` call | Error |
//! | WASM-PANIC-002 | `expect()` call | Error |
//! | WASM-PANIC-003 | `panic!()` macro | Error |
//! | WASM-PANIC-004 | `unreachable!()` macro | Warning |
//! | WASM-PANIC-005 | `todo!()` macro | Error |
//! | WASM-PANIC-006 | `unimplemented!()` macro | Error |
//! | WASM-PANIC-007 | Index access without bounds check | Warning |
//!
//! ## Example
//!
//! ```rust,ignore
//! // BAD: Will panic in WASM
//! let value = some_option.unwrap();
//!
//! // GOOD: Proper error handling
//! let value = some_option.ok_or(MyError::Missing)?;
//! ```

use super::{LintError, LintSeverity, StateSyncReport};
use syn::visit::Visit;
use syn::{ExprMethodCall, Macro};

/// Patterns that indicate panic paths
const PANIC_METHODS: &[&str] = &["unwrap", "expect"];

/// Macros that always panic
const PANIC_MACROS: &[&str] = &["panic", "unreachable", "todo", "unimplemented"];

/// AST visitor for detecting panic paths
#[derive(Debug)]
pub struct PanicPathVisitor {
    /// Current file being analyzed
    file: String,
    /// Collected errors
    errors: Vec<LintError>,
    /// Source code for line lookups
    source: String,
    /// Whether we're inside a test module (relaxed rules)
    in_test_module: bool,
    /// Whether we're inside an unsafe block
    in_unsafe_block: bool,
}

impl PanicPathVisitor {
    /// Create a new panic path visitor
    #[must_use]
    pub fn new(file: String, source: String) -> Self {
        Self {
            file,
            errors: Vec::new(),
            source,
            in_test_module: false,
            in_unsafe_block: false,
        }
    }

    /// Get the line number for a span
    fn span_to_line(&self, span: proc_macro2::Span) -> usize {
        span.start().line
    }

    /// Get the column for a span
    fn span_to_column(&self, span: proc_macro2::Span) -> usize {
        span.start().column + 1
    }

    /// Get the line content for context
    fn get_line_content(&self, line: usize) -> String {
        self.source
            .lines()
            .nth(line.saturating_sub(1))
            .unwrap_or("")
            .trim()
            .to_string()
    }

    /// Check if a method name is a panic method
    fn is_panic_method(method: &str) -> bool {
        PANIC_METHODS.contains(&method)
    }

    /// Check if a macro path is a panic macro
    fn is_panic_macro(path: &syn::Path) -> bool {
        if let Some(ident) = path.get_ident() {
            let name = ident.to_string();
            return PANIC_MACROS.contains(&name.as_str());
        }
        // Check for std::panic, core::panic, etc.
        if let Some(last) = path.segments.last() {
            let name = last.ident.to_string();
            return PANIC_MACROS.contains(&name.as_str());
        }
        false
    }

    /// Get severity for a panic macro
    fn macro_severity(name: &str) -> LintSeverity {
        match name {
            "unreachable" => LintSeverity::Warning, // Sometimes intentional
            _ => LintSeverity::Error,
        }
    }

    /// Get suggestion for a panic method
    fn suggestion_for_method(method: &str) -> String {
        match method {
            "unwrap" => {
                "Use `ok_or(err)?` or `unwrap_or_default()` instead of `unwrap()`".to_string()
            }
            "expect" => "Use `ok_or(err)?` or `unwrap_or_else(|| default)` instead of `expect()`"
                .to_string(),
            _ => format!("Avoid `{method}()` in WASM code"),
        }
    }

    /// Get suggestion for a panic macro
    fn suggestion_for_macro(name: &str) -> String {
        match name {
            "panic" => {
                "Return a Result or use `wasm_bindgen::throw_str` for controlled errors".to_string()
            }
            "unreachable" => {
                "Use `unreachable!()` only when truly unreachable; prefer `debug_assert!`"
                    .to_string()
            }
            "todo" => "Implement the function or return `Err(\"not implemented\")`.to_string()`"
                .to_string(),
            "unimplemented" => {
                "Implement the function or return an error instead of panicking".to_string()
            }
            _ => format!("Avoid `{name}!()` in WASM code"),
        }
    }

    /// Convert to report
    #[must_use]
    pub fn into_report(self, lines_analyzed: usize) -> StateSyncReport {
        StateSyncReport {
            errors: self.errors,
            files_analyzed: 1,
            lines_analyzed,
        }
    }

    /// Get errors
    #[must_use]
    pub fn errors(&self) -> &[LintError] {
        &self.errors
    }
}

impl<'ast> Visit<'ast> for PanicPathVisitor {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        // Check if this is a test module
        let is_test = node.attrs.iter().any(|attr| {
            attr.path().is_ident("cfg")
                && attr
                    .meta
                    .require_list()
                    .ok()
                    .and_then(|list| list.parse_args::<syn::Ident>().ok())
                    .is_some_and(|ident| ident == "test")
        });

        let was_in_test = self.in_test_module;
        if is_test {
            self.in_test_module = true;
        }

        syn::visit::visit_item_mod(self, node);

        self.in_test_module = was_in_test;
    }

    fn visit_expr_unsafe(&mut self, node: &'ast syn::ExprUnsafe) {
        let was_unsafe = self.in_unsafe_block;
        self.in_unsafe_block = true;
        syn::visit::visit_expr_unsafe(self, node);
        self.in_unsafe_block = was_unsafe;
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        // Skip if in test module
        if self.in_test_module {
            syn::visit::visit_expr_method_call(self, node);
            return;
        }

        let method_name = node.method.to_string();

        if Self::is_panic_method(&method_name) {
            let line = self.span_to_line(node.method.span());
            let column = self.span_to_column(node.method.span());
            let line_content = self.get_line_content(line);

            // Check for allow attribute on the line (basic heuristic)
            let has_allow = line_content.contains("#[allow(")
                || line_content.contains("// SAFETY:")
                || line_content.contains("// PANIC:");

            if !has_allow {
                let rule = match method_name.as_str() {
                    "unwrap" => "WASM-PANIC-001",
                    "expect" => "WASM-PANIC-002",
                    _ => "WASM-PANIC-000",
                };

                self.errors.push(LintError {
                    rule: rule.to_string(),
                    message: format!(
                        "`{method_name}()` can panic, which terminates WASM execution"
                    ),
                    file: self.file.clone(),
                    line,
                    column,
                    severity: LintSeverity::Error,
                    suggestion: Some(Self::suggestion_for_method(&method_name)),
                });
            }
        }

        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_macro(&mut self, node: &'ast Macro) {
        // Skip if in test module
        if self.in_test_module {
            syn::visit::visit_macro(self, node);
            return;
        }

        if Self::is_panic_macro(&node.path) {
            let macro_name = node
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();

            let line = self.span_to_line(
                node.path
                    .segments
                    .first()
                    .map_or_else(|| proc_macro2::Span::call_site(), |s| s.ident.span()),
            );
            let column = self.span_to_column(
                node.path
                    .segments
                    .first()
                    .map_or_else(|| proc_macro2::Span::call_site(), |s| s.ident.span()),
            );

            let rule = match macro_name.as_str() {
                "panic" => "WASM-PANIC-003",
                "unreachable" => "WASM-PANIC-004",
                "todo" => "WASM-PANIC-005",
                "unimplemented" => "WASM-PANIC-006",
                _ => "WASM-PANIC-000",
            };

            self.errors.push(LintError {
                rule: rule.to_string(),
                message: format!("`{macro_name}!()` panics, which terminates WASM execution"),
                file: self.file.clone(),
                line,
                column,
                severity: Self::macro_severity(&macro_name),
                suggestion: Some(Self::suggestion_for_macro(&macro_name)),
            });
        }

        syn::visit::visit_macro(self, node);
    }

    fn visit_expr_index(&mut self, node: &'ast syn::ExprIndex) {
        // Skip if in test module or unsafe block
        if self.in_test_module || self.in_unsafe_block {
            syn::visit::visit_expr_index(self, node);
            return;
        }

        // Direct indexing like arr[i] can panic
        let line = self.span_to_line(node.bracket_token.span.open());
        let column = self.span_to_column(node.bracket_token.span.open());

        self.errors.push(LintError {
            rule: "WASM-PANIC-007".to_string(),
            message: "Direct indexing can panic on out-of-bounds access".to_string(),
            file: self.file.clone(),
            line,
            column,
            severity: LintSeverity::Warning,
            suggestion: Some("Use `.get(index)` with proper error handling instead".to_string()),
        });

        syn::visit::visit_expr_index(self, node);
    }
}

/// Lint source code for panic paths
///
/// # Arguments
/// * `source` - Rust source code to analyze
/// * `file` - File name for error reporting
///
/// # Returns
/// A report containing all panic path violations found
///
/// # Errors
/// Returns error if source cannot be parsed
pub fn lint_panic_paths(source: &str, file: &str) -> Result<StateSyncReport, String> {
    let syntax = syn::parse_file(source).map_err(|e| format!("Parse error: {e}"))?;

    let lines = source.lines().count();
    let mut visitor = PanicPathVisitor::new(file.to_string(), source.to_string());

    visitor.visit_file(&syntax);

    Ok(visitor.into_report(lines))
}

/// Summary of panic path analysis
#[derive(Debug, Default)]
pub struct PanicPathSummary {
    /// Total unwrap() calls
    pub unwrap_count: usize,
    /// Total expect() calls
    pub expect_count: usize,
    /// Total panic!() macros
    pub panic_count: usize,
    /// Total unreachable!() macros
    pub unreachable_count: usize,
    /// Total todo!() macros
    pub todo_count: usize,
    /// Total unimplemented!() macros
    pub unimplemented_count: usize,
    /// Total index operations
    pub index_count: usize,
}

impl PanicPathSummary {
    /// Create summary from report
    #[must_use]
    pub fn from_report(report: &StateSyncReport) -> Self {
        let mut summary = Self::default();

        for error in &report.errors {
            match error.rule.as_str() {
                "WASM-PANIC-001" => summary.unwrap_count += 1,
                "WASM-PANIC-002" => summary.expect_count += 1,
                "WASM-PANIC-003" => summary.panic_count += 1,
                "WASM-PANIC-004" => summary.unreachable_count += 1,
                "WASM-PANIC-005" => summary.todo_count += 1,
                "WASM-PANIC-006" => summary.unimplemented_count += 1,
                "WASM-PANIC-007" => summary.index_count += 1,
                _ => {}
            }
        }

        summary
    }

    /// Total panic path count
    #[must_use]
    pub fn total(&self) -> usize {
        self.unwrap_count
            + self.expect_count
            + self.panic_count
            + self.unreachable_count
            + self.todo_count
            + self.unimplemented_count
            + self.index_count
    }

    /// Total error-level count (excludes warnings)
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.unwrap_count
            + self.expect_count
            + self.panic_count
            + self.todo_count
            + self.unimplemented_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_unwrap() {
        let source = r#"
            fn example() {
                let x = Some(5);
                let y = x.unwrap();
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        assert!(!report.errors.is_empty());
        assert!(report.errors.iter().any(|e| e.rule == "WASM-PANIC-001"));
    }

    #[test]
    fn test_detect_expect() {
        let source = r#"
            fn example() {
                let x = Some(5);
                let y = x.expect("should exist");
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        assert!(report.errors.iter().any(|e| e.rule == "WASM-PANIC-002"));
    }

    #[test]
    fn test_detect_panic_macro() {
        let source = r#"
            fn example() {
                panic!("something went wrong");
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        assert!(report.errors.iter().any(|e| e.rule == "WASM-PANIC-003"));
    }

    #[test]
    fn test_detect_unreachable() {
        let source = r#"
            fn example(x: bool) {
                if x {
                    return;
                }
                unreachable!();
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        assert!(report.errors.iter().any(|e| e.rule == "WASM-PANIC-004"));
        // unreachable should be a warning, not error
        let unreachable_error = report
            .errors
            .iter()
            .find(|e| e.rule == "WASM-PANIC-004")
            .unwrap();
        assert_eq!(unreachable_error.severity, LintSeverity::Warning);
    }

    #[test]
    fn test_detect_todo() {
        let source = r#"
            fn example() {
                todo!("implement this");
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        assert!(report.errors.iter().any(|e| e.rule == "WASM-PANIC-005"));
    }

    #[test]
    fn test_detect_unimplemented() {
        let source = r#"
            fn example() {
                unimplemented!();
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        assert!(report.errors.iter().any(|e| e.rule == "WASM-PANIC-006"));
    }

    #[test]
    fn test_detect_index() {
        let source = r#"
            fn example() {
                let arr = [1, 2, 3];
                let x = arr[0];
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        assert!(report.errors.iter().any(|e| e.rule == "WASM-PANIC-007"));
    }

    #[test]
    fn test_skip_in_test_module() {
        let source = r#"
            #[cfg(test)]
            mod tests {
                fn test_example() {
                    let x = Some(5);
                    let y = x.unwrap();  // Should be skipped
                }
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        // Should have no errors since we're in a test module
        assert!(
            report.errors.is_empty(),
            "Test modules should be skipped: {:?}",
            report.errors
        );
    }

    #[test]
    fn test_summary() {
        let source = r#"
            fn example() {
                let x = Some(5);
                x.unwrap();
                x.unwrap();
                x.expect("msg");
                panic!("oops");
                todo!();
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        let summary = PanicPathSummary::from_report(&report);

        assert_eq!(summary.unwrap_count, 2);
        assert_eq!(summary.expect_count, 1);
        assert_eq!(summary.panic_count, 1);
        assert_eq!(summary.todo_count, 1);
        assert_eq!(summary.total(), 5);
    }

    #[test]
    fn test_clean_code_passes() {
        let source = r#"
            fn example() -> Option<i32> {
                let x = Some(5);
                let y = x?;
                Some(y + 1)
            }

            fn example2() -> Result<i32, &'static str> {
                let x: Option<i32> = Some(5);
                let y = x.ok_or("missing")?;
                Ok(y + 1)
            }
        "#;

        let report = lint_panic_paths(source, "test.rs").expect("parse failed");
        // Clean code should have no panic path errors
        assert!(
            report.errors.is_empty(),
            "Clean code should pass: {:?}",
            report.errors
        );
    }
}
