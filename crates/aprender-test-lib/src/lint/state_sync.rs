//! State Synchronization Linter (PROBAR-SPEC-WASM-001)
//!
//! Static analysis to detect disconnected state patterns in WASM code.
//!
//! ## Motivation
//!
//! The WAPR-QA-REGRESSION-005 bug occurred because:
//! ```rust,ignore
//! // BUG: spawn() created LOCAL state_ptr, not using self.state_ptr
//! pub fn spawn(&mut self) {
//!     let state_ptr = Rc::new(RefCell::new(State::Spawning));  // LOCAL!
//!     let closure = move || {
//!         *state_ptr.borrow_mut() = State::Ready;  // Updates LOCAL, not self
//!     };
//! }
//! ```
//!
//! The fix was to clone from self:
//! ```rust,ignore
//! pub fn spawn(&mut self) {
//!     let state_ptr_clone = self.state_ptr.clone();  // Clone from self
//!     let closure = move || {
//!         *state_ptr_clone.borrow_mut() = State::Ready;  // Updates shared
//!     };
//! }
//! ```

use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Severity of lint errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintSeverity {
    /// Error: Must be fixed
    Error,
    /// Warning: Should be reviewed
    Warning,
    /// Info: Informational note
    Info,
}

impl std::fmt::Display for LintSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "error"),
            Self::Warning => write!(f, "warning"),
            Self::Info => write!(f, "info"),
        }
    }
}

/// A lint error with location and suggestion
#[derive(Debug, Clone)]
pub struct LintError {
    /// Rule identifier (e.g., "WASM-SS-001")
    pub rule: String,
    /// Human-readable message
    pub message: String,
    /// File path
    pub file: String,
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
    /// Severity level
    pub severity: LintSeverity,
    /// Suggested fix
    pub suggestion: Option<String>,
}

impl std::fmt::Display for LintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}[{}]: {} ({}:{}:{})",
            self.severity, self.rule, self.message, self.file, self.line, self.column
        )?;
        if let Some(suggestion) = &self.suggestion {
            write!(f, "\n  = help: {suggestion}")?;
        }
        Ok(())
    }
}

/// Result of linting
pub type LintResult = Result<StateSyncReport, String>;

/// Report from linting one or more files
#[derive(Debug, Default)]
pub struct StateSyncReport {
    /// All errors found
    pub errors: Vec<LintError>,
    /// Files analyzed
    pub files_analyzed: usize,
    /// Lines analyzed
    pub lines_analyzed: usize,
}

impl StateSyncReport {
    /// Check if there are any errors
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.errors
            .iter()
            .any(|e| e.severity == LintSeverity::Error)
    }

    /// Count errors by severity
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors
            .iter()
            .filter(|e| e.severity == LintSeverity::Error)
            .count()
    }

    /// Count warnings
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.errors
            .iter()
            .filter(|e| e.severity == LintSeverity::Warning)
            .count()
    }

    /// Merge another report into this one
    pub fn merge(&mut self, other: Self) {
        self.errors.extend(other.errors);
        self.files_analyzed += other.files_analyzed;
        self.lines_analyzed += other.lines_analyzed;
    }
}

/// State synchronization linter
///
/// Detects anti-patterns that cause state desync in WASM closures.
///
/// ## Rules
///
/// | Rule | Description | Severity |
/// |------|-------------|----------|
/// | WASM-SS-001 | Local Rc::new() in method with closure | Error |
/// | WASM-SS-002 | Both self.field and local reference exist | Warning |
/// | WASM-SS-005 | Missing self.*.clone() before closure | Warning |
/// | WASM-SS-006 | Type alias for Rc<RefCell<T>> used with ::new() | Warning |
/// | WASM-SS-007 | Function returning Rc<RefCell<T>> used in closure context | Warning |
#[derive(Debug)]
pub struct StateSyncLinter {
    /// Track local Rc variables per function
    local_rcs: HashMap<String, Vec<(String, usize)>>,
    /// Track closure captures
    closure_captures: HashSet<String>,
    /// Current file being analyzed
    current_file: String,
    /// Function/method names that create closures
    closure_creators: HashSet<String>,
    /// Type aliases that resolve to Rc<RefCell<T>>
    rc_type_aliases: HashSet<String>,
    /// Functions that return Rc<RefCell<T>>
    rc_returning_functions: HashSet<String>,
}

impl Default for StateSyncLinter {
    fn default() -> Self {
        Self::new()
    }
}

impl StateSyncLinter {
    /// Create a new linter
    #[must_use]
    pub fn new() -> Self {
        let mut closure_creators = HashSet::new();
        // Common patterns that create closures in WASM code
        closure_creators.insert("Closure::wrap".to_string());
        closure_creators.insert("Closure::once".to_string());
        closure_creators.insert("move ||".to_string());
        closure_creators.insert("move |".to_string());

        Self {
            local_rcs: HashMap::new(),
            closure_captures: HashSet::new(),
            current_file: String::new(),
            closure_creators,
            rc_type_aliases: HashSet::new(),
            rc_returning_functions: HashSet::new(),
        }
    }

    /// Lint a single file
    pub fn lint_file(&mut self, path: &Path) -> LintResult {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

        self.current_file = path.display().to_string();
        self.lint_source(&content)
    }

    /// Lint source code directly (uses AST-based analysis by default)
    ///
    /// This method first attempts AST-based analysis using `syn`, which is more
    /// accurate and handles edge cases like turbofish syntax. Falls back to
    /// text-based analysis if AST parsing fails.
    pub fn lint_source(&mut self, source: &str) -> LintResult {
        // Try AST-based analysis first (PROBAR-WASM-003)
        if let Ok(ast_report) = super::ast_visitor::lint_source_ast(source, &self.current_file) {
            // Merge AST findings with text-based for comprehensive coverage
            let mut report = ast_report;
            if let Ok(text_report) = self.lint_source_text_based(source) {
                // Only add text-based errors that aren't duplicates
                for error in text_report.errors {
                    if !report.errors.iter().any(|e| {
                        e.rule == error.rule && e.line == error.line && e.file == error.file
                    }) {
                        report.errors.push(error);
                    }
                }
            }
            return Ok(report);
        }

        // Fallback to text-based analysis
        self.lint_source_text_based(source)
    }

    /// Text-based lint analysis (legacy, for edge cases)
    fn lint_source_text_based(&mut self, source: &str) -> LintResult {
        let mut report = StateSyncReport {
            files_analyzed: 1,
            lines_analyzed: source.lines().count(),
            ..Default::default()
        };

        // Reset state
        self.local_rcs.clear();
        self.closure_captures.clear();
        self.rc_type_aliases.clear();
        self.rc_returning_functions.clear();

        // Pre-pass: Collect type aliases and function signatures
        self.collect_type_info(source, &mut report);

        // Pre-pass: Identify functions that contain closures
        let fns_with_closures = self.find_functions_with_closures(source);

        // Track context
        let mut current_fn: Option<String> = None;
        let mut fn_has_closure = false;
        let mut brace_depth = 0;
        let mut fn_start_depth = 0;

        for (line_num, line) in source.lines().enumerate() {
            let line_num = line_num + 1; // 1-indexed

            // Track brace depth
            brace_depth += line.matches('{').count();
            brace_depth = brace_depth.saturating_sub(line.matches('}').count());

            // Detect function start
            if let Some(fn_name) = self.detect_function_start(line) {
                current_fn = Some(fn_name);
                fn_start_depth = brace_depth;
                fn_has_closure = false;
                self.local_rcs.clear();
            }

            // Detect function end
            if current_fn.is_some() && brace_depth < fn_start_depth {
                current_fn = None;
            }

            // Check for closure patterns
            if self.line_creates_closure(line) {
                fn_has_closure = true;
            }

            // WASM-SS-001: Local Rc::new() in method with closure
            if let Some(var_name) = self.detect_local_rc_new(line) {
                let fn_name = current_fn
                    .clone()
                    .unwrap_or_else(|| "<unknown>".to_string());
                self.local_rcs
                    .entry(fn_name.clone())
                    .or_default()
                    .push((var_name.clone(), line_num));

                // If this function creates closures, this is suspicious
                let fn_has_any_closure = fn_has_closure
                    || fns_with_closures.contains(&fn_name)
                    || self.function_likely_creates_closure(&fn_name);
                if fn_has_any_closure {
                    report.errors.push(LintError {
                        rule: "WASM-SS-001".to_string(),
                        message: format!(
                            "Local `{var_name}` creates new Rc - if captured by closure, \
                             it will be disconnected from self"
                        ),
                        file: self.current_file.clone(),
                        line: line_num,
                        column: line.find(&var_name).unwrap_or(0) + 1,
                        severity: LintSeverity::Error,
                        suggestion: Some(format!(
                            "Use `let {var_name}_clone = self.{var_name}.clone()` instead"
                        )),
                    });
                }
            }

            // WASM-SS-006: Type alias ::new() pattern
            if let Some((alias_name, var_name)) = self.detect_type_alias_new(line) {
                if fn_has_closure
                    || self.function_likely_creates_closure(
                        current_fn.as_deref().unwrap_or("<unknown>"),
                    )
                {
                    report.errors.push(LintError {
                        rule: "WASM-SS-006".to_string(),
                        message: format!(
                            "Type alias `{alias_name}::new()` creates local Rc - \
                             may cause state desync if captured in closure"
                        ),
                        file: self.current_file.clone(),
                        line: line_num,
                        column: line.find(&var_name).unwrap_or(0) + 1,
                        severity: LintSeverity::Warning,
                        suggestion: Some(format!(
                            "Use `self.{var_name}.clone()` instead of `{alias_name}::new()`"
                        )),
                    });
                }
            }

            // WASM-SS-007: Helper function returning Rc pattern
            if let Some((fn_name_called, var_name)) = self.detect_rc_function_call(line) {
                if fn_has_closure
                    || self.function_likely_creates_closure(
                        current_fn.as_deref().unwrap_or("<unknown>"),
                    )
                {
                    report.errors.push(LintError {
                        rule: "WASM-SS-007".to_string(),
                        message: format!(
                            "Function `{fn_name_called}()` returns Rc - \
                             local assignment may cause state desync in closure"
                        ),
                        file: self.current_file.clone(),
                        line: line_num,
                        column: line.find(&var_name).unwrap_or(0) + 1,
                        severity: LintSeverity::Warning,
                        suggestion: Some(
                            "Clone from self instead of calling helper function".to_string(),
                        ),
                    });
                }
            }

            // WASM-SS-003: Closure captures local instead of self field
            if self.line_creates_closure(line) {
                // Check what variables are referenced in the closure context
                self.check_closure_captures(line, line_num, source, &mut report);
            }

            // WASM-SS-005: Check for missing self.*.clone() pattern
            if fn_has_closure && current_fn.is_some() {
                self.check_missing_self_clone(line, line_num, &mut report);
            }
        }

        Ok(report)
    }

    /// Pre-pass to collect type aliases and function signatures
    fn collect_type_info(&mut self, source: &str, report: &mut StateSyncReport) {
        for (line_num, line) in source.lines().enumerate() {
            let line_num = line_num + 1;
            let trimmed = line.trim();

            // Detect type aliases: type Foo = Rc<RefCell<...>>
            if trimmed.starts_with("type ") && trimmed.contains("Rc<") {
                if let Some(alias_name) = self.extract_type_alias_name(trimmed) {
                    self.rc_type_aliases.insert(alias_name.clone());
                    report.errors.push(LintError {
                        rule: "WASM-SS-006".to_string(),
                        message: format!(
                            "Type alias `{alias_name}` wraps Rc - usage with ::new() may cause state desync"
                        ),
                        file: self.current_file.clone(),
                        line: line_num,
                        column: 1,
                        severity: LintSeverity::Info,
                        suggestion: Some("Consider using self.field.clone() pattern instead".to_string()),
                    });
                }
            }

            // Detect functions returning Rc: fn foo() -> Rc<...>
            if trimmed.contains("fn ") && trimmed.contains("-> Rc<") {
                if let Some(fn_name) = self.detect_function_start(trimmed) {
                    self.rc_returning_functions.insert(fn_name.clone());
                    report.errors.push(LintError {
                        rule: "WASM-SS-007".to_string(),
                        message: format!(
                            "Function `{fn_name}` returns Rc - callers may create disconnected state"
                        ),
                        file: self.current_file.clone(),
                        line: line_num,
                        column: 1,
                        severity: LintSeverity::Info,
                        suggestion: Some("Document that callers should use self.field.clone() instead".to_string()),
                    });
                }
            }
        }
    }

    /// Extract type alias name from a type declaration
    fn extract_type_alias_name(&self, line: &str) -> Option<String> {
        // Pattern: type AliasName = ...
        let trimmed = line.trim();
        if !trimmed.starts_with("type ") {
            return None;
        }
        let after_type = &trimmed[5..];
        let name_end = after_type
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(after_type.len());
        let name = &after_type[..name_end];
        if !name.is_empty() {
            Some(name.to_string())
        } else {
            None
        }
    }

    /// Detect type alias ::new() pattern
    fn detect_type_alias_new(&self, line: &str) -> Option<(String, String)> {
        let trimmed = line.trim();

        // Look for patterns like: let var = AliasName::new(...)
        for alias in &self.rc_type_aliases {
            let pattern = format!("{alias}::new(");
            if trimmed.contains(&pattern) {
                // Extract variable name
                if let Some(after_let) = trimmed.strip_prefix("let ") {
                    let after_mut = after_let.strip_prefix("mut ").unwrap_or(after_let);
                    let name_end = after_mut
                        .find(|c: char| !c.is_alphanumeric() && c != '_')
                        .unwrap_or(after_mut.len());
                    let var_name = &after_mut[..name_end];
                    if !var_name.is_empty() {
                        return Some((alias.clone(), var_name.to_string()));
                    }
                }
            }
        }
        None
    }

    /// Detect helper function call returning Rc
    fn detect_rc_function_call(&self, line: &str) -> Option<(String, String)> {
        let trimmed = line.trim();

        // Look for patterns like: let var = Self::make_state() or self.make_state()
        for fn_name in &self.rc_returning_functions {
            // Check for Self::fn_name() or self.fn_name()
            let patterns = [
                format!("Self::{fn_name}("),
                format!("self.{fn_name}("),
                format!("{fn_name}("), // Direct call
            ];

            for pattern in &patterns {
                if trimmed.contains(pattern) {
                    // Extract variable name if it's an assignment
                    if let Some(after_let) = trimmed.strip_prefix("let ") {
                        let after_mut = after_let.strip_prefix("mut ").unwrap_or(after_let);
                        let name_end = after_mut
                            .find(|c: char| !c.is_alphanumeric() && c != '_')
                            .unwrap_or(after_mut.len());
                        let var_name = &after_mut[..name_end];
                        if !var_name.is_empty() {
                            return Some((fn_name.clone(), var_name.to_string()));
                        }
                    }
                }
            }
        }
        None
    }

    /// Detect function/method start, return function name
    fn detect_function_start(&self, line: &str) -> Option<String> {
        let trimmed = line.trim();

        // Match: pub fn name, fn name, pub async fn name, etc.
        if trimmed.contains("fn ")
            && (trimmed.starts_with("fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("pub(crate) fn ")
                || trimmed.starts_with("async fn ")
                || trimmed.starts_with("pub async fn "))
        {
            // Extract function name
            if let Some(fn_pos) = trimmed.find("fn ") {
                let after_fn = &trimmed[fn_pos + 3..];
                let name_end = after_fn
                    .find(|c: char| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(after_fn.len());
                let name = &after_fn[..name_end];
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
        None
    }

    /// Check if line creates a closure
    fn line_creates_closure(&self, line: &str) -> bool {
        let trimmed = line.trim();
        for pattern in &self.closure_creators {
            if trimmed.contains(pattern.as_str()) {
                return true;
            }
        }
        false
    }

    /// Pre-pass to identify which functions contain closures
    fn find_functions_with_closures(&self, source: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut current_fn: Option<String> = None;
        let mut brace_depth = 0;
        let mut fn_start_depth = 0;

        for line in source.lines() {
            brace_depth += line.matches('{').count();
            brace_depth = brace_depth.saturating_sub(line.matches('}').count());

            if let Some(fn_name) = self.detect_function_start(line) {
                current_fn = Some(fn_name);
                fn_start_depth = brace_depth;
            }

            if current_fn.is_some() && brace_depth < fn_start_depth {
                current_fn = None;
            }

            if self.line_creates_closure(line) {
                if let Some(ref fn_name) = current_fn {
                    result.insert(fn_name.clone());
                }
            }
        }

        result
    }

    /// Detect local Rc::new() pattern
    fn detect_local_rc_new(&self, line: &str) -> Option<String> {
        let trimmed = line.trim();

        // Pattern: let var_name = Rc::new(RefCell::new(
        // Pattern: let var_name = Rc::new(
        if let Some(after_let) = trimmed.strip_prefix("let ") {
            if trimmed.contains("Rc::new(") {
                // Handle: let var_name = or let mut var_name =
                let after_mut = after_let.strip_prefix("mut ").unwrap_or(after_let);

                let name_end = after_mut
                    .find(|c: char| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(after_mut.len());
                let name = &after_mut[..name_end];

                // Exclude patterns like `let state_ptr_clone = self.state_ptr.clone()`
                // which are the CORRECT pattern
                if !line.contains(".clone()") && !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
        None
    }

    /// Check if function likely creates closures (heuristic)
    fn function_likely_creates_closure(&self, fn_name: &str) -> bool {
        // Common function names that typically create closures
        let closure_fn_names = [
            "spawn",
            "start",
            "on_message",
            "on_click",
            "on_event",
            "set_callback",
            "register",
            "subscribe",
            "listen",
        ];
        closure_fn_names.iter().any(|&n| fn_name.contains(n))
    }

    /// Check closure captures for anti-patterns
    fn check_closure_captures(
        &self,
        _line: &str,
        line_num: usize,
        source: &str,
        report: &mut StateSyncReport,
    ) {
        // Look at context around closure creation
        let lines: Vec<&str> = source.lines().collect();
        let start = line_num.saturating_sub(10);
        let end = (line_num + 10).min(lines.len());

        let context = &lines[start..end];

        // Check if we have a local Rc that's not from self.*.clone()
        for line in context {
            if line.contains("let ") && line.contains("Rc::new(") && !line.contains(".clone()") {
                // Already reported by WASM-SS-001, skip
                continue;
            }

            // WASM-SS-002: Both self.field and local_clone exist
            if line.contains("self.state") && line.contains("state_ptr") {
                // This is potentially a desync pattern
                report.errors.push(LintError {
                    rule: "WASM-SS-002".to_string(),
                    message: "Potential state desync: both `self.state` and local \
                              `state_ptr` reference exist"
                        .to_string(),
                    file: self.current_file.clone(),
                    line: line_num,
                    column: 1,
                    severity: LintSeverity::Warning,
                    suggestion: Some(
                        "Ensure closure uses `self.state_ptr.clone()`, not a local Rc".to_string(),
                    ),
                });
            }
        }
    }

    /// Check for missing self.*.clone() before closure
    fn check_missing_self_clone(&self, line: &str, line_num: usize, report: &mut StateSyncReport) {
        // Pattern 1: Closure::wrap or move || with state_ptr reference
        if self.line_creates_closure(line)
            && line.contains("state_ptr")
            && !line.contains("state_ptr_clone")
        {
            report.errors.push(LintError {
                rule: "WASM-SS-005".to_string(),
                message: "Closure may capture local state - ensure \
                          `self.state_ptr.clone()` is used"
                    .to_string(),
                file: self.current_file.clone(),
                line: line_num,
                column: 1,
                severity: LintSeverity::Warning,
                suggestion: Some(
                    "Add `let state_ptr_clone = self.state_ptr.clone();` before closure"
                        .to_string(),
                ),
            });
            return;
        }

        // Pattern 2: Usage of state_ptr inside a function with closures (not cloned from self)
        // Detects: state_ptr.borrow_mut() or state_ptr.borrow() when state_ptr isn't cloned
        if line.contains("state_ptr.borrow") && !line.contains("self.") && !line.contains("_clone")
        {
            report.errors.push(LintError {
                rule: "WASM-SS-005".to_string(),
                message: "Using `state_ptr` directly - may be disconnected from self".to_string(),
                file: self.current_file.clone(),
                line: line_num,
                column: 1,
                severity: LintSeverity::Warning,
                suggestion: Some(
                    "Use `let state_ptr_clone = self.state_ptr.clone();` before closure"
                        .to_string(),
                ),
            });
        }
    }

    /// Lint all Rust files in a directory
    pub fn lint_directory(&mut self, dir: &Path) -> LintResult {
        fn visit_dir(linter: &mut StateSyncLinter, dir: &Path, report: &mut StateSyncReport) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        // Skip target, .git, etc.
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if !name.starts_with('.') && name != "target" {
                            visit_dir(linter, &path, report);
                        }
                    } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
                        if let Ok(file_report) = linter.lint_file(&path) {
                            report.merge(file_report);
                        }
                    }
                }
            }
        }

        let mut report = StateSyncReport::default();
        visit_dir(self, dir, &mut report);
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_local_rc_new() {
        let linter = StateSyncLinter::new();

        // Should detect
        assert!(linter
            .detect_local_rc_new("let state_ptr = Rc::new(RefCell::new(State::Init));")
            .is_some());
        assert!(linter
            .detect_local_rc_new("    let foo = Rc::new(42);")
            .is_some());

        // Should NOT detect (correct pattern - cloning from self)
        assert!(linter
            .detect_local_rc_new("let state_ptr_clone = self.state_ptr.clone();")
            .is_none());
    }

    #[test]
    fn test_detect_function_start() {
        let linter = StateSyncLinter::new();

        assert_eq!(
            linter.detect_function_start("fn foo() {"),
            Some("foo".to_string())
        );
        assert_eq!(
            linter.detect_function_start("pub fn spawn(&mut self) {"),
            Some("spawn".to_string())
        );
        assert_eq!(
            linter.detect_function_start("pub async fn start() {"),
            Some("start".to_string())
        );
        assert_eq!(linter.detect_function_start("// fn not_a_function"), None);
    }

    #[test]
    fn test_line_creates_closure() {
        let linter = StateSyncLinter::new();

        assert!(linter.line_creates_closure("let f = move || { do_stuff(); };"));
        assert!(linter.line_creates_closure("let cb = Closure::wrap(Box::new(move |e| {}));"));
        assert!(!linter.line_creates_closure("fn regular_function() {}"));
    }

    #[test]
    fn test_lint_buggy_code() {
        let mut linter = StateSyncLinter::new();

        let buggy_code = r#"
impl WorkerManager {
    pub fn spawn(&mut self) {
        // BUG: Creates local Rc, not from self
        let state_ptr = Rc::new(RefCell::new(ManagerState::Spawning));

        let on_message = Closure::wrap(Box::new(move |event| {
            *state_ptr.borrow_mut() = ManagerState::Ready;
        }));
    }
}
"#;

        let report = linter.lint_source(buggy_code).expect("lint failed");

        // Should detect WASM-SS-001
        assert!(!report.errors.is_empty(), "Expected lint errors");
        assert!(
            report.errors.iter().any(|e| e.rule == "WASM-SS-001"),
            "Expected WASM-SS-001 error"
        );
    }

    #[test]
    fn test_lint_correct_code() {
        let mut linter = StateSyncLinter::new();

        let correct_code = r#"
impl WorkerManager {
    pub fn spawn(&mut self) {
        // CORRECT: Clone from self
        let state_ptr_clone = self.state_ptr.clone();

        let on_message = Closure::wrap(Box::new(move |event| {
            *state_ptr_clone.borrow_mut() = ManagerState::Ready;
        }));
    }
}
"#;

        let report = linter.lint_source(correct_code).expect("lint failed");

        // Should NOT detect WASM-SS-001 (the correct pattern doesn't trigger it)
        let ss001_errors: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.rule == "WASM-SS-001")
            .collect();
        assert!(
            ss001_errors.is_empty(),
            "Should not report WASM-SS-001 for correct pattern"
        );
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(LintSeverity::Error.to_string(), "error");
        assert_eq!(LintSeverity::Warning.to_string(), "warning");
        assert_eq!(LintSeverity::Info.to_string(), "info");
    }

    #[test]
    fn test_lint_error_display() {
        let err = LintError {
            rule: "WASM-SS-001".to_string(),
            message: "Local Rc captured".to_string(),
            file: "src/lib.rs".to_string(),
            line: 42,
            column: 13,
            severity: LintSeverity::Error,
            suggestion: Some("Use self.state_ptr.clone()".to_string()),
        };

        let display = err.to_string();
        assert!(display.contains("WASM-SS-001"));
        assert!(display.contains("Local Rc captured"));
        assert!(display.contains("src/lib.rs:42:13"));
        assert!(display.contains("self.state_ptr.clone()"));
    }

    #[test]
    fn test_report_counts() {
        let mut report = StateSyncReport::default();

        report.errors.push(LintError {
            rule: "WASM-SS-001".to_string(),
            message: "test".to_string(),
            file: "test.rs".to_string(),
            line: 1,
            column: 1,
            severity: LintSeverity::Error,
            suggestion: None,
        });

        report.errors.push(LintError {
            rule: "WASM-SS-002".to_string(),
            message: "test".to_string(),
            file: "test.rs".to_string(),
            line: 2,
            column: 1,
            severity: LintSeverity::Warning,
            suggestion: None,
        });

        assert_eq!(report.error_count(), 1);
        assert_eq!(report.warning_count(), 1);
        assert!(report.has_errors());
    }

    // Additional tests for improved coverage

    #[test]
    fn test_lint_error_display_without_suggestion() {
        let err = LintError {
            rule: "WASM-SS-002".to_string(),
            message: "Potential desync".to_string(),
            file: "src/worker.rs".to_string(),
            line: 10,
            column: 5,
            severity: LintSeverity::Warning,
            suggestion: None,
        };

        let display = err.to_string();
        assert!(display.contains("WASM-SS-002"));
        assert!(display.contains("Potential desync"));
        assert!(display.contains("src/worker.rs:10:5"));
        // Should not contain "help:" when no suggestion
        assert!(!display.contains("help:"));
    }

    #[test]
    fn test_report_merge() {
        let mut report1 = StateSyncReport {
            errors: vec![LintError {
                rule: "WASM-SS-001".to_string(),
                message: "error1".to_string(),
                file: "file1.rs".to_string(),
                line: 1,
                column: 1,
                severity: LintSeverity::Error,
                suggestion: None,
            }],
            files_analyzed: 1,
            lines_analyzed: 100,
        };

        let report2 = StateSyncReport {
            errors: vec![LintError {
                rule: "WASM-SS-002".to_string(),
                message: "error2".to_string(),
                file: "file2.rs".to_string(),
                line: 2,
                column: 1,
                severity: LintSeverity::Warning,
                suggestion: None,
            }],
            files_analyzed: 2,
            lines_analyzed: 200,
        };

        report1.merge(report2);

        assert_eq!(report1.errors.len(), 2);
        assert_eq!(report1.files_analyzed, 3);
        assert_eq!(report1.lines_analyzed, 300);
    }

    #[test]
    fn test_report_no_errors() {
        let report = StateSyncReport::default();
        assert!(!report.has_errors());
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn test_report_only_warnings_no_errors() {
        let mut report = StateSyncReport::default();
        report.errors.push(LintError {
            rule: "WASM-SS-002".to_string(),
            message: "warning".to_string(),
            file: "test.rs".to_string(),
            line: 1,
            column: 1,
            severity: LintSeverity::Warning,
            suggestion: None,
        });
        report.errors.push(LintError {
            rule: "WASM-SS-006".to_string(),
            message: "warning2".to_string(),
            file: "test.rs".to_string(),
            line: 2,
            column: 1,
            severity: LintSeverity::Warning,
            suggestion: None,
        });

        assert!(!report.has_errors());
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 2);
    }

    #[test]
    fn test_extract_type_alias_name() {
        let linter = StateSyncLinter::new();

        // Valid type alias
        assert_eq!(
            linter.extract_type_alias_name("type StatePtr = Rc<RefCell<State>>;"),
            Some("StatePtr".to_string())
        );

        // Type alias with underscores
        assert_eq!(
            linter.extract_type_alias_name("type My_State_Ptr = Rc<RefCell<State>>;"),
            Some("My_State_Ptr".to_string())
        );

        // Not a type declaration
        assert_eq!(linter.extract_type_alias_name("let x = 5;"), None);

        // Empty after type
        assert_eq!(linter.extract_type_alias_name("type "), None);

        // Type with generic
        assert_eq!(
            linter.extract_type_alias_name("type Handler<T> = Rc<RefCell<T>>;"),
            Some("Handler".to_string())
        );
    }

    #[test]
    fn test_detect_type_alias_new_pattern() {
        let mut linter = StateSyncLinter::new();
        linter.rc_type_aliases.insert("StatePtr".to_string());

        // Should detect type alias ::new()
        let result = linter.detect_type_alias_new("let state = StatePtr::new(Default::default());");
        assert!(result.is_some());
        let (alias, var) = result.unwrap();
        assert_eq!(alias, "StatePtr");
        assert_eq!(var, "state");

        // Should detect with mut
        let result =
            linter.detect_type_alias_new("let mut state = StatePtr::new(Default::default());");
        assert!(result.is_some());
        let (alias, var) = result.unwrap();
        assert_eq!(alias, "StatePtr");
        assert_eq!(var, "state");

        // Should not detect non-alias
        let result = linter.detect_type_alias_new("let x = Rc::new(5);");
        assert!(result.is_none());

        // Should not detect without let
        let result = linter.detect_type_alias_new("StatePtr::new(Default::default());");
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_rc_function_call() {
        let mut linter = StateSyncLinter::new();
        linter
            .rc_returning_functions
            .insert("make_state".to_string());

        // Self:: pattern
        let result = linter.detect_rc_function_call("let state = Self::make_state();");
        assert!(result.is_some());
        let (fn_name, var) = result.unwrap();
        assert_eq!(fn_name, "make_state");
        assert_eq!(var, "state");

        // self. pattern
        let result = linter.detect_rc_function_call("let state = self.make_state();");
        assert!(result.is_some());

        // Direct call pattern
        let result = linter.detect_rc_function_call("let state = make_state();");
        assert!(result.is_some());

        // With mut
        let result = linter.detect_rc_function_call("let mut state = Self::make_state();");
        assert!(result.is_some());

        // Non-matching function
        let result = linter.detect_rc_function_call("let x = other_func();");
        assert!(result.is_none());

        // No assignment
        let result = linter.detect_rc_function_call("Self::make_state();");
        assert!(result.is_none());
    }

    #[test]
    fn test_function_likely_creates_closure() {
        let linter = StateSyncLinter::new();

        // Closure-likely function names
        assert!(linter.function_likely_creates_closure("spawn"));
        assert!(linter.function_likely_creates_closure("start"));
        assert!(linter.function_likely_creates_closure("on_message"));
        assert!(linter.function_likely_creates_closure("on_click"));
        assert!(linter.function_likely_creates_closure("on_event"));
        assert!(linter.function_likely_creates_closure("set_callback"));
        assert!(linter.function_likely_creates_closure("register"));
        assert!(linter.function_likely_creates_closure("subscribe"));
        assert!(linter.function_likely_creates_closure("listen"));

        // Names containing closure patterns
        assert!(linter.function_likely_creates_closure("spawn_worker"));
        assert!(linter.function_likely_creates_closure("do_spawn"));

        // Non-closure function names
        assert!(!linter.function_likely_creates_closure("calculate"));
        assert!(!linter.function_likely_creates_closure("get_value"));
        assert!(!linter.function_likely_creates_closure("process"));
    }

    #[test]
    fn test_detect_function_start_pub_crate() {
        let linter = StateSyncLinter::new();

        // pub(crate) fn
        assert_eq!(
            linter.detect_function_start("pub(crate) fn internal_func() {"),
            Some("internal_func".to_string())
        );

        // Just fn
        assert_eq!(
            linter.detect_function_start("    fn helper() {"),
            Some("helper".to_string())
        );

        // async fn
        assert_eq!(
            linter.detect_function_start("async fn async_work() {"),
            Some("async_work".to_string())
        );

        // Not a function (impl block)
        assert_eq!(linter.detect_function_start("impl Foo {"), None);

        // Not a function (closure)
        assert_eq!(linter.detect_function_start("let f = || {};"), None);
    }

    #[test]
    fn test_detect_local_rc_new_edge_cases() {
        let linter = StateSyncLinter::new();

        // With mut
        assert_eq!(
            linter.detect_local_rc_new("let mut counter = Rc::new(0);"),
            Some("counter".to_string())
        );

        // No let keyword
        assert!(linter
            .detect_local_rc_new("counter = Rc::new(0);")
            .is_none());

        // With clone (correct pattern)
        assert!(linter
            .detect_local_rc_new("let ptr = self.state.clone();")
            .is_none());

        // Nested in expression - should still detect due to simple pattern matching
        assert!(linter
            .detect_local_rc_new("    let x = Rc::new(RefCell::new(vec![]));")
            .is_some());
    }

    #[test]
    fn test_lint_type_alias_detection() {
        let mut linter = StateSyncLinter::new();

        let code_with_type_alias = r#"
type StatePtr = Rc<RefCell<State>>;

impl Worker {
    pub fn spawn(&mut self) {
        let state = StatePtr::new(State::default());
        let closure = move || {
            state.borrow_mut().update();
        };
    }
}
"#;

        let report = linter
            .lint_source(code_with_type_alias)
            .expect("lint failed");

        // Should detect WASM-SS-006 for type alias
        assert!(
            report.errors.iter().any(|e| e.rule == "WASM-SS-006"),
            "Expected WASM-SS-006 for type alias"
        );
    }

    #[test]
    fn test_lint_rc_returning_function() {
        let mut linter = StateSyncLinter::new();

        let code_with_rc_fn = r#"
fn make_state() -> Rc<RefCell<State>> {
    Rc::new(RefCell::new(State::default()))
}

impl Worker {
    pub fn spawn(&mut self) {
        let state = make_state();
        let closure = move || {
            state.borrow_mut().update();
        };
    }
}
"#;

        let report = linter.lint_source(code_with_rc_fn).expect("lint failed");

        // Should detect WASM-SS-007 for function returning Rc
        assert!(
            report.errors.iter().any(|e| e.rule == "WASM-SS-007"),
            "Expected WASM-SS-007 for Rc-returning function"
        );
    }

    #[test]
    fn test_lint_wasm_ss_005_missing_clone() {
        let mut linter = StateSyncLinter::new();

        let code_with_missing_clone = r#"
impl Worker {
    pub fn process(&mut self) {
        let closure = move || {
            // Uses state_ptr directly without clone from self
            state_ptr.borrow_mut().process();
        };
    }
}
"#;

        let report = linter
            .lint_source(code_with_missing_clone)
            .expect("lint failed");

        // Should detect WASM-SS-005
        assert!(
            report.errors.iter().any(|e| e.rule == "WASM-SS-005"),
            "Expected WASM-SS-005 for missing self clone"
        );
    }

    #[test]
    fn test_lint_wasm_ss_002_desync_pattern() {
        let mut linter = StateSyncLinter::new();

        let code_with_desync = r#"
impl Worker {
    pub fn spawn(&mut self) {
        // Both self.state and state_ptr exist - potential desync
        let state_ptr = Rc::new(RefCell::new(self.state.clone()));
        let closure = move || {
            state_ptr.borrow_mut().update();
        };
    }
}
"#;

        let report = linter.lint_source(code_with_desync).expect("lint failed");

        // Should detect some error (WASM-SS-001 for local Rc::new at minimum)
        assert!(
            !report.errors.is_empty(),
            "Expected lint errors for desync pattern"
        );
    }

    #[test]
    fn test_lint_empty_source() {
        let mut linter = StateSyncLinter::new();
        let report = linter.lint_source("").expect("lint failed");
        assert!(report.errors.is_empty());
        assert_eq!(report.files_analyzed, 1);
        assert_eq!(report.lines_analyzed, 0);
    }

    #[test]
    fn test_lint_source_with_no_functions() {
        let mut linter = StateSyncLinter::new();

        let code = r#"
// Just constants and types
const MAX: usize = 100;
type MyType = Vec<u32>;
"#;

        let report = linter.lint_source(code).expect("lint failed");
        // Should have no WASM-SS-001 errors (no functions with closures)
        assert!(
            !report.errors.iter().any(|e| e.rule == "WASM-SS-001"),
            "Should not report WASM-SS-001 for code without functions"
        );
    }

    #[test]
    fn test_lint_function_without_closure() {
        let mut linter = StateSyncLinter::new();

        let code = r#"
impl Calculator {
    pub fn add(&self, a: i32, b: i32) -> i32 {
        let result = Rc::new(a + b);
        *result
    }
}
"#;

        let report = linter.lint_source(code).expect("lint failed");
        // Should not detect WASM-SS-001 (no closure in function)
        assert!(
            !report.errors.iter().any(|e| e.rule == "WASM-SS-001"),
            "Should not report WASM-SS-001 for function without closure"
        );
    }

    #[test]
    fn test_lint_closure_with_move_pipe() {
        let linter = StateSyncLinter::new();

        // move |x|
        assert!(linter.line_creates_closure("let f = move |x| x + 1;"));
        // move ||
        assert!(linter.line_creates_closure("let f = move || println!(\"hi\");"));
        // Closure::once
        assert!(linter.line_creates_closure("let cb = Closure::once(Box::new(|| {}));"));
    }

    #[test]
    fn test_lint_brace_depth_tracking() {
        let mut linter = StateSyncLinter::new();

        // Code with nested braces
        let code = r#"
impl Outer {
    pub fn outer_fn(&mut self) {
        {
            let inner_scope = Rc::new(RefCell::new(0));
        }
        // After inner scope closes, we're back in outer_fn
        let closure = move || {};
    }
}
"#;

        let report = linter.lint_source(code).expect("lint failed");
        // This tests that brace depth tracking works correctly
        assert!(report.lines_analyzed > 0);
    }

    #[test]
    fn test_lint_multiple_functions() {
        let mut linter = StateSyncLinter::new();

        let code = r#"
impl Multi {
    pub fn first(&mut self) {
        let state = Rc::new(RefCell::new(0));
        let closure = move || {};
    }

    pub fn second(&mut self) {
        let state_clone = self.state.clone();
        let closure = move || {};
    }
}
"#;

        let report = linter.lint_source(code).expect("lint failed");
        // Should detect error in first function, not in second
        let ss001_count = report
            .errors
            .iter()
            .filter(|e| e.rule == "WASM-SS-001")
            .count();
        assert!(ss001_count >= 1, "Expected at least one WASM-SS-001 error");
    }

    #[test]
    fn test_lint_directory_with_tempdir() {
        use std::io::Write;

        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let rs_file_path = temp_dir.path().join("test.rs");

        let code = r#"
impl Test {
    pub fn spawn(&mut self) {
        let state = Rc::new(RefCell::new(0));
        let closure = move || {};
    }
}
"#;

        std::fs::File::create(&rs_file_path)
            .expect("Failed to create file")
            .write_all(code.as_bytes())
            .expect("Failed to write file");

        let mut linter = StateSyncLinter::new();
        let report = linter
            .lint_directory(temp_dir.path())
            .expect("lint_directory failed");

        assert_eq!(report.files_analyzed, 1);
        assert!(report.lines_analyzed > 0);
    }

    #[test]
    fn test_lint_directory_skips_hidden_and_target() {
        use std::io::Write;

        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

        // Create .hidden directory with a file
        let hidden_dir = temp_dir.path().join(".hidden");
        std::fs::create_dir(&hidden_dir).expect("Failed to create .hidden dir");
        let hidden_file = hidden_dir.join("hidden.rs");
        std::fs::File::create(&hidden_file)
            .expect("Failed to create hidden file")
            .write_all(b"fn hidden() {}")
            .expect("Failed to write");

        // Create target directory with a file
        let target_dir = temp_dir.path().join("target");
        std::fs::create_dir(&target_dir).expect("Failed to create target dir");
        let target_file = target_dir.join("generated.rs");
        std::fs::File::create(&target_file)
            .expect("Failed to create target file")
            .write_all(b"fn generated() {}")
            .expect("Failed to write");

        // Create a regular file
        let regular_file = temp_dir.path().join("src.rs");
        std::fs::File::create(&regular_file)
            .expect("Failed to create regular file")
            .write_all(b"fn regular() {}")
            .expect("Failed to write");

        let mut linter = StateSyncLinter::new();
        let report = linter
            .lint_directory(temp_dir.path())
            .expect("lint_directory failed");

        // Should only analyze the regular file, not hidden or target
        assert_eq!(report.files_analyzed, 1);
    }

    #[test]
    fn test_lint_file_not_found() {
        let mut linter = StateSyncLinter::new();
        let result = linter.lint_file(std::path::Path::new("/nonexistent/path/file.rs"));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Failed to read /nonexistent/path/file.rs"));
    }

    #[test]
    fn test_lint_file_success() {
        use std::io::Write;

        let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
        let rs_file = temp_dir.path().join("test.rs");

        let code = "fn test() { let x = 1; }";
        std::fs::File::create(&rs_file)
            .expect("Failed to create file")
            .write_all(code.as_bytes())
            .expect("Failed to write");

        let mut linter = StateSyncLinter::new();
        let report = linter.lint_file(&rs_file).expect("lint_file failed");

        assert_eq!(report.files_analyzed, 1);
        assert_eq!(report.lines_analyzed, 1);
    }

    #[test]
    fn test_collect_type_info_function_returning_rc() {
        let mut linter = StateSyncLinter::new();
        let mut report = StateSyncReport::default();

        let code = r#"
fn create_state() -> Rc<RefCell<State>> {
    Rc::new(RefCell::new(State::default()))
}
"#;

        linter.collect_type_info(code, &mut report);

        assert!(linter.rc_returning_functions.contains("create_state"));
        assert!(report.errors.iter().any(|e| e.rule == "WASM-SS-007"));
    }

    #[test]
    fn test_collect_type_info_type_alias() {
        let mut linter = StateSyncLinter::new();
        let mut report = StateSyncReport::default();

        let code = r#"
type SharedState = Rc<RefCell<State>>;
"#;

        linter.collect_type_info(code, &mut report);

        assert!(linter.rc_type_aliases.contains("SharedState"));
        assert!(report.errors.iter().any(|e| e.rule == "WASM-SS-006"));
    }

    #[test]
    fn test_lint_source_text_based_directly() {
        let mut linter = StateSyncLinter::new();
        linter.current_file = "test.rs".to_string();

        let code = r#"
impl Worker {
    pub fn on_event(&mut self) {
        let state = Rc::new(RefCell::new(0));
        let cb = Closure::wrap(Box::new(move || {}));
    }
}
"#;

        let report = linter
            .lint_source_text_based(code)
            .expect("lint_source_text_based failed");

        assert!(report.files_analyzed == 1);
        assert!(report.lines_analyzed > 0);
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(LintSeverity::Error, LintSeverity::Error);
        assert_eq!(LintSeverity::Warning, LintSeverity::Warning);
        assert_eq!(LintSeverity::Info, LintSeverity::Info);
        assert_ne!(LintSeverity::Error, LintSeverity::Warning);
        assert_ne!(LintSeverity::Warning, LintSeverity::Info);
    }

    #[test]
    fn test_lint_error_clone() {
        let err = LintError {
            rule: "TEST-001".to_string(),
            message: "test message".to_string(),
            file: "test.rs".to_string(),
            line: 1,
            column: 1,
            severity: LintSeverity::Error,
            suggestion: Some("fix it".to_string()),
        };

        let cloned = err.clone();
        assert_eq!(err.rule, cloned.rule);
        assert_eq!(err.message, cloned.message);
        assert_eq!(err.file, cloned.file);
        assert_eq!(err.line, cloned.line);
        assert_eq!(err.column, cloned.column);
        assert_eq!(err.severity, cloned.severity);
        assert_eq!(err.suggestion, cloned.suggestion);
    }

    #[test]
    fn test_linter_default() {
        let linter = StateSyncLinter::default();
        // Default should be same as new()
        assert!(linter.closure_creators.contains("Closure::wrap"));
        assert!(linter.closure_creators.contains("move ||"));
    }
}
