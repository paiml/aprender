//! Structured compiler diagnostic types.
//!
//! Following Yasunaga & Liang (2020), structured diagnostics enable
//! the program-feedback graph construction needed for GNN-based learning.

use super::ErrorCode;

/// Structured compiler diagnostic.
///
/// Per Yasunaga & Liang (2020), the program-feedback graph requires
/// structured diagnostics with:
/// - Error code for classification
/// - Span for localization
/// - Expected/found types for type errors
/// - Suggestions for fix generation
#[derive(Debug, Clone)]
pub struct CompilerDiagnostic {
    /// Unique error code (e.g., E0308, E0382 for rustc)
    pub code: ErrorCode,
    /// Severity level
    pub severity: DiagnosticSeverity,
    /// Human-readable message
    pub message: String,
    /// Primary source span
    pub span: SourceSpan,
    /// Additional labeled spans
    pub labels: Vec<DiagnosticLabel>,
    /// Compiler-suggested fixes (if any)
    pub suggestions: Vec<CompilerSuggestion>,
    /// Expected type (for type errors)
    pub expected: Option<TypeInfo>,
    /// Actual type found (for type errors)
    pub found: Option<TypeInfo>,
    /// Related notes
    pub notes: Vec<String>,
}

impl CompilerDiagnostic {
    /// Create a new diagnostic with required fields.
    #[must_use]
    pub fn new(
        code: ErrorCode,
        severity: DiagnosticSeverity,
        message: &str,
        span: SourceSpan,
    ) -> Self {
        Self {
            code,
            severity,
            message: message.to_string(),
            span,
            labels: Vec::new(),
            suggestions: Vec::new(),
            expected: None,
            found: None,
            notes: Vec::new(),
        }
    }

    /// Add a label to the diagnostic.
    #[must_use]
    pub fn with_label(mut self, label: DiagnosticLabel) -> Self {
        self.labels.push(label);
        self
    }

    /// Add a suggestion to the diagnostic.
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: CompilerSuggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Set expected type.
    #[must_use]
    pub fn with_expected(mut self, expected: TypeInfo) -> Self {
        self.expected = Some(expected);
        self
    }

    /// Set found type.
    #[must_use]
    pub fn with_found(mut self, found: TypeInfo) -> Self {
        self.found = Some(found);
        self
    }

    /// Add a note.
    #[must_use]
    pub fn with_note(mut self, note: &str) -> Self {
        self.notes.push(note.to_string());
        self
    }

    /// Check if this is a type mismatch error.
    #[must_use]
    pub fn is_type_error(&self) -> bool {
        self.expected.is_some() && self.found.is_some()
    }

    /// Check if compiler suggestions are available.
    #[must_use]
    pub fn has_suggestions(&self) -> bool {
        !self.suggestions.is_empty()
    }

    /// Get machine-applicable suggestions only.
    #[must_use]
    pub fn machine_applicable_suggestions(&self) -> Vec<&CompilerSuggestion> {
        self.suggestions
            .iter()
            .filter(|s| s.applicability == SuggestionApplicability::MachineApplicable)
            .collect()
    }
}

/// Diagnostic severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticSeverity {
    /// Fatal error preventing compilation
    Error,
    /// Warning (compilation may succeed)
    Warning,
    /// Informational note
    Note,
    /// Help message
    Help,
}

impl DiagnosticSeverity {
    /// Check if this severity prevents compilation.
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        matches!(self, DiagnosticSeverity::Error)
    }
}

/// Source code span for error location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSpan {
    /// File path (may be empty for in-memory sources)
    pub file: String,
    /// Start line (1-indexed)
    pub line_start: usize,
    /// End line (1-indexed)
    pub line_end: usize,
    /// Start column (1-indexed)
    pub column_start: usize,
    /// End column (1-indexed)
    pub column_end: usize,
    /// Byte offset start
    pub byte_start: usize,
    /// Byte offset end
    pub byte_end: usize,
}

impl SourceSpan {
    /// Create a new source span.
    #[must_use]
    pub fn new(
        file: &str,
        line_start: usize,
        line_end: usize,
        column_start: usize,
        column_end: usize,
    ) -> Self {
        Self {
            file: file.to_string(),
            line_start,
            line_end,
            column_start,
            column_end,
            byte_start: 0,
            byte_end: 0,
        }
    }

    /// Create a single-line span.
    #[must_use]
    pub fn single_line(file: &str, line: usize, column_start: usize, column_end: usize) -> Self {
        Self::new(file, line, line, column_start, column_end)
    }

    /// Create a point span (single character).
    #[must_use]
    pub fn point(file: &str, line: usize, column: usize) -> Self {
        Self::new(file, line, line, column, column + 1)
    }

    /// Check if this span is a single line.
    #[must_use]
    pub fn is_single_line(&self) -> bool {
        self.line_start == self.line_end
    }

    /// Get span length in characters (approximate).
    #[must_use]
    pub fn len(&self) -> usize {
        if self.byte_end > self.byte_start {
            self.byte_end - self.byte_start
        } else if self.is_single_line() {
            self.column_end.saturating_sub(self.column_start)
        } else {
            // Multi-line, estimate
            (self.line_end - self.line_start + 1) * 40
        }
    }

    /// Check if span is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if two spans overlap.
    #[must_use]
    pub fn overlaps(&self, other: &SourceSpan) -> bool {
        self.file == other.file
            && !(self.line_end < other.line_start || other.line_end < self.line_start)
    }
}

impl Default for SourceSpan {
    fn default() -> Self {
        Self {
            file: String::new(),
            line_start: 1,
            line_end: 1,
            column_start: 1,
            column_end: 1,
            byte_start: 0,
            byte_end: 0,
        }
    }
}

/// Additional labeled span within a diagnostic.
#[derive(Debug, Clone)]
pub struct DiagnosticLabel {
    /// Label text
    pub message: String,
    /// Span this label refers to
    pub span: SourceSpan,
    /// Whether this is the primary label
    pub primary: bool,
}

impl DiagnosticLabel {
    /// Create a new primary label.
    #[must_use]
    pub fn primary(message: &str, span: SourceSpan) -> Self {
        Self {
            message: message.to_string(),
            span,
            primary: true,
        }
    }

    /// Create a new secondary label.
    #[must_use]
    pub fn secondary(message: &str, span: SourceSpan) -> Self {
        Self {
            message: message.to_string(),
            span,
            primary: false,
        }
    }
}

/// Compiler suggestion for fixing the error.
///
/// Rust's excellent diagnostics provide suggestions that can bootstrap
/// the fix generator per DeepDelta (Mesbah et al., 2019).
#[derive(Debug, Clone)]
pub struct CompilerSuggestion {
    /// Suggestion text
    pub message: String,
    /// Applicability confidence
    pub applicability: SuggestionApplicability,
    /// Code replacement
    pub replacement: CodeReplacement,
}

impl CompilerSuggestion {
    /// Create a new suggestion.
    #[must_use]
    pub fn new(
        message: &str,
        applicability: SuggestionApplicability,
        replacement: CodeReplacement,
    ) -> Self {
        Self {
            message: message.to_string(),
            applicability,
            replacement,
        }
    }

    /// Check if this suggestion can be safely auto-applied.
    #[must_use]
    pub fn is_auto_applicable(&self) -> bool {
        self.applicability == SuggestionApplicability::MachineApplicable
    }
}

/// Suggestion applicability level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionApplicability {
    /// Fix is mechanical and safe to apply automatically
    MachineApplicable,
    /// Fix might be correct but needs human review
    MaybeIncorrect,
    /// Fix has placeholders that need filling
    HasPlaceholders,
    /// Suggestion is informational only
    Unspecified,
}

/// Code replacement for a suggestion.
#[derive(Debug, Clone)]
pub struct CodeReplacement {
    /// Span to replace
    pub span: SourceSpan,
    /// Replacement text
    pub replacement: String,
}

impl CodeReplacement {
    /// Create a new code replacement.
    #[must_use]
    pub fn new(span: SourceSpan, replacement: &str) -> Self {
        Self {
            span,
            replacement: replacement.to_string(),
        }
    }

    /// Apply this replacement to source code.
    #[must_use]
    pub fn apply(&self, source: &str) -> String {
        if self.span.byte_start > 0 && self.span.byte_end > 0 {
            // Use byte offsets if available
            let mut result = source[..self.span.byte_start].to_string();
            result.push_str(&self.replacement);
            result.push_str(&source[self.span.byte_end..]);
            result
        } else {
            // Fallback to line/column-based replacement
            let lines: Vec<&str> = source.lines().collect();
            let mut result = String::new();

            for (i, line) in lines.iter().enumerate() {
                let line_num = i + 1; // 1-indexed

                if line_num < self.span.line_start || line_num > self.span.line_end {
                    result.push_str(line);
                    result.push('\n');
                } else if line_num == self.span.line_start && line_num == self.span.line_end {
                    // Single line replacement
                    let before = &line[..self.span.column_start.saturating_sub(1).min(line.len())];
                    let after = &line[self.span.column_end.saturating_sub(1).min(line.len())..];
                    result.push_str(before);
                    result.push_str(&self.replacement);
                    result.push_str(after);
                    result.push('\n');
                }
                // Multi-line replacements are more complex - simplified here
            }

            // Remove trailing newline if original didn't have one
            if !source.ends_with('\n') && result.ends_with('\n') {
                result.pop();
            }

            result
        }
    }
}

/// Type information extracted from type errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    /// Full type string (e.g., `Vec<String>`)
    pub full: String,
    /// Base type without generics (e.g., "Vec")
    pub base: String,
    /// Generic parameters if any
    pub generics: Vec<String>,
    /// Whether this is a reference
    pub is_reference: bool,
    /// Whether this is mutable
    pub is_mutable: bool,
    /// Lifetime if specified
    pub lifetime: Option<String>,
}

impl TypeInfo {
    /// Create a new type info from a type string.
    #[must_use]
    pub fn new(type_str: &str) -> Self {
        let trimmed = type_str.trim();

        // Check for reference
        let (is_reference, is_mutable, rest) = if let Some(stripped) = trimmed.strip_prefix("&mut ")
        {
            (true, true, stripped)
        } else if let Some(stripped) = trimmed.strip_prefix('&') {
            (true, false, stripped)
        } else {
            (false, false, trimmed)
        };

        // Extract lifetime if present
        let (lifetime, rest) = if rest.starts_with('\'') {
            let end = rest.find(' ').unwrap_or(rest.len());
            (Some(rest[..end].to_string()), rest[end..].trim())
        } else {
            (None, rest)
        };

        // Extract base and generics
        let (base, generics) = if let Some(start) = rest.find('<') {
            let base = rest[..start].to_string();
            let generics_str = &rest[start + 1..rest.len() - 1];
            let generics: Vec<String> = generics_str
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            (base, generics)
        } else {
            (rest.to_string(), Vec::new())
        };

        Self {
            full: type_str.to_string(),
            base,
            generics,
            is_reference,
            is_mutable,
            lifetime,
        }
    }

    /// Check if types are compatible (same base, possibly different references).
    #[must_use]
    pub fn is_compatible_with(&self, other: &TypeInfo) -> bool {
        self.base == other.base && self.generics == other.generics
    }

    /// Get suggested conversion from this type to target.
    #[must_use]
    pub fn suggest_conversion_to(&self, target: &TypeInfo) -> Option<String> {
        // String <-> &str conversions
        if self.base == "String" && target.base == "str" && target.is_reference {
            return Some(".as_str()".to_string());
        }
        if self.base == "str" && self.is_reference && target.base == "String" {
            return Some(".to_string()".to_string());
        }

        // Vec <-> slice conversions
        if self.base == "Vec" && target.is_reference {
            return Some(".as_slice()".to_string());
        }
        if self.is_reference && target.base == "Vec" {
            return Some(".to_vec()".to_string());
        }

        // Reference addition/removal
        if !self.is_reference && target.is_reference && !target.is_mutable {
            return Some("&".to_string());
        }
        if self.is_reference && !target.is_reference {
            return Some(".clone()".to_string());
        }

        None
    }
}

impl std::fmt::Display for TypeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::citl::{Difficulty, ErrorCategory};

    // ==================== SourceSpan Tests ====================

    #[test]
    fn test_source_span_new() {
        let span = SourceSpan::new("test.rs", 10, 12, 5, 20);
        assert_eq!(span.file, "test.rs");
        assert_eq!(span.line_start, 10);
        assert_eq!(span.line_end, 12);
        assert_eq!(span.column_start, 5);
        assert_eq!(span.column_end, 20);
    }

    #[test]
    fn test_source_span_single_line() {
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        assert!(span.is_single_line());
        assert_eq!(span.line_start, 10);
        assert_eq!(span.line_end, 10);
    }

    #[test]
    fn test_source_span_point() {
        let span = SourceSpan::point("test.rs", 10, 5);
        assert!(span.is_single_line());
        assert_eq!(span.column_start, 5);
        assert_eq!(span.column_end, 6);
    }

    #[test]
    fn test_source_span_len() {
        let span = SourceSpan::single_line("test.rs", 10, 5, 15);
        assert_eq!(span.len(), 10);
    }

    #[test]
    fn test_source_span_overlaps() {
        let span1 = SourceSpan::new("test.rs", 10, 15, 1, 80);
        let span2 = SourceSpan::new("test.rs", 12, 20, 1, 80);
        let span3 = SourceSpan::new("test.rs", 20, 25, 1, 80);
        let span4 = SourceSpan::new("other.rs", 10, 15, 1, 80);

        assert!(span1.overlaps(&span2));
        assert!(!span1.overlaps(&span3));
        assert!(!span1.overlaps(&span4)); // Different file
    }

    // ==================== DiagnosticSeverity Tests ====================

    #[test]
    fn test_severity_is_fatal() {
        assert!(DiagnosticSeverity::Error.is_fatal());
        assert!(!DiagnosticSeverity::Warning.is_fatal());
        assert!(!DiagnosticSeverity::Note.is_fatal());
        assert!(!DiagnosticSeverity::Help.is_fatal());
    }

    // ==================== CompilerDiagnostic Tests ====================

    #[test]
    fn test_diagnostic_new() {
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        let diag =
            CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span);

        assert_eq!(diag.code.code, "E0308");
        assert_eq!(diag.message, "mismatched types");
        assert!(!diag.is_type_error());
        assert!(!diag.has_suggestions());
    }

    #[test]
    fn test_diagnostic_with_types() {
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        let diag =
            CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
                .with_expected(TypeInfo::new("String"))
                .with_found(TypeInfo::new("&str"));

        assert!(diag.is_type_error());
        assert_eq!(
            diag.expected.as_ref().map(|t| &t.base),
            Some(&"String".to_string())
        );
        assert_eq!(
            diag.found.as_ref().map(|t| &t.base),
            Some(&"str".to_string())
        );
    }

    #[test]
    fn test_diagnostic_with_suggestions() {
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        let suggestion = CompilerSuggestion::new(
            "try using `.to_string()`",
            SuggestionApplicability::MachineApplicable,
            CodeReplacement::new(span.clone(), "value.to_string()"),
        );
        let diag =
            CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
                .with_suggestion(suggestion);

        assert!(diag.has_suggestions());
        assert_eq!(diag.machine_applicable_suggestions().len(), 1);
    }

    #[test]
    fn test_diagnostic_with_label() {
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        let label = DiagnosticLabel::primary("expected `String`", span.clone());
        let diag =
            CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
                .with_label(label);

        assert_eq!(diag.labels.len(), 1);
        assert!(diag.labels[0].primary);
    }

    #[test]
    fn test_diagnostic_with_note() {
        let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        let diag =
            CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
                .with_note("help: consider using `.to_string()`");

        assert_eq!(diag.notes.len(), 1);
    }

    // ==================== DiagnosticLabel Tests ====================

    #[test]
    fn test_label_primary() {
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        let label = DiagnosticLabel::primary("primary message", span);
        assert!(label.primary);
    }

    #[test]
    fn test_label_secondary() {
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);
        let label = DiagnosticLabel::secondary("secondary message", span);
        assert!(!label.primary);
    }

    // ==================== CompilerSuggestion Tests ====================

    #[test]
    fn test_suggestion_is_auto_applicable() {
        let span = SourceSpan::single_line("test.rs", 10, 5, 20);

        let auto = CompilerSuggestion::new(
            "auto fix",
            SuggestionApplicability::MachineApplicable,
            CodeReplacement::new(span.clone(), "fixed"),
        );
        assert!(auto.is_auto_applicable());

        let manual = CompilerSuggestion::new(
            "manual fix",
            SuggestionApplicability::MaybeIncorrect,
            CodeReplacement::new(span, "fixed"),
        );
        assert!(!manual.is_auto_applicable());
    }

    // ==================== CodeReplacement Tests ====================

    #[test]
    fn test_code_replacement_apply_single_line() {
        let span = SourceSpan {
            file: "test.rs".to_string(),
            line_start: 1,
            line_end: 1,
            column_start: 5,
            column_end: 10,
            byte_start: 0,
            byte_end: 0,
        };
        let replacement = CodeReplacement::new(span, "world");
        let source = "let hello = 42;";
        let result = replacement.apply(source);
        assert!(result.contains("world"));
    }

    // ==================== TypeInfo Tests ====================

    #[test]
    fn test_type_info_simple() {
        let ti = TypeInfo::new("String");
        assert_eq!(ti.base, "String");
        assert!(!ti.is_reference);
        assert!(!ti.is_mutable);
        assert!(ti.generics.is_empty());
    }

    #[test]
    fn test_type_info_reference() {
        let ti = TypeInfo::new("&str");
        assert_eq!(ti.base, "str");
        assert!(ti.is_reference);
        assert!(!ti.is_mutable);
    }

    #[test]
    fn test_type_info_mutable_reference() {
        let ti = TypeInfo::new("&mut Vec<i32>");
        assert_eq!(ti.base, "Vec");
        assert!(ti.is_reference);
        assert!(ti.is_mutable);
        assert_eq!(ti.generics, vec!["i32"]);
    }

    #[test]
    fn test_type_info_generics() {
        let ti = TypeInfo::new("HashMap<String, i32>");
        assert_eq!(ti.base, "HashMap");
        assert_eq!(ti.generics.len(), 2);
        assert_eq!(ti.generics[0], "String");
        assert_eq!(ti.generics[1], "i32");
    }

    #[test]
    fn test_type_info_compatibility() {
        let t1 = TypeInfo::new("Vec<i32>");
        let t2 = TypeInfo::new("&Vec<i32>");
        let t3 = TypeInfo::new("Vec<String>");

        assert!(t1.is_compatible_with(&t2)); // Same base, different reference
        assert!(!t1.is_compatible_with(&t3)); // Different generics
    }

    #[test]
    fn test_type_info_conversion_string_to_str() {
        let from = TypeInfo::new("String");
        let to = TypeInfo::new("&str");
        assert_eq!(
            from.suggest_conversion_to(&to),
            Some(".as_str()".to_string())
        );
    }

    #[test]
    fn test_type_info_conversion_str_to_string() {
        let from = TypeInfo::new("&str");
        let to = TypeInfo::new("String");
        assert_eq!(
            from.suggest_conversion_to(&to),
            Some(".to_string()".to_string())
        );
    }

    #[test]
    fn test_type_info_conversion_add_reference() {
        let from = TypeInfo::new("String");
        let to = TypeInfo::new("&String");
        assert_eq!(from.suggest_conversion_to(&to), Some("&".to_string()));
    }

    #[test]
    fn test_type_info_conversion_clone() {
        let from = TypeInfo::new("&String");
        let to = TypeInfo::new("String");
        assert_eq!(
            from.suggest_conversion_to(&to),
            Some(".clone()".to_string())
        );
    }
}
