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
/// the fix generator per `DeepDelta` (Mesbah et al., 2019).
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
#[path = "diagnostic_tests.rs"]
mod tests;
