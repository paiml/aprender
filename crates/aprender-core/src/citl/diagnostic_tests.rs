pub(crate) use super::*;
pub(crate) use crate::citl::{Difficulty, ErrorCategory};

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
    let diag = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span);

    assert_eq!(diag.code.code, "E0308");
    assert_eq!(diag.message, "mismatched types");
    assert!(!diag.is_type_error());
    assert!(!diag.has_suggestions());
}

#[test]
fn test_diagnostic_with_types() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diag = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
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
    let diag = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
        .with_suggestion(suggestion);

    assert!(diag.has_suggestions());
    assert_eq!(diag.machine_applicable_suggestions().len(), 1);
}

#[test]
fn test_diagnostic_with_label() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let label = DiagnosticLabel::primary("expected `String`", span.clone());
    let diag = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
        .with_label(label);

    assert_eq!(diag.labels.len(), 1);
    assert!(diag.labels[0].primary);
}

#[test]
fn test_diagnostic_with_note() {
    let code = ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy);
    let span = SourceSpan::single_line("test.rs", 10, 5, 20);
    let diag = CompilerDiagnostic::new(code, DiagnosticSeverity::Error, "mismatched types", span)
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

// ========================================================================
// Coverage Tests for uncovered branches
// ========================================================================

#[test]
fn test_source_span_len_with_byte_offsets() {
    // Cover line 200: byte_end > byte_start branch
    let span = SourceSpan {
        file: "test.rs".to_string(),
        line_start: 1,
        line_end: 1,
        column_start: 5,
        column_end: 15,
        byte_start: 10,
        byte_end: 25, // byte offsets specified
    };
    assert_eq!(span.len(), 15); // Uses byte_end - byte_start
}

#[test]
fn test_source_span_len_multiline_estimate() {
    // Cover line 205: multi-line estimate branch
    let span = SourceSpan {
        file: "test.rs".to_string(),
        line_start: 10,
        line_end: 15, // Multi-line
        column_start: 1,
        column_end: 80,
        byte_start: 0,
        byte_end: 0, // No byte offsets
    };
    // Multi-line formula: (line_end - line_start + 1) * 40 = 6 * 40 = 240
    assert_eq!(span.len(), 240);
}

#[test]
fn test_source_span_is_empty() {
    // Cover lines 211-213: is_empty() based on len()
    let empty_span = SourceSpan {
        file: "test.rs".to_string(),
        line_start: 10,
        line_end: 10,
        column_start: 5,
        column_end: 5, // Same column = zero length
        byte_start: 0,
        byte_end: 0,
    };
    assert!(empty_span.is_empty());

    let non_empty_span = SourceSpan::single_line("test.rs", 10, 5, 10);
    assert!(!non_empty_span.is_empty());
}

#[test]
fn test_code_replacement_apply_with_byte_offsets() {
    // Cover lines 343-346: byte offset code replacement path
    let span = SourceSpan {
        file: "test.rs".to_string(),
        line_start: 1,
        line_end: 1,
        column_start: 5,
        column_end: 10,
        byte_start: 4, // "let " (0-3), then target starts at 4
        byte_end: 9,   // "hello"
    };
    let replacement = CodeReplacement::new(span, "world");
    let source = "let hello = 42;";
    let result = replacement.apply(source);
    assert_eq!(result, "let world = 42;");
}

#[test]
fn test_code_replacement_apply_multiline_preservation() {
    // Cover lines 356-357: lines outside replacement range
    let span = SourceSpan {
        file: "test.rs".to_string(),
        line_start: 2,
        line_end: 2,
        column_start: 5,
        column_end: 10,
        byte_start: 0,
        byte_end: 0,
    };
    let replacement = CodeReplacement::new(span, "world");
    let source = "line one\nlet hello = 42;\nline three";
    let result = replacement.apply(source);
    // Line 1 and line 3 should be preserved, line 2 modified
    assert!(result.contains("line one"));
    assert!(result.contains("world"));
    assert!(result.contains("line three"));
}

#[test]
fn test_type_info_with_lifetime() {
    // Cover lines 415-416: lifetime parsing with space
    let ti = TypeInfo::new("&'a str");
    assert!(ti.is_reference);
    assert!(!ti.is_mutable);
    assert_eq!(ti.lifetime, Some("'a".to_string()));
    assert_eq!(ti.base, "str");
}

#[test]
fn test_type_info_with_lifetime_no_space() {
    // Cover lifetime parsing where find(' ') returns None
    let ti = TypeInfo::new("&'static");
    assert!(ti.is_reference);
    assert_eq!(ti.lifetime, Some("'static".to_string()));
}

#[test]
fn test_type_info_conversion_vec_to_slice() {
    // Cover line 463: Vec -> slice conversion
    let from = TypeInfo::new("Vec<i32>");
    let to = TypeInfo::new("&[i32]");
    assert_eq!(
        from.suggest_conversion_to(&to),
        Some(".as_slice()".to_string())
    );
}

#[test]
fn test_type_info_conversion_slice_to_vec() {
    // Cover line 466: slice -> Vec conversion
    let from = TypeInfo::new("&[i32]");
    let to = TypeInfo::new("Vec<i32>");
    assert_eq!(
        from.suggest_conversion_to(&to),
        Some(".to_vec()".to_string())
    );
}

#[test]
fn test_type_info_conversion_no_conversion() {
    // Cover line 477: None return path
    let from = TypeInfo::new("HashMap<String, i32>");
    let to = TypeInfo::new("BTreeMap<String, i32>");
    assert_eq!(from.suggest_conversion_to(&to), None);
}

#[test]
fn test_type_info_conversion_ref_to_owned() {
    // Cover line 474: reference to non-reference (clone path)
    // This is a different case from &String -> String which was already tested
    let from = TypeInfo::new("&i32");
    let to = TypeInfo::new("i32");
    assert_eq!(
        from.suggest_conversion_to(&to),
        Some(".clone()".to_string())
    );
}

#[test]
fn test_type_info_conversion_same_ref() {
    // When both are same reference type, no conversion available
    let from = TypeInfo::new("&String");
    let to = TypeInfo::new("&String");
    // Neither adds ref (both refs) nor removes ref (both refs) -> None
    // Actually from.is_reference=true, target.is_reference=true,
    // so line 473 condition: self.is_reference && !target.is_reference is false
    // and line 470: !self.is_reference && target.is_reference is false
    // So it returns None
    assert_eq!(from.suggest_conversion_to(&to), None);
}

#[test]
fn test_type_info_display() {
    let ti = TypeInfo::new("Vec<String>");
    let display = format!("{}", ti);
    assert_eq!(display, "Vec<String>");
}

#[test]
fn test_source_span_default() {
    let span = SourceSpan::default();
    assert!(span.file.is_empty());
    assert_eq!(span.line_start, 1);
    assert_eq!(span.line_end, 1);
    assert_eq!(span.column_start, 1);
    assert_eq!(span.column_end, 1);
}

#[test]
fn test_code_replacement_apply_preserves_no_trailing_newline() {
    // Cover lines 371-373: remove trailing newline if original didn't have one
    let span = SourceSpan {
        file: "test.rs".to_string(),
        line_start: 1,
        line_end: 1,
        column_start: 1,
        column_end: 6,
        byte_start: 0,
        byte_end: 0,
    };
    let replacement = CodeReplacement::new(span, "world");
    let source = "hello"; // No trailing newline
    let result = replacement.apply(source);
    assert!(!result.ends_with('\n'));
}
