//! Source span types for CITL trainer

use serde::{Deserialize, Serialize};

/// A source code span (location in source file)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceSpan {
    /// File path
    pub file: String,
    /// Start line (1-indexed)
    pub start_line: u32,
    /// Start column (1-indexed)
    pub start_col: u32,
    /// End line (1-indexed)
    pub end_line: u32,
    /// End column (1-indexed)
    pub end_col: u32,
}

impl SourceSpan {
    /// Create a new source span
    #[must_use]
    pub fn new(
        file: impl Into<String>,
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    ) -> Self {
        Self { file: file.into(), start_line, start_col, end_line, end_col }
    }

    /// Create a single-line span
    #[must_use]
    pub fn line(file: impl Into<String>, line: u32) -> Self {
        Self::new(file, line, 1, line, u32::MAX)
    }

    /// Check if this span overlaps with another
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.file != other.file {
            return false;
        }

        // Check if ranges overlap
        !(self.end_line < other.start_line || other.end_line < self.start_line)
    }

    /// Check if this span contains another
    #[must_use]
    pub fn contains(&self, other: &Self) -> bool {
        if self.file != other.file {
            return false;
        }

        self.start_line <= other.start_line && self.end_line >= other.end_line
    }
}

impl std::fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}-{}:{}",
            self.file, self.start_line, self.start_col, self.end_line, self.end_col
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_span_new() {
        let span = SourceSpan::new("main.rs", 1, 1, 10, 80);
        assert_eq!(span.file, "main.rs");
        assert_eq!(span.start_line, 1);
        assert_eq!(span.end_line, 10);
    }

    #[test]
    fn test_source_span_line() {
        let span = SourceSpan::line("main.rs", 5);
        assert_eq!(span.file, "main.rs");
        assert_eq!(span.start_line, 5);
        assert_eq!(span.end_line, 5);
    }

    #[test]
    fn test_source_span_overlaps_same_line() {
        let span1 = SourceSpan::line("main.rs", 5);
        let span2 = SourceSpan::line("main.rs", 5);
        assert!(span1.overlaps(&span2));
    }

    #[test]
    fn test_source_span_overlaps_different_lines() {
        let span1 = SourceSpan::new("main.rs", 1, 1, 10, 80);
        let span2 = SourceSpan::new("main.rs", 5, 1, 15, 80);
        assert!(span1.overlaps(&span2));
    }

    #[test]
    fn test_source_span_no_overlap() {
        let span1 = SourceSpan::new("main.rs", 1, 1, 5, 80);
        let span2 = SourceSpan::new("main.rs", 10, 1, 15, 80);
        assert!(!span1.overlaps(&span2));
    }

    #[test]
    fn test_source_span_no_overlap_different_files() {
        let span1 = SourceSpan::line("main.rs", 5);
        let span2 = SourceSpan::line("lib.rs", 5);
        assert!(!span1.overlaps(&span2));
    }

    #[test]
    fn test_source_span_contains() {
        let outer = SourceSpan::new("main.rs", 1, 1, 20, 80);
        let inner = SourceSpan::new("main.rs", 5, 1, 10, 80);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_source_span_display() {
        let span = SourceSpan::new("main.rs", 5, 10, 5, 20);
        let display = format!("{span}");
        assert!(display.contains("main.rs"));
        assert!(display.contains('5'));
    }

    #[test]
    fn test_source_span_serialization() {
        let span = SourceSpan::line("main.rs", 5);
        let json = serde_json::to_string(&span).expect("JSON serialization should succeed");
        let deserialized: SourceSpan =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(span, deserialized);
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_source_span_overlap_symmetric(
            line1 in 1u32..100,
            line2 in 1u32..100
        ) {
            let span1 = SourceSpan::line("file.rs", line1);
            let span2 = SourceSpan::line("file.rs", line2);

            prop_assert_eq!(span1.overlaps(&span2), span2.overlaps(&span1));
        }
    }
}
