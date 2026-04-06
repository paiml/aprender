//! TUI-specific error types
//!
//! Provides error handling for the TUI dataset viewer without panic paths.

use std::fmt;

/// TUI-specific error type
///
/// All errors are recoverable - no panics in TUI code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiError {
    /// Dataset is empty (no rows)
    EmptyDataset,
    /// Row index out of bounds
    RowOutOfBounds {
        /// Requested row index
        requested: usize,
        /// Total row count
        total: usize,
    },
    /// Column index out of bounds
    ColumnOutOfBounds {
        /// Requested column index
        requested: usize,
        /// Total column count
        total: usize,
    },
    /// Failed to format cell value
    FormatError {
        /// Row index
        row: usize,
        /// Column index
        col: usize,
        /// Reason for failure
        reason: String,
    },
    /// Schema mismatch between batches
    SchemaMismatch {
        /// Description of mismatch
        description: String,
    },
    /// Unsupported Arrow data type
    UnsupportedType {
        /// The unsupported type name
        type_name: String,
    },
    /// Render constraint violation
    RenderConstraint {
        /// Description of constraint violation
        description: String,
    },
    /// Invalid scroll position
    InvalidScroll {
        /// Requested offset
        requested: usize,
        /// Maximum valid offset
        max_valid: usize,
    },
}

impl fmt::Display for TuiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => write!(f, "Dataset is empty"),
            Self::RowOutOfBounds { requested, total } => {
                write!(f, "Row index {requested} out of bounds (total: {total})")
            }
            Self::ColumnOutOfBounds { requested, total } => {
                write!(f, "Column index {requested} out of bounds (total: {total})")
            }
            Self::FormatError { row, col, reason } => {
                write!(f, "Failed to format cell ({row}, {col}): {reason}")
            }
            Self::SchemaMismatch { description } => {
                write!(f, "Schema mismatch: {description}")
            }
            Self::UnsupportedType { type_name } => {
                write!(f, "Unsupported Arrow type: {type_name}")
            }
            Self::RenderConstraint { description } => {
                write!(f, "Render constraint violation: {description}")
            }
            Self::InvalidScroll {
                requested,
                max_valid,
            } => {
                write!(f, "Invalid scroll position {requested} (max: {max_valid})")
            }
        }
    }
}

impl std::error::Error for TuiError {}

/// Result type for TUI operations
pub type TuiResult<T> = Result<T, TuiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f_error_display_empty_dataset() {
        let err = TuiError::EmptyDataset;
        assert_eq!(err.to_string(), "Dataset is empty");
    }

    #[test]
    fn f_error_display_row_out_of_bounds() {
        let err = TuiError::RowOutOfBounds {
            requested: 100,
            total: 80,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("80"));
    }

    #[test]
    fn f_error_display_column_out_of_bounds() {
        let err = TuiError::ColumnOutOfBounds {
            requested: 15,
            total: 11,
        };
        assert!(err.to_string().contains("15"));
        assert!(err.to_string().contains("11"));
    }

    #[test]
    fn f_error_display_format_error() {
        let err = TuiError::FormatError {
            row: 5,
            col: 3,
            reason: "null value".to_string(),
        };
        assert!(err.to_string().contains('5'));
        assert!(err.to_string().contains('3'));
        assert!(err.to_string().contains("null value"));
    }

    #[test]
    fn f_error_is_clone() {
        let err = TuiError::EmptyDataset;
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn f_error_is_debug() {
        let err = TuiError::EmptyDataset;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("EmptyDataset"));
    }

    #[test]
    fn f_error_implements_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(TuiError::EmptyDataset);
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn f_error_display_schema_mismatch() {
        let err = TuiError::SchemaMismatch {
            description: "column count differs".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("Schema mismatch"));
        assert!(s.contains("column count differs"));
    }

    #[test]
    fn f_error_display_unsupported_type() {
        let err = TuiError::UnsupportedType {
            type_name: "FixedSizeBinary".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("Unsupported Arrow type"));
        assert!(s.contains("FixedSizeBinary"));
    }

    #[test]
    fn f_error_display_render_constraint() {
        let err = TuiError::RenderConstraint {
            description: "width exceeds terminal".to_string(),
        };
        let s = err.to_string();
        assert!(s.contains("Render constraint"));
        assert!(s.contains("width exceeds terminal"));
    }

    #[test]
    fn f_error_display_invalid_scroll() {
        let err = TuiError::InvalidScroll {
            requested: 100,
            max_valid: 80,
        };
        let s = err.to_string();
        assert!(s.contains("Invalid scroll position"));
        assert!(s.contains("100"));
        assert!(s.contains("80"));
    }
}
