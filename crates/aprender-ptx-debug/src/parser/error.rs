//! Parser error types

use super::ast::SourceLocation;

/// Parse error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParseError {
    /// Unexpected token
    #[error("Unexpected token at {location}: expected {expected}, found {found}")]
    UnexpectedToken {
        /// Expected token description
        expected: String,
        /// Found token description
        found: String,
        /// Source location
        location: SourceLocation,
    },

    /// Unexpected end of file
    #[error("Unexpected end of file at {location}")]
    UnexpectedEof {
        /// Source location
        location: SourceLocation,
    },

    /// Invalid directive
    #[error("Invalid directive '{directive}' at {location}")]
    InvalidDirective {
        /// Directive text
        directive: String,
        /// Source location
        location: SourceLocation,
    },

    /// Invalid instruction
    #[error("Invalid instruction at {location}: {message}")]
    InvalidInstruction {
        /// Error message
        message: String,
        /// Source location
        location: SourceLocation,
    },

    /// Invalid operand
    #[error("Invalid operand '{operand}' at {location}")]
    InvalidOperand {
        /// Operand text
        operand: String,
        /// Source location
        location: SourceLocation,
    },

    /// Invalid type
    #[error("Invalid type '{ty}' at {location}")]
    InvalidType {
        /// Type text
        ty: String,
        /// Source location
        location: SourceLocation,
    },

    /// Duplicate label
    #[error("Duplicate label '{label}' at {location}")]
    DuplicateLabel {
        /// Label name
        label: String,
        /// Source location
        location: SourceLocation,
    },

    /// Undefined label
    #[error("Undefined label '{label}' at {location}")]
    UndefinedLabel {
        /// Label name
        label: String,
        /// Source location
        location: SourceLocation,
    },

    /// Lexer error
    #[error("Lexer error at {location}: {message}")]
    LexerError {
        /// Error message
        message: String,
        /// Source location
        location: SourceLocation,
    },
}

impl ParseError {
    /// Get the source location of the error
    pub fn location(&self) -> &SourceLocation {
        match self {
            ParseError::UnexpectedToken { location, .. } => location,
            ParseError::UnexpectedEof { location, .. } => location,
            ParseError::InvalidDirective { location, .. } => location,
            ParseError::InvalidInstruction { location, .. } => location,
            ParseError::InvalidOperand { location, .. } => location,
            ParseError::InvalidType { location, .. } => location,
            ParseError::DuplicateLabel { location, .. } => location,
            ParseError::UndefinedLabel { location, .. } => location,
            ParseError::LexerError { location, .. } => location,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ParseError::UnexpectedToken {
            expected: "identifier".into(),
            found: "number".into(),
            location: SourceLocation { line: 10, column: 5, file: None },
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Unexpected token"));
        assert!(msg.contains("10:5"));
    }

    #[test]
    fn test_error_location() {
        let loc = SourceLocation { line: 42, column: 10, file: None };
        let err = ParseError::InvalidDirective { directive: ".foo".into(), location: loc.clone() };
        assert_eq!(err.location().line, 42);
        assert_eq!(err.location().column, 10);
    }
}
