//! Error types for YAML parsing.

use std::fmt;

/// Error type for manifest parsing.
#[derive(Debug)]
pub enum ParseError {
    /// YAML parsing error
    Yaml(serde_yaml_ng::Error),
    /// Expression parsing error
    Expression(crate::expression::ExpressionError),
    /// Validation error
    Validation(String),
    /// Missing required field
    MissingField(String),
    /// Invalid value
    InvalidValue {
        /// Field name
        field: String,
        /// Error message
        message: String,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Yaml(e) => write!(f, "YAML error: {e}"),
            Self::Expression(e) => write!(f, "Expression error: {e}"),
            Self::Validation(msg) => write!(f, "Validation error: {msg}"),
            Self::MissingField(field) => write!(f, "Missing required field: {field}"),
            Self::InvalidValue { field, message } => {
                write!(f, "Invalid value for '{field}': {message}")
            }
        }
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Yaml(e) => Some(e),
            Self::Expression(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_yaml_ng::Error> for ParseError {
    fn from(e: serde_yaml_ng::Error) -> Self {
        Self::Yaml(e)
    }
}

impl From<crate::expression::ExpressionError> for ParseError {
    fn from(e: crate::expression::ExpressionError) -> Self {
        Self::Expression(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_parse_error_display() {
        let err = ParseError::MissingField("name".to_string());
        assert_eq!(err.to_string(), "Missing required field: name");

        let err = ParseError::InvalidValue {
            field: "columns".to_string(),
            message: "must be positive".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Invalid value for 'columns': must be positive"
        );

        let err = ParseError::Validation("layout is required".to_string());
        assert_eq!(err.to_string(), "Validation error: layout is required");
    }

    #[test]
    fn test_parse_error_yaml_display() {
        // Create a YAML error by parsing invalid YAML
        let yaml_err: serde_yaml_ng::Error =
            serde_yaml_ng::from_str::<serde_yaml_ng::Value>("{{").unwrap_err();
        let err = ParseError::Yaml(yaml_err);
        assert!(err.to_string().contains("YAML error"));
    }

    #[test]
    fn test_parse_error_expression_display() {
        use crate::expression::ExpressionError;
        let expr_err = ExpressionError::UnknownTransform("missing".to_string());
        let err = ParseError::Expression(expr_err);
        assert!(err.to_string().contains("Expression error"));
    }

    #[test]
    fn test_parse_error_from_yaml() {
        let yaml_err: serde_yaml_ng::Error =
            serde_yaml_ng::from_str::<serde_yaml_ng::Value>("{{").unwrap_err();
        let err: ParseError = yaml_err.into();
        assert!(matches!(err, ParseError::Yaml(_)));
    }

    #[test]
    fn test_parse_error_from_expression() {
        use crate::expression::ExpressionError;
        let expr_err = ExpressionError::EmptyExpression;
        let err: ParseError = expr_err.into();
        assert!(matches!(err, ParseError::Expression(_)));
    }

    #[test]
    fn test_parse_error_source_yaml() {
        let yaml_err: serde_yaml_ng::Error =
            serde_yaml_ng::from_str::<serde_yaml_ng::Value>("{{").unwrap_err();
        let err = ParseError::Yaml(yaml_err);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_parse_error_source_expression() {
        use crate::expression::ExpressionError;
        let expr_err = ExpressionError::InvalidArgument("bad arg".to_string());
        let err = ParseError::Expression(expr_err);
        assert!(err.source().is_some());
    }

    #[test]
    fn test_parse_error_source_validation() {
        let err = ParseError::Validation("test".to_string());
        assert!(err.source().is_none());
    }

    #[test]
    fn test_parse_error_source_missing_field() {
        let err = ParseError::MissingField("test".to_string());
        assert!(err.source().is_none());
    }

    #[test]
    fn test_parse_error_source_invalid_value() {
        let err = ParseError::InvalidValue {
            field: "x".to_string(),
            message: "y".to_string(),
        };
        assert!(err.source().is_none());
    }
}
