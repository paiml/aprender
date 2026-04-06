//! HPO error types

use thiserror::Error;

/// HPO errors
#[derive(Debug, Error)]
pub enum HPOError {
    #[error("Empty search space")]
    EmptySpace,

    #[error("Parameter not found: {0}")]
    ParameterNotFound(String),

    #[error("Invalid parameter value for {0}: {1}")]
    InvalidValue(String, String),

    #[error("No trials completed")]
    NoTrials,

    #[error("HPO error: {0}")]
    Internal(String),
}

/// Result type for HPO operations
pub type Result<T> = std::result::Result<T, HPOError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hpo_error_display() {
        let err = HPOError::EmptySpace;
        assert!(format!("{err}").contains("Empty search space"));

        let err = HPOError::ParameterNotFound("lr".to_string());
        assert!(format!("{err}").contains("Parameter not found"));
        assert!(format!("{err}").contains("lr"));

        let err = HPOError::InvalidValue("lr".to_string(), "invalid".to_string());
        assert!(format!("{err}").contains("Invalid parameter value"));

        let err = HPOError::NoTrials;
        assert!(format!("{err}").contains("No trials completed"));

        let err = HPOError::Internal("test error".to_string());
        assert!(format!("{err}").contains("HPO error"));
    }
}
