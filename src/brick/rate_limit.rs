//! DoS Prevention Rate Limiting
//!
//! AWP-15: Request validation and rate limiting for DoS prevention.

use std::fmt;
use std::time::Duration;

// ----------------------------------------------------------------------------
// AWP-15: DoS Prevention Limits
// ----------------------------------------------------------------------------

/// DoS prevention limits for serving.
///
/// # Example
/// ```rust
/// use trueno::brick::ServeLimits;
///
/// let limits = ServeLimits::default();
/// assert!(limits.validate_request(50, 1024).is_ok());
/// assert!(limits.validate_request(200, 1024).is_err());  // Too many headers
/// ```
#[derive(Debug, Clone)]
pub struct ServeLimits {
    /// Maximum request body size (bytes).
    pub max_request_size: usize,
    /// Maximum number of headers.
    pub max_headers: usize,
    /// Maximum header size (bytes).
    pub max_header_size: usize,
    /// Keep-alive timeout.
    pub keep_alive_timeout: Duration,
    /// Client request timeout.
    pub client_timeout: Duration,
    /// Maximum pipelined requests.
    pub max_pipelined: usize,
    /// Maximum concurrent connections.
    pub max_connections: usize,
}

impl Default for ServeLimits {
    fn default() -> Self {
        Self {
            max_request_size: 2 * 1024 * 1024, // 2MB
            max_headers: 100,
            max_header_size: 8 * 1024, // 8KB
            keep_alive_timeout: Duration::from_secs(5),
            client_timeout: Duration::from_secs(5),
            max_pipelined: 16,
            max_connections: 1024,
        }
    }
}

impl ServeLimits {
    /// Create new limits with custom values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set max request size.
    #[must_use]
    pub fn with_max_request_size(mut self, size: usize) -> Self {
        self.max_request_size = size;
        self
    }

    /// Builder: set max headers.
    #[must_use]
    pub fn with_max_headers(mut self, count: usize) -> Self {
        self.max_headers = count;
        self
    }

    /// Builder: set max connections.
    #[must_use]
    pub fn with_max_connections(mut self, count: usize) -> Self {
        self.max_connections = count;
        self
    }

    /// Validate incoming request against limits.
    pub fn validate_request(
        &self,
        headers_count: usize,
        body_size: usize,
    ) -> Result<(), LimitError> {
        if headers_count > self.max_headers {
            return Err(LimitError::TooManyHeaders { count: headers_count, max: self.max_headers });
        }
        if body_size > self.max_request_size {
            return Err(LimitError::BodyTooLarge { size: body_size, max: self.max_request_size });
        }
        Ok(())
    }

    /// Validate header size.
    pub fn validate_header_size(&self, size: usize) -> Result<(), LimitError> {
        if size > self.max_header_size {
            return Err(LimitError::HeaderTooLarge { size, max: self.max_header_size });
        }
        Ok(())
    }

    /// Validate pipelined request count.
    pub fn validate_pipelined(&self, count: usize) -> Result<(), LimitError> {
        if count > self.max_pipelined {
            return Err(LimitError::TooManyPipelined { count, max: self.max_pipelined });
        }
        Ok(())
    }

    /// Validate connection count.
    pub fn validate_connections(&self, current: usize) -> Result<(), LimitError> {
        if current >= self.max_connections {
            return Err(LimitError::ConnectionLimitReached { current, max: self.max_connections });
        }
        Ok(())
    }
}

/// Error when a limit is exceeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitError {
    /// Too many headers in request.
    TooManyHeaders {
        /// Actual count.
        count: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// Request body too large.
    BodyTooLarge {
        /// Actual size.
        size: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// Header too large.
    HeaderTooLarge {
        /// Actual size.
        size: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// Too many pipelined requests.
    TooManyPipelined {
        /// Actual count.
        count: usize,
        /// Maximum allowed.
        max: usize,
    },
    /// Connection limit reached.
    ConnectionLimitReached {
        /// Current connections.
        current: usize,
        /// Maximum allowed.
        max: usize,
    },
}

impl fmt::Display for LimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LimitError::TooManyHeaders { count, max } => {
                write!(f, "too many headers: {} (max {})", count, max)
            }
            LimitError::BodyTooLarge { size, max } => {
                write!(f, "body too large: {} bytes (max {})", size, max)
            }
            LimitError::HeaderTooLarge { size, max } => {
                write!(f, "header too large: {} bytes (max {})", size, max)
            }
            LimitError::TooManyPipelined { count, max } => {
                write!(f, "too many pipelined requests: {} (max {})", count, max)
            }
            LimitError::ConnectionLimitReached { current, max } => {
                write!(f, "connection limit reached: {} (max {})", current, max)
            }
        }
    }
}

impl std::error::Error for LimitError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_limits_default() {
        let limits = ServeLimits::default();
        assert_eq!(limits.max_request_size, 2 * 1024 * 1024);
        assert_eq!(limits.max_headers, 100);
        assert_eq!(limits.max_header_size, 8 * 1024);
        assert_eq!(limits.max_pipelined, 16);
        assert_eq!(limits.max_connections, 1024);
    }

    #[test]
    fn test_serve_limits_builder() {
        let limits = ServeLimits::new()
            .with_max_request_size(1024)
            .with_max_headers(50)
            .with_max_connections(100);

        assert_eq!(limits.max_request_size, 1024);
        assert_eq!(limits.max_headers, 50);
        assert_eq!(limits.max_connections, 100);
    }

    #[test]
    fn test_validate_request_ok() {
        let limits = ServeLimits::default();
        assert!(limits.validate_request(50, 1024).is_ok());
    }

    #[test]
    fn test_validate_request_too_many_headers() {
        let limits = ServeLimits::default();
        let result = limits.validate_request(200, 1024);
        assert!(matches!(result, Err(LimitError::TooManyHeaders { .. })));
    }

    #[test]
    fn test_validate_request_body_too_large() {
        let limits = ServeLimits::default();
        let result = limits.validate_request(50, 10 * 1024 * 1024);
        assert!(matches!(result, Err(LimitError::BodyTooLarge { .. })));
    }

    #[test]
    fn test_validate_header_size_ok() {
        let limits = ServeLimits::default();
        assert!(limits.validate_header_size(1024).is_ok());
    }

    #[test]
    fn test_validate_header_size_too_large() {
        let limits = ServeLimits::default();
        let result = limits.validate_header_size(16 * 1024);
        assert!(matches!(result, Err(LimitError::HeaderTooLarge { .. })));
    }

    #[test]
    fn test_validate_pipelined_ok() {
        let limits = ServeLimits::default();
        assert!(limits.validate_pipelined(10).is_ok());
    }

    #[test]
    fn test_validate_pipelined_too_many() {
        let limits = ServeLimits::default();
        let result = limits.validate_pipelined(20);
        assert!(matches!(result, Err(LimitError::TooManyPipelined { .. })));
    }

    #[test]
    fn test_validate_connections_ok() {
        let limits = ServeLimits::default();
        assert!(limits.validate_connections(500).is_ok());
    }

    #[test]
    fn test_validate_connections_limit_reached() {
        let limits = ServeLimits::default();
        let result = limits.validate_connections(1024);
        assert!(matches!(result, Err(LimitError::ConnectionLimitReached { .. })));
    }

    #[test]
    fn test_limit_error_display() {
        let err = LimitError::TooManyHeaders { count: 150, max: 100 };
        assert_eq!(format!("{}", err), "too many headers: 150 (max 100)");

        let err = LimitError::BodyTooLarge { size: 5000000, max: 2097152 };
        assert_eq!(format!("{}", err), "body too large: 5000000 bytes (max 2097152)");
    }

    #[test]
    fn test_limit_error_eq() {
        let err1 = LimitError::TooManyHeaders { count: 150, max: 100 };
        let err2 = LimitError::TooManyHeaders { count: 150, max: 100 };
        let err3 = LimitError::TooManyHeaders { count: 200, max: 100 };

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    /// FALSIFICATION TEST: Verify boundary conditions
    ///
    /// Limits must reject exactly at the boundary, not before or after.
    #[test]
    fn test_falsify_exact_boundaries() {
        let limits = ServeLimits::new()
            .with_max_headers(100)
            .with_max_request_size(1000)
            .with_max_connections(10);

        // Headers: 100 is the max, so 100 should be OK but 101 should fail
        assert!(limits.validate_request(100, 0).is_ok());
        assert!(limits.validate_request(101, 0).is_err());

        // Body: 1000 is the max, so 1000 should be OK but 1001 should fail
        assert!(limits.validate_request(0, 1000).is_ok());
        assert!(limits.validate_request(0, 1001).is_err());

        // Connections: Using >=, so 10 should fail but 9 should be OK
        assert!(limits.validate_connections(9).is_ok());
        assert!(limits.validate_connections(10).is_err());
    }

    /// FALSIFICATION TEST: Verify all validation passes for zero values
    #[test]
    fn test_falsify_zero_values_pass() {
        let limits = ServeLimits::default();

        // Zero headers and zero body should always pass
        assert!(
            limits.validate_request(0, 0).is_ok(),
            "FALSIFICATION FAILED: Zero values should always pass"
        );
        assert!(
            limits.validate_header_size(0).is_ok(),
            "FALSIFICATION FAILED: Zero header size should pass"
        );
        assert!(
            limits.validate_pipelined(0).is_ok(),
            "FALSIFICATION FAILED: Zero pipelined should pass"
        );
        assert!(
            limits.validate_connections(0).is_ok(),
            "FALSIFICATION FAILED: Zero connections should pass"
        );
    }
}
