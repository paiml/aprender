//! Error types for trueno-cupti.

use std::fmt;

/// Result type for CUPTI operations.
pub type CuptiResult<T> = Result<T, CuptiError>;

/// Errors from CUPTI operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CuptiError {
    /// CUPTI library not available on this system.
    NotAvailable,

    /// CUPTI initialization failed.
    InitializationFailed,

    /// Invalid parameter passed to CUPTI.
    InvalidParameter,

    /// CUPTI out of memory.
    OutOfMemory,

    /// Activity kind not supported.
    ActivityNotSupported,

    /// Metric not supported on this device.
    MetricNotSupported,

    /// Profiling is already active.
    AlreadyActive,

    /// Profiling is not active.
    NotActive,

    /// Buffer too small for requested data.
    BufferTooSmall,

    /// Hardware resource limit exceeded.
    HardwareLimitExceeded,

    /// CUDA driver error.
    CudaError(i32),

    /// Unknown CUPTI error code.
    Unknown(i32),
}

impl fmt::Display for CuptiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CuptiError::NotAvailable => write!(f, "CUPTI not available on this system"),
            CuptiError::InitializationFailed => write!(f, "CUPTI initialization failed"),
            CuptiError::InvalidParameter => write!(f, "Invalid parameter"),
            CuptiError::OutOfMemory => write!(f, "Out of memory"),
            CuptiError::ActivityNotSupported => write!(f, "Activity kind not supported"),
            CuptiError::MetricNotSupported => write!(f, "Metric not supported on this device"),
            CuptiError::AlreadyActive => write!(f, "Profiling is already active"),
            CuptiError::NotActive => write!(f, "Profiling is not active"),
            CuptiError::BufferTooSmall => write!(f, "Buffer too small"),
            CuptiError::HardwareLimitExceeded => write!(f, "Hardware resource limit exceeded"),
            CuptiError::CudaError(code) => write!(f, "CUDA driver error: {}", code),
            CuptiError::Unknown(code) => write!(f, "Unknown CUPTI error: {}", code),
        }
    }
}

impl std::error::Error for CuptiError {}

impl CuptiError {
    /// Create error from CUPTI result code.
    pub fn from_cupti_result(code: i32) -> CuptiResult<()> {
        match code {
            0 => Ok(()), // CUPTI_SUCCESS
            1 => Err(CuptiError::InvalidParameter),
            2 => Err(CuptiError::OutOfMemory),
            3 => Err(CuptiError::InitializationFailed),
            _ => Err(CuptiError::Unknown(code)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CuptiError::NotAvailable;
        assert_eq!(format!("{}", err), "CUPTI not available on this system");
    }

    #[test]
    fn test_from_cupti_result() {
        assert!(CuptiError::from_cupti_result(0).is_ok());
        assert!(CuptiError::from_cupti_result(1).is_err());
    }
}
