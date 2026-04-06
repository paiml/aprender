//! GPU sharing error types (GPU-SHARE-001/002/003).

use std::fmt;

/// Errors from GPU resource management.
#[derive(Debug)]
pub enum GpuError {
    /// Not enough VRAM for requested budget.
    InsufficientMemory {
        budget_mb: usize,
        available_mb: usize,
        reserved_mb: usize,
        total_mb: usize,
    },
    /// Wait-for-VRAM timed out.
    Timeout { budget_mb: usize, timeout_secs: u64 },
    /// Ledger file is corrupt or unreadable.
    LedgerCorrupt(String),
    /// I/O error accessing ledger.
    Io(std::io::Error),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientMemory { budget_mb, available_mb, reserved_mb, total_mb } => {
                write!(
                    f,
                    "insufficient VRAM: need {budget_mb} MB, available {available_mb} MB \
                     (reserved {reserved_mb} MB / total {total_mb} MB)"
                )
            }
            Self::Timeout { budget_mb, timeout_secs } => {
                write!(f, "VRAM wait timed out after {timeout_secs}s (need {budget_mb} MB)")
            }
            Self::LedgerCorrupt(msg) => write!(f, "ledger corrupt: {msg}"),
            Self::Io(e) => write!(f, "GPU ledger I/O: {e}"),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GpuError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_error_insufficient_memory_display() {
        let err = GpuError::InsufficientMemory {
            budget_mb: 8000,
            available_mb: 4000,
            reserved_mb: 12000,
            total_mb: 16000,
        };
        let msg = format!("{err}");
        assert!(msg.contains("8000"));
        assert!(msg.contains("4000"));
        assert!(msg.contains("12000"));
        assert!(msg.contains("16000"));
    }

    #[test]
    fn test_gpu_error_timeout_display() {
        let err = GpuError::Timeout { budget_mb: 4000, timeout_secs: 30 };
        let msg = format!("{err}");
        assert!(msg.contains("30"));
        assert!(msg.contains("4000"));
    }

    #[test]
    fn test_gpu_error_ledger_corrupt_display() {
        let err = GpuError::LedgerCorrupt("bad checksum".into());
        let msg = format!("{err}");
        assert!(msg.contains("bad checksum"));
    }

    #[test]
    fn test_gpu_error_io_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = GpuError::Io(io_err);
        let msg = format!("{err}");
        assert!(msg.contains("file missing"));
    }

    #[test]
    fn test_gpu_error_source_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let err = GpuError::Io(io_err);
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn test_gpu_error_source_non_io() {
        let err = GpuError::Timeout { budget_mb: 100, timeout_secs: 5 };
        assert!(std::error::Error::source(&err).is_none());

        let err = GpuError::LedgerCorrupt("test".into());
        assert!(std::error::Error::source(&err).is_none());

        let err = GpuError::InsufficientMemory {
            budget_mb: 1,
            available_mb: 0,
            reserved_mb: 1,
            total_mb: 1,
        };
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn test_gpu_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no access");
        let err: GpuError = io_err.into();
        match err {
            GpuError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied),
            other => panic!("Expected Io variant, got: {other:?}"),
        }
    }

    #[test]
    fn test_gpu_error_debug() {
        let err = GpuError::Timeout { budget_mb: 100, timeout_secs: 5 };
        let debug = format!("{err:?}");
        assert!(debug.contains("Timeout"));
    }
}
