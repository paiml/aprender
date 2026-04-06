//! Error types for the `trueno-cuda-edge` crate.
//!
//! All edge-case test failures funnel through [`EdgeError`], which unifies
//! null-pointer violations, shared-memory overflows, quantization mismatches,
//! PTX verification failures, lifecycle faults, and supervision crashes.

use std::fmt;

/// Unified error type for GPU edge-case testing.
#[derive(Debug, thiserror::Error)]
pub enum EdgeError {
    /// A null device pointer was encountered where a valid pointer is required.
    #[error("null device pointer: {context}")]
    NullPointer {
        /// Human-readable context for the null pointer.
        context: String,
    },

    /// Shared memory allocation exceeds hardware limits.
    #[error("shared memory overflow: requested {requested} bytes, limit {limit} bytes")]
    SharedMemoryOverflow {
        /// Bytes requested.
        requested: u64,
        /// Hardware limit in bytes.
        limit: u64,
    },

    /// A quantization parity violation was detected.
    #[error("quantization parity violation: {message}")]
    QuantizationParity {
        /// Description of the violation.
        message: String,
    },

    /// PTX verification failed.
    #[error("PTX verification failed: {reason}")]
    PtxVerification {
        /// Reason for failure.
        reason: String,
    },

    /// A lifecycle chaos scenario triggered a fault.
    #[error("lifecycle fault: {scenario}")]
    LifecycleFault {
        /// The scenario that caused the fault.
        scenario: String,
    },

    /// A supervised worker crashed beyond the restart budget.
    #[error("supervisor: worker exhausted restart budget ({restarts} restarts)")]
    SupervisionExhausted {
        /// Number of restarts attempted.
        restarts: u32,
    },

    /// A boundary sentinel was corrupted.
    #[error("boundary sentinel corrupted at offset {offset}")]
    SentinelCorruption {
        /// Byte offset of the corrupted sentinel.
        offset: u64,
    },

    /// A bank conflict was detected in shared memory.
    #[error("bank conflict: {detail}")]
    BankConflict {
        /// Detail of the conflict.
        detail: String,
    },

    /// A context resource leak was detected.
    #[error("context leak: {leaked_bytes} bytes leaked across {context_count} context(s)")]
    ContextLeak {
        /// Total bytes leaked.
        leaked_bytes: u64,
        /// Number of contexts that leaked.
        context_count: u32,
    },

    /// PTX mutation was not detected (mutation survived).
    #[error("mutation survived: {mutation}")]
    MutationSurvived {
        /// Description of the surviving mutation.
        mutation: String,
    },
}

/// Convenience alias for `Result<T, EdgeError>`.
pub type Result<T> = std::result::Result<T, EdgeError>;

/// Classification of error severity for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    /// Informational — the test found something noteworthy but not a defect.
    Info,
    /// Warning — potential issue that may indicate a latent bug.
    Warning,
    /// Error — confirmed defect or invariant violation.
    Error,
    /// Critical — the GPU may be in an unrecoverable state.
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_pointer_error_display() {
        let err = EdgeError::NullPointer {
            context: "kernel arg 0".into(),
        };
        assert!(err.to_string().contains("null device pointer"));
        assert!(err.to_string().contains("kernel arg 0"));
    }

    #[test]
    fn shared_memory_overflow_display() {
        let err = EdgeError::SharedMemoryOverflow {
            requested: 65536,
            limit: 49152,
        };
        let msg = err.to_string();
        assert!(msg.contains("65536"));
        assert!(msg.contains("49152"));
    }

    #[test]
    fn quantization_parity_display() {
        let err = EdgeError::QuantizationParity {
            message: "Q4_K drift exceeds 0.01".into(),
        };
        assert!(err.to_string().contains("Q4_K drift"));
    }

    #[test]
    fn ptx_verification_display() {
        let err = EdgeError::PtxVerification {
            reason: "missing .version directive".into(),
        };
        assert!(err.to_string().contains("missing .version"));
    }

    #[test]
    fn lifecycle_fault_display() {
        let err = EdgeError::LifecycleFault {
            scenario: "double destroy".into(),
        };
        assert!(err.to_string().contains("double destroy"));
    }

    #[test]
    fn supervision_exhausted_display() {
        let err = EdgeError::SupervisionExhausted { restarts: 5 };
        assert!(err.to_string().contains("5 restarts"));
    }

    #[test]
    fn sentinel_corruption_display() {
        let err = EdgeError::SentinelCorruption { offset: 4096 };
        assert!(err.to_string().contains("4096"));
    }

    #[test]
    fn bank_conflict_display() {
        let err = EdgeError::BankConflict {
            detail: "32-way conflict on bank 0".into(),
        };
        assert!(err.to_string().contains("32-way"));
    }

    #[test]
    fn context_leak_display() {
        let err = EdgeError::ContextLeak {
            leaked_bytes: 1_048_576,
            context_count: 2,
        };
        let msg = err.to_string();
        assert!(msg.contains("1048576"));
        assert!(msg.contains("2 context(s)"));
    }

    #[test]
    fn mutation_survived_display() {
        let err = EdgeError::MutationSurvived {
            mutation: "flipped add→sub".into(),
        };
        assert!(err.to_string().contains("flipped add"));
    }

    #[test]
    fn severity_display() {
        assert_eq!(Severity::Info.to_string(), "INFO");
        assert_eq!(Severity::Warning.to_string(), "WARN");
        assert_eq!(Severity::Error.to_string(), "ERROR");
        assert_eq!(Severity::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn severity_equality() {
        assert_eq!(Severity::Info, Severity::Info);
        assert_ne!(Severity::Info, Severity::Critical);
    }
}
