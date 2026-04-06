//! Safety Integrity Level (IEC 61508)

use serde::{Deserialize, Serialize};

/// Safety Integrity Level (IEC 61508)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyIntegrityLevel {
    /// QM: No safety requirements (games, entertainment)
    /// - Ring buffer traces
    /// - Best-effort logging
    QM,

    /// SIL 1: Low safety requirements
    /// - Persistent traces
    /// - Hash verification
    SIL1,

    /// SIL 2: Medium safety requirements
    /// - Hash chain
    /// - Redundant storage
    SIL2,

    /// SIL 3: High safety requirements (automotive ASIL C)
    /// - Hash chain with signatures
    /// - Triple redundant storage
    /// - Hardware security module
    SIL3,

    /// SIL 4: Highest safety requirements (automotive ASIL D)
    /// - All SIL 3 requirements
    /// - Formal verification of trace system
    /// - Independent safety monitor
    SIL4,
}

impl SafetyIntegrityLevel {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            SafetyIntegrityLevel::QM => "QM",
            SafetyIntegrityLevel::SIL1 => "SIL1",
            SafetyIntegrityLevel::SIL2 => "SIL2",
            SafetyIntegrityLevel::SIL3 => "SIL3",
            SafetyIntegrityLevel::SIL4 => "SIL4",
        }
    }

    /// Get minimum confidence threshold for this level
    pub fn min_confidence(&self) -> f32 {
        match self {
            SafetyIntegrityLevel::QM => 0.0, // No requirement
            SafetyIntegrityLevel::SIL1 => 0.5,
            SafetyIntegrityLevel::SIL2 => 0.7,
            SafetyIntegrityLevel::SIL3 => 0.8,
            SafetyIntegrityLevel::SIL4 => 0.9,
        }
    }

    /// Get maximum allowed latency in nanoseconds
    pub fn max_latency_ns(&self) -> u64 {
        match self {
            SafetyIntegrityLevel::QM => u64::MAX,      // No requirement
            SafetyIntegrityLevel::SIL1 => 100_000_000, // 100ms
            SafetyIntegrityLevel::SIL2 => 50_000_000,  // 50ms
            SafetyIntegrityLevel::SIL3 => 10_000_000,  // 10ms
            SafetyIntegrityLevel::SIL4 => 1_000_000,   // 1ms
        }
    }
}
