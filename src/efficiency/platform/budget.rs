//! WASM deployment budget constraints and violations.

use serde::{Deserialize, Serialize};

/// Maximum startup latency for mobile web targets (milliseconds)
const MOBILE_MAX_STARTUP_MS: u64 = 200;
/// Default maximum startup latency for WASM deployment (milliseconds)
const DEFAULT_MAX_STARTUP_MS: u64 = 500;

/// WASM deployment budget constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WasmBudget {
    /// Maximum binary size in bytes (default: 5 MB)
    pub max_binary_size: u64,
    /// Maximum startup latency in milliseconds (default: 500 ms)
    pub max_startup_ms: u64,
    /// Maximum memory usage in bytes (default: 256 MB)
    pub max_memory_bytes: u64,
}

impl WasmBudget {
    /// Create new WASM budget
    pub fn new(max_binary_size: u64, max_startup_ms: u64, max_memory_bytes: u64) -> Self {
        Self { max_binary_size, max_startup_ms, max_memory_bytes }
    }

    /// Create a strict budget for mobile web
    pub fn mobile() -> Self {
        Self {
            max_binary_size: 2 * 1024 * 1024, // 2 MB
            max_startup_ms: MOBILE_MAX_STARTUP_MS,
            max_memory_bytes: 128 * 1024 * 1024, // 128 MB
        }
    }

    /// Create a standard budget for desktop web
    pub fn desktop() -> Self {
        Self {
            max_binary_size: 10 * 1024 * 1024,   // 10 MB
            max_startup_ms: 1000,                // 1 second
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
        }
    }

    /// Create a relaxed budget for embedded/IoT
    pub fn embedded() -> Self {
        Self {
            max_binary_size: 1024 * 1024,       // 1 MB
            max_startup_ms: 100,                // 100 ms
            max_memory_bytes: 64 * 1024 * 1024, // 64 MB
        }
    }

    /// Get budget sizes in MB
    pub fn sizes_mb(&self) -> (f64, f64) {
        (
            self.max_binary_size as f64 / (1024.0 * 1024.0),
            self.max_memory_bytes as f64 / (1024.0 * 1024.0),
        )
    }
}

impl Default for WasmBudget {
    fn default() -> Self {
        Self {
            max_binary_size: 5 * 1024 * 1024, // 5 MB
            max_startup_ms: DEFAULT_MAX_STARTUP_MS,
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
        }
    }
}

/// Budget violation details
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetViolation {
    /// Binary size exceeds limit
    BinarySize { actual: u64, limit: u64 },
    /// Startup latency exceeds limit
    StartupLatency { actual: u64, limit: u64 },
    /// Memory footprint exceeds limit
    MemoryFootprint { actual: u64, limit: u64 },
}

impl std::fmt::Display for BudgetViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BinarySize { actual, limit } => {
                write!(
                    f,
                    "Binary size {} MB exceeds {} MB limit",
                    *actual as f64 / (1024.0 * 1024.0),
                    *limit as f64 / (1024.0 * 1024.0)
                )
            }
            Self::StartupLatency { actual, limit } => {
                write!(f, "Startup latency {actual} ms exceeds {limit} ms limit")
            }
            Self::MemoryFootprint { actual, limit } => {
                write!(
                    f,
                    "Memory footprint {} MB exceeds {} MB limit",
                    *actual as f64 / (1024.0 * 1024.0),
                    *limit as f64 / (1024.0 * 1024.0)
                )
            }
        }
    }
}
