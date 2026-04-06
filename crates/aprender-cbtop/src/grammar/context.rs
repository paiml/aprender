//! Execution context (Coordinates equivalent in Grammar of Graphics).

/// CPU affinity specification
#[derive(Debug, Clone, PartialEq)]
pub struct CpuAffinity {
    /// List of CPU cores to use
    pub cores: Vec<usize>,
}

/// Execution context (analogous to Coordinates)
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionContext {
    /// Local CPU execution
    Cpu {
        affinity: Option<CpuAffinity>,
        numa_node: Option<usize>,
    },
    /// GPU execution
    Gpu { device_id: u32, stream: Option<u32> },
    /// Heterogeneous (multiple contexts)
    Heterogeneous { contexts: Vec<ExecutionContext> },
}

impl Default for ExecutionContext {
    fn default() -> Self {
        ExecutionContext::Cpu {
            affinity: None,
            numa_node: None,
        }
    }
}

impl ExecutionContext {
    /// Create CPU context
    pub fn cpu() -> Self {
        ExecutionContext::Cpu {
            affinity: None,
            numa_node: None,
        }
    }

    /// Create GPU context
    pub fn gpu(device_id: u32) -> Self {
        ExecutionContext::Gpu {
            device_id,
            stream: None,
        }
    }
}
