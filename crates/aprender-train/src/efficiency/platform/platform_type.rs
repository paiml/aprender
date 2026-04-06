//! Platform efficiency abstraction enum.

use serde::{Deserialize, Serialize};

use super::edge::EdgeEfficiency;
use super::server::ServerEfficiency;

/// Platform efficiency abstraction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlatformEfficiency {
    /// Server/cloud deployment
    Server(ServerEfficiency),
    /// Edge/embedded deployment
    Edge(EdgeEfficiency),
}

impl PlatformEfficiency {
    /// Check if this is a server deployment
    pub fn is_server(&self) -> bool {
        matches!(self, Self::Server(_))
    }

    /// Check if this is an edge deployment
    pub fn is_edge(&self) -> bool {
        matches!(self, Self::Edge(_))
    }

    /// Get throughput in samples per second
    pub fn throughput(&self) -> f64 {
        match self {
            Self::Server(s) => s.throughput_samples_per_sec,
            Self::Edge(e) => e.max_throughput_per_sec(),
        }
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        match self {
            Self::Server(_) => 0, // Server typically has abundant memory
            Self::Edge(e) => e.memory_footprint_bytes,
        }
    }
}
