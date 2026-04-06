//! Remote SSH/Headless Agent Integration (PMAT-044)
//!
//! Distributed benchmark collection from remote hosts via SSH.
//!
//! # Design
//!
//! - SSH-based command execution with connection pooling
//! - Multi-host result aggregation with statistical merging
//! - Host health monitoring and automatic failover
//! - Secure credential handling (no plaintext storage)
//!
//! # Falsification (FKR-045)
//!
//! H₀: Remote agent cannot collect metrics from heterogeneous hosts
//! Test: Connect to 3+ hosts with different architectures, verify metric aggregation

mod agent;
mod json;
mod metrics;
mod types;

pub use agent::RemoteAgent;
pub use types::{
    AggregatedResult, AggregationStrategy, AuthMethod, CommandResult, HostBenchmark, HostConfig,
    HostHealth, HostState, RemoteAgentConfig, RemoteError, RemoteResult,
    DEFAULT_HEALTH_CHECK_INTERVAL_SEC, DEFAULT_MAX_CONCURRENT, DEFAULT_RETRY_DELAY_MS,
};

#[cfg(test)]
mod tests;
