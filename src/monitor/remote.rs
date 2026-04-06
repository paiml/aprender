//! Multi-system monitoring support.
//!
//! Provides TCP/TLS transport for distributed monitoring with MessagePack serialization.
//!
//! # Components
//!
//! - **Agent**: Runs on monitored nodes, collects metrics, sends to aggregator
//! - **Protocol**: MessagePack-based wire format with <10% overhead
//!
//! # Feature Flags
//!
//! - `monitor-remote`: Basic TCP transport
//! - `monitor-tls`: TLS encryption via rustls

// Placeholder - remote monitoring is under development
