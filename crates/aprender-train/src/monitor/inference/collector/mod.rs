//! Trace Collectors (ENT-105, ENT-106, ENT-107)
//!
//! Strategies for collecting decision traces:
//! - RingCollector: Stack-allocated, <100ns, for games/drones
//! - StreamCollector: Write-through, <1µs, for persistent logging
//! - HashChainCollector: SHA-256 chain, <10µs, for safety-critical

mod hash_chain;
mod ring;
mod stream;
mod traits;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use hash_chain::{ChainEntry, ChainVerification, HashChainCollector};
pub use ring::RingCollector;
pub use stream::{StreamCollector, StreamFormat};
pub use traits::TraceCollector;
