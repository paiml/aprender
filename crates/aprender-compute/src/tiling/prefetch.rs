//! Prefetch locality hints for cache optimization.

use super::geometry::{TcbGeometry, TcbLevel};

/// Prefetch locality hint
#[derive(Debug, Clone, Copy)]
pub enum PrefetchLocality {
    /// Non-temporal (streaming, evict soon)
    NonTemporal,
    /// L3 cache
    T2,
    /// L2 cache
    T1,
    /// L1 cache (highest priority)
    T0,
}

/// Calculate optimal prefetch distance based on tile geometry and cache level
///
/// Per Ding & Kennedy (2004): distance = memory_latency / compute_time_per_iter
#[must_use]
pub fn optimal_prefetch_distance(geometry: &TcbGeometry, level: TcbLevel) -> usize {
    // Approximate cycles per micro-tile
    let compute_cycles = geometry.m as usize * geometry.n as usize * geometry.k as usize / 8;

    // Memory latency in cycles (approximate for modern x86)
    let mem_latency = match level {
        TcbLevel::Micro => 4,  // L1: ~4 cycles
        TcbLevel::Midi => 12,  // L2: ~12 cycles
        TcbLevel::Macro => 40, // L3: ~40 cycles
    };

    // Distance = latency / compute_time, minimum 1
    (mem_latency / compute_cycles.max(1)).max(1)
}
