//! BrickProfiler: Token-Centric Profiling System
//!
//! TILING-SPEC-001: Tile-Level Profiling Support
//!
//! This module provides hierarchical profiling for compute bricks:
//! - Per-brick timing and throughput (PAR-073)
//! - O(1) hot path with BrickId enum (PAR-200)
//! - Tile-level profiling for cache-blocked operations (TILING-SPEC-001)
//! - Kernel checksum capture for divergence detection (CORRECTNESS-011)

mod checksum;
mod tile_stats;

#[cfg(test)]
mod tests;

mod divergence;
mod exec_graph_ext;
mod recording;
mod reporting;
mod tiling;

pub use checksum::{fnv1a_f32_checksum, DivergenceInfo, KernelChecksum};
pub use tile_stats::{TileLevel, TileStats, TileTimer};

use std::time::Instant;

use super::exec_graph::{
    BrickCategory, BrickId, BrickStats, CategoryStats, ExecutionGraph, SyncMode,
};

/// Pending measurement for deferred sync mode.
#[derive(Debug, Clone)]
struct PendingMeasurement {
    /// Brick ID (if known)
    brick_id: Option<BrickId>,
    /// Brick name (for dynamic bricks)
    name: Option<String>,
    /// Start time in nanoseconds (from Instant::now())
    start_ns: u64,
    /// Number of elements processed
    elements: u64,
}

/// Per-brick profiler using pure Rust timing.
///
/// # Design (PAR-073, PAR-200)
///
/// - Uses `std::time::Instant` for timing (no CUDA event FFI)
/// - PAR-200: O(1) hot path with `BrickId` enum + array storage
/// - GPU operations require explicit sync before timing point
/// - Supports deferred sync mode for low-overhead production profiling
/// - Aggregates statistics per brick name
///
/// # Usage
///
/// ```rust,ignore
/// use trueno::brick::{BrickProfiler, BrickId, SyncMode};
///
/// let mut profiler = BrickProfiler::new();
/// profiler.enable();
///
/// // Fast path: use BrickId for known bricks (PAR-200)
/// let timer = profiler.start_brick(BrickId::RmsNorm);
/// // ... do work ...
/// // For GPU: cuda_stream.synchronize() HERE
/// profiler.stop_brick(timer, 1);
///
/// // Legacy path: string-based (slower, for unknown bricks)
/// let timer = profiler.start("CustomBrick");
/// profiler.stop(timer, 1);
///
/// // Deferred sync mode (production)
/// profiler.set_sync_mode(SyncMode::Deferred);
/// profiler.record_deferred(BrickId::RmsNorm, start_ns, 1);
/// // ... more operations ...
/// cuda_stream.synchronize();
/// profiler.finalize(end_ns);
///
/// // Get statistics
/// let stats = profiler.brick_stats(BrickId::RmsNorm);
/// println!("RmsNorm avg: {:.2}µs", stats.avg_us());
///
/// // Get category breakdown
/// let cats = profiler.category_stats();
/// println!("Attention: {:.1}%", cats[BrickCategory::Attention as usize].percentage(profiler.total_ns()));
/// ```
#[derive(Debug)]
pub struct BrickProfiler {
    // PAR-200: Fast path - pre-allocated array for known bricks
    /// Per-brick statistics for known BrickId types (O(1) lookup)
    brick_stats: [BrickStats; BrickId::COUNT],

    // Legacy path - HashMap for dynamic/unknown brick names
    /// Per-brick statistics for unknown brick names (slower, O(1) amortized)
    dynamic_stats: std::collections::HashMap<String, BrickStats>,

    // PAR-200: Deferred sync support
    /// Pending measurements awaiting GPU sync
    pending: Vec<PendingMeasurement>,
    /// Synchronization mode
    sync_mode: SyncMode,
    /// Reference instant for deferred timing
    epoch: Instant,

    /// Whether profiling is enabled
    enabled: bool,
    /// Total tokens processed
    total_tokens: u64,
    /// Total time (ns) across all bricks
    total_ns: u64,
    /// L2 cache hit rate (0.0-1.0) - v1.1.0 OBSERVE phase
    l2_cache_hit_rate: Option<f32>,
    /// Whether zero-copy memory transfers are enabled - v1.1.0 OBSERVE phase
    is_zero_copy: bool,
    /// CORRECTNESS-011: Per-kernel checksums for divergence detection
    kernel_checksums: Vec<KernelChecksum>,

    // PAR-201: Execution path graph
    /// Whether execution graph tracking is enabled
    graph_enabled: bool,
    /// Execution path graph for PTX→kernel→brick relationships
    execution_graph: ExecutionGraph,

    // TILING-SPEC-001: Tile-level profiling
    /// Per-level tile statistics (Macro, Midi, Micro)
    tile_stats: [TileStats; 3],
    /// Whether tile profiling is enabled
    tile_profiling_enabled: bool,
}

/// Timer handle returned by `start()` (legacy string-based API).
#[derive(Debug)]
pub struct BrickTimer {
    /// Brick name
    name: String,
    /// Start time
    start: Instant,
}

/// Timer handle returned by `start_brick()` (PAR-200 fast path).
#[derive(Debug)]
pub struct BrickIdTimer {
    /// Brick ID
    brick_id: BrickId,
    /// Start time
    start: Instant,
}

impl Default for BrickProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickProfiler {
    /// Create a new profiler (disabled by default for zero overhead).
    pub fn new() -> Self {
        Self {
            brick_stats: std::array::from_fn(|i| BrickStats::new(BrickId::ALL[i].name())),
            dynamic_stats: std::collections::HashMap::new(),
            pending: Vec::new(),
            sync_mode: SyncMode::Deferred,
            epoch: Instant::now(),
            enabled: false,
            total_tokens: 0,
            total_ns: 0,
            l2_cache_hit_rate: None,
            is_zero_copy: false,
            kernel_checksums: Vec::new(),
            graph_enabled: false,
            execution_graph: ExecutionGraph::new(),
            tile_stats: [
                TileStats::new(TileLevel::Macro),
                TileStats::new(TileLevel::Midi),
                TileStats::new(TileLevel::Micro),
            ],
            tile_profiling_enabled: false,
        }
    }

    /// Create an enabled profiler.
    pub fn enabled() -> Self {
        let mut profiler = Self::new();
        profiler.enabled = true;
        profiler
    }

    // ========================================================================
    // PAR-200: Sync Mode Configuration
    // ========================================================================

    /// Set the synchronization mode for GPU profiling.
    ///
    /// # Modes
    /// - `Immediate`: Sync after each kernel (accurate but slow)
    /// - `PerLayer`: Sync once per transformer layer
    /// - `Deferred`: Sync once per forward pass (default, fast)
    /// - `None`: No synchronization
    pub fn set_sync_mode(&mut self, mode: SyncMode) {
        contract_pre_sync_verification!();
        self.sync_mode = mode;
    }

    /// Get the current synchronization mode.
    #[must_use]
    pub fn sync_mode(&self) -> SyncMode {
        self.sync_mode
    }

    /// Reset the epoch for deferred timing.
    /// Call this at the start of a forward pass.
    pub fn reset_epoch(&mut self) {
        self.epoch = Instant::now();
    }

    /// Get nanoseconds elapsed since epoch.
    #[inline]
    pub fn elapsed_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    // ========================================================================
    // PAR-200: Fast Path API (BrickId-based)
    // ========================================================================

    /// Start timing a brick using BrickId (O(1) hot path).
    ///
    /// This is the preferred API for known brick types.
    /// For GPU operations, call `stream.synchronize()` before `stop_brick()`.
    #[inline]
    #[must_use]
    pub fn start_brick(&self, brick_id: BrickId) -> BrickIdTimer {
        BrickIdTimer { brick_id, start: Instant::now() }
    }

    /// Stop timing and record the sample (O(1) hot path).
    #[inline]
    pub fn stop_brick(&mut self, timer: BrickIdTimer, elements: u64) {
        if !self.enabled {
            return;
        }

        let elapsed = timer.start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;

        // O(1) array access — CB-BUDGET: bounds-check brick_id
        debug_assert!(
            (timer.brick_id as usize) < self.brick_stats.len(),
            "CB-BUDGET: brick_id {} out of bounds (max {})",
            timer.brick_id as usize,
            self.brick_stats.len()
        );
        let stats = &mut self.brick_stats[timer.brick_id as usize];
        stats.add_sample(elapsed_ns, elements);

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// Get statistics for a known brick type (O(1)).
    #[inline]
    #[must_use]
    pub fn brick_stats(&self, brick_id: BrickId) -> &BrickStats {
        contract_pre_brick_ordering!();
        &self.brick_stats[brick_id as usize]
    }

    /// Get mutable statistics for a known brick type (O(1)).
    #[inline]
    pub fn brick_stats_mut(&mut self, brick_id: BrickId) -> &mut BrickStats {
        &mut self.brick_stats[brick_id as usize]
    }

    // ========================================================================
    // PAR-200: Deferred Sync API
    // ========================================================================

    /// Record a measurement without GPU sync (deferred mode).
    ///
    /// Call `finalize()` after GPU sync to apply all pending measurements.
    ///
    /// # Arguments
    /// - `brick_id`: The brick type
    /// - `start_ns`: Start time (from `elapsed_ns()` at operation start)
    /// - `elements`: Number of elements processed
    #[inline]
    pub fn record_deferred(&mut self, brick_id: BrickId, start_ns: u64, elements: u64) {
        if !self.enabled {
            return;
        }
        self.pending.push(PendingMeasurement {
            brick_id: Some(brick_id),
            name: None,
            start_ns,
            elements,
        });
    }

    /// Record a measurement for a dynamic brick (deferred mode).
    #[inline]
    pub fn record_deferred_dynamic(&mut self, name: &str, start_ns: u64, elements: u64) {
        if !self.enabled {
            return;
        }
        self.pending.push(PendingMeasurement {
            brick_id: BrickId::from_str(name),
            name: Some(name.to_string()),
            start_ns,
            elements,
        });
    }

    /// Finalize all pending measurements after GPU sync.
    ///
    /// Must be called after `stream.synchronize()` to get accurate timing.
    ///
    /// # Arguments
    /// - `end_ns`: End time (from `elapsed_ns()` after sync)
    pub fn finalize(&mut self, end_ns: u64) {
        if self.pending.is_empty() {
            return;
        }

        // Calculate elapsed time for each pending measurement
        for m in self.pending.drain(..) {
            let elapsed_ns = end_ns.saturating_sub(m.start_ns);

            if let Some(brick_id) = m.brick_id {
                // Fast path: known brick
                let stats = &mut self.brick_stats[brick_id as usize];
                stats.add_sample(elapsed_ns, m.elements);
            } else if let Some(name) = m.name {
                // Fallback path: dynamic brick lookup
                let stats = self
                    .dynamic_stats
                    .entry(name.clone())
                    .or_insert_with(|| BrickStats::new(&name));
                stats.add_sample(elapsed_ns, m.elements);
            }

            self.total_tokens += m.elements;
            self.total_ns += elapsed_ns;
        }
    }

    /// Check if there are pending measurements.
    #[inline]
    #[must_use]
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get number of pending measurements.
    #[inline]
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    // ========================================================================
    // PAR-200: Category Aggregation
    // ========================================================================

    /// Get aggregated statistics by category.
    ///
    /// Returns an array indexed by `BrickCategory as usize`.
    #[must_use]
    pub fn category_stats(&self) -> [CategoryStats; BrickCategory::COUNT] {
        let mut result = [CategoryStats::default(); BrickCategory::COUNT];

        for (i, stats) in self.brick_stats.iter().enumerate() {
            let brick_id = BrickId::ALL[i];
            let cat = brick_id.category() as usize;
            result[cat].total_ns += stats.total_ns;
            result[cat].total_elements += stats.total_elements;
            result[cat].count += stats.count;
        }

        // Include dynamic stats in "Other" category
        for stats in self.dynamic_stats.values() {
            let cat = BrickCategory::Other as usize;
            result[cat].total_ns += stats.total_ns;
            result[cat].total_elements += stats.total_elements;
            result[cat].count += stats.count;
        }

        result
    }

    /// Set L2 cache hit rate (v1.1.0 OBSERVE phase)
    pub fn set_l2_cache_hit_rate(&mut self, rate: f32) {
        self.l2_cache_hit_rate = Some(rate.clamp(0.0, 1.0));
    }

    /// Get L2 cache hit rate
    pub fn l2_cache_hit_rate(&self) -> Option<f32> {
        self.l2_cache_hit_rate
    }

    /// Set zero-copy mode (v1.1.0 OBSERVE phase)
    pub fn set_zero_copy(&mut self, enabled: bool) {
        self.is_zero_copy = enabled;
    }

    /// Check if zero-copy is enabled
    pub fn is_zero_copy(&self) -> bool {
        self.is_zero_copy
    }

    /// Enable profiling.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get total throughput across all bricks.
    #[must_use]
    pub fn total_throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_tokens as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Get total tokens processed.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        contract_pre_token_accounting!();
        self.total_tokens
    }

    /// Get total time in nanoseconds.
    #[must_use]
    pub fn total_ns(&self) -> u64 {
        contract_pre_wall_coverage!();
        self.total_ns
    }
}
