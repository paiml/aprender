//! Tile-level profiling methods for BrickProfiler.
//!
//! TILING-SPEC-001: Tile-Level Profiling Support (Phase 15).
//! Extracted from mod.rs to keep file sizes manageable.

use super::BrickProfiler;
use crate::brick::profiler::tile_stats::{TileLevel, TileStats, TileTimer};
use std::time::Instant;

impl BrickProfiler {
    // ========================================================================
    // TILING-SPEC-001: Tile-Level Profiling (Phase 15)
    // ========================================================================

    /// Enable tile-level profiling.
    ///
    /// When enabled, `start_tile()`/`stop_tile()` record per-tile statistics
    /// for Macro/Midi/Micro tile hierarchy.
    pub fn enable_tile_profiling(&mut self) {
        self.tile_profiling_enabled = true;
    }

    /// Disable tile-level profiling.
    pub fn disable_tile_profiling(&mut self) {
        self.tile_profiling_enabled = false;
    }

    /// Check if tile profiling is enabled.
    #[must_use]
    pub fn is_tile_profiling_enabled(&self) -> bool {
        self.tile_profiling_enabled
    }

    /// Start timing a tile execution.
    ///
    /// Returns a `TileTimer` that should be passed to `stop_tile()` after
    /// the tile computation completes.
    ///
    /// # Arguments
    /// - `level`: Tile hierarchy level (Macro/Midi/Micro)
    /// - `row`: Row index within parent tile
    /// - `col`: Column index within parent tile
    ///
    /// # Example
    /// ```rust,ignore
    /// let timer = profiler.start_tile(TileLevel::Macro, 0, 0);
    /// // ... execute tile computation ...
    /// profiler.stop_tile(timer, 256 * 256, 2 * 256 * 256 * 256);
    /// ```
    #[must_use]
    pub fn start_tile(&self, level: TileLevel, row: u32, col: u32) -> TileTimer {
        TileTimer { level, _row: row, _col: col, start: Instant::now() }
    }

    /// Stop timing and record tile statistics.
    ///
    /// # Arguments
    /// - `timer`: Timer handle from `start_tile()`
    /// - `elements`: Number of elements processed by this tile
    /// - `flops`: Number of floating-point operations performed
    pub fn stop_tile(&mut self, timer: TileTimer, elements: u64, flops: u64) {
        if !self.tile_profiling_enabled {
            return;
        }

        let elapsed_ns = timer.start.elapsed().as_nanos() as u64;
        let idx = timer.level as usize;
        self.tile_stats[idx].add_sample(elapsed_ns, elements, flops);
    }

    /// Get tile statistics for a given level.
    #[must_use]
    pub fn tile_stats(&self, level: TileLevel) -> &TileStats {
        &self.tile_stats[level as usize]
    }

    /// Get mutable tile statistics for a given level.
    pub fn tile_stats_mut(&mut self, level: TileLevel) -> &mut TileStats {
        &mut self.tile_stats[level as usize]
    }

    /// Get all tile statistics as a slice.
    #[must_use]
    pub fn all_tile_stats(&self) -> &[TileStats; 3] {
        &self.tile_stats
    }

    /// Reset tile statistics for all levels.
    pub fn reset_tile_stats(&mut self) {
        self.tile_stats = [
            TileStats::new(TileLevel::Macro),
            TileStats::new(TileLevel::Midi),
            TileStats::new(TileLevel::Micro),
        ];
    }
}
