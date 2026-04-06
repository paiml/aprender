//! Tile-level profiling statistics.

use std::time::Instant;

/// Tile-level profiling statistics.
///
/// Tracks per-tile performance metrics for hierarchical cache-blocked operations.
/// Used in conjunction with `TcbGeometry` and `TilingConfig` from the tiling module.
///
/// # Example
///
/// ```ignore
/// let mut profiler = BrickProfiler::new();
/// profiler.enable();
///
/// let tile_timer = profiler.start_tile(TileLevel::Macro, 0, 0);
/// // ... execute tile ...
/// profiler.stop_tile(tile_timer, 1024 * 1024);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TileStats {
    /// Tile level (Macro/Midi/Micro)
    pub level: TileLevel,
    /// Total samples collected
    pub count: u64,
    /// Total elapsed time (nanoseconds)
    pub total_ns: u64,
    /// Min elapsed time (nanoseconds)
    pub min_ns: u64,
    /// Max elapsed time (nanoseconds)
    pub max_ns: u64,
    /// Total elements processed
    pub total_elements: u64,
    /// Total cache misses (estimated)
    pub cache_misses: u64,
    /// Total arithmetic operations
    pub total_flops: u64,
}

/// Tile hierarchy level for profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TileLevel {
    /// Macro-tile: L3 cache / GPU global memory
    #[default]
    Macro,
    /// Midi-tile: L2 cache / GPU shared memory
    Midi,
    /// Micro-tile: Registers / SIMD lanes
    Micro,
}

impl TileLevel {
    /// Get the name of this tile level.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            TileLevel::Macro => "macro",
            TileLevel::Midi => "midi",
            TileLevel::Micro => "micro",
        }
    }
}

impl TileStats {
    /// Create new tile stats for a given level.
    pub fn new(level: TileLevel) -> Self {
        Self {
            level,
            count: 0,
            total_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            total_elements: 0,
            cache_misses: 0,
            total_flops: 0,
        }
    }

    /// Add a sample to statistics.
    pub fn add_sample(&mut self, elapsed_ns: u64, elements: u64, flops: u64) {
        debug_assert!(elements > 0, "CB-BUDGET: tile sample elements must be > 0");
        self.count += 1;
        self.total_ns += elapsed_ns;
        self.min_ns = self.min_ns.min(elapsed_ns);
        self.max_ns = self.max_ns.max(elapsed_ns);
        self.total_elements += elements;
        self.total_flops += flops;
    }

    /// Average time in microseconds.
    #[must_use]
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64 / 1000.0
        }
    }

    /// Throughput in elements/second.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_elements as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Compute throughput in GFLOP/s.
    #[must_use]
    pub fn gflops(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_flops as f64 / (self.total_ns as f64 / 1_000_000_000.0) / 1e9
        }
    }

    /// Arithmetic intensity (FLOP/byte) estimate.
    ///
    /// Assumes 4 bytes per element (f32).
    #[must_use]
    pub fn arithmetic_intensity(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            self.total_flops as f64 / (self.total_elements as f64 * 4.0)
        }
    }

    /// Estimated cache efficiency (0.0-1.0).
    ///
    /// Based on ratio of actual throughput vs theoretical peak.
    #[must_use]
    pub fn cache_efficiency(&self, peak_gflops: f64) -> f64 {
        if peak_gflops <= 0.0 {
            0.0
        } else {
            (self.gflops() / peak_gflops).min(1.0)
        }
    }
}

/// Timer handle for tile-level profiling.
#[derive(Debug)]
pub struct TileTimer {
    /// Tile level
    pub(crate) level: TileLevel,
    /// Row index within parent tile (reserved for spatial analysis)
    pub(crate) _row: u32,
    /// Column index within parent tile (reserved for spatial analysis)
    pub(crate) _col: u32,
    /// Start time
    pub(crate) start: Instant,
}
