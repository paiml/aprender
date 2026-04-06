//! TCB index calculation utilities.

use super::config::TilingConfig;

/// Index calculator for hierarchical tiling
///
/// Converts between linear indices and (row, col) coordinates at each tiling level.
#[derive(Debug, Clone)]
pub struct TcbIndexCalculator {
    /// Tiling configuration
    pub config: TilingConfig,
    /// Problem dimensions
    problem_m: u32,
    problem_n: u32,
    problem_k: u32,
}

impl TcbIndexCalculator {
    /// Create a new index calculator for the given problem size
    #[must_use]
    pub fn new(config: TilingConfig, m: u32, n: u32, k: u32) -> Self {
        Self { config, problem_m: m, problem_n: n, problem_k: k }
    }

    /// Get macro-tile offset for a given block index
    ///
    /// Returns (row_offset, col_offset) in the output matrix.
    #[must_use]
    pub fn macro_tile_offset(&self, block_idx: u32) -> (u32, u32) {
        let tiles_per_row =
            (self.problem_n + self.config.macro_tile.n - 1) / self.config.macro_tile.n;
        let row = (block_idx / tiles_per_row) * self.config.macro_tile.m;
        let col = (block_idx % tiles_per_row) * self.config.macro_tile.n;
        (row, col)
    }

    /// Get midi-tile offset within a macro-tile
    #[must_use]
    pub fn midi_tile_offset(&self, midi_idx: u32) -> (u32, u32) {
        let tiles_per_row = self.config.macro_tile.n / self.config.midi_tile.n;
        let row = (midi_idx / tiles_per_row) * self.config.midi_tile.m;
        let col = (midi_idx % tiles_per_row) * self.config.midi_tile.n;
        (row, col)
    }

    /// Get micro-tile offset within a midi-tile
    #[must_use]
    pub fn micro_tile_offset(&self, micro_idx: u32) -> (u32, u32) {
        let tiles_per_row = self.config.midi_tile.n / self.config.micro_tile.n;
        let row = (micro_idx / tiles_per_row) * self.config.micro_tile.m;
        let col = (micro_idx % tiles_per_row) * self.config.micro_tile.n;
        (row, col)
    }

    /// Convert block index to linear memory offset
    ///
    /// For row-major C matrix with given stride.
    #[must_use]
    #[inline]
    pub fn block_to_linear_offset(&self, block_idx: u32, stride: u32) -> usize {
        let (row, col) = self.macro_tile_offset(block_idx);
        (row * stride + col) as usize
    }

    /// Calculate A matrix offset for K-dimension blocking
    #[must_use]
    #[inline]
    pub fn a_offset(&self, macro_row: u32, k_block: u32) -> usize {
        let row = macro_row * self.config.macro_tile.m;
        let col = k_block * self.config.macro_tile.k;
        (row * self.problem_k + col) as usize
    }

    /// Calculate B matrix offset for K-dimension blocking
    #[must_use]
    #[inline]
    pub fn b_offset(&self, k_block: u32, macro_col: u32) -> usize {
        let row = k_block * self.config.macro_tile.k;
        let col = macro_col * self.config.macro_tile.n;
        (row * self.problem_n + col) as usize
    }

    /// Get number of K blocks needed
    #[must_use]
    pub fn num_k_blocks(&self) -> u32 {
        (self.problem_k + self.config.macro_tile.k - 1) / self.config.macro_tile.k
    }

    /// Check if this is a boundary tile (may need masking)
    #[must_use]
    pub fn is_boundary_tile(&self, block_idx: u32) -> bool {
        let (row, col) = self.macro_tile_offset(block_idx);
        row + self.config.macro_tile.m > self.problem_m
            || col + self.config.macro_tile.n > self.problem_n
    }

    /// Get actual tile dimensions (may be smaller at boundaries)
    #[must_use]
    pub fn actual_tile_dims(&self, block_idx: u32) -> (u32, u32) {
        let (row, col) = self.macro_tile_offset(block_idx);
        let actual_m = (self.problem_m - row).min(self.config.macro_tile.m);
        let actual_n = (self.problem_n - col).min(self.config.macro_tile.n);
        (actual_m, actual_n)
    }
}
