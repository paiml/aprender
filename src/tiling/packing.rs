//! Memory packing layout utilities.

/// Memory layout for packed matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingLayout {
    /// Row-major (C-style)
    RowMajor,
    /// Column-major (Fortran-style)
    ColumnMajor,
    /// Panel-major for A (Goto algorithm)
    PanelMajorA,
    /// Panel-major for B (Goto algorithm)
    PanelMajorB,
}

/// Calculate packed index for panel-major A layout
///
/// Panel-major stores micro-panels contiguously for sequential access.
#[must_use]
#[inline]
pub fn pack_a_index(row: usize, col: usize, mr: usize, kc: usize, _mc: usize) -> usize {
    let panel = row / mr;
    let row_in_panel = row % mr;
    panel * mr * kc + col * mr + row_in_panel
}

/// Calculate packed index for panel-major B layout
#[must_use]
#[inline]
pub fn pack_b_index(row: usize, col: usize, nr: usize, kc: usize, _nc: usize) -> usize {
    let panel = col / nr;
    let col_in_panel = col % nr;
    panel * kc * nr + row * nr + col_in_panel
}

/// Apply XOR swizzling for shared memory bank conflict avoidance
///
/// Pattern: idx_swizzled = idx ^ (idx >> 5) for 32-bank architectures.
#[must_use]
#[inline]
pub fn swizzle_index(idx: usize) -> usize {
    idx ^ (idx >> 5)
}
