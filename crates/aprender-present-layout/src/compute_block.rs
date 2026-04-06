#![allow(clippy::cast_lossless)] // u16 to u32/f32 casts are intentional and always safe
//! ComputeBlock Grid Compositor
//!
//! Solves two critical TUI layout issues:
//! - **Issue A**: Automatic space utilization via intrinsic sizing
//! - **Issue B**: Artifact prevention via cell ownership and clipping
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                      Frame Compositor                          │
//! ├────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐ │
//! │  │ GridLayout   │───▶│ ComputeBlock │───▶│ ClippedRenderer  │ │
//! │  │              │    │              │    │                  │ │
//! │  │ - Define NxM │    │ - claim(r,c) │    │ - clip to bounds │ │
//! │  │ - gutters    │    │ - bounds()   │    │ - z-order        │ │
//! │  │ - flex sizes │    │ - clear()    │    │ - no overflow    │ │
//! │  └──────────────┘    └──────────────┘    └──────────────────┘ │
//! └────────────────────────────────────────────────────────────────┘
//! ```

use crate::grid::{compute_grid_layout, GridArea, GridTemplate};
use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// INTRINSIC SIZING (Issue A)
// ============================================================================

/// Size in terminal cells (u16 for compatibility with ratatui).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Size {
    /// Width in terminal columns.
    pub width: u16,
    /// Height in terminal rows.
    pub height: u16,
}

impl Size {
    /// Create a new size.
    #[must_use]
    pub const fn new(width: u16, height: u16) -> Self {
        Self { width, height }
    }

    /// Zero size.
    pub const ZERO: Self = Self {
        width: 0,
        height: 0,
    };
}

/// Rectangle in terminal coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    /// X position (column).
    pub x: u16,
    /// Y position (row).
    pub y: u16,
    /// Width in columns.
    pub width: u16,
    /// Height in rows.
    pub height: u16,
}

impl Rect {
    /// Create a new rectangle.
    #[must_use]
    pub const fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Calculate the intersection of two rectangles.
    #[must_use]
    pub fn intersection(&self, other: Self) -> Self {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x2 > x1 && y2 > y1 {
            Self {
                x: x1,
                y: y1,
                width: x2 - x1,
                height: y2 - y1,
            }
        } else {
            Self::default()
        }
    }

    /// Check if a point is within this rectangle.
    #[must_use]
    pub const fn contains(&self, x: u16, y: u16) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    /// Get the area of this rectangle.
    #[must_use]
    pub const fn area(&self) -> u32 {
        self.width as u32 * self.height as u32
    }
}

/// Size hints for content-aware layout.
///
/// Widgets report their sizing requirements through this struct,
/// allowing the layout engine to make intelligent decisions about
/// space allocation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SizeHint {
    /// Minimum size needed to render at all.
    pub min: Size,
    /// Preferred size for comfortable rendering.
    pub preferred: Size,
    /// Maximum useful size (content won't expand beyond).
    pub max: Option<Size>,
}

impl SizeHint {
    /// Create a new size hint.
    #[must_use]
    pub const fn new(min: Size, preferred: Size, max: Option<Size>) -> Self {
        Self {
            min,
            preferred,
            max,
        }
    }

    /// Create a fixed-size hint (all sizes equal).
    #[must_use]
    pub const fn fixed(size: Size) -> Self {
        Self {
            min: size,
            preferred: size,
            max: Some(size),
        }
    }

    /// Create a flexible hint with only minimum.
    #[must_use]
    pub const fn flexible(min: Size) -> Self {
        Self {
            min,
            preferred: min,
            max: None,
        }
    }
}

/// Extended constraint with Fill support.
///
/// This extends the standard constraint system with:
/// - `Fill`: Distributes remaining space proportionally
/// - `Content`: Uses widget's `SizeHint` for sizing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlexConstraint {
    /// Fixed size in terminal cells.
    Fixed(u16),
    /// Minimum size (can grow).
    Min(u16),
    /// Maximum size (can shrink).
    Max(u16),
    /// Percentage of parent (0-100).
    Percentage(u16),
    /// Ratio of remaining space (numerator, denominator).
    Ratio(u16, u16),
    /// Fill remaining space with weight.
    ///
    /// Multiple Fill constraints share remaining space
    /// proportionally to their weights.
    Fill(u16),
    /// Content-based: use widget's SizeHint.
    Content,
}

impl Default for FlexConstraint {
    fn default() -> Self {
        Self::Fill(1)
    }
}

/// Trait for widgets with intrinsic sizing.
pub trait IntrinsicSize {
    /// Report size requirements given available space.
    fn size_hint(&self, available: Size) -> SizeHint;
}

// ============================================================================
// GRID COMPOSITOR (Issue B)
// ============================================================================

/// A named region in the grid with ownership semantics.
///
/// ComputeBlocks prevent rendering conflicts by:
/// 1. Claiming exclusive ownership of grid cells
/// 2. Enforcing clipping at render time
/// 3. Supporting z-ordering for overlays
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeBlock {
    /// Unique name for this block.
    pub name: String,
    /// Grid area this block occupies.
    pub area: GridArea,
    /// Z-order for overlapping blocks (higher = on top).
    pub z_index: i16,
    /// Whether this block is visible.
    pub visible: bool,
    /// Clipping mode.
    pub clip: ClipMode,
}

impl ComputeBlock {
    /// Create a new compute block.
    #[must_use]
    pub fn new(name: impl Into<String>, area: GridArea) -> Self {
        Self {
            name: name.into(),
            area,
            z_index: 0,
            visible: true,
            clip: ClipMode::default(),
        }
    }

    /// Set z-index.
    #[must_use]
    pub const fn with_z_index(mut self, z_index: i16) -> Self {
        self.z_index = z_index;
        self
    }

    /// Set visibility.
    #[must_use]
    pub const fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Set clip mode.
    #[must_use]
    pub const fn with_clip(mut self, clip: ClipMode) -> Self {
        self.clip = clip;
        self
    }
}

/// Clipping behavior for blocks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClipMode {
    /// Render only within bounds (default, prevents artifacts).
    #[default]
    Strict,
    /// Allow overflow (for tooltips, dropdowns).
    Overflow,
    /// Scroll if content exceeds bounds.
    Scroll,
}

/// Grid compositor managing block ownership.
///
/// The compositor ensures:
/// - No two blocks claim the same cell
/// - Blocks are rendered in z-order
/// - Dirty regions are tracked for efficient redraw
#[derive(Debug, Clone)]
pub struct GridCompositor {
    /// Grid template definition.
    template: GridTemplate,
    /// Registered blocks.
    blocks: Vec<ComputeBlock>,
    /// Cell ownership map: (row, col) -> block index.
    ownership: Vec<Vec<Option<usize>>>,
    /// Dirty rectangles for incremental redraw.
    dirty: Vec<Rect>,
}

impl GridCompositor {
    /// Create a new compositor with the given template.
    #[must_use]
    pub fn new(template: GridTemplate) -> Self {
        let rows = template.row_count().max(1);
        let cols = template.column_count().max(1);
        Self {
            template,
            blocks: Vec::new(),
            ownership: vec![vec![None; cols]; rows],
            dirty: Vec::new(),
        }
    }

    /// Get the grid template.
    #[must_use]
    pub fn template(&self) -> &GridTemplate {
        &self.template
    }

    /// Register a block, claiming grid cells.
    ///
    /// Returns the block index on success, or an error if:
    /// - The block area is out of grid bounds
    /// - The block overlaps with an existing block
    pub fn register(&mut self, block: ComputeBlock) -> Result<usize, CompositorError> {
        // Validate area is within grid bounds
        if block.area.col_end > self.template.column_count() {
            return Err(CompositorError::OutOfBounds {
                block: block.name.clone(),
                reason: format!(
                    "column {} exceeds grid width {}",
                    block.area.col_end,
                    self.template.column_count()
                ),
            });
        }
        if block.area.row_end > self.ownership.len() {
            return Err(CompositorError::OutOfBounds {
                block: block.name.clone(),
                reason: format!(
                    "row {} exceeds grid height {}",
                    block.area.row_end,
                    self.ownership.len()
                ),
            });
        }

        // Check for ownership conflicts
        for row in block.area.row_start..block.area.row_end {
            for col in block.area.col_start..block.area.col_end {
                if let Some(existing_idx) = self.ownership[row][col] {
                    return Err(CompositorError::CellConflict {
                        cell: (row, col),
                        existing: self.blocks[existing_idx].name.clone(),
                        new: block.name,
                    });
                }
            }
        }

        // Claim cells
        let idx = self.blocks.len();
        for row in block.area.row_start..block.area.row_end {
            for col in block.area.col_start..block.area.col_end {
                self.ownership[row][col] = Some(idx);
            }
        }

        self.blocks.push(block);
        Ok(idx)
    }

    /// Unregister a block by name, freeing its cells.
    pub fn unregister(&mut self, name: &str) -> Result<ComputeBlock, CompositorError> {
        let idx = self
            .blocks
            .iter()
            .position(|b| b.name == name)
            .ok_or_else(|| CompositorError::BlockNotFound(name.to_string()))?;

        let block = self.blocks.remove(idx);

        // Free cells
        for row in block.area.row_start..block.area.row_end {
            for col in block.area.col_start..block.area.col_end {
                self.ownership[row][col] = None;
            }
        }

        // Update indices in ownership map (shift down after removal)
        for row in &mut self.ownership {
            for i in row.iter_mut().flatten() {
                if *i > idx {
                    *i -= 1;
                }
            }
        }

        Ok(block)
    }

    /// Get a block by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ComputeBlock> {
        self.blocks.iter().find(|b| b.name == name)
    }

    /// Get a mutable block by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ComputeBlock> {
        self.blocks.iter_mut().find(|b| b.name == name)
    }

    /// Get computed bounds for a block.
    #[must_use]
    pub fn bounds(&self, name: &str, total_area: Rect) -> Option<Rect> {
        let block = self.blocks.iter().find(|b| b.name == name)?;
        let layout = compute_grid_layout(
            &self.template,
            total_area.width as f32,
            total_area.height as f32,
            &[],
        );
        let (x, y, w, h) = layout.area_bounds(&block.area)?;
        Some(Rect::new(
            total_area.x + x as u16,
            total_area.y + y as u16,
            w as u16,
            h as u16,
        ))
    }

    /// Get all registered blocks.
    #[must_use]
    pub fn blocks(&self) -> &[ComputeBlock] {
        &self.blocks
    }

    /// Mark a region as dirty (needs redraw).
    pub fn mark_dirty(&mut self, rect: Rect) {
        self.dirty.push(rect);
    }

    /// Clear dirty rectangles and return them.
    pub fn take_dirty(&mut self) -> Vec<Rect> {
        std::mem::take(&mut self.dirty)
    }

    /// Check if any regions are dirty.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// Get blocks sorted by z-index for rendering.
    #[must_use]
    pub fn render_order(&self) -> Vec<&ComputeBlock> {
        contract_pre_render!();
        let mut sorted: Vec<_> = self.blocks.iter().filter(|b| b.visible).collect();
        sorted.sort_by_key(|b| b.z_index);
        sorted
    }

    /// Get the block that owns a specific cell.
    #[must_use]
    pub fn owner_at(&self, row: usize, col: usize) -> Option<&ComputeBlock> {
        self.ownership
            .get(row)
            .and_then(|r| r.get(col))
            .and_then(|&idx| idx)
            .map(|idx| &self.blocks[idx])
    }
}

/// Errors from compositor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositorError {
    /// Block area extends beyond grid bounds.
    OutOfBounds { block: String, reason: String },
    /// Two blocks claim the same cell.
    CellConflict {
        cell: (usize, usize),
        existing: String,
        new: String,
    },
    /// Block not found by name.
    BlockNotFound(String),
}

impl fmt::Display for CompositorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { block, reason } => {
                write!(f, "block '{}' out of bounds: {}", block, reason)
            }
            Self::CellConflict {
                cell,
                existing,
                new,
            } => {
                write!(
                    f,
                    "cell ({}, {}) already owned by '{}', cannot assign to '{}'",
                    cell.0, cell.1, existing, new
                )
            }
            Self::BlockNotFound(name) => {
                write!(f, "block '{}' not found", name)
            }
        }
    }
}

impl std::error::Error for CompositorError {}

// ============================================================================
// INTRINSIC LAYOUT COMPUTATION
// ============================================================================

/// Compute width allocation for a single constraint.
/// Returns the allocated width (0 for Fill which is handled in phase 2).
#[inline]
fn compute_constraint_width(
    hint: &SizeHint,
    constraint: FlexConstraint,
    available_width: u16,
) -> u16 {
    match constraint {
        FlexConstraint::Fixed(size) => size,
        FlexConstraint::Min(size) => size.max(hint.min.width),
        FlexConstraint::Max(size) => size.min(hint.preferred.width),
        FlexConstraint::Percentage(pct) => (available_width as u32 * pct as u32 / 100) as u16,
        FlexConstraint::Ratio(num, den) if den > 0 => {
            (available_width as u32 * num as u32 / den as u32) as u16
        }
        FlexConstraint::Ratio(_, _) => 0,
        FlexConstraint::Content => hint.preferred.width,
        FlexConstraint::Fill(_) => 0, // Handle in phase 2
    }
}

/// Distribute remaining space among Fill constraints.
fn distribute_fill_space(
    allocated: &mut [Size],
    hints: &[SizeHint],
    constraints: &[FlexConstraint],
    remaining_width: u16,
    count: usize,
) {
    let fill_total: u16 = constraints
        .iter()
        .take(count)
        .filter_map(|c| {
            if let FlexConstraint::Fill(w) = c {
                Some(*w)
            } else {
                None
            }
        })
        .sum();

    if fill_total == 0 || remaining_width == 0 {
        return;
    }

    for (i, constraint) in constraints.iter().enumerate().take(count) {
        if let FlexConstraint::Fill(weight) = constraint {
            let share = (remaining_width as u32 * *weight as u32 / fill_total as u32) as u16;
            allocated[i].width = hints[i].max.map_or(share, |max| share.min(max.width));
        }
    }
}

/// Compute layout respecting intrinsic sizes.
///
/// This implements a flexbox-like algorithm:
/// 1. Allocate fixed and min sizes
/// 2. Distribute remaining space to Fill constraints
/// 3. Respect max sizes
#[must_use]
pub fn compute_intrinsic_layout(
    hints: &[SizeHint],
    constraints: &[FlexConstraint],
    available: Size,
) -> Vec<Rect> {
    if hints.is_empty() || constraints.is_empty() {
        return Vec::new();
    }

    let count = hints.len().min(constraints.len());
    let mut allocated = vec![Size::ZERO; count];
    let mut remaining_width = available.width;

    // Phase 1: Allocate fixed and min sizes
    for (i, (hint, constraint)) in hints.iter().zip(constraints).enumerate().take(count) {
        let width = compute_constraint_width(hint, *constraint, available.width);
        if !matches!(constraint, FlexConstraint::Fill(_)) {
            allocated[i].width = width;
            if matches!(constraint, FlexConstraint::Content) {
                allocated[i].height = hint.preferred.height;
            }
            remaining_width = remaining_width.saturating_sub(width);
        }
    }

    // Phase 2: Distribute Fill constraints
    distribute_fill_space(&mut allocated, hints, constraints, remaining_width, count);

    // Phase 3: Convert to Rects
    let mut x = 0u16;
    allocated
        .iter()
        .map(|size| {
            let rect = Rect::new(x, 0, size.width, available.height);
            x = x.saturating_add(size.width);
            rect
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::TrackSize;

    // =========================================================================
    // Size Tests
    // =========================================================================

    #[test]
    fn test_size_new() {
        let size = Size::new(80, 24);
        assert_eq!(size.width, 80);
        assert_eq!(size.height, 24);
    }

    #[test]
    fn test_size_zero() {
        assert_eq!(Size::ZERO, Size::new(0, 0));
    }

    // =========================================================================
    // Rect Tests
    // =========================================================================

    #[test]
    fn test_rect_intersection() {
        let r1 = Rect::new(0, 0, 10, 10);
        let r2 = Rect::new(5, 5, 10, 10);
        let intersection = r1.intersection(r2);

        assert_eq!(intersection.x, 5);
        assert_eq!(intersection.y, 5);
        assert_eq!(intersection.width, 5);
        assert_eq!(intersection.height, 5);
    }

    #[test]
    fn test_rect_no_intersection() {
        let r1 = Rect::new(0, 0, 5, 5);
        let r2 = Rect::new(10, 10, 5, 5);
        let intersection = r1.intersection(r2);

        assert_eq!(intersection.area(), 0);
    }

    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(10, 10, 20, 20);

        assert!(rect.contains(10, 10));
        assert!(rect.contains(15, 15));
        assert!(rect.contains(29, 29));
        assert!(!rect.contains(30, 30));
        assert!(!rect.contains(9, 10));
    }

    // =========================================================================
    // SizeHint Tests
    // =========================================================================

    #[test]
    fn test_size_hint_fixed() {
        let hint = SizeHint::fixed(Size::new(40, 10));
        assert_eq!(hint.min, hint.preferred);
        assert_eq!(hint.preferred, hint.max.unwrap());
    }

    #[test]
    fn test_size_hint_flexible() {
        let hint = SizeHint::flexible(Size::new(10, 3));
        assert_eq!(hint.min, Size::new(10, 3));
        assert!(hint.max.is_none());
    }

    // =========================================================================
    // FlexConstraint Tests
    // =========================================================================

    #[test]
    fn test_flex_constraint_default() {
        assert_eq!(FlexConstraint::default(), FlexConstraint::Fill(1));
    }

    // =========================================================================
    // ComputeBlock Tests
    // =========================================================================

    #[test]
    fn test_compute_block_new() {
        let block = ComputeBlock::new("test", GridArea::cell(0, 0));
        assert_eq!(block.name, "test");
        assert_eq!(block.z_index, 0);
        assert!(block.visible);
        assert_eq!(block.clip, ClipMode::Strict);
    }

    #[test]
    fn test_compute_block_builder() {
        let block = ComputeBlock::new("overlay", GridArea::cell(1, 1))
            .with_z_index(10)
            .with_visible(true)
            .with_clip(ClipMode::Overflow);

        assert_eq!(block.z_index, 10);
        assert_eq!(block.clip, ClipMode::Overflow);
    }

    // =========================================================================
    // GridCompositor Tests
    // =========================================================================

    #[test]
    fn test_compositor_register() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0), TrackSize::Fr(1.0)])
            .with_rows([TrackSize::Fr(1.0), TrackSize::Fr(1.0)]);
        let mut compositor = GridCompositor::new(template);

        let idx = compositor
            .register(ComputeBlock::new("header", GridArea::row_span(0, 0, 2)))
            .unwrap();
        assert_eq!(idx, 0);

        let idx = compositor
            .register(ComputeBlock::new("main", GridArea::cell(1, 0)))
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_compositor_cell_conflict() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0), TrackSize::Fr(1.0)]);
        let mut compositor = GridCompositor::new(template);

        compositor
            .register(ComputeBlock::new("first", GridArea::cell(0, 0)))
            .unwrap();

        let result = compositor.register(ComputeBlock::new("second", GridArea::cell(0, 0)));
        assert!(matches!(result, Err(CompositorError::CellConflict { .. })));
    }

    #[test]
    fn test_compositor_out_of_bounds() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0)]);
        let mut compositor = GridCompositor::new(template);

        let result = compositor.register(ComputeBlock::new("bad", GridArea::cell(0, 5)));
        assert!(matches!(result, Err(CompositorError::OutOfBounds { .. })));
    }

    #[test]
    fn test_compositor_bounds() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0), TrackSize::Fr(1.0)])
            .with_rows([TrackSize::Fr(1.0)]);
        let mut compositor = GridCompositor::new(template);

        compositor
            .register(ComputeBlock::new("left", GridArea::cell(0, 0)))
            .unwrap();
        compositor
            .register(ComputeBlock::new("right", GridArea::cell(0, 1)))
            .unwrap();

        let total = Rect::new(0, 0, 100, 50);
        let left_bounds = compositor.bounds("left", total).unwrap();
        let right_bounds = compositor.bounds("right", total).unwrap();

        assert_eq!(left_bounds.x, 0);
        assert_eq!(left_bounds.width, 50);
        assert_eq!(right_bounds.x, 50);
        assert_eq!(right_bounds.width, 50);
    }

    #[test]
    fn test_compositor_render_order() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0), TrackSize::Fr(1.0)]);
        let mut compositor = GridCompositor::new(template);

        compositor
            .register(ComputeBlock::new("back", GridArea::cell(0, 0)).with_z_index(0))
            .unwrap();
        compositor
            .register(ComputeBlock::new("front", GridArea::cell(0, 1)).with_z_index(10))
            .unwrap();

        let order = compositor.render_order();
        assert_eq!(order[0].name, "back");
        assert_eq!(order[1].name, "front");
    }

    #[test]
    fn test_compositor_hidden_blocks() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0)]);
        let mut compositor = GridCompositor::new(template);

        compositor
            .register(ComputeBlock::new("visible", GridArea::cell(0, 0)))
            .unwrap();

        // Need a second row for the hidden block
        let template2 = GridTemplate::columns([TrackSize::Fr(1.0)])
            .with_rows([TrackSize::Fr(1.0), TrackSize::Fr(1.0)]);
        let mut compositor2 = GridCompositor::new(template2);

        compositor2
            .register(ComputeBlock::new("visible", GridArea::cell(0, 0)))
            .unwrap();
        compositor2
            .register(ComputeBlock::new("hidden", GridArea::cell(1, 0)).with_visible(false))
            .unwrap();

        let order = compositor2.render_order();
        assert_eq!(order.len(), 1);
        assert_eq!(order[0].name, "visible");
    }

    #[test]
    fn test_compositor_unregister() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0)]);
        let mut compositor = GridCompositor::new(template);

        compositor
            .register(ComputeBlock::new("block", GridArea::cell(0, 0)))
            .unwrap();

        let block = compositor.unregister("block").unwrap();
        assert_eq!(block.name, "block");

        // Can register same area again
        compositor
            .register(ComputeBlock::new("new", GridArea::cell(0, 0)))
            .unwrap();
    }

    #[test]
    fn test_compositor_dirty_tracking() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0)]);
        let mut compositor = GridCompositor::new(template);

        assert!(!compositor.is_dirty());

        compositor.mark_dirty(Rect::new(0, 0, 10, 10));
        assert!(compositor.is_dirty());

        let dirty = compositor.take_dirty();
        assert_eq!(dirty.len(), 1);
        assert!(!compositor.is_dirty());
    }

    #[test]
    fn test_compositor_owner_at() {
        let template = GridTemplate::columns([TrackSize::Fr(1.0), TrackSize::Fr(1.0)]);
        let mut compositor = GridCompositor::new(template);

        compositor
            .register(ComputeBlock::new("left", GridArea::cell(0, 0)))
            .unwrap();

        assert_eq!(compositor.owner_at(0, 0).unwrap().name, "left");
        assert!(compositor.owner_at(0, 1).is_none());
    }

    // =========================================================================
    // Intrinsic Layout Tests
    // =========================================================================

    #[test]
    fn test_gc001_fill_distributes_space() {
        let hints = vec![
            SizeHint::flexible(Size::new(10, 5)),
            SizeHint::flexible(Size::new(10, 5)),
            SizeHint::flexible(Size::new(10, 5)),
        ];
        let constraints = vec![
            FlexConstraint::Fill(1),
            FlexConstraint::Fill(1),
            FlexConstraint::Fill(1),
        ];

        let rects = compute_intrinsic_layout(&hints, &constraints, Size::new(120, 24));

        assert_eq!(rects.len(), 3);
        assert_eq!(rects[0].width, 40);
        assert_eq!(rects[1].width, 40);
        assert_eq!(rects[2].width, 40);
    }

    #[test]
    fn test_gc002_content_uses_size_hint() {
        let hints = vec![SizeHint::new(
            Size::new(10, 3),
            Size::new(40, 8),
            Some(Size::new(80, 16)),
        )];
        let constraints = vec![FlexConstraint::Content];

        let rects = compute_intrinsic_layout(&hints, &constraints, Size::new(200, 50));

        assert_eq!(rects[0].width, 40); // Uses preferred
    }

    #[test]
    fn test_fill_with_weights() {
        let hints = vec![
            SizeHint::flexible(Size::new(0, 5)),
            SizeHint::flexible(Size::new(0, 5)),
        ];
        let constraints = vec![FlexConstraint::Fill(2), FlexConstraint::Fill(1)];

        let rects = compute_intrinsic_layout(&hints, &constraints, Size::new(90, 24));

        assert_eq!(rects[0].width, 60); // 2/3
        assert_eq!(rects[1].width, 30); // 1/3
    }

    #[test]
    fn test_mixed_constraints() {
        let hints = vec![
            SizeHint::fixed(Size::new(20, 5)),
            SizeHint::flexible(Size::new(10, 5)),
            SizeHint::fixed(Size::new(20, 5)),
        ];
        let constraints = vec![
            FlexConstraint::Fixed(20),
            FlexConstraint::Fill(1),
            FlexConstraint::Fixed(20),
        ];

        let rects = compute_intrinsic_layout(&hints, &constraints, Size::new(100, 24));

        assert_eq!(rects[0].width, 20);
        assert_eq!(rects[1].width, 60); // Fills remaining
        assert_eq!(rects[2].width, 20);
    }

    #[test]
    fn test_fill_respects_max() {
        let hints = vec![SizeHint::new(
            Size::new(10, 5),
            Size::new(30, 5),
            Some(Size::new(50, 5)),
        )];
        let constraints = vec![FlexConstraint::Fill(1)];

        let rects = compute_intrinsic_layout(&hints, &constraints, Size::new(200, 24));

        assert_eq!(rects[0].width, 50); // Capped at max
    }

    // =========================================================================
    // Error Display Tests
    // =========================================================================

    #[test]
    fn test_compositor_error_display() {
        let err = CompositorError::CellConflict {
            cell: (1, 2),
            existing: "first".to_string(),
            new: "second".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("first"));
        assert!(msg.contains("second"));
    }
}
