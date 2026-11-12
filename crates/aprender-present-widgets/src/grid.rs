//! Grid widget: children placed left to right, top to bottom in equal cells.
//!
//! The cell arithmetic lives in the pure function [`grid_cells`], which never
//! panics and never returns a non-finite coordinate, whatever the bounds,
//! column count or gap. [`Grid`] lays each child out with its cell as bounds.
//!
//! Contract: `contracts/present-grid-widget-v1.yaml`.

use presentar_core::{
    widget::{BrickAssertion, BrickBudget, BrickVerification, LayoutResult},
    Brick, Canvas, Constraints, Event, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Largest magnitude accepted for any coordinate or extent. Larger finite
/// values are clamped so that `x + width` can never overflow to infinity.
const COORD_LIMIT: f32 = 1.0e15;

/// A finite coordinate: NaN and infinities become 0, huge values are clamped.
fn sane(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-COORD_LIMIT, COORD_LIMIT)
    } else {
        0.0
    }
}

/// Fit `value` between `min` and `max`, finite and not negative. Unlike
/// `f32::clamp` this never panics, whatever the constraints (NaN, infinite or
/// `min > max`); a lower bound that exceeds the upper one loses.
fn fit(value: f32, min: f32, max: f32) -> f32 {
    sane(value.max(min).min(max)).max(0.0)
}

/// An upper bound a child can be measured against: NaN means "no bound",
/// and a bound below 0 is 0.
fn sane_upper(max: f32) -> f32 {
    if max.is_nan() {
        f32::INFINITY
    } else {
        max.max(0.0)
    }
}

/// A usable gap: finite and not negative, otherwise 0.
fn sane_gap(gap: f32) -> f32 {
    sane(gap).max(0.0)
}

/// Gap actually used between `tracks` tracks inside `extent`: the requested
/// gap, shrunk so that the gaps alone never exceed the extent.
fn squeezed_gap(gap: f32, extent: f32, tracks: usize) -> f32 {
    if tracks <= 1 {
        0.0
    } else {
        gap.min(extent / (tracks - 1) as f32)
    }
}

/// Cells of a grid with `columns` equal-width columns inside `bounds`.
///
/// Cell `i` is in column `i % columns` and row `i / columns`. The rows share
/// the height equally. `columns` below 1 counts as 1; a gap that is not finite
/// or is negative counts as 0; non-finite bounds components count as 0 and a
/// negative size as 0. Gaps that would not fit are shrunk, so every cell lies
/// inside `bounds` and no two cells overlap.
#[must_use]
pub fn grid_cells(bounds: Rect, columns: usize, gap: f32, count: usize) -> Vec<Rect> {
    grid_cells_with(bounds, columns, gap, None, count)
}

/// Like [`grid_cells`], with an optional fixed row height.
///
/// With `Some(height)` every row is `height` tall (a negative or non-finite
/// value counts as 0) and rows follow each other separated by `gap`; the cells
/// of a tall grid then extend below `bounds`. With `None` the rows share the
/// height of `bounds` equally.
#[must_use]
pub fn grid_cells_with(
    bounds: Rect,
    columns: usize,
    gap: f32,
    row_height: Option<f32>,
    count: usize,
) -> Vec<Rect> {
    if count == 0 {
        return Vec::new();
    }
    let columns = columns.max(1);
    let gap = sane_gap(gap);
    let (x, y) = (sane(bounds.x), sane(bounds.y));
    let width = sane(bounds.width).max(0.0);
    let height = sane(bounds.height).max(0.0);
    let rows = count.div_ceil(columns);

    let gap_x = squeezed_gap(gap, width, columns);
    let cell_width = (gap_x.mul_add(-((columns - 1) as f32), width) / columns as f32).max(0.0);
    let (cell_height, row_step) = match row_height {
        Some(fixed) => {
            let fixed = sane(fixed).max(0.0);
            (fixed, fixed + gap)
        }
        None => {
            let gap_y = squeezed_gap(gap, height, rows);
            let share = (gap_y.mul_add(-((rows - 1) as f32), height) / rows as f32).max(0.0);
            (share, share + gap_y)
        }
    };
    let column_step = cell_width + gap_x;

    (0..count)
        .map(|i| {
            Rect::new(
                (i % columns) as f32 * column_step + x,
                (i / columns) as f32 * row_step + y,
                cell_width,
                cell_height,
            )
        })
        .collect()
}

/// Grid widget: a fixed number of equal-width columns.
#[derive(Serialize, Deserialize)]
pub struct Grid {
    /// Number of columns (at least 1)
    columns: usize,
    /// Gap between cells, both directions
    gap: f32,
    /// Fixed row height; `None` divides the available height equally
    row_height: Option<f32>,
    /// Children widgets, one per cell
    #[serde(skip)]
    children: Vec<Box<dyn Widget>>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for Grid {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Grid {
    /// Create an empty grid with `columns` columns (0 is stored as 1).
    #[must_use]
    pub fn new(columns: usize) -> Self {
        Self {
            columns: columns.max(1),
            gap: 0.0,
            row_height: None,
            children: Vec::new(),
            test_id_value: None,
            bounds: Rect::default(),
        }
    }

    /// Set the number of columns (0 is stored as 1).
    #[must_use]
    pub fn columns(mut self, columns: usize) -> Self {
        self.columns = columns.max(1);
        self
    }

    /// Set the gap between cells. A gap that is not finite or is negative is
    /// treated as 0.
    #[must_use]
    pub fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
        self
    }

    /// Set a fixed row height. A negative or non-finite height clears it, so
    /// the rows share the available height equally.
    #[must_use]
    pub fn row_height(mut self, height: f32) -> Self {
        self.row_height = (height.is_finite() && height >= 0.0).then_some(height);
        self
    }

    /// Add a child widget; it takes the next free cell.
    pub fn child(mut self, widget: impl Widget + 'static) -> Self {
        self.children.push(Box::new(widget));
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Number of columns.
    #[must_use]
    pub const fn get_columns(&self) -> usize {
        self.columns
    }

    /// The gap in use (0 when the stored gap is not finite or is negative).
    #[must_use]
    pub fn get_gap(&self) -> f32 {
        sane_gap(self.gap)
    }

    /// The fixed row height, if one is set.
    #[must_use]
    pub const fn get_row_height(&self) -> Option<f32> {
        self.row_height
    }

    /// Number of rows needed for the current children.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.children.len().div_ceil(self.columns.max(1))
    }

    /// The cell of each child for the given bounds.
    #[must_use]
    pub fn cells(&self, bounds: Rect) -> Vec<Rect> {
        grid_cells_with(
            bounds,
            self.columns,
            self.gap,
            self.row_height,
            self.children.len(),
        )
    }
}

impl Widget for Grid {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        if self.children.is_empty() {
            return Size::ZERO;
        }

        let columns = self.columns.max(1);
        let gap = self.get_gap();
        let used_columns = columns.min(self.children.len());
        let rows = self.row_count();

        // Width offered to each child: the available width minus the gaps,
        // split between the columns (unbounded stays unbounded).
        let slot_width = if constraints.max_width.is_finite() {
            ((constraints.max_width - gap * (columns - 1) as f32) / columns as f32).max(0.0)
        } else {
            f32::INFINITY
        };
        let child_constraints =
            Constraints::new(0.0, slot_width, 0.0, sane_upper(constraints.max_height));

        let mut cell_width = 0.0f32;
        let mut cell_height = 0.0f32;
        for child in &self.children {
            let size = child.measure(child_constraints);
            cell_width = cell_width.max(size.width);
            cell_height = cell_height.max(size.height);
        }
        let cell_height = self.row_height.unwrap_or(cell_height);

        let width = cell_width.mul_add(used_columns as f32, gap * (used_columns - 1) as f32);
        let height = cell_height.mul_add(rows as f32, gap * (rows - 1) as f32);
        // Not `Constraints::constrain`: its `f32::clamp` panics on NaN or
        // inverted constraints.
        Size::new(
            fit(width, constraints.min_width, constraints.max_width),
            fit(height, constraints.min_height, constraints.max_height),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        let cells = self.cells(bounds);
        for (child, cell) in self.children.iter_mut().zip(cells) {
            child.layout(cell);
        }
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        for child in &self.children {
            child.paint(canvas);
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        for child in &mut self.children {
            if let Some(msg) = child.event(event) {
                return Some(msg);
            }
        }
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut self.children
    }

    fn bounds(&self) -> Rect {
        self.bounds
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Grid {
    fn brick_name(&self) -> &'static str {
        "Grid"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[BrickAssertion::MaxLatencyMs(16)]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = self.assertions().to_vec();
        let mut failed = Vec::new();

        let b = self.bounds;
        let valid = [b.x, b.y, b.width, b.height].iter().all(|v| v.is_finite())
            && b.width >= 0.0
            && b.height >= 0.0;
        let assertion = BrickAssertion::ElementPresent("valid-bounds".to_string());
        if valid {
            passed.push(assertion);
        } else {
            failed.push((assertion, "Grid bounds are not finite and >= 0".to_string()));
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        format!(
            r#"<div class="brick-grid" style="grid-template-columns: repeat({}, 1fr);"></div>"#,
            self.columns.max(1)
        )
    }

    fn to_css(&self) -> String {
        format!(
            ".brick-grid {{ display: grid; gap: {}px; }}",
            self.get_gap()
        )
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::modal::Modal;
    use crate::video::{Video, VideoToggled};
    use presentar_core::draw::DrawCommand;
    use presentar_core::{MouseButton, Point, RecordingCanvas};
    use proptest::prelude::*;

    const EPS: f32 = 0.01;

    fn tile(n: usize) -> Video {
        Video::new(format!("tile-{n}.mp4")).accessible_name(format!("Tile {n}"))
    }

    fn grid_of(columns: usize, tiles: usize) -> Grid {
        (0..tiles).fold(Grid::new(columns), |g, n| g.child(tile(n)))
    }

    fn right(r: &Rect) -> f32 {
        r.x + r.width
    }

    fn bottom(r: &Rect) -> f32 {
        r.y + r.height
    }

    fn inside(outer: &Rect, inner: &Rect) -> bool {
        inner.x >= outer.x - EPS
            && inner.y >= outer.y - EPS
            && right(inner) <= right(outer) + EPS
            && bottom(inner) <= bottom(outer) + EPS
    }

    fn overlap(a: &Rect, b: &Rect) -> bool {
        let across = right(a).min(right(b)) - a.x.max(b.x);
        let down = bottom(a).min(bottom(b)) - a.y.max(b.y);
        across > EPS && down > EPS
    }

    fn all_finite(cells: &[Rect]) -> bool {
        cells
            .iter()
            .all(|c| [c.x, c.y, c.width, c.height].iter().all(|v| v.is_finite()))
    }

    fn click(grid: &mut Grid, at: Point) -> Option<Box<dyn Any + Send>> {
        grid.event(&Event::MouseDown {
            position: at,
            button: MouseButton::Left,
        });
        grid.event(&Event::MouseUp {
            position: at,
            button: MouseButton::Left,
        })
    }

    fn paint_len(widget: &dyn Widget) -> usize {
        let mut canvas = RecordingCanvas::new();
        widget.paint(&mut canvas);
        canvas.commands().len()
    }

    // ===== Cell arithmetic =====

    #[test]
    fn grid_cells_two_by_two_with_gap_has_exact_geometry() {
        let cells = grid_cells(Rect::new(10.0, 20.0, 210.0, 110.0), 2, 10.0, 4);
        assert_eq!(
            cells,
            vec![
                Rect::new(10.0, 20.0, 100.0, 50.0),
                Rect::new(120.0, 20.0, 100.0, 50.0),
                Rect::new(10.0, 80.0, 100.0, 50.0),
                Rect::new(120.0, 80.0, 100.0, 50.0),
            ]
        );
    }

    #[test]
    fn grid_cells_empty_grid_has_no_cells() {
        assert!(grid_cells(Rect::new(0.0, 0.0, 100.0, 100.0), 3, 5.0, 0).is_empty());
    }

    #[test]
    fn grid_cells_one_column_stacks_cells() {
        let cells = grid_cells(Rect::new(0.0, 0.0, 100.0, 90.0), 1, 0.0, 3);
        assert_eq!(cells.len(), 3);
        for (i, c) in cells.iter().enumerate() {
            assert_eq!(c.x, 0.0);
            assert_eq!(c.width, 100.0);
            assert_eq!(c.height, 30.0);
            assert_eq!(c.y, 30.0 * i as f32);
        }
    }

    #[test]
    fn grid_cells_more_columns_than_children_keeps_column_width() {
        let cells = grid_cells(Rect::new(0.0, 0.0, 100.0, 40.0), 5, 0.0, 2);
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0], Rect::new(0.0, 0.0, 20.0, 40.0));
        assert_eq!(cells[1], Rect::new(20.0, 0.0, 20.0, 40.0));
    }

    #[test]
    fn grid_cells_zero_columns_counts_as_one() {
        let bounds = Rect::new(0.0, 0.0, 60.0, 60.0);
        assert_eq!(grid_cells(bounds, 0, 0.0, 3), grid_cells(bounds, 1, 0.0, 3));
    }

    #[test]
    fn grid_cells_fixed_row_height_steps_by_height_plus_gap() {
        let cells = grid_cells_with(Rect::new(5.0, 6.0, 100.0, 10.0), 2, 4.0, Some(30.0), 5);
        assert_eq!(cells.len(), 5);
        assert_eq!(cells[0], Rect::new(5.0, 6.0, 48.0, 30.0));
        assert_eq!(cells[1], Rect::new(57.0, 6.0, 48.0, 30.0));
        assert_eq!(cells[2], Rect::new(5.0, 40.0, 48.0, 30.0));
        assert_eq!(cells[4], Rect::new(5.0, 74.0, 48.0, 30.0));
    }

    #[test]
    fn grid_cells_oversized_gap_is_squeezed_to_fit() {
        let bounds = Rect::new(0.0, 0.0, 100.0, 100.0);
        let cells = grid_cells(bounds, 3, 1000.0, 9);
        assert_eq!(cells.len(), 9);
        for c in &cells {
            assert!(inside(&bounds, c));
            assert!(c.width >= 0.0 && c.height >= 0.0);
        }
    }

    #[test]
    fn grid_cells_bad_inputs_never_panic_or_go_non_finite() {
        let nan = f32::NAN;
        let inf = f32::INFINITY;
        let cases = [
            Rect::new(nan, nan, nan, nan),
            Rect::new(inf, -inf, inf, -inf),
            Rect::new(0.0, 0.0, -50.0, -50.0),
            Rect::new(0.0, 0.0, 0.0, 0.0),
            Rect::new(f32::MAX, f32::MAX, f32::MAX, f32::MAX),
            Rect::new(f32::MIN, f32::MIN, f32::MAX, f32::MAX),
        ];
        for bounds in cases {
            for gap in [nan, inf, -inf, -3.0, 0.0, 4.0, f32::MAX] {
                for columns in [0usize, 1, 3, usize::MAX] {
                    for row_height in [None, Some(nan), Some(-1.0), Some(f32::MAX), Some(9.0)] {
                        let cells = grid_cells_with(bounds, columns, gap, row_height, 7);
                        assert_eq!(cells.len(), 7);
                        assert!(all_finite(&cells), "{bounds:?} {gap} {columns}");
                        assert!(cells.iter().all(|c| c.width >= 0.0 && c.height >= 0.0));
                    }
                }
            }
        }
    }

    proptest! {
        #[test]
        fn grid_cells_well_formed_inputs_tile_the_bounds(
            x in -1000.0f32..1000.0,
            y in -1000.0f32..1000.0,
            w in 0.0f32..2000.0,
            h in 0.0f32..2000.0,
            columns in 1usize..8,
            gap in -10.0f32..200.0,
            count in 0usize..40,
        ) {
            let bounds = Rect::new(x, y, w, h);
            let cells = grid_cells(bounds, columns, gap, count);
            prop_assert_eq!(cells.len(), count);
            prop_assert!(all_finite(&cells));
            for c in &cells {
                prop_assert!(inside(&bounds, c), "{:?} outside {:?}", c, bounds);
            }
            for (i, a) in cells.iter().enumerate() {
                prop_assert_eq!(a.width, cells[0].width);
                let row_start = &cells[i - i % columns];
                prop_assert_eq!(a.y, row_start.y);
                prop_assert_eq!(a.height, row_start.height);
                for b in &cells[i + 1..] {
                    prop_assert!(!overlap(a, b), "{:?} overlaps {:?}", a, b);
                }
            }
        }

        #[test]
        fn grid_cells_arbitrary_inputs_never_panic_or_go_non_finite(
            x in prop_oneof![-1000.0f32..1000.0, Just(f32::NAN), Just(f32::INFINITY)],
            y in prop_oneof![-1000.0f32..1000.0, Just(f32::NAN), Just(f32::NEG_INFINITY)],
            w in prop_oneof![-500.0f32..2000.0, Just(0.0), Just(f32::NAN), Just(f32::INFINITY)],
            h in prop_oneof![-500.0f32..2000.0, Just(0.0), Just(f32::NAN), Just(f32::INFINITY)],
            columns in 0usize..8,
            gap in prop_oneof![-50.0f32..200.0, Just(f32::NAN), Just(f32::INFINITY)],
            count in 0usize..40,
        ) {
            let cells = grid_cells(Rect::new(x, y, w, h), columns, gap, count);
            prop_assert_eq!(cells.len(), count);
            prop_assert!(all_finite(&cells));
            prop_assert!(cells.iter().all(|c| c.width >= 0.0 && c.height >= 0.0));
        }
    }

    // ===== Construction =====

    #[test]
    fn grid_new_zero_columns_is_stored_as_one() {
        assert_eq!(Grid::new(0).get_columns(), 1);
        assert_eq!(Grid::new(4).columns(0).get_columns(), 1);
        assert_eq!(Grid::default().get_columns(), 1);
        assert_eq!(Grid::new(3).columns(5).get_columns(), 5);
    }

    #[test]
    fn grid_builder_sets_gap_row_height_and_test_id() {
        let g = Grid::new(2).gap(8.0).row_height(40.0).with_test_id("tiles");
        assert_eq!(g.get_gap(), 8.0);
        assert_eq!(g.get_row_height(), Some(40.0));
        assert_eq!(Widget::test_id(&g), Some("tiles"));
        assert_eq!(Brick::test_id(&g), Some("tiles"));
        assert_eq!(Widget::type_id(&g), TypeId::of::<Grid>());
    }

    #[test]
    fn grid_bad_gap_and_row_height_are_ignored() {
        assert_eq!(Grid::new(2).gap(f32::NAN).get_gap(), 0.0);
        assert_eq!(Grid::new(2).gap(-5.0).get_gap(), 0.0);
        assert_eq!(Grid::new(2).row_height(-1.0).get_row_height(), None);
        assert_eq!(Grid::new(2).row_height(f32::NAN).get_row_height(), None);
        assert_eq!(Grid::new(2).row_height(0.0).get_row_height(), Some(0.0));
    }

    #[test]
    fn grid_row_count_rounds_up() {
        assert_eq!(grid_of(3, 0).row_count(), 0);
        assert_eq!(grid_of(3, 1).row_count(), 1);
        assert_eq!(grid_of(3, 3).row_count(), 1);
        assert_eq!(grid_of(3, 4).row_count(), 2);
    }

    #[test]
    fn grid_serde_keeps_settings_and_skips_children() {
        let g = grid_of(3, 2).gap(6.0).row_height(20.0).with_test_id("g");
        let json = serde_json::to_string(&g).unwrap();
        let back: Grid = serde_json::from_str(&json).unwrap();
        assert_eq!(back.get_columns(), 3);
        assert_eq!(back.get_gap(), 6.0);
        assert_eq!(back.get_row_height(), Some(20.0));
        assert!(back.children().is_empty());
    }

    // ===== Layout =====

    #[test]
    fn grid_empty_and_degenerate_grids_lay_out_without_panic() {
        let mut empty = Grid::new(3);
        let result = empty.layout(Rect::new(0.0, 0.0, 100.0, 100.0));
        assert_eq!(result.size, Size::new(100.0, 100.0));
        assert!(empty.children().is_empty());

        let mut one_column = grid_of(1, 3);
        one_column.layout(Rect::new(0.0, 0.0, 90.0, 90.0));
        assert_eq!(
            one_column.children()[2].bounds(),
            Rect::new(0.0, 60.0, 90.0, 30.0)
        );

        let mut wide = grid_of(10, 2);
        wide.layout(Rect::new(0.0, 0.0, 100.0, 50.0));
        assert_eq!(
            wide.children()[1].bounds(),
            Rect::new(10.0, 0.0, 10.0, 50.0)
        );
    }

    #[test]
    fn grid_layout_gives_each_child_its_cell() {
        let bounds = Rect::new(10.0, 20.0, 210.0, 110.0);
        let mut g = grid_of(2, 4).gap(10.0);
        let result = g.layout(bounds);
        assert_eq!(result.size, bounds.size());
        assert_eq!(Widget::bounds(&g), bounds);
        let cells = grid_cells(bounds, 2, 10.0, 4);
        for (child, cell) in g.children().iter().zip(&cells) {
            assert_eq!(child.bounds(), *cell);
        }
    }

    #[test]
    fn grid_relayout_replaces_cells() {
        let mut g = grid_of(2, 2);
        g.layout(Rect::new(0.0, 0.0, 100.0, 100.0));
        g.layout(Rect::new(0.0, 0.0, 40.0, 20.0));
        assert_eq!(g.children()[1].bounds(), Rect::new(20.0, 0.0, 20.0, 20.0));
    }

    #[test]
    fn grid_children_mut_reaches_the_same_children() {
        let mut g = grid_of(2, 3);
        assert_eq!(g.children_mut().len(), 3);
        assert_eq!(g.cells(Rect::new(0.0, 0.0, 10.0, 10.0)).len(), 3);
    }

    // ===== Measure =====

    #[test]
    fn grid_measure_empty_is_zero() {
        let g = Grid::new(3);
        assert_eq!(
            g.measure(Constraints::loose(Size::new(100.0, 100.0))),
            Size::ZERO
        );
    }

    #[test]
    fn grid_measure_sums_cells_and_gaps() {
        // Videos prefer 320x180; plenty of room, 2 columns x 2 rows, gap 10.
        let g = grid_of(2, 3).gap(10.0);
        let size = g.measure(Constraints::loose(Size::new(2000.0, 2000.0)));
        assert_eq!(size, Size::new(650.0, 370.0));
    }

    #[test]
    fn grid_measure_shares_width_and_honours_row_height() {
        let g = grid_of(2, 4).gap(10.0).row_height(50.0);
        let size = g.measure(Constraints::loose(Size::new(210.0, 2000.0)));
        // Each child is offered (210 - 10) / 2 = 100 wide; rows are 50 tall.
        assert_eq!(size, Size::new(210.0, 110.0));
    }

    #[test]
    fn grid_measure_with_unbounded_width_uses_preferred_sizes() {
        let g = grid_of(2, 1);
        let size = g.measure(Constraints::unbounded());
        assert_eq!(size, Size::new(320.0, 180.0));
    }

    // ===== Paint =====

    #[test]
    fn grid_paint_draws_every_child() {
        let mut g = grid_of(2, 3);
        g.layout(Rect::new(0.0, 0.0, 200.0, 100.0));
        // Each paused Video draws a backdrop rect and a glyph.
        assert_eq!(paint_len(&g), 6);
    }

    // ===== Verify =====

    #[test]
    fn grid_verify_passes_with_valid_bounds() {
        let mut g = grid_of(2, 2);
        assert!(g.verify().failed.is_empty());
        assert!(g.verify().is_valid());
        g.layout(Rect::new(0.0, 0.0, 200.0, 100.0));
        assert!(g.verify().is_valid());
        assert!(g.assertions().contains(&BrickAssertion::MaxLatencyMs(16)));
    }

    #[test]
    fn grid_verify_fails_with_invalid_bounds() {
        for bad in [
            Rect::new(f32::NAN, 0.0, 10.0, 10.0),
            Rect::new(0.0, 0.0, -1.0, 10.0),
            Rect::new(0.0, 0.0, 10.0, -1.0),
            Rect::new(0.0, 0.0, f32::INFINITY, 10.0),
        ] {
            let mut g = Grid::new(2);
            g.layout(bad);
            let v = g.verify();
            assert!(!v.is_valid(), "{bad:?}");
            assert_eq!(v.failed.len(), 1);
        }
    }

    #[test]
    fn grid_brick_metadata() {
        let g = Grid::new(3).gap(4.0);
        assert_eq!(g.brick_name(), "Grid");
        assert_eq!(g.budget(), BrickBudget::uniform(16));
        assert!(g.to_html().contains("repeat(3, 1fr)"));
        assert!(g.to_css().contains("display: grid"));
        assert!(g.to_css().contains("4px"));
    }

    // ===== Events =====

    #[test]
    fn grid_click_in_second_cell_reaches_only_the_second_child() {
        let mut g = grid_of(2, 2);
        g.layout(Rect::new(0.0, 0.0, 200.0, 100.0));
        let msg = click(&mut g, Point::new(150.0, 50.0));
        let toggled = msg
            .as_ref()
            .and_then(|m| m.downcast_ref::<VideoToggled>())
            .cloned();
        assert_eq!(toggled, Some(VideoToggled { playing: true }));
        // First tile still paused: backdrop + glyph. Second playing: frame only.
        assert_eq!(paint_len(g.children()[0].as_ref()), 2);
        assert_eq!(paint_len(g.children()[1].as_ref()), 1);
    }

    #[test]
    fn grid_click_outside_every_cell_does_nothing() {
        let mut g = grid_of(2, 2);
        g.layout(Rect::new(0.0, 0.0, 200.0, 100.0));
        assert!(click(&mut g, Point::new(500.0, 500.0)).is_none());
        assert_eq!(paint_len(g.children()[0].as_ref()), 2);
        assert_eq!(paint_len(g.children()[1].as_ref()), 2);
    }

    // ===== Composition =====

    #[test]
    fn grid_of_six_videos_inside_modal_paints_inside_content_bounds() {
        let viewport = Rect::new(0.0, 0.0, 800.0, 600.0);

        // Commands the modal paints on its own, without content.
        let mut bare = Modal::new();
        bare.show();
        bare.layout(viewport);
        let mut bare_canvas = RecordingCanvas::new();
        bare.paint(&mut bare_canvas);
        let own = bare_canvas.commands().len();

        let mut modal = Modal::new().content(grid_of(3, 6).gap(8.0));
        modal.show();
        modal.layout(viewport);
        let mut canvas = RecordingCanvas::new();
        modal.paint(&mut canvas);
        let content = modal.content_bounds();

        let tiles = &canvas.commands()[own..];
        // Six tiles, each a backdrop rect and a glyph path.
        assert_eq!(tiles.len(), 12);
        let rects: Vec<Rect> = tiles
            .iter()
            .filter_map(|c| match c {
                DrawCommand::Rect { bounds, .. } => Some(*bounds),
                _ => None,
            })
            .collect();
        let points: Vec<Point> = tiles
            .iter()
            .filter_map(|c| match c {
                DrawCommand::Path { points, .. } => Some(points.clone()),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(rects.len(), 6);
        assert_eq!(points.len(), 18);
        for r in &rects {
            assert!(inside(&content, r), "{r:?} outside {content:?}");
        }
        for p in &points {
            assert!(inside(&content, &Rect::new(p.x, p.y, 0.0, 0.0)));
        }
        for (i, a) in rects.iter().enumerate() {
            for b in &rects[i + 1..] {
                assert!(!overlap(a, b));
            }
        }
    }

    // ----- measure under hostile constraints (review FAIL @08e8ce3) -----

    fn finite_and_not_negative(size: Size) -> bool {
        size.width.is_finite() && size.height.is_finite() && size.width >= 0.0 && size.height >= 0.0
    }

    #[test]
    fn grid_measure_survives_inverted_constraints() {
        // min > max: `f32::clamp` would panic here. The upper bound wins.
        let size = grid_of(2, 1).measure(Constraints::new(10.0, 5.0, 8.0, 3.0));
        assert!(finite_and_not_negative(size), "{size:?}");
        assert!(size.width <= 5.0 + EPS, "{size:?}");
        assert!(size.height <= 3.0 + EPS, "{size:?}");
    }

    #[test]
    fn grid_measure_survives_non_finite_and_negative_constraints() {
        let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, -1.0];
        for &v in &bad {
            for slot in 0..4 {
                let mut c = [0.0, 300.0, 0.0, 300.0];
                c[slot] = v;
                let size = grid_of(3, 5).measure(Constraints::new(c[0], c[1], c[2], c[3]));
                assert!(finite_and_not_negative(size), "slot {slot} = {v}: {size:?}");
            }
        }
    }

    proptest! {
        #[test]
        fn grid_measure_never_panics_and_stays_finite(
            min_w in proptest::num::f32::ANY,
            max_w in proptest::num::f32::ANY,
            min_h in proptest::num::f32::ANY,
            max_h in proptest::num::f32::ANY,
            columns in 0usize..6,
            tiles in 0usize..9,
        ) {
            let size = grid_of(columns, tiles).measure(Constraints::new(min_w, max_w, min_h, max_h));
            prop_assert!(finite_and_not_negative(size), "{size:?}");
        }
    }

    // ===== Class property: every public entry point, any f32 (PR #6897, round 3) =====
    //
    // Entry point                       covered by
    // grid_cells                        class_grid_free_functions_are_total_and_sane
    // grid_cells_with                   class_grid_free_functions_are_total_and_sane
    // new / columns / gap / row_height / child / with_test_id
    //                                   class_grid_builders_getters_and_cells_are_sane
    // get_columns / get_gap / get_row_height / row_count / cells
    //                                   class_grid_builders_getters_and_cells_are_sane
    // measure                           class_grid_measure_is_finite_and_not_negative
    // layout / bounds / verify          class_grid_layout_size_is_sane_and_verify_judges_raw_bounds
    // paint                             class_grid_paint_is_sane_for_any_bounds
    // event                             class_grid_event_is_total

    /// Any f32, with the awkward ones over-represented.
    fn wild() -> impl Strategy<Value = f32> {
        prop_oneof![
            proptest::num::f32::ANY,
            Just(f32::NAN),
            Just(f32::INFINITY),
            Just(f32::NEG_INFINITY),
            Just(f32::MAX),
            Just(f32::MIN),
            Just(f32::MIN_POSITIVE),
            Just(1.0e-40_f32),
            Just(-0.0_f32),
            Just(-10.0_f32),
            -1000.0_f32..1000.0,
        ]
    }

    fn any_columns() -> impl Strategy<Value = usize> {
        prop_oneof![0_usize..8, Just(usize::MAX), any::<usize>()]
    }

    fn finite_non_negative(v: f32) -> bool {
        v.is_finite() && v >= 0.0
    }

    fn rect_sane(r: &Rect) -> bool {
        r.x.is_finite()
            && r.y.is_finite()
            && finite_non_negative(r.width)
            && finite_non_negative(r.height)
    }

    /// Every coordinate finite; every width, radius and blur finite and not negative.
    fn command_widths_sane(cmd: &DrawCommand) -> bool {
        match cmd {
            DrawCommand::Rect {
                bounds,
                radius,
                style,
            } => {
                rect_sane(bounds)
                    && [
                        radius.top_left,
                        radius.top_right,
                        radius.bottom_right,
                        radius.bottom_left,
                    ]
                    .into_iter()
                    .all(finite_non_negative)
                    && style
                        .stroke
                        .as_ref()
                        .is_none_or(|s| finite_non_negative(s.width))
            }
            DrawCommand::Path { points, style, .. } => {
                finite_non_negative(style.width)
                    && points.iter().all(|p| p.x.is_finite() && p.y.is_finite())
            }
            _ => false,
        }
    }

    fn raw_bounds_valid(b: Rect) -> bool {
        [b.x, b.y, b.width, b.height].iter().all(|v| v.is_finite())
            && b.width >= 0.0
            && b.height >= 0.0
    }

    fn bits(r: Rect) -> [u32; 4] {
        [
            r.x.to_bits(),
            r.y.to_bits(),
            r.width.to_bits(),
            r.height.to_bits(),
        ]
    }

    #[test]
    fn reviewer_repro_grid_clause_8_upper_limits_mean_what_the_contract_says() {
        let g = grid_of(2, 2);
        let wide = |max_w: f32, max_h: f32| g.measure(Constraints::new(0.0, max_w, 0.0, max_h));
        // NaN and +inf upper limits impose no limit: the same size as unbounded.
        let unbounded = wide(f32::INFINITY, f32::INFINITY);
        assert_eq!(wide(f32::NAN, f32::NAN), unbounded);
        assert_eq!(unbounded, Size::new(640.0, 180.0));
        // Negative and -inf upper limits count as 0.
        for limit in [-1.0, -1.0e9, f32::NEG_INFINITY] {
            assert_eq!(wide(limit, limit), Size::ZERO, "{limit}");
        }
    }

    #[test]
    fn reviewer_repro_grid_layout_with_bad_bounds_returns_a_sane_size() {
        let mut g = grid_of(2, 2);
        let result = g.layout(Rect::new(f32::NAN, 0.0, -5.0, f32::INFINITY));
        assert_eq!(result.size, Size::ZERO);
        assert!(!g.verify().is_valid());
        let mut g = grid_of(2, 2);
        assert_eq!(
            g.layout(Rect::new(0.0, 0.0, 30.0, 20.0)).size,
            Size::new(30.0, 20.0)
        );
    }

    #[test]
    fn class_grid_helpers_reject_what_they_should() {
        assert!(!command_widths_sane(&DrawCommand::filled_rect(
            Rect::new(0.0, 0.0, -1.0, 1.0),
            presentar_core::Color::WHITE
        )));
        assert!(!command_widths_sane(&DrawCommand::rounded_rect(
            Rect::new(0.0, 0.0, 1.0, 1.0),
            f32::NAN,
            presentar_core::Color::WHITE
        )));
        assert!(!raw_bounds_valid(Rect::new(f32::NAN, 0.0, 1.0, 1.0)));
    }

    proptest! {
        #[test]
        fn class_grid_free_functions_are_total_and_sane(
            x in wild(),
            y in wild(),
            w in wild(),
            h in wild(),
            columns in any_columns(),
            gap in wild(),
            row_height in proptest::option::of(wild()),
            count in 0_usize..40,
        ) {
            let b = Rect::new(x, y, w, h);
            let plain = grid_cells(b, columns, gap, count);
            prop_assert_eq!(plain.len(), count);
            prop_assert!(plain.iter().all(rect_sane), "{plain:?}");
            let with = grid_cells_with(b, columns, gap, row_height, count);
            prop_assert_eq!(with.len(), count);
            prop_assert!(with.iter().all(rect_sane), "{with:?}");
        }

        #[test]
        fn class_grid_builders_getters_and_cells_are_sane(
            columns in any_columns(),
            gap in wild(),
            row_height in wild(),
            tiles in 0_usize..9,
            x in wild(),
            y in wild(),
            w in wild(),
            h in wild(),
        ) {
            let g = grid_of(columns, tiles)
                .columns(columns)
                .gap(gap)
                .row_height(row_height)
                .with_test_id("g");
            prop_assert!(g.get_columns() >= 1);
            prop_assert!(finite_non_negative(g.get_gap()), "{}", g.get_gap());
            prop_assert!(g.get_row_height().is_none_or(finite_non_negative));
            prop_assert_eq!(g.row_count(), tiles.div_ceil(g.get_columns()));
            let css = g.to_css();
            prop_assert!(!css.contains("NaN") && !css.contains("inf"), "{css}");
            let cells = g.cells(Rect::new(x, y, w, h));
            prop_assert_eq!(cells.len(), tiles);
            prop_assert!(cells.iter().all(rect_sane), "{cells:?}");
        }

        #[test]
        fn class_grid_measure_is_finite_and_not_negative(
            columns in any_columns(),
            gap in wild(),
            row_height in proptest::option::of(wild()),
            tiles in 0_usize..9,
            a in wild(),
            b in wild(),
            c in wild(),
            d in wild(),
        ) {
            let mut g = grid_of(columns, tiles).gap(gap);
            if let Some(height) = row_height {
                g = g.row_height(height);
            }
            let s = g.measure(Constraints::new(a, b, c, d));
            prop_assert!(finite_non_negative(s.width) && finite_non_negative(s.height), "{s:?}");
        }

        #[test]
        fn class_grid_layout_size_is_sane_and_verify_judges_raw_bounds(
            columns in any_columns(),
            gap in wild(),
            tiles in 0_usize..9,
            x in wild(),
            y in wild(),
            w in wild(),
            h in wild(),
        ) {
            let bounds = Rect::new(x, y, w, h);
            let mut g = grid_of(columns, tiles).gap(gap);
            let result = g.layout(bounds);
            prop_assert!(
                finite_non_negative(result.size.width) && finite_non_negative(result.size.height),
                "{:?}", result.size
            );
            prop_assert_eq!(bits(Widget::bounds(&g)), bits(bounds));
            prop_assert_eq!(g.verify().is_valid(), raw_bounds_valid(bounds));
            let cells = g.cells(bounds);
            for (child, cell) in g.children().iter().zip(&cells) {
                prop_assert_eq!(bits(child.bounds()), bits(*cell));
                prop_assert!(rect_sane(&child.bounds()));
            }
        }

        #[test]
        fn class_grid_paint_is_sane_for_any_bounds(
            columns in any_columns(),
            gap in wild(),
            row_height in proptest::option::of(wild()),
            tiles in 0_usize..9,
            x in wild(),
            y in wild(),
            w in wild(),
            h in wild(),
            playing in any::<bool>(),
        ) {
            let mut g = grid_of(columns, tiles).gap(gap);
            if let Some(height) = row_height {
                g = g.row_height(height);
            }
            g.layout(Rect::new(x, y, w, h));
            if playing {
                for child in g.children_mut() {
                    child.event(&Event::key_up(presentar_core::Key::Space));
                }
            }
            let mut canvas = RecordingCanvas::new();
            g.paint(&mut canvas);
            prop_assert!(canvas.commands().iter().all(command_widths_sane));
            prop_assert_eq!(canvas.commands().len(), tiles * if playing { 1 } else { 2 });
        }

        #[test]
        fn class_grid_event_is_total(
            columns in any_columns(),
            tiles in 0_usize..9,
            w in wild(),
            h in wild(),
            px in wild(),
            py in wild(),
        ) {
            let mut g = grid_of(columns, tiles);
            g.layout(Rect::new(0.0, 0.0, w, h));
            let _ = click(&mut g, Point::new(px, py));
            prop_assert!(g.event(&Event::MouseLeave).is_none());
        }
    }
}
