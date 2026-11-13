//! Position adjustments: `position_stack` and `position_fill`.
//!
//! These are the numeric parts of ggplot2 4.0.3's `PositionStack` and `PositionFill`
//! (`pos_stack()` and `collide()` in `position-collide.R`), applied
//! to bar rows in data space. Every default is ggplot2's, taken from `formals()` on the reference
//! host rather than from memory, and is a named constant here.
//!
//! Scope: vertical bars only (`orientation = "x"`, no `flipped_aes`). `position_dodge` is not
//! implemented here and has no oracle row.
//!
//! `NaN` stands for R's `NA`. Known differences from R, outside the tested domain: R's `cumsum`
//! accumulates in long double, so a stacked edge can differ from R in the last ulp; and
//! `vjust = NaN`, which R rejects with an error, yields `NaN` positions here.

/// `position_stack(vjust = 1)`: the stack's top edge carries the point (`formals(position_stack)`).
pub const STACK_VJUST: f64 = 1.0;
/// `position_stack(reverse = FALSE)`: the first group sits on top of the stack.
pub const STACK_REVERSE: bool = false;
/// `position_fill(vjust = 1)`.
pub const FILL_VJUST: f64 = 1.0;

/// A position adjustment applied to a bar layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    /// `position_identity()`: leave the rows alone.
    Identity,
    /// `position_stack(vjust, reverse)`: ggplot2's default for `geom_bar()` and `geom_col()`.
    Stack {
        /// Where in each segment the row's `y` falls: 0 = bottom edge, 1 = top edge.
        vjust: f64,
        /// Stack the first group at the bottom instead of the top.
        reverse: bool,
    },
    /// `position_fill(vjust, reverse)`: a stack normalised to a total height of 1.
    Fill {
        /// As [`Position::Stack`].
        vjust: f64,
        /// As [`Position::Stack`].
        reverse: bool,
    },
}

impl Position {
    /// `position_stack()` with ggplot2's defaults.
    #[must_use]
    pub fn stack() -> Self {
        Position::Stack { vjust: STACK_VJUST, reverse: STACK_REVERSE }
    }

    /// `position_fill()` with ggplot2's defaults.
    #[must_use]
    pub fn fill() -> Self {
        Position::Fill { vjust: FILL_VJUST, reverse: STACK_REVERSE }
    }
}

/// One bar: the columns `ggplot_build()` holds for a bar layer that a position adjustment reads
/// or writes.
#[derive(Debug, Clone, PartialEq)]
pub struct BarRow {
    /// Bar centre on the discrete axis (the level's position, 1-based, before dodging).
    pub x: f64,
    /// The row's `y`: the value for `geom_col()`, the count for `geom_bar()`; after stacking, the
    /// point selected by `vjust`.
    pub y: f64,
    /// Lower edge.
    pub ymin: f64,
    /// Upper edge.
    pub ymax: f64,
    /// Left edge.
    pub xmin: f64,
    /// Right edge.
    pub xmax: f64,
    /// Bar width.
    pub width: f64,
    /// ggplot2's group id (1-based).
    pub group: u32,
    /// `stat_count`'s `count`; `None` for `stat_identity`.
    pub count: Option<f64>,
    /// `stat_count`'s `prop`; `None` for `stat_identity`.
    pub prop: Option<f64>,
}

/// Apply `position` in place to the rows of a single-panel layer. Row order is preserved.
pub fn apply(rows: &mut [BarRow], position: Position) {
    if let Some((vjust, reverse, fill)) = stacking(position, has_repeated_x(rows.iter())) {
        stack(rows, vjust, reverse, fill);
    }
}

/// Apply `position` to a layer whose rows are split by panel, as ggplot2 does: stacking and filling
/// happen inside each panel (`Position$compute_layer` splits by `PANEL`).
pub fn apply_panels(panels: &mut [Vec<BarRow>], position: Position) {
    if let Some((vjust, reverse, fill)) =
        stacking(position, has_repeated_x(panels.iter().flatten()))
    {
        for rows in panels.iter_mut() {
            stack(rows, vjust, reverse, fill);
        }
    }
}

/// `(vjust, reverse, fill)` when the layer is stacked, `None` when it is left alone.
/// PositionStack$setup_params: with no repeated x in the whole layer, fill off and every vjust
/// equal to 1, `var` is NULL and neither setup_data nor compute_panel touches the data, so a
/// negative bar over a unique x keeps y = value (stacking would set it to 0).
fn stacking(position: Position, repeated_x: bool) -> Option<(f64, bool, bool)> {
    match position {
        Position::Identity => None,
        Position::Stack { vjust, .. } if vjust == 1.0 && !repeated_x => None,
        Position::Stack { vjust, reverse } => Some((vjust, reverse, false)),
        Position::Fill { vjust, reverse } => Some((vjust, reverse, true)),
    }
}

/// `vec_duplicate_any(data$x)` over every row of the layer.
fn has_repeated_x<'a>(rows: impl Iterator<Item = &'a BarRow>) -> bool {
    let mut seen = std::collections::HashSet::new();
    // vctrs compares values, not bits: `+ 0.0` maps -0.0 to 0.0, and every NaN (either sign, any
    // payload) is one value.
    let key = |x: f64| if x.is_nan() { f64::NAN.to_bits() } else { (x + 0.0).to_bits() };
    !rows.into_iter().all(|r| seen.insert(key(r.x)))
}

/// Row indices grouped by identical `xmin`, in ascending `xmin` (ggplot2's `dapply(data, "xmin")`).
fn by_xmin(rows: &[BarRow], subset: &[usize]) -> Vec<Vec<usize>> {
    let mut idx: Vec<usize> = subset.to_vec();
    idx.sort_by(|a, b| rows[*a].xmin.total_cmp(&rows[*b].xmin));
    let mut out: Vec<Vec<usize>> = Vec::new();
    for i in idx {
        match out.last_mut() {
            Some(last) if rows[last[0]].xmin == rows[i].xmin => last.push(i),
            _ => out.push(vec![i]),
        }
    }
    out
}

/// `PositionStack$compute_panel`: negative bars (`ymax < 0`) and the rest are stacked separately,
/// each outward from zero.
fn stack(rows: &mut [BarRow], vjust: f64, reverse: bool, fill: bool) {
    // PositionStack$setup_data: a bar that ends at zero (ymax == 0) is a negative bar; its ymax
    // becomes its ymin so that the sign test below sees it.
    for r in rows.iter_mut() {
        if r.ymax == 0.0 {
            r.ymax = r.ymin;
        }
        // `detect_missing(data, c("x", "xmin", "xmax", "y"))`: a row missing any of the four has
        // all four set to NA (±Inf counts as present). Such a row stacks as a zero-height bar.
        if [r.x, r.xmin, r.xmax, r.y].iter().any(|v| v.is_nan()) {
            (r.x, r.xmin, r.xmax, r.y) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }
    }
    let (neg, pos): (Vec<usize>, Vec<usize>) = (0..rows.len()).partition(|i| rows[*i].ymax < 0.0);
    for subset in [neg, pos] {
        for mut column in by_xmin(rows, &subset) {
            // collide(): order(xmin, -group), or order(xmin, group) when reversed. The first
            // element is the bottom of the stack.
            column.sort_by(|a, b| {
                let (ga, gb) = (rows[*a].group, rows[*b].group);
                if reverse {
                    ga.cmp(&gb)
                } else {
                    gb.cmp(&ga)
                }
            });
            pos_stack(rows, &column, vjust, fill);
        }
    }
}

/// `pos_stack()`: cumulative heights in stacking order.
fn pos_stack(rows: &mut [BarRow], column: &[usize], vjust: f64, fill: bool) {
    let mut heights = vec![0.0_f64];
    let mut sum = 0.0;
    for i in column {
        let y = if rows[*i].y.is_nan() { 0.0 } else { rows[*i].y };
        sum += y;
        heights.push(sum);
    }
    if fill {
        let total = heights[heights.len() - 1].abs();
        if total > f64::EPSILON.sqrt() {
            for h in &mut heights {
                *h /= total;
            }
        }
    }
    for (k, i) in column.iter().enumerate() {
        let (a, b) = (heights[k], heights[k + 1]);
        // pmin()/pmax() propagate NaN (an Inf/Inf fill total); f64::min/max would drop it.
        let (lo, hi) =
            if a.is_nan() || b.is_nan() { (f64::NAN, f64::NAN) } else { (a.min(b), a.max(b)) };
        rows[*i].y = (1.0 - vjust) * lo + vjust * hi;
        // `max_is_lower <- ymax < ymin` is NA when an edge is NA, and ifelse() then gives NA edges.
        // Otherwise it never holds for rows built by this crate: ymin <= ymax.
        let na_edge = rows[*i].ymin.is_nan() || rows[*i].ymax.is_nan();
        rows[*i].ymin = if na_edge { f64::NAN } else { lo };
        rows[*i].ymax = if na_edge { f64::NAN } else { hi };
    }
}

#[cfg(test)]
mod tests {
    //! Expected values are `ggplot_build()` output from R 4.6.2 + ggplot2 4.0.3 (`LC_ALL=C`), one
    //! `geom_col()` per case with the named position; the inputs are that layer's identity rows.
    use super::{apply, apply_panels, BarRow, Position};

    /// One `geom_col()` row as ggplot2 builds it: width 0.9, centred on `x`.
    fn row(x: f64, group: u32, y: f64, ymin: f64, ymax: f64) -> BarRow {
        BarRow {
            x,
            y,
            ymin,
            ymax,
            xmin: x - 0.45,
            xmax: x + 0.45,
            width: 0.9,
            group,
            count: None,
            prop: None,
        }
    }

    /// Case A: two bars at x = 1, three at x = 2, two of those negative.
    fn case_a() -> Vec<BarRow> {
        let x = [1.0, 1.0, 2.0, 2.0, 2.0];
        let y = [1.0, 2.0, -1.0, 3.0, -2.0];
        (0..5).map(|i| row(x[i], i as u32 + 1, y[i], y[i].min(0.0), y[i].max(0.0))).collect()
    }

    fn col(rows: &[BarRow], f: fn(&BarRow) -> f64) -> Vec<f64> {
        rows.iter().map(f).collect()
    }

    fn adjusted(mut rows: Vec<BarRow>, position: Position) -> Vec<BarRow> {
        let mut panels = vec![std::mem::take(&mut rows)];
        apply_panels(&mut panels, position);
        panels.remove(0)
    }

    /// Equal values, with NaN (R's NA) equal to NaN.
    fn assert_col(rows: &[BarRow], f: fn(&BarRow) -> f64, want: &[f64], name: &str) {
        let got = col(rows, f);
        let same = got.len() == want.len()
            && got.iter().zip(want).all(|(g, w)| g == w || (g.is_nan() && w.is_nan()));
        assert!(same, "{name}: got {got:?}, want {want:?}");
    }

    fn assert_y(rows: &[BarRow], y: &[f64], ymin: &[f64], ymax: &[f64]) {
        assert_col(rows, |r| r.y, y, "y");
        assert_col(rows, |r| r.ymin, ymin, "ymin");
        assert_col(rows, |r| r.ymax, ymax, "ymax");
    }

    #[test]
    fn stack_splits_negatives_and_puts_the_first_group_on_top() {
        let rows = adjusted(case_a(), Position::stack());
        assert_y(
            &rows,
            &[3.0, 2.0, -2.0, 3.0, 0.0],
            &[2.0, 0.0, -3.0, 0.0, -2.0],
            &[3.0, 2.0, -2.0, 3.0, 0.0],
        );
        // Position adjustments never move a stacked bar sideways.
        assert_eq!(col(&rows, |r| r.xmin), col(&case_a(), |r| r.xmin));
    }

    #[test]
    fn stack_reverse_puts_the_first_group_at_the_bottom() {
        let rows = adjusted(case_a(), Position::Stack { vjust: 1.0, reverse: true });
        assert_y(
            &rows,
            &[1.0, 3.0, 0.0, 3.0, -1.0],
            &[0.0, 1.0, -1.0, 0.0, -3.0],
            &[1.0, 3.0, 0.0, 3.0, -1.0],
        );
    }

    #[test]
    fn stack_vjust_half_puts_y_mid_segment() {
        let rows = adjusted(case_a(), Position::Stack { vjust: 0.5, reverse: false });
        assert_y(
            &rows,
            &[2.5, 1.0, -2.5, 1.5, -1.0],
            &[2.0, 0.0, -3.0, 0.0, -2.0],
            &[3.0, 2.0, -2.0, 3.0, 0.0],
        );
    }

    #[test]
    fn fill_normalises_each_side_by_its_own_total() {
        let t = 2.0 / 3.0;
        let rows = adjusted(case_a(), Position::fill());
        assert_y(&rows, &[1.0, t, -t, 1.0, 0.0], &[t, 0.0, -1.0, 0.0, -t], &[1.0, t, -t, 1.0, 0.0]);
    }

    #[test]
    fn fill_leaves_a_zero_total_column_unscaled() {
        let x = [1.0, 1.0, 2.0, 2.0];
        let y = [0.0, 0.0, 1.0, 3.0];
        let rows = (0..4).map(|i| row(x[i], i as u32 + 1, y[i], 0.0, y[i])).collect();
        let rows = adjusted(rows, Position::fill());
        assert_y(&rows, &[0.0, 0.0, 1.0, 0.75], &[0.0, 0.0, 0.75, 0.0], &[0.0, 0.0, 1.0, 0.75]);
    }

    #[test]
    fn fill_scales_only_a_total_above_sqrt_epsilon() {
        // Measured in R: a fill column whose total is sqrt(.Machine$double.eps) = 2^-26 is left
        // unscaled, and one of twice that is scaled to 1.
        let at = 1.490_116_119_384_765_6e-8; // 2^-26, R printed 1.4901161193847656e-08
        let rows = adjusted(vec![row(1.0, 1, at, 0.0, at)], Position::fill());
        assert_y(&rows, &[at], &[0.0], &[at]);
        let above = 2.0 * at;
        let rows = adjusted(vec![row(1.0, 1, above, 0.0, above)], Position::fill());
        assert_y(&rows, &[1.0], &[0.0], &[1.0]);
    }

    fn case_f() -> Vec<BarRow> {
        vec![row(1.0, 1, -1.0, -1.0, 0.0), row(2.0, 2, 2.0, 0.0, 2.0)]
    }

    #[test]
    fn stack_is_a_no_op_when_no_x_repeats() {
        let rows = adjusted(case_f(), Position::stack());
        assert_eq!(rows, case_f());
    }

    #[test]
    fn stack_with_vjust_not_one_stacks_even_without_repeats() {
        let rows = adjusted(case_f(), Position::Stack { vjust: 0.5, reverse: false });
        assert_y(&rows, &[-0.5, 1.0], &[-1.0, 0.0], &[0.0, 2.0]);
    }

    #[test]
    fn fill_stacks_even_without_repeats() {
        let rows = adjusted(case_f(), Position::fill());
        assert_y(&rows, &[0.0, 1.0], &[-1.0, 0.0], &[0.0, 1.0]);
    }

    #[test]
    fn stack_runs_inside_each_panel() {
        let mut panels = vec![
            vec![row(1.0, 1, 1.0, 0.0, 1.0), row(1.0, 2, 2.0, 0.0, 2.0)],
            vec![row(1.0, 1, 4.0, 0.0, 4.0)],
        ];
        apply_panels(&mut panels, Position::stack());
        assert_y(&panels[0], &[3.0, 2.0], &[2.0, 0.0], &[3.0, 2.0]);
        assert_y(&panels[1], &[4.0], &[0.0], &[4.0]);
    }

    #[test]
    fn an_x_repeated_across_panels_turns_stacking_on() {
        let mut panels =
            vec![vec![row(1.0, 1, -1.0, -1.0, 0.0)], vec![row(1.0, 1, -2.0, -2.0, 0.0)]];
        apply_panels(&mut panels, Position::stack());
        assert_y(&panels[0], &[0.0], &[-1.0], &[0.0]);
        assert_y(&panels[1], &[0.0], &[-2.0], &[0.0]);
    }

    #[test]
    fn signed_zeros_are_the_same_x() {
        // vctrs::vec_duplicate_any(c(0, -0)) is TRUE, so these two bars stack.
        let rows = vec![row(0.0, 1, 1.0, 0.0, 1.0), row(-0.0, 2, 1.0, 0.0, 1.0)];
        let rows = adjusted(rows, Position::stack());
        assert_y(&rows, &[2.0, 1.0], &[1.0, 0.0], &[2.0, 1.0]);
    }

    #[test]
    fn a_zero_height_bar_stacks_with_the_positive_bars() {
        let rows = vec![
            row(1.0, 1, 2.0, 0.0, 2.0),
            row(1.0, 2, 0.0, 0.0, 0.0),
            row(1.0, 3, 1.0, 0.0, 1.0),
        ];
        let rows = adjusted(rows, Position::stack());
        assert_y(&rows, &[3.0, 1.0, 1.0], &[1.0, 1.0, 0.0], &[3.0, 1.0, 1.0]);
    }

    const NAN: f64 = f64::NAN;

    /// Each row's x, and its edges x -/+ 0.45 (NaN stays NaN), as R reports them.
    fn assert_x(rows: &[BarRow], x: [f64; 3]) {
        assert_col(rows, |r| r.x, &x, "x");
        assert_col(rows, |r| r.xmin, &x.map(|v| v - 0.45), "xmin");
        assert_col(rows, |r| r.xmax, &x.map(|v| v + 0.45), "xmax");
    }

    #[test]
    fn a_nan_y_row_is_blanked_and_its_neighbours_stack_without_it() {
        // N_nan_y: R sets x, xmin, xmax and y of the NaN row to NA; it stacks as y = 0, NA edges.
        let rows = vec![
            row(1.0, 1, 1.0, 0.0, 1.0),
            row(1.0, 2, NAN, NAN, NAN),
            row(1.0, 3, 2.0, 0.0, 2.0),
        ];
        let rows = adjusted(rows, Position::stack());
        assert_y(&rows, &[3.0, 0.0, 2.0], &[2.0, NAN, 0.0], &[3.0, NAN, 2.0]);
        assert_x(&rows, [1.0, NAN, 1.0]);
    }

    #[test]
    fn nan_x_rows_are_blanked_to_zero_height_bars() {
        // Q_nan_x_dup and Q_negnan_x: a repeated NaN x (either sign) turns stacking on, and each
        // NaN-x row becomes y = ymin = ymax = 0 with NA x edges.
        for second in [NAN, -NAN] {
            let rows = vec![row(NAN, 1, 5.0, 0.0, 5.0), row(second, 2, 5.0, 0.0, 5.0)];
            let rows =
                adjusted([rows, vec![row(1.0, 3, 1.0, 0.0, 1.0)]].concat(), Position::stack());
            assert_y(&rows, &[0.0, 0.0, 1.0], &[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0]);
            assert_x(&rows, [NAN, NAN, 1.0]);
        }
    }

    #[test]
    fn a_nan_x_alone_blanks_the_row_and_a_nan_ymax_gives_na_edges() {
        // C_crafted, R's PositionStack methods on hand-built rows: x alone NaN still blanks the
        // row; a NaN ymax with a finite y stacks (y = 6) but its edges come out NA.
        let mut rows = vec![row(1.0, 1, 1.0, 0.0, 1.0), row(1.0, 2, 2.0, 0.0, NAN)];
        rows[0].x = NAN;
        let rows = adjusted([rows, vec![row(1.0, 3, 4.0, 0.0, 4.0)]].concat(), Position::stack());
        assert_y(&rows, &[0.0, 6.0, 4.0], &[0.0, NAN, 0.0], &[0.0, NAN, 4.0]);
        assert_x(&rows, [NAN, 1.0, 1.0]);
    }

    #[test]
    fn stack_vjust_two_stacks_even_without_repeats() {
        // Q_vjust2_unique: any vjust other than 1 turns stacking on; y lies at twice the top edge.
        let rows = (1..=3).map(|i| row(f64::from(i), i, f64::from(i), 0.0, f64::from(i))).collect();
        let rows = adjusted(rows, Position::Stack { vjust: 2.0, reverse: false });
        assert_y(&rows, &[2.0, 4.0, 6.0], &[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn fill_with_an_overflowing_total_gives_nan_like_r() {
        // Q_huge_fill: the total overflows to Inf, the top edge is Inf/Inf, and R's row 1 is NA.
        let rows = vec![row(1.0, 1, 1e308, 0.0, 1e308), row(1.0, 2, 1e308, 0.0, 1e308)];
        let rows = adjusted(rows, Position::fill());
        assert_y(&rows, &[NAN, 0.0], &[NAN, 0.0], &[NAN, 0.0]);
    }

    #[test]
    fn interleaved_x_rows_stack_by_column() {
        // I_interleaved: rows arrive unsorted by x and are stacked per column, in place.
        let (x, y) = ([2.0, 1.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]);
        let rows = (0..4).map(|i| row(x[i], i as u32 + 1, y[i], 0.0, y[i])).collect();
        let rows = adjusted(rows, Position::stack());
        assert_y(&rows, &[4.0, 6.0, 3.0, 4.0], &[3.0, 4.0, 0.0, 0.0], &[4.0, 6.0, 3.0, 4.0]);
    }

    #[test]
    fn apply_on_one_panel_keeps_the_unique_x_no_op() {
        // F_stack through the single-panel entry point: R keeps the negative bar at y = -1.
        let mut rows = case_f();
        apply(&mut rows, Position::stack());
        assert_eq!(rows, case_f());
        // ...and it stacks a layer with repeated x exactly as apply_panels does (A_stack).
        let mut rows = case_a();
        apply(&mut rows, Position::stack());
        assert_eq!(rows, adjusted(case_a(), Position::stack()));
    }

    #[test]
    fn identity_leaves_rows_alone() {
        let mut rows = case_a();
        apply(&mut rows, Position::Identity);
        assert_eq!(rows, case_a());
    }
}
