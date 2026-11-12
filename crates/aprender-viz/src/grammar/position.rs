//! Position adjustments: `position_stack`, `position_fill` and `position_dodge`.
//!
//! These are the numeric parts of ggplot2 4.0.3's `PositionStack`, `PositionFill` and
//! `PositionDodge` (`pos_stack()`, `pos_dodge()` and `collide()` in `position-collide.R`), applied
//! to bar rows in data space. Every default is ggplot2's, taken from `formals()` on the reference
//! host rather than from memory, and is a named constant here.
//!
//! Scope: vertical bars only (`orientation = "x"`, no `flipped_aes`), and for
//! `position_dodge` neither `reverse = TRUE` nor `position_dodge2`. Those branches are not
//! implemented and have no oracle row.

/// `position_stack(vjust = 1)`: the stack's top edge carries the point (`formals(position_stack)`).
pub const STACK_VJUST: f64 = 1.0;
/// `position_stack(reverse = FALSE)`: the first group sits on top of the stack.
pub const STACK_REVERSE: bool = false;
/// `position_fill(vjust = 1)`.
pub const FILL_VJUST: f64 = 1.0;
/// `position_dodge(width = NULL)`: dodge within the bar's own width.
pub const DODGE_WIDTH: Option<f64> = None;
/// `position_dodge(preserve = "total")`: bars at an `x` share that `x`'s width between them.
pub const DODGE_PRESERVE: Preserve = Preserve::Total;

/// `position_dodge(preserve =)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preserve {
    /// Each `x` position's bars share its full width.
    Total,
    /// Every bar gets the width it would have at the busiest `x` position.
    Single,
}

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
    /// `position_dodge(width, preserve)`.
    Dodge {
        /// Spacing between dodged bar centres; `None` uses the bar width.
        width: Option<f64>,
        /// How the bar width is shared.
        preserve: Preserve,
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

    /// `position_dodge()` with ggplot2's defaults.
    #[must_use]
    pub fn dodge() -> Self {
        Position::Dodge { width: DODGE_WIDTH, preserve: DODGE_PRESERVE }
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

/// Apply `position` to `rows` in place. Row order is preserved.
pub fn apply(rows: &mut [BarRow], position: Position) {
    match position {
        Position::Identity => {}
        Position::Stack { vjust, reverse } => stack(rows, vjust, reverse, false),
        Position::Fill { vjust, reverse } => stack(rows, vjust, reverse, true),
        Position::Dodge { width, preserve } => {
            let n_single = max_groups_per_column(rows);
            dodge(rows, width, preserve, n_single);
        }
    }
}

/// Apply `position` to a layer whose rows are split by panel, as ggplot2 does: stacking, filling and
/// dodging happen inside each panel (`Position$compute_layer` splits by `PANEL`).
///
/// The one quantity shared across panels is `position_dodge(preserve = "single")`'s `n`: it is set
/// once from the whole layer (`PositionDodge$setup_params`, over unique `(PANEL, xmin)` cells), so it
/// is the largest group count at any x in ANY panel.
pub fn apply_panels(panels: &mut [Vec<BarRow>], position: Position) {
    match position {
        Position::Dodge { width, preserve } => {
            let n_single = panels.iter().map(|p| max_groups_per_column(p)).max().unwrap_or(1);
            for rows in panels.iter_mut() {
                dodge(rows, width, preserve, n_single);
            }
        }
        _ => {
            for rows in panels.iter_mut() {
                apply(rows, position);
            }
        }
    }
}

/// The largest number of distinct groups at any one `xmin`.
fn max_groups_per_column(rows: &[BarRow]) -> usize {
    let all: Vec<usize> = (0..rows.len()).collect();
    by_xmin(rows, &all).iter().map(|c| distinct_groups(rows, c).len()).max().unwrap_or(1)
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
        let (lo, hi) = (a.min(b), a.max(b));
        // `max_is_lower` (ymax < ymin) never holds for rows built by this crate: ymin <= ymax.
        rows[*i].y = (1.0 - vjust) * lo + vjust * hi;
        rows[*i].ymin = lo;
        rows[*i].ymax = hi;
    }
}

/// Distinct groups among `column`'s rows, ascending (ggplot2's `vec_unique_count(df$group)` set).
fn distinct_groups(rows: &[BarRow], column: &[usize]) -> Vec<u32> {
    let mut g: Vec<u32> = column.iter().map(|i| rows[*i].group).collect();
    g.sort_unstable();
    g.dedup();
    g
}

/// `PositionDodge`: the groups at an `xmin` split the bar width between them, in group order.
///
/// ggplot2 counts DISTINCT GROUPS, not rows: `n <- vec_unique_count(df$group)` and
/// `groupidx <- df$order`, where `order` is the dense rank of the row's group among the groups at
/// its x (`PositionDodge$setup_data`). Two rows of one group at one x (a `geom_col()` with a
/// repeated x/fill pair) therefore share a slot. With `preserve = "single"`, `n` is the largest
/// number of distinct groups at any one `xmin`.
fn dodge(rows: &mut [BarRow], width: Option<f64>, preserve: Preserve, n_single: usize) {
    let all: Vec<usize> = (0..rows.len()).collect();
    let columns = by_xmin(rows, &all);
    // collide(): without `width`, the first bar's own width (xmax - xmin).
    let width = width.unwrap_or_else(|| rows.first().map_or(0.0, |r| r.xmax - r.xmin));
    for column in columns {
        let groups = distinct_groups(rows, &column);
        let n = match preserve {
            Preserve::Total => groups.len(),
            Preserve::Single => n_single,
        };
        if n == 1 {
            continue; // pos_dodge(): `if (n == 1) return(df)`
        }
        let d_width = column.iter().map(|i| rows[*i].xmax - rows[*i].xmin).fold(0.0, f64::max);
        for i in &column {
            // `order`: dense rank of this row's group among the distinct groups at this x.
            let rank = groups.binary_search(&rows[*i].group).map_or(0, |p| p + 1);
            let x = rows[*i].x + width * ((rank as f64 - 0.5) / n as f64 - 0.5);
            rows[*i].x = x;
            rows[*i].xmin = x - d_width / n as f64 / 2.0;
            rows[*i].xmax = x + d_width / n as f64 / 2.0;
        }
    }
}
