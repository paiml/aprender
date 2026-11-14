//! `stat_count` + `geom_bar`, `stat_identity` + `geom_col`: bar rows and the discrete x axis.
//!
//! [`bar_layout`] builds what `ggplot_build()` holds for a bar layer over a discrete `x`: the
//! counted (or given) heights, `GeomBar`'s `setup_data` edges, the position adjustment, and the
//! trained x and y scales. All of it is computed from the data; nothing is read from a fixture.
//!
//! Defaults, from ggplot2 4.0.3 (`GeomBar$default_aes`, `StatCount$setup_params`, `default_expansion()`):
//! bar width is `0.9 * resolution(x)` and the discrete scale expands by `add = 0.6`.
//!
//! # Reconciliation note (discrete scales)
//!
//! The discrete x axis here is the minimum this row needs: levels sorted bytewise (`LC_COLLATE=C`, the pinned oracle locale), mapped to
//! `1..=n`, expanded by [`DISCRETE_EXPAND_ADD`], and unioned with the bars' unexpanded extent.
//! `scale_discrete` already owns the discrete position scale (same `add = 0.6`); folding
//! [`DiscreteAxis`] and `level_positions` into it is left to a later change. Nothing in the
//! grammar calls [`bar_layout_in`] yet: wiring it into `GGPlot::layout()` is a later slice.
//!
//! # Scope
//!
//! `x` and the optional `fill` are text columns (discrete); `x` numeric is not supported. Every column
//! read must hold one value per row of the frame ([`BarError::ShortColumn`] otherwise), and an identity
//! `y` must be finite ([`BarError::NonFinite`]: ggplot2 keeps a `NaN` row through `ggplot_build()` with an
//! `NA` x and edges, which this crate does not reproduce). Group ids follow `add_group()`: the dense rank
//! of the (x, fill) combinations present over the WHOLE layer data (before any facet split), both compared
//! bytewise (`LC_COLLATE=C`). The first column ranked is the first discrete aesthetic of ggplot2's
//! `computed_mapping`, the layer's own aesthetics before the plot's: x, unless `fill` is mapped on the
//! layer and `x` only on the plot ([`BarSpec::fill_first`]). Within one `aes()` this crate assumes x is
//! written before fill (the builder does not record the order), so `aes(fill = g, x = x)` on one level
//! is out of scope.
//! Weights, `orientation = "y"`, `just`, `na.rm`, `position_dodge(reverse =)` and `position_dodge2` are
//! not implemented. `width =` is implemented (`geom_bar(width =)`).
//!
//! Facets: [`bar_layout_in`] computes stat and position PER PANEL, as ggplot2 does, over x levels
//! shared by all panels (fixed scales).

use std::collections::BTreeMap;

use super::build::PositionScale;
use super::data::{DataFrame, DataValue};
use super::position::{apply_panels, BarRow, Position};

/// `GeomBar$default_aes$width` / `StatCount$setup_params`: a bar fills 90% of its resolution.
pub const DEFAULT_BAR_WIDTH: f64 = 0.9;
/// `GeomBar$setup_data`'s `just = 0.5`: bars are centred on `x`.
pub const BAR_JUST: f64 = 0.5;
/// `default_expansion(discrete = expansion(add = 0.6))`.
pub const DISCRETE_EXPAND_ADD: f64 = 0.6;

/// What went wrong building a bar layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarError {
    /// A named column is not in the data frame.
    MissingColumn(String),
    /// A column that must be discrete (text) holds numbers, or `y` holds text.
    WrongType(String),
    /// A column holds fewer (or more) values than the frame has rows.
    ShortColumn {
        /// The column.
        column: String,
        /// Its length.
        len: usize,
        /// The frame's row count.
        nrow: usize,
    },
    /// An identity `y` holds `NaN` or an infinity.
    NonFinite(String),
}

impl std::fmt::Display for BarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BarError::MissingColumn(c) => write!(f, "column `{c}` not in the data"),
            BarError::WrongType(c) => {
                write!(f, "column `{c}` has the wrong type for this aesthetic")
            }
            BarError::ShortColumn { column, len, nrow } => {
                write!(f, "column `{column}` has {len} values for {nrow} rows")
            }
            BarError::NonFinite(c) => write!(f, "column `{c}` holds a NaN or infinite value"),
        }
    }
}

impl std::error::Error for BarError {}

/// Which statistic feeds the bars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarStat {
    /// `stat_count`: one bar per `x` per group, its height the number of rows.
    Count,
    /// `stat_identity`: the named numeric column is the height of each row (`geom_col()`).
    Identity(String),
}

/// A bar layer over a discrete `x`.
#[derive(Debug, Clone, PartialEq)]
pub struct BarSpec {
    /// The discrete x column.
    pub x: String,
    /// The fill column; it joins the grouping.
    pub fill: Option<String>,
    /// `aes(group = 1)`: every row in one group.
    pub group_all: bool,
    /// The statistic.
    pub stat: BarStat,
    /// The position adjustment (ggplot2's default for bars is [`Position::stack`]).
    pub position: Position,
    /// `geom_bar(width =)`: the bar width as a fraction of the x resolution (1 for a discrete x).
    pub width: f64,
    /// Rank `fill` before `x` in the group ids: ggplot2 does when `fill` is a layer aesthetic and `x`
    /// a plot aesthetic (`computed_mapping` lists the layer's aesthetics first).
    pub fill_first: bool,
}

impl BarSpec {
    /// `geom_bar(aes(x))`: `stat_count`, `position_stack`.
    #[must_use]
    pub fn bar(x: &str) -> Self {
        Self {
            x: x.to_string(),
            fill: None,
            group_all: false,
            stat: BarStat::Count,
            position: Position::stack(),
            width: DEFAULT_BAR_WIDTH,
            fill_first: false,
        }
    }

    /// `geom_col(aes(x, y))`: `stat_identity`, `position_stack`.
    #[must_use]
    pub fn col(x: &str, y: &str) -> Self {
        Self { stat: BarStat::Identity(y.to_string()), ..Self::bar(x) }
    }

    /// Map `fill` to a discrete column.
    #[must_use]
    pub fn fill(mut self, column: &str) -> Self {
        self.fill = Some(column.to_string());
        self
    }

    /// `aes(group = 1)`.
    #[must_use]
    pub fn group_all(mut self) -> Self {
        self.group_all = true;
        self
    }

    /// Set the position adjustment.
    #[must_use]
    pub fn position(mut self, position: Position) -> Self {
        self.position = position;
        self
    }

    /// `geom_bar(width =)`.
    #[must_use]
    pub fn width(mut self, width: f64) -> Self {
        self.width = width;
        self
    }

    /// Rank `fill` before `x` in the group ids (see [`BarSpec::fill_first`]).
    #[must_use]
    pub fn fill_first(mut self, fill_first: bool) -> Self {
        self.fill_first = fill_first;
        self
    }
}

/// A trained discrete position axis, as the panel sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteAxis {
    /// The levels, in scale order.
    pub limits: Vec<String>,
    /// What the panel spans: the levels' `1..=n` expanded by [`DISCRETE_EXPAND_ADD`], unioned with
    /// the bars' extent.
    pub continuous_range: (f64, f64),
    /// Break positions: level `i` sits at `i + 1`.
    pub breaks: Vec<f64>,
    /// One label per break: the level names.
    pub labels: Vec<String>,
}

/// A built bar layer: rows in ggplot2's row order, and the two scales.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltBars {
    /// One entry per bar, panel by panel (ggplot2 orders the layer data by `PANEL`).
    pub rows: Vec<BarRow>,
    /// The panel (0-based, in panel order) of each row in `rows`.
    pub panel: Vec<usize>,
    /// The discrete x axis.
    pub x: DiscreteAxis,
    /// The continuous y axis, trained on `y`, `ymin` and `ymax` after the position adjustment.
    pub y: PositionScale,
}

/// The column `name`, which must hold exactly one value per row of `df`.
fn full_column<'a>(df: &'a DataFrame, name: &str) -> Result<&'a [DataValue], BarError> {
    let col = df.get(name).ok_or_else(|| BarError::MissingColumn(name.to_string()))?;
    if col.len() != df.nrow() {
        return Err(BarError::ShortColumn {
            column: name.to_string(),
            len: col.len(),
            nrow: df.nrow(),
        });
    }
    Ok(col)
}

fn text_column<'a>(df: &'a DataFrame, name: &str) -> Result<Vec<&'a str>, BarError> {
    let col = full_column(df, name)?;
    col.iter().map(|v| v.as_str().ok_or_else(|| BarError::WrongType(name.to_string()))).collect()
}

fn number_column(df: &DataFrame, name: &str) -> Result<Vec<f64>, BarError> {
    let col = full_column(df, name)?;
    col.iter()
        .map(|v| match v {
            DataValue::Number(n) if n.is_finite() => Ok(f64::from(*n)),
            DataValue::Number(_) => Err(BarError::NonFinite(name.to_string())),
            DataValue::Text(_) | DataValue::Null => Err(BarError::WrongType(name.to_string())),
        })
        .collect()
}

/// Sorted distinct values, in BYTEWISE order (`LC_COLLATE=C`). ggplot2 builds factor levels with
/// R's `sort()`, which collates by the locale: `en_US.UTF-8` gives a, B, c where C gives B, a, c.
/// The oracle is therefore pinned to `LC_COLLATE=C` (each `generate.R` sets it and records it in
/// the fixture as `collate`); other locales are out of scope.
pub fn sorted_levels<'a>(values: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut v: Vec<String> = values.into_iter().map(ToString::to_string).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Level `i` of `levels` sits at position `i + 1` on a discrete axis (`mapped_discrete`).
pub fn level_positions(levels: &[String], values: &[&str]) -> Vec<f64> {
    // Reversed, so a repeated level keeps its first position as a linear scan would.
    let at: std::collections::HashMap<&str, usize> =
        levels.iter().enumerate().rev().map(|(i, l)| (l.as_str(), i)).collect();
    values.iter().map(|v| at.get(v).map_or(f64::NAN, |p| (p + 1) as f64)).collect()
}

/// `add_group()`: dense rank of the (x, fill) combinations present; the first column of the key
/// varies slowest, and it is `fill` when `fill_first`.
fn group_ids(x: &[&str], fill: Option<&[&str]>, group_all: bool, fill_first: bool) -> Vec<u32> {
    if group_all {
        return vec![1; x.len()];
    }
    let key = |i: usize| match fill {
        Some(f) if fill_first => (f[i], Some(x[i])),
        Some(f) => (x[i], Some(f[i])),
        None => (x[i], None),
    };
    let mut keys: Vec<_> = (0..x.len()).map(key).collect();
    keys.sort_unstable();
    keys.dedup();
    (0..x.len()).map(|i| keys.binary_search(&key(i)).map_or(0, |p| (p + 1) as u32)).collect()
}

/// `GeomBar$setup_data`: edges from `y` and `width`.
fn bar_row(
    x: f64,
    y: f64,
    width: f64,
    group: u32,
    count: Option<f64>,
    prop: Option<f64>,
) -> BarRow {
    BarRow {
        x,
        y,
        ymin: y.min(0.0),
        ymax: y.max(0.0),
        xmin: x - width * BAR_JUST,
        xmax: x + width * (1.0 - BAR_JUST),
        width,
        group,
        count,
        prop,
    }
}

/// `StatCount$compute_group`, per group in group order: `count` per distinct x and
/// `prop = count / sum(abs(count))` within the group.
fn stat_count(x_pos: &[f64], group: &[u32], width: f64) -> Vec<BarRow> {
    let mut by_group: BTreeMap<u32, BTreeMap<u64, (f64, f64)>> = BTreeMap::new();
    for (x, g) in x_pos.iter().zip(group) {
        // x positions are small positive integers: their bit patterns order like the values.
        by_group.entry(*g).or_default().entry(x.to_bits()).or_insert((*x, 0.0)).1 += 1.0;
    }
    let mut rows = Vec::new();
    for (g, bars) in by_group {
        let total: f64 = bars.values().map(|(_, c)| c.abs()).sum();
        for (x, count) in bars.values() {
            rows.push(bar_row(*x, *count, width, g, Some(*count), Some(count / total)));
        }
    }
    rows
}

/// The discrete x axis for `levels`: the levels' `1..=n` expanded by [`DISCRETE_EXPAND_ADD`],
/// unioned with `extent`, the UNEXPANDED continuous range that layers contribute (`range_c`: bar
/// edges, numeric x of other layers).
#[must_use]
pub fn discrete_axis(levels: &[String], extent: Option<(f64, f64)>) -> DiscreteAxis {
    let n = levels.len() as f64;
    let (mut lo, mut hi) = (1.0 - DISCRETE_EXPAND_ADD, n + DISCRETE_EXPAND_ADD);
    if let Some((a, b)) = extent {
        lo = lo.min(a);
        hi = hi.max(b);
    }
    DiscreteAxis {
        limits: levels.to_vec(),
        continuous_range: (lo, hi),
        breaks: (1..=levels.len()).map(|i| i as f64).collect(),
        labels: levels.to_vec(),
    }
}

/// The `(min, max)` of the x edges of `rows`, if any.
#[must_use]
pub fn x_extent(rows: &[BarRow]) -> Option<(f64, f64)> {
    rows.iter().fold(None, |acc, r| match acc {
        None => Some((r.xmin, r.xmax)),
        Some((lo, hi)) => Some((lo.min(r.xmin), hi.max(r.xmax))),
    })
}

/// The `(min, max)` of `y`, `ymin` and `ymax` over `rows`, if any.
#[must_use]
pub fn y_extent(rows: &[BarRow]) -> Option<(f64, f64)> {
    let mut acc: Option<(f64, f64)> = None;
    for r in rows {
        for v in [r.y, r.ymin, r.ymax] {
            acc = Some(acc.map_or((v, v), |(lo, hi)| (lo.min(v), hi.max(v))));
        }
    }
    acc
}

/// Build a bar layer on one panel holding every row, with x levels taken from this data.
///
/// # Errors
///
/// [`BarError`] if a named column is missing or does not hold one value per row, `x`/`fill` is not
/// text, or the identity `y` is not a finite number.
pub fn bar_layout(df: &DataFrame, spec: &BarSpec) -> Result<BuiltBars, BarError> {
    let x = text_column(df, &spec.x)?;
    let levels = sorted_levels(x.iter().copied());
    bar_layout_in(df, spec, &levels, &[(0..df.nrow()).collect()])
}

/// Build a bar layer with `x_levels` as the shared discrete scale and `panels` as the row indices
/// of each panel.
///
/// Group ids come from the whole frame (`add_group()` runs before the facet split); the statistic
/// and the position adjustment run per panel, in panel order; the y scale and the x extent are
/// trained over all panels (fixed scales). `BuiltBars::panel` gives each row's panel index.
///
/// # Errors
///
/// As [`bar_layout`].
pub fn bar_layout_in(
    df: &DataFrame,
    spec: &BarSpec,
    x_levels: &[String],
    panels: &[Vec<usize>],
) -> Result<BuiltBars, BarError> {
    let x = text_column(df, &spec.x)?;
    let fill = spec.fill.as_deref().map(|f| text_column(df, f)).transpose()?;
    let ycol = match &spec.stat {
        BarStat::Count => None,
        BarStat::Identity(ycol) => Some(number_column(df, ycol)?),
    };
    let x_pos = level_positions(x_levels, &x);
    let group = group_ids(&x, fill.as_deref(), spec.group_all, spec.fill_first);

    let mut per_panel: Vec<Vec<BarRow>> = panels
        .iter()
        .map(|idx| match &ycol {
            None => {
                let xp: Vec<f64> = idx.iter().map(|i| x_pos[*i]).collect();
                let g: Vec<u32> = idx.iter().map(|i| group[*i]).collect();
                stat_count(&xp, &g, spec.width)
            }
            Some(y) => idx
                .iter()
                .map(|i| bar_row(x_pos[*i], y[*i], spec.width, group[*i], None, None))
                .collect(),
        })
        .collect();
    apply_panels(&mut per_panel, spec.position);

    let mut rows = Vec::new();
    let mut panel = Vec::new();
    for (p, panel_rows) in per_panel.into_iter().enumerate() {
        panel.extend(std::iter::repeat(p).take(panel_rows.len()));
        rows.extend(panel_rows);
    }
    let axis = discrete_axis(x_levels, x_extent(&rows));
    let y = PositionScale::continuous(y_extent(&rows).unwrap_or((f64::MAX, f64::MIN)));
    Ok(BuiltBars { rows, panel, x: axis, y })
}

#[cfg(test)]
mod tests {
    //! Every expected value is `ggplot_build()` output from R 4.6.2 + ggplot2 4.0.3 with
    //! `LC_COLLATE=C`: the layer data columns, `panel_params` x and y ranges, and the y scale limits.
    use super::*;

    fn frame(cols: &[(&str, &[&str])]) -> DataFrame {
        let mut df = DataFrame::new();
        for (name, values) in cols {
            df.add_column_str(name, values);
        }
        df
    }

    fn assert_close(got: &[f64], want: &[f64], what: &str) {
        let ok = got.len() == want.len()
            && got
                .iter()
                .zip(want)
                .all(|(g, w)| (g - w).abs() <= 1e-9 * g.abs().max(w.abs()) + 1e-12);
        assert!(ok, "{what}: got {got:?}, want {want:?}");
    }

    /// `x, y, ymin, ymax, xmin, xmax` per row, then group ids, `count`, `prop` and panels.
    fn check(b: &BuiltBars, cols: [&[f64]; 6], group: &[u32], count_prop: Option<[&[f64]; 2]>) {
        let get: [(&str, fn(&BarRow) -> f64); 6] = [
            ("x", |r| r.x),
            ("y", |r| r.y),
            ("ymin", |r| r.ymin),
            ("ymax", |r| r.ymax),
            ("xmin", |r| r.xmin),
            ("xmax", |r| r.xmax),
        ];
        for ((name, f), want) in get.into_iter().zip(cols) {
            assert_close(&b.rows.iter().map(f).collect::<Vec<_>>(), want, name);
        }
        assert_eq!(b.rows.iter().map(|r| r.group).collect::<Vec<_>>(), group);
        let opt = |f: fn(&BarRow) -> Option<f64>| b.rows.iter().map(f).collect::<Option<Vec<_>>>();
        match count_prop {
            Some([count, prop]) => {
                assert_close(&opt(|r| r.count).expect("count"), count, "count");
                assert_close(&opt(|r| r.prop).expect("prop"), prop, "prop");
            }
            None => assert!(b.rows.iter().all(|r| r.count.is_none() && r.prop.is_none())),
        }
    }

    fn check_axes(
        b: &BuiltBars,
        x_range: [f64; 2],
        labels: &[&str],
        y_lim: [f64; 2],
        y_range: [f64; 2],
    ) {
        assert_close(&[b.x.continuous_range.0, b.x.continuous_range.1], &x_range, "x range");
        assert_eq!(b.x.limits, labels);
        assert_eq!(b.x.labels, labels);
        assert_close(
            &b.x.breaks,
            &(1..=labels.len()).map(|i| i as f64).collect::<Vec<_>>(),
            "x breaks",
        );
        assert_close(&[b.y.limits.0, b.y.limits.1], &y_lim, "y limits");
        assert_close(&[b.y.continuous_range.0, b.y.continuous_range.1], &y_range, "y range");
    }

    /// `ggplot(aes(x, group = 1)) + geom_bar()`, x = b B a b a b: levels in C order (B < a < b), and
    /// `prop` is each count over the one group's total.
    #[test]
    fn bar_counts_each_level_in_c_order_with_prop_over_the_group() {
        let df = frame(&[("x", &["b", "B", "a", "b", "a", "b"])]);
        let b = bar_layout(&df, &BarSpec::bar("x").group_all()).expect("builds");
        let cols = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [0.0; 3],
            [1.0, 2.0, 3.0],
            [0.55, 1.55, 2.55],
            [1.45, 2.45, 3.45],
        ];
        let count_prop = [&[1.0, 2.0, 3.0][..], &[1.0 / 6.0, 2.0 / 6.0, 0.5]];
        check(&b, cols.each_ref().map(|c| &c[..]), &[1, 1, 1], Some(count_prop));
        assert!(b.rows.iter().all(|r| r.width == DEFAULT_BAR_WIDTH) && b.panel == [0, 0, 0]);
        check_axes(&b, [0.4, 3.6], &["B", "a", "b"], [0.0, 3.0], [-0.15, 3.15]);
    }

    /// `ggplot(aes(x, fill = g)) + geom_bar(position = "fill")`: groups rank (x, fill), x first.
    #[test]
    fn bar_with_fill_ranks_x_first_and_fills() {
        let x: &[&str] = &["a", "a", "b", "a", "b", "b", "b"];
        let df = frame(&[("x", x), ("g", &["v", "u", "u", "u", "v", "u", "v"])]);
        let b = bar_layout(&df, &BarSpec::bar("x").fill("g").position(Position::fill()))
            .expect("builds");
        let third = 1.0 / 3.0;
        let cols: [&[f64]; 6] = [
            &[1.0, 1.0, 2.0, 2.0],
            &[1.0, third, 1.0, 0.5],
            &[third, 0.0, 0.5, 0.0],
            &[1.0, third, 1.0, 0.5],
            &[0.55, 0.55, 1.55, 1.55],
            &[1.45, 1.45, 2.45, 2.45],
        ];
        check(&b, cols, &[1, 2, 3, 4], Some([&[2.0, 1.0, 2.0, 2.0], &[1.0; 4]]));
        check_axes(&b, [0.4, 2.6], &["a", "b"], [0.0, 1.0], [-0.05, 1.05]);
    }

    /// `ggplot(aes(x)) + geom_bar(aes(fill = g)) + facet_wrap(~p)` on the same rows: fill is a layer
    /// aesthetic, so groups rank (fill, x); ids come from the whole layer, counts and stacks per panel.
    #[test]
    fn bar_with_layer_fill_ranks_fill_first_and_counts_per_panel() {
        let x: &[&str] = &["a", "a", "b", "a", "b", "b", "b"];
        let df = frame(&[("x", x), ("g", &["v", "u", "u", "u", "v", "u", "v"])]);
        let spec = BarSpec::bar("x").fill("g").fill_first(true);
        let panels = [vec![0, 2, 3, 6], vec![1, 4, 5]];
        let b =
            bar_layout_in(&df, &spec, &sorted_levels(x.iter().copied()), &panels).expect("builds");
        let cols: [&[f64]; 6] = [
            &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0],
            &[2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            &[1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            &[0.55, 1.55, 0.55, 1.55, 0.55, 1.55, 1.55],
            &[1.45, 2.45, 1.45, 2.45, 1.45, 2.45, 2.45],
        ];
        check(&b, cols, &[1, 2, 3, 4, 1, 2, 4], Some([&[1.0; 7], &[1.0; 7]]));
        assert_eq!(b.panel, [0, 0, 0, 0, 1, 1, 1]);
        check_axes(&b, [0.4, 2.6], &["a", "b"], [0.0, 2.0], [-0.1, 2.1]);
    }

    /// `ggplot(aes(x, y)) + geom_col(width = 1.5)`, x = q p, y = -2 3: rows keep data order, the
    /// negative bar hangs from 0, and the bar edges widen the x range past the expanded levels.
    #[test]
    fn col_keeps_row_order_and_wide_bars_widen_the_x_range() {
        let mut df = frame(&[("x", &["q", "p"])]);
        df.add_column_f32("y", &[-2.0, 3.0]);
        let b = bar_layout(&df, &BarSpec::col("x", "y").width(1.5)).expect("builds");
        let cols: [&[f64]; 6] =
            [&[2.0, 1.0], &[-2.0, 3.0], &[-2.0, 0.0], &[0.0, 3.0], &[1.25, 0.25], &[2.75, 1.75]];
        check(&b, cols, &[2, 1], None);
        assert!(b.rows.iter().all(|r| r.width == 1.5));
        check_axes(&b, [0.25, 2.75], &["p", "q"], [-2.0, 3.0], [-2.25, 3.25]);
    }

    #[test]
    fn bad_columns_are_refused_naming_the_column() {
        let mut df = frame(&[("x", &["a", "b"]), ("t", &["1", "2"]), ("s", &["a"])]);
        df.add_column_f32("n", &[1.0, f32::NAN]);
        let short = BarError::ShortColumn { column: "s".into(), len: 1, nrow: 2 };
        let cases = [
            (BarSpec::bar("z"), BarError::MissingColumn("z".into()), "column `z` not in the data"),
            (BarSpec::bar("s"), short, "column `s` has 1 values for 2 rows"),
            (
                BarSpec::bar("n"),
                BarError::WrongType("n".into()),
                "column `n` has the wrong type for this aesthetic",
            ),
            (
                BarSpec::bar("x").fill("n"),
                BarError::WrongType("n".into()),
                "column `n` has the wrong type for this aesthetic",
            ),
            (
                BarSpec::col("x", "t"),
                BarError::WrongType("t".into()),
                "column `t` has the wrong type for this aesthetic",
            ),
            (
                BarSpec::col("x", "n"),
                BarError::NonFinite("n".into()),
                "column `n` holds a NaN or infinite value",
            ),
        ];
        for (spec, err, msg) in cases {
            assert_eq!(bar_layout(&df, &spec), Err(err.clone()), "{spec:?}");
            assert_eq!(err.to_string(), msg);
        }
    }

    /// R's `match()`: a repeated level keeps its first position; a value with no level maps to `NA`.
    #[test]
    fn a_repeated_level_keeps_its_first_position_and_an_unknown_value_is_nan() {
        let p = level_positions(&["a".into(), "b".into(), "a".into()], &["a", "b", "z"]);
        assert_eq!(p[..2], [1.0, 2.0]);
        assert!(p[2].is_nan());
    }
}
