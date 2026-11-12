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
//! `1..=n`, expanded by [`DISCRETE_EXPAND_ADD`], and unioned with the bars' unexpanded extent. A
//! scales lane (PR #6916, `feat/plot-gg-001-scales-discrete-log`) owns the discrete position
//! scale and is changing in parallel; when it lands, [`DiscreteAxis`] should be replaced by its
//! type and `level_positions` by its mapping. The grammar reaches this code through ONE call,
//! [`bar_layout_in`], from `GGPlot::layout()`, so that merge is one call site.
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
        return Err(BarError::ShortColumn { column: name.to_string(), len: col.len(), nrow: df.nrow() });
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
    values
        .iter()
        .map(|v| levels.iter().position(|l| l == v).map_or(f64::NAN, |p| (p + 1) as f64))
        .collect()
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
