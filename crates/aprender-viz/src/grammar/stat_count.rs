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
//! The discrete x axis here is the minimum this row needs: levels sorted bytewise, mapped to
//! `1..=n`, expanded by [`DISCRETE_EXPAND_ADD`], and unioned with the bars' unexpanded extent. A
//! scales lane is adding discrete position scales to `build.rs`; when it lands, [`DiscreteAxis`]
//! should be replaced by its type and `level_positions` by its mapping.
//!
//! # Scope
//!
//! `x` and the optional `fill` are text columns (discrete); `x` numeric is not supported. Group
//! ids follow `add_group()`: the dense rank of the (x, fill) combinations present, x varying
//! slowest. Weights, `orientation = "y"`, `just`, `width =` and `na.rm` are not implemented.

use std::collections::BTreeMap;

use super::build::PositionScale;
use super::data::{DataFrame, DataValue};
use super::position::{apply, BarRow, Position};

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
}

impl std::fmt::Display for BarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BarError::MissingColumn(c) => write!(f, "column `{c}` not in the data"),
            BarError::WrongType(c) => {
                write!(f, "column `{c}` has the wrong type for this aesthetic")
            }
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
    /// One entry per bar.
    pub rows: Vec<BarRow>,
    /// The discrete x axis.
    pub x: DiscreteAxis,
    /// The continuous y axis, trained on `y`, `ymin` and `ymax` after the position adjustment.
    pub y: PositionScale,
}

fn text_column<'a>(df: &'a DataFrame, name: &str) -> Result<Vec<&'a str>, BarError> {
    let col = df.get(name).ok_or_else(|| BarError::MissingColumn(name.to_string()))?;
    col.iter().map(|v| v.as_str().ok_or_else(|| BarError::WrongType(name.to_string()))).collect()
}

fn number_column(df: &DataFrame, name: &str) -> Result<Vec<f64>, BarError> {
    let col = df.get(name).ok_or_else(|| BarError::MissingColumn(name.to_string()))?;
    col.iter()
        .map(|v| match v {
            DataValue::Number(n) => Ok(f64::from(*n)),
            DataValue::Text(_) | DataValue::Null => Err(BarError::WrongType(name.to_string())),
        })
        .collect()
}

/// Sorted distinct values (ggplot2 sorts factor levels; ASCII levels sort the same in any locale).
fn levels<'a>(values: &[&'a str]) -> Vec<&'a str> {
    let mut v = values.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// Level `i` of `levels` sits at position `i + 1` on a discrete axis (`mapped_discrete`).
fn level_positions(levels: &[&str], values: &[&str]) -> Vec<f64> {
    values
        .iter()
        .map(|v| levels.iter().position(|l| l == v).map_or(f64::NAN, |p| (p + 1) as f64))
        .collect()
}

/// `add_group()`: dense rank of the (x, fill) combinations present, x slowest.
fn group_ids(x: &[&str], fill: Option<&[&str]>, group_all: bool) -> Vec<u32> {
    if group_all {
        return vec![1; x.len()];
    }
    let key = |i: usize| (x[i], fill.map(|f| f[i]));
    let mut keys: Vec<_> = (0..x.len()).map(key).collect();
    keys.sort_unstable();
    keys.dedup();
    (0..x.len()).map(|i| keys.binary_search(&key(i)).map_or(0, |p| (p + 1) as u32)).collect()
}

/// `GeomBar$setup_data`: edges from `y` and `width`.
fn bar_row(x: f64, y: f64, group: u32, count: Option<f64>, prop: Option<f64>) -> BarRow {
    BarRow {
        x,
        y,
        ymin: y.min(0.0),
        ymax: y.max(0.0),
        xmin: x - DEFAULT_BAR_WIDTH * BAR_JUST,
        xmax: x + DEFAULT_BAR_WIDTH * (1.0 - BAR_JUST),
        width: DEFAULT_BAR_WIDTH,
        group,
        count,
        prop,
    }
}

/// `StatCount$compute_group`, per group in group order: `count` per distinct x and
/// `prop = count / sum(abs(count))` within the group.
fn stat_count(x_pos: &[f64], group: &[u32]) -> Vec<BarRow> {
    let mut by_group: BTreeMap<u32, BTreeMap<u64, (f64, f64)>> = BTreeMap::new();
    for (x, g) in x_pos.iter().zip(group) {
        // x positions are small positive integers: their bit patterns order like the values.
        by_group.entry(*g).or_default().entry(x.to_bits()).or_insert((*x, 0.0)).1 += 1.0;
    }
    let mut rows = Vec::new();
    for (g, bars) in by_group {
        let total: f64 = bars.values().map(|(_, c)| c.abs()).sum();
        for (x, count) in bars.values() {
            rows.push(bar_row(*x, *count, g, Some(*count), Some(count / total)));
        }
    }
    rows
}

/// The discrete x axis for `levels` given the bars' extent.
fn discrete_axis(levels: &[&str], rows: &[BarRow]) -> DiscreteAxis {
    let n = levels.len() as f64;
    let (mut lo, mut hi) = (1.0 - DISCRETE_EXPAND_ADD, n + DISCRETE_EXPAND_ADD);
    for r in rows {
        lo = lo.min(r.xmin);
        hi = hi.max(r.xmax);
    }
    DiscreteAxis {
        limits: levels.iter().map(ToString::to_string).collect(),
        continuous_range: (lo, hi),
        breaks: (1..=levels.len()).map(|i| i as f64).collect(),
        labels: levels.iter().map(ToString::to_string).collect(),
    }
}

/// Build a bar layer.
///
/// # Errors
///
/// [`BarError`] if a named column is missing, `x`/`fill` is not text, or the identity `y` is not
/// numeric.
pub fn bar_layout(df: &DataFrame, spec: &BarSpec) -> Result<BuiltBars, BarError> {
    let x = text_column(df, &spec.x)?;
    let fill = spec.fill.as_deref().map(|f| text_column(df, f)).transpose()?;
    let x_levels = levels(&x);
    let x_pos = level_positions(&x_levels, &x);
    let group = group_ids(&x, fill.as_deref(), spec.group_all);

    let mut rows = match &spec.stat {
        BarStat::Count => stat_count(&x_pos, &group),
        BarStat::Identity(ycol) => {
            let y = number_column(df, ycol)?;
            (0..x.len()).map(|i| bar_row(x_pos[i], y[i], group[i], None, None)).collect()
        }
    };
    apply(&mut rows, spec.position);

    let axis = discrete_axis(&x_levels, &rows);
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for r in &rows {
        for v in [r.y, r.ymin, r.ymax] {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let y = PositionScale::continuous((lo, hi));
    Ok(BuiltBars { rows, x: axis, y })
}
