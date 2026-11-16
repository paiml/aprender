//! `geom_boxplot()` through [`GGPlot::layout`](super::GGPlot::layout) (PLOT-GG-001,
//! paiml/aprender#6909): ggplot2 4.0.3's `stat_boxplot`, then `GeomBoxplot$setup_data`, then its
//! default `position_dodge2`, then the position scales retrained on the result.
//!
//! The steps, each as ggplot2 4.0.3 does them:
//!
//! 1. Rows with a missing `y`, or a missing numeric `x`, are dropped (R warns and drops them). The
//!    discrete `x` scale is trained on every row first, so a level whose rows were all dropped
//!    still takes a position.
//! 2. `add_group()`: the interaction of the discrete mappings (`x` when text, then `fill`, then
//!    `colour`), or a `group` column's dense rank, or `-1` when nothing is discrete. A box is made
//!    per group.
//! 3. `StatBoxplot$compute_group`: `quantile(y, type = 7)` at `0, .25, .5, .75, 1`; an outlier is
//!    a `y` beyond `coef * IQR` of the box; the whiskers reach the furthest value inside; the
//!    notch is `middle +- 1.58 * IQR / sqrt(n)`; `x` is the middle of the group's `x` range and
//!    the width `0.9 * diff(range(x))` when the group spans several `x`, else the `width`
//!    parameter, else `0.75 * resolution(x)`.
//! 4. `GeomBoxplot$setup_data`: `xmin/xmax = x -+ width / 2`, `ymin_final/ymax_final` include the
//!    outliers.
//! 5. `position_dodge2(padding = 0.1)`: boxes whose extents overlap share their cluster's span.
//! 6. Position scales: `y` over `ymin_final`, `ymax_final` and the quartiles; a continuous `x`
//!    over `xmin`..`xmax`; a text `x` keeps its levels and spans the union of the expanded
//!    levels and the unexpanded boxes (`expand_limits_discrete_trans`).
//!
//! Differences from R, each a typed error rather than a guess, are listed on [`layout`].

use std::collections::BTreeMap;

use crate::error::{Error, Result};

use super::aes::Aes;
use super::build::{BuiltLayer, BuiltLayout, PositionScale};
use super::data::{DataFrame, DataValue};
use super::dist::{column, is_text, key_order, keys, numeric, DistSpec, Key};
use super::scale_discrete::{resolve_discrete, train_levels, DiscreteSpec};

/// `position_dodge2(padding = 0.1)`.
const DODGE2_PADDING: f64 = 0.1;
/// `StatBoxplot`'s notch factor.
const NOTCH: f64 = 1.58;
/// `StatBoxplot$setup_params`: the default width is this fraction of the resolution.
const DEFAULT_WIDTH: f64 = 0.75;
/// `StatBoxplot$compute_group`: a group spanning several `x` is `0.9` of its `x` range wide.
const SPAN_WIDTH: f64 = 0.9;

/// One box, as ggplot2's `layer_data()` holds it after `position_dodge2`.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxRow {
    /// ggplot2's `group` (`-1` when nothing is discrete).
    pub group: i64,
    /// Box centre, after dodging.
    pub x: f64,
    /// Left edge, after dodging and padding.
    pub xmin: f64,
    /// Right edge, after dodging and padding.
    pub xmax: f64,
    /// Lower whisker end.
    pub ymin: f64,
    /// First quartile.
    pub lower: f64,
    /// Median.
    pub middle: f64,
    /// Third quartile.
    pub upper: f64,
    /// Upper whisker end.
    pub ymax: f64,
    /// `middle - 1.58 * IQR / sqrt(n)`.
    pub notchlower: f64,
    /// `middle + 1.58 * IQR / sqrt(n)`.
    pub notchupper: f64,
    /// `ymin` or the smallest outlier, whichever is lower.
    pub ymin_final: f64,
    /// `ymax` or the largest outlier, whichever is higher.
    pub ymax_final: f64,
    /// `sqrt(n)`: what `varwidth = TRUE` would scale by. Not in `layer_data()`; it is on the
    /// stat's own output.
    pub relvarwidth: f64,
    /// The values beyond the whiskers, in data order.
    pub outliers: Vec<f64>,
}

fn refuse(msg: impl Into<String>) -> Error {
    Error::Rendering(msg.into())
}

/// The `x` aesthetic: text levels, numbers, or absent (`x = 0`).
struct XInput {
    /// One position per row; `NaN` is a missing numeric `x`.
    values: Vec<f64>,
    /// The levels of a text `x`, in position order.
    levels: Option<Vec<String>>,
    /// Name of the text column, when `x` is text.
    text_column: Option<String>,
    /// Whether `x` is mapped_discrete (text) or absent: `resolution()` is then 1.
    resolution_one: bool,
}

fn read_x(data: &DataFrame, aes: &Aes) -> Result<XInput> {
    let n = data.nrow();
    let Some(name) = aes.x.as_deref() else {
        return Ok(XInput {
            values: vec![0.0; n],
            levels: None,
            text_column: None,
            resolution_one: true,
        });
    };
    if !is_text(data, name)? {
        return Ok(XInput {
            values: numeric(data, name)?,
            levels: None,
            text_column: None,
            resolution_one: false,
        });
    }
    let cells = column(data, name)?;
    let texts: Vec<&str> = cells.iter().filter_map(DataValue::as_str).collect();
    if texts.len() != n {
        return Err(refuse(format!(
            "column `{name}` mixes text with numbers or missing values: R would add a level for the \
             missing value, which is not supported"
        )));
    }
    let levels = train_levels(texts.iter().copied());
    let values = texts
        .iter()
        .map(|t| levels.iter().position(|l| l == t).map_or(f64::NAN, |p| (p + 1) as f64))
        .collect();
    Ok(XInput {
        values,
        levels: Some(levels),
        text_column: Some(name.to_string()),
        resolution_one: true,
    })
}

/// Dense ranks (0-based) of a column's values under [`key_order`]; a missing value ranks last, as
/// in R's `id()`. The caller decides whether a row with a missing value may survive.
fn ranks(data: &DataFrame, name: &str) -> Result<Vec<usize>> {
    let ks: Vec<Key> = keys(data, name)?;
    let mut distinct = ks.clone();
    distinct.sort_by(key_order);
    distinct.dedup_by(|a, b| key_order(a, b).is_eq());
    Ok(ks.iter().map(|k| distinct.binary_search_by(|d| key_order(d, k)).unwrap_or(0)).collect())
}

/// The discrete mappings ggplot2's `add_group()` interacts, in mapping order: `x` (text), `fill`,
/// `colour`. A continuous `fill` or `colour` is refused: the palette is not resolved.
fn discrete_vars<'a>(
    data: &DataFrame,
    aes: &'a Aes,
    x_text: Option<&'a str>,
) -> Result<Vec<&'a str>> {
    let mut vars: Vec<&str> = x_text.into_iter().collect();
    for name in [aes.fill.as_deref(), aes.color.as_deref()].into_iter().flatten() {
        if !is_text(data, name)? {
            return Err(refuse(format!(
                "`{name}` is a continuous colour or fill mapping: continuous colour scales are not supported"
            )));
        }
        if keys(data, name)?.iter().any(|k| k.0 == 2) {
            return Err(refuse(format!(
                "column `{name}` holds a missing value: R makes a level of it, which is not supported"
            )));
        }
        if !vars.contains(&name) {
            vars.push(name);
        }
    }
    Ok(vars)
}

/// ggplot2's `add_group()`: `id()` of the discrete mappings (first variable slowest), re-ranked
/// densely; a `group` column overrides them; `-1` when nothing is discrete.
fn group_ids(data: &DataFrame, aes: &Aes, x_text: Option<&str>) -> Result<Vec<i64>> {
    let vars = match &aes.group {
        Some(g) => vec![g.as_str()],
        None => discrete_vars(data, aes, x_text)?,
    };
    if vars.is_empty() {
        return Ok(vec![-1; data.nrow()]);
    }
    let per_var: Vec<Vec<usize>> = vars.iter().map(|v| ranks(data, v)).collect::<Result<_>>()?;
    let tuples: Vec<Vec<usize>> = (0..data.nrow())
        .map(|i| per_var.iter().map(|r| r.get(i).copied().unwrap_or(0)).collect())
        .collect();
    let mut distinct = tuples.clone();
    distinct.sort();
    distinct.dedup();
    Ok(tuples
        .iter()
        .map(|t| distinct.binary_search(t).map_or(-1, |p| i64::try_from(p + 1).unwrap_or(i64::MAX)))
        .collect())
}

/// `scales::zero_range()` of `(lo, hi)`.
fn zero_range(lo: f64, hi: f64) -> bool {
    if lo == hi {
        return true;
    }
    let m = lo.abs().min(hi.abs());
    m != 0.0 && ((lo - hi) / m).abs() < 1000.0 * f64::EPSILON
}

/// ggplot2's `resolution(x, zero = TRUE, discrete = TRUE)` of a numeric `x`: the smallest gap
/// between the distinct values, `0` counted as one of them.
fn resolution(xs: &[f64]) -> f64 {
    let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if xs.is_empty() || zero_range(lo, hi) {
        return 1.0;
    }
    let mut v: Vec<f64> = xs.iter().copied().chain(std::iter::once(0.0)).collect();
    v.sort_by(f64::total_cmp);
    v.dedup();
    let tol = f64::EPSILON.sqrt();
    let gaps = v.windows(2).filter_map(|w| match w {
        [a, b] => Some(b - a),
        _ => None,
    });
    gaps.filter(|d| *d > tol)
        .fold(None, |m: Option<f64>, d| Some(m.map_or(d, |m| m.min(d))))
        .unwrap_or(1.0)
}

/// `quantile(x, p, type = 7)` of an ascending, non-empty `sorted`.
fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let index = 1.0 + (sorted.len() as f64 - 1.0).max(0.0) * p;
    let (lo, hi) = (index.floor(), index.ceil());
    let at = |k: f64| sorted.get((k as usize).saturating_sub(1)).copied().unwrap_or(f64::NAN);
    let (below, above) = (at(lo), at(hi));
    if index > lo && above != below {
        let h = index - lo;
        (1.0 - h) * below + h * above
    } else {
        below
    }
}

fn extent(vs: impl IntoIterator<Item = f64>) -> (f64, f64) {
    vs.into_iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)))
}

/// One group's `x` and `y` values, in data order.
struct GroupData {
    group: i64,
    xs: Vec<f64>,
    ys: Vec<f64>,
}

/// `StatBoxplot$compute_group` and `GeomBoxplot$setup_data` for one group. `width` is the stat's
/// width parameter, used when the group has a single `x`.
fn summarise(g: &GroupData, coef: f64, width: f64) -> BoxRow {
    let mut sorted = g.ys.clone();
    sorted.sort_by(f64::total_cmp);
    let (lower, middle, upper) =
        (quantile7(&sorted, 0.25), quantile7(&sorted, 0.5), quantile7(&sorted, 0.75));
    let iqr = upper - lower;
    let (fence_lo, fence_hi) = (lower - coef * iqr, upper + coef * iqr);
    let (outliers, inside): (Vec<f64>, Vec<f64>) =
        g.ys.iter().partition(|y| **y < fence_lo || **y > fence_hi);
    let (ymin, ymax) = if outliers.is_empty() {
        extent(sorted.iter().copied())
    } else {
        extent(inside.iter().copied().chain([lower, middle, upper]))
    };
    let (out_lo, out_hi) = extent(outliers.iter().copied());
    let (x_lo, x_hi) = extent(g.xs.iter().copied());
    let n = g.ys.len() as f64;
    let w = if x_lo == x_hi { width } else { (x_hi - x_lo) * SPAN_WIDTH };
    let x = (x_lo + x_hi) / 2.0;
    BoxRow {
        group: g.group,
        x,
        xmin: x - w / 2.0,
        xmax: x + w / 2.0,
        ymin,
        lower,
        middle,
        upper,
        ymax,
        notchlower: middle - NOTCH * iqr / n.sqrt(),
        notchupper: middle + NOTCH * iqr / n.sqrt(),
        ymin_final: out_lo.min(ymin),
        ymax_final: out_hi.max(ymax),
        relvarwidth: n.sqrt(),
        outliers,
    }
}

/// `find_x_overlaps()` over rows sorted by `(x, group)`: the cluster id of each row. A row starts
/// a new cluster when it begins past every earlier box's right edge (or exactly on it, for a box
/// that has width).
fn cluster_ids(rows: &[BoxRow]) -> Vec<usize> {
    let mut ids = Vec::with_capacity(rows.len());
    let mut id = 0_usize;
    let mut end = rows.first().map_or(f64::NEG_INFINITY, |r| r.xmax);
    for (i, r) in rows.iter().enumerate() {
        if i > 0 && (r.xmin > end || (r.xmin == end && r.xmax != r.xmin)) {
            id += 1;
        }
        end = end.max(r.xmax);
        ids.push(id);
    }
    ids
}

/// `position_dodge2(padding = 0.1)` (`pos_dodge2()`): rows sorted by `(x, group)`; each cluster of
/// overlapping boxes is split into equal slots centred on the cluster's span; when any cluster
/// holds several boxes every box is padded to 90% of its slot.
fn dodge2(mut rows: Vec<BoxRow>) -> Vec<BoxRow> {
    rows.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.group.cmp(&b.group)));
    let ids = cluster_ids(&rows);
    let clusters = ids.last().map_or(0, |l| l + 1);
    // (min xmin, max xmax, box count, sum of slot widths) per cluster
    let mut span = vec![(f64::INFINITY, f64::NEG_INFINITY, 0_usize, 0.0_f64); clusters];
    for (r, id) in rows.iter().zip(&ids) {
        if let Some(s) = span.get_mut(*id) {
            s.0 = s.0.min(r.xmin);
            s.1 = s.1.max(r.xmax);
            s.2 += 1;
        }
    }
    for (r, id) in rows.iter().zip(&ids) {
        if let Some(s) = span.get_mut(*id) {
            s.3 += (r.xmax - r.xmin) / s.2 as f64;
        }
    }
    let mut cursor: Vec<f64> = span.iter().map(|s| (s.0 + s.1) / 2.0 - s.3 / 2.0).collect();
    let padded = span.iter().any(|s| s.2 > 1);
    for (r, id) in rows.iter_mut().zip(&ids) {
        let (Some(s), Some(at)) = (span.get(*id), cursor.get_mut(*id)) else {
            continue;
        };
        let slot = (r.xmax - r.xmin) / s.2 as f64;
        (r.xmin, r.xmax) = (*at, *at + slot);
        *at = r.xmax;
        r.x = (r.xmin + r.xmax) / 2.0;
        if padded {
            let pad = slot * (1.0 - DODGE2_PADDING);
            (r.xmin, r.xmax) = (r.x - pad / 2.0, r.x + pad / 2.0);
        }
    }
    rows
}

/// `StatBoxplot$setup_params`: the `width` parameter, or `0.75 * resolution(x)`.
fn default_width(spec: &DistSpec, x: &XInput) -> f64 {
    if let Some(w) = spec.width {
        return w;
    }
    let finite: Vec<f64> = x.values.iter().copied().filter(|v| v.is_finite()).collect();
    DEFAULT_WIDTH * if x.resolution_one { 1.0 } else { resolution(&finite) }
}

fn check_params(spec: &DistSpec, aes: &Aes) -> Result<()> {
    if aes.y.is_none() {
        return Err(refuse(
            "geom_boxplot needs a `y` mapping: a boxplot of `x` alone is the flipped orientation, \
             which is not supported",
        ));
    }
    if let Some(w) = spec.width {
        if !(w.is_finite() && w > 0.0) {
            return Err(refuse(format!("boxplot `width` must be positive and finite, got {w}")));
        }
    }
    if !spec.coef.is_finite() {
        return Err(refuse(format!("boxplot `coef` must be finite, got {}", spec.coef)));
    }
    Ok(())
}

/// The rows ggplot2 keeps, grouped, in group order: a missing `y` or `x` is dropped, and a group
/// that loses every row vanishes (group ids are not renumbered). No surviving row at all is refused.
fn split_groups(
    x: &XInput,
    y: &[f64],
    group: &[i64],
    missing_group: &[bool],
) -> Result<Vec<GroupData>> {
    let mut by_group: BTreeMap<i64, GroupData> = BTreeMap::new();
    for (i, ((xv, yv), g)) in x.values.iter().zip(y).zip(group).enumerate() {
        if !(xv.is_finite() && yv.is_finite()) {
            continue;
        }
        if missing_group.get(i).copied().unwrap_or(false) {
            return Err(refuse(
                "a row with a missing `group` value survives: R makes a group of the missing value, \
                 which is not supported",
            ));
        }
        let entry =
            by_group.entry(*g).or_insert_with(|| GroupData { group: *g, xs: vec![], ys: vec![] });
        entry.xs.push(*xv);
        entry.ys.push(*yv);
    }
    if by_group.is_empty() {
        return Err(refuse("no row has a finite `x` and `y`: R would draw an empty layer"));
    }
    Ok(by_group.into_values().collect())
}

/// Rows whose explicit `group` value is missing (all `false` without a `group` mapping).
fn missing_group_rows(data: &DataFrame, aes: &Aes) -> Result<Vec<bool>> {
    match aes.group.as_deref() {
        Some(g) => Ok(keys(data, g)?.iter().map(|k| k.0 == 2).collect()),
        None => Ok(vec![false; data.nrow()]),
    }
}

fn y_scale(rows: &[BoxRow]) -> PositionScale {
    let spread = rows.iter().flat_map(|r| [r.ymin_final, r.ymax_final, r.lower, r.middle, r.upper]);
    PositionScale::continuous(extent(spread))
}

/// The `x` scale retrained on the boxes. A text `x` keeps every level (trained before the stat)
/// and spans the union of the expanded levels and the unexpanded `xmin..xmax`.
fn x_scale(rows: &[BoxRow], levels: Option<Vec<String>>) -> PositionScale {
    let (lo, hi) = extent(rows.iter().flat_map(|r| [r.xmin, r.xmax]));
    let Some(levels) = levels else {
        return PositionScale::continuous((lo, hi));
    };
    let mut scale = resolve_discrete(levels, &DiscreteSpec::new());
    scale.continuous_range = (scale.continuous_range.0.min(lo), scale.continuous_range.1.max(hi));
    scale
}

/// Build one `geom_boxplot()` layer and its scales.
///
/// Out of scope, each refused with [`Error::Rendering`] instead of approximated: no `y` mapping
/// (flipped orientation), a text or infinite `y`, a missing value in a text `x`, `fill` or `colour`,
/// or a surviving row with a missing `group` (R adds a level or a group for it), a `width` that is not positive and finite, a `coef`
/// that is not finite, no row with a finite `x` and `y`, and a continuous `fill` or `colour`.
/// A mapped `fill`/`colour` only groups: no palette is resolved. Known differences from R, kept
/// as such: the order of two discrete mappings inside one `aes()` call is taken as `x`, `fill`,
/// `colour`; an integer-typed numeric `x` in R has resolution 1, a number here is always a double.
///
/// # Errors
///
/// [`Error::Rendering`] as above, [`Error::UnknownColumn`] for an absent column.
pub(super) fn layout(data: &DataFrame, aes: &Aes, spec: &DistSpec) -> Result<BuiltLayout> {
    check_params(spec, aes)?;
    let x = read_x(data, aes)?;
    let y = numeric(data, aes.y.as_deref().unwrap_or_default())?;
    let group = group_ids(data, aes, x.text_column.as_deref())?;
    let width = default_width(spec, &x);
    let missing = missing_group_rows(data, aes)?;
    let boxes: Vec<BoxRow> = split_groups(&x, &y, &group, &missing)?
        .iter()
        .map(|g| summarise(g, spec.coef, width))
        .collect();
    let boxes = dodge2(boxes);
    let (x_axis, y_axis) = (x_scale(&boxes, x.levels), y_scale(&boxes));
    let layer = BuiltLayer {
        x: boxes.iter().map(|b| b.x).collect(),
        y: boxes.iter().map(|b| b.middle).collect(),
        points: None,
        path: None,
        filled: None,
        boxes: Some(boxes),
    };
    Ok(BuiltLayout { layers: vec![layer], x: x_axis, y: y_axis })
}
