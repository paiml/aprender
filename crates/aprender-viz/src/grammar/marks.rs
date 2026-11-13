//! Filled marks and text: `geom_area`, `geom_ribbon`, `geom_tile`, `geom_rect`, `geom_text`
//! (PLOT-GG-001, paiml/aprender#6909).
//!
//! [`build`] resolves one layer the way `ggplot_build()` in ggplot2 4.0.3 does: the stat
//! (`stat_align` for `geom_area`), the geom's `setup_data` (sorting, tile bounds), and the position
//! (`position_stack` for `geom_area`). The result is the layer's data frame: positions, group ids,
//! style columns and, for text, the label strings as passthrough.
//!
//! Scope is layout only. Anything this module does not model returns a typed [`Error`] instead of
//! a plausible wrong answer: data-mapped colour/size/shape/alpha/fill (they split groups and train
//! further scales), discrete or infinite positions, negative `y` in `geom_area` (stacking then
//! splits on sign), numeric labels, an area whose groups share fewer than two x values.
//!
//! Companion layers. The x and y scales are shared, so a plot with one of these layers may hold
//! only these five geoms and `geom_point` with the identity stat; every other geom is an
//! [`Error::UnsupportedLayer`] and every other stat on a `geom_point` an [`Error::UnsupportedStat`]
//! (see [`require_modelled_companions`]). `ggplot2_marks_oracle::companions` proves the shared
//! scales of the mixes that do lay out against ggplot2.
//!
//! Every default (fill `grey20`, rect fill `grey35`, line widths, text size `11 / .pt`) is read
//! off ggplot2 4.0.3's `default_aes`, and `NA` is carried as `NaN`/`None`.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use super::aes::Aes;
use super::build::PT_PER_MM;
use super::data::{DataFrame, DataValue};
use super::geom::{Geom, GeomType};
use super::ggplot::Layer;
use super::stat::Stat;
use crate::color::Rgba;
use crate::error::{Error, Result};

/// Which of the five geoms a layer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkKind {
    /// `geom_area()`: `stat_align` + `position_stack`.
    Area,
    /// `geom_ribbon()`.
    Ribbon,
    /// `geom_tile()`.
    Tile,
    /// `geom_rect()`.
    Rect,
    /// `geom_text()`.
    Text,
}

impl MarkKind {
    /// The ggplot2 geom class name, as `ggplot_build()` reports it.
    #[must_use]
    pub const fn geom_class(self) -> &'static str {
        match self {
            MarkKind::Area => "GeomArea",
            MarkKind::Ribbon => "GeomRibbon",
            MarkKind::Tile => "GeomTile",
            MarkKind::Rect => "GeomRect",
            MarkKind::Text => "GeomText",
        }
    }

    fn of(geom: &GeomType) -> Option<Self> {
        match geom {
            GeomType::Area { .. } => Some(MarkKind::Area),
            GeomType::Ribbon => Some(MarkKind::Ribbon),
            GeomType::Tile => Some(MarkKind::Tile),
            GeomType::Rect => Some(MarkKind::Rect),
            GeomType::Text => Some(MarkKind::Text),
            _ => None,
        }
    }
}

/// `default_aes` of the five geoms in ggplot2 4.0.3.
pub mod mark_defaults {
    /// `fill = from_theme(fill %||% col_mix(ink, paper, 0.2))` = `#333333FF`: area, ribbon, tile.
    pub const FILL_GREY20: [u8; 4] = [51, 51, 51, 255];
    /// `GeomRect`: `col_mix(ink, paper, 0.35)` = `#595959FF`.
    pub const FILL_RECT: [u8; 4] = [89, 89, 89, 255];
    /// `GeomText`: `colour = ink` = black.
    pub const TEXT_COLOUR: [u8; 4] = [0, 0, 0, 255];
    /// `linewidth` of area, ribbon and rect (`borderwidth`).
    pub const LINEWIDTH: f64 = 0.5;
    /// `linewidth` of `GeomTile` (`borderwidth * 0.4`).
    pub const TILE_LINEWIDTH: f64 = 0.2;
    /// `lineheight` of `GeomText`.
    pub const LINEHEIGHT: f64 = 1.2;
    /// `fontsize` of `GeomText` in points (`from_theme(fontsize)`, 11); `size = fontsize / .pt`.
    pub const FONTSIZE_PT: f64 = 11.0;
}

/// One layer's built data frame. `NaN` is `NA`.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltMarks {
    /// Which geom produced it.
    pub kind: MarkKind,
    /// The `group` column: the 1-based rank of the sorted distinct group value, `-1` if unmapped.
    pub group: Vec<i32>,
    /// Per-row numeric columns ggplot2 emits (`x`, `y`, `xmin`, ..., `align_padding` as 0/1).
    pub num: BTreeMap<&'static str, Vec<f64>>,
    /// Per-row text columns (`label`, `family`); `None` is `NA`.
    pub texts: BTreeMap<&'static str, Vec<Option<String>>>,
    /// Constant numeric style columns (`linewidth`, `alpha`, `size`, ...), one value for every row.
    pub scalars: BTreeMap<&'static str, f64>,
    /// Constant colour columns as RGBA; `None` is `NA`.
    pub colours: BTreeMap<&'static str, Option<[u8; 4]>>,
}

impl BuiltMarks {
    /// Rows in the layer.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.group.len()
    }

    /// The finite range of the columns the x scale trains on (`x`, `xmin`, `xmax`).
    #[must_use]
    pub fn x_extent(&self) -> Option<(f64, f64)> {
        self.extent(&["x", "xmin", "xmax"])
    }

    /// The finite range of the columns the y scale trains on (`y`, `ymin`, `ymax`).
    #[must_use]
    pub fn y_extent(&self) -> Option<(f64, f64)> {
        self.extent(&["y", "ymin", "ymax"])
    }

    fn extent(&self, cols: &[&str]) -> Option<(f64, f64)> {
        let mut acc: Option<(f64, f64)> = None;
        for v in cols.iter().filter_map(|c| self.num.get(c)).flatten().filter(|v| v.is_finite()) {
            acc = Some(acc.map_or((*v, *v), |(lo, hi)| (lo.min(*v), hi.max(*v))));
        }
        acc
    }
}

/// The constant style of a layer, from `geom_*(colour =, fill =, alpha =, size =)`.
struct Style {
    colour: Option<[u8; 4]>,
    fill: Option<[u8; 4]>,
    alpha: f64,
    size: Option<f64>,
}

fn rgba(c: Rgba) -> [u8; 4] {
    [c.r, c.g, c.b, c.a]
}

impl Style {
    fn of(aes: &Aes) -> Self {
        Self {
            colour: aes.color_value.map(rgba),
            fill: aes.fill_value.map(rgba),
            alpha: aes.alpha_value.map_or(f64::NAN, f64::from),
            size: aes.size_value.map(f64::from),
        }
    }
}

/// A built layer's positional core, before the style columns are attached.
struct Core {
    group: Vec<i32>,
    num: BTreeMap<&'static str, Vec<f64>>,
}

/// Resolve a layer if its geom is one of the five; `Ok(None)` for every other geom.
///
/// # Errors
///
/// [`Error::EmptyData`] when no row survives; [`Error::UnknownColumn`] for a mapped column the
/// frame lacks; [`Error::Rendering`] / [`Error::ScaleDomain`] for an input this module does not
/// model (see the module docs).
pub(super) fn build(geom: &Geom, data: &DataFrame, aes: &Aes) -> Result<Option<BuiltMarks>> {
    let Some(kind) = MarkKind::of(&geom.geom_type) else { return Ok(None) };
    refuse_layer_parameters(kind, geom)?;
    refuse_mapped_style(kind, aes)?;
    refuse_unmodelled_aesthetics(kind, aes)?;
    if data.nrow() == 0 {
        return Err(Error::EmptyData);
    }
    let group = group_ids(data, aes)?;
    let style = Style::of(aes);
    let (core, texts) = match kind {
        MarkKind::Area => (area_core(&read_xy(kind, data, aes)?, &group)?, BTreeMap::new()),
        MarkKind::Ribbon => (ribbon_core(kind, data, aes, &group)?, BTreeMap::new()),
        MarkKind::Tile => (tile_core(&read_xy(kind, data, aes)?, &group), BTreeMap::new()),
        MarkKind::Rect => (rect_core(kind, data, aes, &group)?, BTreeMap::new()),
        MarkKind::Text => text_parts(kind, data, aes, &group)?,
    };
    Ok(Some(finish(kind, core, texts, &style)))
}

/// Is any layer one of the five geoms?
pub(super) fn any_marks(layers: &[Layer]) -> bool {
    layers.iter().any(|l| MarkKind::of(&l.geom.geom_type).is_some())
}

/// Refuse, while one of the five geoms is present, every layer `layout()` does not resolve beside
/// it. The scales are shared, so a companion whose ggplot2 stat, setup or position widens them
/// (measured on ggplot2 4.0.3 beside `geom_area`: `geom_bar` takes y to 0 and x to the bar edges,
/// `geom_histogram` x to the bin edges, `geom_point(stat = "count")` y to 0) would make `layout()`
/// answer with the wrong limits. Only the five geoms and `geom_point` with the identity stat may
/// share the plot; a `geom_point` beside them must map `x` and `y` to complete numeric columns of
/// the frame, with no infinite value (ggplot2 squishes those to the panel), and must not inherit
/// `xmin`/`xmax`/`ymin`/`ymax` (ggplot2 keeps them in the point's data and trains the scales on
/// them). Path, step and reference layers are refused before this by `path`.
///
/// There is no position-adjustment parameter on a layer here (every layer is
/// `position = "identity"`, and `geom_area`'s `position_stack` is applied by [`build`]), so there
/// is no position to refuse.
///
/// # Errors
///
/// [`Error::UnsupportedLayer`] for a companion geom, [`Error::UnsupportedStat`] for a stat on
/// `geom_point`, [`Error::UnknownColumn`] for a missing column, [`Error::Rendering`] for a missing
/// mapping, a text or short column, or an inherited `xmin`/`xmax`/`ymin`/`ymax`, and
/// [`Error::ScaleDomain`] for an infinite position.
pub(super) fn require_modelled_companions(
    layers: &[Layer],
    plot_aes: &Aes,
    plot_data: &DataFrame,
) -> Result<()> {
    if !any_marks(layers) {
        return Ok(());
    }
    for l in layers {
        let g = &l.geom.geom_type;
        if MarkKind::of(g).is_some() {
            continue;
        }
        if !matches!(g, GeomType::Point { .. }) {
            return Err(Error::UnsupportedLayer(unsupported_companion(g)));
        }
        if let Some(msg) = l.geom.stat.as_ref().and_then(unsupported_point_stat) {
            return Err(Error::UnsupportedStat(msg));
        }
        let data = l.data.as_ref().unwrap_or(plot_data);
        require_point_companion(data, &plot_aes.merge(&l.aes))?;
    }
    Ok(())
}

/// The refusal naming a companion geom that may not share a plot with the five.
fn unsupported_companion(g: &GeomType) -> &'static str {
    match g {
        GeomType::Bar { .. } => {
            "geom_bar beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_count and the bar widths widen the shared scales)"
        }
        GeomType::Histogram { .. } => {
            "geom_histogram beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_bin widens the shared scales)"
        }
        GeomType::Boxplot => {
            "geom_boxplot beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_boxplot and the box widths widen the shared scales)"
        }
        GeomType::Violin => {
            "geom_violin beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_ydensity and the violin widths widen the shared scales)"
        }
        GeomType::Smooth { .. } => {
            "geom_smooth beside area/ribbon/tile/rect/text: layout() does not resolve it there \
             (stat_smooth and its ribbon widen the shared scales)"
        }
        _ => {
            "a line, path, step or reference layer beside area/ribbon/tile/rect/text: layout() \
             does not resolve it there"
        }
    }
}

/// The refusal naming a stat on a `geom_point` beside the five; `None` for the identity stat.
fn unsupported_point_stat(s: &Stat) -> Option<&'static str> {
    let msg = match s {
        Stat::Identity => return None,
        Stat::Count => "stat_count on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat",
        Stat::Bin { .. } => "stat_bin on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat",
        Stat::Density => {
            "stat_density on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Boxplot => {
            "stat_boxplot on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Smooth => {
            "stat_smooth on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Bin2d { .. } => {
            "stat_bin_2d on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Sum => {
            "stat_summary(fun = sum) on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
        Stat::Mean => {
            "stat_summary(fun = mean) on geom_point beside area/ribbon/tile/rect/text: layout() computes no stat"
        }
    };
    Some(msg)
}

/// A `geom_point` beside the five trains the shared scales on its raw `x` and `y`: both must be
/// mapped, present, complete-length numeric columns without an infinite value, and no bounds
/// aesthetic may reach it.
fn require_point_companion(data: &DataFrame, aes: &Aes) -> Result<()> {
    let bounds =
        [("xmin", &aes.xmin), ("xmax", &aes.xmax), ("ymin", &aes.ymin), ("ymax", &aes.ymax)];
    if let Some((name, _)) = bounds.iter().find(|(_, v)| v.is_some()) {
        return Err(Error::Rendering(format!(
            "geom_point beside area/ribbon/tile/rect/text: a mapped {name} is kept by ggplot2 in \
             the point's data and trains a scale; not modelled"
        )));
    }
    for (axis, name) in [("x", &aes.x), ("y", &aes.y)] {
        let Some(name) = name.as_deref() else {
            return Err(Error::Rendering(format!(
                "geom_point beside area/ribbon/tile/rect/text needs a mapped {axis}"
            )));
        };
        require_point_column(data, name)?;
    }
    Ok(())
}

/// One position column of a `geom_point` beside the five: present, as long as the frame, and
/// numeric without an infinite value (an NA cell is kept: the scale trains on the other rows).
fn require_point_column(data: &DataFrame, name: &str) -> Result<()> {
    let cells = data.get(name).ok_or_else(|| Error::UnknownColumn(name.to_string()))?;
    if cells.len() < data.nrow() {
        return Err(Error::Rendering(format!(
            "geom_point beside area/ribbon/tile/rect/text: column {name} is shorter than the frame"
        )));
    }
    if cells.iter().any(|v| matches!(v, DataValue::Text(_))) {
        return Err(Error::Rendering(format!(
            "geom_point beside area/ribbon/tile/rect/text: column {name} is text, which needs a \
             discrete scale; not modelled"
        )));
    }
    if cells.iter().any(|v| matches!(v, DataValue::Number(n) if n.is_infinite())) {
        return Err(Error::ScaleDomain(format!(
            "geom_point beside area/ribbon/tile/rect/text: column {name}: infinite position is \
             not modelled"
        )));
    }
    Ok(())
}

/// Layer parameters that change ggplot2's numbers and that this module does not model. Measured on
/// ggplot2 4.0.3: `geom_area(stat = "identity")` drops `stat_align`'s resampling and padding rows,
/// and `geom_area(alpha = 0.5)` writes `alpha = 0.5`. Neither may be read as the default layer.
///
/// There is no position-adjustment parameter in the API (`position_stack` for `geom_area` is the
/// only one applied, and it is the default), so there is nothing to refuse there. The `stat` of
/// ribbon, tile, rect and text is `identity` by default, so an explicit identity is that default.
fn refuse_layer_parameters(kind: MarkKind, geom: &Geom) -> Result<()> {
    let class = kind.geom_class();
    match (&geom.stat, kind) {
        (None, _) => {}
        (
            Some(Stat::Identity),
            MarkKind::Ribbon | MarkKind::Tile | MarkKind::Rect | MarkKind::Text,
        ) => {}
        (Some(stat), _) => {
            return Err(Error::Rendering(format!(
                "{class}: stat {stat:?} is not modelled (only the geom's default stat is)"
            )))
        }
    }
    if let GeomType::Area { alpha } = geom.geom_type {
        // `Geom::area()` carries 0.3 as this crate's drawing default, which ggplot2 does not have
        // (its alpha is NA); the built alpha column comes only from `Aes::alpha_value`.
        if alpha.to_bits() != DRAWING_ALPHA_AREA.to_bits() {
            return Err(Error::Rendering(format!(
                "{class}: Geom::alpha({alpha}) is not read by layout; set Aes::alpha_value instead"
            )));
        }
    }
    Ok(())
}

/// The `alpha` `Geom::area()` starts with.
const DRAWING_ALPHA_AREA: f32 = 0.3;

/// Aesthetics ggplot2 keeps in the layer's data and trains scales on, or turns into another column,
/// where this module would drop them. Measured on ggplot2 4.0.3:
/// - `geom_rect(aes(x, y))` keeps `x` and `y` and trains both scales on them;
/// - `geom_text(aes(ymin = ))` keeps the column and trains the y scale on it (xmin, xmax, ymax
///   likewise);
/// - `size` on area, ribbon, tile or rect is the deprecated spelling of `linewidth`.
///
/// Not refused because ggplot2 overwrites them, so the default result is right: `ymin`/`ymax` on
/// area and tile, `xmin`/`xmax` on tile, `y` on ribbon. Tests `overwritten_aesthetics_equal_default` pin each.
fn refuse_unmodelled_aesthetics(kind: MarkKind, aes: &Aes) -> Result<()> {
    let class = kind.geom_class();
    let extra: &[(&str, bool)] = match kind {
        MarkKind::Rect => &[("x", aes.x.is_some()), ("y", aes.y.is_some())],
        MarkKind::Text => &[
            ("xmin", aes.xmin.is_some()),
            ("xmax", aes.xmax.is_some()),
            ("ymin", aes.ymin.is_some()),
            ("ymax", aes.ymax.is_some()),
        ],
        _ => &[],
    };
    if let Some((name, _)) = extra.iter().find(|(_, set)| *set) {
        return Err(Error::Rendering(format!(
            "{class}: a mapped {name} aesthetic is kept by ggplot2 and trains a scale; not modelled"
        )));
    }
    if kind != MarkKind::Text && aes.size_value.is_some() {
        return Err(Error::Rendering(format!(
            "{class}: a fixed size is ggplot2's deprecated linewidth here; not modelled"
        )));
    }
    Ok(())
}

/// Data-mapped style aesthetics split groups and train more scales; not modelled.
fn refuse_mapped_style(kind: MarkKind, aes: &Aes) -> Result<()> {
    let mapped = [
        ("colour", &aes.color),
        ("size", &aes.size),
        ("shape", &aes.shape),
        ("alpha", &aes.alpha),
        ("fill", &aes.fill),
    ];
    match mapped.iter().find(|(_, v)| v.is_some()) {
        Some((name, _)) => Err(Error::Rendering(format!(
            "{}: a data-mapped {name} aesthetic is not modelled (it adds a scale and splits groups)",
            kind.geom_class()
        ))),
        None => Ok(()),
    }
}

// --- reading columns ----------------------------------------------------------------------------

/// One position column as f64 with `NA` as `NaN`. Text positions (a discrete scale) and infinite
/// values (ggplot2 squishes them to the panel) are refused. `-0.0` becomes `0.0`, as R compares.
fn position_column(data: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let cells = data.get(name).ok_or_else(|| Error::UnknownColumn(name.to_string()))?;
    (0..data.nrow())
        .map(|i| match cells.get(i) {
            Some(DataValue::Number(v)) if v.is_infinite() => {
                Err(Error::ScaleDomain(format!("column {name}: infinite position is not modelled")))
            }
            Some(DataValue::Number(v)) => Ok(f64::from(*v) + 0.0),
            Some(DataValue::Text(_)) => Err(Error::Rendering(format!(
                "column {name}: a text position needs a discrete scale, which is not modelled"
            ))),
            Some(DataValue::Null) => Ok(f64::NAN),
            None => Err(ragged(name, cells.len(), data.nrow())),
        })
        .collect()
}

/// A column shorter than the frame: R cannot build such a data frame, so there is no ggplot2
/// answer to match, and padding it with `NA` would invent one.
fn ragged(name: &str, len: usize, nrow: usize) -> Error {
    Error::Rendering(format!(
        "column {name} has {len} cells but the frame has {nrow} rows: ragged columns are not modelled"
    ))
}

fn required<'a>(kind: MarkKind, slot: &'a Option<String>, role: &str) -> Result<&'a str> {
    slot.as_deref().ok_or_else(|| {
        Error::Rendering(format!("{} needs the {role} aesthetic", kind.geom_class()))
    })
}

/// The `x` and `y` columns of a layer.
struct Xy {
    x: Vec<f64>,
    y: Vec<f64>,
}

fn read_xy(kind: MarkKind, data: &DataFrame, aes: &Aes) -> Result<Xy> {
    let x = position_column(data, required(kind, &aes.x, "x")?)?;
    let y = position_column(data, required(kind, &aes.y, "y")?)?;
    Ok(Xy { x, y })
}

/// ggplot2's group id: `id()` of the group column, i.e. the rank of the sorted distinct value
/// (numbers numerically, text bytewise: C collation), `-1` when no group is mapped.
fn group_ids(data: &DataFrame, aes: &Aes) -> Result<Vec<i32>> {
    let n = data.nrow();
    let Some(col) = &aes.group else { return Ok(vec![-1; n]) };
    let cells = data.get(col).ok_or_else(|| Error::UnknownColumn(col.clone()))?;
    let keys: Vec<GroupKey> = (0..n)
        .map(|i| match cells.get(i) {
            Some(DataValue::Number(v)) if v.is_finite() => Ok(GroupKey::Num(f64::from(*v) + 0.0)),
            Some(DataValue::Text(s)) => Ok(GroupKey::Text(s.clone())),
            None => Err(ragged(col, cells.len(), n)),
            _ => Err(Error::Rendering(format!(
                "group column {col}: a missing group is not modelled"
            ))),
        })
        .collect::<Result<_>>()?;
    let mut distinct = keys.clone();
    distinct.sort_by(GroupKey::order);
    distinct.dedup();
    Ok(keys
        .iter()
        .map(|k| {
            let rank = distinct.iter().position(|d| d == k).unwrap_or(0);
            i32::try_from(rank + 1).unwrap_or(i32::MAX)
        })
        .collect())
}

#[derive(Debug, Clone, PartialEq)]
enum GroupKey {
    Num(f64),
    Text(String),
}

impl GroupKey {
    fn order(a: &GroupKey, b: &GroupKey) -> Ordering {
        match (a, b) {
            (GroupKey::Num(x), GroupKey::Num(y)) => x.total_cmp(y),
            (GroupKey::Text(x), GroupKey::Text(y)) => x.as_bytes().cmp(y.as_bytes()),
            (GroupKey::Num(_), GroupKey::Text(_)) => Ordering::Less,
            (GroupKey::Text(_), GroupKey::Num(_)) => Ordering::Greater,
        }
    }
}

// --- geom_area ----------------------------------------------------------------------------------

/// One row on its way through `stat_align` and `position_stack`.
#[derive(Clone, Copy)]
struct AreaRow {
    group: i32,
    x: f64,
    y: f64,
    /// `align_padding`, present only when `stat_align` ran.
    pad: Option<bool>,
}

fn area_core(xy: &Xy, group: &[i32]) -> Result<Core> {
    let rows: Vec<AreaRow> = (0..xy.x.len())
        .filter(|&i| !xy.x[i].is_nan() && !xy.y[i].is_nan())
        .map(|i| AreaRow { group: group[i], x: xy.x[i], y: xy.y[i], pad: None })
        .collect();
    let first = rows.first().ok_or(Error::EmptyData)?;
    if rows.iter().any(|r| r.y < 0.0) {
        return Err(Error::Rendering(
            "GeomArea: negative y is not modelled (position_stack splits on sign)".to_string(),
        ));
    }
    let single = rows.iter().all(|r| r.group == first.group);
    let ordered = if single { sort_by_x(rows) } else { align_groups(&rows)? };
    Ok(stack(ordered))
}

/// `order(PANEL, group, x)`, stable: `GeomArea$setup_data`. One panel, one group.
fn sort_by_x(mut rows: Vec<AreaRow>) -> Vec<AreaRow> {
    rows.sort_by(|a, b| a.x.total_cmp(&b.x));
    rows
}

/// `StatAlign$compute_panel` for several groups: every group is resampled onto one shared grid.
fn align_groups(rows: &[AreaRow]) -> Result<Vec<AreaRow>> {
    let mut ul: Vec<f64> = rows.iter().map(|r| r.x).collect();
    ul.sort_by(f64::total_cmp);
    ul.dedup();
    if ul.len() < 2 {
        return Err(Error::Rendering(
            "GeomArea: several groups need at least two distinct x values to align".to_string(),
        ));
    }
    let span = ul[ul.len() - 1] - ul[0];
    let min_gap = ul.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
    let adjust = (span * 0.001).min(min_gap / 3.0);
    let mut loc: Vec<f64> = ul.iter().flat_map(|&u| [u - adjust, u, u + adjust]).collect();
    loc.sort_by(f64::total_cmp);
    loc.dedup();

    let mut ids: Vec<i32> = rows.iter().map(|r| r.group).collect();
    ids.sort_unstable();
    ids.dedup();
    let out: Vec<AreaRow> = ids
        .iter()
        .flat_map(|&g| {
            let pts: Vec<(f64, f64)> =
                rows.iter().filter(|r| r.group == g).map(|r| (r.x, r.y)).collect();
            align_group(g, &pts, &loc, adjust)
        })
        .collect();
    if out.is_empty() {
        return Err(Error::EmptyData);
    }
    Ok(sort_by_group_x(out))
}

/// `order(PANEL, group, x)` over the aligned rows (groups are already ascending).
fn sort_by_group_x(mut rows: Vec<AreaRow>) -> Vec<AreaRow> {
    rows.sort_by(|a, b| a.group.cmp(&b.group).then(a.x.total_cmp(&b.x)));
    rows
}

/// `StatAlign$compute_group` for one group. A group whose x are all identical yields nothing.
fn align_group(group: i32, pts: &[(f64, f64)], loc: &[f64], adjust: f64) -> Vec<AreaRow> {
    if pts.iter().all(|p| p.0 == pts[0].0) {
        return Vec::new();
    }
    let knots = regularize(&shift_duplicates(pts, adjust));
    let (xs, ys): (Vec<f64>, Vec<f64>) =
        loc.iter().filter_map(|&u| approx(&knots, u).map(|v| (u, v))).unzip();
    let (Some(&lo), Some(&hi)) = (xs.first(), xs.last()) else { return Vec::new() };
    let mut out = vec![AreaRow { group, x: lo - adjust, y: 0.0, pad: Some(true) }];
    out.extend(xs.iter().zip(&ys).map(|(&x, &y)| AreaRow { group, x, y, pad: Some(false) }));
    out.push(AreaRow { group, x: hi + adjust, y: 0.0, pad: Some(true) });
    out
}

/// Where a group repeats an x, keep its first and last row, the first moved left by `adjust`.
fn shift_duplicates(pts: &[(f64, f64)], adjust: f64) -> Vec<(f64, f64)> {
    let mut order: Vec<usize> = (0..pts.len()).collect();
    order.sort_by(|&a, &b| pts[a].0.total_cmp(&pts[b].0));
    let mut out = Vec::with_capacity(pts.len());
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && pts[order[j + 1]].0 == pts[order[i]].0 {
            j += 1;
        }
        let (first, last) = (pts[order[i]], pts[order[j]]);
        if j > i {
            out.push((first.0 - adjust, first.1));
        }
        out.push(last);
        i = j + 1;
    }
    out
}

/// `approxfun`'s regularisation: sort by x. R also averages the y of equal x (`ties = mean`), but
/// `shift_duplicates` has already left each x once: a repeated x keeps two rows, the first moved
/// left by `adjust <= gap / 3`, so no two knots share an x and the averaging has nothing to do.
fn regularize(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut sorted = pts.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    sorted
}

/// R's `approx1()` with `rule = 1`: linear interpolation, `None` (NA) outside the knots.
fn approx(knots: &[(f64, f64)], v: f64) -> Option<f64> {
    let (mut i, mut j) = (0, knots.len().checked_sub(1)?);
    if v < knots[i].0 || v > knots[j].0 {
        return None;
    }
    while i + 1 < j {
        let ij = (i + j) / 2;
        if v < knots[ij].0 {
            j = ij;
        } else {
            i = ij;
        }
    }
    if v == knots[j].0 {
        return Some(knots[j].1);
    }
    if v == knots[i].0 {
        return Some(knots[i].1);
    }
    Some(knots[i].1 + (knots[j].1 - knots[i].1) * ((v - knots[i].0) / (knots[j].0 - knots[i].0)))
}

/// `GeomArea$setup_data` (`ymin = 0, ymax = y`) then `position_stack`. Stacking runs only when an
/// x repeats; within one x the rows pile up in `order(xmin, -group)`, the highest group lowest.
fn stack(rows: Vec<AreaRow>) -> Core {
    let n = rows.len();
    let mut ymin = vec![0.0; n];
    let mut ymax: Vec<f64> = rows.iter().map(|r| r.y).collect();
    let x: Vec<f64> = rows.iter().map(|r| r.x).collect();
    let mut sorted_x = x.clone();
    sorted_x.sort_by(f64::total_cmp);
    let stacked = sorted_x.windows(2).any(|w| w[0] == w[1]);
    if stacked {
        let mut ord: Vec<usize> = (0..n).collect();
        ord.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(rows[b].group.cmp(&rows[a].group)));
        let (mut run, mut at) = (0.0_f64, f64::NAN);
        for &i in &ord {
            if x[i] != at {
                (run, at) = (0.0, x[i]);
            }
            let below = run;
            run += rows[i].y;
            (ymin[i], ymax[i]) = (below.min(run), below.max(run));
        }
    }
    let mut num: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    num.insert("x", x.clone());
    num.insert("y", if stacked { ymax.clone() } else { rows.iter().map(|r| r.y).collect() });
    num.insert("ymin", ymin);
    num.insert("ymax", ymax);
    if stacked {
        num.insert("xmin", x.clone());
        num.insert("xmax", x);
    }
    if rows.iter().any(|r| r.pad.is_some()) {
        let pad = rows.iter().map(|r| f64::from(u8::from(r.pad == Some(true)))).collect();
        num.insert("align_padding", pad);
    }
    Core { group: rows.iter().map(|r| r.group).collect(), num }
}

// --- geom_ribbon --------------------------------------------------------------------------------

fn ribbon_core(kind: MarkKind, data: &DataFrame, aes: &Aes, group: &[i32]) -> Result<Core> {
    if aes.xmin.is_some() || aes.xmax.is_some() {
        return Err(Error::Rendering(
            "GeomRibbon: the flipped (xmin/xmax) form is not modelled".to_string(),
        ));
    }
    let x = position_column(data, required(kind, &aes.x, "x")?)?;
    let ymin = position_column(data, required(kind, &aes.ymin, "ymin")?)?;
    let ymax = position_column(data, required(kind, &aes.ymax, "ymax")?)?;
    // order(PANEL, group, x), stable, NA x last.
    let mut ord: Vec<usize> = (0..x.len()).collect();
    ord.sort_by(|&a, &b| group[a].cmp(&group[b]).then_with(|| na_last(x[a], x[b])));
    let pick = |v: &[f64]| ord.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let mut num: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    num.insert("x", pick(&x));
    num.insert("ymin", pick(&ymin));
    num.insert("ymax", pick(&ymax));
    num.insert("y", pick(&ymin));
    Ok(Core { group: ord.iter().map(|&i| group[i]).collect(), num })
}

fn na_last(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}

// --- geom_tile ----------------------------------------------------------------------------------

/// `scales::zero_range()` with its default tolerance, on a finite range.
fn zero_range(lo: f64, hi: f64) -> bool {
    if lo == hi {
        return true;
    }
    let m = lo.abs().min(hi.abs());
    m != 0.0 && ((lo - hi) / m).abs() < 1000.0 * f64::EPSILON
}

/// `resolution(x, zero = FALSE, discrete = TRUE)`: the smallest gap between distinct values,
/// `1` for a zero-width range.
fn resolution(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(f64::total_cmp);
    v.dedup();
    let (Some(&lo), Some(&hi)) = (v.first(), v.last()) else { return 1.0 };
    if zero_range(lo, hi) {
        return 1.0;
    }
    let tol = f64::EPSILON.sqrt();
    v.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > tol).fold(f64::INFINITY, f64::min)
}

fn tile_core(xy: &Xy, group: &[i32]) -> Core {
    let (w, h) = (resolution(&xy.x), resolution(&xy.y));
    let w = if w.is_finite() { w } else { 1.0 };
    let h = if h.is_finite() { h } else { 1.0 };
    let mut num: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    num.insert("x", xy.x.clone());
    num.insert("y", xy.y.clone());
    num.insert("xmin", xy.x.iter().map(|x| x - w / 2.0).collect());
    num.insert("xmax", xy.x.iter().map(|x| x + w / 2.0).collect());
    num.insert("ymin", xy.y.iter().map(|y| y - h / 2.0).collect());
    num.insert("ymax", xy.y.iter().map(|y| y + h / 2.0).collect());
    Core { group: group.to_vec(), num }
}

// --- geom_rect ----------------------------------------------------------------------------------

fn rect_core(kind: MarkKind, data: &DataFrame, aes: &Aes, group: &[i32]) -> Result<Core> {
    let mut num: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    for (name, slot) in
        [("xmin", &aes.xmin), ("xmax", &aes.xmax), ("ymin", &aes.ymin), ("ymax", &aes.ymax)]
    {
        num.insert(name, position_column(data, required(kind, slot, name)?)?);
    }
    Ok(Core { group: group.to_vec(), num })
}

// --- geom_text ----------------------------------------------------------------------------------

type TextColumns = BTreeMap<&'static str, Vec<Option<String>>>;

fn text_parts(
    kind: MarkKind,
    data: &DataFrame,
    aes: &Aes,
    group: &[i32],
) -> Result<(Core, TextColumns)> {
    let xy = read_xy(kind, data, aes)?;
    let name = required(kind, &aes.label, "label")?;
    let cells = data.get(name).ok_or_else(|| Error::UnknownColumn(name.to_string()))?;
    let label: Vec<Option<String>> = (0..data.nrow())
        .map(|i| match cells.get(i) {
            Some(DataValue::Text(s)) => Ok(Some(s.clone())),
            Some(DataValue::Null) => Ok(None),
            None => Err(ragged(name, cells.len(), data.nrow())),
            Some(DataValue::Number(_)) => Err(Error::Rendering(format!(
                "label column {name}: a numeric label is not modelled (R keeps it numeric)"
            ))),
        })
        .collect::<Result<_>>()?;
    let mut num: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    num.insert("x", xy.x);
    num.insert("y", xy.y);
    let mut texts: TextColumns = BTreeMap::new();
    texts.insert("family", vec![Some(String::new()); label.len()]);
    texts.insert("label", label);
    Ok((Core { group: group.to_vec(), num }, texts))
}

// --- style --------------------------------------------------------------------------------------

fn finish(kind: MarkKind, core: Core, texts: TextColumns, style: &Style) -> BuiltMarks {
    let mut scalars: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut colours: BTreeMap<&'static str, Option<[u8; 4]>> = BTreeMap::new();
    scalars.insert("linetype", 1.0);
    scalars.insert("alpha", style.alpha);
    match kind {
        MarkKind::Area | MarkKind::Ribbon => {
            scalars.insert("flipped_aes", 0.0);
            scalars.insert("linewidth", mark_defaults::LINEWIDTH);
            colours.insert("colour", style.colour);
            colours.insert("fill", Some(style.fill.unwrap_or(mark_defaults::FILL_GREY20)));
        }
        MarkKind::Tile => {
            scalars.insert("linewidth", mark_defaults::TILE_LINEWIDTH);
            scalars.insert("width", 1.0);
            scalars.insert("height", 1.0);
            colours.insert("colour", style.colour);
            colours.insert("fill", Some(style.fill.unwrap_or(mark_defaults::FILL_GREY20)));
        }
        MarkKind::Rect => {
            scalars.insert("linewidth", mark_defaults::LINEWIDTH);
            colours.insert("colour", style.colour);
            colours.insert("fill", Some(style.fill.unwrap_or(mark_defaults::FILL_RECT)));
        }
        MarkKind::Text => {
            scalars.remove("linetype");
            text_scalars(&mut scalars, style);
            colours.insert("colour", Some(style.colour.unwrap_or(mark_defaults::TEXT_COLOUR)));
        }
    }
    BuiltMarks { kind, group: core.group, num: core.num, texts, scalars, colours }
}

fn text_scalars(scalars: &mut BTreeMap<&'static str, f64>, style: &Style) {
    let size = style.size.unwrap_or(mark_defaults::FONTSIZE_PT / PT_PER_MM);
    for (k, v) in [
        ("nudge_x", 0.0),
        ("nudge_y", 0.0),
        ("size", size),
        ("angle", 0.0),
        ("hjust", 0.5),
        ("vjust", 0.5),
        ("fontface", 1.0),
        ("lineheight", mark_defaults::LINEHEIGHT),
    ] {
        scalars.insert(k, v);
    }
}
