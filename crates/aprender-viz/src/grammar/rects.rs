//! `geom_tile`, `geom_rect` and `geom_raster` (PLOT-GG-001, paiml/aprender#6909): the layout
//! `ggplot_build()` gives each, as ggplot2 4.0.3 computes it.
//!
//! * `geom_tile` (`GeomTile$setup_data`): `width` and `height` default to the data
//!   [`resolution`] of `x` and `y` (one panel here); `xmin`/`xmax` are `x -+ width/2` and
//!   `ymin`/`ymax` are `y -+ height/2`. The built `width` and `height` columns hold the
//!   parameter, or ggplot2's default aesthetic `1`, never the resolved size.
//! * `geom_rect`: the four bounds are the data and `x`/`y` do not exist.
//! * `geom_raster` (`GeomRaster$setup_data`): the cell size is the spacing of the sorted distinct
//!   `x` (`y`), `1` for a single value; uneven spacing warns (ggplot2's own message, reproduced
//!   in [`BuiltRects::warnings`]) and uses the smallest step. `xmin = x - w * (1 - hjust)` and
//!   `xmax = x + w * hjust`.
//!
//! The scales train on `x`, `y` and the four bounds, in f64 (the bounds of a half-width tile are
//! not f32 numbers). `NA` rows stay in the layer with `NaN` bounds, as in `ggplot_build()`.
//!
//! Scope. Continuous `x`/`y`, one panel, the identity stat and the default coordinate system.
//! A data-mapped colour, fill, alpha, size, shape, label or group, a facet, a non-default coord,
//! a text position and an infinite value are typed refusals, never a guess. R integer columns
//! make `resolution()` return 1; this frame has no integer type, so every column is a double
//! (the fixtures force doubles), and an all-integer-valued column resolves like a double one.

use super::aes::Aes;
use super::build::BuiltLayer;
use super::coord::Coord;
use super::data::DataFrame;
use super::facet::Facet;
use super::geom::GeomType;
use super::ggplot::Layer;
use super::path;
use crate::color::Rgba;
use crate::error::{Error, Result};

/// Which of the three geoms a [`BuiltRects`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RectKind {
    /// `geom_tile()`.
    Tile,
    /// `geom_rect()`.
    Rect,
    /// `geom_raster()`.
    Raster,
}

/// One tile, rect or raster layer, as `ggplot_build()` resolves it. One entry per data row.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltRects {
    /// Which geom.
    pub kind: RectKind,
    /// Left edges (`NaN` for a row the data cannot place).
    pub xmin: Vec<f64>,
    /// Right edges.
    pub xmax: Vec<f64>,
    /// Bottom edges.
    pub ymin: Vec<f64>,
    /// Top edges.
    pub ymax: Vec<f64>,
    /// The built `width` column of a tile: the parameter, or `1` (`None` for rect and raster).
    pub width: Option<f64>,
    /// The built `height` column of a tile: the parameter, or `1` (`None` for rect and raster).
    pub height: Option<f64>,
    /// Fill colour.
    pub fill: Rgba,
    /// Outline colour; `None` is `NA` (ggplot2's default for tile and rect; raster has none).
    pub colour: Option<Rgba>,
    /// Outline width in mm (tile 0.2, rect 0.5; `None` for raster).
    pub linewidth: Option<f64>,
    /// Outline linetype (1 = solid; `None` for raster).
    pub linetype: Option<u8>,
    /// Alpha (`None` is `NA`).
    pub alpha: Option<f64>,
    /// ggplot2's warnings for the layer (the first line of each), in order.
    pub warnings: Vec<String>,
}

/// What the renderers return for a tile, rect or raster layer: `layout()` resolves them, but no
/// renderer draws them yet, so a plot is refused rather than returned with the layer missing.
pub(super) const NOT_DRAWN: &str =
    "tile, rect and raster layers are not drawn by the renderer yet; use layout()";

/// Is this geom a tile, rect or raster?
pub(super) fn is_rect_family(g: &GeomType) -> bool {
    matches!(g, GeomType::Tile { .. } | GeomType::Rect | GeomType::Raster { .. })
}

fn refuse(msg: impl Into<String>) -> Error {
    Error::Rendering(msg.into())
}

/// The default `coord_cartesian` is the only coordinate system `layout()` models.
fn require_default_coord(coord: &Coord) -> Result<()> {
    match coord {
        Coord::Cartesian { xlim: None, ylim: None, flip: false } => {}
        Coord::Cartesian { flip: true, .. } => {
            return Err(Error::UnsupportedCoord(
                "coord_flip with tile, rect or raster layers: layout() does not flip",
            ))
        }
        Coord::Cartesian { .. } => return Err(Error::UnsupportedCoord(
            "coord_cartesian limits with tile, rect or raster layers: layout() does not apply them",
        )),
        Coord::Polar { .. } => {
            return Err(Error::UnsupportedCoord("coord_polar with tile, rect or raster layers"))
        }
        Coord::Fixed { .. } => {
            return Err(Error::UnsupportedCoord("coord_fixed with tile, rect or raster layers"))
        }
    }
    Ok(())
}

/// Refuse what `layout()` does not model while a tile, rect or raster layer is present: a
/// coordinate system but the default, a stat but the identity (a facet is refused by
/// [`build_layer`]).
///
/// # Errors
///
/// [`Error::UnsupportedCoord`], [`Error::UnsupportedStat`].
pub(super) fn require_supported(layers: &[Layer], coord: &Coord) -> Result<()> {
    if !layers.iter().any(|l| is_rect_family(&l.geom.geom_type)) {
        return Ok(());
    }
    require_default_coord(coord)?;
    for l in layers.iter().filter(|l| is_rect_family(&l.geom.geom_type)) {
        if l.geom.stat.as_ref().and_then(path::unsupported_stat).is_some() {
            return Err(Error::UnsupportedStat(
                "a stat other than identity on a tile, rect or raster layer: layout() computes no stat",
            ));
        }
    }
    Ok(())
}

/// The tolerance ggplot2 uses for "no spacing": `sqrt(.Machine$double.eps)`.
fn spacing_tolerance() -> f64 {
    f64::EPSILON.sqrt()
}

/// Sorted distinct finite values.
fn sorted_unique(v: &[f64]) -> Vec<f64> {
    let mut u: Vec<f64> = v.iter().copied().filter(|a| a.is_finite()).collect();
    u.sort_by(f64::total_cmp);
    u.dedup();
    u
}

/// ggplot2's `resolution(x, zero = FALSE)` for a double column: the smallest gap between distinct
/// values that exceeds `sqrt(eps)`, `1` when none does (a constant column). R's extra
/// `zero_range()` early return cannot fire for f32-exact values, whose distinct gaps are at least
/// 1e-7 relative against its 2e-13 tolerance, so it is not modelled.
pub(super) fn resolution(x: &[f64]) -> f64 {
    let u = sorted_unique(x);
    let tol = spacing_tolerance();
    u.windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > tol)
        .fold(None, |m: Option<f64>, d| Some(m.map_or(d, |m| m.min(d))))
        .unwrap_or(1.0)
}

/// `GeomRaster$setup_data`'s cell size along one axis; pushes ggplot2's warning when uneven.
pub(super) fn raster_step(v: &[f64], warnings: &mut Vec<String>) -> f64 {
    let u = sorted_unique(v);
    let diffs: Vec<f64> = u.windows(2).map(|w| w[1] - w[0]).collect();
    let Some(&first) = diffs.first() else { return 1.0 };
    let tol = spacing_tolerance();
    if diffs.windows(2).any(|p| (p[1] - p[0]).abs() > tol) {
        // Verbatim ggplot2 4.0.3, and the same text for the y axis: its y branch says "horizontal"
        // too (R's copy-paste, confirmed by running it). Not a typo here; do not reword.
        warnings.push(
            "Raster pixels are placed at uneven horizontal intervals and will be shifted"
                .to_string(),
        );
        return diffs.iter().copied().fold(first, f64::min);
    }
    first
}

/// A fixed f32 aesthetic as ggplot2 saw the decimal the caller typed.
fn widen(v: f32) -> f64 {
    format!("{v}").parse().unwrap_or_else(|_| f64::from(v))
}

fn require_plain_aes(aes: &Aes) -> Result<()> {
    if aes.color.is_some()
        || aes.size.is_some()
        || aes.shape.is_some()
        || aes.alpha.is_some()
        || aes.fill.is_some()
        || aes.label.is_some()
    {
        return Err(refuse(
            "a data-mapped colour, size, shape, alpha, fill or label is not supported on a \
             tile, rect or raster layer",
        ));
    }
    if aes.size_value.is_some() {
        return Err(refuse("a fixed size is not supported on a tile, rect or raster layer"));
    }
    if aes.group.is_some() {
        return Err(refuse("a group aesthetic is not supported on a tile, rect or raster layer"));
    }
    Ok(())
}

/// Build one tile, rect or raster layer.
///
/// # Errors
///
/// [`Error::Rendering`] for an aesthetic this row does not model or a missing mapping,
/// [`Error::UnknownColumn`] for an absent column. A layer with no placeable row builds empty bounds
/// and `layout()` refuses it with [`Error::ScaleDomain`] when it finds the axis untrained.
pub(super) fn build_layer(
    layer: &Layer,
    plot_aes: &Aes,
    plot_data: &DataFrame,
    facet: &Facet,
) -> Result<BuiltLayer> {
    if !matches!(facet, Facet::None) {
        return Err(refuse("a facet with tile, rect or raster layers: layout() builds one panel"));
    }
    let data = layer.data.as_ref().unwrap_or(plot_data);
    let aes = plot_aes.merge(&layer.aes);
    require_plain_aes(&aes)?;
    let bounds_mapped =
        aes.xmin.is_some() || aes.xmax.is_some() || aes.ymin.is_some() || aes.ymax.is_some();
    if bounds_mapped && !matches!(layer.geom.geom_type, GeomType::Rect) {
        return Err(refuse(
            "xmin, xmax, ymin or ymax on a tile or raster layer: ggplot2 resolves those bounds as \
             a rect, which this row does not model",
        ));
    }
    let mut warnings = Vec::new();
    let p = match layer.geom.geom_type {
        GeomType::Rect => rect_parts(&aes, data)?,
        GeomType::Tile { width, height } => tile_parts(&aes, data, width, height)?,
        GeomType::Raster { hjust, vjust } => raster_parts(&aes, data, hjust, vjust, &mut warnings)?,
        _ => return Err(refuse("not a tile, rect or raster layer")),
    };
    let raster = p.kind == RectKind::Raster;
    let (xmin, xmax, ymin, ymax) = p.bounds;
    let rects = BuiltRects {
        kind: p.kind,
        xmin,
        xmax,
        ymin,
        ymax,
        width: p.width,
        height: p.height,
        fill: p.fill,
        colour: if raster { None } else { aes.color_value },
        linewidth: p.linewidth,
        linetype: p.linetype,
        alpha: aes.alpha_value.map(widen),
        warnings,
    };
    Ok(BuiltLayer {
        x: p.x,
        y: p.y,
        points: None,
        path: None,
        rects: Some(rects),
        bins: None,
        density: None,
        filled: None,
    })
}

/// What one geom contributes before the shared outline, colour and alpha are applied.
struct Parts {
    kind: RectKind,
    x: Vec<f64>,
    y: Vec<f64>,
    bounds: (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>),
    width: Option<f64>,
    height: Option<f64>,
    fill: Rgba,
    linewidth: Option<f64>,
    linetype: Option<u8>,
}

/// geom_rect: the four bounds are the data.
fn rect_parts(aes: &Aes, data: &DataFrame) -> Result<Parts> {
    if aes.x.is_some() || aes.y.is_some() {
        return Err(refuse("geom_rect takes xmin, xmax, ymin and ymax, not x or y"));
    }
    let col = |name: &Option<String>, what: &str| match name {
        Some(n) => path::read_column(data, n, what),
        None => Err(refuse(format!("geom_rect needs the {what} aesthetic"))),
    };
    let xmin = col(&aes.xmin, "xmin")?;
    let xmax = col(&aes.xmax, "xmax")?;
    let ymin = col(&aes.ymin, "ymin")?;
    let ymax = col(&aes.ymax, "ymax")?;
    Ok(Parts {
        kind: RectKind::Rect,
        x: Vec::new(),
        y: Vec::new(),
        bounds: (xmin, xmax, ymin, ymax),
        width: None,
        height: None,
        fill: aes.fill_value.unwrap_or(Rgba { r: 89, g: 89, b: 89, a: 255 }),
        linewidth: Some(0.5),
        linetype: Some(1),
    })
}

/// geom_tile: centred on (x, y), `resolution` wide unless a size is given.
fn tile_parts(aes: &Aes, data: &DataFrame, w: Option<f64>, h: Option<f64>) -> Result<Parts> {
    if w.is_some_and(|v| !v.is_finite()) || h.is_some_and(|v| !v.is_finite()) {
        return Err(refuse("a tile width or height must be a finite number"));
    }
    let xs = path::read_column(data, aes.x.as_deref().unwrap_or("x"), "x")?;
    let ys = path::read_column(data, aes.y.as_deref().unwrap_or("y"), "y")?;
    let (dw, dh) = (w.unwrap_or_else(|| resolution(&xs)), h.unwrap_or_else(|| resolution(&ys)));
    let half = |v: &[f64], d: f64, sign: f64| -> Vec<f64> {
        v.iter().map(|a| a + sign * d / 2.0).collect()
    };
    let bounds = (half(&xs, dw, -1.0), half(&xs, dw, 1.0), half(&ys, dh, -1.0), half(&ys, dh, 1.0));
    Ok(Parts {
        kind: RectKind::Tile,
        x: xs,
        y: ys,
        bounds,
        width: Some(w.unwrap_or(1.0)),
        height: Some(h.unwrap_or(1.0)),
        fill: aes.fill_value.unwrap_or(Rgba { r: 51, g: 51, b: 51, a: 255 }),
        linewidth: Some(0.2),
        linetype: Some(1),
    })
}

/// geom_raster: cells one grid step wide, placed by `hjust` and `vjust`.
fn raster_parts(
    aes: &Aes,
    data: &DataFrame,
    hjust: f64,
    vjust: f64,
    warnings: &mut Vec<String>,
) -> Result<Parts> {
    if !hjust.is_finite() || !vjust.is_finite() {
        return Err(refuse("a raster hjust or vjust must be a finite number"));
    }
    let xs = path::read_column(data, aes.x.as_deref().unwrap_or("x"), "x")?;
    let ys = path::read_column(data, aes.y.as_deref().unwrap_or("y"), "y")?;
    let dw = raster_step(&xs, warnings);
    let dh = raster_step(&ys, warnings);
    let at = |v: &[f64], d: f64, k: f64| -> Vec<f64> { v.iter().map(|a| a + d * k).collect() };
    let bounds = (
        at(&xs, dw, -(1.0 - hjust)),
        at(&xs, dw, hjust),
        at(&ys, dh, -(1.0 - vjust)),
        at(&ys, dh, vjust),
    );
    Ok(Parts {
        kind: RectKind::Raster,
        x: xs,
        y: ys,
        bounds,
        width: None,
        height: None,
        fill: aes.fill_value.unwrap_or(Rgba { r: 51, g: 51, b: 51, a: 255 }),
        linewidth: None,
        linetype: None,
    })
}

/// Extend `acc` with every finite value of `values`.
fn extend(acc: (f64, f64), values: &[f64]) -> (f64, f64) {
    values.iter().filter(|v| v.is_finite()).fold(acc, |(lo, hi), v| (lo.min(*v), hi.max(*v)))
}

/// Train f64 x and y domains on one built tile, rect or raster layer (`x`, `y` and the bounds).
pub(super) fn train(built: &BuiltLayer, x: (f64, f64), y: (f64, f64)) -> ((f64, f64), (f64, f64)) {
    let Some(r) = built.rects.as_ref() else { return (x, y) };
    let x = extend(extend(extend(x, &built.x), &r.xmin), &r.xmax);
    let y = extend(extend(extend(y, &built.y), &r.ymin), &r.ymax);
    (x, y)
}
