//! Map plots (`geo` feature): the framebuffer and the SVG of `geom_polygon` under `coord_geo`,
//! and of `geom_hex`.
//!
//! The plot area is the one panel a single-panel plot gets: the theme margin on every side.
//! Each polygon's rings are projected, fitted to that area with one uniform scale (north up),
//! and filled even-odd. A hex map has no projection: its cells are fitted with the same uniform
//! scale (so hexagons keep their aspect ratio), north up as for polygons, and each hexagon is
//! filled with the colour of its value. The SVG carries the same rings as `fill-rule="evenodd"`
//! paths on top of the framebuffer image. No axes or grid are drawn, since a map has no limits to
//! label.
//!
//! Refused rather than drawn: any layer other than `geom_polygon` under `coord_geo`, a `geom_hex`
//! beside another layer or under any coord but the plain Cartesian one, and any facet. Without the
//! `geo` feature neither exists, so every entry point here is a no-op.

use super::coord::Coord;
use super::facet::Facet;
#[cfg(feature = "geo")]
use super::geom::GeomType;
use super::ggplot::Layer;
use super::theme::Theme;
#[cfg(feature = "geo")]
use crate::color::Rgba;
#[cfg(feature = "geo")]
use crate::error::Error;
use crate::error::Result;
use crate::framebuffer::Framebuffer;
use crate::output::SvgEncoder;

/// One filled shape: its rings in pixel coordinates, and its colour.
#[cfg(feature = "geo")]
type Shape = (Vec<Vec<(f64, f64)>>, Rgba);

/// The fitted pixel rings of every polygon feature, or `None` when the coord is not a map.
#[cfg(feature = "geo")]
fn map_pixels(
    layers: &[Layer],
    coord: &Coord,
    facet: &Facet,
    size: (u32, u32),
    margin: u32,
) -> Result<Option<crate::geo::plot::Features>> {
    use crate::geo::plot::{pixel_features, Features};

    let Coord::Geo { projection } = coord else {
        return Ok(None);
    };
    if !matches!(facet, Facet::None) {
        return Err(Error::UnsupportedLayer("coord_geo with facets"));
    }
    let mut features: Features = Vec::new();
    for layer in layers {
        match &layer.geom.geom_type {
            GeomType::Polygon { features: polygons } => features.extend(polygons.iter().cloned()),
            _ => return Err(Error::UnsupportedLayer("coord_geo draws geom_polygon layers only")),
        }
    }
    let area = plot_area(size, margin);
    pixel_features(projection.projection(), &features, area)
        .map(Some)
        .map_err(|e| Error::ScaleDomain(e.to_string()))
}

/// The `(x, y, width, height)` of the plot area, in pixels, as `f64` for the fit.
#[cfg(feature = "geo")]
fn plot_area(size: (u32, u32), margin: u32) -> (f64, f64, f64, f64) {
    let (width, height) = size;
    let inner_w = width.saturating_sub(margin.saturating_mul(2));
    let inner_h = height.saturating_sub(margin.saturating_mul(2));
    (f64::from(margin), f64::from(margin), f64::from(inner_w), f64::from(inner_h))
}

/// Every filled shape of a map plot, or `None` when the plot is not a map.
///
/// # Errors
///
/// The errors of [`hex_shapes`] for a hex plot, and of `map_pixels` for a polygon plot.
#[cfg(feature = "geo")]
pub(super) fn map_shapes(
    layers: &[Layer],
    coord: &Coord,
    facet: &Facet,
    size: (u32, u32),
    margin: u32,
) -> Result<Option<Vec<Shape>>> {
    if layers.iter().any(|l| matches!(l.geom.geom_type, GeomType::Hex { .. })) {
        return hex_shapes(layers, coord, facet, size, margin).map(Some);
    }
    let Some(pixels) = map_pixels(layers, coord, facet, size, margin)? else {
        return Ok(None);
    };
    Ok(Some(pixels.into_iter().map(|rings| (rings, crate::geo::plot::POLYGON_FILL)).collect()))
}

/// The shapes of a `geom_hex` plot: every cell's hexagon, fitted to the plot area with the same
/// uniform, north-up fit as a polygon map. Larger plane `y` is drawn higher, so a hexagon's row
/// is oriented exactly as the same plane point would be under `geom_polygon`.
///
/// # Errors
///
/// [`Error::UnsupportedLayer`] for anything but exactly one `geom_hex` layer, a facet, a flipped
/// or axis-limited Cartesian coord, or any other coord; the errors of [`hex_colours`] for the cells.
#[cfg(feature = "geo")]
fn hex_shapes(
    layers: &[Layer],
    coord: &Coord,
    facet: &Facet,
    size: (u32, u32),
    margin: u32,
) -> Result<Vec<Shape>> {
    use crate::geo::plot::{fit_to_area, Features};

    if !matches!(coord, Coord::Cartesian { xlim: None, ylim: None, flip: false }) {
        return Err(Error::UnsupportedLayer(
            "geom_hex draws on the plain Cartesian coord, with no flip or axis limits",
        ));
    }
    if !matches!(facet, Facet::None) {
        return Err(Error::UnsupportedLayer("geom_hex with facets"));
    }
    let alone = || Error::UnsupportedLayer("geom_hex draws alone: one geom_hex layer, no other");
    let [layer] = layers else {
        return Err(alone());
    };
    let GeomType::Hex { layout, cells } = &layer.geom.geom_type else {
        return Err(alone());
    };
    let colours = hex_colours(cells)?;
    let plane: Features =
        cells.iter().map(|&(cell, _)| vec![hex_plane_ring(layout, cell)]).collect();
    let fitted = fit_to_area(&plane, plot_area(size, margin));
    Ok(fitted.into_iter().zip(colours).collect())
}

/// One cell's six corners in plane coordinates, the same plane a polygon map projects into.
#[cfg(feature = "geo")]
fn hex_plane_ring(
    layout: &crate::geo::hex::HexLayout,
    cell: crate::geo::hex::Cell,
) -> Vec<(f64, f64)> {
    layout.corners(cell).to_vec()
}

/// The colour of each hexagon: its value on the viridis scale over the range of all values.
///
/// All values equal leaves no range, so every hexagon takes the scale's midpoint colour.
///
/// # Errors
///
/// [`Error::ScaleDomain`] for an empty cell list, a non-finite value, or values the scale cannot
/// hold (outside the `f32` range, or too close together to tell apart).
#[cfg(feature = "geo")]
fn hex_colours(cells: &[(crate::geo::hex::Cell, f64)]) -> Result<Vec<Rgba>> {
    use crate::scale::{ColorScale, Scale};

    if cells.is_empty() {
        return Err(Error::ScaleDomain("geom_hex has no cells".into()));
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &(_, value) in cells {
        if !value.is_finite() {
            return Err(Error::ScaleDomain(format!("geom_hex value {value} is not finite")));
        }
        lo = lo.min(value);
        hi = hi.max(value);
    }
    if hi <= lo {
        let mid = ColorScale::viridis((0.0, 1.0)).map(|s| s.scale(0.5));
        let mid = mid.ok_or_else(|| Error::ScaleDomain("viridis scale unavailable".into()))?;
        return Ok(vec![mid; cells.len()]);
    }
    let (lo32, hi32) = (lo as f32, hi as f32);
    if !(lo32.is_finite() && hi32.is_finite()) {
        return Err(Error::ScaleDomain("geom_hex values exceed the f32 range".into()));
    }
    let scale = ColorScale::viridis((lo32, hi32)).ok_or_else(|| {
        Error::ScaleDomain("geom_hex values are too close together to colour apart".into())
    })?;
    Ok(cells.iter().map(|&(_, value)| scale.scale(value as f32)).collect())
}

/// Draws a map plot into a framebuffer of `size`, or returns `None` when the plot is not a map.
///
/// # Errors
///
/// [`Error::UnsupportedLayer`] for a non-polygon layer under `coord_geo` or a facet, [`Error::ScaleDomain`]
/// when the projection refuses a vertex or a hex value is refused, and [`Error::Rendering`] when a
/// ring is degenerate.
#[cfg(feature = "geo")]
pub(super) fn draw(
    layers: &[Layer],
    coord: &Coord,
    facet: &Facet,
    size: (u32, u32),
    theme: &Theme,
) -> Result<Option<Framebuffer>> {
    use crate::geo::fill::fill_polygon_even_odd;

    let Some(shapes) = map_shapes(layers, coord, facet, size, theme.margin)? else {
        return Ok(None);
    };
    let mut fb = Framebuffer::new(size.0, size.1)?;
    fb.clear(theme.background);
    let (x, y, w, h) = plot_area(size, theme.margin);
    let (x, y, w, h) = (x as u32, y as u32, w as u32, h as u32);
    for py in y..y.saturating_add(h) {
        for px in x..x.saturating_add(w) {
            fb.set_pixel(px, py, theme.panel_background);
        }
    }
    for (rings, colour) in &shapes {
        fill_polygon_even_odd(&mut fb, rings, *colour)
            .map_err(|e| Error::Rendering(e.to_string()))?;
    }
    Ok(Some(fb))
}

/// Draws a map plot into a framebuffer of `size`, or returns `None` when `coord` is not a map.
#[cfg(not(feature = "geo"))]
pub(super) fn draw(
    _layers: &[Layer],
    _coord: &Coord,
    _facet: &Facet,
    _size: (u32, u32),
    _theme: &Theme,
) -> Result<Option<Framebuffer>> {
    Ok(None)
}

/// Adds the map's shapes to `svg` as `fill-rule="evenodd"` paths. Other plots return `svg`.
///
/// # Errors
///
/// The same as [`draw`].
#[cfg(feature = "geo")]
pub(super) fn svg_polygons(
    layers: &[Layer],
    coord: &Coord,
    facet: &Facet,
    size: (u32, u32),
    margin: u32,
    mut svg: SvgEncoder,
) -> Result<SvgEncoder> {
    use crate::geo::fill::svg_path_d;

    let Some(shapes) = map_shapes(layers, coord, facet, size, margin)? else {
        return Ok(svg);
    };
    for (rings, colour) in &shapes {
        let d = svg_path_d(rings).map_err(|e| Error::Rendering(e.to_string()))?;
        svg = svg.path_even_odd(&d, *colour);
    }
    Ok(svg)
}

/// Adds the map's polygons to `svg`. Without the `geo` feature there are none.
#[cfg(not(feature = "geo"))]
pub(super) fn svg_polygons(
    _layers: &[Layer],
    _coord: &Coord,
    _facet: &Facet,
    _size: (u32, u32),
    _margin: u32,
    svg: SvgEncoder,
) -> Result<SvgEncoder> {
    Ok(svg)
}

/// Whether `coord` is a map projection, which draws no axes.
#[cfg(feature = "geo")]
pub(super) fn is_geo(coord: &Coord) -> bool {
    matches!(coord, Coord::Geo { .. })
}

/// Whether `coord` is a map projection. Without the `geo` feature it never is.
#[cfg(not(feature = "geo"))]
pub(super) fn is_geo(_coord: &Coord) -> bool {
    false
}

/// Whether the plot is drawn as a map, with no axes: a `coord_geo` plot or a `geom_hex` plot.
#[cfg(feature = "geo")]
pub(super) fn is_map(layers: &[Layer], coord: &Coord) -> bool {
    is_geo(coord) || layers.iter().any(|l| matches!(l.geom.geom_type, GeomType::Hex { .. }))
}

/// Whether the plot is drawn as a map. Without the `geo` feature only `is_geo` can say so.
#[cfg(not(feature = "geo"))]
pub(super) fn is_map(_layers: &[Layer], coord: &Coord) -> bool {
    is_geo(coord)
}
