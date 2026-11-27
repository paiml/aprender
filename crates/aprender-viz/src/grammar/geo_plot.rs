//! Map plots under `coord_geo` (`geo` feature): the framebuffer and the SVG of `geom_polygon`.
//!
//! The plot area is the one panel a single-panel plot gets: the theme margin on every side.
//! Each polygon's rings are projected, fitted to that area with one uniform scale (north up),
//! and filled even-odd. The SVG carries the same rings as `fill-rule="evenodd"` paths on top of
//! the framebuffer image. No axes or grid are drawn, since a map has no limits to label.
//!
//! Refused rather than drawn: any layer other than `geom_polygon`, and any facet. Without the
//! `geo` feature `coord_geo` does not exist, so every entry point here is a no-op.

use super::coord::Coord;
use super::facet::Facet;
#[cfg(feature = "geo")]
use super::geom::GeomType;
use super::ggplot::Layer;
use super::theme::Theme;
#[cfg(feature = "geo")]
use crate::error::Error;
use crate::error::Result;
use crate::framebuffer::Framebuffer;
use crate::output::SvgEncoder;

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

/// Draws a map plot into a framebuffer of `size`, or returns `None` when `coord` is not a map.
///
/// # Errors
///
/// [`Error::UnsupportedLayer`] for a non-polygon layer or a facet, [`Error::ScaleDomain`] when
/// the projection refuses a vertex, and [`Error::Rendering`] when a ring is degenerate.
#[cfg(feature = "geo")]
pub(super) fn draw(
    layers: &[Layer],
    coord: &Coord,
    facet: &Facet,
    size: (u32, u32),
    theme: &Theme,
) -> Result<Option<Framebuffer>> {
    use crate::geo::fill::fill_polygon_even_odd;
    use crate::geo::plot::POLYGON_FILL;

    let Some(pixels) = map_pixels(layers, coord, facet, size, theme.margin)? else {
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
    for feature in &pixels {
        fill_polygon_even_odd(&mut fb, feature, POLYGON_FILL)
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

/// Adds the map's polygons to `svg` as `fill-rule="evenodd"` paths. Other coords return `svg`.
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
    use crate::geo::plot::POLYGON_FILL;

    let Some(pixels) = map_pixels(layers, coord, facet, size, margin)? else {
        return Ok(svg);
    };
    for feature in &pixels {
        let d = svg_path_d(feature).map_err(|e| Error::Rendering(e.to_string()))?;
        svg = svg.path_even_odd(&d, POLYGON_FILL);
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
