//! Map polygons under `coord_geo`: project the rings, fit them to a panel, hand back pixels.
//!
//! A polygon feature is a list of rings of `(lon, lat)` degrees; the first ring is the outer
//! boundary and the rest are holes under even-odd fill. [`project_features`] applies a
//! [`Projection`] to every vertex and refuses the whole plot on the first error, so no point is
//! skipped. [`fit_to_area`] maps the plane coordinates onto a pixel rectangle with one uniform
//! scale (the aspect ratio is kept) and north up (larger plane `y` is a smaller pixel row).

use super::projection::{GeoError, Projection};
use crate::color::Rgba;
use std::fmt;
use std::sync::Arc;

/// The fill of every map polygon, in both the framebuffer and the SVG.
pub const POLYGON_FILL: Rgba = Rgba::rgb(78, 121, 167);

/// Polygon features: each feature is a list of rings, and each ring a list of `(x, y)` vertices.
pub type Features = Vec<Vec<Vec<(f64, f64)>>>;

/// A cloneable, shared projection, carried by [`crate::grammar::Coord::Geo`].
///
/// `Coord` derives `Debug` and `Clone`, and a bare `dyn Projection` is neither, so the plot holds
/// the projection behind an [`Arc`] and prints only its type.
#[derive(Clone)]
pub struct GeoProjection(Arc<dyn Projection>);

impl GeoProjection {
    /// Wraps a projection so a plot can own it.
    #[must_use]
    pub fn new<P: Projection + 'static>(projection: P) -> Self {
        Self(Arc::new(projection))
    }

    /// The wrapped projection.
    #[must_use]
    pub fn projection(&self) -> &dyn Projection {
        &*self.0
    }
}

impl fmt::Debug for GeoProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GeoProjection")
    }
}

/// Projects every vertex of every ring. The first [`GeoError`] is returned and nothing is kept.
///
/// # Errors
///
/// The [`GeoError`] of the first vertex the projection refuses.
pub fn project_features(
    projection: &dyn Projection,
    features: &[Vec<Vec<(f64, f64)>>],
) -> Result<Features, GeoError> {
    let project_ring = |ring: &[(f64, f64)]| -> Result<Vec<(f64, f64)>, GeoError> {
        ring.iter().map(|&(lon, lat)| projection.forward(lon, lat)).collect()
    };
    features
        .iter()
        .map(|rings| rings.iter().map(|ring| project_ring(ring)).collect::<Result<Vec<_>, _>>())
        .collect()
}

/// Maps plane coordinates onto the pixel rectangle `area = (x, y, width, height)`.
///
/// The bounding box of every vertex is scaled by the largest factor that keeps it inside the
/// area, then centred. The scale is the same on both axes, so shapes keep their proportions,
/// and a larger plane `y` lands on a smaller pixel row (north up). A box with no extent on one
/// axis takes the scale of the other axis; a box with no extent at all collapses to the centre,
/// where the even-odd fill paints nothing. No vertices give no rings.
#[must_use]
pub fn fit_to_area(plane: &Features, area: (f64, f64, f64, f64)) -> Features {
    let Some(((x0, y0), (x1, y1))) = extent(plane) else {
        return Vec::new();
    };
    let (area_x, area_y, area_w, area_h) = area;
    // Spans are computed on halved values: `x1 - x0` can overflow f64 to infinity, while
    // `x1 / 2 - x0 / 2` cannot. Halving is exact, so for ordinary inputs the scale and every
    // output coordinate are bit-identical to the unhalved computation.
    let (half_w, half_h) = (x1 / 2.0 - x0 / 2.0, y1 / 2.0 - y0 / 2.0);
    let scale = uniform_scale(half_w, half_h, area_w, area_h);
    let off_x = area_x + (area_w - half_w * scale) / 2.0;
    let off_y = area_y + (area_h - half_h * scale) / 2.0;
    plane
        .iter()
        .map(|rings| {
            rings
                .iter()
                .map(|ring| {
                    ring.iter()
                        .map(|&(x, y)| {
                            (
                                off_x + (x / 2.0 - x0 / 2.0) * scale,
                                off_y + (y1 / 2.0 - y / 2.0) * scale,
                            )
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Projects and fits in one step: the pixel rings a plot fills.
///
/// # Errors
///
/// The [`GeoError`] of [`project_features`].
pub fn pixel_features(
    projection: &dyn Projection,
    features: &[Vec<Vec<(f64, f64)>>],
    area: (f64, f64, f64, f64),
) -> Result<Features, GeoError> {
    Ok(fit_to_area(&project_features(projection, features)?, area))
}

/// The lower-left and upper-right corners of the box holding every vertex, if there is one.
fn extent(plane: &Features) -> Option<((f64, f64), (f64, f64))> {
    plane.iter().flatten().flatten().fold(None, |acc, &(x, y)| {
        Some(match acc {
            None => ((x, y), (x, y)),
            Some(((x0, y0), (x1, y1))) => ((x0.min(x), y0.min(y)), (x1.max(x), y1.max(y))),
        })
    })
}

/// The largest scale that fits a `box_w` by `box_h` box in an `area_w` by `area_h` area.
/// An axis with no extent does not limit the scale; if neither axis has extent the scale is 0.
fn uniform_scale(box_w: f64, box_h: f64, area_w: f64, area_h: f64) -> f64 {
    let sx = if box_w > 0.0 { area_w / box_w } else { f64::INFINITY };
    let sy = if box_h > 0.0 { area_h / box_h } else { f64::INFINITY };
    let scale = sx.min(sy);
    if scale.is_finite() {
        scale
    } else {
        0.0
    }
}
