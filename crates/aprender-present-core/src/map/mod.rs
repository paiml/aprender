//! Headless map runtime core (APR-MAP-001 MAP-5).
//!
//! Target-neutral plain Rust: no rendering, no clock, no platform calls. The camera, the
//! level-of-detail tiers, the culling and the layer-cache scheduler are pure functions over
//! `f64`, so the contracts `aprender-viz-map-camera-v1` and `aprender-viz-map-layer-cache-v1`
//! can be checked without a browser or a GPU.

use std::fmt;

pub mod camera;
pub mod cull;
pub mod layers;
pub mod lod;
pub mod sprites;
pub mod urlstate;

pub use camera::{Camera2D, FlyTo, ZoomAnim};
pub use cull::{visible_indices, Bounds};
pub use layers::{
    EntityId, Event, Layers, LensId, Outcome, OverlayChange, Renders, CACHE_MARGIN_PX,
};
pub use lod::{tier_for_radius, LodTier, BOUNDARIES_PX};
pub use sprites::{generate_sprite, side_for_bucket, size_bucket, Sprite, SpriteAtlas};

/// Largest magnitude accepted for any coordinate, pixel offset or viewport size. With the scale
/// range below, every product, quotient and difference the runtime forms stays finite.
pub const MAX_COORD: f64 = 1e12;
/// Smallest accepted scale, in screen pixels per world unit.
pub const MIN_SCALE: f64 = 1e-9;
/// Largest accepted scale, in screen pixels per world unit.
pub const MAX_SCALE: f64 = 1e9;

/// Errors from the map runtime. Constructors and mutators return these instead of panicking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MapError {
    /// A value that must be finite was NaN or infinite.
    NonFinite(&'static str),
    /// A value that must be strictly greater than zero was zero or negative.
    NonPositive(&'static str),
    /// A value that must not be negative was negative.
    Negative(&'static str),
    /// Bounds had a minimum above its maximum on some axis.
    InvertedBounds,
    /// The start and target cameras of a fly-to do not share a viewport size.
    ViewportMismatch,
    /// A finite value lay outside the supported numeric domain.
    OutOfRange(&'static str),
    /// Text that does not parse as the named value (URL state codec).
    Malformed(&'static str),
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(what) => write!(f, "{what} must be finite"),
            Self::NonPositive(what) => write!(f, "{what} must be greater than zero"),
            Self::Negative(what) => write!(f, "{what} must not be negative"),
            Self::InvertedBounds => write!(f, "bounds minimum exceeds its maximum"),
            Self::ViewportMismatch => write!(f, "fly-to cameras must share a viewport size"),
            Self::OutOfRange(what) => write!(f, "{what} is outside the supported range"),
            Self::Malformed(what) => write!(f, "{what} is malformed"),
        }
    }
}

impl std::error::Error for MapError {}

/// Fails unless both components of `p` are finite and within `MAX_COORD`.
pub(crate) fn check_point(p: (f64, f64), what: &'static str) -> Result<(), MapError> {
    check_coord(p.0, what)?;
    check_coord(p.1, what)
}

/// Fails unless `v` is finite and within `MAX_COORD`.
pub(crate) fn check_coord(v: f64, what: &'static str) -> Result<(), MapError> {
    check_finite(v, what)?;
    if v.abs() <= MAX_COORD {
        Ok(())
    } else {
        Err(MapError::OutOfRange(what))
    }
}

/// Fails with `NonFinite(what)` unless `v` is finite.
pub(crate) fn check_finite(v: f64, what: &'static str) -> Result<(), MapError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(MapError::NonFinite(what))
    }
}

/// Returns `v` when it is finite and strictly positive.
pub(crate) fn check_scale(v: f64, what: &'static str) -> Result<f64, MapError> {
    check_finite(v, what)?;
    if v > 0.0 {
        Ok(v)
    } else {
        Err(MapError::NonPositive(what))
    }
}

/// Returns `v` when it is a scale inside `[MIN_SCALE, MAX_SCALE]`.
pub(crate) fn check_zoom(v: f64, what: &'static str) -> Result<f64, MapError> {
    let v = check_scale(v, what)?;
    if (MIN_SCALE..=MAX_SCALE).contains(&v) {
        Ok(v)
    } else {
        Err(MapError::OutOfRange(what))
    }
}

/// Fails unless `v` is finite and not negative.
pub(crate) fn check_non_negative(v: f64, what: &'static str) -> Result<(), MapError> {
    check_finite(v, what)?;
    if v >= 0.0 {
        Ok(())
    } else {
        Err(MapError::Negative(what))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_error_display_names_the_offending_value() {
        let cases = [
            (MapError::NonFinite("scale"), "scale must be finite"),
            (
                MapError::NonPositive("duration"),
                "duration must be greater than zero",
            ),
            (MapError::Negative("margin"), "margin must not be negative"),
            (
                MapError::InvertedBounds,
                "bounds minimum exceeds its maximum",
            ),
            (
                MapError::ViewportMismatch,
                "fly-to cameras must share a viewport size",
            ),
            (
                MapError::OutOfRange("pan"),
                "pan is outside the supported range",
            ),
            (MapError::Malformed("sel"), "sel is malformed"),
        ];
        for (err, want) in cases {
            assert_eq!(err.to_string(), want);
            let as_std: &dyn std::error::Error = &err;
            assert!(as_std.source().is_none());
        }
    }

    #[test]
    fn check_helpers_accept_valid_and_reject_invalid_values() {
        assert_eq!(check_scale(2.0, "s"), Ok(2.0));
        assert_eq!(check_scale(0.0, "s"), Err(MapError::NonPositive("s")));
        assert_eq!(check_scale(f64::NAN, "s"), Err(MapError::NonFinite("s")));
        assert_eq!(check_non_negative(0.0, "m"), Ok(()));
        assert_eq!(check_non_negative(-1e-9, "m"), Err(MapError::Negative("m")));
        assert_eq!(
            check_point((1.0, f64::INFINITY), "p"),
            Err(MapError::NonFinite("p"))
        );
        assert_eq!(check_finite(3.0, "f"), Ok(()));
    }
}
