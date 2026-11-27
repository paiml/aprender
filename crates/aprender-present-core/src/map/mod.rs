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

pub use camera::{Camera2D, FlyTo, ZoomAnim};
pub use cull::{visible_indices, Bounds};
pub use layers::{
    EntityId, Event, Layers, LensId, Outcome, OverlayChange, Renders, CACHE_MARGIN_PX,
};
pub use lod::{tier_for_radius, LodTier, BOUNDARIES_PX};

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
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(what) => write!(f, "{what} must be finite"),
            Self::NonPositive(what) => write!(f, "{what} must be greater than zero"),
            Self::Negative(what) => write!(f, "{what} must not be negative"),
            Self::InvertedBounds => write!(f, "bounds minimum exceeds its maximum"),
            Self::ViewportMismatch => write!(f, "fly-to cameras must share a viewport size"),
        }
    }
}

impl std::error::Error for MapError {}

/// Fails with `NonFinite(what)` unless both components of `p` are finite.
pub(crate) fn check_point(p: (f64, f64), what: &'static str) -> Result<(), MapError> {
    check_finite(p.0, what)?;
    check_finite(p.1, what)
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

/// Fails unless `v` is finite and not negative.
pub(crate) fn check_non_negative(v: f64, what: &'static str) -> Result<(), MapError> {
    check_finite(v, what)?;
    if v >= 0.0 {
        Ok(())
    } else {
        Err(MapError::Negative(what))
    }
}
