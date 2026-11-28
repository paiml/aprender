//! Headless map runtime core (APR-MAP-001 MAP-5).
//!
//! Target-neutral plain Rust: no rendering, no clock, no platform calls. The camera, the
//! level-of-detail tiers, the culling and the layer-cache scheduler are pure functions over
//! `f64`, so the contracts `aprender-viz-map-camera-v1` and `aprender-viz-map-layer-cache-v1`
//! can be checked without a browser or a GPU.

use std::fmt;

pub mod camera;
pub mod client;
pub mod cull;
pub mod delta;
pub mod hex;
pub mod histogram;
pub mod interact;
pub mod layers;
pub mod lens;
pub mod lod;
pub mod pick;
pub mod search;
pub mod sprites;
pub mod urlstate;

pub use camera::{Camera2D, FlyTo, ZoomAnim};
pub use client::Client;
pub use cull::{visible_indices, Bounds};
pub use delta::Delta;
pub use hex::HexGrid;
pub use histogram::{Brush, Histogram, HISTOGRAM_BINS};
pub use interact::Interaction;
pub use layers::{
    EntityId, Event, Layers, LensId, Outcome, OverlayChange, Renders, CACHE_MARGIN_PX,
};
pub use lens::{Lens, LensRegistry, Scale};
pub use lod::{tier_for_radius, LodTier, BOUNDARIES_PX};
pub use search::SearchIndex;
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
    /// A value domain whose lower bound is not strictly below its upper bound.
    EmptyDomain,
    /// A lens id that is already registered.
    Duplicate(&'static str),
    /// A lens whose per-cell values do not match the grid's cell count.
    Mismatch(&'static str),
    /// A lens id that the registry does not hold.
    UnknownLens(LensId),
    /// A wire buffer that ends before the length its header declares.
    Truncated(&'static str),
    /// A wire buffer with bytes after the end its header declares.
    TrailingBytes(&'static str),
    /// A cell tag byte that is neither "no data" nor "value follows".
    UnknownTag(u8),
    /// A cell id at or beyond the grid's cell count.
    CellOutOfRange(u32),
    /// A delta whose sequence number is not after the last one applied to its lens.
    Stale { seq: u64, last: u64 },
    /// A delta whose sequence number skips past the next expected one.
    Gap { expected: u64, seq: u64 },
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
            Self::EmptyDomain => write!(f, "value domain must have lo < hi"),
            Self::Duplicate(what) => write!(f, "{what} is already registered"),
            Self::Mismatch(what) => write!(f, "{what} does not match the grid"),
            Self::UnknownLens(id) => write!(f, "lens {id} is not registered"),
            Self::Truncated(what) => write!(f, "{what} is truncated"),
            Self::TrailingBytes(what) => write!(f, "{what} has trailing bytes"),
            Self::UnknownTag(tag) => write!(f, "tag {tag} is not a cell tag"),
            Self::CellOutOfRange(cell) => write!(f, "cell {cell} is outside the grid"),
            Self::Stale { seq, last } => {
                write!(f, "sequence {seq} is not after the last applied {last}")
            }
            Self::Gap { expected, seq } => {
                write!(f, "sequence {seq} skips the expected {expected}")
            }
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
#[path = "lens_switch_tests.rs"]
mod lens_switch_tests;

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
            (MapError::EmptyDomain, "value domain must have lo < hi"),
            (MapError::Duplicate("lens"), "lens is already registered"),
            (MapError::Mismatch("lens"), "lens does not match the grid"),
            (MapError::UnknownLens(9), "lens 9 is not registered"),
            (MapError::Truncated("delta"), "delta is truncated"),
            (MapError::TrailingBytes("delta"), "delta has trailing bytes"),
            (MapError::UnknownTag(7), "tag 7 is not a cell tag"),
            (
                MapError::CellOutOfRange(10_530),
                "cell 10530 is outside the grid",
            ),
            (
                MapError::Stale { seq: 2, last: 3 },
                "sequence 2 is not after the last applied 3",
            ),
            (
                MapError::Gap {
                    expected: 4,
                    seq: 6,
                },
                "sequence 6 skips the expected 4",
            ),
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
