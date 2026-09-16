//! Coordinate systems for Grammar of Graphics.
//!
//! Defines how positions are mapped to the plotting area.
//!
//! # `apply` refuses what it cannot do
//!
//! [`apply`] is the reader for [`Coord`]. It implements `Cartesian` — limits and `flip` — and
//! returns [`Error::UnsupportedCoord`] for `Polar` and `Fixed` rather than passing the point
//! through unchanged. A silent identity transform is precisely the defect this module is being
//! repaired for: `flip` was a field nothing read, so `coord_flip()` type-checked, ran, and did
//! nothing. Reintroducing that shape for `Polar` would rebuild the bug next to its own fix.
//!
//! # These are `f64`
//!
//! The fields were `f32`. Axis limits feed the scale that positions every mark, and APEX-001
//! compares rendered bytes across architectures; `f32` carries ~7 decimal digits, which is where
//! rounding divergence appears first. This is a breaking change, deliberately taken while the
//! crate is pre-1.0.

use crate::error::{Error, Result};

/// Coordinate system type.
#[derive(Debug, Clone)]
pub enum Coord {
    /// Cartesian coordinates (x, y).
    Cartesian {
        /// X axis limits.
        xlim: Option<(f64, f64)>,
        /// Y axis limits.
        ylim: Option<(f64, f64)>,
        /// Whether to flip x and y.
        flip: bool,
    },
    /// Polar coordinates (r, theta).
    Polar {
        /// Start angle in radians.
        start: f64,
        /// Direction: 1 for clockwise, -1 for counter-clockwise.
        direction: i8,
    },
    /// Fixed aspect ratio coordinates.
    Fixed {
        /// Aspect ratio (y/x).
        ratio: f64,
    },
}

/// Map one data point through the coordinate system.
///
/// Applied *before* geoms draw, so a geom never sees coordinate concerns.
///
/// # Errors
///
/// [`Error::UnsupportedCoord`] for [`Coord::Polar`] and [`Coord::Fixed`], whose transforms are not
/// implemented. They refuse loudly instead of returning the point unchanged — see the module docs.
///
/// ```
/// use trueno_viz::grammar::{apply, Coord};
///
/// assert_eq!(apply(&Coord::cartesian(), 3.0, 7.0).unwrap(), (3.0, 7.0));
/// assert_eq!(apply(&Coord::cartesian().flip(), 3.0, 7.0).unwrap(), (7.0, 3.0));
/// assert!(apply(&Coord::polar(), 3.0, 7.0).is_err());
/// ```
pub fn apply(coord: &Coord, x: f64, y: f64) -> Result<(f64, f64)> {
    match coord {
        Coord::Cartesian { flip: true, .. } => Ok((y, x)),
        Coord::Cartesian { flip: false, .. } => Ok((x, y)),
        Coord::Polar { .. } => Err(Error::UnsupportedCoord("polar")),
        Coord::Fixed { .. } => Err(Error::UnsupportedCoord("fixed aspect ratio")),
    }
}

/// Resolve the axis domains: declared limits win over the data's own range, and `flip` swaps the
/// two axes so the flip reaches the scales and not only the points.
///
/// # Errors
///
/// [`Error::UnsupportedCoord`], as [`apply`].
pub fn apply_limits(
    coord: &Coord,
    x_range: (f64, f64),
    y_range: (f64, f64),
) -> Result<((f64, f64), (f64, f64))> {
    match coord {
        Coord::Cartesian { xlim, ylim, flip } => {
            let x = xlim.unwrap_or(x_range);
            let y = ylim.unwrap_or(y_range);
            Ok(if *flip { (y, x) } else { (x, y) })
        }
        Coord::Polar { .. } => Err(Error::UnsupportedCoord("polar")),
        Coord::Fixed { .. } => Err(Error::UnsupportedCoord("fixed aspect ratio")),
    }
}

impl Default for Coord {
    fn default() -> Self {
        Coord::cartesian()
    }
}

impl Coord {
    /// Create a Cartesian coordinate system.
    #[must_use]
    pub fn cartesian() -> Self {
        Coord::Cartesian { xlim: None, ylim: None, flip: false }
    }

    /// Create a polar coordinate system.
    #[must_use]
    pub fn polar() -> Self {
        Coord::Polar { start: 0.0, direction: 1 }
    }

    /// Create a fixed aspect ratio coordinate system.
    #[must_use]
    pub fn fixed(ratio: f64) -> Self {
        Coord::Fixed { ratio }
    }

    /// Set x-axis limits.
    #[must_use]
    pub fn xlim(mut self, min: f64, max: f64) -> Self {
        if let Coord::Cartesian { ref mut xlim, .. } = self {
            *xlim = Some((min, max));
        }
        self
    }

    /// Set y-axis limits.
    #[must_use]
    pub fn ylim(mut self, min: f64, max: f64) -> Self {
        if let Coord::Cartesian { ref mut ylim, .. } = self {
            *ylim = Some((min, max));
        }
        self
    }

    /// Flip x and y axes.
    #[must_use]
    pub fn flip(mut self) -> Self {
        if let Coord::Cartesian { flip: ref mut f, .. } = self {
            *f = true;
        }
        self
    }

    /// Set polar start angle.
    #[must_use]
    pub fn start_angle(mut self, start: f64) -> Self {
        if let Coord::Polar { start: ref mut s, .. } = self {
            *s = start;
        }
        self
    }

    /// Set polar direction (1 = clockwise, -1 = counter-clockwise).
    #[must_use]
    pub fn direction(mut self, dir: i8) -> Self {
        if let Coord::Polar { direction: ref mut d, .. } = self {
            *d = if dir >= 0 { 1 } else { -1 };
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_cartesian() {
        let c = Coord::cartesian().xlim(0.0, 10.0).ylim(-5.0, 5.0);
        match c {
            Coord::Cartesian { xlim, ylim, flip } => {
                assert_eq!(xlim, Some((0.0, 10.0)));
                assert_eq!(ylim, Some((-5.0, 5.0)));
                assert!(!flip);
            }
            _ => panic!("Expected Cartesian"),
        }
    }

    #[test]
    fn test_coord_flip() {
        let c = Coord::cartesian().flip();
        match c {
            Coord::Cartesian { flip, .. } => assert!(flip),
            _ => panic!("Expected Cartesian"),
        }
    }

    #[test]
    fn test_coord_polar() {
        let c = Coord::polar().start_angle(std::f64::consts::PI).direction(-1);
        match c {
            Coord::Polar { start, direction } => {
                assert!((start - std::f64::consts::PI).abs() < 0.001);
                assert_eq!(direction, -1);
            }
            _ => panic!("Expected Polar"),
        }
    }

    #[test]
    fn test_coord_fixed() {
        let c = Coord::fixed(1.5);
        match c {
            Coord::Fixed { ratio } => {
                assert!((ratio - 1.5).abs() < 0.001);
            }
            _ => panic!("Expected Fixed"),
        }
    }

    #[test]
    fn test_coord_default() {
        let c = Coord::default();
        assert!(matches!(c, Coord::Cartesian { xlim: None, ylim: None, flip: false }));
    }

    #[test]
    fn test_xlim_on_non_cartesian() {
        // xlim on Polar should do nothing
        let c = Coord::polar().xlim(0.0, 10.0);
        assert!(matches!(c, Coord::Polar { .. }));
    }

    #[test]
    fn test_ylim_on_non_cartesian() {
        // ylim on Fixed should do nothing
        let c = Coord::fixed(1.0).ylim(0.0, 10.0);
        assert!(matches!(c, Coord::Fixed { .. }));
    }

    #[test]
    fn test_flip_on_non_cartesian() {
        // flip on Polar should do nothing
        let c = Coord::polar().flip();
        assert!(matches!(c, Coord::Polar { .. }));
    }

    #[test]
    fn test_start_angle_on_non_polar() {
        // start_angle on Cartesian should do nothing
        let c = Coord::cartesian().start_angle(1.0);
        assert!(matches!(c, Coord::Cartesian { .. }));
    }

    #[test]
    fn test_direction_on_non_polar() {
        // direction on Fixed should do nothing
        let c = Coord::fixed(1.0).direction(-1);
        assert!(matches!(c, Coord::Fixed { .. }));
    }

    #[test]
    fn test_direction_positive() {
        let c = Coord::polar().direction(1);
        match c {
            Coord::Polar { direction, .. } => {
                assert_eq!(direction, 1);
            }
            _ => panic!("Expected Polar"),
        }
    }

    #[test]
    fn test_direction_zero() {
        // Zero should be treated as positive (>=0)
        let c = Coord::polar().direction(0);
        match c {
            Coord::Polar { direction, .. } => {
                assert_eq!(direction, 1);
            }
            _ => panic!("Expected Polar"),
        }
    }

    #[test]
    fn test_coord_debug_clone() {
        let c = Coord::cartesian().xlim(0.0, 10.0);
        let c2 = c.clone();
        let _ = format!("{c2:?}");
    }

    #[test]
    fn test_polar_debug_clone() {
        let c = Coord::polar().start_angle(0.5);
        let c2 = c.clone();
        let _ = format!("{c2:?}");
    }

    #[test]
    fn test_fixed_debug_clone() {
        let c = Coord::fixed(2.0);
        let c2 = c.clone();
        let _ = format!("{c2:?}");
    }
}
