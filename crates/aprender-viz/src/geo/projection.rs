//! Projection types: Albers equal-area, AlbersUSA, Mercator, equirectangular.
//!
//! Angles are degrees at the API. Internally longitude differences are wrapped into
//! `[-180, 180)` before any trigonometry, so `lon` and `lon + 360` project identically.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// Absorbs rounding at a domain edge (in radians, or in units of `R` where stated).
const DOMAIN_TOL: f64 = 1e-12;

/// Mercator latitude cutoff in degrees.
///
/// Mercator's `y` diverges at the poles. [`Mercator`] therefore accepts `|lat| <= 85`
/// and refuses anything beyond it, including the poles themselves.
pub const MERCATOR_MAX_LAT_DEG: f64 = 85.0;

/// Slack, in degrees, for a plane point whose inverse lands on a sub-region box edge with
/// rounding error. A point within this distance of a box is clamped into the box.
const BOX_SLACK_DEG: f64 = 1e-9;

/// Lon/lat boxes of the three AlbersUSA sub-regions as `[lon_min, lon_max, lat_min, lat_max]`
/// in degrees: contiguous states, Alaska, Hawaii. The boxes do not overlap.
pub(crate) const US_SUB_REGION_BOXES: [[f64; 4]; 3] =
    [[-125.0, -66.0, 24.0, 50.0], [-170.0, -129.0, 51.0, 72.0], [-161.0, -154.0, 18.0, 23.0]];

/// Errors raised by map projections.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum GeoError {
    /// An input coordinate or a constructor parameter was NaN or infinite.
    #[error("non-finite input")]
    NonFiniteInput,
    /// A computed plane coordinate was NaN or infinite (overflow or cancellation).
    #[error("non-finite result")]
    NonFiniteResult,
    /// A value lies outside the projection's domain.
    #[error("{what} = {value} is outside the projection domain")]
    OutOfDomain {
        /// Which quantity was checked.
        what: &'static str,
        /// The offending value.
        value: f64,
    },
    /// A constructor parameter is invalid.
    #[error("invalid parameter {name} = {value}")]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// The offending value.
        value: f64,
    },
    /// A lon/lat point lies in none of the AlbersUSA sub-region boxes.
    #[error("({lon}, {lat}) lies in no AlbersUSA sub-region")]
    NoSubRegion {
        /// Longitude, wrapped into `[-180, 180)`.
        lon: f64,
        /// Latitude in degrees.
        lat: f64,
    },
    /// A plane point maps back into none of the AlbersUSA sub-region boxes.
    #[error("({x}, {y}) lies in no AlbersUSA sub-region image")]
    NoPlacement {
        /// Plane x coordinate.
        x: f64,
        /// Plane y coordinate.
        y: f64,
    },
}

/// A map projection between geographic degrees and plane coordinates.
pub trait Projection {
    /// Projects `(lon_deg, lat_deg)` to plane coordinates in units of the radius.
    ///
    /// # Errors
    ///
    /// Returns [`GeoError`] when an input is non-finite or outside the domain, or when the
    /// result is non-finite.
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> Result<(f64, f64), GeoError>;

    /// Inverts [`Projection::forward`]. Longitude of the result is in `[-180, 180)`.
    ///
    /// # Errors
    ///
    /// Returns [`GeoError`] when a plane point is non-finite or lies outside the image of
    /// the forward domain.
    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), GeoError>;
}

/// Wraps an angle in degrees into `[-180, 180)`.
pub(crate) fn wrap180(deg: f64) -> f64 {
    let w = (deg + 180.0).rem_euclid(360.0) - 180.0;
    // `rem_euclid` can round up to the modulus for a tiny negative dividend.
    if w >= 180.0 {
        -180.0
    } else {
        w
    }
}

/// Rejects NaN and infinities on the input path.
fn finite_input(value: f64) -> Result<f64, GeoError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GeoError::NonFiniteInput)
    }
}

/// Rejects `|value| > limit`. Callers pass the limit including any [`DOMAIN_TOL`].
fn within(what: &'static str, value: f64, limit: f64) -> Result<f64, GeoError> {
    if value.abs() <= limit {
        Ok(value)
    } else {
        Err(GeoError::OutOfDomain { what, value })
    }
}

/// Rejects a non-finite result pair.
fn finite_result(x: f64, y: f64) -> Result<(f64, f64), GeoError> {
    if x.is_finite() && y.is_finite() {
        Ok((x, y))
    } else {
        Err(GeoError::NonFiniteResult)
    }
}

/// Validates a sphere radius: finite and strictly positive.
fn checked_radius(radius: f64) -> Result<f64, GeoError> {
    if radius.is_finite() && radius > 0.0 {
        Ok(radius)
    } else {
        Err(GeoError::InvalidParameter { name: "radius", value: radius })
    }
}

/// Validates a free parameter: finite.
fn checked_param(name: &'static str, value: f64) -> Result<f64, GeoError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GeoError::InvalidParameter { name, value })
    }
}

/// Validates a latitude parameter: finite and within `[-90, 90]`.
fn checked_lat(name: &'static str, value: f64) -> Result<f64, GeoError> {
    if value.is_finite() && value.abs() <= 90.0 {
        Ok(value)
    } else {
        Err(GeoError::InvalidParameter { name, value })
    }
}

/// Mercator `y` for a latitude in radians, on the unit sphere.
fn mercator_y(lat_rad: f64) -> f64 {
    libm::log((FRAC_PI_4 + lat_rad / 2.0).tan())
}

/// Albers equal-area conic projection on a sphere, with two standard parallels.
///
/// With standard parallels `phi1`, `phi2` and `n = (sin phi1 + sin phi2) / 2`, the cone
/// constant is `C = cos^2 phi1 + 2 n sin phi1`. Forward:
/// `rho = R sqrt(C - 2 n sin phi) / n`, `theta = n (lambda - lambda0)`,
/// `x = rho sin theta`, `y = rho0 - rho cos theta`, where `rho0` is `rho` at the origin
/// latitude. The domain is all latitudes in `[-90, 90]` and longitudes wrapped into
/// `[-180, 180)`; the image is the sector `|theta| <= n pi`.
///
/// Conditioning: inverse latitude uses `asin`, which is ill-conditioned near the poles.
/// Round-trip latitude error is below 1e-7 degrees for `|lat| <= 89.9` and below 1e-5 degrees
/// for `|lat| > 89.9`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlbersEqualArea {
    radius: f64,
    lon0_deg: f64,
    n: f64,
    c: f64,
    rho0: f64,
}

impl AlbersEqualArea {
    /// Builds an Albers projection.
    ///
    /// `std1_deg` and `std2_deg` are the standard parallels and `origin_lat_deg` the latitude
    /// of the projection origin; all three lie in `[-90, 90]`. `origin_lon_deg` is any finite
    /// longitude. The cone constant `n` must be positive, so both parallels in the
    /// southern hemisphere (a southern-only cone) are rejected by this phase.
    ///
    /// # Errors
    ///
    /// Returns [`GeoError::InvalidParameter`] for a non-finite or out-of-range angle, a
    /// non-positive or non-finite `radius`, or `n <= 0` (for example `std2 = -std1`, where
    /// `n = 0`).
    pub fn new(
        std1_deg: f64,
        std2_deg: f64,
        origin_lat_deg: f64,
        origin_lon_deg: f64,
        radius: f64,
    ) -> Result<Self, GeoError> {
        let radius = checked_radius(radius)?;
        let std1 = checked_lat("standard_parallel_1", std1_deg)?;
        let std2 = checked_lat("standard_parallel_2", std2_deg)?;
        let lat0 = checked_lat("origin_lat", origin_lat_deg)?;
        let lon0 = checked_param("origin_lon", origin_lon_deg)?;
        let n = (std1.to_radians().sin() + std2.to_radians().sin()) / 2.0;
        if n <= 0.0 {
            return Err(GeoError::InvalidParameter { name: "n", value: n });
        }
        Ok(Self::from_params(radius, lon0, std1, std2, lat0))
    }

    /// Infallible constructor for parameters already validated by the caller.
    ///
    /// Requires `n > 0`, which the callers guarantee: [`AlbersEqualArea::new`] checks it and
    /// the AlbersUSA constants satisfy it. With `n > 0` and both parallels in `[-90, 90]`, the
    /// square-root argument `C - 2 n sin phi` is non-negative on the whole latitude range.
    fn from_params(radius: f64, lon0_deg: f64, std1: f64, std2: f64, lat0: f64) -> Self {
        let s1 = std1.to_radians().sin();
        let n = (s1 + std2.to_radians().sin()) / 2.0;
        // Factored so that near `s1 = -1` it rounds consistently with `n`: `1 + s1` is exact there.
        let c = (1.0 - s1) * (1.0 + s1) + 2.0 * n * s1;
        let rho0 = radius * (c - 2.0 * n * lat0.to_radians().sin()).max(0.0).sqrt() / n;
        Self { radius, lon0_deg, n, c, rho0 }
    }
}

impl Projection for AlbersEqualArea {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> Result<(f64, f64), GeoError> {
        let lon = finite_input(lon_deg)?;
        let lat = within("latitude", finite_input(lat_deg)?, 90.0)?;
        let theta = self.n * wrap180(lon - self.lon0_deg).to_radians();
        // The radicand is non-negative in exact arithmetic; rounding can push it just below zero.
        let radicand = (self.c - 2.0 * self.n * lat.to_radians().sin()).max(0.0);
        let rho = self.radius * radicand.sqrt() / self.n;
        finite_result(rho * theta.sin(), self.rho0 - rho * theta.cos())
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), GeoError> {
        let x = finite_input(x)?;
        let dy = self.rho0 - finite_input(y)?;
        let theta = within("sector angle", x.atan2(dy), self.n * PI + DOMAIN_TOL)?;
        let rho = x.hypot(dy);
        let q = rho * self.n / self.radius;
        let sin_lat = (self.c - q * q) / (2.0 * self.n);
        let sin_lat = within("sin(latitude)", sin_lat, 1.0 + DOMAIN_TOL)?.clamp(-1.0, 1.0);
        let lon = self.lon0_deg + (theta / self.n).to_degrees();
        finite_result(wrap180(lon), sin_lat.asin().to_degrees())
    }
}

/// One AlbersUSA sub-region: its projection, its lon/lat box, and its placement in the plane.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SubRegion {
    proj: AlbersEqualArea,
    /// `[lon_min, lon_max, lat_min, lat_max]` in degrees, inclusive.
    bbox: [f64; 4],
    /// Scale applied to the sub-projection's plane coordinates.
    scale: f64,
    /// Plane translation, in units of the radius, applied after scaling.
    shift: (f64, f64),
}

impl SubRegion {
    /// True when a wrapped lon/lat lies in this sub-region's box (edges inclusive).
    fn contains(&self, lon: f64, lat: f64) -> bool {
        let [lon_min, lon_max, lat_min, lat_max] = self.bbox;
        (lon_min..=lon_max).contains(&lon) && (lat_min..=lat_max).contains(&lat)
    }

    /// Inverts a composite plane point through this sub-region, if it lands in the box.
    /// A sub-projection error means "not this sub-region", so the caller tries the next one.
    /// A point within `BOX_SLACK_DEG` of the box counts as inside and is clamped into it, so a
    /// valid box-edge point that inverts with rounding error still round-trips.
    fn unplace(&self, x: f64, y: f64, radius: f64) -> Option<(f64, f64)> {
        let sub_x = (x - self.shift.0 * radius) / self.scale;
        let sub_y = (y - self.shift.1 * radius) / self.scale;
        let (lon, lat) = self.proj.inverse(sub_x, sub_y).ok()?;
        let [lon_min, lon_max, lat_min, lat_max] = self.bbox;
        let near = |v: f64, lo: f64, hi: f64| v >= lo - BOX_SLACK_DEG && v <= hi + BOX_SLACK_DEG;
        (near(lon, lon_min, lon_max) && near(lat, lat_min, lat_max))
            .then(|| (lon.clamp(lon_min, lon_max), lat.clamp(lat_min, lat_max)))
    }
}

/// The AlbersUSA composite: contiguous states, Alaska, and Hawaii, each in its own Albers
/// projection, placed in one plane.
///
/// The contiguous projection is the standard USA Albers (parallels 29.5 and 45.5, origin
/// 37.5 N 96 W), placed unscaled at the plane origin. Alaska uses parallels 55 and 65 with
/// origin 50 N 154 W, scaled by 0.35 and shifted by `(-0.90, -0.45) R`. Hawaii uses
/// parallels 8 and 18 with origin 13 N 157 W, unscaled and shifted by `(-0.60, -0.62) R`.
/// The plane images of the three boxes are disjoint (contiguous `y >= -0.17 R`, Alaska
/// `x` in `[-0.96, -0.80] R`, Hawaii `x` in `[-0.67, -0.55] R` and `y` in `[-0.53, -0.45] R`).
/// A point in no box is an error, not a fallback to another projection.
#[derive(Debug, Clone, PartialEq)]
pub struct AlbersUsa {
    radius: f64,
    regions: [SubRegion; 3],
}

impl AlbersUsa {
    /// Builds the composite projection on a sphere of radius `radius`.
    ///
    /// # Errors
    ///
    /// Returns [`GeoError::InvalidParameter`] when `radius` is non-finite or not positive.
    pub fn new(radius: f64) -> Result<Self, GeoError> {
        let radius = checked_radius(radius)?;
        let [contiguous, alaska, hawaii] = US_SUB_REGION_BOXES;
        let regions = [
            SubRegion {
                proj: AlbersEqualArea::from_params(radius, -96.0, 29.5, 45.5, 37.5),
                bbox: contiguous,
                scale: 1.0,
                shift: (0.0, 0.0),
            },
            SubRegion {
                proj: AlbersEqualArea::from_params(radius, -154.0, 55.0, 65.0, 50.0),
                bbox: alaska,
                scale: 0.35,
                shift: (-0.90, -0.45),
            },
            SubRegion {
                proj: AlbersEqualArea::from_params(radius, -157.0, 8.0, 18.0, 13.0),
                bbox: hawaii,
                scale: 1.0,
                shift: (-0.60, -0.62),
            },
        ];
        Ok(Self { radius, regions })
    }
}

impl Projection for AlbersUsa {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> Result<(f64, f64), GeoError> {
        let lon = wrap180(finite_input(lon_deg)?);
        let lat = finite_input(lat_deg)?;
        let region = self
            .regions
            .iter()
            .find(|region| region.contains(lon, lat))
            .ok_or(GeoError::NoSubRegion { lon, lat })?;
        let (x, y) = region.proj.forward(lon, lat)?;
        finite_result(
            region.scale * x + region.shift.0 * self.radius,
            region.scale * y + region.shift.1 * self.radius,
        )
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), GeoError> {
        let x = finite_input(x)?;
        let y = finite_input(y)?;
        self.regions
            .iter()
            .find_map(|region| region.unplace(x, y, self.radius))
            .ok_or(GeoError::NoPlacement { x, y })
    }
}

/// Mercator projection on a sphere, with a documented latitude cutoff.
///
/// `x = R (lambda - lambda0)` and `y = R ln tan(pi/4 + phi/2)`. The domain is
/// `|lat| <= MERCATOR_MAX_LAT_DEG` (85 degrees). The poles and everything beyond the cutoff
/// are errors. The inverse accepts `|x| <= pi R` and `|y| <= y_cut`, where `y_cut` is the
/// forward `y` at the cutoff latitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mercator {
    radius: f64,
    lon0_deg: f64,
    /// `y` at the latitude cutoff, in units of the radius.
    y_max: f64,
}

impl Mercator {
    /// Builds a Mercator projection with origin longitude `origin_lon_deg`.
    ///
    /// # Errors
    ///
    /// Returns [`GeoError::InvalidParameter`] for a non-finite or non-positive `radius`, or a
    /// non-finite origin longitude.
    pub fn new(origin_lon_deg: f64, radius: f64) -> Result<Self, GeoError> {
        let radius = checked_radius(radius)?;
        let lon0_deg = checked_param("origin_lon", origin_lon_deg)?;
        let y_max = mercator_y(MERCATOR_MAX_LAT_DEG.to_radians());
        Ok(Self { radius, lon0_deg, y_max })
    }
}

impl Projection for Mercator {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> Result<(f64, f64), GeoError> {
        let lon = finite_input(lon_deg)?;
        let lat = within("latitude", finite_input(lat_deg)?, MERCATOR_MAX_LAT_DEG)?;
        let x = self.radius * wrap180(lon - self.lon0_deg).to_radians();
        let y = self.radius * mercator_y(lat.to_radians());
        finite_result(x, y)
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), GeoError> {
        let x_unit = within("x/R", finite_input(x)? / self.radius, PI + DOMAIN_TOL)?;
        let y_unit = within("y/R", finite_input(y)? / self.radius, self.y_max + DOMAIN_TOL)?;
        let lat = (2.0 * libm::exp(y_unit).atan() - FRAC_PI_2)
            .to_degrees()
            .clamp(-MERCATOR_MAX_LAT_DEG, MERCATOR_MAX_LAT_DEG);
        let lon = wrap180(self.lon0_deg + x_unit.to_degrees());
        finite_result(lon, lat)
    }
}

/// Equirectangular (plate carree) projection with a standard parallel.
///
/// `x = R (lambda - lambda0) cos(phi_s)` and `y = R phi`, where `phi_s` is the standard
/// parallel. The domain is all latitudes in `[-90, 90]` and longitudes wrapped into
/// `[-180, 180)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equirectangular {
    radius: f64,
    lon0_deg: f64,
    cos_std: f64,
}

impl Equirectangular {
    /// Builds an equirectangular projection.
    ///
    /// The standard parallel must satisfy `|phi_s| < 90`: at a pole `cos(phi_s)` is zero and
    /// the x scale would collapse.
    ///
    /// # Errors
    ///
    /// Returns [`GeoError::InvalidParameter`] for a non-finite or non-positive `radius`, a
    /// non-finite origin longitude, or a standard parallel that is non-finite or has
    /// `|phi_s| >= 90`.
    pub fn new(
        standard_parallel_deg: f64,
        origin_lon_deg: f64,
        radius: f64,
    ) -> Result<Self, GeoError> {
        let radius = checked_radius(radius)?;
        let lon0_deg = checked_param("origin_lon", origin_lon_deg)?;
        let std = standard_parallel_deg;
        if !(std.is_finite() && std.abs() < 90.0) {
            return Err(GeoError::InvalidParameter { name: "standard_parallel", value: std });
        }
        Ok(Self { radius, lon0_deg, cos_std: std.to_radians().cos() })
    }
}

impl Projection for Equirectangular {
    fn forward(&self, lon_deg: f64, lat_deg: f64) -> Result<(f64, f64), GeoError> {
        let lon = finite_input(lon_deg)?;
        let lat = within("latitude", finite_input(lat_deg)?, 90.0)?;
        let x = self.radius * wrap180(lon - self.lon0_deg).to_radians() * self.cos_std;
        finite_result(x, self.radius * lat.to_radians())
    }

    fn inverse(&self, x: f64, y: f64) -> Result<(f64, f64), GeoError> {
        let lat_unit = within("y/R", finite_input(y)? / self.radius, FRAC_PI_2 + DOMAIN_TOL)?;
        let lon_unit = within(
            "x/(R cos phi_s)",
            finite_input(x)? / (self.radius * self.cos_std),
            PI + DOMAIN_TOL,
        )?;
        let lat = lat_unit.clamp(-FRAC_PI_2, FRAC_PI_2).to_degrees();
        finite_result(wrap180(self.lon0_deg + lon_unit.to_degrees()), lat)
    }
}
