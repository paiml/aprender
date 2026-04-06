//! Type-safe physical units (Poka-Yoke).
//!
//! Implements Kennedy's dimensional analysis principles [28] to eliminate
//! dimensional errors at compile time. All physical quantities use newtype
//! wrappers from the `uom` crate.
//!
//! # Toyota Way Alignment
//!
//! - **Poka-Yoke (ポカヨケ)**: Mistake-proofing through design constraints
//! - Compile-time dimensional analysis prevents position + velocity errors
//!
//! # References
//!
//! [28] A. J. Kennedy, "Dimension Types," ESOP '94, LNCS vol. 788, pp. 348-362, 1994.

use serde::{Deserialize, Serialize};
use std::ops::{Add, Div, Mul, Neg, Sub};
use uom::si::acceleration::meter_per_second_squared;
use uom::si::f64::{Acceleration, Length, Mass, Time, Velocity};
use uom::si::length::meter;
use uom::si::mass::kilogram;
use uom::si::time::second;
use uom::si::velocity::meter_per_second;

/// Gravitational constant (m³ kg⁻¹ s⁻²).
pub const G: f64 = 6.674_30e-11;

/// Astronomical unit in meters.
pub const AU: f64 = 1.495_978_707e11;

/// Solar mass in kilograms.
pub const SOLAR_MASS: f64 = 1.988_92e30;

/// Earth mass in kilograms.
pub const EARTH_MASS: f64 = 5.972_2e24;

/// Type-safe 3D position vector with dimensional safety.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position3D {
    pub x: Length,
    pub y: Length,
    pub z: Length,
}

impl Position3D {
    /// Create a new position vector from meter values.
    #[must_use]
    pub fn from_meters(x: f64, y: f64, z: f64) -> Self {
        Self {
            x: Length::new::<meter>(x),
            y: Length::new::<meter>(y),
            z: Length::new::<meter>(z),
        }
    }

    /// Create a new position vector from AU values.
    #[must_use]
    pub fn from_au(x: f64, y: f64, z: f64) -> Self {
        Self::from_meters(x * AU, y * AU, z * AU)
    }

    /// Get the zero position.
    #[must_use]
    pub fn zero() -> Self {
        Self::from_meters(0.0, 0.0, 0.0)
    }

    /// Calculate the magnitude (distance from origin).
    #[must_use]
    pub fn magnitude(&self) -> Length {
        let x = self.x.get::<meter>();
        let y = self.y.get::<meter>();
        let z = self.z.get::<meter>();
        Length::new::<meter>((x * x + y * y + z * z).sqrt())
    }

    /// Calculate squared magnitude (avoids sqrt).
    #[must_use]
    pub fn magnitude_squared(&self) -> f64 {
        let x = self.x.get::<meter>();
        let y = self.y.get::<meter>();
        let z = self.z.get::<meter>();
        x * x + y * y + z * z
    }

    /// Normalize to unit vector (returns raw f64 components).
    #[must_use]
    pub fn normalize(&self) -> (f64, f64, f64) {
        let mag = self.magnitude().get::<meter>();
        if mag < f64::EPSILON {
            return (0.0, 0.0, 0.0);
        }
        (
            self.x.get::<meter>() / mag,
            self.y.get::<meter>() / mag,
            self.z.get::<meter>() / mag,
        )
    }

    /// Dot product (returns Length² as f64 in m²).
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x.get::<meter>() * other.x.get::<meter>()
            + self.y.get::<meter>() * other.y.get::<meter>()
            + self.z.get::<meter>() * other.z.get::<meter>()
    }

    /// Cross product.
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        let ax = self.x.get::<meter>();
        let ay = self.y.get::<meter>();
        let az = self.z.get::<meter>();
        let bx = other.x.get::<meter>();
        let by = other.y.get::<meter>();
        let bz = other.z.get::<meter>();

        Self::from_meters(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)
    }

    /// Scale by a dimensionless factor.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }

    /// Check if all components are finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x.get::<meter>().is_finite()
            && self.y.get::<meter>().is_finite()
            && self.z.get::<meter>().is_finite()
    }

    /// Get raw meter values as tuple.
    #[must_use]
    pub fn as_meters(&self) -> (f64, f64, f64) {
        (
            self.x.get::<meter>(),
            self.y.get::<meter>(),
            self.z.get::<meter>(),
        )
    }
}

impl Add for Position3D {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Position3D {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Neg for Position3D {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Type-safe 3D velocity vector with dimensional safety.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Velocity3D {
    pub x: Velocity,
    pub y: Velocity,
    pub z: Velocity,
}

impl Velocity3D {
    /// Create a new velocity vector from m/s values.
    #[must_use]
    pub fn from_mps(x: f64, y: f64, z: f64) -> Self {
        Self {
            x: Velocity::new::<meter_per_second>(x),
            y: Velocity::new::<meter_per_second>(y),
            z: Velocity::new::<meter_per_second>(z),
        }
    }

    /// Get the zero velocity.
    #[must_use]
    pub fn zero() -> Self {
        Self::from_mps(0.0, 0.0, 0.0)
    }

    /// Calculate the magnitude (speed).
    #[must_use]
    pub fn magnitude(&self) -> Velocity {
        let x = self.x.get::<meter_per_second>();
        let y = self.y.get::<meter_per_second>();
        let z = self.z.get::<meter_per_second>();
        Velocity::new::<meter_per_second>((x * x + y * y + z * z).sqrt())
    }

    /// Calculate squared magnitude (avoids sqrt).
    #[must_use]
    pub fn magnitude_squared(&self) -> f64 {
        let x = self.x.get::<meter_per_second>();
        let y = self.y.get::<meter_per_second>();
        let z = self.z.get::<meter_per_second>();
        x * x + y * y + z * z
    }

    /// Scale by a dimensionless factor.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }

    /// Check if all components are finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x.get::<meter_per_second>().is_finite()
            && self.y.get::<meter_per_second>().is_finite()
            && self.z.get::<meter_per_second>().is_finite()
    }

    /// Get raw m/s values as tuple.
    #[must_use]
    pub fn as_mps(&self) -> (f64, f64, f64) {
        (
            self.x.get::<meter_per_second>(),
            self.y.get::<meter_per_second>(),
            self.z.get::<meter_per_second>(),
        )
    }
}

impl Add for Velocity3D {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Velocity3D {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Neg for Velocity3D {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Type-safe 3D acceleration vector with dimensional safety.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Acceleration3D {
    pub x: Acceleration,
    pub y: Acceleration,
    pub z: Acceleration,
}

impl Acceleration3D {
    /// Create a new acceleration vector from m/s² values.
    #[must_use]
    pub fn from_mps2(x: f64, y: f64, z: f64) -> Self {
        Self {
            x: Acceleration::new::<meter_per_second_squared>(x),
            y: Acceleration::new::<meter_per_second_squared>(y),
            z: Acceleration::new::<meter_per_second_squared>(z),
        }
    }

    /// Get the zero acceleration.
    #[must_use]
    pub fn zero() -> Self {
        Self::from_mps2(0.0, 0.0, 0.0)
    }

    /// Calculate the magnitude.
    #[must_use]
    pub fn magnitude(&self) -> Acceleration {
        let x = self.x.get::<meter_per_second_squared>();
        let y = self.y.get::<meter_per_second_squared>();
        let z = self.z.get::<meter_per_second_squared>();
        Acceleration::new::<meter_per_second_squared>((x * x + y * y + z * z).sqrt())
    }

    /// Scale by a dimensionless factor.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }

    /// Get raw m/s² values as tuple.
    #[must_use]
    pub fn as_mps2(&self) -> (f64, f64, f64) {
        (
            self.x.get::<meter_per_second_squared>(),
            self.y.get::<meter_per_second_squared>(),
            self.z.get::<meter_per_second_squared>(),
        )
    }
}

impl Add for Acceleration3D {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Neg for Acceleration3D {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Type-safe mass wrapper.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct OrbitMass(pub Mass);

impl OrbitMass {
    /// Create from kilograms.
    #[must_use]
    pub fn from_kg(kg: f64) -> Self {
        Self(Mass::new::<kilogram>(kg))
    }

    /// Create from solar masses.
    #[must_use]
    pub fn from_solar_masses(m: f64) -> Self {
        Self::from_kg(m * SOLAR_MASS)
    }

    /// Create from earth masses.
    #[must_use]
    pub fn from_earth_masses(m: f64) -> Self {
        Self::from_kg(m * EARTH_MASS)
    }

    /// Get value in kilograms.
    #[must_use]
    pub fn as_kg(&self) -> f64 {
        self.0.get::<kilogram>()
    }
}

/// Type-safe time wrapper.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct OrbitTime(pub Time);

impl OrbitTime {
    /// Create from seconds.
    #[must_use]
    pub fn from_seconds(s: f64) -> Self {
        Self(Time::new::<second>(s))
    }

    /// Create from days.
    #[must_use]
    pub fn from_days(d: f64) -> Self {
        Self::from_seconds(d * 86400.0)
    }

    /// Create from years (Julian years).
    #[must_use]
    pub fn from_years(y: f64) -> Self {
        Self::from_days(y * 365.25)
    }

    /// Get value in seconds.
    #[must_use]
    pub fn as_seconds(&self) -> f64 {
        self.0.get::<second>()
    }

    /// Get value in days.
    #[must_use]
    pub fn as_days(&self) -> f64 {
        self.as_seconds() / 86400.0
    }
}

/// Velocity = Position / Time (dimensional operation).
impl Div<OrbitTime> for Position3D {
    type Output = Velocity3D;

    fn div(self, dt: OrbitTime) -> Velocity3D {
        let t = dt.as_seconds();
        Velocity3D::from_mps(
            self.x.get::<meter>() / t,
            self.y.get::<meter>() / t,
            self.z.get::<meter>() / t,
        )
    }
}

/// Position = Velocity * Time (dimensional operation).
impl Mul<OrbitTime> for Velocity3D {
    type Output = Position3D;

    fn mul(self, dt: OrbitTime) -> Position3D {
        let t = dt.as_seconds();
        Position3D::from_meters(
            self.x.get::<meter_per_second>() * t,
            self.y.get::<meter_per_second>() * t,
            self.z.get::<meter_per_second>() * t,
        )
    }
}

/// Velocity = Acceleration * Time (dimensional operation).
impl Mul<OrbitTime> for Acceleration3D {
    type Output = Velocity3D;

    fn mul(self, dt: OrbitTime) -> Velocity3D {
        let t = dt.as_seconds();
        Velocity3D::from_mps(
            self.x.get::<meter_per_second_squared>() * t,
            self.y.get::<meter_per_second_squared>() * t,
            self.z.get::<meter_per_second_squared>() * t,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_position_from_meters() {
        let pos = Position3D::from_meters(1.0, 2.0, 3.0);
        let (x, y, z) = pos.as_meters();
        assert!((x - 1.0).abs() < EPSILON);
        assert!((y - 2.0).abs() < EPSILON);
        assert!((z - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_position_from_au() {
        let pos = Position3D::from_au(1.0, 0.0, 0.0);
        let (x, _, _) = pos.as_meters();
        assert!((x - AU).abs() < 1.0); // 1 meter tolerance
    }

    #[test]
    fn test_position_magnitude() {
        let pos = Position3D::from_meters(3.0, 4.0, 0.0);
        let mag = pos.magnitude().get::<meter>();
        assert!((mag - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_position_normalize() {
        let pos = Position3D::from_meters(3.0, 4.0, 0.0);
        let (nx, ny, nz) = pos.normalize();
        assert!((nx - 0.6).abs() < EPSILON);
        assert!((ny - 0.8).abs() < EPSILON);
        assert!(nz.abs() < EPSILON);
    }

    #[test]
    fn test_position_dot_product() {
        let a = Position3D::from_meters(1.0, 2.0, 3.0);
        let b = Position3D::from_meters(4.0, 5.0, 6.0);
        let dot = a.dot(&b);
        assert!((dot - 32.0).abs() < EPSILON); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_position_cross_product() {
        let a = Position3D::from_meters(1.0, 0.0, 0.0);
        let b = Position3D::from_meters(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        let (x, y, z) = c.as_meters();
        assert!(x.abs() < EPSILON);
        assert!(y.abs() < EPSILON);
        assert!((z - 1.0).abs() < EPSILON); // i × j = k
    }

    #[test]
    fn test_position_add_sub() {
        let a = Position3D::from_meters(1.0, 2.0, 3.0);
        let b = Position3D::from_meters(4.0, 5.0, 6.0);
        let sum = a + b;
        let (x, y, z) = sum.as_meters();
        assert!((x - 5.0).abs() < EPSILON);
        assert!((y - 7.0).abs() < EPSILON);
        assert!((z - 9.0).abs() < EPSILON);
    }

    #[test]
    fn test_velocity_from_mps() {
        let vel = Velocity3D::from_mps(10.0, 20.0, 30.0);
        let (x, y, z) = vel.as_mps();
        assert!((x - 10.0).abs() < EPSILON);
        assert!((y - 20.0).abs() < EPSILON);
        assert!((z - 30.0).abs() < EPSILON);
    }

    #[test]
    fn test_velocity_magnitude() {
        let vel = Velocity3D::from_mps(3.0, 4.0, 0.0);
        let mag = vel.magnitude().get::<meter_per_second>();
        assert!((mag - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_acceleration_from_mps2() {
        let acc = Acceleration3D::from_mps2(1.0, 2.0, 3.0);
        let (x, y, z) = acc.as_mps2();
        assert!((x - 1.0).abs() < EPSILON);
        assert!((y - 2.0).abs() < EPSILON);
        assert!((z - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_mass_from_kg() {
        let m = OrbitMass::from_kg(1000.0);
        assert!((m.as_kg() - 1000.0).abs() < EPSILON);
    }

    #[test]
    fn test_mass_from_solar_masses() {
        let m = OrbitMass::from_solar_masses(1.0);
        assert!((m.as_kg() - SOLAR_MASS).abs() < 1e20);
    }

    #[test]
    fn test_time_from_seconds() {
        let t = OrbitTime::from_seconds(3600.0);
        assert!((t.as_seconds() - 3600.0).abs() < EPSILON);
    }

    #[test]
    fn test_time_from_days() {
        let t = OrbitTime::from_days(1.0);
        assert!((t.as_seconds() - 86400.0).abs() < EPSILON);
    }

    #[test]
    fn test_time_from_years() {
        let t = OrbitTime::from_years(1.0);
        let expected = 365.25 * 86400.0;
        assert!((t.as_seconds() - expected).abs() < 1.0);
    }

    #[test]
    fn test_position_div_time_gives_velocity() {
        let pos = Position3D::from_meters(1000.0, 2000.0, 3000.0);
        let dt = OrbitTime::from_seconds(10.0);
        let vel = pos / dt;
        let (vx, vy, vz) = vel.as_mps();
        assert!((vx - 100.0).abs() < EPSILON);
        assert!((vy - 200.0).abs() < EPSILON);
        assert!((vz - 300.0).abs() < EPSILON);
    }

    #[test]
    fn test_velocity_mul_time_gives_position() {
        let vel = Velocity3D::from_mps(100.0, 200.0, 300.0);
        let dt = OrbitTime::from_seconds(10.0);
        let pos = vel * dt;
        let (x, y, z) = pos.as_meters();
        assert!((x - 1000.0).abs() < EPSILON);
        assert!((y - 2000.0).abs() < EPSILON);
        assert!((z - 3000.0).abs() < EPSILON);
    }

    #[test]
    fn test_acceleration_mul_time_gives_velocity() {
        let acc = Acceleration3D::from_mps2(10.0, 20.0, 30.0);
        let dt = OrbitTime::from_seconds(5.0);
        let vel = acc * dt;
        let (vx, vy, vz) = vel.as_mps();
        assert!((vx - 50.0).abs() < EPSILON);
        assert!((vy - 100.0).abs() < EPSILON);
        assert!((vz - 150.0).abs() < EPSILON);
    }

    #[test]
    fn test_is_finite() {
        let pos = Position3D::from_meters(1.0, 2.0, 3.0);
        assert!(pos.is_finite());

        let vel = Velocity3D::from_mps(1.0, 2.0, 3.0);
        assert!(vel.is_finite());
    }

    #[test]
    fn test_gravitational_constant() {
        assert!((G - 6.674_30e-11).abs() < 1e-15);
    }

    #[test]
    fn test_au_constant() {
        assert!((AU - 1.495_978_707e11).abs() < 1.0);
    }

    #[test]
    fn test_position_scale() {
        let pos = Position3D::from_meters(1.0, 2.0, 3.0);
        let scaled = pos.scale(2.0);
        let (x, y, z) = scaled.as_meters();
        assert!((x - 2.0).abs() < EPSILON);
        assert!((y - 4.0).abs() < EPSILON);
        assert!((z - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_position_magnitude_squared() {
        let pos = Position3D::from_meters(3.0, 4.0, 0.0);
        let mag_sq = pos.magnitude_squared();
        assert!((mag_sq - 25.0).abs() < EPSILON);
    }

    #[test]
    fn test_position_zero() {
        let pos = Position3D::zero();
        let (x, y, z) = pos.as_meters();
        assert!(x.abs() < EPSILON);
        assert!(y.abs() < EPSILON);
        assert!(z.abs() < EPSILON);
    }

    #[test]
    fn test_position_normalize_zero() {
        let pos = Position3D::zero();
        let (nx, ny, nz) = pos.normalize();
        assert!(nx.abs() < EPSILON);
        assert!(ny.abs() < EPSILON);
        assert!(nz.abs() < EPSILON);
    }

    #[test]
    fn test_position_sub() {
        let a = Position3D::from_meters(5.0, 7.0, 9.0);
        let b = Position3D::from_meters(1.0, 2.0, 3.0);
        let diff = a - b;
        let (x, y, z) = diff.as_meters();
        assert!((x - 4.0).abs() < EPSILON);
        assert!((y - 5.0).abs() < EPSILON);
        assert!((z - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_position_neg() {
        let pos = Position3D::from_meters(1.0, 2.0, 3.0);
        let neg = -pos;
        let (x, y, z) = neg.as_meters();
        assert!((x + 1.0).abs() < EPSILON);
        assert!((y + 2.0).abs() < EPSILON);
        assert!((z + 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_position_is_finite_with_nan() {
        let mut pos = Position3D::from_meters(1.0, 2.0, 3.0);
        assert!(pos.is_finite());
        pos.x = Length::new::<meter>(f64::NAN);
        assert!(!pos.is_finite());
    }

    #[test]
    fn test_velocity_zero() {
        let vel = Velocity3D::zero();
        let (x, y, z) = vel.as_mps();
        assert!(x.abs() < EPSILON);
        assert!(y.abs() < EPSILON);
        assert!(z.abs() < EPSILON);
    }

    #[test]
    fn test_velocity_scale() {
        let vel = Velocity3D::from_mps(1.0, 2.0, 3.0);
        let scaled = vel.scale(3.0);
        let (x, y, z) = scaled.as_mps();
        assert!((x - 3.0).abs() < EPSILON);
        assert!((y - 6.0).abs() < EPSILON);
        assert!((z - 9.0).abs() < EPSILON);
    }

    #[test]
    fn test_velocity_magnitude_squared() {
        let vel = Velocity3D::from_mps(3.0, 4.0, 0.0);
        let mag_sq = vel.magnitude_squared();
        assert!((mag_sq - 25.0).abs() < EPSILON);
    }

    #[test]
    fn test_velocity_add() {
        let a = Velocity3D::from_mps(1.0, 2.0, 3.0);
        let b = Velocity3D::from_mps(4.0, 5.0, 6.0);
        let sum = a + b;
        let (x, y, z) = sum.as_mps();
        assert!((x - 5.0).abs() < EPSILON);
        assert!((y - 7.0).abs() < EPSILON);
        assert!((z - 9.0).abs() < EPSILON);
    }

    #[test]
    fn test_velocity_sub() {
        let a = Velocity3D::from_mps(5.0, 7.0, 9.0);
        let b = Velocity3D::from_mps(1.0, 2.0, 3.0);
        let diff = a - b;
        let (x, y, z) = diff.as_mps();
        assert!((x - 4.0).abs() < EPSILON);
        assert!((y - 5.0).abs() < EPSILON);
        assert!((z - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_velocity_neg() {
        let vel = Velocity3D::from_mps(1.0, 2.0, 3.0);
        let neg = -vel;
        let (x, y, z) = neg.as_mps();
        assert!((x + 1.0).abs() < EPSILON);
        assert!((y + 2.0).abs() < EPSILON);
        assert!((z + 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_velocity_is_finite_with_inf() {
        let mut vel = Velocity3D::from_mps(1.0, 2.0, 3.0);
        assert!(vel.is_finite());
        vel.y = Velocity::new::<meter_per_second>(f64::INFINITY);
        assert!(!vel.is_finite());
    }

    #[test]
    fn test_acceleration_zero() {
        let acc = Acceleration3D::zero();
        let (x, y, z) = acc.as_mps2();
        assert!(x.abs() < EPSILON);
        assert!(y.abs() < EPSILON);
        assert!(z.abs() < EPSILON);
    }

    #[test]
    fn test_acceleration_scale() {
        let acc = Acceleration3D::from_mps2(1.0, 2.0, 3.0);
        let scaled = acc.scale(2.0);
        let (x, y, z) = scaled.as_mps2();
        assert!((x - 2.0).abs() < EPSILON);
        assert!((y - 4.0).abs() < EPSILON);
        assert!((z - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_acceleration_magnitude() {
        let acc = Acceleration3D::from_mps2(3.0, 4.0, 0.0);
        let mag = acc.magnitude().get::<meter_per_second_squared>();
        assert!((mag - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_acceleration_add() {
        let a = Acceleration3D::from_mps2(1.0, 2.0, 3.0);
        let b = Acceleration3D::from_mps2(4.0, 5.0, 6.0);
        let sum = a + b;
        let (x, y, z) = sum.as_mps2();
        assert!((x - 5.0).abs() < EPSILON);
        assert!((y - 7.0).abs() < EPSILON);
        assert!((z - 9.0).abs() < EPSILON);
    }

    #[test]
    fn test_acceleration_neg() {
        let acc = Acceleration3D::from_mps2(1.0, 2.0, 3.0);
        let neg = -acc;
        let (x, y, z) = neg.as_mps2();
        assert!((x + 1.0).abs() < EPSILON);
        assert!((y + 2.0).abs() < EPSILON);
        assert!((z + 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_mass_from_earth_masses() {
        let m = OrbitMass::from_earth_masses(1.0);
        assert!((m.as_kg() - EARTH_MASS).abs() < 1e18);
    }

    #[test]
    fn test_time_as_days() {
        let t = OrbitTime::from_seconds(172800.0); // 2 days
        assert!((t.as_days() - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_constants() {
        // Sanity-check physical constants at compile time
        const _: () = assert!(SOLAR_MASS > 1.9e30);
        const _: () = assert!(EARTH_MASS > 5.9e24);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Position conversion round-trip preserves value.
        #[test]
        fn prop_position_roundtrip(
            meters in 1e3f64..1e15,
        ) {
            let p = Position3D::from_meters(meters, 0.0, 0.0);
            let (x, _, _) = p.as_meters();
            prop_assert!((x - meters).abs() / meters < 1e-10);
        }

        /// AU to meters conversion is consistent.
        #[test]
        fn prop_au_conversion(
            au in 0.1f64..100.0,
        ) {
            let p = Position3D::from_au(au, 0.0, 0.0);
            let (x, _, _) = p.as_meters();
            let expected = au * AU;
            prop_assert!((x - expected).abs() / expected < 1e-10);
        }

        /// Velocity magnitude is non-negative.
        #[test]
        fn prop_velocity_magnitude_nonneg(
            vx in -1e5f64..1e5,
            vy in -1e5f64..1e5,
            vz in -1e5f64..1e5,
        ) {
            let v = Velocity3D::from_mps(vx, vy, vz);
            let mag = v.magnitude().get::<meter_per_second>();
            prop_assert!(mag >= 0.0);
        }

        /// Position magnitude is non-negative.
        #[test]
        fn prop_position_magnitude_nonneg(
            x in -1e12f64..1e12,
            y in -1e12f64..1e12,
            z in -1e12f64..1e12,
        ) {
            let p = Position3D::from_meters(x, y, z);
            let mag = p.magnitude().get::<meter>();
            prop_assert!(mag >= 0.0);
        }

        /// Mass is always positive.
        #[test]
        fn prop_mass_positive(
            kg in 1.0f64..1e35,
        ) {
            let m = OrbitMass::from_kg(kg);
            prop_assert!(m.as_kg() > 0.0);
        }

        /// Time conversion round-trip.
        #[test]
        fn prop_time_roundtrip(
            seconds in 1.0f64..1e10,
        ) {
            let t = OrbitTime::from_seconds(seconds);
            let recovered = t.as_seconds();
            prop_assert!((recovered - seconds).abs() / seconds < 1e-10);
        }

        /// Negation preserves magnitude.
        #[test]
        fn prop_velocity_neg_magnitude(
            vx in -1e5f64..1e5,
            vy in -1e5f64..1e5,
            vz in -1e5f64..1e5,
        ) {
            let v = Velocity3D::from_mps(vx, vy, vz);
            let neg_v = -v;
            let mag1 = v.magnitude().get::<meter_per_second>();
            let mag2 = neg_v.magnitude().get::<meter_per_second>();
            prop_assert!((mag1 - mag2).abs() < 1e-10);
        }
    }
}
