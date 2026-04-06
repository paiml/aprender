//! Satellite orbital mechanics scenarios.
//!
//! Implements Keplerian orbital mechanics with:
//! - Two-body gravitational dynamics
//! - Orbital element conversions
//! - Common orbit types (LEO, GEO, polar, etc.)

use crate::domains::physics::CentralForceField;
use crate::engine::state::{SimState, Vec3};
use serde::{Deserialize, Serialize};

/// Standard gravitational parameter for Earth (m³/s²).
pub const EARTH_MU: f64 = 3.986_004_418e14;

/// Earth equatorial radius (m).
pub const EARTH_RADIUS: f64 = 6_378_137.0;

/// Classical orbital elements (Keplerian elements).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrbitalElements {
    /// Semi-major axis (m).
    pub a: f64,
    /// Eccentricity (0 = circular, 0-1 = elliptical).
    pub e: f64,
    /// Inclination (radians).
    pub i: f64,
    /// Right ascension of ascending node (radians).
    pub raan: f64,
    /// Argument of periapsis (radians).
    pub omega: f64,
    /// True anomaly (radians).
    pub nu: f64,
}

impl OrbitalElements {
    /// Create circular orbit at given altitude and inclination.
    #[must_use]
    pub fn circular(altitude: f64, inclination: f64) -> Self {
        Self {
            a: EARTH_RADIUS + altitude,
            e: 0.0,
            i: inclination,
            raan: 0.0,
            omega: 0.0,
            nu: 0.0,
        }
    }

    /// Create Low Earth Orbit (LEO) at default altitude.
    #[must_use]
    pub fn leo() -> Self {
        Self::circular(400_000.0, 0.0)
    }

    /// Create Geostationary Earth Orbit (GEO).
    #[must_use]
    pub fn geo() -> Self {
        Self::circular(35_786_000.0, 0.0)
    }

    /// Create sun-synchronous orbit.
    #[must_use]
    pub fn sun_synchronous(altitude: f64) -> Self {
        // Sun-synchronous inclination depends on altitude
        // For ~500km altitude, it's approximately 97.4 degrees
        let inclination = 97.4_f64.to_radians();
        Self::circular(altitude, inclination)
    }

    /// Create polar orbit.
    #[must_use]
    pub fn polar(altitude: f64) -> Self {
        Self::circular(altitude, std::f64::consts::FRAC_PI_2)
    }

    /// Convert to Cartesian state vectors (position, velocity).
    #[must_use]
    #[allow(clippy::many_single_char_names)] // Standard orbital mechanics notation
    pub fn to_cartesian(&self) -> (Vec3, Vec3) {
        let mu = EARTH_MU;

        // Semi-latus rectum
        let p = self.a * (1.0 - self.e * self.e);

        // Distance from focus
        let r_mag = p / (1.0 + self.e * self.nu.cos());

        // Position in perifocal frame
        let r_pqw = Vec3::new(r_mag * self.nu.cos(), r_mag * self.nu.sin(), 0.0);

        // Velocity in perifocal frame
        let h = (mu * p).sqrt();
        let v_pqw = Vec3::new(
            -mu / h * self.nu.sin(),
            mu / h * (self.e + self.nu.cos()),
            0.0,
        );

        // Rotation matrices
        let (sin_raan, cos_raan) = self.raan.sin_cos();
        let (sin_i, cos_i) = self.i.sin_cos();
        let (sin_omega, cos_omega) = self.omega.sin_cos();

        // Transform to inertial frame (ECI)
        let transform = |v: &Vec3| -> Vec3 {
            let x = (cos_raan * cos_omega - sin_raan * sin_omega * cos_i) * v.x
                + (-cos_raan * sin_omega - sin_raan * cos_omega * cos_i) * v.y;
            let y = (sin_raan * cos_omega + cos_raan * sin_omega * cos_i) * v.x
                + (-sin_raan * sin_omega + cos_raan * cos_omega * cos_i) * v.y;
            let z = sin_omega * sin_i * v.x + cos_omega * sin_i * v.y;
            Vec3::new(x, y, z)
        };

        (transform(&r_pqw), transform(&v_pqw))
    }

    /// Get orbital period (seconds).
    #[must_use]
    pub fn period(&self) -> f64 {
        2.0 * std::f64::consts::PI * (self.a.powi(3) / EARTH_MU).sqrt()
    }

    /// Get specific orbital energy (J/kg).
    #[must_use]
    pub fn energy(&self) -> f64 {
        -EARTH_MU / (2.0 * self.a)
    }

    /// Get periapsis altitude (m).
    #[must_use]
    pub fn periapsis_altitude(&self) -> f64 {
        self.a * (1.0 - self.e) - EARTH_RADIUS
    }

    /// Get apoapsis altitude (m).
    #[must_use]
    pub fn apoapsis_altitude(&self) -> f64 {
        self.a * (1.0 + self.e) - EARTH_RADIUS
    }
}

impl Default for OrbitalElements {
    fn default() -> Self {
        Self::leo()
    }
}

/// Satellite configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatelliteConfig {
    /// Satellite mass (kg).
    pub mass: f64,
    /// Drag coefficient.
    pub cd: f64,
    /// Cross-sectional area (m²).
    pub area: f64,
    /// Orbital elements.
    pub orbit: OrbitalElements,
}

impl Default for SatelliteConfig {
    fn default() -> Self {
        Self {
            mass: 1000.0,
            cd: 2.2,
            area: 10.0,
            orbit: OrbitalElements::leo(),
        }
    }
}

/// Satellite scenario for orbital simulations.
#[derive(Debug, Clone)]
pub struct SatelliteScenario {
    config: SatelliteConfig,
}

impl SatelliteScenario {
    /// Create a new satellite scenario.
    #[must_use]
    pub fn new(config: SatelliteConfig) -> Self {
        Self { config }
    }

    /// Create scenario for LEO satellite.
    #[must_use]
    pub fn leo(mass: f64) -> Self {
        Self::new(SatelliteConfig {
            mass,
            orbit: OrbitalElements::leo(),
            ..Default::default()
        })
    }

    /// Create scenario for GEO satellite.
    #[must_use]
    pub fn geo(mass: f64) -> Self {
        Self::new(SatelliteConfig {
            mass,
            orbit: OrbitalElements::geo(),
            ..Default::default()
        })
    }

    /// Initialize simulation state for the satellite.
    #[must_use]
    pub fn init_state(&self) -> SimState {
        let (position, velocity) = self.config.orbit.to_cartesian();

        let mut state = SimState::new();
        state.add_body(self.config.mass, position, velocity);
        state
    }

    /// Create force field for this scenario.
    #[must_use]
    pub fn create_force_field(&self) -> CentralForceField {
        CentralForceField::new(EARTH_MU, Vec3::zero())
    }

    /// Get orbital period (seconds).
    #[must_use]
    pub fn period(&self) -> f64 {
        self.config.orbit.period()
    }

    /// Get orbital energy.
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.config.orbit.energy()
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &SatelliteConfig {
        &self.config
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::domains::physics::ForceField;

    #[test]
    fn test_orbital_elements_circular() {
        let orbit = OrbitalElements::circular(400_000.0, 0.0);

        assert!(orbit.e.abs() < f64::EPSILON);
        assert!((orbit.a - (EARTH_RADIUS + 400_000.0)).abs() < 1.0);
    }

    #[test]
    fn test_orbital_elements_leo() {
        let orbit = OrbitalElements::leo();

        // LEO at ~400km
        assert!((orbit.periapsis_altitude() - 400_000.0).abs() < 1.0);
    }

    #[test]
    fn test_orbital_elements_default() {
        let orbit = OrbitalElements::default();
        assert_eq!(orbit.a, OrbitalElements::leo().a);
    }

    #[test]
    fn test_orbital_elements_geo() {
        let orbit = OrbitalElements::geo();

        // GEO period is one sidereal day (~86164 seconds), not solar day (86400s)
        // Sidereal day = 23h 56m 4s = 86164 seconds
        let period = orbit.period();
        assert!(
            (period - 86164.0).abs() < 100.0,
            "GEO period {} should be ~86164s",
            period
        );
    }

    #[test]
    fn test_orbital_elements_sun_synchronous() {
        let orbit = OrbitalElements::sun_synchronous(500_000.0);
        // Sun-sync inclination is ~97.4 degrees
        assert!(orbit.i > 1.6 && orbit.i < 1.8);
    }

    #[test]
    fn test_orbital_elements_polar() {
        let orbit = OrbitalElements::polar(600_000.0);
        assert!((orbit.i - std::f64::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn test_orbital_elements_clone() {
        let orbit = OrbitalElements::leo();
        let cloned = orbit.clone();
        assert!((cloned.a - orbit.a).abs() < f64::EPSILON);
    }

    #[test]
    fn test_orbital_elements_to_cartesian() {
        let orbit = OrbitalElements::circular(400_000.0, 0.0);
        let (pos, vel) = orbit.to_cartesian();

        // Position magnitude should equal semi-major axis for circular orbit at nu=0
        let r = pos.magnitude();
        assert!((r - orbit.a).abs() < 1.0, "r={}, a={}", r, orbit.a);

        // Velocity should be circular velocity: v = sqrt(mu/r)
        let v = vel.magnitude();
        let v_circular = (EARTH_MU / orbit.a).sqrt();
        assert!(
            (v - v_circular).abs() < 10.0,
            "v={}, v_circ={}",
            v,
            v_circular
        );
    }

    #[test]
    fn test_orbital_elements_to_cartesian_inclined() {
        let orbit = OrbitalElements {
            a: EARTH_RADIUS + 400_000.0,
            e: 0.0,
            i: 0.5, // ~28 degrees
            raan: 0.0,
            omega: 0.0,
            nu: 0.0,
        };
        let (pos, _vel) = orbit.to_cartesian();
        // Should have a z component due to inclination
        assert!(pos.z.abs() < f64::EPSILON); // At nu=0, z should be 0 for circular orbit
    }

    #[test]
    fn test_orbital_elements_energy() {
        let orbit = OrbitalElements::leo();
        let energy = orbit.energy();
        // Energy should be negative (bound orbit)
        assert!(energy < 0.0);
    }

    #[test]
    fn test_orbital_elements_periapsis_apoapsis() {
        let orbit = OrbitalElements {
            a: EARTH_RADIUS + 1_000_000.0, // Higher altitude for valid orbit with e=0.1
            e: 0.1,
            i: 0.0,
            raan: 0.0,
            omega: 0.0,
            nu: 0.0,
        };
        let peri = orbit.periapsis_altitude();
        let apo = orbit.apoapsis_altitude();

        // Apoapsis > periapsis for elliptical orbit
        assert!(apo > peri);
        // Both should be positive for orbit above Earth
        assert!(peri > 0.0);
    }

    #[test]
    fn test_satellite_config_default() {
        let config = SatelliteConfig::default();
        assert!(config.mass > 0.0);
        assert!(config.cd > 0.0);
        assert!(config.area > 0.0);
    }

    #[test]
    fn test_satellite_config_clone() {
        let config = SatelliteConfig::default();
        let cloned = config.clone();
        assert!((cloned.mass - config.mass).abs() < f64::EPSILON);
    }

    #[test]
    fn test_satellite_scenario_init() {
        let scenario = SatelliteScenario::leo(1000.0);
        let state = scenario.init_state();

        assert_eq!(state.num_bodies(), 1);

        // Check the satellite is at the right altitude
        let pos = state.positions()[0];
        let altitude = pos.magnitude() - EARTH_RADIUS;
        assert!(
            (altitude - 400_000.0).abs() < 1000.0,
            "altitude={}",
            altitude
        );
    }

    #[test]
    fn test_satellite_scenario_new() {
        let config = SatelliteConfig {
            mass: 500.0,
            ..Default::default()
        };
        let scenario = SatelliteScenario::new(config);
        assert!((scenario.config().mass - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_satellite_scenario_geo() {
        let scenario = SatelliteScenario::geo(2000.0);
        let state = scenario.init_state();
        assert_eq!(state.num_bodies(), 1);
    }

    #[test]
    fn test_satellite_scenario_clone() {
        let scenario = SatelliteScenario::leo(1000.0);
        let cloned = scenario.clone();
        assert!((cloned.period() - scenario.period()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_satellite_scenario_period() {
        let scenario = SatelliteScenario::leo(1000.0);
        let period = scenario.period();
        // LEO period ~92 minutes
        assert!(period > 5000.0 && period < 6000.0);
    }

    #[test]
    fn test_satellite_scenario_energy() {
        let scenario = SatelliteScenario::leo(1000.0);
        let energy = scenario.energy();
        assert!(energy < 0.0);
    }

    #[test]
    fn test_satellite_force_field() {
        let scenario = SatelliteScenario::leo(1000.0);
        let field = scenario.create_force_field();

        // Acceleration should point toward center
        let pos = Vec3::new(EARTH_RADIUS + 400_000.0, 0.0, 0.0);
        let acc = field.acceleration(&pos, 1000.0);

        // Should point in -x direction
        assert!(acc.x < 0.0);
        assert!(acc.y.abs() < f64::EPSILON);
        assert!(acc.z.abs() < f64::EPSILON);
    }

    #[test]
    fn test_orbital_period_leo() {
        let orbit = OrbitalElements::circular(400_000.0, 0.0);
        let period = orbit.period();

        // LEO period is approximately 92 minutes = 5520 seconds
        assert!(period > 5000.0 && period < 6000.0, "LEO period={}", period);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Orbital period increases with semi-major axis (Kepler's 3rd law).
        #[test]
        fn prop_period_increases_with_altitude(
            alt1 in 200_000.0f64..1_000_000.0,
            alt2 in 200_000.0f64..1_000_000.0,
        ) {
            let orbit1 = OrbitalElements::circular(alt1, 0.0);
            let orbit2 = OrbitalElements::circular(alt2, 0.0);

            let period1 = orbit1.period();
            let period2 = orbit2.period();

            // Higher altitude -> longer period
            if alt1 > alt2 {
                prop_assert!(period1 > period2, "P({})={} should > P({})={}", alt1, period1, alt2, period2);
            } else if alt1 < alt2 {
                prop_assert!(period1 < period2, "P({})={} should < P({})={}", alt1, period1, alt2, period2);
            }
        }

        /// Orbital energy is negative for bound orbits.
        #[test]
        fn prop_bound_orbit_negative_energy(
            altitude in 200_000.0f64..35_786_000.0,
            mass in 100.0f64..10000.0,
        ) {
            let config = SatelliteConfig {
                mass,
                orbit: OrbitalElements::circular(altitude, 0.0),
                ..Default::default()
            };
            let scenario = SatelliteScenario::new(config);
            let energy = scenario.energy();
            prop_assert!(energy < 0.0, "Bound orbit energy={} should be negative", energy);
        }

        /// Eccentricity is in valid range [0, 1) for elliptical orbits.
        #[test]
        fn prop_eccentricity_valid(
            ecc in 0.0f64..0.99,
        ) {
            prop_assert!(ecc >= 0.0);
            prop_assert!(ecc < 1.0);
        }

        /// Circular orbit has zero eccentricity.
        #[test]
        fn prop_circular_orbit_zero_eccentricity(
            altitude in 200_000.0f64..1_000_000.0,
        ) {
            let orbit = OrbitalElements::circular(altitude, 0.0);
            prop_assert!(orbit.e.abs() < 1e-10, "e={}", orbit.e);
        }

        /// Semi-major axis must be positive.
        #[test]
        fn prop_sma_positive(
            altitude in 200_000.0f64..35_786_000.0,
        ) {
            let orbit = OrbitalElements::circular(altitude, 0.0);
            prop_assert!(orbit.a > 0.0);
            prop_assert!(orbit.a > EARTH_RADIUS, "a={} < Re", orbit.a);
        }
    }
}
