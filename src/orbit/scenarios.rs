//! Pre-built orbital scenarios.
//!
//! Implements canonical orbital mechanics demonstrations:
//! - Kepler two-body problem (Earth-Sun)
//! - N-body solar system
//! - Hohmann transfer orbits
//! - Lagrange point dynamics
//!
//! # References
//!
//! [6] Bate, Mueller, White, "Fundamentals of Astrodynamics," 1971.
//! [23] Hohmann, "Die Erreichbarkeit der Himmelskörper," 1925.
//! [24] Szebehely, "Theory of Orbits," 1967.

use crate::orbit::physics::{NBodyState, OrbitBody};
use crate::orbit::units::{OrbitMass, Position3D, Velocity3D, AU, EARTH_MASS, G, SOLAR_MASS};
use serde::{Deserialize, Serialize};

/// Scenario type enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScenarioType {
    /// Two-body Keplerian orbit.
    Kepler(KeplerConfig),
    /// N-body gravitational system.
    NBody(NBodyConfig),
    /// Hohmann transfer between circular orbits.
    Hohmann(HohmannConfig),
    /// Lagrange point dynamics.
    Lagrange(LagrangeConfig),
}

/// Kepler two-body configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeplerConfig {
    /// Central body mass (kg).
    pub central_mass: f64,
    /// Orbiting body mass (kg).
    pub orbiter_mass: f64,
    /// Semi-major axis (m).
    pub semi_major_axis: f64,
    /// Eccentricity (0-1 for ellipses).
    pub eccentricity: f64,
    /// Initial true anomaly (radians).
    pub initial_anomaly: f64,
}

impl Default for KeplerConfig {
    fn default() -> Self {
        Self {
            central_mass: SOLAR_MASS,
            orbiter_mass: EARTH_MASS,
            semi_major_axis: AU,
            eccentricity: 0.0167, // Earth's eccentricity
            initial_anomaly: 0.0,
        }
    }
}

impl KeplerConfig {
    /// Create Earth-Sun system.
    #[must_use]
    pub fn earth_sun() -> Self {
        Self::default()
    }

    /// Create circular orbit.
    #[must_use]
    pub fn circular(central_mass: f64, orbiter_mass: f64, radius: f64) -> Self {
        Self {
            central_mass,
            orbiter_mass,
            semi_major_axis: radius,
            eccentricity: 0.0,
            initial_anomaly: 0.0,
        }
    }

    /// Calculate orbital period (seconds).
    #[must_use]
    pub fn period(&self) -> f64 {
        2.0 * std::f64::consts::PI * (self.semi_major_axis.powi(3) / (G * self.central_mass)).sqrt()
    }

    /// Calculate circular orbital velocity at semi-major axis.
    #[must_use]
    pub fn circular_velocity(&self) -> f64 {
        (G * self.central_mass / self.semi_major_axis).sqrt()
    }

    /// Build N-body state from this configuration.
    #[must_use]
    #[allow(clippy::many_single_char_names)] // Standard orbital mechanics notation
    pub fn build(&self, softening: f64) -> NBodyState {
        let mu = G * self.central_mass;

        // Calculate position and velocity at initial anomaly
        // p = semi-latus rectum, r = radius, h = specific angular momentum
        let semi_latus = self.semi_major_axis * (1.0 - self.eccentricity * self.eccentricity);
        let radius = semi_latus / (1.0 + self.eccentricity * self.initial_anomaly.cos());

        // Position in orbital plane
        let pos_x = radius * self.initial_anomaly.cos();
        let pos_y = radius * self.initial_anomaly.sin();

        // Velocity in orbital plane
        let ang_momentum = (mu * semi_latus).sqrt();
        let vel_x = -mu / ang_momentum * self.initial_anomaly.sin();
        let vel_y = mu / ang_momentum * (self.eccentricity + self.initial_anomaly.cos());

        let bodies = vec![
            OrbitBody::new(
                OrbitMass::from_kg(self.central_mass),
                Position3D::zero(),
                Velocity3D::zero(),
            ),
            OrbitBody::new(
                OrbitMass::from_kg(self.orbiter_mass),
                Position3D::from_meters(pos_x, pos_y, 0.0),
                Velocity3D::from_mps(vel_x, vel_y, 0.0),
            ),
        ];

        NBodyState::new(bodies, softening)
    }
}

/// N-body configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NBodyConfig {
    /// Bodies in the system.
    pub bodies: Vec<BodyConfig>,
}

/// Single body configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyConfig {
    pub name: String,
    pub mass: f64,
    pub position: (f64, f64, f64),
    pub velocity: (f64, f64, f64),
}

impl Default for NBodyConfig {
    fn default() -> Self {
        Self::inner_solar_system()
    }
}

impl NBodyConfig {
    /// Create inner solar system (Sun + Mercury, Venus, Earth, Mars).
    #[must_use]
    pub fn inner_solar_system() -> Self {
        // Simplified orbital data (circular approximation)
        let sun = BodyConfig {
            name: "Sun".to_string(),
            mass: SOLAR_MASS,
            position: (0.0, 0.0, 0.0),
            velocity: (0.0, 0.0, 0.0),
        };

        let mercury = {
            let r = 0.387 * AU;
            let v = (G * SOLAR_MASS / r).sqrt();
            BodyConfig {
                name: "Mercury".to_string(),
                mass: 3.301e23,
                position: (r, 0.0, 0.0),
                velocity: (0.0, v, 0.0),
            }
        };

        let venus = {
            let r = 0.723 * AU;
            let v = (G * SOLAR_MASS / r).sqrt();
            BodyConfig {
                name: "Venus".to_string(),
                mass: 4.867e24,
                position: (0.0, r, 0.0),
                velocity: (-v, 0.0, 0.0),
            }
        };

        let earth = {
            let r = AU;
            let v = (G * SOLAR_MASS / r).sqrt();
            BodyConfig {
                name: "Earth".to_string(),
                mass: EARTH_MASS,
                position: (-r, 0.0, 0.0),
                velocity: (0.0, -v, 0.0),
            }
        };

        let mars = {
            let r = 1.524 * AU;
            let v = (G * SOLAR_MASS / r).sqrt();
            BodyConfig {
                name: "Mars".to_string(),
                mass: 6.417e23,
                position: (0.0, -r, 0.0),
                velocity: (v, 0.0, 0.0),
            }
        };

        Self {
            bodies: vec![sun, mercury, venus, earth, mars],
        }
    }

    /// Build N-body state from this configuration.
    #[must_use]
    pub fn build(&self, softening: f64) -> NBodyState {
        let bodies = self
            .bodies
            .iter()
            .map(|b| {
                OrbitBody::new(
                    OrbitMass::from_kg(b.mass),
                    Position3D::from_meters(b.position.0, b.position.1, b.position.2),
                    Velocity3D::from_mps(b.velocity.0, b.velocity.1, b.velocity.2),
                )
            })
            .collect();

        NBodyState::new(bodies, softening)
    }
}

/// Hohmann transfer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HohmannConfig {
    /// Central body mass (kg).
    pub central_mass: f64,
    /// Spacecraft mass (kg).
    pub spacecraft_mass: f64,
    /// Initial orbit radius (m).
    pub r1: f64,
    /// Target orbit radius (m).
    pub r2: f64,
    /// Current phase of transfer (0 = initial, 1 = first burn, 2 = transfer, 3 = second burn).
    pub phase: usize,
}

impl Default for HohmannConfig {
    fn default() -> Self {
        Self::earth_to_mars()
    }
}

impl HohmannConfig {
    /// Earth-to-Mars transfer.
    #[must_use]
    pub fn earth_to_mars() -> Self {
        Self {
            central_mass: SOLAR_MASS,
            spacecraft_mass: 1000.0,
            r1: AU,
            r2: 1.524 * AU,
            phase: 0,
        }
    }

    /// LEO to GEO transfer.
    #[must_use]
    pub fn leo_to_geo() -> Self {
        let _earth_mu = 3.986e14; // m³/s²
        let r_earth = 6.378e6; // m
        Self {
            central_mass: EARTH_MASS,
            spacecraft_mass: 1000.0,
            r1: r_earth + 400_000.0,    // 400 km LEO
            r2: r_earth + 35_786_000.0, // GEO
            phase: 0,
        }
    }

    /// Calculate first delta-v (departure burn).
    #[must_use]
    pub fn delta_v1(&self) -> f64 {
        let mu = G * self.central_mass;
        let v_circular = (mu / self.r1).sqrt();
        let v_transfer = (2.0 * mu * self.r2 / (self.r1 * (self.r1 + self.r2))).sqrt();
        v_transfer - v_circular
    }

    /// Calculate second delta-v (arrival burn).
    #[must_use]
    pub fn delta_v2(&self) -> f64 {
        let mu = G * self.central_mass;
        let v_transfer = (2.0 * mu * self.r1 / (self.r2 * (self.r1 + self.r2))).sqrt();
        let v_circular = (mu / self.r2).sqrt();
        v_circular - v_transfer
    }

    /// Calculate total delta-v.
    #[must_use]
    pub fn total_delta_v(&self) -> f64 {
        self.delta_v1().abs() + self.delta_v2().abs()
    }

    /// Calculate transfer time (seconds).
    #[must_use]
    pub fn transfer_time(&self) -> f64 {
        let mu = G * self.central_mass;
        let a = (self.r1 + self.r2) / 2.0;
        std::f64::consts::PI * (a.powi(3) / mu).sqrt()
    }

    /// Build initial state (spacecraft at r1 with circular velocity).
    #[must_use]
    pub fn build_initial(&self, softening: f64) -> NBodyState {
        let mu = G * self.central_mass;
        let v = (mu / self.r1).sqrt();

        let bodies = vec![
            OrbitBody::new(
                OrbitMass::from_kg(self.central_mass),
                Position3D::zero(),
                Velocity3D::zero(),
            ),
            OrbitBody::new(
                OrbitMass::from_kg(self.spacecraft_mass),
                Position3D::from_meters(self.r1, 0.0, 0.0),
                Velocity3D::from_mps(0.0, v, 0.0),
            ),
        ];

        NBodyState::new(bodies, softening)
    }

    /// Build state after first burn (on transfer orbit).
    #[must_use]
    pub fn build_transfer(&self, softening: f64) -> NBodyState {
        let mu = G * self.central_mass;
        let v = (mu / self.r1).sqrt() + self.delta_v1();

        let bodies = vec![
            OrbitBody::new(
                OrbitMass::from_kg(self.central_mass),
                Position3D::zero(),
                Velocity3D::zero(),
            ),
            OrbitBody::new(
                OrbitMass::from_kg(self.spacecraft_mass),
                Position3D::from_meters(self.r1, 0.0, 0.0),
                Velocity3D::from_mps(0.0, v, 0.0),
            ),
        ];

        NBodyState::new(bodies, softening)
    }
}

/// Lagrange point selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LagrangePoint {
    L1,
    L2,
    L3,
    L4,
    L5,
}

/// Lagrange point configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagrangeConfig {
    /// Primary body mass (kg).
    pub primary_mass: f64,
    /// Secondary body mass (kg).
    pub secondary_mass: f64,
    /// Distance between primary and secondary (m).
    pub separation: f64,
    /// Selected Lagrange point.
    pub point: LagrangePoint,
    /// Initial perturbation from Lagrange point (m).
    pub perturbation: (f64, f64, f64),
}

impl Default for LagrangeConfig {
    fn default() -> Self {
        Self::sun_earth_l2()
    }
}

impl LagrangeConfig {
    /// Sun-Earth L2 (like JWST).
    #[must_use]
    pub fn sun_earth_l2() -> Self {
        Self {
            primary_mass: SOLAR_MASS,
            secondary_mass: EARTH_MASS,
            separation: AU,
            point: LagrangePoint::L2,
            perturbation: (0.0, 0.0, 0.0),
        }
    }

    /// Sun-Earth L4 (Trojan point).
    #[must_use]
    pub fn sun_earth_l4() -> Self {
        Self {
            primary_mass: SOLAR_MASS,
            secondary_mass: EARTH_MASS,
            separation: AU,
            point: LagrangePoint::L4,
            perturbation: (0.0, 0.0, 0.0),
        }
    }

    /// Calculate mass ratio μ = m2 / (m1 + m2).
    #[must_use]
    pub fn mass_ratio(&self) -> f64 {
        self.secondary_mass / (self.primary_mass + self.secondary_mass)
    }

    /// Calculate approximate Lagrange point position (in rotating frame).
    #[must_use]
    pub fn lagrange_position(&self) -> (f64, f64, f64) {
        let mu = self.mass_ratio();
        let r = self.separation;

        match self.point {
            LagrangePoint::L1 => {
                // Approximate L1: between primary and secondary
                let x = r * (1.0 - (mu / 3.0).cbrt());
                (x, 0.0, 0.0)
            }
            LagrangePoint::L2 => {
                // Approximate L2: beyond secondary
                let x = r * (1.0 + (mu / 3.0).cbrt());
                (x, 0.0, 0.0)
            }
            LagrangePoint::L3 => {
                // Approximate L3: opposite side
                let x = -r * (1.0 + 5.0 * mu / 12.0);
                (x, 0.0, 0.0)
            }
            LagrangePoint::L4 => {
                // L4: 60° ahead of secondary
                let x = r * 0.5;
                let y = r * (3.0_f64.sqrt() / 2.0);
                (x, y, 0.0)
            }
            LagrangePoint::L5 => {
                // L5: 60° behind secondary
                let x = r * 0.5;
                let y = -r * (3.0_f64.sqrt() / 2.0);
                (x, y, 0.0)
            }
        }
    }

    /// Build state with test particle near Lagrange point.
    #[must_use]
    pub fn build(&self, softening: f64) -> NBodyState {
        let mu = self.mass_ratio();
        let total_mass = self.primary_mass + self.secondary_mass;
        let r = self.separation;

        // Barycenter-centered coordinates
        let x1 = -mu * r; // Primary position
        let x2 = (1.0 - mu) * r; // Secondary position

        // Orbital angular velocity
        let omega = (G * total_mass / r.powi(3)).sqrt();

        // Secondary velocity (circular orbit around barycenter)
        let v2 = omega * x2.abs();

        // Lagrange point position
        let (lx, ly, lz) = self.lagrange_position();
        let lx = lx + self.perturbation.0;
        let ly = ly + self.perturbation.1;
        let lz = lz + self.perturbation.2;

        // Test particle velocity (co-rotating with system)
        let test_r = (lx * lx + ly * ly).sqrt();
        let test_v = omega * test_r;
        let test_vx = -test_v * ly / test_r.max(1e-10);
        let test_vy = test_v * lx / test_r.max(1e-10);

        let bodies = vec![
            OrbitBody::new(
                OrbitMass::from_kg(self.primary_mass),
                Position3D::from_meters(x1, 0.0, 0.0),
                Velocity3D::zero(),
            ),
            OrbitBody::new(
                OrbitMass::from_kg(self.secondary_mass),
                Position3D::from_meters(x2, 0.0, 0.0),
                Velocity3D::from_mps(0.0, v2, 0.0),
            ),
            OrbitBody::new(
                OrbitMass::from_kg(1.0), // Negligible mass test particle
                Position3D::from_meters(lx, ly, lz),
                Velocity3D::from_mps(test_vx, test_vy, 0.0),
            ),
        ];

        NBodyState::new(bodies, softening)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn test_kepler_config_default() {
        let config = KeplerConfig::default();
        assert!((config.central_mass - SOLAR_MASS).abs() < 1e20);
        assert!((config.orbiter_mass - EARTH_MASS).abs() < 1e20);
    }

    #[test]
    fn test_kepler_period() {
        let config = KeplerConfig::earth_sun();
        let period = config.period();
        let expected = 365.25 * 86400.0; // ~1 year in seconds
        let relative_error = (period - expected).abs() / expected;
        assert!(relative_error < 0.001, "Period error: {relative_error}");
    }

    #[test]
    fn test_kepler_circular_velocity() {
        let config = KeplerConfig::earth_sun();
        let v = config.circular_velocity();
        let expected = 29780.0; // ~29.78 km/s
        let relative_error = (v - expected).abs() / expected;
        assert!(relative_error < 0.001, "Velocity error: {relative_error}");
    }

    #[test]
    fn test_kepler_build() {
        let config = KeplerConfig::earth_sun();
        let state = config.build(0.0);
        assert_eq!(state.num_bodies(), 2);
    }

    #[test]
    fn test_nbody_inner_solar_system() {
        let config = NBodyConfig::inner_solar_system();
        assert_eq!(config.bodies.len(), 5);
        assert_eq!(config.bodies[0].name, "Sun");
    }

    #[test]
    fn test_nbody_build() {
        let config = NBodyConfig::inner_solar_system();
        let state = config.build(1e6);
        assert_eq!(state.num_bodies(), 5);
    }

    #[test]
    fn test_hohmann_delta_v1() {
        let config = HohmannConfig::earth_to_mars();
        let dv1 = config.delta_v1();
        assert!(dv1 > 0.0); // Prograde burn
        assert!(dv1 < 5000.0); // Reasonable km/s
    }

    #[test]
    fn test_hohmann_delta_v2() {
        let config = HohmannConfig::earth_to_mars();
        let dv2 = config.delta_v2();
        assert!(dv2 > 0.0); // Prograde burn
        assert!(dv2 < 5000.0);
    }

    #[test]
    fn test_hohmann_total_delta_v() {
        let config = HohmannConfig::earth_to_mars();
        let total = config.total_delta_v();
        let expected = 5600.0; // ~5.6 km/s for Earth-Mars
        let relative_error = (total - expected).abs() / expected;
        assert!(
            relative_error < 0.1,
            "Total delta-v error: {relative_error}"
        );
    }

    #[test]
    fn test_hohmann_transfer_time() {
        let config = HohmannConfig::earth_to_mars();
        let time = config.transfer_time();
        let expected = 259.0 * 86400.0; // ~259 days
        let relative_error = (time - expected).abs() / expected;
        assert!(
            relative_error < 0.1,
            "Transfer time error: {relative_error}"
        );
    }

    #[test]
    fn test_hohmann_build_initial() {
        let config = HohmannConfig::earth_to_mars();
        let state = config.build_initial(0.0);
        assert_eq!(state.num_bodies(), 2);
    }

    #[test]
    fn test_lagrange_mass_ratio() {
        let config = LagrangeConfig::sun_earth_l2();
        let mu = config.mass_ratio();
        assert!(mu > 0.0);
        assert!(mu < 1e-5); // Earth is much smaller than Sun
    }

    #[test]
    fn test_lagrange_l1_position() {
        let config = LagrangeConfig {
            point: LagrangePoint::L1,
            ..LagrangeConfig::default()
        };
        let (x, y, z) = config.lagrange_position();
        assert!(x > 0.0); // Between Sun and Earth
        assert!(x < AU);
        assert!(y.abs() < EPSILON);
        assert!(z.abs() < EPSILON);
    }

    #[test]
    fn test_lagrange_l2_position() {
        let config = LagrangeConfig::sun_earth_l2();
        let (x, y, z) = config.lagrange_position();
        assert!(x > AU); // Beyond Earth
        assert!(y.abs() < EPSILON);
        assert!(z.abs() < EPSILON);
    }

    #[test]
    fn test_lagrange_l4_position() {
        let config = LagrangeConfig::sun_earth_l4();
        let (x, y, z) = config.lagrange_position();
        // L4 is at 60° ahead, so x ≈ 0.5*AU, y ≈ 0.866*AU
        assert!((x - 0.5 * AU).abs() / AU < 0.01);
        assert!((y - 0.866 * AU).abs() / AU < 0.01);
        assert!(z.abs() < EPSILON);
    }

    #[test]
    fn test_lagrange_build() {
        let config = LagrangeConfig::sun_earth_l2();
        let state = config.build(1e6);
        assert_eq!(state.num_bodies(), 3); // Sun, Earth, test particle
    }

    #[test]
    fn test_scenario_type_kepler() {
        let scenario = ScenarioType::Kepler(KeplerConfig::default());
        match scenario {
            ScenarioType::Kepler(config) => {
                assert!((config.eccentricity - 0.0167).abs() < 0.001);
            }
            _ => panic!("Expected Kepler scenario"),
        }
    }

    #[test]
    fn test_kepler_circular() {
        let config = KeplerConfig::circular(SOLAR_MASS, EARTH_MASS, AU);
        assert!((config.eccentricity).abs() < EPSILON);
        assert!((config.central_mass - SOLAR_MASS).abs() < 1e20);
    }

    #[test]
    fn test_nbody_default() {
        let config = NBodyConfig::default();
        assert_eq!(config.bodies.len(), 5);
    }

    #[test]
    fn test_hohmann_default() {
        let config = HohmannConfig::default();
        assert!((config.r1 - AU).abs() < 1e6);
    }

    #[test]
    fn test_hohmann_leo_to_geo() {
        let config = HohmannConfig::leo_to_geo();
        assert!(config.r1 > 6e6); // Above Earth's surface
        assert!(config.r2 > 4e7); // GEO radius
        assert!(config.central_mass > 5e24); // Earth mass
    }

    #[test]
    fn test_hohmann_build_transfer() {
        let config = HohmannConfig::earth_to_mars();
        let state = config.build_transfer(0.0);
        assert_eq!(state.num_bodies(), 2);

        // Spacecraft should be at r1 with transfer velocity (higher than circular)
        let (x, _, _) = state.bodies[1].position.as_meters();
        assert!((x - config.r1).abs() < 1e6);
    }

    #[test]
    fn test_lagrange_default() {
        let config = LagrangeConfig::default();
        assert_eq!(config.point, LagrangePoint::L2);
    }

    #[test]
    fn test_lagrange_l3_position() {
        let config = LagrangeConfig {
            point: LagrangePoint::L3,
            ..LagrangeConfig::default()
        };
        let (x, y, z) = config.lagrange_position();
        assert!(x < 0.0); // Opposite side from Earth
        assert!(x < -AU * 0.9);
        assert!(y.abs() < EPSILON);
        assert!(z.abs() < EPSILON);
    }

    #[test]
    fn test_lagrange_l5_position() {
        let config = LagrangeConfig {
            point: LagrangePoint::L5,
            ..LagrangeConfig::default()
        };
        let (x, y, z) = config.lagrange_position();
        // L5 is at 60° behind, so x ≈ 0.5*AU, y ≈ -0.866*AU
        assert!((x - 0.5 * AU).abs() / AU < 0.01);
        assert!((y + 0.866 * AU).abs() / AU < 0.01); // Negative y
        assert!(z.abs() < EPSILON);
    }

    #[test]
    fn test_lagrange_with_perturbation() {
        let config = LagrangeConfig {
            perturbation: (1000.0, 2000.0, 3000.0),
            ..LagrangeConfig::sun_earth_l2()
        };
        let state = config.build(1e6);
        assert_eq!(state.num_bodies(), 3);

        // Test particle should be offset from nominal L2
        let (lx, _, _) = state.bodies[2].position.as_meters();
        let nominal_l2 = config.lagrange_position().0;
        assert!((lx - nominal_l2 - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_scenario_type_nbody() {
        let scenario = ScenarioType::NBody(NBodyConfig::default());
        match scenario {
            ScenarioType::NBody(config) => {
                assert_eq!(config.bodies.len(), 5);
            }
            _ => panic!("Expected NBody scenario"),
        }
    }

    #[test]
    fn test_scenario_type_hohmann() {
        let scenario = ScenarioType::Hohmann(HohmannConfig::default());
        match scenario {
            ScenarioType::Hohmann(config) => {
                assert!((config.r1 - AU).abs() < 1e6);
            }
            _ => panic!("Expected Hohmann scenario"),
        }
    }

    #[test]
    fn test_scenario_type_lagrange() {
        let scenario = ScenarioType::Lagrange(LagrangeConfig::default());
        match scenario {
            ScenarioType::Lagrange(config) => {
                assert_eq!(config.point, LagrangePoint::L2);
            }
            _ => panic!("Expected Lagrange scenario"),
        }
    }

    #[test]
    fn test_kepler_build_with_nonzero_anomaly() {
        let config = KeplerConfig {
            initial_anomaly: std::f64::consts::PI / 2.0, // 90 degrees
            ..KeplerConfig::default()
        };
        let state = config.build(0.0);
        assert_eq!(state.num_bodies(), 2);

        // At 90° anomaly, Earth should be above or below the x-axis
        let (_, y, _) = state.bodies[1].position.as_meters();
        assert!(y.abs() > AU * 0.5);
    }

    #[test]
    fn test_body_config_fields() {
        let body = BodyConfig {
            name: "TestBody".to_string(),
            mass: 1e24,
            position: (1e11, 2e11, 3e11),
            velocity: (1000.0, 2000.0, 3000.0),
        };
        assert_eq!(body.name, "TestBody");
        assert!((body.mass - 1e24).abs() < 1e18);
        assert!((body.position.0 - 1e11).abs() < 1e6);
        assert!((body.velocity.0 - 1000.0).abs() < 0.1);
    }
}
