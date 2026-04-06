//! Rocket launch and stage separation scenarios.
//!
//! Implements multi-stage rocket dynamics with:
//! - Stage separation events
//! - Atmospheric drag
//! - Variable mass (fuel consumption)

use crate::domains::physics::ForceField;
use crate::engine::state::{SimState, Vec3};
use serde::{Deserialize, Serialize};

/// Configuration for a rocket stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Dry mass (kg).
    pub dry_mass: f64,
    /// Fuel mass (kg).
    pub fuel_mass: f64,
    /// Thrust (N).
    pub thrust: f64,
    /// Specific impulse (s).
    pub isp: f64,
    /// Drag coefficient.
    pub cd: f64,
    /// Cross-sectional area (m²).
    pub area: f64,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            dry_mass: 25_000.0,
            fuel_mass: 100_000.0,
            thrust: 1_000_000.0,
            isp: 300.0,
            cd: 0.3,
            area: 10.0,
        }
    }
}

/// Rocket configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocketConfig {
    /// First stage configuration.
    pub stage1: StageConfig,
    /// Second stage configuration (optional).
    pub stage2: Option<StageConfig>,
    /// Separation altitude (m).
    pub separation_altitude: f64,
    /// Initial position.
    pub initial_position: Vec3,
    /// Initial velocity.
    pub initial_velocity: Vec3,
}

impl Default for RocketConfig {
    fn default() -> Self {
        Self {
            stage1: StageConfig {
                dry_mass: 25_000.0,
                fuel_mass: 400_000.0,
                thrust: 7_600_000.0,
                isp: 282.0,
                cd: 0.3,
                area: 10.0,
            },
            stage2: Some(StageConfig {
                dry_mass: 4_000.0,
                fuel_mass: 110_000.0,
                thrust: 934_000.0,
                isp: 348.0,
                cd: 0.3,
                area: 10.0,
            }),
            separation_altitude: 80_000.0,
            initial_position: Vec3::zero(),
            initial_velocity: Vec3::zero(),
        }
    }
}

/// Stage separation event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageSeparation {
    /// First stage active.
    Stage1Active,
    /// Separation in progress.
    Separating,
    /// Second stage active.
    Stage2Active,
    /// All stages exhausted.
    Complete,
}

/// Rocket scenario for launch simulations.
#[derive(Debug, Clone)]
pub struct RocketScenario {
    config: RocketConfig,
    stage: StageSeparation,
    fuel_remaining_1: f64,
    fuel_remaining_2: f64,
}

impl RocketScenario {
    /// Create a new rocket scenario.
    #[must_use]
    pub fn new(config: RocketConfig) -> Self {
        let fuel_remaining_1 = config.stage1.fuel_mass;
        let fuel_remaining_2 = config.stage2.as_ref().map_or(0.0, |s| s.fuel_mass);

        Self {
            config,
            stage: StageSeparation::Stage1Active,
            fuel_remaining_1,
            fuel_remaining_2,
        }
    }

    /// Initialize simulation state for the rocket.
    #[must_use]
    pub fn init_state(&self) -> SimState {
        let mut state = SimState::new();

        let total_mass = self.total_mass();
        state.add_body(
            total_mass,
            self.config.initial_position,
            self.config.initial_velocity,
        );

        state
    }

    /// Get total current mass.
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        match self.stage {
            StageSeparation::Stage1Active | StageSeparation::Separating => {
                self.config.stage1.dry_mass
                    + self.fuel_remaining_1
                    + self
                        .config
                        .stage2
                        .as_ref()
                        .map_or(0.0, |s| s.dry_mass + s.fuel_mass)
            }
            StageSeparation::Stage2Active => self
                .config
                .stage2
                .as_ref()
                .map_or(0.0, |s| s.dry_mass + self.fuel_remaining_2),
            StageSeparation::Complete => self
                .config
                .stage2
                .as_ref()
                .map_or(self.config.stage1.dry_mass, |s| s.dry_mass),
        }
    }

    /// Get current thrust.
    #[must_use]
    pub fn current_thrust(&self) -> f64 {
        match self.stage {
            StageSeparation::Stage1Active if self.fuel_remaining_1 > 0.0 => {
                self.config.stage1.thrust
            }
            StageSeparation::Stage2Active if self.fuel_remaining_2 > 0.0 => {
                self.config.stage2.as_ref().map_or(0.0, |s| s.thrust)
            }
            _ => 0.0,
        }
    }

    /// Consume fuel for a timestep.
    pub fn consume_fuel(&mut self, dt: f64) {
        const G0: f64 = 9.80665; // Standard gravity

        match self.stage {
            StageSeparation::Stage1Active => {
                let mass_flow = self.config.stage1.thrust / (self.config.stage1.isp * G0);
                self.fuel_remaining_1 = (self.fuel_remaining_1 - mass_flow * dt).max(0.0);
            }
            StageSeparation::Stage2Active => {
                if let Some(ref stage2) = self.config.stage2 {
                    let mass_flow = stage2.thrust / (stage2.isp * G0);
                    self.fuel_remaining_2 = (self.fuel_remaining_2 - mass_flow * dt).max(0.0);
                }
            }
            _ => {}
        }
    }

    /// Check and perform stage separation.
    pub fn check_separation(&mut self, altitude: f64) -> bool {
        if self.stage == StageSeparation::Stage1Active
            && altitude >= self.config.separation_altitude
            && self.config.stage2.is_some()
        {
            self.stage = StageSeparation::Separating;
            true
        } else {
            false
        }
    }

    /// Complete stage separation.
    pub fn complete_separation(&mut self) {
        if self.stage == StageSeparation::Separating {
            self.stage = StageSeparation::Stage2Active;
        }
    }

    /// Get current stage.
    #[must_use]
    pub const fn current_stage(&self) -> StageSeparation {
        self.stage
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &RocketConfig {
        &self.config
    }

    /// Get fuel remaining (stage 1).
    #[must_use]
    pub const fn fuel_remaining_stage1(&self) -> f64 {
        self.fuel_remaining_1
    }

    /// Get fuel remaining (stage 2).
    #[must_use]
    pub const fn fuel_remaining_stage2(&self) -> f64 {
        self.fuel_remaining_2
    }
}

/// Atmospheric model for drag calculation.
#[derive(Debug, Clone)]
pub struct AtmosphericModel {
    /// Sea-level density (kg/m³).
    pub rho_0: f64,
    /// Scale height (m).
    pub scale_height: f64,
}

impl Default for AtmosphericModel {
    fn default() -> Self {
        Self {
            rho_0: 1.225,
            scale_height: 8500.0,
        }
    }
}

impl AtmosphericModel {
    /// Get density at altitude.
    #[must_use]
    pub fn density(&self, altitude: f64) -> f64 {
        if altitude < 0.0 {
            self.rho_0
        } else {
            self.rho_0 * (-altitude / self.scale_height).exp()
        }
    }
}

/// Rocket force field combining gravity, thrust, and drag.
#[derive(Debug, Clone)]
pub struct RocketForceField {
    /// Gravitational acceleration at surface.
    pub g_surface: f64,
    /// Earth radius (m).
    pub earth_radius: f64,
    /// Atmospheric model.
    pub atmosphere: AtmosphericModel,
    /// Current thrust (updated externally).
    pub thrust: f64,
    /// Current mass (updated externally).
    pub mass: f64,
    /// Drag coefficient.
    pub cd: f64,
    /// Cross-sectional area.
    pub area: f64,
}

impl Default for RocketForceField {
    fn default() -> Self {
        Self {
            g_surface: 9.80665,
            earth_radius: 6_371_000.0,
            atmosphere: AtmosphericModel::default(),
            thrust: 0.0,
            mass: 1.0,
            cd: 0.3,
            area: 10.0,
        }
    }
}

impl RocketForceField {
    /// Update thrust and mass.
    pub fn update(&mut self, thrust: f64, mass: f64, cd: f64, area: f64) {
        self.thrust = thrust;
        self.mass = mass;
        self.cd = cd;
        self.area = area;
    }
}

impl ForceField for RocketForceField {
    fn acceleration(&self, position: &Vec3, _mass: f64) -> Vec3 {
        let altitude = position.z;

        // Gravity (inverse square law)
        let r = self.earth_radius + altitude;
        let g = self.g_surface * (self.earth_radius / r).powi(2);

        // Thrust (upward)
        let thrust_acc = self.thrust / self.mass;

        // Drag (opposing velocity direction)
        // Note: this is simplified - real drag depends on velocity
        let rho = self.atmosphere.density(altitude);
        let drag_factor = 0.5 * rho * self.cd * self.area / self.mass;

        // Net acceleration (thrust up, gravity down)
        Vec3::new(0.0, 0.0, thrust_acc - g - drag_factor)
    }

    fn potential(&self, position: &Vec3, mass: f64) -> f64 {
        // Gravitational potential energy
        let altitude = position.z;
        let r = self.earth_radius + altitude;
        -self.g_surface * self.earth_radius * self.earth_radius * mass / r
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_config_default() {
        let config = StageConfig::default();
        assert!(config.dry_mass > 0.0);
        assert!(config.fuel_mass > 0.0);
        assert!(config.thrust > 0.0);
    }

    #[test]
    fn test_stage_config_clone() {
        let config = StageConfig::default();
        let cloned = config.clone();
        assert!((cloned.thrust - config.thrust).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rocket_config_default() {
        let config = RocketConfig::default();
        assert!(config.stage1.thrust > 0.0);
        assert!(config.stage2.is_some());
        assert!(config.separation_altitude > 0.0);
    }

    #[test]
    fn test_rocket_config_clone() {
        let config = RocketConfig::default();
        let cloned = config.clone();
        assert!((cloned.separation_altitude - config.separation_altitude).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rocket_scenario_init() {
        let scenario = RocketScenario::new(RocketConfig::default());
        let state = scenario.init_state();

        assert_eq!(state.num_bodies(), 1);
        assert!(scenario.total_mass() > 0.0);
        assert_eq!(scenario.current_stage(), StageSeparation::Stage1Active);
    }

    #[test]
    fn test_rocket_scenario_clone() {
        let scenario = RocketScenario::new(RocketConfig::default());
        let cloned = scenario.clone();
        assert_eq!(cloned.current_stage(), scenario.current_stage());
    }

    #[test]
    fn test_rocket_scenario_config() {
        let config = RocketConfig {
            separation_altitude: 50000.0,
            ..Default::default()
        };
        let scenario = RocketScenario::new(config);
        assert!((scenario.config().separation_altitude - 50000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rocket_fuel_remaining() {
        let config = RocketConfig::default();
        let scenario = RocketScenario::new(config.clone());
        assert!((scenario.fuel_remaining_stage1() - config.stage1.fuel_mass).abs() < f64::EPSILON);
        assert!(
            (scenario.fuel_remaining_stage2() - config.stage2.as_ref().unwrap().fuel_mass).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_rocket_fuel_consumption() {
        let mut scenario = RocketScenario::new(RocketConfig::default());
        let initial_fuel = scenario.fuel_remaining_stage1();

        scenario.consume_fuel(1.0);

        assert!(scenario.fuel_remaining_stage1() < initial_fuel);
    }

    #[test]
    fn test_rocket_fuel_consumption_stage2() {
        let config = RocketConfig {
            separation_altitude: 0.0, // Immediate separation
            ..Default::default()
        };
        let mut scenario = RocketScenario::new(config);

        // Trigger separation
        scenario.check_separation(100.0);
        scenario.complete_separation();

        let initial_fuel = scenario.fuel_remaining_stage2();
        scenario.consume_fuel(1.0);
        assert!(scenario.fuel_remaining_stage2() < initial_fuel);
    }

    #[test]
    fn test_rocket_current_thrust_stage1() {
        let scenario = RocketScenario::new(RocketConfig::default());
        let thrust = scenario.current_thrust();
        assert!(thrust > 0.0);
    }

    #[test]
    fn test_rocket_current_thrust_stage2() {
        let config = RocketConfig {
            separation_altitude: 0.0,
            ..Default::default()
        };
        let mut scenario = RocketScenario::new(config);

        // Transition to stage 2
        scenario.check_separation(100.0);
        scenario.complete_separation();

        let thrust = scenario.current_thrust();
        assert!(thrust > 0.0);
    }

    #[test]
    fn test_rocket_current_thrust_no_fuel() {
        let config = RocketConfig {
            stage1: StageConfig {
                fuel_mass: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let scenario = RocketScenario::new(config);
        let thrust = scenario.current_thrust();
        assert!((thrust - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rocket_total_mass_stages() {
        let config = RocketConfig::default();
        let mut scenario = RocketScenario::new(config.clone());
        let stage1_mass = scenario.total_mass();

        // Transition to stage 2
        scenario.check_separation(100000.0);
        scenario.complete_separation();
        let stage2_mass = scenario.total_mass();

        // Stage 2 should be lighter (no stage 1)
        assert!(stage2_mass < stage1_mass);
    }

    #[test]
    fn test_rocket_stage_separation() {
        let config = RocketConfig {
            separation_altitude: 100.0,
            ..Default::default()
        };
        let mut scenario = RocketScenario::new(config);

        // Below separation altitude
        assert!(!scenario.check_separation(50.0));
        assert_eq!(scenario.current_stage(), StageSeparation::Stage1Active);

        // At separation altitude
        assert!(scenario.check_separation(100.0));
        assert_eq!(scenario.current_stage(), StageSeparation::Separating);

        // Complete separation
        scenario.complete_separation();
        assert_eq!(scenario.current_stage(), StageSeparation::Stage2Active);
    }

    #[test]
    fn test_rocket_no_stage2() {
        let config = RocketConfig {
            stage2: None,
            ..Default::default()
        };
        let mut scenario = RocketScenario::new(config);

        // No separation without stage 2
        assert!(!scenario.check_separation(100000.0));
        assert_eq!(scenario.current_stage(), StageSeparation::Stage1Active);
    }

    #[test]
    fn test_rocket_complete_separation_when_not_separating() {
        let mut scenario = RocketScenario::new(RocketConfig::default());
        // Not separating
        scenario.complete_separation();
        assert_eq!(scenario.current_stage(), StageSeparation::Stage1Active);
    }

    #[test]
    fn test_stage_separation_eq() {
        assert_eq!(StageSeparation::Stage1Active, StageSeparation::Stage1Active);
        assert_ne!(StageSeparation::Stage1Active, StageSeparation::Stage2Active);
    }

    #[test]
    fn test_atmospheric_model() {
        let atm = AtmosphericModel::default();

        // Sea level
        assert!((atm.density(0.0) - 1.225).abs() < 0.01);

        // Decreases with altitude
        assert!(atm.density(10000.0) < atm.density(0.0));
        assert!(atm.density(50000.0) < atm.density(10000.0));
    }

    #[test]
    fn test_atmospheric_model_negative_altitude() {
        let atm = AtmosphericModel::default();
        // Below sea level returns sea level density
        assert!((atm.density(-100.0) - 1.225).abs() < f64::EPSILON);
    }

    #[test]
    fn test_atmospheric_model_clone() {
        let atm = AtmosphericModel::default();
        let cloned = atm.clone();
        assert!((cloned.rho_0 - atm.rho_0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rocket_force_field() {
        let field = RocketForceField {
            thrust: 1_000_000.0,
            mass: 100_000.0,
            ..Default::default()
        };

        let acc = field.acceleration(&Vec3::new(0.0, 0.0, 0.0), 100_000.0);

        // Should have positive z acceleration (thrust > gravity at launch)
        // thrust/mass = 10 m/s², gravity ≈ 9.8 m/s², so net positive
        assert!(acc.z > 0.0);
    }

    #[test]
    fn test_rocket_force_field_default() {
        let field = RocketForceField::default();
        assert!((field.g_surface - 9.80665).abs() < 0.001);
        assert!(field.earth_radius > 0.0);
    }

    #[test]
    fn test_rocket_force_field_update() {
        let mut field = RocketForceField::default();
        field.update(1000.0, 500.0, 0.5, 20.0);
        assert!((field.thrust - 1000.0).abs() < f64::EPSILON);
        assert!((field.mass - 500.0).abs() < f64::EPSILON);
        assert!((field.cd - 0.5).abs() < f64::EPSILON);
        assert!((field.area - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rocket_force_field_clone() {
        let field = RocketForceField {
            thrust: 1000.0,
            ..Default::default()
        };
        let cloned = field.clone();
        assert!((cloned.thrust - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rocket_force_field_potential() {
        let field = RocketForceField::default();
        let pos = Vec3::new(0.0, 0.0, 0.0);
        let pe = field.potential(&pos, 1000.0);
        // Potential energy should be negative (gravitational)
        assert!(pe < 0.0);
    }

    #[test]
    fn test_rocket_force_field_high_altitude() {
        let field = RocketForceField {
            thrust: 0.0,
            mass: 1000.0,
            ..Default::default()
        };

        // At high altitude, gravity should be weaker
        let acc_surface = field.acceleration(&Vec3::new(0.0, 0.0, 0.0), 1000.0);
        let acc_high = field.acceleration(&Vec3::new(0.0, 0.0, 100_000.0), 1000.0);

        assert!(acc_high.z.abs() < acc_surface.z.abs());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Thrust-to-weight ratio determines liftoff capability.
        #[test]
        fn prop_twr_liftoff(
            thrust in 5000.0f64..50000.0,
            mass in 500.0f64..5000.0,
        ) {
            let g = 9.81;
            let twr = thrust / (mass * g);

            // TWR > 1 means rocket can lift off
            if twr > 1.0 {
                let net_accel = thrust / mass - g;
                prop_assert!(net_accel > 0.0, "TWR={} but no liftoff", twr);
            }
        }

        /// Gravity decreases with altitude (inverse square law).
        #[test]
        fn prop_gravity_decreases_with_altitude(
            alt1 in 0.0f64..100_000.0,
            alt2 in 0.0f64..100_000.0,
        ) {
            let field = RocketForceField::default();
            let pos1 = Vec3::new(0.0, 0.0, alt1);
            let pos2 = Vec3::new(0.0, 0.0, alt2);

            let acc1 = field.acceleration(&pos1, 1000.0).z.abs();
            let acc2 = field.acceleration(&pos2, 1000.0).z.abs();

            // Higher altitude -> weaker gravity
            if alt1 > alt2 {
                prop_assert!(acc1 <= acc2 + 0.001, "g at {}m = {}, at {}m = {}", alt1, acc1, alt2, acc2);
            } else if alt1 < alt2 {
                prop_assert!(acc1 >= acc2 - 0.001, "g at {}m = {}, at {}m = {}", alt1, acc1, alt2, acc2);
            }
        }

        /// Specific impulse relates thrust and mass flow rate.
        #[test]
        fn prop_specific_impulse(
            isp in 200.0f64..500.0,
            mass_flow in 10.0f64..100.0,
        ) {
            // Thrust = Isp * g0 * mass_flow
            let g0 = 9.81;
            let thrust = isp * g0 * mass_flow;

            prop_assert!(thrust > 0.0);
            prop_assert!(thrust > isp * mass_flow, "thrust={}", thrust);
        }

        /// Rocket mass must be positive.
        #[test]
        fn prop_mass_positive(
            dry_mass in 100.0f64..10000.0,
            fuel_mass in 100.0f64..50000.0,
        ) {
            let total_mass = dry_mass + fuel_mass;
            prop_assert!(total_mass > 0.0);
            prop_assert!(total_mass > dry_mass);
        }

        /// Delta-v from Tsiolkovsky rocket equation.
        #[test]
        fn prop_delta_v_tsiolkovsky(
            isp in 200.0f64..450.0,
            mass_ratio in 1.5f64..10.0,
        ) {
            // delta_v = Isp * g0 * ln(mass_ratio)
            let g0 = 9.81;
            let delta_v = isp * g0 * mass_ratio.ln();

            prop_assert!(delta_v > 0.0);
            // Higher mass ratio -> higher delta-v
            let delta_v_higher = isp * g0 * (mass_ratio * 1.1).ln();
            prop_assert!(delta_v_higher > delta_v);
        }
    }
}
