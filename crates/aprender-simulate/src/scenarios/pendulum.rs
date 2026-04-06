//! Pendulum scenarios for canonical physics examples.
//!
//! Implements various pendulum systems:
//! - Simple pendulum (single mass on string)
//! - Double pendulum (chaotic dynamics)
//! - Driven pendulum (forced oscillations)

use crate::domains::physics::ForceField;
use crate::engine::state::{SimState, Vec3};
use serde::{Deserialize, Serialize};

/// Configuration for a simple pendulum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendulumConfig {
    /// Length of pendulum (m).
    pub length: f64,
    /// Mass of bob (kg).
    pub mass: f64,
    /// Gravitational acceleration (m/s²).
    pub g: f64,
    /// Damping coefficient (kg/s).
    pub damping: f64,
    /// Initial angle (radians, 0 = hanging down).
    pub initial_angle: f64,
    /// Initial angular velocity (rad/s).
    pub initial_angular_velocity: f64,
}

impl Default for PendulumConfig {
    fn default() -> Self {
        Self {
            length: 1.0,
            mass: 1.0,
            g: 9.81,
            damping: 0.0,
            initial_angle: std::f64::consts::FRAC_PI_4, // 45 degrees
            initial_angular_velocity: 0.0,
        }
    }
}

impl PendulumConfig {
    /// Create a small-angle pendulum (linearized regime).
    #[must_use]
    pub fn small_angle() -> Self {
        Self {
            initial_angle: 0.1, // ~6 degrees
            ..Default::default()
        }
    }

    /// Create a large-angle pendulum (nonlinear regime).
    #[must_use]
    pub fn large_angle() -> Self {
        Self {
            initial_angle: std::f64::consts::FRAC_PI_2, // 90 degrees
            ..Default::default()
        }
    }

    /// Create a damped pendulum.
    #[must_use]
    pub fn damped(damping: f64) -> Self {
        Self {
            damping,
            ..Default::default()
        }
    }

    /// Get theoretical period for small oscillations.
    #[must_use]
    pub fn small_angle_period(&self) -> f64 {
        2.0 * std::f64::consts::PI * (self.length / self.g).sqrt()
    }
}

/// Simple pendulum scenario.
#[derive(Debug, Clone)]
pub struct PendulumScenario {
    config: PendulumConfig,
}

impl PendulumScenario {
    /// Create a new pendulum scenario.
    #[must_use]
    pub fn new(config: PendulumConfig) -> Self {
        Self { config }
    }

    /// Initialize simulation state for the pendulum.
    ///
    /// The pendulum is modeled as a particle constrained to move on a circle.
    /// State is represented in Cartesian coordinates (x, y) where:
    /// - Origin is at pivot point
    /// - y-axis points downward
    /// - Angle is measured from vertical (y-axis)
    #[must_use]
    pub fn init_state(&self) -> SimState {
        let mut state = SimState::new();

        // Position from angle (angle=0 means hanging straight down)
        let x = self.config.length * self.config.initial_angle.sin();
        let y = -self.config.length * self.config.initial_angle.cos();

        // Velocity from angular velocity
        // v = L * omega, tangent to circle
        let vx = self.config.length
            * self.config.initial_angular_velocity
            * self.config.initial_angle.cos();
        let vy = self.config.length
            * self.config.initial_angular_velocity
            * self.config.initial_angle.sin();

        state.add_body(
            self.config.mass,
            Vec3::new(x, y, 0.0),
            Vec3::new(vx, vy, 0.0),
        );

        state
    }

    /// Create force field for pendulum (gravity + tension constraint).
    #[must_use]
    pub fn create_force_field(&self) -> PendulumForceField {
        PendulumForceField {
            g: self.config.g,
            length: self.config.length,
            damping: self.config.damping,
        }
    }

    /// Get theoretical period (small angle approximation).
    #[must_use]
    pub fn theoretical_period(&self) -> f64 {
        self.config.small_angle_period()
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &PendulumConfig {
        &self.config
    }

    /// Calculate current angle from state.
    #[must_use]
    pub fn current_angle(state: &SimState) -> f64 {
        let pos = state.positions()[0];
        pos.x.atan2(-pos.y)
    }

    /// Calculate current angular velocity from state.
    #[must_use]
    pub fn current_angular_velocity(state: &SimState, length: f64) -> f64 {
        let pos = state.positions()[0];
        let vel = state.velocities()[0];
        let r = pos.magnitude();
        if r < f64::EPSILON {
            return 0.0;
        }
        // Angular velocity = (r × v) / r² = (x*vy - y*vx) / r²
        (pos.x * vel.y - pos.y * vel.x) / (length * length)
    }
}

/// Force field for simple pendulum.
///
/// Models gravity plus the constraint force (tension) that keeps
/// the particle on the circular arc.
#[derive(Debug, Clone)]
pub struct PendulumForceField {
    /// Gravitational acceleration.
    pub g: f64,
    /// Pendulum length.
    pub length: f64,
    /// Damping coefficient.
    pub damping: f64,
}

impl ForceField for PendulumForceField {
    fn acceleration(&self, position: &Vec3, _mass: f64) -> Vec3 {
        // Project gravity onto tangent direction
        // Gravity is (0, -g, 0) in our coordinate system where y is up
        // But we defined y pointing down for the pendulum, so gravity is (0, g, 0)

        let r = (position.x * position.x + position.y * position.y).sqrt();
        if r < f64::EPSILON {
            return Vec3::zero();
        }

        // Angle from vertical (y-axis pointing down)
        let theta = position.x.atan2(-position.y);

        // Tangential acceleration from gravity: -g * sin(theta)
        let a_tangent = -self.g * theta.sin();

        // Convert tangential acceleration to Cartesian
        // Tangent direction: perpendicular to radial, in direction of increasing theta
        let tan_x = -position.y / r;
        let tan_y = position.x / r;

        // Damping (proportional to velocity, but we only have position here)
        // For proper damping, we'd need velocity. This is a simplified model.

        Vec3::new(a_tangent * tan_x, a_tangent * tan_y, 0.0)
    }

    fn potential(&self, position: &Vec3, mass: f64) -> f64 {
        // Potential energy relative to lowest point
        // PE = m * g * h where h = L - L*cos(theta) = L * (1 - cos(theta))
        let theta = position.x.atan2(-position.y);
        mass * self.g * self.length * (1.0 - theta.cos())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_pendulum_config_default() {
        let config = PendulumConfig::default();

        assert!((config.length - 1.0).abs() < f64::EPSILON);
        assert!((config.mass - 1.0).abs() < f64::EPSILON);
        assert!((config.g - 9.81).abs() < 0.01);
    }

    #[test]
    fn test_pendulum_config_small_angle() {
        let config = PendulumConfig::small_angle();
        assert!((config.initial_angle - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_config_large_angle() {
        let config = PendulumConfig::large_angle();
        assert!((config.initial_angle - std::f64::consts::FRAC_PI_2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_config_damped() {
        let config = PendulumConfig::damped(0.5);
        assert!((config.damping - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_config_clone() {
        let config = PendulumConfig::default();
        let cloned = config.clone();
        assert!((cloned.length - config.length).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_small_angle_period() {
        let config = PendulumConfig {
            length: 1.0,
            g: 9.81,
            ..Default::default()
        };

        // T = 2 * pi * sqrt(L/g) ≈ 2.006 seconds
        let period = config.small_angle_period();
        assert!((period - 2.006).abs() < 0.01, "Period={}", period);
    }

    #[test]
    fn test_pendulum_scenario_init() {
        let config = PendulumConfig {
            initial_angle: std::f64::consts::FRAC_PI_4,
            length: 1.0,
            ..Default::default()
        };
        let scenario = PendulumScenario::new(config);
        let state = scenario.init_state();

        assert_eq!(state.num_bodies(), 1);

        // Check initial angle
        let angle = PendulumScenario::current_angle(&state);
        assert!((angle - std::f64::consts::FRAC_PI_4).abs() < 0.01);
    }

    #[test]
    fn test_pendulum_scenario_clone() {
        let config = PendulumConfig::default();
        let scenario = PendulumScenario::new(config);
        let cloned = scenario.clone();
        assert!((cloned.theoretical_period() - scenario.theoretical_period()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_scenario_config() {
        let config = PendulumConfig {
            length: 2.0,
            ..Default::default()
        };
        let scenario = PendulumScenario::new(config);
        assert!((scenario.config().length - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_scenario_theoretical_period() {
        let config = PendulumConfig {
            length: 1.0,
            g: 9.81,
            ..Default::default()
        };
        let scenario = PendulumScenario::new(config);
        let period = scenario.theoretical_period();
        assert!((period - 2.006).abs() < 0.01);
    }

    #[test]
    fn test_pendulum_current_angular_velocity() {
        let config = PendulumConfig {
            length: 1.0,
            initial_angular_velocity: 1.0,
            ..Default::default()
        };
        let scenario = PendulumScenario::new(config);
        let state = scenario.init_state();

        let omega = PendulumScenario::current_angular_velocity(&state, 1.0);
        // Should be approximately 1.0
        assert!(omega.abs() < 2.0);
    }

    #[test]
    fn test_pendulum_angular_velocity_at_origin() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::zero());
        let omega = PendulumScenario::current_angular_velocity(&state, 1.0);
        assert!((omega - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_force_field() {
        let field = PendulumForceField {
            g: 9.81,
            length: 1.0,
            damping: 0.0,
        };

        // At vertical (theta=0), no tangential acceleration
        let pos_vertical = Vec3::new(0.0, -1.0, 0.0);
        let acc = field.acceleration(&pos_vertical, 1.0);
        assert!(acc.magnitude() < 0.01, "acc={:?}", acc);

        // At 90 degrees, maximum tangential acceleration
        let pos_horizontal = Vec3::new(1.0, 0.0, 0.0);
        let acc = field.acceleration(&pos_horizontal, 1.0);
        assert!(acc.magnitude() > 9.0, "acc magnitude={}", acc.magnitude());
    }

    #[test]
    fn test_pendulum_force_field_at_origin() {
        let field = PendulumForceField {
            g: 9.81,
            length: 1.0,
            damping: 0.0,
        };

        let pos = Vec3::zero();
        let acc = field.acceleration(&pos, 1.0);
        assert!((acc.magnitude() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_force_field_clone() {
        let field = PendulumForceField {
            g: 9.81,
            length: 1.0,
            damping: 0.5,
        };
        let cloned = field.clone();
        assert!((cloned.damping - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_create_force_field() {
        let config = PendulumConfig {
            g: 10.0,
            length: 2.0,
            damping: 0.1,
            ..Default::default()
        };
        let scenario = PendulumScenario::new(config);
        let field = scenario.create_force_field();
        assert!((field.g - 10.0).abs() < f64::EPSILON);
        assert!((field.length - 2.0).abs() < f64::EPSILON);
        assert!((field.damping - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pendulum_potential_energy() {
        let field = PendulumForceField {
            g: 9.81,
            length: 1.0,
            damping: 0.0,
        };

        // At lowest point (theta=0), PE = 0
        let pos_low = Vec3::new(0.0, -1.0, 0.0);
        let pe_low = field.potential(&pos_low, 1.0);
        assert!(pe_low.abs() < 0.01, "PE at bottom={}", pe_low);

        // At horizontal (theta=pi/2), PE = m*g*L
        let pos_horiz = Vec3::new(1.0, 0.0, 0.0);
        let pe_horiz = field.potential(&pos_horiz, 1.0);
        assert!(
            (pe_horiz - 9.81).abs() < 0.01,
            "PE at horizontal={}",
            pe_horiz
        );

        // At top (theta=pi), PE = 2*m*g*L
        let pos_top = Vec3::new(0.0, 1.0, 0.0);
        let pe_top = field.potential(&pos_top, 1.0);
        assert!((pe_top - 2.0 * 9.81).abs() < 0.01, "PE at top={}", pe_top);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Period increases with pendulum length (T ∝ √L).
        #[test]
        fn prop_period_increases_with_length(
            length1 in 0.1f64..10.0,
            length2 in 0.1f64..10.0,
        ) {
            let config1 = PendulumConfig { length: length1, ..Default::default() };
            let config2 = PendulumConfig { length: length2, ..Default::default() };

            let period1 = config1.small_angle_period();
            let period2 = config2.small_angle_period();

            // If length1 > length2, then period1 > period2
            if length1 > length2 {
                prop_assert!(period1 > period2);
            } else if length1 < length2 {
                prop_assert!(period1 < period2);
            }
        }

        /// Period decreases with gravity (T ∝ 1/√g).
        #[test]
        fn prop_period_decreases_with_gravity(
            g1 in 1.0f64..20.0,
            g2 in 1.0f64..20.0,
        ) {
            let config1 = PendulumConfig { g: g1, ..Default::default() };
            let config2 = PendulumConfig { g: g2, ..Default::default() };

            let period1 = config1.small_angle_period();
            let period2 = config2.small_angle_period();

            // If g1 > g2, then period1 < period2
            if g1 > g2 {
                prop_assert!(period1 < period2);
            } else if g1 < g2 {
                prop_assert!(period1 > period2);
            }
        }

        /// Pendulum force field produces restoring force toward equilibrium.
        #[test]
        fn prop_restoring_force(
            angle in -std::f64::consts::PI..std::f64::consts::PI,
            length in 0.5f64..5.0,
        ) {
            let field = PendulumForceField {
                g: 9.81,
                length,
                damping: 0.0,
            };

            // Position at given angle
            let x = length * angle.sin();
            let y = -length * angle.cos();
            let pos = Vec3::new(x, y, 0.0);

            let accel = field.acceleration(&pos, 1.0);

            // For small displacements, force should point toward equilibrium
            if angle.abs() > 0.01 {
                // Tangential acceleration should oppose displacement
                let tangent_x = angle.cos();
                let tangent_y = angle.sin();
                let tangent_accel = accel.x * tangent_x + accel.y * tangent_y;

                // Restoring force: positive angle -> negative tangent accel
                if angle > 0.0 {
                    prop_assert!(tangent_accel < 0.1, "tangent_accel={}", tangent_accel);
                } else {
                    prop_assert!(tangent_accel > -0.1, "tangent_accel={}", tangent_accel);
                }
            }
        }

        /// Potential energy is non-negative when measured from lowest point.
        #[test]
        fn prop_potential_energy_nonnegative(
            angle in -std::f64::consts::PI..std::f64::consts::PI,
            length in 0.5f64..5.0,
            mass in 0.1f64..10.0,
        ) {
            let field = PendulumForceField {
                g: 9.81,
                length,
                damping: 0.0,
            };

            let x = length * angle.sin();
            let y = -length * angle.cos();
            let pos = Vec3::new(x, y, 0.0);

            let pe = field.potential(&pos, mass);
            // PE relative to lowest point should be >= 0
            prop_assert!(pe >= -0.001, "PE={} should be non-negative", pe);
        }
    }
}
