//! Jidoka (自働化) - Graceful degradation anomaly detection.
//!
//! Implements Toyota's Jidoka principle with graceful degradation:
//! - Pause simulation and highlight defects (don't crash)
//! - Allow user intervention and recovery
//! - Visual management (Mieruka) of status
//!
//! # TPS Review Feedback (v1.1.0)
//!
//! Jidoka should **pause, not crash** — the system stops and highlights
//! the defect while remaining available for intervention [27].
//!
//! # References
//!
//! [27] Avizienis et al., "Dependable and Secure Computing," IEEE TDSC, 2004.

use crate::orbit::physics::NBodyState;
use serde::{Deserialize, Serialize};

/// Jidoka response types with graceful degradation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JidokaResponse {
    /// All checks passed, continue normally.
    Continue,

    /// Warning detected, continue with visual indicator.
    Warning {
        message: String,
        body_index: Option<usize>,
        metric: String,
        current: f64,
        threshold: f64,
    },

    /// Critical violation, pause simulation for user intervention.
    Pause {
        violation: OrbitJidokaViolation,
        recoverable: bool,
        suggestion: String,
    },

    /// Fatal unrecoverable error, halt with state snapshot.
    Halt { violation: OrbitJidokaViolation },
}

impl JidokaResponse {
    /// Check if this response allows continuation.
    #[must_use]
    pub fn can_continue(&self) -> bool {
        matches!(self, Self::Continue | Self::Warning { .. })
    }

    /// Check if this is a warning.
    #[must_use]
    pub fn is_warning(&self) -> bool {
        matches!(self, Self::Warning { .. })
    }

    /// Check if simulation should pause.
    #[must_use]
    pub fn should_pause(&self) -> bool {
        matches!(self, Self::Pause { .. })
    }

    /// Check if simulation should halt.
    #[must_use]
    pub fn should_halt(&self) -> bool {
        matches!(self, Self::Halt { .. })
    }
}

/// Jidoka violation types for orbital simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrbitJidokaViolation {
    /// Non-finite value (NaN or Inf) detected.
    NonFinite {
        body_index: usize,
        field: String,
        value: f64,
    },

    /// Energy conservation violated.
    EnergyDrift {
        initial: f64,
        current: f64,
        relative_error: f64,
        tolerance: f64,
    },

    /// Angular momentum conservation violated.
    AngularMomentumDrift {
        initial: f64,
        current: f64,
        relative_error: f64,
        tolerance: f64,
    },

    /// Close encounter (collision risk).
    CloseEncounter {
        body_i: usize,
        body_j: usize,
        separation: f64,
        threshold: f64,
    },

    /// Escape velocity exceeded (body leaving system).
    EscapeVelocity {
        body_index: usize,
        velocity: f64,
        escape_velocity: f64,
    },
}

impl std::fmt::Display for OrbitJidokaViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite {
                body_index,
                field,
                value,
            } => {
                write!(f, "Non-finite {field} at body {body_index}: {value}")
            }
            Self::EnergyDrift {
                relative_error,
                tolerance,
                ..
            } => {
                write!(
                    f,
                    "Energy drift {relative_error:.2e} exceeds tolerance {tolerance:.2e}"
                )
            }
            Self::AngularMomentumDrift {
                relative_error,
                tolerance,
                ..
            } => {
                write!(
                    f,
                    "Angular momentum drift {relative_error:.2e} exceeds tolerance {tolerance:.2e}"
                )
            }
            Self::CloseEncounter {
                body_i,
                body_j,
                separation,
                threshold,
            } => {
                write!(f, "Close encounter: bodies {body_i}-{body_j} at {separation:.2e}m (threshold: {threshold:.2e}m)")
            }
            Self::EscapeVelocity {
                body_index,
                velocity,
                escape_velocity,
            } => {
                write!(f, "Body {body_index} at escape velocity: {velocity:.2e} > {escape_velocity:.2e} m/s")
            }
        }
    }
}

/// Jidoka configuration for orbital simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)] // Config struct with feature flags
pub struct OrbitJidokaConfig {
    /// Check for non-finite values.
    pub check_finite: bool,
    /// Check energy conservation.
    pub check_energy: bool,
    /// Energy drift tolerance (relative).
    pub energy_tolerance: f64,
    /// Energy warning threshold (fraction of tolerance).
    pub energy_warning_fraction: f64,
    /// Check angular momentum conservation.
    pub check_angular_momentum: bool,
    /// Angular momentum tolerance (relative).
    pub angular_momentum_tolerance: f64,
    /// Check for close encounters.
    pub check_close_encounters: bool,
    /// Close encounter threshold (meters).
    pub close_encounter_threshold: f64,
    /// Maximum warnings before pausing.
    pub max_warnings_before_pause: usize,
}

impl Default for OrbitJidokaConfig {
    fn default() -> Self {
        Self {
            check_finite: true,
            check_energy: true,
            energy_tolerance: 1e-6,
            energy_warning_fraction: 0.8,
            check_angular_momentum: true,
            angular_momentum_tolerance: 1e-9,
            check_close_encounters: true,
            close_encounter_threshold: 1e6, // 1000 km
            max_warnings_before_pause: 10,
        }
    }
}

/// Jidoka status for visualization (Mieruka).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)] // Status struct with multiple checks
pub struct JidokaStatus {
    /// Energy relative error.
    pub energy_error: f64,
    /// Energy check passed.
    pub energy_ok: bool,
    /// Angular momentum relative error.
    pub angular_momentum_error: f64,
    /// Angular momentum check passed.
    pub angular_momentum_ok: bool,
    /// All values finite.
    pub finite_ok: bool,
    /// Minimum separation between bodies.
    pub min_separation: f64,
    /// Close encounter warning.
    pub close_encounter_warning: bool,
    /// Total warning count.
    pub warning_count: usize,
}

/// Orbital Jidoka guard with graceful degradation.
#[derive(Debug, Clone)]
pub struct OrbitJidokaGuard {
    config: OrbitJidokaConfig,
    initial_energy: Option<f64>,
    initial_angular_momentum: Option<f64>,
    warning_count: usize,
    status: JidokaStatus,
}

impl OrbitJidokaGuard {
    /// Create a new Jidoka guard.
    #[must_use]
    pub fn new(config: OrbitJidokaConfig) -> Self {
        Self {
            config,
            initial_energy: None,
            initial_angular_momentum: None,
            warning_count: 0,
            status: JidokaStatus::default(),
        }
    }

    /// Initialize with initial state values.
    pub fn initialize(&mut self, state: &NBodyState) {
        self.initial_energy = Some(state.total_energy());
        self.initial_angular_momentum = Some(state.angular_momentum_magnitude());
        self.warning_count = 0;
    }

    /// Reset warning count (e.g., after user intervention).
    pub fn reset_warnings(&mut self) {
        self.warning_count = 0;
    }

    /// Get current status for visualization.
    #[must_use]
    pub fn status(&self) -> &JidokaStatus {
        &self.status
    }

    /// Check state and return response (graceful degradation).
    pub fn check(&mut self, state: &NBodyState) -> JidokaResponse {
        // Reset status
        self.status = JidokaStatus::default();

        // 1. Check finite values (fatal if violated)
        if self.config.check_finite {
            if let Some(response) = self.check_finite(state) {
                return response;
            }
        }

        // 2. Check energy conservation
        if self.config.check_energy {
            if let Some(response) = self.check_energy(state) {
                return response;
            }
        }

        // 3. Check angular momentum conservation
        if self.config.check_angular_momentum {
            if let Some(response) = self.check_angular_momentum(state) {
                return response;
            }
        }

        // 4. Check close encounters
        if self.config.check_close_encounters {
            if let Some(response) = self.check_close_encounters(state) {
                return response;
            }
        }

        // All checks passed
        JidokaResponse::Continue
    }

    fn check_finite(&mut self, state: &NBodyState) -> Option<JidokaResponse> {
        for (i, body) in state.bodies.iter().enumerate() {
            if !body.position.is_finite() {
                self.status.finite_ok = false;
                return Some(JidokaResponse::Halt {
                    violation: OrbitJidokaViolation::NonFinite {
                        body_index: i,
                        field: "position".to_string(),
                        value: f64::NAN,
                    },
                });
            }
            if !body.velocity.is_finite() {
                self.status.finite_ok = false;
                return Some(JidokaResponse::Halt {
                    violation: OrbitJidokaViolation::NonFinite {
                        body_index: i,
                        field: "velocity".to_string(),
                        value: f64::NAN,
                    },
                });
            }
        }
        self.status.finite_ok = true;
        None
    }

    fn check_energy(&mut self, state: &NBodyState) -> Option<JidokaResponse> {
        let initial = self.initial_energy?;

        let current = state.total_energy();
        let relative_error = if initial.abs() > f64::EPSILON {
            (current - initial).abs() / initial.abs()
        } else {
            (current - initial).abs()
        };

        self.status.energy_error = relative_error;
        self.status.energy_ok = relative_error <= self.config.energy_tolerance;

        // Critical violation
        if relative_error > self.config.energy_tolerance {
            self.warning_count += 1;

            if self.warning_count >= self.config.max_warnings_before_pause {
                return Some(JidokaResponse::Pause {
                    violation: OrbitJidokaViolation::EnergyDrift {
                        initial,
                        current,
                        relative_error,
                        tolerance: self.config.energy_tolerance,
                    },
                    recoverable: true,
                    suggestion: "Consider reducing time step or using Yoshida integrator"
                        .to_string(),
                });
            }

            return Some(JidokaResponse::Warning {
                message: format!("Energy drift: {relative_error:.2e}"),
                body_index: None,
                metric: "energy".to_string(),
                current: relative_error,
                threshold: self.config.energy_tolerance,
            });
        }

        // Warning threshold
        let warning_threshold = self.config.energy_tolerance * self.config.energy_warning_fraction;
        if relative_error > warning_threshold {
            return Some(JidokaResponse::Warning {
                message: format!("Energy approaching tolerance: {relative_error:.2e}"),
                body_index: None,
                metric: "energy".to_string(),
                current: relative_error,
                threshold: self.config.energy_tolerance,
            });
        }

        None
    }

    fn check_angular_momentum(&mut self, state: &NBodyState) -> Option<JidokaResponse> {
        let initial = self.initial_angular_momentum?;

        let current = state.angular_momentum_magnitude();
        let relative_error = if initial.abs() > f64::EPSILON {
            (current - initial).abs() / initial.abs()
        } else {
            (current - initial).abs()
        };

        self.status.angular_momentum_error = relative_error;
        self.status.angular_momentum_ok = relative_error <= self.config.angular_momentum_tolerance;

        if relative_error > self.config.angular_momentum_tolerance {
            self.warning_count += 1;

            if self.warning_count >= self.config.max_warnings_before_pause {
                return Some(JidokaResponse::Pause {
                    violation: OrbitJidokaViolation::AngularMomentumDrift {
                        initial,
                        current,
                        relative_error,
                        tolerance: self.config.angular_momentum_tolerance,
                    },
                    recoverable: true,
                    suggestion:
                        "Angular momentum should be conserved - check for numerical instability"
                            .to_string(),
                });
            }
        }

        None
    }

    fn check_close_encounters(&mut self, state: &NBodyState) -> Option<JidokaResponse> {
        let min_sep = state.min_separation();
        self.status.min_separation = min_sep;
        self.status.close_encounter_warning = min_sep < self.config.close_encounter_threshold;

        if min_sep < self.config.close_encounter_threshold {
            // Find the close pair
            let n = state.bodies.len();
            for i in 0..n {
                for j in (i + 1)..n {
                    let r = state.bodies[i].position - state.bodies[j].position;
                    let sep = r.magnitude().get::<uom::si::length::meter>();

                    if sep < self.config.close_encounter_threshold {
                        self.warning_count += 1;

                        if self.warning_count >= self.config.max_warnings_before_pause {
                            return Some(JidokaResponse::Pause {
                                violation: OrbitJidokaViolation::CloseEncounter {
                                    body_i: i,
                                    body_j: j,
                                    separation: sep,
                                    threshold: self.config.close_encounter_threshold,
                                },
                                recoverable: true,
                                suggestion:
                                    "Reduce time step or enable softening for close encounters"
                                        .to_string(),
                            });
                        }

                        return Some(JidokaResponse::Warning {
                            message: format!("Close encounter: bodies {i}-{j} at {sep:.2e}m"),
                            body_index: Some(i),
                            metric: "separation".to_string(),
                            current: sep,
                            threshold: self.config.close_encounter_threshold,
                        });
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::physics::{NBodyState, OrbitBody};
    use crate::orbit::units::{OrbitMass, Position3D, Velocity3D, AU, EARTH_MASS, G, SOLAR_MASS};

    fn create_sun_earth_system() -> NBodyState {
        let v_circular = (G * SOLAR_MASS / AU).sqrt();
        let bodies = vec![
            OrbitBody::new(
                OrbitMass::from_kg(SOLAR_MASS),
                Position3D::zero(),
                Velocity3D::zero(),
            ),
            OrbitBody::new(
                OrbitMass::from_kg(EARTH_MASS),
                Position3D::from_au(1.0, 0.0, 0.0),
                Velocity3D::from_mps(0.0, v_circular, 0.0),
            ),
        ];
        NBodyState::new(bodies, 0.0)
    }

    #[test]
    fn test_jidoka_response_can_continue() {
        assert!(JidokaResponse::Continue.can_continue());
        assert!(JidokaResponse::Warning {
            message: "test".to_string(),
            body_index: None,
            metric: "test".to_string(),
            current: 0.0,
            threshold: 1.0,
        }
        .can_continue());

        assert!(!JidokaResponse::Pause {
            violation: OrbitJidokaViolation::EnergyDrift {
                initial: 1.0,
                current: 2.0,
                relative_error: 1.0,
                tolerance: 0.1,
            },
            recoverable: true,
            suggestion: String::new(),
        }
        .can_continue());
    }

    #[test]
    fn test_jidoka_config_default() {
        let config = OrbitJidokaConfig::default();
        assert!(config.check_finite);
        assert!(config.check_energy);
        assert!(config.check_angular_momentum);
        assert!((config.energy_tolerance - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_jidoka_guard_initialize() {
        let config = OrbitJidokaConfig::default();
        let mut guard = OrbitJidokaGuard::new(config);
        let state = create_sun_earth_system();

        guard.initialize(&state);

        assert!(guard.initial_energy.is_some());
        assert!(guard.initial_angular_momentum.is_some());
    }

    #[test]
    fn test_jidoka_guard_check_healthy_state() {
        let config = OrbitJidokaConfig::default();
        let mut guard = OrbitJidokaGuard::new(config);
        let state = create_sun_earth_system();

        guard.initialize(&state);
        let response = guard.check(&state);

        assert!(response.can_continue());
        assert!(guard.status().energy_ok);
        assert!(guard.status().angular_momentum_ok);
        assert!(guard.status().finite_ok);
    }

    #[test]
    fn test_jidoka_guard_energy_violation() {
        let mut config = OrbitJidokaConfig::default();
        config.energy_tolerance = 1e-20; // Impossibly tight tolerance
        config.max_warnings_before_pause = 1;

        let mut guard = OrbitJidokaGuard::new(config);
        let mut state = create_sun_earth_system();

        guard.initialize(&state);

        // Perturb energy by changing velocity
        state.bodies[1].velocity = Velocity3D::from_mps(0.0, 30000.0, 0.0);

        let response = guard.check(&state);

        // Should pause after violation
        assert!(response.should_pause() || response.is_warning());
    }

    #[test]
    fn test_jidoka_guard_close_encounter() {
        let mut config = OrbitJidokaConfig::default();
        config.close_encounter_threshold = 1e15; // Very large threshold
        config.max_warnings_before_pause = 1;

        let mut guard = OrbitJidokaGuard::new(config);
        let state = create_sun_earth_system();

        guard.initialize(&state);
        let response = guard.check(&state);

        // Should warn about close encounter (Earth is within 1e15m of Sun)
        assert!(response.is_warning() || response.should_pause());
    }

    #[test]
    fn test_jidoka_violation_display() {
        let violation = OrbitJidokaViolation::EnergyDrift {
            initial: -1e33,
            current: -1.1e33,
            relative_error: 0.1,
            tolerance: 1e-6,
        };

        let display = format!("{violation}");
        assert!(display.contains("Energy drift"));
        assert!(display.contains("1.00e-1"));
    }

    #[test]
    fn test_jidoka_status_default() {
        let status = JidokaStatus::default();
        assert!(!status.energy_ok);
        assert!(!status.angular_momentum_ok);
        assert!(!status.finite_ok);
        assert_eq!(status.warning_count, 0);
    }

    #[test]
    fn test_jidoka_reset_warnings() {
        let config = OrbitJidokaConfig::default();
        let mut guard = OrbitJidokaGuard::new(config);

        guard.warning_count = 5;
        guard.reset_warnings();

        assert_eq!(guard.warning_count, 0);
    }

    #[test]
    fn test_jidoka_warning_accumulation() {
        let mut config = OrbitJidokaConfig::default();
        config.energy_tolerance = 1e-20; // Tight tolerance
        config.max_warnings_before_pause = 5;

        let mut guard = OrbitJidokaGuard::new(config);
        let mut state = create_sun_earth_system();

        guard.initialize(&state);
        state.bodies[1].velocity = Velocity3D::from_mps(0.0, 35000.0, 0.0);

        // Accumulate warnings
        for _ in 0..4 {
            let response = guard.check(&state);
            assert!(response.is_warning());
        }

        // Fifth check should pause
        let response = guard.check(&state);
        assert!(response.should_pause());
    }

    #[test]
    fn test_jidoka_response_is_warning() {
        assert!(!JidokaResponse::Continue.is_warning());
        assert!(JidokaResponse::Warning {
            message: "test".to_string(),
            body_index: None,
            metric: "test".to_string(),
            current: 0.0,
            threshold: 1.0,
        }
        .is_warning());
    }

    #[test]
    fn test_jidoka_response_should_pause() {
        assert!(!JidokaResponse::Continue.should_pause());
        assert!(JidokaResponse::Pause {
            violation: OrbitJidokaViolation::EnergyDrift {
                initial: 1.0,
                current: 2.0,
                relative_error: 1.0,
                tolerance: 0.1,
            },
            recoverable: true,
            suggestion: String::new(),
        }
        .should_pause());
    }

    #[test]
    fn test_jidoka_response_should_halt() {
        assert!(!JidokaResponse::Continue.should_halt());
        assert!(JidokaResponse::Halt {
            violation: OrbitJidokaViolation::NonFinite {
                body_index: 0,
                field: "position".to_string(),
                value: f64::NAN,
            },
        }
        .should_halt());
    }

    #[test]
    fn test_jidoka_violation_display_non_finite() {
        let violation = OrbitJidokaViolation::NonFinite {
            body_index: 1,
            field: "velocity".to_string(),
            value: f64::INFINITY,
        };
        let display = format!("{violation}");
        assert!(display.contains("Non-finite"));
        assert!(display.contains("velocity"));
        assert!(display.contains("body 1"));
    }

    #[test]
    fn test_jidoka_violation_display_angular_momentum() {
        let violation = OrbitJidokaViolation::AngularMomentumDrift {
            initial: 1e40,
            current: 1.1e40,
            relative_error: 0.1,
            tolerance: 1e-9,
        };
        let display = format!("{violation}");
        assert!(display.contains("Angular momentum drift"));
    }

    #[test]
    fn test_jidoka_violation_display_close_encounter() {
        let violation = OrbitJidokaViolation::CloseEncounter {
            body_i: 0,
            body_j: 1,
            separation: 1e6,
            threshold: 1e7,
        };
        let display = format!("{violation}");
        assert!(display.contains("Close encounter"));
        assert!(display.contains("0-1"));
    }

    #[test]
    fn test_jidoka_violation_display_escape() {
        let violation = OrbitJidokaViolation::EscapeVelocity {
            body_index: 1,
            velocity: 50000.0,
            escape_velocity: 42000.0,
        };
        let display = format!("{violation}");
        assert!(display.contains("escape velocity"));
        assert!(display.contains("Body 1"));
    }

    #[test]
    fn test_jidoka_guard_halt_on_non_finite_position() {
        let config = OrbitJidokaConfig::default();
        let mut guard = OrbitJidokaGuard::new(config);
        let mut state = create_sun_earth_system();

        guard.initialize(&state);

        // Inject NaN into position
        state.bodies[0].position = Position3D::from_meters(f64::NAN, 0.0, 0.0);

        let response = guard.check(&state);
        assert!(response.should_halt());
    }

    #[test]
    fn test_jidoka_guard_halt_on_non_finite_velocity() {
        let config = OrbitJidokaConfig::default();
        let mut guard = OrbitJidokaGuard::new(config);
        let mut state = create_sun_earth_system();

        guard.initialize(&state);

        // Inject Inf into velocity
        state.bodies[1].velocity = Velocity3D::from_mps(f64::INFINITY, 0.0, 0.0);

        let response = guard.check(&state);
        assert!(response.should_halt());
    }

    #[test]
    fn test_jidoka_guard_check_without_initialize() {
        let config = OrbitJidokaConfig::default();
        let mut guard = OrbitJidokaGuard::new(config);
        let state = create_sun_earth_system();

        // Check without initializing - should still work for finite checks
        let response = guard.check(&state);
        assert!(response.can_continue());
    }

    #[test]
    fn test_jidoka_guard_disabled_checks() {
        let config = OrbitJidokaConfig {
            check_finite: false,
            check_energy: false,
            check_angular_momentum: false,
            check_close_encounters: false,
            ..Default::default()
        };
        let mut guard = OrbitJidokaGuard::new(config);
        let mut state = create_sun_earth_system();

        guard.initialize(&state);

        // Even with NaN, should continue because checks are disabled
        state.bodies[0].position = Position3D::from_meters(f64::NAN, 0.0, 0.0);

        let response = guard.check(&state);
        assert!(response.can_continue());
    }

    #[test]
    fn test_jidoka_energy_warning_threshold() {
        let mut config = OrbitJidokaConfig::default();
        config.energy_tolerance = 1e-3;
        config.energy_warning_fraction = 0.8; // Warn at 80%

        let mut guard = OrbitJidokaGuard::new(config);
        let mut state = create_sun_earth_system();

        guard.initialize(&state);

        // Small perturbation to trigger warning but not violation
        state.bodies[1].velocity = Velocity3D::from_mps(0.0, 29784.0, 0.0);

        let response = guard.check(&state);
        // Should be continue or warning, not pause
        assert!(response.can_continue());
    }

    #[test]
    fn test_jidoka_angular_momentum_violation() {
        let mut config = OrbitJidokaConfig::default();
        config.angular_momentum_tolerance = 1e-20; // Impossible tolerance
        config.max_warnings_before_pause = 1;

        let mut guard = OrbitJidokaGuard::new(config);
        let mut state = create_sun_earth_system();

        guard.initialize(&state);

        // Change velocity direction to alter angular momentum
        state.bodies[1].velocity = Velocity3D::from_mps(29784.0, 0.0, 0.0);

        let response = guard.check(&state);
        // Should pause on angular momentum violation
        assert!(response.should_pause());
    }
}
