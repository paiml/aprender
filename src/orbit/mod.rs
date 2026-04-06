//! Simular Orbit Demo module.
//!
//! Implements orbital mechanics demonstration with:
//! - Type-safe units (Poka-Yoke)
//! - Symplectic integrators (Yoshida 4th order)
//! - Jidoka guards with graceful degradation
//! - Heijunka time-budget scheduler
//! - Pre-built scenarios (Kepler, N-body, Hohmann, Lagrange)
//!
//! # Toyota Way Principles
//!
//! - **Jidoka (自働化)**: Graceful degradation anomaly detection
//! - **Poka-Yoke (ポカヨケ)**: Type-safe dimensional analysis
//! - **Heijunka (平準化)**: Load-leveled frame delivery
//! - **Mieruka (見える化)**: Visual status management
//!
//! # NASA/JPL Standards
//!
//! Follows Power of 10 rules for mission-critical software [1].
//!
//! # Example
//!
//! ```rust
//! use simular::orbit::prelude::*;
//!
//! // Create Earth-Sun Keplerian orbit
//! let config = KeplerConfig::earth_sun();
//! let mut state = config.build(1e6);
//!
//! // Initialize guards
//! let mut jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
//! jidoka.initialize(&state);
//!
//! // Create integrator
//! let yoshida = YoshidaIntegrator::new();
//!
//! // Simulate one day
//! let dt = OrbitTime::from_seconds(3600.0);
//! for _ in 0..24 {
//!     yoshida.step(&mut state, dt).expect("step failed");
//!     let response = jidoka.check(&state);
//!     if !response.can_continue() {
//!         break;
//!     }
//! }
//! ```

pub mod heijunka;
pub mod jidoka;
pub mod metamorphic;
pub mod physics;
pub mod render;
pub mod scenarios;
pub mod units;

#[cfg(feature = "wasm")]
pub mod wasm;

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::heijunka::{
        FrameResult, HeijunkaConfig, HeijunkaScheduler, HeijunkaStatus, QualityLevel,
    };
    pub use super::jidoka::{
        JidokaResponse, JidokaStatus, OrbitJidokaConfig, OrbitJidokaGuard, OrbitJidokaViolation,
    };
    pub use super::metamorphic::{
        run_all_metamorphic_tests, test_angular_momentum_conservation, test_deterministic_replay,
        test_energy_conservation, test_rotation_invariance, test_time_reversal, MetamorphicResult,
    };
    pub use super::physics::{AdaptiveIntegrator, NBodyState, OrbitBody, YoshidaIntegrator};
    pub use super::scenarios::{
        BodyConfig, HohmannConfig, KeplerConfig, LagrangeConfig, LagrangePoint, NBodyConfig,
        ScenarioType,
    };
    pub use super::units::{
        Acceleration3D, OrbitMass, OrbitTime, Position3D, Velocity3D, AU, EARTH_MASS, G, SOLAR_MASS,
    };
}

/// Run a complete orbital simulation with Jidoka monitoring.
///
/// # Arguments
///
/// * `scenario` - Scenario type to simulate
/// * `duration_seconds` - Total simulation duration
/// * `dt_seconds` - Time step size
/// * `softening` - Softening parameter for close encounters
///
/// # Returns
///
/// Final state and simulation statistics.
///
/// # Example
///
/// ```rust
/// use simular::orbit::{run_simulation, scenarios::ScenarioType, scenarios::KeplerConfig};
///
/// let result = run_simulation(
///     &ScenarioType::Kepler(KeplerConfig::earth_sun()),
///     365.25 * 86400.0,  // 1 year
///     3600.0,            // 1 hour steps
///     1e6,               // 1000 km softening
/// );
/// ```
#[must_use]
pub fn run_simulation(
    scenario: &scenarios::ScenarioType,
    duration_seconds: f64,
    dt_seconds: f64,
    softening: f64,
) -> SimulationResult {
    use jidoka::{JidokaResponse, OrbitJidokaConfig, OrbitJidokaGuard};
    use physics::YoshidaIntegrator;
    use units::OrbitTime;

    // Build initial state
    let mut state = match scenario {
        scenarios::ScenarioType::Kepler(config) => config.build(softening),
        scenarios::ScenarioType::NBody(config) => config.build(softening),
        scenarios::ScenarioType::Hohmann(config) => config.build_initial(softening),
        scenarios::ScenarioType::Lagrange(config) => config.build(softening),
    };

    // Initialize Jidoka guard
    let mut jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
    jidoka.initialize(&state);

    // Create integrator
    let yoshida = YoshidaIntegrator::new();
    let dt = OrbitTime::from_seconds(dt_seconds);

    // Track statistics
    let initial_energy = state.total_energy();
    let initial_angular_momentum = state.angular_momentum_magnitude();
    let mut steps = 0u64;
    let mut warnings = 0u64;
    let mut paused = false;

    // Run simulation
    let num_steps = (duration_seconds / dt_seconds) as u64;
    for _ in 0..num_steps {
        // Integrate
        if yoshida.step(&mut state, dt).is_err() {
            paused = true;
            break;
        }

        steps += 1;

        // Check Jidoka guards
        let response = jidoka.check(&state);
        match response {
            JidokaResponse::Continue => {}
            JidokaResponse::Warning { .. } => {
                warnings += 1;
            }
            JidokaResponse::Pause { .. } | JidokaResponse::Halt { .. } => {
                paused = true;
                break;
            }
        }
    }

    // Calculate final statistics
    let final_energy = state.total_energy();
    let final_angular_momentum = state.angular_momentum_magnitude();
    let energy_error = (final_energy - initial_energy).abs() / initial_energy.abs();
    let angular_momentum_error =
        (final_angular_momentum - initial_angular_momentum).abs() / initial_angular_momentum.abs();

    SimulationResult {
        final_state: state,
        steps,
        warnings,
        paused,
        energy_error,
        angular_momentum_error,
        sim_time: steps as f64 * dt_seconds,
    }
}

/// Result of running a simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Final simulation state.
    pub final_state: physics::NBodyState,
    /// Number of steps completed.
    pub steps: u64,
    /// Number of warnings encountered.
    pub warnings: u64,
    /// Whether simulation was paused due to Jidoka violation.
    pub paused: bool,
    /// Relative energy error.
    pub energy_error: f64,
    /// Relative angular momentum error.
    pub angular_momentum_error: f64,
    /// Simulated time (seconds).
    pub sim_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scenarios::{KeplerConfig, ScenarioType};

    #[test]
    fn test_run_simulation_kepler() {
        let result = run_simulation(
            &ScenarioType::Kepler(KeplerConfig::earth_sun()),
            86400.0 * 10.0, // 10 days
            3600.0,         // 1 hour steps
            1e6,
        );

        assert!(result.steps > 0);
        assert!(!result.paused);
        assert!(
            result.energy_error < 1e-6,
            "Energy error: {}",
            result.energy_error
        );
    }

    #[test]
    fn test_run_simulation_nbody() {
        let result = run_simulation(
            &ScenarioType::NBody(scenarios::NBodyConfig::inner_solar_system()),
            86400.0, // 1 day
            3600.0,  // 1 hour steps
            1e9,     // Larger softening for N-body
        );

        assert!(result.steps > 0);
        assert!(result.energy_error < 1e-4);
    }

    #[test]
    fn test_prelude_imports() {
        use prelude::*;

        let pos = Position3D::from_au(1.0, 0.0, 0.0);
        let _vel = Velocity3D::from_mps(0.0, 29780.0, 0.0);
        let _mass = OrbitMass::from_solar_masses(1.0);
        let _time = OrbitTime::from_days(365.25);
        let _config = KeplerConfig::earth_sun();

        assert!(pos.is_finite());
    }

    #[test]
    fn test_run_simulation_hohmann() {
        let result = run_simulation(
            &ScenarioType::Hohmann(scenarios::HohmannConfig::earth_to_mars()),
            86400.0 * 10.0, // 10 days
            3600.0,         // 1 hour steps
            1e6,
        );

        assert!(result.steps > 0);
        assert!(!result.paused);
    }

    #[test]
    fn test_run_simulation_lagrange() {
        let result = run_simulation(
            &ScenarioType::Lagrange(scenarios::LagrangeConfig::sun_earth_l2()),
            86400.0, // 1 day
            3600.0,  // 1 hour steps
            1e9,
        );

        assert!(result.steps > 0);
    }

    #[test]
    fn test_simulation_result_fields() {
        let result = run_simulation(
            &ScenarioType::Kepler(KeplerConfig::earth_sun()),
            3600.0, // 1 hour
            3600.0, // 1 step
            1e6,
        );

        assert_eq!(result.steps, 1);
        assert_eq!(result.warnings, 0);
        assert!(!result.paused);
        assert!(result.sim_time > 0.0);
        assert!(result.final_state.num_bodies() == 2);
    }
}
