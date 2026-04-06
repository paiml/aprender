//! Core simulation engine.
//!
//! Implements the central simulation loop with:
//! - Deterministic RNG (PCG with partitioned seeds)
//! - Event scheduling with deterministic ordering
//! - Jidoka guards for stop-on-error
//! - State management

pub mod clock;
pub mod jidoka;
pub mod rng;
pub mod scheduler;
pub mod state;

use serde::{Deserialize, Serialize};

pub use clock::SimClock;
pub use jidoka::{JidokaGuard, JidokaViolation};
pub use rng::SimRng;
pub use scheduler::{EventScheduler, ScheduledEvent};
pub use state::SimState;

use crate::config::{IntegratorType, PhysicsEngine as ConfigPhysicsEngine, SimConfig};
use crate::domains::physics::{
    CentralForceField, EulerIntegrator, ForceField, GravityField, Integrator, PhysicsEngine,
    RK4Integrator, VerletIntegrator,
};
use crate::engine::state::Vec3;
use crate::error::SimResult;

/// Simulation time representation.
///
/// Uses a fixed-point representation for reproducibility across platforms.
/// Internal representation is in nanoseconds to avoid floating-point issues.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct SimTime {
    /// Time in nanoseconds from simulation start.
    nanos: u64,
}

impl SimTime {
    /// Zero time (simulation start).
    pub const ZERO: Self = Self { nanos: 0 };

    /// Create time from seconds.
    ///
    /// # Panics
    ///
    /// Panics if seconds is negative or not finite.
    #[must_use]
    pub fn from_secs(secs: f64) -> Self {
        assert!(secs >= 0.0, "SimTime cannot be negative");
        assert!(secs.is_finite(), "SimTime must be finite");
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let nanos = (secs * 1_000_000_000.0) as u64;
        Self { nanos }
    }

    /// Create time from nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }

    /// Get time as seconds (f64).
    #[must_use]
    pub fn as_secs_f64(&self) -> f64 {
        self.nanos as f64 / 1_000_000_000.0
    }

    /// Get time as nanoseconds.
    #[must_use]
    pub const fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Add duration to time.
    #[must_use]
    pub const fn add_nanos(self, nanos: u64) -> Self {
        Self {
            nanos: self.nanos + nanos,
        }
    }

    /// Subtract duration from time, saturating at zero.
    #[must_use]
    pub const fn saturating_sub_nanos(self, nanos: u64) -> Self {
        Self {
            nanos: self.nanos.saturating_sub(nanos),
        }
    }
}

impl std::ops::Add for SimTime {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            nanos: self.nanos + rhs.nanos,
        }
    }
}

impl std::ops::Sub for SimTime {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            nanos: self.nanos.saturating_sub(rhs.nanos),
        }
    }
}

impl std::fmt::Display for SimTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.9}s", self.as_secs_f64())
    }
}

/// Main simulation engine.
///
/// Coordinates all subsystems:
/// - Event scheduling
/// - State management
/// - Jidoka monitoring
/// - Checkpointing
/// - Physics simulation
pub struct SimEngine {
    /// Current simulation state.
    state: SimState,
    /// Event scheduler.
    scheduler: EventScheduler,
    /// Jidoka guard for anomaly detection.
    jidoka: JidokaGuard,
    /// Simulation clock.
    clock: SimClock,
    /// Random number generator.
    rng: SimRng,
    /// Physics engine (optional).
    physics: Option<PhysicsEngine>,
    /// Configuration (stored for reference and future use).
    #[allow(dead_code)]
    config: SimConfig,
}

impl SimEngine {
    /// Create a new simulation engine from configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration validation fails.
    pub fn new(config: SimConfig) -> SimResult<Self> {
        let seed = config.reproducibility.seed;
        let rng = SimRng::new(seed);
        let jidoka = JidokaGuard::from_config(&config);
        let clock = SimClock::new(config.get_timestep());

        // Initialize Physics Engine based on config
        let physics = if config.domains.physics.enabled {
            let integrator: Box<dyn Integrator + Send + Sync> =
                match config.domains.physics.integrator.integrator_type {
                    IntegratorType::Euler => Box::new(EulerIntegrator::new()),
                    IntegratorType::Rk4 => Box::new(RK4Integrator::new()),
                    // Default to Verlet for Verlet, Rk78, SymplecticEuler, etc.
                    _ => Box::new(VerletIntegrator::new()),
                };

            let force_field: Box<dyn ForceField + Send + Sync> = match config.domains.physics.engine
            {
                ConfigPhysicsEngine::Orbital => Box::new(CentralForceField::new(1.0, Vec3::zero())), // Default mu=1.0 for now
                // Default to GravityField for RigidBody, Fluid, Discrete, etc.
                _ => Box::new(GravityField::default()),
            };

            Some(PhysicsEngine::new_boxed(force_field, integrator))
        } else {
            None
        };

        Ok(Self {
            state: SimState::default(),
            scheduler: EventScheduler::new(),
            jidoka,
            clock,
            rng,
            physics,
            config,
        })
    }

    /// Get current simulation time.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Delegating to non-const method
    pub fn current_time(&self) -> SimTime {
        self.clock.current_time()
    }

    /// Get current simulation state.
    #[must_use]
    pub const fn state(&self) -> &SimState {
        &self.state
    }

    /// Get mutable reference to state.
    #[must_use]
    pub fn state_mut(&mut self) -> &mut SimState {
        &mut self.state
    }

    /// Get reference to RNG.
    #[must_use]
    pub const fn rng(&self) -> &SimRng {
        &self.rng
    }

    /// Get mutable reference to RNG.
    #[must_use]
    pub fn rng_mut(&mut self) -> &mut SimRng {
        &mut self.rng
    }

    /// Step the simulation forward by one timestep.
    ///
    /// # Errors
    ///
    /// Returns `SimError` if:
    /// - Jidoka violation detected (NaN, energy drift, constraint)
    /// - Domain engine error
    pub fn step(&mut self) -> SimResult<()> {
        // Advance clock
        self.clock.tick();

        // Process scheduled events
        while let Some(event) = self.scheduler.next_before(self.clock.current_time()) {
            self.state.apply_event(&event.event)?;
        }

        // Run Physics Step
        if let Some(physics) = &self.physics {
            physics.step(&mut self.state, self.clock.dt())?;
        }

        // Jidoka check (stop-on-error)
        self.jidoka.check(&self.state)?;

        Ok(())
    }

    /// Run simulation for specified duration.
    ///
    /// # Errors
    ///
    /// Returns error if any step fails.
    pub fn run_for(&mut self, duration: SimTime) -> SimResult<()> {
        let end_time = self.clock.current_time() + duration;

        while self.clock.current_time() < end_time {
            self.step()?;
        }

        Ok(())
    }

    /// Run simulation until predicate returns true.
    ///
    /// # Errors
    ///
    /// Returns error if any step fails.
    pub fn run_until<F>(&mut self, predicate: F) -> SimResult<()>
    where
        F: Fn(&SimState) -> bool,
    {
        while !predicate(&self.state) {
            self.step()?;
        }

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::SimConfig;

    #[test]
    fn test_sim_time_creation() {
        let t1 = SimTime::from_secs(1.5);
        assert!((t1.as_secs_f64() - 1.5).abs() < 1e-9);

        let t2 = SimTime::from_nanos(1_500_000_000);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_sim_time_arithmetic() {
        let t1 = SimTime::from_secs(1.0);
        let t2 = SimTime::from_secs(0.5);

        let sum = t1 + t2;
        assert!((sum.as_secs_f64() - 1.5).abs() < 1e-9);

        let diff = t1 - t2;
        assert!((diff.as_secs_f64() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_sim_time_ordering() {
        let t1 = SimTime::from_secs(1.0);
        let t2 = SimTime::from_secs(2.0);

        assert!(t1 < t2);
        assert!(t2 > t1);
        assert_eq!(t1, t1);
    }

    #[test]
    fn test_sim_time_display() {
        let t = SimTime::from_secs(1.234_567_890);
        let s = t.to_string();
        assert!(s.contains("1.234567890"));
    }

    #[test]
    fn test_sim_time_zero() {
        let t = SimTime::ZERO;
        assert_eq!(t.as_nanos(), 0);
        assert!((t.as_secs_f64() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sim_time_add_nanos() {
        let t = SimTime::from_secs(1.0);
        let t2 = t.add_nanos(500_000_000);
        assert!((t2.as_secs_f64() - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_sim_time_saturating_sub() {
        let t = SimTime::from_secs(1.0);
        let t2 = t.saturating_sub_nanos(500_000_000);
        assert!((t2.as_secs_f64() - 0.5).abs() < 1e-9);

        // Saturating at zero
        let t3 = t.saturating_sub_nanos(2_000_000_000);
        assert_eq!(t3.as_nanos(), 0);
    }

    #[test]
    fn test_sim_time_default() {
        let t: SimTime = Default::default();
        assert_eq!(t.as_nanos(), 0);
    }

    #[test]
    fn test_sim_time_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SimTime::from_secs(1.0));
        set.insert(SimTime::from_secs(2.0));
        set.insert(SimTime::from_secs(1.0)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_sim_time_clone() {
        let t1 = SimTime::from_secs(1.0);
        let t2 = t1;
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_sim_time_sub_saturating() {
        let t1 = SimTime::from_secs(1.0);
        let t2 = SimTime::from_secs(2.0);
        // Sub uses saturating_sub
        let diff = t1 - t2;
        assert_eq!(diff.as_nanos(), 0);
    }

    #[test]
    fn test_sim_engine_new() {
        let config = SimConfig::builder().seed(42).build();
        let engine = SimEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_sim_engine_initial_time() {
        let config = SimConfig::builder().seed(42).build();
        let engine = SimEngine::new(config).unwrap();
        assert_eq!(engine.current_time(), SimTime::ZERO);
    }

    #[test]
    fn test_sim_engine_state() {
        let config = SimConfig::builder().seed(42).build();
        let engine = SimEngine::new(config).unwrap();
        let state = engine.state();
        assert_eq!(state.num_bodies(), 0);
    }

    #[test]
    fn test_sim_engine_state_mut() {
        let config = SimConfig::builder().seed(42).build();
        let mut engine = SimEngine::new(config).unwrap();
        let state = engine.state_mut();
        // Add a body to test mutability
        state.add_body(1.0, state::Vec3::zero(), state::Vec3::zero());
        assert_eq!(state.num_bodies(), 1);
    }

    #[test]
    fn test_sim_engine_rng() {
        let config = SimConfig::builder().seed(42).build();
        let engine = SimEngine::new(config).unwrap();
        let _rng = engine.rng();
    }

    #[test]
    fn test_sim_engine_rng_mut() {
        let config = SimConfig::builder().seed(42).build();
        let mut engine = SimEngine::new(config).unwrap();
        let rng = engine.rng_mut();
        let _ = rng.gen_f64();
    }

    #[test]
    fn test_sim_engine_step() {
        let config = SimConfig::builder().seed(42).build();
        let mut engine = SimEngine::new(config).unwrap();
        let result = engine.step();
        assert!(result.is_ok());
        assert!(engine.current_time() > SimTime::ZERO);
    }

    #[test]
    fn test_sim_engine_multiple_steps() {
        let config = SimConfig::builder().seed(42).build();
        let mut engine = SimEngine::new(config).unwrap();

        for _ in 0..10 {
            engine.step().unwrap();
        }

        assert!(engine.current_time() > SimTime::ZERO);
    }

    #[test]
    fn test_sim_engine_run_for() {
        let config = SimConfig::builder().seed(42).build();
        let mut engine = SimEngine::new(config).unwrap();

        let duration = SimTime::from_secs(0.1);
        let result = engine.run_for(duration);
        assert!(result.is_ok());
        assert!(engine.current_time() >= duration);
    }

    #[test]
    fn test_sim_engine_run_until() {
        let config = SimConfig::builder().seed(42).timestep(0.001).build();
        let mut engine = SimEngine::new(config).unwrap();

        // Add a body so we have something to check
        engine
            .state_mut()
            .add_body(1.0, state::Vec3::zero(), state::Vec3::zero());

        // run_until checks predicate based on state
        // Stop when bodies exist (immediate)
        let result = engine.run_until(|state| state.num_bodies() > 0);

        // This is a test that it runs without error
        assert!(result.is_ok());
    }

    #[test]
    fn test_sim_engine_run_until_immediate() {
        let config = SimConfig::builder().seed(42).build();
        let mut engine = SimEngine::new(config).unwrap();

        // Predicate immediately true
        let result = engine.run_until(|_state| true);
        assert!(result.is_ok());
        assert_eq!(engine.current_time(), SimTime::ZERO);
    }
}
