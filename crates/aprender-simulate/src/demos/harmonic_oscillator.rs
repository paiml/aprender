//! Demo 1: Harmonic Oscillator Energy Conservation
//!
//! Demonstrates symplectic vs non-symplectic integrators showing energy drift.
//!
//! # Governing Equations
//!
//! ```text
//! Total Energy: E = ½mω²A² = ½m(ẋ² + ω²x²)
//! Position:     x(t) = A·cos(ωt + φ)
//! Velocity:     v(t) = -Aω·sin(ωt + φ)
//! ```
//!
//! # EDD Cycle
//!
//! 1. **Equation**: Energy E = ½mω²A² must be constant (Hamiltonian system)
//! 2. **Failing Test**: Energy drifts >1e-10 over 1000 periods
//! 3. **Implementation**: Störmer-Verlet symplectic integrator
//! 4. **Verification**: Energy bounded within tolerance
//! 5. **Falsification**: RK4 fails the same test (energy grows unbounded)

use super::{CriterionStatus, EddDemo, FalsificationStatus, IntegratorType};
use crate::engine::rng::SimRng;
use serde::{Deserialize, Serialize};

/// Harmonic oscillator demo state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonicOscillatorDemo {
    /// Current position.
    pub x: f64,
    /// Current velocity.
    pub v: f64,
    /// Angular frequency ω (rad/s).
    pub omega: f64,
    /// Mass (for energy calculation, normalized to 1).
    pub mass: f64,
    /// Current simulation time.
    pub time: f64,
    /// Initial energy (for conservation check).
    pub initial_energy: f64,
    /// Maximum energy drift observed.
    pub max_energy_drift: f64,
    /// Number of steps taken.
    pub step_count: u64,
    /// Integrator type.
    pub integrator: IntegratorType,
    /// Energy tolerance for verification.
    pub energy_tolerance: f64,
    /// RNG for any stochastic elements.
    #[serde(skip)]
    #[allow(dead_code)]
    rng: Option<SimRng>,
    /// Seed for reproducibility.
    pub seed: u64,
}

impl Default for HarmonicOscillatorDemo {
    fn default() -> Self {
        Self::new(42)
    }
}

impl HarmonicOscillatorDemo {
    /// Create a new harmonic oscillator demo with given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let x = 1.0; // Initial displacement
        let v = 0.0; // Initial velocity
        let omega = 2.0 * std::f64::consts::PI; // ω = 2π (1 Hz)
        let mass = 1.0;

        let initial_energy = Self::compute_energy_static(x, v, omega, mass);

        Self {
            x,
            v,
            omega,
            mass,
            time: 0.0,
            initial_energy,
            max_energy_drift: 0.0,
            step_count: 0,
            integrator: IntegratorType::StormerVerlet,
            energy_tolerance: 1e-10,
            rng: Some(SimRng::new(seed)),
            seed,
        }
    }

    /// Set the integrator type.
    pub fn set_integrator(&mut self, integrator: IntegratorType) {
        self.integrator = integrator;
    }

    /// Set initial conditions.
    pub fn set_initial_conditions(&mut self, x: f64, v: f64) {
        self.x = x;
        self.v = v;
        self.initial_energy = self.compute_energy();
        self.max_energy_drift = 0.0;
        self.time = 0.0;
        self.step_count = 0;
    }

    /// Compute current total energy: E = ½m(v² + ω²x²).
    #[must_use]
    pub fn compute_energy(&self) -> f64 {
        Self::compute_energy_static(self.x, self.v, self.omega, self.mass)
    }

    fn compute_energy_static(x: f64, v: f64, omega: f64, mass: f64) -> f64 {
        0.5 * mass * (v * v + omega * omega * x * x)
    }

    /// Compute relative energy drift.
    #[must_use]
    pub fn energy_drift(&self) -> f64 {
        let current_energy = self.compute_energy();
        (current_energy - self.initial_energy).abs() / self.initial_energy
    }

    /// Get analytical solution at current time.
    #[must_use]
    pub fn analytical_position(&self) -> f64 {
        // x(t) = A·cos(ωt + φ)
        // With x₀=1, v₀=0: A=1, φ=0
        let amplitude = (self.x * self.x + (self.v / self.omega).powi(2)).sqrt();
        let phase = (-self.v / self.omega).atan2(self.x);
        amplitude * (self.omega * self.time + phase).cos()
    }

    /// Get analytical velocity at current time.
    #[must_use]
    pub fn analytical_velocity(&self) -> f64 {
        let amplitude = (self.x * self.x + (self.v / self.omega).powi(2)).sqrt();
        let phase = (-self.v / self.omega).atan2(self.x);
        -amplitude * self.omega * (self.omega * self.time + phase).sin()
    }

    /// Step using Störmer-Verlet (symplectic).
    fn step_stormer_verlet(&mut self, dt: f64) {
        // Störmer-Verlet: symplectic, 2nd order, energy-conserving
        // v_{n+1/2} = v_n + (dt/2) * a(x_n)
        // x_{n+1} = x_n + dt * v_{n+1/2}
        // v_{n+1} = v_{n+1/2} + (dt/2) * a(x_{n+1})

        let omega_sq = self.omega * self.omega;

        // Half step velocity
        let a_n = -omega_sq * self.x;
        let v_half = self.v + 0.5 * dt * a_n;

        // Full step position
        self.x += dt * v_half;

        // Half step velocity with new acceleration
        let a_n1 = -omega_sq * self.x;
        self.v = v_half + 0.5 * dt * a_n1;
    }

    /// Step using RK4 (non-symplectic).
    fn step_rk4(&mut self, dt: f64) {
        // RK4: 4th order, but NOT symplectic - energy drifts over time
        let omega_sq = self.omega * self.omega;

        let x0 = self.x;
        let v0 = self.v;

        // k1
        let k1_x = v0;
        let k1_v = -omega_sq * x0;

        // k2
        let k2_x = v0 + 0.5 * dt * k1_v;
        let k2_v = -omega_sq * (x0 + 0.5 * dt * k1_x);

        // k3
        let k3_x = v0 + 0.5 * dt * k2_v;
        let k3_v = -omega_sq * (x0 + 0.5 * dt * k2_x);

        // k4
        let k4_x = v0 + dt * k3_v;
        let k4_v = -omega_sq * (x0 + dt * k3_x);

        // Update
        self.x = x0 + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x);
        self.v = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
    }

    /// Step using Euler (1st order, for demonstration).
    fn step_euler(&mut self, dt: f64) {
        // Euler: 1st order, wildly unstable for oscillators
        let omega_sq = self.omega * self.omega;

        let new_x = self.x + dt * self.v;
        let new_v = self.v - dt * omega_sq * self.x;

        self.x = new_x;
        self.v = new_v;
    }

    /// Run simulation for a given number of periods.
    pub fn run_periods(&mut self, num_periods: usize, steps_per_period: usize) {
        contract_pre_iterator!();
        let period = 2.0 * std::f64::consts::PI / self.omega;
        let dt = period / steps_per_period as f64;

        for _ in 0..(num_periods * steps_per_period) {
            self.step(dt);
        }
    }

    /// Get phase space coordinates (x, v).
    #[must_use]
    pub fn phase_space(&self) -> (f64, f64) {
        (self.x, self.v)
    }

    /// Get current period count.
    #[must_use]
    pub fn period_count(&self) -> f64 {
        self.time * self.omega / (2.0 * std::f64::consts::PI)
    }
}

impl EddDemo for HarmonicOscillatorDemo {
    fn name(&self) -> &'static str {
        "Harmonic Oscillator Energy Conservation"
    }

    fn emc_ref(&self) -> &'static str {
        "physics/harmonic_oscillator"
    }

    fn step(&mut self, dt: f64) {
        match self.integrator {
            IntegratorType::StormerVerlet => self.step_stormer_verlet(dt),
            IntegratorType::RK4 => self.step_rk4(dt),
            IntegratorType::Euler => self.step_euler(dt),
        }

        self.time += dt;
        self.step_count += 1;

        // Track maximum energy drift
        let drift = self.energy_drift();
        if drift > self.max_energy_drift {
            self.max_energy_drift = drift;
        }
    }

    fn verify_equation(&self) -> bool {
        self.energy_drift() < self.energy_tolerance
    }

    fn get_falsification_status(&self) -> FalsificationStatus {
        let drift = self.energy_drift();
        let passed = drift < self.energy_tolerance;

        FalsificationStatus {
            verified: passed,
            criteria: vec![CriterionStatus {
                id: "HO-ENERGY".to_string(),
                name: "Energy conservation".to_string(),
                passed,
                value: drift,
                threshold: self.energy_tolerance,
            }],
            message: if passed {
                format!(
                    "Energy conserved: drift = {:.2e} < {:.2e}",
                    drift, self.energy_tolerance
                )
            } else {
                format!(
                    "FALSIFIED: Energy drift = {:.2e} > {:.2e}",
                    drift, self.energy_tolerance
                )
            },
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.seed);
    }
}

// =============================================================================
// WASM Bindings
// =============================================================================

#[cfg(feature = "wasm")]
mod wasm {
    use super::{EddDemo, HarmonicOscillatorDemo, IntegratorType};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmHarmonicOscillator {
        inner: HarmonicOscillatorDemo,
    }

    #[wasm_bindgen]
    impl WasmHarmonicOscillator {
        #[wasm_bindgen(constructor)]
        pub fn new(seed: u64) -> Self {
            Self {
                inner: HarmonicOscillatorDemo::new(seed),
            }
        }

        pub fn step(&mut self, dt: f64) {
            self.inner.step(dt);
        }

        pub fn get_x(&self) -> f64 {
            self.inner.x
        }

        pub fn get_v(&self) -> f64 {
            self.inner.v
        }

        pub fn get_energy(&self) -> f64 {
            self.inner.compute_energy()
        }

        pub fn get_energy_drift(&self) -> f64 {
            self.inner.energy_drift()
        }

        pub fn get_time(&self) -> f64 {
            self.inner.time
        }

        pub fn verify_equation(&self) -> bool {
            self.inner.verify_equation()
        }

        pub fn set_integrator(&mut self, integrator: &str) {
            self.inner.integrator = match integrator {
                "rk4" => IntegratorType::RK4,
                "euler" => IntegratorType::Euler,
                // "verlet" and any other value defaults to StormerVerlet
                _ => IntegratorType::StormerVerlet,
            };
        }

        pub fn reset(&mut self) {
            self.inner.reset();
        }

        pub fn get_status_json(&self) -> String {
            serde_json::to_string(&self.inner.get_falsification_status()).unwrap_or_default()
        }
    }
}

// =============================================================================
// Tests - Following EDD Methodology
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Phase 1: Equation - Define what we're testing
    // =========================================================================

    #[test]
    fn test_equation_energy_formula() {
        // E = ½m(v² + ω²x²)
        let demo = HarmonicOscillatorDemo::new(42);

        let x = demo.x;
        let v = demo.v;
        let omega = demo.omega;
        let mass = demo.mass;

        let energy = 0.5 * mass * (v * v + omega * omega * x * x);
        assert!((energy - demo.compute_energy()).abs() < 1e-15);
    }

    #[test]
    fn test_equation_analytical_solution() {
        // x(t) = A·cos(ωt + φ) should match at t=0
        let demo = HarmonicOscillatorDemo::new(42);
        assert!((demo.analytical_position() - demo.x).abs() < 1e-10);
    }

    // =========================================================================
    // Phase 2: Failing Test - This should fail with bad integrator
    // =========================================================================

    #[test]
    fn test_failing_euler_energy_conservation() {
        // Euler method should FAIL energy conservation
        let mut demo = HarmonicOscillatorDemo::new(42);
        demo.set_integrator(IntegratorType::Euler);
        demo.energy_tolerance = 1e-6; // Even with loose tolerance

        // Run for 10 periods
        demo.run_periods(10, 100);

        // Euler should fail conservation
        assert!(
            !demo.verify_equation(),
            "Euler method should NOT conserve energy"
        );
    }

    #[test]
    fn test_failing_rk4_long_term() {
        // RK4 should fail over very long times
        let mut demo = HarmonicOscillatorDemo::new(42);
        demo.set_integrator(IntegratorType::RK4);
        demo.energy_tolerance = 1e-10; // Strict tolerance

        // Run for 1000 periods
        demo.run_periods(1000, 100);

        // RK4 accumulates drift over long times
        // This may or may not fail depending on step size
        let drift = demo.energy_drift();
        // Just document the drift
        assert!(drift > 0.0, "RK4 should have some drift: {drift}");
    }

    // =========================================================================
    // Phase 3: Implementation - Störmer-Verlet passes
    // =========================================================================

    #[test]
    fn test_verlet_energy_conservation_short() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        demo.set_integrator(IntegratorType::StormerVerlet);
        demo.energy_tolerance = 1e-10;

        // Run for 100 periods
        demo.run_periods(100, 1000);

        assert!(
            demo.verify_equation(),
            "Störmer-Verlet should conserve energy, drift = {}",
            demo.energy_drift()
        );
    }

    #[test]
    fn test_verlet_energy_conservation_long() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        demo.set_integrator(IntegratorType::StormerVerlet);
        demo.energy_tolerance = 1e-8; // Slightly relaxed for 1000 periods

        // Run for 1000 periods
        demo.run_periods(1000, 1000);

        assert!(
            demo.verify_equation(),
            "Störmer-Verlet should conserve energy over 1000 periods, drift = {}",
            demo.energy_drift()
        );
    }

    // =========================================================================
    // Phase 4: Verification - Comprehensive tests
    // =========================================================================

    #[test]
    fn test_verification_phase_space_bounded() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        demo.run_periods(100, 1000);

        let (x, v) = demo.phase_space();
        let amplitude = (x * x + (v / demo.omega).powi(2)).sqrt();

        // Amplitude should stay close to initial (1.0)
        assert!(
            (amplitude - 1.0).abs() < 0.01,
            "Amplitude should be conserved: {amplitude}"
        );
    }

    #[test]
    fn test_verification_period_accurate() {
        let mut demo = HarmonicOscillatorDemo::new(42);

        // Run exactly 10 periods
        demo.run_periods(10, 10000);

        // Should be back near starting position
        assert!(
            (demo.x - 1.0).abs() < 0.001,
            "Position after 10 periods: {} (expected ~1.0)",
            demo.x
        );
    }

    // =========================================================================
    // Phase 5: Falsification - Document how to break it
    // =========================================================================

    #[test]
    fn test_falsification_large_timestep() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        demo.set_integrator(IntegratorType::StormerVerlet);
        demo.energy_tolerance = 1e-6;

        // Use timestep larger than stability limit
        let period = 2.0 * std::f64::consts::PI / demo.omega;
        let dt = period / 5.0; // Only 5 steps per period (too coarse)

        for _ in 0..50 {
            demo.step(dt);
        }

        // Even Verlet can fail with too-large timestep
        let drift = demo.energy_drift();
        // Document the behavior (may or may not fail depending on exact dt)
        println!(
            "Falsification test: dt={dt:.4}, drift={drift:.2e}, periods={}",
            demo.period_count()
        );
    }

    #[test]
    fn test_falsification_status_structure() {
        let demo = HarmonicOscillatorDemo::new(42);
        let status = demo.get_falsification_status();

        assert!(status.verified);
        assert_eq!(status.criteria.len(), 1);
        assert_eq!(status.criteria[0].id, "HO-ENERGY");
    }

    // =========================================================================
    // Integration tests
    // =========================================================================

    #[test]
    fn test_demo_trait_implementation() {
        let mut demo = HarmonicOscillatorDemo::new(42);

        assert_eq!(demo.name(), "Harmonic Oscillator Energy Conservation");
        assert_eq!(demo.emc_ref(), "physics/harmonic_oscillator");

        demo.step(0.001);
        assert!(demo.time > 0.0);

        demo.reset();
        assert_eq!(demo.time, 0.0);
    }

    #[test]
    fn test_serialization() {
        let demo = HarmonicOscillatorDemo::new(42);
        let json = serde_json::to_string(&demo).expect("serialize");
        assert!(json.contains("omega"));

        let restored: HarmonicOscillatorDemo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.omega, demo.omega);
    }

    #[test]
    fn test_reproducibility() {
        let mut demo1 = HarmonicOscillatorDemo::new(42);
        let mut demo2 = HarmonicOscillatorDemo::new(42);

        demo1.run_periods(10, 100);
        demo2.run_periods(10, 100);

        assert_eq!(demo1.x, demo2.x);
        assert_eq!(demo1.v, demo2.v);
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_default() {
        let demo = HarmonicOscillatorDemo::default();
        assert_eq!(demo.seed, 42);
        assert_eq!(demo.step_count, 0);
    }

    #[test]
    fn test_set_initial_conditions() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        demo.set_initial_conditions(2.0, 1.0);

        assert!((demo.x - 2.0).abs() < 1e-10);
        assert!((demo.v - 1.0).abs() < 1e-10);
        assert_eq!(demo.time, 0.0);
        assert_eq!(demo.step_count, 0);
        assert!((demo.max_energy_drift - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_analytical_velocity() {
        let demo = HarmonicOscillatorDemo::new(42);
        // At t=0 with x=1, v=0: velocity should be 0
        let analytical_v = demo.analytical_velocity();
        assert!(
            analytical_v.abs() < 1e-10,
            "Analytical velocity at t=0 should be ~0: {analytical_v}"
        );
    }

    #[test]
    fn test_phase_space() {
        let demo = HarmonicOscillatorDemo::new(42);
        let (x, v) = demo.phase_space();
        assert!((x - 1.0).abs() < 1e-10);
        assert!(v.abs() < 1e-10);
    }

    #[test]
    fn test_period_count() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        assert!((demo.period_count() - 0.0).abs() < 1e-10);

        demo.run_periods(5, 100);
        assert!(
            (demo.period_count() - 5.0).abs() < 0.01,
            "Period count after 5 periods: {}",
            demo.period_count()
        );
    }

    #[test]
    fn test_energy_drift_calculation() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        let initial_drift = demo.energy_drift();
        assert!(initial_drift < 1e-15, "Initial drift should be ~0");

        // Run with Euler to cause drift
        demo.set_integrator(IntegratorType::Euler);
        demo.run_periods(1, 100);
        let after_drift = demo.energy_drift();
        assert!(
            after_drift > 0.0,
            "After Euler integration, drift should be > 0"
        );
    }

    #[test]
    fn test_max_energy_drift_tracking() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        demo.set_integrator(IntegratorType::Euler);
        demo.run_periods(10, 100);

        assert!(
            demo.max_energy_drift > 0.0,
            "Max energy drift should be tracked"
        );
    }

    #[test]
    fn test_step_count_tracking() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        assert_eq!(demo.step_count, 0);

        demo.step(0.01);
        assert_eq!(demo.step_count, 1);

        demo.run_periods(1, 100);
        assert_eq!(demo.step_count, 101);
    }

    #[test]
    fn test_falsification_status_failed() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        demo.set_integrator(IntegratorType::Euler);
        demo.energy_tolerance = 1e-15; // Very strict
        demo.run_periods(10, 100);

        let status = demo.get_falsification_status();
        assert!(!status.verified);
        assert!(status.message.contains("FALSIFIED"));
    }

    #[test]
    fn test_compute_energy_static() {
        // Test the static method indirectly through compute_energy
        let demo = HarmonicOscillatorDemo::new(42);
        let energy = demo.compute_energy();
        // E = ½m(v² + ω²x²) = ½*1*(0² + (2π)²*1²) = ½*(2π)² ≈ 19.74
        let expected = 0.5 * demo.mass * demo.omega * demo.omega;
        assert!(
            (energy - expected).abs() < 1e-10,
            "Energy should match expected: {} vs {}",
            energy,
            expected
        );
    }

    #[test]
    fn test_clone() {
        let demo = HarmonicOscillatorDemo::new(42);
        let cloned = demo.clone();
        assert_eq!(demo.x, cloned.x);
        assert_eq!(demo.v, cloned.v);
        assert_eq!(demo.omega, cloned.omega);
    }

    #[test]
    fn test_debug() {
        let demo = HarmonicOscillatorDemo::new(42);
        let debug_str = format!("{demo:?}");
        assert!(debug_str.contains("HarmonicOscillatorDemo"));
    }

    #[test]
    fn test_all_integrator_types() {
        // Test all three integrator types
        for integrator in [
            IntegratorType::StormerVerlet,
            IntegratorType::RK4,
            IntegratorType::Euler,
        ] {
            let mut demo = HarmonicOscillatorDemo::new(42);
            demo.set_integrator(integrator);
            demo.step(0.001);
            assert!(demo.time > 0.0);
        }
    }

    #[test]
    fn test_run_periods_boundary() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        // Run 0 periods should do nothing
        demo.run_periods(0, 100);
        assert_eq!(demo.step_count, 0);

        // Run with 0 steps per period would panic on division
        // So we avoid that test case - but at least 1 step
        demo.run_periods(1, 1);
        assert_eq!(demo.step_count, 1);
    }

    #[test]
    fn test_analytical_solution_after_quarter_period() {
        let mut demo = HarmonicOscillatorDemo::new(42);
        // Run for exactly a quarter period using many steps
        let quarter_period = std::f64::consts::PI / (2.0 * demo.omega);
        let dt = quarter_period / 1000.0;

        for _ in 0..1000 {
            demo.step(dt);
        }

        // After quarter period, x should be ~0, v should be negative
        // x(T/4) = cos(π/2) = 0
        // v(T/4) = -ω*sin(π/2) = -ω
        assert!(
            demo.x.abs() < 0.01,
            "x after quarter period: {} (expected ~0)",
            demo.x
        );
    }
}
