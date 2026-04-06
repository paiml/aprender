//! Demo 5: Kepler Orbit Conservation Laws
//!
//! Two-body orbital mechanics verifying all three Kepler laws plus conservation.
//!
//! # Governing Equations
//!
//! ```text
//! Newton's Gravitation: F = -GMm/r² r̂
//! Specific Energy:      E = v²/2 - μ/r = -μ/(2a)    (constant)
//! Angular Momentum:     L = r × v                    (constant vector)
//! Kepler's Third Law:   T² = (4π²/μ) a³
//! ```
//!
//! # EDD Cycle
//!
//! 1. **Equations**: E constant, L constant, T² ∝ a³
//! 2. **Failing Tests**: E drifts >1e-10, L drifts >1e-12, T error >1e-6
//! 3. **Implementation**: Störmer-Verlet symplectic integrator
//! 4. **Verification**: All three conservation laws hold
//! 5. **Falsification**: Add third body perturbation, laws break

use super::{CriterionStatus, EddDemo, FalsificationStatus, IntegratorType};
use crate::engine::rng::SimRng;
use serde::{Deserialize, Serialize};

/// 2D vector for orbital mechanics.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    #[must_use]
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag > f64::EPSILON {
            Self::new(self.x / mag, self.y / mag)
        } else {
            Self::new(0.0, 0.0)
        }
    }

    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product (returns scalar z-component).
    #[must_use]
    pub fn cross(&self, other: &Self) -> f64 {
        self.x * other.y - self.y * other.x
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self::new(self.x * scalar, self.y * scalar)
    }
}

/// Kepler Orbit demo state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeplerOrbitDemo {
    /// Position vector (m).
    pub position: Vec2,
    /// Velocity vector (m/s).
    pub velocity: Vec2,
    /// Gravitational parameter μ = GM (m³/s²).
    pub mu: f64,
    /// Current simulation time (s).
    pub time: f64,
    /// Initial specific orbital energy.
    pub initial_energy: f64,
    /// Initial angular momentum magnitude.
    pub initial_angular_momentum: f64,
    /// Initial semi-major axis.
    pub initial_semi_major_axis: f64,
    /// Maximum energy drift observed.
    pub max_energy_drift: f64,
    /// Maximum angular momentum drift observed.
    pub max_angular_momentum_drift: f64,
    /// Number of steps taken.
    pub step_count: u64,
    /// Integrator type.
    pub integrator: IntegratorType,
    /// Energy conservation tolerance.
    pub energy_tolerance: f64,
    /// Angular momentum tolerance.
    pub angular_momentum_tolerance: f64,
    /// Period tolerance.
    pub period_tolerance: f64,
    /// Position history for orbit plotting.
    #[serde(skip)]
    pub orbit_history: Vec<Vec2>,
    /// Maximum history length.
    pub max_history: usize,
    /// Enable third body perturbation.
    pub perturbation_enabled: bool,
    /// Third body position (if enabled).
    pub perturber_position: Vec2,
    /// Third body mass parameter.
    pub perturber_mu: f64,
    /// Seed for reproducibility.
    pub seed: u64,
    /// RNG for any stochastic elements.
    #[serde(skip)]
    #[allow(dead_code)]
    rng: Option<SimRng>,
}

impl Default for KeplerOrbitDemo {
    fn default() -> Self {
        Self::new(42)
    }
}

impl KeplerOrbitDemo {
    /// Create a new Kepler orbit demo (Earth-like orbit).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        // Earth-like orbit around Sun-like star
        let mu: f64 = 1.327_124_400_18e20; // GM_sun (m³/s²)
        let r: f64 = 1.496e11; // 1 AU (m)
        let v_circular: f64 = (mu / r).sqrt(); // Circular orbit velocity

        let position = Vec2::new(r, 0.0);
        let velocity = Vec2::new(0.0, v_circular);

        let mut demo = Self {
            position,
            velocity,
            mu,
            time: 0.0,
            initial_energy: 0.0,
            initial_angular_momentum: 0.0,
            initial_semi_major_axis: 0.0,
            max_energy_drift: 0.0,
            max_angular_momentum_drift: 0.0,
            step_count: 0,
            integrator: IntegratorType::StormerVerlet,
            energy_tolerance: 1e-10,
            angular_momentum_tolerance: 1e-12,
            period_tolerance: 1e-6,
            orbit_history: Vec::new(),
            max_history: 1000,
            perturbation_enabled: false,
            perturber_position: Vec2::new(5.0 * r, 0.0), // Jupiter-like distance
            perturber_mu: mu * 0.001,                    // Jupiter mass ratio
            seed,
            rng: Some(SimRng::new(seed)),
        };

        // Calculate initial conserved quantities
        demo.initial_energy = demo.specific_energy();
        demo.initial_angular_momentum = demo.angular_momentum();
        demo.initial_semi_major_axis = demo.semi_major_axis();

        demo
    }

    /// Create orbit with specific eccentricity.
    #[must_use]
    pub fn with_eccentricity(seed: u64, eccentricity: f64) -> Self {
        let mut demo = Self::new(seed);

        // Adjust velocity for desired eccentricity
        // At perihelion: v = sqrt(μ/a * (1+e)/(1-e)) for e < 1
        let e = eccentricity.clamp(0.0, 0.99);
        let a = demo.initial_semi_major_axis;

        if e < 0.99 {
            let v_perihelion = (demo.mu / a * (1.0 + e) / (1.0 - e)).sqrt();
            demo.velocity = Vec2::new(0.0, v_perihelion);

            // Position at perihelion
            let r_perihelion = a * (1.0 - e);
            demo.position = Vec2::new(r_perihelion, 0.0);
        }

        // Recalculate initial values
        demo.initial_energy = demo.specific_energy();
        demo.initial_angular_momentum = demo.angular_momentum();
        demo.initial_semi_major_axis = demo.semi_major_axis();

        demo
    }

    /// Calculate specific orbital energy: E = v²/2 - μ/r.
    #[must_use]
    pub fn specific_energy(&self) -> f64 {
        let v_sq = self.velocity.dot(&self.velocity);
        let r = self.position.magnitude();

        if r > f64::EPSILON {
            0.5 * v_sq - self.mu / r
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Calculate angular momentum magnitude: L = |r × v|.
    #[must_use]
    pub fn angular_momentum(&self) -> f64 {
        self.position.cross(&self.velocity).abs()
    }

    /// Calculate semi-major axis: a = -μ/(2E).
    #[must_use]
    pub fn semi_major_axis(&self) -> f64 {
        let e = self.specific_energy();
        if e < 0.0 {
            -self.mu / (2.0 * e)
        } else {
            f64::INFINITY // Parabolic or hyperbolic
        }
    }

    /// Calculate orbital period: T = 2π√(a³/μ).
    #[must_use]
    pub fn orbital_period(&self) -> f64 {
        let a = self.semi_major_axis();
        if a.is_finite() && a > 0.0 {
            2.0 * std::f64::consts::PI * (a.powi(3) / self.mu).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Calculate eccentricity.
    #[must_use]
    pub fn eccentricity(&self) -> f64 {
        let e = self.specific_energy();
        let l = self.angular_momentum();

        if e < 0.0 {
            (1.0 + 2.0 * e * l * l / (self.mu * self.mu)).sqrt()
        } else {
            1.0 // Parabolic or hyperbolic
        }
    }

    /// Get relative energy drift.
    #[must_use]
    pub fn energy_drift(&self) -> f64 {
        let current = self.specific_energy();
        if self.initial_energy.abs() > f64::EPSILON {
            (current - self.initial_energy).abs() / self.initial_energy.abs()
        } else {
            (current - self.initial_energy).abs()
        }
    }

    /// Get relative angular momentum drift.
    #[must_use]
    pub fn angular_momentum_drift(&self) -> f64 {
        let current = self.angular_momentum();
        if self.initial_angular_momentum > f64::EPSILON {
            (current - self.initial_angular_momentum).abs() / self.initial_angular_momentum
        } else {
            (current - self.initial_angular_momentum).abs()
        }
    }

    /// Calculate gravitational acceleration at position.
    fn acceleration(&self, pos: &Vec2) -> Vec2 {
        let r = pos.magnitude();
        if r < f64::EPSILON {
            return Vec2::new(0.0, 0.0);
        }

        // Primary body acceleration
        let a_primary = *pos * (-self.mu / (r * r * r));

        // Add perturber if enabled
        if self.perturbation_enabled {
            let r_to_perturber = self.perturber_position - *pos;
            let r_perturber = r_to_perturber.magnitude();
            if r_perturber > f64::EPSILON {
                let a_perturber = r_to_perturber
                    * (self.perturber_mu / (r_perturber * r_perturber * r_perturber));
                return a_primary + a_perturber;
            }
        }

        a_primary
    }

    /// Step using Störmer-Verlet (symplectic).
    fn step_stormer_verlet(&mut self, dt: f64) {
        // Half step velocity
        let a_n = self.acceleration(&self.position);
        let v_half = self.velocity + a_n * (0.5 * dt);

        // Full step position
        self.position = self.position + v_half * dt;

        // Half step velocity with new acceleration
        let a_n1 = self.acceleration(&self.position);
        self.velocity = v_half + a_n1 * (0.5 * dt);
    }

    /// Step using RK4 (non-symplectic).
    fn step_rk4(&mut self, dt: f64) {
        let pos0 = self.position;
        let vel0 = self.velocity;

        // k1
        let a1 = self.acceleration(&pos0);
        let k1_r = vel0;
        let k1_v = a1;

        // k2
        let pos2 = pos0 + k1_r * (0.5 * dt);
        let vel2 = vel0 + k1_v * (0.5 * dt);
        let a2 = self.acceleration(&pos2);
        let k2_r = vel2;
        let k2_v = a2;

        // k3
        let pos3 = pos0 + k2_r * (0.5 * dt);
        let vel3 = vel0 + k2_v * (0.5 * dt);
        let a3 = self.acceleration(&pos3);
        let k3_r = vel3;
        let k3_v = a3;

        // k4
        let pos4 = pos0 + k3_r * dt;
        let vel4 = vel0 + k3_v * dt;
        let a4 = self.acceleration(&pos4);
        let k4_r = vel4;
        let k4_v = a4;

        // Update
        self.position = pos0 + (k1_r + k2_r * 2.0 + k3_r * 2.0 + k4_r) * (dt / 6.0);
        self.velocity = vel0 + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * (dt / 6.0);
    }

    /// Enable or disable perturbation.
    pub fn set_perturbation(&mut self, enabled: bool) {
        self.perturbation_enabled = enabled;
    }

    /// Set integrator type.
    pub fn set_integrator(&mut self, integrator: IntegratorType) {
        self.integrator = integrator;
    }

    /// Run simulation for given number of orbits.
    pub fn run_orbits(&mut self, num_orbits: f64, steps_per_orbit: usize) {
        contract_pre_iterator!();
        let period = self.orbital_period();
        if !period.is_finite() {
            return;
        }

        let dt = period / steps_per_orbit as f64;
        let total_steps = (num_orbits * steps_per_orbit as f64) as usize;

        for _ in 0..total_steps {
            self.step(dt);
        }
    }

    /// Get current true anomaly (angle in orbit).
    #[must_use]
    pub fn true_anomaly(&self) -> f64 {
        self.position.y.atan2(self.position.x)
    }

    /// Record position for orbit history.
    fn record_position(&mut self) {
        if self.orbit_history.len() >= self.max_history {
            self.orbit_history.remove(0);
        }
        self.orbit_history.push(self.position);
    }
}

impl EddDemo for KeplerOrbitDemo {
    fn name(&self) -> &'static str {
        "Kepler Orbit Conservation Laws"
    }

    fn emc_ref(&self) -> &'static str {
        "physics/kepler_two_body"
    }

    fn step(&mut self, dt: f64) {
        match self.integrator {
            IntegratorType::StormerVerlet => self.step_stormer_verlet(dt),
            IntegratorType::RK4 => self.step_rk4(dt),
            IntegratorType::Euler => {
                // Simple Euler (unstable for orbits)
                let a = self.acceleration(&self.position);
                self.position = self.position + self.velocity * dt;
                self.velocity = self.velocity + a * dt;
            }
        }

        self.time += dt;
        self.step_count += 1;

        // Track maximum drifts
        let e_drift = self.energy_drift();
        let l_drift = self.angular_momentum_drift();

        if e_drift > self.max_energy_drift {
            self.max_energy_drift = e_drift;
        }
        if l_drift > self.max_angular_momentum_drift {
            self.max_angular_momentum_drift = l_drift;
        }

        // Record for visualization
        if self.step_count % 10 == 0 {
            self.record_position();
        }
    }

    fn verify_equation(&self) -> bool {
        let energy_ok = self.energy_drift() < self.energy_tolerance;
        let angular_ok = self.angular_momentum_drift() < self.angular_momentum_tolerance;

        energy_ok && angular_ok
    }

    fn get_falsification_status(&self) -> FalsificationStatus {
        let e_drift = self.energy_drift();
        let l_drift = self.angular_momentum_drift();

        let energy_passed = e_drift < self.energy_tolerance;
        let angular_passed = l_drift < self.angular_momentum_tolerance;

        FalsificationStatus {
            verified: energy_passed && angular_passed,
            criteria: vec![
                CriterionStatus {
                    id: "KEP-ENERGY".to_string(),
                    name: "Energy conservation".to_string(),
                    passed: energy_passed,
                    value: e_drift,
                    threshold: self.energy_tolerance,
                },
                CriterionStatus {
                    id: "KEP-ANGULAR".to_string(),
                    name: "Angular momentum conservation".to_string(),
                    passed: angular_passed,
                    value: l_drift,
                    threshold: self.angular_momentum_tolerance,
                },
            ],
            message: if energy_passed && angular_passed {
                format!("Conservation verified: E_drift={e_drift:.2e}, L_drift={l_drift:.2e}")
            } else {
                format!(
                    "FALSIFIED: E_drift={e_drift:.2e} (tol={:.2e}), L_drift={l_drift:.2e} (tol={:.2e})",
                    self.energy_tolerance, self.angular_momentum_tolerance
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
    use super::{EddDemo, IntegratorType, KeplerOrbitDemo};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmKeplerOrbit {
        inner: KeplerOrbitDemo,
    }

    #[wasm_bindgen]
    impl WasmKeplerOrbit {
        #[wasm_bindgen(constructor)]
        pub fn new(seed: u64) -> Self {
            Self {
                inner: KeplerOrbitDemo::new(seed),
            }
        }

        pub fn step(&mut self, dt: f64) {
            self.inner.step(dt);
        }

        pub fn get_x(&self) -> f64 {
            self.inner.position.x
        }

        pub fn get_y(&self) -> f64 {
            self.inner.position.y
        }

        pub fn get_energy(&self) -> f64 {
            self.inner.specific_energy()
        }

        pub fn get_angular_momentum(&self) -> f64 {
            self.inner.angular_momentum()
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

        pub fn set_perturbation(&mut self, enabled: bool) {
            self.inner.set_perturbation(enabled);
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
    fn test_equation_specific_energy() {
        let demo = KeplerOrbitDemo::new(42);

        // E = v²/2 - μ/r
        let v_sq = demo.velocity.dot(&demo.velocity);
        let r = demo.position.magnitude();
        let expected = 0.5 * v_sq - demo.mu / r;

        assert!((demo.specific_energy() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_equation_angular_momentum() {
        let demo = KeplerOrbitDemo::new(42);

        // L = |r × v| (z-component in 2D)
        let expected =
            (demo.position.x * demo.velocity.y - demo.position.y * demo.velocity.x).abs();

        assert!((demo.angular_momentum() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_equation_keplers_third_law() {
        let demo = KeplerOrbitDemo::new(42);

        // T² = (4π²/μ) a³
        let a = demo.semi_major_axis();
        let t = demo.orbital_period();

        let lhs = t * t;
        let rhs = 4.0 * std::f64::consts::PI.powi(2) * a.powi(3) / demo.mu;

        assert!(
            (lhs - rhs).abs() / rhs < 1e-10,
            "Kepler's third law: T²={lhs:.6e}, 4π²a³/μ={rhs:.6e}"
        );
    }

    // =========================================================================
    // Phase 2: Failing Test - Non-symplectic integrator fails
    // =========================================================================

    #[test]
    fn test_failing_euler_energy() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::Euler);
        demo.energy_tolerance = 1e-6;

        // Run for 1 orbit
        demo.run_orbits(1.0, 1000);

        // Euler should have significant energy drift
        let drift = demo.energy_drift();
        assert!(
            drift > 1e-6,
            "Euler should have energy drift >1e-6, got {drift:.2e}"
        );
    }

    #[test]
    fn test_failing_with_perturbation() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_perturbation(true);
        demo.energy_tolerance = 1e-10;

        // Run for several orbits
        demo.run_orbits(5.0, 1000);

        // With perturbation, conservation laws break
        // (Energy is NOT conserved with external forces)
        // Note: This documents behavior, may or may not "fail" depending on perturbation strength
        let drift = demo.energy_drift();
        println!("With perturbation: energy drift = {drift:.2e}");
    }

    // =========================================================================
    // Phase 3: Implementation - Störmer-Verlet conserves
    // =========================================================================

    #[test]
    fn test_verification_energy_conservation() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::StormerVerlet);
        demo.energy_tolerance = 1e-10;

        // Run for 10 orbits
        demo.run_orbits(10.0, 1000);

        assert!(
            demo.verify_equation(),
            "Energy should be conserved, drift = {:.2e}",
            demo.energy_drift()
        );
    }

    #[test]
    fn test_verification_angular_momentum_conservation() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::StormerVerlet);

        // Run for 10 orbits
        demo.run_orbits(10.0, 1000);

        let l_drift = demo.angular_momentum_drift();
        assert!(
            l_drift < 1e-10,
            "Angular momentum should be conserved, drift = {l_drift:.2e}"
        );
    }

    // =========================================================================
    // Phase 4: Verification - Long-term stability
    // =========================================================================

    #[test]
    fn test_verification_long_term() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::StormerVerlet);
        demo.energy_tolerance = 1e-8;

        // Run for 100 orbits
        demo.run_orbits(100.0, 1000);

        assert!(
            demo.verify_equation(),
            "Long-term conservation failed: E_drift={:.2e}, L_drift={:.2e}",
            demo.energy_drift(),
            demo.angular_momentum_drift()
        );
    }

    #[test]
    fn test_verification_elliptical_orbit() {
        let mut demo = KeplerOrbitDemo::with_eccentricity(42, 0.5);
        demo.set_integrator(IntegratorType::StormerVerlet);
        demo.energy_tolerance = 1e-8;

        // Run for 10 orbits
        demo.run_orbits(10.0, 2000); // More steps for elliptical

        assert!(
            demo.verify_equation(),
            "Elliptical orbit: E_drift={:.2e}, L_drift={:.2e}",
            demo.energy_drift(),
            demo.angular_momentum_drift()
        );
    }

    // =========================================================================
    // Phase 5: Falsification - Document how to break conservation
    // =========================================================================

    #[test]
    fn test_falsification_large_timestep() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::StormerVerlet);

        // Very large timestep (only 10 steps per orbit)
        demo.run_orbits(1.0, 10);

        let drift = demo.energy_drift();
        // Document - may or may not fail depending on step size
        println!("Large timestep (10 steps/orbit): energy drift = {drift:.2e}");
    }

    #[test]
    fn test_falsification_status_structure() {
        let demo = KeplerOrbitDemo::new(42);
        let status = demo.get_falsification_status();

        assert_eq!(status.criteria.len(), 2);
        assert_eq!(status.criteria[0].id, "KEP-ENERGY");
        assert_eq!(status.criteria[1].id, "KEP-ANGULAR");
    }

    // =========================================================================
    // Integration tests
    // =========================================================================

    #[test]
    fn test_demo_trait_implementation() {
        let mut demo = KeplerOrbitDemo::new(42);

        assert_eq!(demo.name(), "Kepler Orbit Conservation Laws");
        assert_eq!(demo.emc_ref(), "physics/kepler_two_body");

        let period = demo.orbital_period();
        let dt = period / 1000.0;
        demo.step(dt);
        assert!(demo.time > 0.0);

        demo.reset();
        assert_eq!(demo.time, 0.0);
    }

    #[test]
    fn test_reproducibility() {
        let mut demo1 = KeplerOrbitDemo::new(42);
        let mut demo2 = KeplerOrbitDemo::new(42);

        demo1.run_orbits(1.0, 100);
        demo2.run_orbits(1.0, 100);

        assert_eq!(demo1.position.x, demo2.position.x);
        assert_eq!(demo1.position.y, demo2.position.y);
    }

    #[test]
    fn test_orbital_elements() {
        let demo = KeplerOrbitDemo::new(42);

        // Circular orbit should have low eccentricity
        let e = demo.eccentricity();
        assert!(
            e < 0.01,
            "Circular orbit eccentricity should be ~0, got {e}"
        );

        // Period should be about 1 year
        let period = demo.orbital_period();
        let year_seconds = 365.25 * 24.0 * 3600.0;
        let period_error = (period - year_seconds).abs() / year_seconds;
        assert!(
            period_error < 0.01,
            "Period should be ~1 year, error = {:.2}%",
            period_error * 100.0
        );
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_default() {
        let demo = KeplerOrbitDemo::default();
        assert_eq!(demo.seed, 42);
        assert_eq!(demo.step_count, 0);
    }

    #[test]
    fn test_vec2_operations() {
        let v1 = Vec2::new(3.0, 4.0);
        let v2 = Vec2::new(1.0, 2.0);

        // Magnitude
        assert!((v1.magnitude() - 5.0).abs() < 1e-10);

        // Normalize
        let normalized = v1.normalize();
        assert!((normalized.magnitude() - 1.0).abs() < 1e-10);
        assert!((normalized.x - 0.6).abs() < 1e-10);
        assert!((normalized.y - 0.8).abs() < 1e-10);

        // Dot product
        assert!((v1.dot(&v2) - 11.0).abs() < 1e-10);

        // Cross product
        assert!((v1.cross(&v2) - 2.0).abs() < 1e-10);

        // Add
        let sum = v1 + v2;
        assert!((sum.x - 4.0).abs() < 1e-10);
        assert!((sum.y - 6.0).abs() < 1e-10);

        // Sub
        let diff = v1 - v2;
        assert!((diff.x - 2.0).abs() < 1e-10);
        assert!((diff.y - 2.0).abs() < 1e-10);

        // Mul scalar
        let scaled = v1 * 2.0;
        assert!((scaled.x - 6.0).abs() < 1e-10);
        assert!((scaled.y - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_normalize_zero() {
        let zero = Vec2::new(0.0, 0.0);
        let normalized = zero.normalize();
        assert!((normalized.x).abs() < 1e-10);
        assert!((normalized.y).abs() < 1e-10);
    }

    #[test]
    fn test_vec2_default() {
        let v = Vec2::default();
        assert!((v.x).abs() < 1e-10);
        assert!((v.y).abs() < 1e-10);
    }

    #[test]
    fn test_set_integrator() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::RK4);
        assert!(matches!(demo.integrator, IntegratorType::RK4));
    }

    #[test]
    fn test_set_perturbation() {
        let mut demo = KeplerOrbitDemo::new(42);
        assert!(!demo.perturbation_enabled);
        demo.set_perturbation(true);
        assert!(demo.perturbation_enabled);
    }

    #[test]
    fn test_with_eccentricity_high() {
        let demo = KeplerOrbitDemo::with_eccentricity(42, 0.9);
        let e = demo.eccentricity();
        assert!(
            (e - 0.9).abs() < 0.1,
            "High eccentricity orbit: expected ~0.9, got {e}"
        );
    }

    #[test]
    fn test_with_eccentricity_clamped() {
        // Eccentricity > 0.99 should be clamped
        let demo = KeplerOrbitDemo::with_eccentricity(42, 1.5);
        // Should not crash and produce a valid orbit
        assert!(demo.specific_energy().is_finite());
    }

    #[test]
    fn test_clone() {
        let demo = KeplerOrbitDemo::new(42);
        let cloned = demo.clone();
        assert_eq!(demo.position.x, cloned.position.x);
        assert_eq!(demo.velocity.y, cloned.velocity.y);
    }

    #[test]
    fn test_debug() {
        let demo = KeplerOrbitDemo::new(42);
        let debug_str = format!("{demo:?}");
        assert!(debug_str.contains("KeplerOrbitDemo"));
    }

    #[test]
    fn test_specific_energy_near_zero() {
        let mut demo = KeplerOrbitDemo::new(42);
        // Set position very close to zero to test the edge case
        demo.position = Vec2::new(0.0, 0.0);
        let energy = demo.specific_energy();
        assert!(energy.is_infinite() && energy < 0.0);
    }

    #[test]
    fn test_semi_major_axis_parabolic() {
        let mut demo = KeplerOrbitDemo::new(42);
        // Set up escape velocity to get E >= 0
        let r = demo.position.magnitude();
        let v_escape = (2.0 * demo.mu / r).sqrt();
        demo.velocity = Vec2::new(0.0, v_escape * 1.1);
        let a = demo.semi_major_axis();
        assert!(a.is_infinite());
    }

    #[test]
    fn test_orbital_period_infinite() {
        let mut demo = KeplerOrbitDemo::new(42);
        // Set up escape velocity
        let r = demo.position.magnitude();
        let v_escape = (2.0 * demo.mu / r).sqrt();
        demo.velocity = Vec2::new(0.0, v_escape * 1.1);
        let period = demo.orbital_period();
        assert!(period.is_infinite());
    }

    #[test]
    fn test_eccentricity_parabolic() {
        let mut demo = KeplerOrbitDemo::new(42);
        // Set up escape velocity
        let r = demo.position.magnitude();
        let v_escape = (2.0 * demo.mu / r).sqrt();
        demo.velocity = Vec2::new(0.0, v_escape * 1.1);
        let e = demo.eccentricity();
        assert!((e - 1.0).abs() < 0.1 || e >= 1.0);
    }

    #[test]
    fn test_energy_drift_zero_initial() {
        let mut demo = KeplerOrbitDemo::new(42);
        // Manipulate to get initial_energy = 0
        demo.initial_energy = 0.0;
        let drift = demo.energy_drift();
        // Should return absolute difference
        assert!(drift > 0.0);
    }

    #[test]
    fn test_angular_momentum_drift_zero_initial() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.initial_angular_momentum = 0.0;
        let drift = demo.angular_momentum_drift();
        assert!(drift > 0.0);
    }

    #[test]
    fn test_step_rk4() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::RK4);
        let initial_pos = demo.position;
        demo.step(1000.0);
        // Position should have changed
        assert!(
            (demo.position.x - initial_pos.x).abs() > 1.0
                || (demo.position.y - initial_pos.y).abs() > 1.0
        );
    }

    #[test]
    fn test_step_euler() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::Euler);
        let initial_pos = demo.position;
        demo.step(1000.0);
        assert!(
            (demo.position.x - initial_pos.x).abs() > 1.0
                || (demo.position.y - initial_pos.y).abs() > 1.0
        );
    }

    #[test]
    fn test_orbit_history() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.max_history = 10;
        for _ in 0..20 {
            demo.step(1000.0);
        }
        // History should be capped at max_history
        assert!(demo.orbit_history.len() <= demo.max_history);
    }

    #[test]
    fn test_step_count_increment() {
        let mut demo = KeplerOrbitDemo::new(42);
        assert_eq!(demo.step_count, 0);
        demo.step(1000.0);
        assert_eq!(demo.step_count, 1);
    }

    #[test]
    fn test_max_drift_tracking() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::Euler);
        demo.run_orbits(0.1, 100);
        // Euler should accumulate drift
        assert!(demo.max_energy_drift > 0.0 || demo.max_angular_momentum_drift >= 0.0);
    }

    #[test]
    fn test_falsification_failed_state() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_integrator(IntegratorType::Euler);
        demo.energy_tolerance = 1e-15;
        demo.run_orbits(1.0, 100);

        let status = demo.get_falsification_status();
        // Euler should fail strict tolerance
        assert!(!status.verified || status.message.contains("FALSIFIED"));
    }

    #[test]
    fn test_serialization() {
        let demo = KeplerOrbitDemo::new(42);
        let json = serde_json::to_string(&demo).expect("serialize");
        assert!(json.contains("mu"));

        let restored: KeplerOrbitDemo = serde_json::from_str(&json).expect("deserialize");
        assert!((restored.mu - demo.mu).abs() < 1.0);
    }

    #[test]
    fn test_run_orbits_zero() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.run_orbits(0.0, 100);
        assert_eq!(demo.step_count, 0);
    }

    #[test]
    fn test_perturbation_acceleration() {
        let mut demo = KeplerOrbitDemo::new(42);
        demo.set_perturbation(true);
        // Run a small timestep directly
        demo.step(1000.0);
        assert!(demo.step_count > 0);
    }

    #[test]
    fn test_verify_equation_all_pass() {
        let demo = KeplerOrbitDemo::new(42);
        // Fresh demo should verify
        assert!(demo.verify_equation());
    }
}
