//! WASM bindings for orbit simulation.
//!
//! Provides JavaScript-callable functions for orbital mechanics simulation
//! with the same physics as the native implementation (AC-6: epsilon-identical).
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { OrbitSimulation, SimulationConfig } from 'simular';
//!
//! async function main() {
//!     await init();
//!
//!     const sim = OrbitSimulation.earth_sun();
//!     for (let i = 0; i < 365; i++) {
//!         sim.step_days(1.0);
//!         const state = sim.get_state();
//!         console.log(`Day ${i}: Earth at (${state.earth_x}, ${state.earth_y})`);
//!     }
//! }
//! ```

// WASM-bindgen exports don't need #[must_use] - values returned to JS
#![allow(clippy::must_use_candidate)]

use wasm_bindgen::prelude::*;

use crate::demos::{DemoEngine, OrbitalEngine};
use crate::orbit::jidoka::{JidokaResponse, OrbitJidokaConfig, OrbitJidokaGuard};
use crate::orbit::physics::{NBodyState, YoshidaIntegrator};
use crate::orbit::scenarios::KeplerConfig;
use crate::orbit::units::{OrbitTime, AU};

/// Embedded default Earth-Sun YAML configuration.
const DEFAULT_ORBIT_YAML: &str = include_str!("../../examples/experiments/orbit_earth_sun.yaml");

/// WASM-exported orbit simulation state.
#[wasm_bindgen]
pub struct OrbitSimulation {
    state: NBodyState,
    integrator: YoshidaIntegrator,
    jidoka: OrbitJidokaGuard,
    sim_time_seconds: f64,
    paused: bool,
}

#[wasm_bindgen]
impl OrbitSimulation {
    /// Create a new Earth-Sun Keplerian orbit simulation.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::earth_sun()
    }

    /// Create Earth-Sun system from YAML configuration.
    #[wasm_bindgen]
    pub fn earth_sun() -> Self {
        // Load from embedded YAML for YAML-first architecture
        Self::from_config(&Self::load_default_config())
    }

    /// Load the default configuration from embedded YAML.
    fn load_default_config() -> KeplerConfig {
        OrbitalEngine::from_yaml(DEFAULT_ORBIT_YAML)
            .map(|e| e.kepler_config())
            .unwrap_or_default()
    }

    /// Create from a specific configuration.
    fn from_config(config: &KeplerConfig) -> Self {
        let state = config.build(1e6);

        let mut jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
        jidoka.initialize(&state);

        Self {
            state,
            integrator: YoshidaIntegrator::new(),
            jidoka,
            sim_time_seconds: 0.0,
            paused: false,
        }
    }

    /// Create a circular orbit with custom parameters.
    #[wasm_bindgen]
    pub fn circular_orbit(central_mass_kg: f64, orbiter_mass_kg: f64, radius_m: f64) -> Self {
        let config = KeplerConfig::circular(central_mass_kg, orbiter_mass_kg, radius_m);
        let state = config.build(radius_m * 1e-6);

        let mut jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
        jidoka.initialize(&state);

        Self {
            state,
            integrator: YoshidaIntegrator::new(),
            jidoka,
            sim_time_seconds: 0.0,
            paused: false,
        }
    }

    /// Step the simulation forward by the given time in seconds.
    #[wasm_bindgen]
    pub fn step(&mut self, dt_seconds: f64) -> bool {
        if self.paused {
            return false;
        }

        let dt = OrbitTime::from_seconds(dt_seconds);
        if self.integrator.step(&mut self.state, dt).is_err() {
            self.paused = true;
            return false;
        }

        self.sim_time_seconds += dt_seconds;

        // Check Jidoka guards
        let response = self.jidoka.check(&self.state);
        match response {
            JidokaResponse::Continue | JidokaResponse::Warning { .. } => true,
            JidokaResponse::Pause { .. } | JidokaResponse::Halt { .. } => {
                self.paused = true;
                false
            }
        }
    }

    /// Step the simulation forward by days.
    #[wasm_bindgen]
    pub fn step_days(&mut self, days: f64) -> bool {
        contract_pre_iterator!(days);
        self.step(days * 86400.0)
    }

    /// Step the simulation forward by hours.
    #[wasm_bindgen]
    pub fn step_hours(&mut self, hours: f64) -> bool {
        contract_pre_iterator!(hours);
        self.step(hours * 3600.0)
    }

    /// Run multiple steps with a given dt (for performance).
    #[wasm_bindgen]
    pub fn run_steps(&mut self, num_steps: u32, dt_seconds: f64) -> u32 {
        contract_pre_iterator!(num_steps);
        let mut completed = 0;
        for _ in 0..num_steps {
            if !self.step(dt_seconds) {
                break;
            }
            completed += 1;
        }
        completed
    }

    /// Get simulation time in seconds.
    #[wasm_bindgen(getter)]
    pub fn sim_time(&self) -> f64 {
        self.sim_time_seconds
    }

    /// Get simulation time in days.
    #[wasm_bindgen(getter)]
    pub fn sim_time_days(&self) -> f64 {
        self.sim_time_seconds / 86400.0
    }

    /// Check if simulation is paused.
    #[wasm_bindgen(getter)]
    pub fn paused(&self) -> bool {
        self.paused
    }

    /// Resume a paused simulation.
    #[wasm_bindgen]
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Reset the simulation to initial state.
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        let config = Self::load_default_config();
        self.state = config.build(1e6);
        self.jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
        self.jidoka.initialize(&self.state);
        self.sim_time_seconds = 0.0;
        self.paused = false;
    }

    /// Get the number of bodies.
    #[wasm_bindgen(getter)]
    pub fn num_bodies(&self) -> usize {
        self.state.num_bodies()
    }

    /// Get total energy of the system (J).
    #[wasm_bindgen(getter)]
    pub fn total_energy(&self) -> f64 {
        self.state.total_energy()
    }

    /// Get total angular momentum magnitude (kg*m^2/s).
    #[wasm_bindgen(getter)]
    pub fn angular_momentum(&self) -> f64 {
        self.state.angular_momentum_magnitude()
    }

    /// Get body position X in meters.
    #[wasm_bindgen]
    pub fn body_x(&self, index: usize) -> f64 {
        if index < self.state.bodies.len() {
            let (x, _, _) = self.state.bodies[index].position.as_meters();
            x
        } else {
            f64::NAN
        }
    }

    /// Get body position Y in meters.
    #[wasm_bindgen]
    pub fn body_y(&self, index: usize) -> f64 {
        if index < self.state.bodies.len() {
            let (_, y, _) = self.state.bodies[index].position.as_meters();
            y
        } else {
            f64::NAN
        }
    }

    /// Get body position Z in meters.
    #[wasm_bindgen]
    pub fn body_z(&self, index: usize) -> f64 {
        if index < self.state.bodies.len() {
            let (_, _, z) = self.state.bodies[index].position.as_meters();
            z
        } else {
            f64::NAN
        }
    }

    /// Get body position X in AU.
    #[wasm_bindgen]
    pub fn body_x_au(&self, index: usize) -> f64 {
        self.body_x(index) / AU
    }

    /// Get body position Y in AU.
    #[wasm_bindgen]
    pub fn body_y_au(&self, index: usize) -> f64 {
        self.body_y(index) / AU
    }

    /// Get body velocity X in m/s.
    #[wasm_bindgen]
    pub fn body_vx(&self, index: usize) -> f64 {
        if index < self.state.bodies.len() {
            let (vx, _, _) = self.state.bodies[index].velocity.as_mps();
            vx
        } else {
            f64::NAN
        }
    }

    /// Get body velocity Y in m/s.
    #[wasm_bindgen]
    pub fn body_vy(&self, index: usize) -> f64 {
        if index < self.state.bodies.len() {
            let (_, vy, _) = self.state.bodies[index].velocity.as_mps();
            vy
        } else {
            f64::NAN
        }
    }

    /// Get body velocity Z in m/s.
    #[wasm_bindgen]
    pub fn body_vz(&self, index: usize) -> f64 {
        if index < self.state.bodies.len() {
            let (_, _, vz) = self.state.bodies[index].velocity.as_mps();
            vz
        } else {
            f64::NAN
        }
    }

    /// Get body mass in kg.
    #[wasm_bindgen]
    pub fn body_mass(&self, index: usize) -> f64 {
        if index < self.state.bodies.len() {
            self.state.bodies[index].mass.as_kg()
        } else {
            f64::NAN
        }
    }

    /// Get Jidoka status as JSON string.
    #[wasm_bindgen]
    pub fn jidoka_status_json(&self) -> String {
        let status = self.jidoka.status();
        format!(
            r#"{{"energy_ok":{},"angular_momentum_ok":{},"finite_ok":{},"energy_error":{},"angular_momentum_error":{},"warning_count":{}}}"#,
            status.energy_ok,
            status.angular_momentum_ok,
            status.finite_ok,
            status.energy_error,
            status.angular_momentum_error,
            status.warning_count,
        )
    }

    /// Get all body positions as a flat array [x0, y0, z0, x1, y1, z1, ...] in meters.
    #[wasm_bindgen]
    pub fn positions_flat(&self) -> Vec<f64> {
        let mut positions = Vec::with_capacity(self.state.bodies.len() * 3);
        for body in &self.state.bodies {
            let (x, y, z) = body.position.as_meters();
            positions.push(x);
            positions.push(y);
            positions.push(z);
        }
        positions
    }

    /// Get all body velocities as a flat array [vx0, vy0, vz0, ...] in m/s.
    #[wasm_bindgen]
    pub fn velocities_flat(&self) -> Vec<f64> {
        let mut velocities = Vec::with_capacity(self.state.bodies.len() * 3);
        for body in &self.state.bodies {
            let (vx, vy, vz) = body.velocity.as_mps();
            velocities.push(vx);
            velocities.push(vy);
            velocities.push(vz);
        }
        velocities
    }

    /// Get all body positions in AU as flat array.
    #[wasm_bindgen]
    pub fn positions_au_flat(&self) -> Vec<f64> {
        let mut positions = Vec::with_capacity(self.state.bodies.len() * 3);
        for body in &self.state.bodies {
            let (x, y, z) = body.position.as_meters();
            positions.push(x / AU);
            positions.push(y / AU);
            positions.push(z / AU);
        }
        positions
    }
}

impl Default for OrbitSimulation {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize WASM module.
/// Called automatically when the WASM module is loaded.
#[wasm_bindgen(start)]
pub fn init() {
    // Future: Add console_error_panic_hook for better error messages
    // when debugging WASM in browser dev tools.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbit_simulation_new() {
        let sim = OrbitSimulation::new();
        assert_eq!(sim.num_bodies(), 2);
        assert!(!sim.paused());
    }

    #[test]
    fn test_orbit_simulation_step() {
        let mut sim = OrbitSimulation::new();
        let initial_energy = sim.total_energy();

        assert!(sim.step(3600.0)); // 1 hour
        assert!(sim.sim_time() > 0.0);

        // Energy should be conserved
        let final_energy = sim.total_energy();
        let rel_error = (final_energy - initial_energy).abs() / initial_energy.abs();
        assert!(rel_error < 1e-9);
    }

    #[test]
    fn test_orbit_simulation_step_days() {
        let mut sim = OrbitSimulation::new();
        assert!(sim.step_days(1.0));
        assert!((sim.sim_time_days() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_orbit_simulation_run_steps() {
        let mut sim = OrbitSimulation::new();
        let completed = sim.run_steps(24, 3600.0); // 24 hours
        assert_eq!(completed, 24);
        assert!((sim.sim_time_days() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_orbit_simulation_body_positions() {
        let sim = OrbitSimulation::new();

        // Sun at origin
        assert!(sim.body_x(0).abs() < 1e-10);
        assert!(sim.body_y(0).abs() < 1e-10);

        // Earth at ~1 AU
        let earth_r = (sim.body_x(1).powi(2) + sim.body_y(1).powi(2)).sqrt();
        assert!((earth_r / AU - 0.983).abs() < 0.01); // Perihelion
    }

    #[test]
    fn test_orbit_simulation_reset() {
        let mut sim = OrbitSimulation::new();
        let initial_x = sim.body_x(1);

        sim.step_days(10.0);
        assert!((sim.body_x(1) - initial_x).abs() > 1e6); // Moved

        sim.reset();
        assert!((sim.body_x(1) - initial_x).abs() < 1.0); // Back to start
        assert!(sim.sim_time() < 0.001);
    }

    #[test]
    fn test_orbit_simulation_positions_flat() {
        let sim = OrbitSimulation::new();
        let positions = sim.positions_flat();

        assert_eq!(positions.len(), 6); // 2 bodies * 3 coords
        assert!(positions[0].abs() < 1e-10); // Sun x
        assert!(positions[1].abs() < 1e-10); // Sun y
    }

    #[test]
    fn test_orbit_simulation_jidoka_status() {
        let sim = OrbitSimulation::new();
        let status = sim.jidoka_status_json();

        assert!(status.contains("energy_ok"));
        assert!(status.contains("angular_momentum_ok"));
        assert!(status.contains("finite_ok"));
        assert!(status.contains("warning_count"));
    }

    #[test]
    fn test_circular_orbit() {
        let sim = OrbitSimulation::circular_orbit(
            1.989e30, // Sun mass
            5.972e24, // Earth mass
            AU,       // 1 AU
        );
        assert_eq!(sim.num_bodies(), 2);
    }

    #[test]
    fn test_orbit_simulation_invalid_body_index() {
        let sim = OrbitSimulation::new();
        assert!(sim.body_x(999).is_nan());
        assert!(sim.body_y(999).is_nan());
        assert!(sim.body_vx(999).is_nan());
        assert!(sim.body_mass(999).is_nan());
    }

    #[test]
    fn test_orbit_simulation_body_z() {
        let sim = OrbitSimulation::new();
        // Z should be near zero for planar orbit
        assert!(sim.body_z(0).abs() < 1e-10);
        assert!(sim.body_z(1).abs() < 1e-10);
        // Invalid index
        assert!(sim.body_z(999).is_nan());
    }

    #[test]
    fn test_orbit_simulation_body_y_au() {
        let sim = OrbitSimulation::new();
        // Sun at origin
        assert!(sim.body_y_au(0).abs() < 1e-15);
    }

    #[test]
    fn test_orbit_simulation_body_vy() {
        let sim = OrbitSimulation::new();
        // Earth has velocity in y direction
        let vy = sim.body_vy(1);
        assert!(vy.abs() > 20000.0); // Should be ~30 km/s
                                     // Invalid index
        assert!(sim.body_vy(999).is_nan());
    }

    #[test]
    fn test_orbit_simulation_body_vz() {
        let sim = OrbitSimulation::new();
        // Z velocity should be zero for planar orbit
        assert!(sim.body_vz(0).abs() < 1e-10);
        assert!(sim.body_vz(1).abs() < 1e-10);
        // Invalid index
        assert!(sim.body_vz(999).is_nan());
    }

    #[test]
    fn test_orbit_simulation_step_hours() {
        let mut sim = OrbitSimulation::new();
        assert!(sim.step_hours(1.0)); // 1 hour
        assert!((sim.sim_time() - 3600.0).abs() < 0.001);
    }

    #[test]
    fn test_orbit_simulation_pause_resume() {
        let mut sim = OrbitSimulation::new();
        assert!(!sim.paused());

        // Manually pause
        sim.paused = true;
        assert!(sim.paused());

        // Step should fail when paused
        assert!(!sim.step(3600.0));

        // Resume
        sim.resume();
        assert!(!sim.paused());

        // Step should work again
        assert!(sim.step(3600.0));
    }

    #[test]
    fn test_orbit_simulation_velocities_flat() {
        let sim = OrbitSimulation::new();
        let velocities = sim.velocities_flat();

        assert_eq!(velocities.len(), 6); // 2 bodies * 3 coords
                                         // Sun should have near-zero velocity
        assert!(velocities[0].abs() < 1e-10);
        assert!(velocities[1].abs() < 1e-10);
        assert!(velocities[2].abs() < 1e-10);
    }

    #[test]
    fn test_orbit_simulation_positions_au_flat() {
        let sim = OrbitSimulation::new();
        let positions = sim.positions_au_flat();

        assert_eq!(positions.len(), 6); // 2 bodies * 3 coords
                                        // Sun at origin
        assert!(positions[0].abs() < 1e-15);
        assert!(positions[1].abs() < 1e-15);
        assert!(positions[2].abs() < 1e-15);
        // Earth at ~1 AU in x
        assert!((positions[3] - 0.983).abs() < 0.02); // Perihelion
    }

    #[test]
    fn test_orbit_simulation_default() {
        let sim = OrbitSimulation::default();
        assert_eq!(sim.num_bodies(), 2);
    }

    #[test]
    fn test_orbit_simulation_angular_momentum() {
        let sim = OrbitSimulation::new();
        let l = sim.angular_momentum();
        assert!(l > 0.0);
    }

    #[test]
    fn test_orbit_simulation_run_steps_paused() {
        let mut sim = OrbitSimulation::new();
        sim.paused = true;

        // Should complete 0 steps when paused
        let completed = sim.run_steps(10, 3600.0);
        assert_eq!(completed, 0);
    }

    #[test]
    fn test_orbit_simulation_body_x_au() {
        let sim = OrbitSimulation::new();
        // Earth at perihelion ~0.983 AU
        let x_au = sim.body_x_au(1);
        assert!((x_au - 0.983).abs() < 0.02);
    }

    #[test]
    fn test_wasm_init() {
        // The init function should not panic
        init();
    }

    #[test]
    fn test_orbit_simulation_sim_time_days() {
        let mut sim = OrbitSimulation::new();
        sim.step_days(10.5);
        assert!((sim.sim_time_days() - 10.5).abs() < 0.001);
    }
}
