//! Physics domain engine.
//!
//! Implements numerical integration for physical simulations:
//! - Verlet (symplectic, 2nd order)
//! - Euler (1st order)
//! - RK4 (4th order)
//!
//! # Energy Conservation
//!
//! Symplectic integrators (Verlet) preserve phase space volume,
//! leading to bounded energy error over long simulations.

use crate::engine::state::{SimState, Vec3};
use crate::error::SimResult;

/// Force field trait for computing accelerations.
pub trait ForceField {
    /// Compute acceleration for a body at given position.
    fn acceleration(&self, position: &Vec3, mass: f64) -> Vec3;

    /// Compute potential energy at given position.
    fn potential(&self, position: &Vec3, mass: f64) -> f64;
}

/// Simple gravity force field.
#[derive(Debug, Clone)]
pub struct GravityField {
    /// Gravitational acceleration (default: -9.81 m/s² in z).
    pub g: Vec3,
}

impl Default for GravityField {
    fn default() -> Self {
        Self {
            g: Vec3::new(0.0, 0.0, -9.81),
        }
    }
}

impl ForceField for GravityField {
    fn acceleration(&self, _position: &Vec3, _mass: f64) -> Vec3 {
        self.g
    }

    fn potential(&self, position: &Vec3, mass: f64) -> f64 {
        // PE = m * g * h (assuming g points in -z direction)
        -mass * self.g.dot(position)
    }
}

/// Central force field (e.g., gravity toward origin).
#[derive(Debug, Clone)]
pub struct CentralForceField {
    /// Gravitational parameter (G * M).
    pub mu: f64,
    /// Center position.
    pub center: Vec3,
}

impl CentralForceField {
    /// Create a new central force field.
    #[must_use]
    pub const fn new(mu: f64, center: Vec3) -> Self {
        Self { mu, center }
    }
}

impl ForceField for CentralForceField {
    fn acceleration(&self, position: &Vec3, _mass: f64) -> Vec3 {
        let r = *position - self.center;
        let r_mag = r.magnitude();

        if r_mag < f64::EPSILON {
            return Vec3::zero();
        }

        // a = -μ/r² * r̂
        let r_hat = r.normalize();
        r_hat.scale(-self.mu / (r_mag * r_mag))
    }

    fn potential(&self, position: &Vec3, mass: f64) -> f64 {
        let r = (*position - self.center).magnitude();

        if r < f64::EPSILON {
            return 0.0;
        }

        // PE = -G*M*m/r = -μ*m/r
        -self.mu * mass / r
    }
}

/// Numerical integrator trait.
pub trait Integrator {
    /// Step the state forward by one timestep.
    ///
    /// # Errors
    ///
    /// Returns error if integration fails.
    fn step(&self, state: &mut SimState, force_field: &dyn ForceField, dt: f64) -> SimResult<()>;

    /// Get the error order of this integrator.
    fn error_order(&self) -> u32;

    /// Check if integrator is symplectic (preserves phase space volume).
    fn is_symplectic(&self) -> bool;
}

/// Störmer-Verlet symplectic integrator.
///
/// Second-order, symplectic method with excellent energy conservation.
/// Error is O(h²) but energy oscillates around true value without drift.
///
/// Algorithm:
/// ```text
/// q_{n+1/2} = q_n + (h/2) * v_n
/// v_{n+1}   = v_n + h * a(q_{n+1/2})
/// q_{n+1}   = q_{n+1/2} + (h/2) * v_{n+1}
/// ```
#[derive(Debug, Clone, Default)]
pub struct VerletIntegrator;

impl VerletIntegrator {
    /// Create a new Verlet integrator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Integrator for VerletIntegrator {
    fn step(&self, state: &mut SimState, force_field: &dyn ForceField, dt: f64) -> SimResult<()> {
        let n = state.num_bodies();
        let half_dt = dt / 2.0;

        // Half-step position update: q_{n+1/2} = q_n + (h/2) * v_n
        for i in 0..n {
            let pos = state.positions()[i];
            let vel = state.velocities()[i];
            state.set_position(i, pos + vel * half_dt);
        }

        // Compute accelerations at half-step positions
        let mut accelerations = Vec::with_capacity(n);
        let mut potential_energy = 0.0;

        for i in 0..n {
            let pos = state.positions()[i];
            let mass = state.masses()[i];
            accelerations.push(force_field.acceleration(&pos, mass));
            potential_energy += force_field.potential(&pos, mass);
        }

        // Full-step velocity update: v_{n+1} = v_n + h * a(q_{n+1/2})
        for i in 0..n {
            let vel = state.velocities()[i];
            state.set_velocity(i, vel + accelerations[i] * dt);
        }

        // Half-step position update: q_{n+1} = q_{n+1/2} + (h/2) * v_{n+1}
        for i in 0..n {
            let pos = state.positions()[i];
            let vel = state.velocities()[i];
            state.set_position(i, pos + vel * half_dt);
        }

        // Update potential energy
        state.set_potential_energy(potential_energy);

        Ok(())
    }

    fn error_order(&self) -> u32 {
        2
    }

    fn is_symplectic(&self) -> bool {
        true
    }
}

/// Runge-Kutta 4th order integrator.
///
/// Fourth-order accurate, non-symplectic. Excellent for smooth problems
/// but energy may drift in long-term simulations.
///
/// Algorithm (classical RK4):
/// ```text
/// k1_v = a(q_n)
/// k1_q = v_n
///
/// k2_v = a(q_n + h/2 * k1_q)
/// k2_q = v_n + h/2 * k1_v
///
/// k3_v = a(q_n + h/2 * k2_q)
/// k3_q = v_n + h/2 * k2_v
///
/// k4_v = a(q_n + h * k3_q)
/// k4_q = v_n + h * k3_v
///
/// v_{n+1} = v_n + h/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
/// q_{n+1} = q_n + h/6 * (k1_q + 2*k2_q + 2*k3_q + k4_q)
/// ```
#[derive(Debug, Clone, Default)]
pub struct RK4Integrator;

impl RK4Integrator {
    /// Create a new RK4 integrator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Integrator for RK4Integrator {
    fn step(&self, state: &mut SimState, force_field: &dyn ForceField, dt: f64) -> SimResult<()> {
        let n = state.num_bodies();
        let half_dt = dt / 2.0;
        let sixth_dt = dt / 6.0;

        // Save initial state
        let initial_positions: Vec<Vec3> = state.positions().to_vec();
        let initial_velocities: Vec<Vec3> = state.velocities().to_vec();

        // Stage 1: k1 at t_n
        let mut k1_v = Vec::with_capacity(n);
        let k1_q: Vec<Vec3> = initial_velocities.clone();

        for i in 0..n {
            let mass = state.masses()[i];
            k1_v.push(force_field.acceleration(&initial_positions[i], mass));
        }

        // Stage 2: k2 at t_n + h/2
        let mut k2_v = Vec::with_capacity(n);
        let mut k2_q = Vec::with_capacity(n);

        for i in 0..n {
            let pos = initial_positions[i] + k1_q[i] * half_dt;
            let vel = initial_velocities[i] + k1_v[i] * half_dt;
            let mass = state.masses()[i];

            k2_v.push(force_field.acceleration(&pos, mass));
            k2_q.push(vel);
        }

        // Stage 3: k3 at t_n + h/2
        let mut k3_v = Vec::with_capacity(n);
        let mut k3_q = Vec::with_capacity(n);

        for i in 0..n {
            let pos = initial_positions[i] + k2_q[i] * half_dt;
            let vel = initial_velocities[i] + k2_v[i] * half_dt;
            let mass = state.masses()[i];

            k3_v.push(force_field.acceleration(&pos, mass));
            k3_q.push(vel);
        }

        // Stage 4: k4 at t_n + h
        let mut k4_v = Vec::with_capacity(n);
        let mut k4_q = Vec::with_capacity(n);

        for i in 0..n {
            let pos = initial_positions[i] + k3_q[i] * dt;
            let vel = initial_velocities[i] + k3_v[i] * dt;
            let mass = state.masses()[i];

            k4_v.push(force_field.acceleration(&pos, mass));
            k4_q.push(vel);
        }

        // Final update
        let mut potential_energy = 0.0;

        for i in 0..n {
            // v_{n+1} = v_n + h/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            let new_vel = initial_velocities[i]
                + (k1_v[i] + k2_v[i] * 2.0 + k3_v[i] * 2.0 + k4_v[i]) * sixth_dt;

            // q_{n+1} = q_n + h/6 * (k1_q + 2*k2_q + 2*k3_q + k4_q)
            let new_pos = initial_positions[i]
                + (k1_q[i] + k2_q[i] * 2.0 + k3_q[i] * 2.0 + k4_q[i]) * sixth_dt;

            state.set_velocity(i, new_vel);
            state.set_position(i, new_pos);

            let mass = state.masses()[i];
            potential_energy += force_field.potential(&new_pos, mass);
        }

        state.set_potential_energy(potential_energy);
        Ok(())
    }

    fn error_order(&self) -> u32 {
        4
    }

    fn is_symplectic(&self) -> bool {
        false
    }
}

/// Euler integrator (1st order, non-symplectic).
///
/// Simple but inaccurate. Energy drifts over time.
/// Useful for comparison and debugging.
#[derive(Debug, Clone, Default)]
pub struct EulerIntegrator;

impl EulerIntegrator {
    /// Create a new Euler integrator.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Integrator for EulerIntegrator {
    fn step(&self, state: &mut SimState, force_field: &dyn ForceField, dt: f64) -> SimResult<()> {
        let n = state.num_bodies();
        let mut potential_energy = 0.0;

        for i in 0..n {
            let pos = state.positions()[i];
            let vel = state.velocities()[i];
            let mass = state.masses()[i];

            let acc = force_field.acceleration(&pos, mass);
            potential_energy += force_field.potential(&pos, mass);

            // x_{n+1} = x_n + v_n * dt
            // v_{n+1} = v_n + a_n * dt
            state.set_position(i, pos + vel * dt);
            state.set_velocity(i, vel + acc * dt);
        }

        state.set_potential_energy(potential_energy);
        Ok(())
    }

    fn error_order(&self) -> u32 {
        1
    }

    fn is_symplectic(&self) -> bool {
        false
    }
}

/// Physics engine wrapper.
pub struct PhysicsEngine {
    /// Force field.
    force_field: Box<dyn ForceField + Send + Sync>,
    /// Integrator.
    integrator: Box<dyn Integrator + Send + Sync>,
}

impl std::fmt::Debug for PhysicsEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicsEngine")
            .field("force_field", &"<dyn ForceField>")
            .field("integrator", &"<dyn Integrator>")
            .finish()
    }
}

impl PhysicsEngine {
    /// Create a new physics engine.
    pub fn new<F, I>(force_field: F, integrator: I) -> Self
    where
        F: ForceField + Send + Sync + 'static,
        I: Integrator + Send + Sync + 'static,
    {
        Self {
            force_field: Box::new(force_field),
            integrator: Box::new(integrator),
        }
    }

    /// Create a new physics engine from boxed components.
    #[must_use]
    pub fn new_boxed(
        force_field: Box<dyn ForceField + Send + Sync>,
        integrator: Box<dyn Integrator + Send + Sync>,
    ) -> Self {
        Self {
            force_field,
            integrator,
        }
    }

    /// Step the physics simulation.
    ///
    /// # Errors
    ///
    /// Returns error if integration fails.
    pub fn step(&self, state: &mut SimState, dt: f64) -> SimResult<()> {
        self.integrator.step(state, self.force_field.as_ref(), dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gravity_field() {
        let field = GravityField::default();
        let pos = Vec3::new(0.0, 0.0, 10.0);
        let mass = 1.0;

        let acc = field.acceleration(&pos, mass);
        assert!((acc.z - (-9.81)).abs() < f64::EPSILON);

        let pe = field.potential(&pos, mass);
        assert!((pe - 98.1).abs() < 0.01); // m * g * h = 1 * 9.81 * 10
    }

    #[test]
    fn test_central_force_field() {
        let field = CentralForceField::new(1.0, Vec3::zero());
        let pos = Vec3::new(1.0, 0.0, 0.0);
        let mass = 1.0;

        let acc = field.acceleration(&pos, mass);
        // Should point toward origin
        assert!(acc.x < 0.0);
        assert!(acc.y.abs() < f64::EPSILON);
        assert!(acc.z.abs() < f64::EPSILON);
    }

    #[test]
    fn test_verlet_energy_conservation() {
        let mut state = SimState::new();

        // Simple harmonic oscillator: mass on spring
        // F = -kx, k = 1, m = 1
        state.add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero());

        // Custom spring force
        struct SpringField;
        impl ForceField for SpringField {
            fn acceleration(&self, position: &Vec3, _mass: f64) -> Vec3 {
                // a = -x (k/m = 1)
                Vec3::new(-position.x, -position.y, -position.z)
            }
            fn potential(&self, position: &Vec3, _mass: f64) -> f64 {
                // PE = 0.5 * k * x^2
                0.5 * position.magnitude_squared()
            }
        }

        let integrator = VerletIntegrator::new();
        let dt = 0.001;

        // Initial energy
        integrator.step(&mut state, &SpringField, dt).ok();
        let initial_energy = state.total_energy();

        // Simulate many steps
        for _ in 0..10000 {
            integrator.step(&mut state, &SpringField, dt).ok();
        }

        let final_energy = state.total_energy();
        let drift = (final_energy - initial_energy).abs() / initial_energy;

        // Verlet should have very small energy drift
        assert!(drift < 0.01, "Energy drift {} too large", drift);
    }

    #[test]
    fn test_euler_vs_verlet_energy() {
        // Euler should have worse energy conservation than Verlet

        fn run_simulation<I: Integrator>(integrator: &I, steps: usize, dt: f64) -> f64 {
            let mut state = SimState::new();
            state.add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero());

            struct SpringField;
            impl ForceField for SpringField {
                fn acceleration(&self, position: &Vec3, _mass: f64) -> Vec3 {
                    Vec3::new(-position.x, -position.y, -position.z)
                }
                fn potential(&self, position: &Vec3, _mass: f64) -> f64 {
                    0.5 * position.magnitude_squared()
                }
            }

            integrator.step(&mut state, &SpringField, dt).ok();
            let initial_energy = state.total_energy();

            for _ in 0..steps {
                integrator.step(&mut state, &SpringField, dt).ok();
            }

            (state.total_energy() - initial_energy).abs() / initial_energy
        }

        let verlet_drift = run_simulation(&VerletIntegrator::new(), 1000, 0.01);
        let euler_drift = run_simulation(&EulerIntegrator::new(), 1000, 0.01);

        // Verlet should be much better
        assert!(
            verlet_drift < euler_drift,
            "Verlet drift {} should be less than Euler drift {}",
            verlet_drift,
            euler_drift
        );
    }

    #[test]
    fn test_integrator_properties() {
        let verlet = VerletIntegrator::new();
        assert_eq!(verlet.error_order(), 2);
        assert!(verlet.is_symplectic());

        let euler = EulerIntegrator::new();
        assert_eq!(euler.error_order(), 1);
        assert!(!euler.is_symplectic());

        let rk4 = RK4Integrator::new();
        assert_eq!(rk4.error_order(), 4);
        assert!(!rk4.is_symplectic());
    }

    #[test]
    fn test_rk4_accuracy() {
        // RK4 should be more accurate than Euler for short-term simulation
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero());

        struct SpringField;
        impl ForceField for SpringField {
            fn acceleration(&self, position: &Vec3, _mass: f64) -> Vec3 {
                Vec3::new(-position.x, -position.y, -position.z)
            }
            fn potential(&self, position: &Vec3, _mass: f64) -> f64 {
                0.5 * position.magnitude_squared()
            }
        }

        let rk4 = RK4Integrator::new();
        let dt = 0.01;

        // Initial energy
        rk4.step(&mut state, &SpringField, dt).ok();
        let initial_energy = state.total_energy();

        // Simulate
        for _ in 0..100 {
            rk4.step(&mut state, &SpringField, dt).ok();
        }

        let final_energy = state.total_energy();
        let drift = (final_energy - initial_energy).abs() / initial_energy;

        // RK4 should have good short-term accuracy
        assert!(drift < 0.001, "RK4 energy drift {} too large", drift);
    }

    #[test]
    fn test_rk4_vs_euler() {
        // RK4 should be more accurate than Euler
        fn run_simulation<I: Integrator>(integrator: &I, steps: usize, dt: f64) -> f64 {
            let mut state = SimState::new();
            state.add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero());

            struct SpringField;
            impl ForceField for SpringField {
                fn acceleration(&self, position: &Vec3, _mass: f64) -> Vec3 {
                    Vec3::new(-position.x, -position.y, -position.z)
                }
                fn potential(&self, position: &Vec3, _mass: f64) -> f64 {
                    0.5 * position.magnitude_squared()
                }
            }

            integrator.step(&mut state, &SpringField, dt).ok();
            let initial_energy = state.total_energy();

            for _ in 0..steps {
                integrator.step(&mut state, &SpringField, dt).ok();
            }

            (state.total_energy() - initial_energy).abs() / initial_energy
        }

        let rk4_drift = run_simulation(&RK4Integrator::new(), 100, 0.01);
        let euler_drift = run_simulation(&EulerIntegrator::new(), 100, 0.01);

        assert!(
            rk4_drift < euler_drift,
            "RK4 drift {} should be less than Euler drift {}",
            rk4_drift,
            euler_drift
        );
    }

    #[test]
    fn test_rk4_free_particle() {
        // Free particle should move at constant velocity
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 3.0));

        struct ZeroField;
        impl ForceField for ZeroField {
            fn acceleration(&self, _: &Vec3, _: f64) -> Vec3 {
                Vec3::zero()
            }
            fn potential(&self, _: &Vec3, _: f64) -> f64 {
                0.0
            }
        }

        let rk4 = RK4Integrator::new();
        let dt = 0.1;

        for _ in 0..100 {
            rk4.step(&mut state, &ZeroField, dt).ok();
        }

        // After 10 seconds, position should be (10, 20, 30)
        let pos = state.positions()[0];
        assert!((pos.x - 10.0).abs() < 1e-10, "x={}", pos.x);
        assert!((pos.y - 20.0).abs() < 1e-10, "y={}", pos.y);
        assert!((pos.z - 30.0).abs() < 1e-10, "z={}", pos.z);

        // Velocity should be unchanged
        let vel = state.velocities()[0];
        assert!((vel.x - 1.0).abs() < 1e-10);
        assert!((vel.y - 2.0).abs() < 1e-10);
        assert!((vel.z - 3.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Falsification: Verlet preserves energy for harmonic oscillator.
        #[test]
        fn prop_verlet_energy_conservation(
            x0 in 0.1f64..10.0,  // Ensure non-zero initial displacement
            v0 in -10.0f64..10.0,
            mass in 0.1f64..10.0,
        ) {
            let mut state = SimState::new();
            state.add_body(mass, Vec3::new(x0, 0.0, 0.0), Vec3::new(v0, 0.0, 0.0));

            struct SpringField;
            impl ForceField for SpringField {
                fn acceleration(&self, position: &Vec3, _mass: f64) -> Vec3 {
                    Vec3::new(-position.x, 0.0, 0.0)
                }
                fn potential(&self, position: &Vec3, mass: f64) -> f64 {
                    // PE for spring: 0.5 * k * x^2, with k = mass (so omega = 1)
                    0.5 * mass * position.x * position.x
                }
            }

            let integrator = VerletIntegrator::new();
            let dt = 0.001;

            // Get initial energy after first step
            integrator.step(&mut state, &SpringField, dt).ok();
            let initial_energy = state.total_energy();

            // Simulate
            for _ in 0..1000 {
                integrator.step(&mut state, &SpringField, dt).ok();
            }

            let final_energy = state.total_energy();
            let drift = (final_energy - initial_energy).abs() / initial_energy.abs().max(0.001);

            prop_assert!(drift < 0.1, "Energy drift {} too large", drift);
        }

        /// Falsification: free particle moves at constant velocity.
        #[test]
        fn prop_free_particle_constant_velocity(
            x0 in -100.0f64..100.0,
            v0 in -10.0f64..10.0,
            mass in 0.1f64..10.0,
            steps in 10usize..100,
        ) {
            let mut state = SimState::new();
            state.add_body(mass, Vec3::new(x0, 0.0, 0.0), Vec3::new(v0, 0.0, 0.0));

            // Zero force field
            struct ZeroField;
            impl ForceField for ZeroField {
                fn acceleration(&self, _: &Vec3, _: f64) -> Vec3 { Vec3::zero() }
                fn potential(&self, _: &Vec3, _: f64) -> f64 { 0.0 }
            }

            let integrator = VerletIntegrator::new();
            let dt = 0.01;

            for _ in 0..steps {
                integrator.step(&mut state, &ZeroField, dt).ok();
            }

            // Velocity should be unchanged
            let final_vel = state.velocities()[0].x;
            prop_assert!((final_vel - v0).abs() < 1e-9,
                "Velocity changed from {} to {}", v0, final_vel);
        }
    }
}
