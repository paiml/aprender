//! Simulation state management.
//!
//! Implements the world state with:
//! - Entity positions and velocities
//! - Mass properties
//! - Energy computation
//! - Constraint tracking

use crate::error::SimResult;
use serde::{Deserialize, Serialize};

/// 3D vector for positions and velocities.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vec3 {
    /// Create a new vector.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Zero vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Magnitude squared.
    #[must_use]
    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Magnitude (length).
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Dot product.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Normalize to unit vector.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let mag = self.magnitude();
        if mag < f64::EPSILON {
            Self::zero()
        } else {
            Self {
                x: self.x / mag,
                y: self.y / mag,
                z: self.z / mag,
            }
        }
    }

    /// Scale by scalar.
    #[must_use]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Check if all components are finite.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // is_finite not const
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Simulation event for state updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimEvent {
    /// Add a body to the simulation.
    AddBody {
        /// Body mass.
        mass: f64,
        /// Initial position.
        position: Vec3,
        /// Initial velocity.
        velocity: Vec3,
    },
    /// Apply force to a body.
    ApplyForce {
        /// Body index.
        body_index: usize,
        /// Force vector.
        force: Vec3,
    },
    /// Set body position.
    SetPosition {
        /// Body index.
        body_index: usize,
        /// New position.
        position: Vec3,
    },
    /// Set body velocity.
    SetVelocity {
        /// Body index.
        body_index: usize,
        /// New velocity.
        velocity: Vec3,
    },
    /// Custom event.
    Custom {
        /// Event name.
        name: String,
        /// Serialized data.
        data: Vec<u8>,
    },
}

/// Simulation state.
///
/// Contains all state variables needed to reproduce the simulation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimState {
    /// Body masses.
    masses: Vec<f64>,
    /// Body positions.
    positions: Vec<Vec3>,
    /// Body velocities.
    velocities: Vec<Vec3>,
    /// Accumulated forces (cleared each step).
    forces: Vec<Vec3>,
    /// Active constraints and their violations.
    constraints: Vec<(String, f64)>,
    /// Potential energy (set by domain engine).
    potential_energy: f64,
}

impl SimState {
    /// Create a new empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get number of bodies.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Vec::len not const in older Rust
    pub fn num_bodies(&self) -> usize {
        self.masses.len()
    }

    /// Add a body to the simulation.
    pub fn add_body(&mut self, mass: f64, position: Vec3, velocity: Vec3) {
        self.masses.push(mass);
        self.positions.push(position);
        self.velocities.push(velocity);
        self.forces.push(Vec3::zero());
    }

    /// Get body masses.
    #[must_use]
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// Get body positions.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Get mutable body positions.
    #[must_use]
    pub fn positions_mut(&mut self) -> &mut [Vec3] {
        &mut self.positions
    }

    /// Get body velocities.
    #[must_use]
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }

    /// Get mutable body velocities.
    #[must_use]
    pub fn velocities_mut(&mut self) -> &mut [Vec3] {
        &mut self.velocities
    }

    /// Get body forces.
    #[must_use]
    pub fn forces(&self) -> &[Vec3] {
        &self.forces
    }

    /// Get mutable body forces.
    #[must_use]
    pub fn forces_mut(&mut self) -> &mut [Vec3] {
        &mut self.forces
    }

    /// Set position for a body.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    pub fn set_position(&mut self, index: usize, position: Vec3) {
        self.positions[index] = position;
    }

    /// Set velocity for a body.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    pub fn set_velocity(&mut self, index: usize, velocity: Vec3) {
        self.velocities[index] = velocity;
    }

    /// Apply force to a body.
    ///
    /// # Panics
    ///
    /// Panics if index is out of bounds.
    pub fn apply_force(&mut self, index: usize, force: Vec3) {
        self.forces[index] = self.forces[index] + force;
    }

    /// Clear all forces (called at start of each step).
    pub fn clear_forces(&mut self) {
        for f in &mut self.forces {
            *f = Vec3::zero();
        }
    }

    /// Set potential energy (called by domain engine).
    #[allow(clippy::missing_const_for_fn)] // Mutable const not stable
    pub fn set_potential_energy(&mut self, energy: f64) {
        self.potential_energy = energy;
    }

    /// Get total kinetic energy.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        self.masses
            .iter()
            .zip(&self.velocities)
            .map(|(m, v)| 0.5 * m * v.magnitude_squared())
            .sum()
    }

    /// Get potential energy.
    #[must_use]
    pub const fn potential_energy(&self) -> f64 {
        self.potential_energy
    }

    /// Get total energy (kinetic + potential).
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy
    }

    /// Add a constraint violation.
    pub fn add_constraint(&mut self, name: impl Into<String>, violation: f64) {
        self.constraints.push((name.into(), violation));
    }

    /// Clear all constraints.
    pub fn clear_constraints(&mut self) {
        self.constraints.clear();
    }

    /// Get constraint violations.
    pub fn constraint_violations(&self) -> impl Iterator<Item = (String, f64)> + '_ {
        self.constraints.iter().cloned()
    }

    /// Check if all state values are finite.
    #[must_use]
    pub fn all_finite(&self) -> bool {
        self.positions.iter().all(Vec3::is_finite)
            && self.velocities.iter().all(Vec3::is_finite)
            && self.masses.iter().all(|m| m.is_finite())
    }

    /// Apply an event to the state.
    ///
    /// # Errors
    ///
    /// Returns error if event cannot be applied.
    pub fn apply_event(&mut self, event: &SimEvent) -> SimResult<()> {
        match event {
            SimEvent::AddBody {
                mass,
                position,
                velocity,
            } => {
                self.add_body(*mass, *position, *velocity);
            }
            SimEvent::ApplyForce { body_index, force } => {
                if *body_index < self.num_bodies() {
                    self.apply_force(*body_index, *force);
                }
            }
            SimEvent::SetPosition {
                body_index,
                position,
            } => {
                if *body_index < self.num_bodies() {
                    self.set_position(*body_index, *position);
                }
            }
            SimEvent::SetVelocity {
                body_index,
                velocity,
            } => {
                if *body_index < self.num_bodies() {
                    self.set_velocity(*body_index, *velocity);
                }
            }
            SimEvent::Custom { .. } => {
                // Custom events are handled by domain-specific code
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_operations() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);

        // Addition
        let sum = v1 + v2;
        assert!((sum.x - 5.0).abs() < f64::EPSILON);
        assert!((sum.y - 7.0).abs() < f64::EPSILON);
        assert!((sum.z - 9.0).abs() < f64::EPSILON);

        // Subtraction
        let diff = v2 - v1;
        assert!((diff.x - 3.0).abs() < f64::EPSILON);

        // Dot product
        let dot = v1.dot(&v2);
        assert!((dot - 32.0).abs() < f64::EPSILON); // 1*4 + 2*5 + 3*6 = 32

        // Cross product
        let cross = v1.cross(&v2);
        assert!((cross.x - (-3.0)).abs() < f64::EPSILON);
        assert!((cross.y - 6.0).abs() < f64::EPSILON);
        assert!((cross.z - (-3.0)).abs() < f64::EPSILON);

        // Magnitude
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.magnitude() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();

        assert!((n.magnitude() - 1.0).abs() < f64::EPSILON);
        assert!((n.x - 0.6).abs() < f64::EPSILON);
        assert!((n.y - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_add_body() {
        let mut state = SimState::new();

        state.add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero());
        state.add_body(2.0, Vec3::new(0.0, 1.0, 0.0), Vec3::zero());

        assert_eq!(state.num_bodies(), 2);
        assert!((state.masses()[0] - 1.0).abs() < f64::EPSILON);
        assert!((state.masses()[1] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_energy() {
        let mut state = SimState::new();

        // Body with mass 2, velocity (3, 0, 0) -> KE = 0.5 * 2 * 9 = 9
        state.add_body(2.0, Vec3::zero(), Vec3::new(3.0, 0.0, 0.0));

        assert!((state.kinetic_energy() - 9.0).abs() < f64::EPSILON);

        state.set_potential_energy(-5.0);
        assert!((state.total_energy() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_apply_event() {
        let mut state = SimState::new();

        let event = SimEvent::AddBody {
            mass: 1.0,
            position: Vec3::new(1.0, 2.0, 3.0),
            velocity: Vec3::zero(),
        };

        state.apply_event(&event).ok();
        assert_eq!(state.num_bodies(), 1);
        assert!((state.positions()[0].x - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_constraints() {
        let mut state = SimState::new();

        state.add_constraint("test1", 0.001);
        state.add_constraint("test2", -0.002);

        let violations: Vec<_> = state.constraint_violations().collect();
        assert_eq!(violations.len(), 2);

        state.clear_constraints();
        let violations: Vec<_> = state.constraint_violations().collect();
        assert!(violations.is_empty());
    }

    #[test]
    fn test_state_all_finite() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::zero());

        assert!(state.all_finite());

        state.set_position(0, Vec3::new(f64::NAN, 0.0, 0.0));
        assert!(!state.all_finite());
    }

    #[test]
    fn test_vec3_scale() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let scaled = v.scale(2.0);
        assert!((scaled.x - 2.0).abs() < f64::EPSILON);
        assert!((scaled.y - 4.0).abs() < f64::EPSILON);
        assert!((scaled.z - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vec3_mul_scalar() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let scaled = v * 2.5;
        assert!((scaled.x - 2.5).abs() < f64::EPSILON);
        assert!((scaled.y - 5.0).abs() < f64::EPSILON);
        assert!((scaled.z - 7.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vec3_neg() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let neg = -v;
        assert!((neg.x - (-1.0)).abs() < f64::EPSILON);
        assert!((neg.y - 2.0).abs() < f64::EPSILON);
        assert!((neg.z - (-3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vec3_is_finite() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        assert!(v1.is_finite());

        let v2 = Vec3::new(f64::INFINITY, 0.0, 0.0);
        assert!(!v2.is_finite());

        let v3 = Vec3::new(0.0, f64::NEG_INFINITY, 0.0);
        assert!(!v3.is_finite());

        let v4 = Vec3::new(0.0, 0.0, f64::NAN);
        assert!(!v4.is_finite());
    }

    #[test]
    fn test_vec3_normalize_zero() {
        let v = Vec3::zero();
        let n = v.normalize();
        assert!((n.x).abs() < f64::EPSILON);
        assert!((n.y).abs() < f64::EPSILON);
        assert!((n.z).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vec3_default() {
        let v = Vec3::default();
        assert!((v.x).abs() < f64::EPSILON);
        assert!((v.y).abs() < f64::EPSILON);
        assert!((v.z).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vec3_partial_eq() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(1.0, 2.0, 3.0);
        let v3 = Vec3::new(1.0, 2.0, 4.0);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_vec3_magnitude_squared() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.magnitude_squared() - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_positions_mut() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::zero());

        {
            let positions = state.positions_mut();
            positions[0] = Vec3::new(10.0, 20.0, 30.0);
        }

        assert!((state.positions()[0].x - 10.0).abs() < f64::EPSILON);
        assert!((state.positions()[0].y - 20.0).abs() < f64::EPSILON);
        assert!((state.positions()[0].z - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_velocities_mut() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::new(1.0, 2.0, 3.0));

        {
            let velocities = state.velocities_mut();
            velocities[0] = Vec3::new(5.0, 6.0, 7.0);
        }

        assert!((state.velocities()[0].x - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_forces_mut() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::zero());

        {
            let forces = state.forces_mut();
            forces[0] = Vec3::new(100.0, 200.0, 300.0);
        }

        assert!((state.forces()[0].x - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_clear_forces() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::zero());
        state.apply_force(0, Vec3::new(100.0, 200.0, 300.0));

        assert!((state.forces()[0].x - 100.0).abs() < f64::EPSILON);

        state.clear_forces();

        assert!((state.forces()[0].x).abs() < f64::EPSILON);
        assert!((state.forces()[0].y).abs() < f64::EPSILON);
        assert!((state.forces()[0].z).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_apply_event_apply_force() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::zero());

        let event = SimEvent::ApplyForce {
            body_index: 0,
            force: Vec3::new(10.0, 20.0, 30.0),
        };

        state.apply_event(&event).unwrap();
        assert!((state.forces()[0].x - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_apply_event_apply_force_out_of_bounds() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::zero());

        let event = SimEvent::ApplyForce {
            body_index: 100, // out of bounds
            force: Vec3::new(10.0, 20.0, 30.0),
        };

        // Should not panic, just ignore
        state.apply_event(&event).unwrap();
    }

    #[test]
    fn test_state_apply_event_set_position() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::zero());

        let event = SimEvent::SetPosition {
            body_index: 0,
            position: Vec3::new(5.0, 6.0, 7.0),
        };

        state.apply_event(&event).unwrap();
        assert!((state.positions()[0].x - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_apply_event_set_position_out_of_bounds() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::zero());

        let event = SimEvent::SetPosition {
            body_index: 100, // out of bounds
            position: Vec3::new(5.0, 6.0, 7.0),
        };

        // Should not panic, just ignore
        state.apply_event(&event).unwrap();
    }

    #[test]
    fn test_state_apply_event_set_velocity() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::zero());

        let event = SimEvent::SetVelocity {
            body_index: 0,
            velocity: Vec3::new(8.0, 9.0, 10.0),
        };

        state.apply_event(&event).unwrap();
        assert!((state.velocities()[0].x - 8.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_apply_event_set_velocity_out_of_bounds() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::zero());

        let event = SimEvent::SetVelocity {
            body_index: 100, // out of bounds
            velocity: Vec3::new(8.0, 9.0, 10.0),
        };

        // Should not panic, just ignore
        state.apply_event(&event).unwrap();
    }

    #[test]
    fn test_state_apply_event_custom() {
        let mut state = SimState::new();

        let event = SimEvent::Custom {
            name: "custom_event".to_string(),
            data: vec![1, 2, 3, 4],
        };

        // Custom events are no-ops at this level
        state.apply_event(&event).unwrap();
    }

    #[test]
    fn test_sim_event_clone() {
        let event = SimEvent::AddBody {
            mass: 1.0,
            position: Vec3::new(1.0, 2.0, 3.0),
            velocity: Vec3::zero(),
        };
        let cloned = event.clone();
        match cloned {
            SimEvent::AddBody { mass, .. } => assert!((mass - 1.0).abs() < f64::EPSILON),
            _ => panic!("unexpected event type"),
        }
    }

    #[test]
    fn test_sim_event_debug() {
        let event = SimEvent::AddBody {
            mass: 1.0,
            position: Vec3::zero(),
            velocity: Vec3::zero(),
        };
        let debug = format!("{:?}", event);
        assert!(debug.contains("AddBody"));
    }

    #[test]
    fn test_state_clone() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::zero());
        state.set_potential_energy(-10.0);
        state.add_constraint("test", 0.5);

        let cloned = state.clone();
        assert_eq!(cloned.num_bodies(), 1);
        assert!((cloned.potential_energy() - (-10.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_state_debug() {
        let state = SimState::new();
        let debug = format!("{:?}", state);
        assert!(debug.contains("SimState"));
    }

    #[test]
    fn test_state_all_finite_with_infinity_in_velocity() {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::zero(), Vec3::new(f64::INFINITY, 0.0, 0.0));
        assert!(!state.all_finite());
    }

    #[test]
    fn test_state_all_finite_with_nan_in_mass() {
        let mut state = SimState::new();
        state.add_body(f64::NAN, Vec3::zero(), Vec3::zero());
        assert!(!state.all_finite());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Falsification: dot product is commutative.
        #[test]
        fn prop_dot_commutative(
            x1 in -1e6f64..1e6, y1 in -1e6f64..1e6, z1 in -1e6f64..1e6,
            x2 in -1e6f64..1e6, y2 in -1e6f64..1e6, z2 in -1e6f64..1e6,
        ) {
            let v1 = Vec3::new(x1, y1, z1);
            let v2 = Vec3::new(x2, y2, z2);

            let d1 = v1.dot(&v2);
            let d2 = v2.dot(&v1);

            prop_assert!((d1 - d2).abs() < 1e-9 * d1.abs().max(1.0));
        }

        /// Falsification: normalized vectors have unit length.
        #[test]
        fn prop_normalize_unit_length(
            x in -1e6f64..1e6, y in -1e6f64..1e6, z in -1e6f64..1e6,
        ) {
            let v = Vec3::new(x, y, z);

            // Skip zero vectors
            if v.magnitude() < f64::EPSILON {
                return Ok(());
            }

            let n = v.normalize();
            prop_assert!((n.magnitude() - 1.0).abs() < 1e-9);
        }

        /// Falsification: kinetic energy is non-negative.
        #[test]
        fn prop_kinetic_energy_nonnegative(
            mass in 0.1f64..1e6,
            vx in -1e3f64..1e3, vy in -1e3f64..1e3, vz in -1e3f64..1e3,
        ) {
            let mut state = SimState::new();
            state.add_body(mass, Vec3::zero(), Vec3::new(vx, vy, vz));

            prop_assert!(state.kinetic_energy() >= 0.0);
        }
    }
}
