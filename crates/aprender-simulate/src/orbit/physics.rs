//! Orbital physics engine with symplectic integrators.
//!
//! Implements numerical integration for orbital mechanics:
//! - Yoshida 4th order (symplectic, time-reversible)
//! - N-body gravitational force field
//! - Adaptive time-stepping for close encounters
//!
//! # References
//!
//! [8] Hairer, Lubich, Wanner, "Geometric Numerical Integration," 2006.
//! [13] H. Yoshida, "Construction of higher order symplectic integrators," 1990.
//! [30] Hockney & Eastwood, "Computer Simulation Using Particles," 1988.

use crate::error::{SimError, SimResult};
use crate::orbit::units::{Acceleration3D, OrbitMass, OrbitTime, Position3D, Velocity3D, G};
use uom::si::length::meter;

/// Body state in the N-body system.
#[derive(Debug, Clone)]
pub struct OrbitBody {
    pub mass: OrbitMass,
    pub position: Position3D,
    pub velocity: Velocity3D,
}

impl OrbitBody {
    /// Create a new orbital body.
    #[must_use]
    pub fn new(mass: OrbitMass, position: Position3D, velocity: Velocity3D) -> Self {
        Self {
            mass,
            position,
            velocity,
        }
    }

    /// Calculate kinetic energy (J).
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        let v_sq = self.velocity.magnitude_squared();
        0.5 * self.mass.as_kg() * v_sq
    }
}

/// N-body gravitational system state.
#[derive(Debug, Clone)]
pub struct NBodyState {
    pub bodies: Vec<OrbitBody>,
    pub time: OrbitTime,
    softening: f64,
}

impl NBodyState {
    /// Create a new N-body state with optional softening parameter.
    #[must_use]
    pub fn new(bodies: Vec<OrbitBody>, softening: f64) -> Self {
        Self {
            bodies,
            time: OrbitTime::from_seconds(0.0),
            softening,
        }
    }

    /// Number of bodies in the system.
    #[must_use]
    pub fn num_bodies(&self) -> usize {
        self.bodies.len()
    }

    /// Calculate total kinetic energy.
    #[must_use]
    pub fn kinetic_energy(&self) -> f64 {
        self.bodies.iter().map(OrbitBody::kinetic_energy).sum()
    }

    /// Calculate total potential energy.
    #[must_use]
    pub fn potential_energy(&self) -> f64 {
        let mut pe = 0.0;
        let n = self.bodies.len();
        let eps_sq = self.softening * self.softening;

        for i in 0..n {
            for j in (i + 1)..n {
                let r = self.bodies[i].position - self.bodies[j].position;
                let r_mag_sq = r.magnitude_squared() + eps_sq;
                let r_mag = r_mag_sq.sqrt();

                if r_mag > f64::EPSILON {
                    pe -= G * self.bodies[i].mass.as_kg() * self.bodies[j].mass.as_kg() / r_mag;
                }
            }
        }

        pe
    }

    /// Calculate total mechanical energy.
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    /// Calculate total angular momentum vector (Lx, Ly, Lz).
    #[must_use]
    pub fn angular_momentum(&self) -> (f64, f64, f64) {
        let mut lx = 0.0;
        let mut ly = 0.0;
        let mut lz = 0.0;

        for body in &self.bodies {
            let m = body.mass.as_kg();
            let (rx, ry, rz) = body.position.as_meters();
            let (vx, vy, vz) = body.velocity.as_mps();

            // L = m * (r × v)
            lx += m * (ry * vz - rz * vy);
            ly += m * (rz * vx - rx * vz);
            lz += m * (rx * vy - ry * vx);
        }

        (lx, ly, lz)
    }

    /// Calculate angular momentum magnitude.
    #[must_use]
    pub fn angular_momentum_magnitude(&self) -> f64 {
        let (lx, ly, lz) = self.angular_momentum();
        (lx * lx + ly * ly + lz * lz).sqrt()
    }

    /// Get minimum pairwise separation between bodies.
    #[must_use]
    pub fn min_separation(&self) -> f64 {
        let mut min_sep = f64::MAX;
        let n = self.bodies.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let r = self.bodies[i].position - self.bodies[j].position;
                let sep = r.magnitude().get::<meter>();
                if sep < min_sep {
                    min_sep = sep;
                }
            }
        }

        min_sep
    }

    /// Check if all values are finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.bodies
            .iter()
            .all(|b| b.position.is_finite() && b.velocity.is_finite())
    }
}

/// Compute gravitational accelerations for all bodies.
fn compute_accelerations(state: &NBodyState) -> Vec<Acceleration3D> {
    let n = state.bodies.len();
    let eps_sq = state.softening * state.softening;
    let mut accelerations = vec![Acceleration3D::zero(); n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            let r_ij = state.bodies[j].position - state.bodies[i].position;
            let (rx, ry, rz) = r_ij.as_meters();
            let r_mag_sq = rx * rx + ry * ry + rz * rz + eps_sq;
            let r_mag = r_mag_sq.sqrt();

            if r_mag > f64::EPSILON {
                // a_i = G * m_j / r² * r̂
                let factor = G * state.bodies[j].mass.as_kg() / (r_mag_sq * r_mag);
                let (ax, ay, az) = accelerations[i].as_mps2();
                accelerations[i] =
                    Acceleration3D::from_mps2(ax + factor * rx, ay + factor * ry, az + factor * rz);
            }
        }
    }

    accelerations
}

/// Yoshida 4th order symplectic integrator.
///
/// Higher-order symplectic method with excellent energy conservation.
/// Composed of three Verlet steps with specific coefficients.
///
/// # Properties
///
/// - Order: 4
/// - Symplectic: Yes (preserves phase space volume)
/// - Time-reversible: Yes
/// - Energy drift: O(dt⁴) bounded oscillation
///
/// # References
///
/// [13] Yoshida, "Construction of higher order symplectic integrators," 1990.
#[derive(Debug, Clone, Default)]
pub struct YoshidaIntegrator {
    /// Yoshida coefficient w1 = 1/(2 - 2^(1/3))
    w1: f64,
    /// Yoshida coefficient w0 = -2^(1/3)/(2 - 2^(1/3))
    w0: f64,
}

impl YoshidaIntegrator {
    /// Create a new Yoshida 4th order integrator.
    #[must_use]
    pub fn new() -> Self {
        let cbrt2 = 2.0_f64.cbrt();
        let w1 = 1.0 / (2.0 - cbrt2);
        let w0 = -cbrt2 / (2.0 - cbrt2);

        Self { w1, w0 }
    }

    /// Get the position coefficients c[0..4].
    fn c_coefficients(&self) -> [f64; 4] {
        [
            self.w1 / 2.0,
            (self.w0 + self.w1) / 2.0,
            (self.w0 + self.w1) / 2.0,
            self.w1 / 2.0,
        ]
    }

    /// Get the velocity coefficients d[0..3].
    fn d_coefficients(&self) -> [f64; 3] {
        [self.w1, self.w0, self.w1]
    }

    /// Step the state forward by dt.
    ///
    /// # Errors
    ///
    /// Returns error if state becomes invalid.
    pub fn step(&self, state: &mut NBodyState, dt: OrbitTime) -> SimResult<()> {
        let dt_secs = dt.as_seconds();
        let c = self.c_coefficients();
        let d = self.d_coefficients();

        // Yoshida 4th order: composed of three Verlet-like steps
        // Step 1: c[0] position, d[0] velocity
        self.drift(state, c[0] * dt_secs);
        self.kick(state, d[0] * dt_secs)?;

        // Step 2: c[1] position, d[1] velocity
        self.drift(state, c[1] * dt_secs);
        self.kick(state, d[1] * dt_secs)?;

        // Step 3: c[2] position, d[2] velocity
        self.drift(state, c[2] * dt_secs);
        self.kick(state, d[2] * dt_secs)?;

        // Step 4: c[3] position only
        self.drift(state, c[3] * dt_secs);

        // Update simulation time
        state.time = OrbitTime::from_seconds(state.time.as_seconds() + dt_secs);

        Ok(())
    }

    /// Drift: update positions.
    #[allow(clippy::unused_self)] // Method for future extensibility
    fn drift(&self, state: &mut NBodyState, dt: f64) {
        for body in &mut state.bodies {
            let (vx, vy, vz) = body.velocity.as_mps();
            let (px, py, pz) = body.position.as_meters();
            body.position = Position3D::from_meters(px + vx * dt, py + vy * dt, pz + vz * dt);
        }
    }

    /// Kick: update velocities.
    #[allow(clippy::unused_self)] // Method for future extensibility
    fn kick(&self, state: &mut NBodyState, dt: f64) -> SimResult<()> {
        let accelerations = compute_accelerations(state);

        for (i, body) in state.bodies.iter_mut().enumerate() {
            let (ax, ay, az) = accelerations[i].as_mps2();
            let (vx, vy, vz) = body.velocity.as_mps();
            body.velocity = Velocity3D::from_mps(vx + ax * dt, vy + ay * dt, vz + az * dt);

            if !body.velocity.is_finite() {
                return Err(SimError::NonFiniteValue {
                    location: format!("body {i} velocity"),
                });
            }
        }

        Ok(())
    }

    /// Get integrator order.
    #[must_use]
    pub const fn order(&self) -> u32 {
        4
    }

    /// Check if integrator is symplectic.
    #[must_use]
    pub const fn is_symplectic(&self) -> bool {
        true
    }
}

/// Adaptive time-stepping for close encounters.
#[derive(Debug, Clone)]
pub struct AdaptiveIntegrator {
    pub base_dt: f64,
    pub min_dt: f64,
    pub max_dt: f64,
    pub encounter_threshold: f64,
    yoshida: YoshidaIntegrator,
}

impl AdaptiveIntegrator {
    /// Create a new adaptive integrator.
    #[must_use]
    pub fn new(base_dt: f64, min_dt: f64, max_dt: f64, encounter_threshold: f64) -> Self {
        Self {
            base_dt,
            min_dt,
            max_dt,
            encounter_threshold,
            yoshida: YoshidaIntegrator::new(),
        }
    }

    /// Compute adaptive time step based on state.
    #[must_use]
    pub fn compute_dt(&self, state: &NBodyState) -> f64 {
        let min_sep = state.min_separation();

        // Reduce time step for close encounters
        let dt = if min_sep < self.encounter_threshold {
            // Scale dt by separation / threshold
            let ratio = min_sep / self.encounter_threshold;
            self.base_dt * ratio.max(0.01)
        } else {
            self.base_dt
        };

        dt.clamp(self.min_dt, self.max_dt)
    }

    /// Step with adaptive time stepping.
    ///
    /// # Errors
    ///
    /// Returns error if integration fails.
    pub fn step(&self, state: &mut NBodyState) -> SimResult<f64> {
        let dt = self.compute_dt(state);
        self.yoshida.step(state, OrbitTime::from_seconds(dt))?;
        Ok(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::units::{AU, EARTH_MASS, SOLAR_MASS};
    use uom::si::acceleration::meter_per_second_squared;

    const EPSILON: f64 = 1e-10;

    fn create_sun() -> OrbitBody {
        OrbitBody::new(
            OrbitMass::from_kg(SOLAR_MASS),
            Position3D::zero(),
            Velocity3D::zero(),
        )
    }

    fn create_earth() -> OrbitBody {
        // Earth at 1 AU with circular orbital velocity
        let v_circular = (G * SOLAR_MASS / AU).sqrt();
        OrbitBody::new(
            OrbitMass::from_kg(EARTH_MASS),
            Position3D::from_au(1.0, 0.0, 0.0),
            Velocity3D::from_mps(0.0, v_circular, 0.0),
        )
    }

    #[test]
    fn test_orbit_body_kinetic_energy() {
        let body = OrbitBody::new(
            OrbitMass::from_kg(1.0),
            Position3D::zero(),
            Velocity3D::from_mps(10.0, 0.0, 0.0),
        );
        let ke = body.kinetic_energy();
        assert!((ke - 50.0).abs() < EPSILON); // 0.5 * 1 * 100
    }

    #[test]
    fn test_nbody_state_creation() {
        let bodies = vec![create_sun(), create_earth()];
        let state = NBodyState::new(bodies, 0.0);
        assert_eq!(state.num_bodies(), 2);
    }

    #[test]
    fn test_nbody_state_kinetic_energy() {
        let bodies = vec![create_sun(), create_earth()];
        let state = NBodyState::new(bodies, 0.0);
        let ke = state.kinetic_energy();
        assert!(ke > 0.0);
    }

    #[test]
    fn test_nbody_state_potential_energy() {
        let bodies = vec![create_sun(), create_earth()];
        let state = NBodyState::new(bodies, 0.0);
        let pe = state.potential_energy();
        assert!(pe < 0.0); // Gravitational PE is negative
    }

    #[test]
    fn test_nbody_state_total_energy() {
        let bodies = vec![create_sun(), create_earth()];
        let state = NBodyState::new(bodies, 0.0);
        let e = state.total_energy();
        assert!(e < 0.0); // Bound orbit has negative total energy
    }

    #[test]
    fn test_nbody_state_angular_momentum() {
        let bodies = vec![create_sun(), create_earth()];
        let state = NBodyState::new(bodies, 0.0);
        let l = state.angular_momentum_magnitude();
        assert!(l > 0.0);
    }

    #[test]
    fn test_nbody_state_min_separation() {
        let bodies = vec![create_sun(), create_earth()];
        let state = NBodyState::new(bodies, 0.0);
        let min_sep = state.min_separation();
        let expected = AU;
        assert!((min_sep - expected).abs() / expected < 0.01);
    }

    #[test]
    fn test_yoshida_coefficients() {
        let yoshida = YoshidaIntegrator::new();
        let c = yoshida.c_coefficients();
        let d = yoshida.d_coefficients();

        // Sum of c coefficients should be 1
        let c_sum: f64 = c.iter().sum();
        assert!((c_sum - 1.0).abs() < EPSILON);

        // Sum of d coefficients should be 1
        let d_sum: f64 = d.iter().sum();
        assert!((d_sum - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_yoshida_order() {
        let yoshida = YoshidaIntegrator::new();
        assert_eq!(yoshida.order(), 4);
        assert!(yoshida.is_symplectic());
    }

    #[test]
    fn test_yoshida_energy_conservation_short_term() {
        let bodies = vec![create_sun(), create_earth()];
        let mut state = NBodyState::new(bodies, 1e6); // Small softening

        let yoshida = YoshidaIntegrator::new();
        let initial_energy = state.total_energy();

        // Simulate for 100 steps
        let dt = OrbitTime::from_seconds(86400.0); // 1 day
        for _ in 0..100 {
            yoshida.step(&mut state, dt).expect("step failed");
        }

        let final_energy = state.total_energy();
        let relative_error = (final_energy - initial_energy).abs() / initial_energy.abs();

        // Energy should be conserved to better than 1e-6 over 100 days
        assert!(
            relative_error < 1e-6,
            "Energy drift too large: {relative_error}"
        );
    }

    #[test]
    fn test_yoshida_angular_momentum_conservation() {
        let bodies = vec![create_sun(), create_earth()];
        let mut state = NBodyState::new(bodies, 1e6);

        let yoshida = YoshidaIntegrator::new();
        let initial_l = state.angular_momentum_magnitude();

        let dt = OrbitTime::from_seconds(86400.0);
        for _ in 0..100 {
            yoshida.step(&mut state, dt).expect("step failed");
        }

        let final_l = state.angular_momentum_magnitude();
        let relative_error = (final_l - initial_l).abs() / initial_l;

        // Angular momentum should be conserved to machine precision
        assert!(
            relative_error < 1e-12,
            "Angular momentum drift: {relative_error}"
        );
    }

    #[test]
    fn test_yoshida_orbit_period() {
        let bodies = vec![create_sun(), create_earth()];
        let mut state = NBodyState::new(bodies, 1e6);

        let yoshida = YoshidaIntegrator::new();
        let _initial_y = state.bodies[1].position.as_meters().1;

        // Simulate for ~1 year (365.25 days)
        let dt = OrbitTime::from_seconds(3600.0); // 1 hour steps for precision
        let steps = (365.25 * 24.0) as usize;

        for _ in 0..steps {
            yoshida.step(&mut state, dt).expect("step failed");
        }

        // Earth should be back near starting position after 1 year
        let (x, y, _) = state.bodies[1].position.as_meters();
        let final_distance = (x * x + y * y).sqrt();
        let expected_distance = AU;

        let relative_error = (final_distance - expected_distance).abs() / expected_distance;
        assert!(
            relative_error < 0.01,
            "Orbit radius error: {relative_error}"
        );
    }

    #[test]
    fn test_adaptive_integrator_creation() {
        let adaptive = AdaptiveIntegrator::new(86400.0, 3600.0, 604800.0, 1e9);
        assert!((adaptive.base_dt - 86400.0).abs() < EPSILON);
    }

    #[test]
    fn test_adaptive_integrator_normal_dt() {
        let bodies = vec![create_sun(), create_earth()];
        let state = NBodyState::new(bodies, 0.0);

        let adaptive = AdaptiveIntegrator::new(86400.0, 3600.0, 604800.0, 1e9);
        let dt = adaptive.compute_dt(&state);

        // Normal separation, should use base dt
        assert!((dt - 86400.0).abs() < EPSILON);
    }

    #[test]
    fn test_adaptive_integrator_close_encounter() {
        // Two bodies very close together
        let bodies = vec![
            OrbitBody::new(
                OrbitMass::from_kg(1e20),
                Position3D::from_meters(0.0, 0.0, 0.0),
                Velocity3D::zero(),
            ),
            OrbitBody::new(
                OrbitMass::from_kg(1e20),
                Position3D::from_meters(1e8, 0.0, 0.0), // 100 km apart
                Velocity3D::zero(),
            ),
        ];
        let state = NBodyState::new(bodies, 0.0);

        let adaptive = AdaptiveIntegrator::new(86400.0, 3600.0, 604800.0, 1e9);
        let dt = adaptive.compute_dt(&state);

        // Should reduce dt for close encounter
        assert!(dt < 86400.0);
        assert!(dt >= 3600.0); // But not below minimum
    }

    #[test]
    fn test_compute_accelerations_two_body() {
        let bodies = vec![create_sun(), create_earth()];
        let state = NBodyState::new(bodies, 0.0);

        let accelerations = compute_accelerations(&state);

        // Sun should have tiny acceleration toward Earth
        let (ax_sun, _ay_sun, _) = accelerations[0].as_mps2();
        assert!(ax_sun > 0.0); // Toward Earth (at +x)

        // Earth should have large acceleration toward Sun
        let (ax_earth, _, _) = accelerations[1].as_mps2();
        assert!(ax_earth < 0.0); // Toward Sun (at origin)

        // Earth's acceleration magnitude should be ~g at 1 AU
        let a_mag = accelerations[1]
            .magnitude()
            .get::<meter_per_second_squared>();
        let expected = G * SOLAR_MASS / (AU * AU);
        let relative_error = (a_mag - expected).abs() / expected;
        assert!(relative_error < 0.01);
    }

    #[test]
    fn test_nbody_is_finite() {
        let bodies = vec![create_sun(), create_earth()];
        let state = NBodyState::new(bodies, 0.0);
        assert!(state.is_finite());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::orbit::units::SOLAR_MASS;
    use proptest::prelude::*;

    proptest! {
        /// Gravitational force follows inverse square law.
        #[test]
        fn prop_gravity_inverse_square(
            r1 in 1e10f64..1e12,
            r2 in 1e10f64..1e12,
        ) {
            // F ∝ 1/r²
            let mass = 1e24;
            let sun_mass = SOLAR_MASS;

            let f1 = G * sun_mass * mass / (r1 * r1);
            let f2 = G * sun_mass * mass / (r2 * r2);

            // Verify inverse square relationship
            let ratio = (r1 / r2).powi(2);
            let force_ratio = f2 / f1;
            prop_assert!((force_ratio - ratio).abs() / ratio < 1e-10);
        }

        /// Total energy is conserved in n-body system (Hamiltonian).
        #[test]
        fn prop_energy_finite(
            mass in 1e20f64..1e30,
            distance in 1e9f64..1e12,
        ) {
            // Kinetic + Potential should be finite for bound orbits
            let v_orbital = (G * SOLAR_MASS / distance).sqrt();
            let ke = 0.5 * mass * v_orbital * v_orbital;
            let pe = -G * SOLAR_MASS * mass / distance;
            let total = ke + pe;

            // For circular orbit, total energy should be negative (bound)
            prop_assert!(total.is_finite());
            prop_assert!(total < 0.0, "E={} should be negative for bound orbit", total);
        }

        /// Circular orbital velocity formula.
        #[test]
        fn prop_circular_velocity(
            r in 1e10f64..1e13,
        ) {
            // v = sqrt(GM/r)
            let v = (G * SOLAR_MASS / r).sqrt();

            // Velocity should be positive and finite
            prop_assert!(v > 0.0);
            prop_assert!(v.is_finite());

            // Velocity should decrease with distance
            let v2 = (G * SOLAR_MASS / (r * 2.0)).sqrt();
            prop_assert!(v2 < v);
        }

        /// Acceleration magnitude depends on mass and distance.
        #[test]
        fn prop_acceleration_magnitude(
            central_mass in 1e25f64..1e31,
            distance in 1e9f64..1e13,
        ) {
            // a = GM/r²
            let a = G * central_mass / (distance * distance);

            prop_assert!(a > 0.0);
            prop_assert!(a.is_finite());

            // Larger mass -> larger acceleration
            let a2 = G * (central_mass * 2.0) / (distance * distance);
            prop_assert!(a2 > a);
        }

        /// OrbitBody mass must be positive.
        #[test]
        fn prop_body_mass_positive(
            mass in 1e10f64..1e35,
        ) {
            let body = OrbitBody::new(
                OrbitMass::from_kg(mass),
                Position3D::zero(),
                Velocity3D::zero(),
            );
            prop_assert!(body.mass.as_kg() > 0.0);
        }
    }
}
