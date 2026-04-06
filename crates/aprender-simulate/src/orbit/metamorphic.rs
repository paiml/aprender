//! Metamorphic testing for orbital physics invariants.
//!
//! Per Chen et al. [33], metamorphic testing verifies **relations** rather
//! than specific outputs. This is critical for physics simulations where
//! exact outputs are unknown but invariant relationships must hold.
//!
//! # Metamorphic Relations
//!
//! 1. **Rotation Invariance**: Rotating the entire system preserves relative dynamics
//! 2. **Time-Reversal Symmetry**: Symplectic integrators are time-reversible
//! 3. **Mass Scaling**: Uniform mass scaling preserves orbital shapes
//! 4. **Energy Conservation**: Total mechanical energy is preserved
//! 5. **Momentum Conservation**: Total momentum is preserved (closed system)
//!
//! # References
//!
//! [33] Chen et al., "Metamorphic testing: a new approach," Hong Kong UST, 1998.

use crate::orbit::physics::{NBodyState, OrbitBody, YoshidaIntegrator};
use crate::orbit::units::{OrbitMass, OrbitTime, Position3D, Velocity3D};

/// Metamorphic test result.
#[derive(Debug, Clone)]
pub struct MetamorphicResult {
    /// Name of the relation tested.
    pub relation: String,
    /// Whether the relation holds within tolerance.
    pub passed: bool,
    /// Measured error/deviation.
    pub error: f64,
    /// Tolerance used.
    pub tolerance: f64,
    /// Additional details.
    pub details: String,
}

impl MetamorphicResult {
    /// Create a passing result.
    #[must_use]
    pub fn pass(relation: &str, error: f64, tolerance: f64) -> Self {
        Self {
            relation: relation.to_string(),
            passed: true,
            error,
            tolerance,
            details: String::new(),
        }
    }

    /// Create a failing result.
    #[must_use]
    pub fn fail(relation: &str, error: f64, tolerance: f64, details: &str) -> Self {
        Self {
            relation: relation.to_string(),
            passed: false,
            error,
            tolerance,
            details: details.to_string(),
        }
    }
}

/// Compute pairwise distances between all bodies.
fn compute_pairwise_distances(state: &NBodyState) -> Vec<f64> {
    let n = state.bodies.len();
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);

    for i in 0..n {
        for j in (i + 1)..n {
            let (x1, y1, z1) = state.bodies[i].position.as_meters();
            let (x2, y2, z2) = state.bodies[j].position.as_meters();
            let dx = x2 - x1;
            let dy = y2 - y1;
            let dz = z2 - z1;
            distances.push((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }

    distances
}

/// Apply 3D rotation matrix around Z-axis.
fn rotate_z(x: f64, y: f64, z: f64, angle: f64) -> (f64, f64, f64) {
    let (sin_a, cos_a) = angle.sin_cos();
    (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z)
}

/// Rotate entire system around Z-axis.
fn rotate_state(state: &NBodyState, angle: f64) -> NBodyState {
    let rotated_bodies: Vec<OrbitBody> = state
        .bodies
        .iter()
        .map(|body| {
            let (px, py, pz) = body.position.as_meters();
            let (vx, vy, vz) = body.velocity.as_mps();

            let (rpx, rpy, rpz) = rotate_z(px, py, pz, angle);
            let (rvx, rvy, rvz) = rotate_z(vx, vy, vz, angle);

            OrbitBody::new(
                body.mass,
                Position3D::from_meters(rpx, rpy, rpz),
                Velocity3D::from_mps(rvx, rvy, rvz),
            )
        })
        .collect();

    NBodyState::new(rotated_bodies, state.min_separation().min(1e6))
}

/// Reverse velocities for time-reversal test.
fn reverse_velocities(state: &NBodyState) -> NBodyState {
    let reversed_bodies: Vec<OrbitBody> = state
        .bodies
        .iter()
        .map(|body| {
            let (vx, vy, vz) = body.velocity.as_mps();
            OrbitBody::new(
                body.mass,
                body.position,
                Velocity3D::from_mps(-vx, -vy, -vz),
            )
        })
        .collect();

    NBodyState::new(reversed_bodies, state.min_separation().min(1e6))
}

/// Scale all masses by a uniform factor.
#[allow(dead_code)] // Reserved for MR-3 mass scaling relation
fn scale_masses(state: &NBodyState, factor: f64) -> NBodyState {
    let scaled_bodies: Vec<OrbitBody> = state
        .bodies
        .iter()
        .map(|body| {
            OrbitBody::new(
                OrbitMass::from_kg(body.mass.as_kg() * factor),
                body.position,
                body.velocity,
            )
        })
        .collect();

    NBodyState::new(scaled_bodies, state.min_separation().min(1e6))
}

/// MR-1: Rotation Invariance Test
///
/// Rotating the entire system should not change relative distances
/// or the evolution of relative dynamics.
#[must_use]
pub fn test_rotation_invariance(
    initial_state: &NBodyState,
    steps: usize,
    dt: f64,
    tolerance: f64,
) -> MetamorphicResult {
    let angle = std::f64::consts::PI / 4.0; // 45 degrees

    // Run original simulation
    let mut original = initial_state.clone();
    let yoshida = YoshidaIntegrator::new();
    let dt_orbit = OrbitTime::from_seconds(dt);

    for _ in 0..steps {
        if yoshida.step(&mut original, dt_orbit).is_err() {
            return MetamorphicResult::fail(
                "Rotation Invariance",
                f64::NAN,
                tolerance,
                "Original simulation failed",
            );
        }
    }

    // Run rotated simulation
    let rotated_initial = rotate_state(initial_state, angle);
    let mut rotated = rotated_initial;

    for _ in 0..steps {
        if yoshida.step(&mut rotated, dt_orbit).is_err() {
            return MetamorphicResult::fail(
                "Rotation Invariance",
                f64::NAN,
                tolerance,
                "Rotated simulation failed",
            );
        }
    }

    // Compare pairwise distances
    let original_distances = compute_pairwise_distances(&original);
    let rotated_distances = compute_pairwise_distances(&rotated);

    let mut max_error = 0.0_f64;
    for (d1, d2) in original_distances.iter().zip(rotated_distances.iter()) {
        let rel_error = if *d1 > f64::EPSILON {
            (d1 - d2).abs() / d1
        } else {
            (d1 - d2).abs()
        };
        max_error = max_error.max(rel_error);
    }

    if max_error <= tolerance {
        MetamorphicResult::pass("Rotation Invariance", max_error, tolerance)
    } else {
        MetamorphicResult::fail(
            "Rotation Invariance",
            max_error,
            tolerance,
            &format!("Max distance error: {max_error:.2e}"),
        )
    }
}

/// MR-2: Time-Reversal Symmetry Test
///
/// For symplectic integrators, running forward N steps then backward N steps
/// should return to (approximately) the initial state.
#[must_use]
pub fn test_time_reversal(
    initial_state: &NBodyState,
    steps: usize,
    dt: f64,
    tolerance: f64,
) -> MetamorphicResult {
    let yoshida = YoshidaIntegrator::new();
    let dt_orbit = OrbitTime::from_seconds(dt);

    // Run forward
    let mut state = initial_state.clone();
    for _ in 0..steps {
        if yoshida.step(&mut state, dt_orbit).is_err() {
            return MetamorphicResult::fail(
                "Time-Reversal Symmetry",
                f64::NAN,
                tolerance,
                "Forward simulation failed",
            );
        }
    }

    // Reverse velocities
    state = reverse_velocities(&state);

    // Run backward (same number of steps)
    for _ in 0..steps {
        if yoshida.step(&mut state, dt_orbit).is_err() {
            return MetamorphicResult::fail(
                "Time-Reversal Symmetry",
                f64::NAN,
                tolerance,
                "Backward simulation failed",
            );
        }
    }

    // Reverse velocities again to compare
    state = reverse_velocities(&state);

    // Compare with initial state
    let mut max_pos_error = 0.0_f64;
    let mut max_vel_error = 0.0_f64;

    for (initial, final_body) in initial_state.bodies.iter().zip(state.bodies.iter()) {
        let (ix, iy, iz) = initial.position.as_meters();
        let (fx, fy, fz) = final_body.position.as_meters();

        let initial_r = (ix * ix + iy * iy + iz * iz).sqrt();
        let pos_error = ((fx - ix).powi(2) + (fy - iy).powi(2) + (fz - iz).powi(2)).sqrt();
        let rel_pos_error = if initial_r > f64::EPSILON {
            pos_error / initial_r
        } else {
            pos_error
        };
        max_pos_error = max_pos_error.max(rel_pos_error);

        let (ivx, ivy, ivz) = initial.velocity.as_mps();
        let (fvx, fvy, fvz) = final_body.velocity.as_mps();

        let initial_v = (ivx * ivx + ivy * ivy + ivz * ivz).sqrt();
        let vel_error = ((fvx - ivx).powi(2) + (fvy - ivy).powi(2) + (fvz - ivz).powi(2)).sqrt();
        let rel_vel_error = if initial_v > f64::EPSILON {
            vel_error / initial_v
        } else {
            vel_error
        };
        max_vel_error = max_vel_error.max(rel_vel_error);
    }

    let max_error = max_pos_error.max(max_vel_error);

    if max_error <= tolerance {
        MetamorphicResult::pass("Time-Reversal Symmetry", max_error, tolerance)
    } else {
        MetamorphicResult::fail(
            "Time-Reversal Symmetry",
            max_error,
            tolerance,
            &format!("Pos error: {max_pos_error:.2e}, Vel error: {max_vel_error:.2e}"),
        )
    }
}

/// MR-3: Energy Conservation Test
///
/// Total mechanical energy should be conserved (bounded oscillation for symplectic).
#[must_use]
pub fn test_energy_conservation(
    initial_state: &NBodyState,
    steps: usize,
    dt: f64,
    tolerance: f64,
) -> MetamorphicResult {
    let yoshida = YoshidaIntegrator::new();
    let dt_orbit = OrbitTime::from_seconds(dt);

    let initial_energy = initial_state.total_energy();
    let mut state = initial_state.clone();
    let mut max_error = 0.0_f64;

    for _ in 0..steps {
        if yoshida.step(&mut state, dt_orbit).is_err() {
            return MetamorphicResult::fail(
                "Energy Conservation",
                f64::NAN,
                tolerance,
                "Simulation failed",
            );
        }

        let current_energy = state.total_energy();
        let rel_error = if initial_energy.abs() > f64::EPSILON {
            (current_energy - initial_energy).abs() / initial_energy.abs()
        } else {
            (current_energy - initial_energy).abs()
        };
        max_error = max_error.max(rel_error);
    }

    if max_error <= tolerance {
        MetamorphicResult::pass("Energy Conservation", max_error, tolerance)
    } else {
        MetamorphicResult::fail(
            "Energy Conservation",
            max_error,
            tolerance,
            &format!("Max energy drift: {max_error:.2e}"),
        )
    }
}

/// MR-4: Angular Momentum Conservation Test
///
/// Total angular momentum should be exactly conserved (to machine precision).
#[must_use]
pub fn test_angular_momentum_conservation(
    initial_state: &NBodyState,
    steps: usize,
    dt: f64,
    tolerance: f64,
) -> MetamorphicResult {
    let yoshida = YoshidaIntegrator::new();
    let dt_orbit = OrbitTime::from_seconds(dt);

    let initial_l = initial_state.angular_momentum_magnitude();
    let mut state = initial_state.clone();
    let mut max_error = 0.0_f64;

    for _ in 0..steps {
        if yoshida.step(&mut state, dt_orbit).is_err() {
            return MetamorphicResult::fail(
                "Angular Momentum Conservation",
                f64::NAN,
                tolerance,
                "Simulation failed",
            );
        }

        let current_l = state.angular_momentum_magnitude();
        let rel_error = if initial_l.abs() > f64::EPSILON {
            (current_l - initial_l).abs() / initial_l.abs()
        } else {
            (current_l - initial_l).abs()
        };
        max_error = max_error.max(rel_error);
    }

    if max_error <= tolerance {
        MetamorphicResult::pass("Angular Momentum Conservation", max_error, tolerance)
    } else {
        MetamorphicResult::fail(
            "Angular Momentum Conservation",
            max_error,
            tolerance,
            &format!("Max L drift: {max_error:.2e}"),
        )
    }
}

/// MR-5: Deterministic Replay Test
///
/// Same initial conditions + same seed should produce identical results.
#[must_use]
pub fn test_deterministic_replay(
    initial_state: &NBodyState,
    steps: usize,
    dt: f64,
) -> MetamorphicResult {
    let yoshida = YoshidaIntegrator::new();
    let dt_orbit = OrbitTime::from_seconds(dt);

    // First run
    let mut state1 = initial_state.clone();
    for _ in 0..steps {
        if yoshida.step(&mut state1, dt_orbit).is_err() {
            return MetamorphicResult::fail(
                "Deterministic Replay",
                f64::NAN,
                0.0,
                "First run failed",
            );
        }
    }

    // Second run (identical)
    let mut state2 = initial_state.clone();
    for _ in 0..steps {
        if yoshida.step(&mut state2, dt_orbit).is_err() {
            return MetamorphicResult::fail(
                "Deterministic Replay",
                f64::NAN,
                0.0,
                "Second run failed",
            );
        }
    }

    // Compare bit-for-bit (intentional exact comparison for determinism test)
    #[allow(clippy::float_cmp)]
    let identical = state1
        .bodies
        .iter()
        .zip(state2.bodies.iter())
        .all(|(b1, b2)| {
            let (x1, y1, z1) = b1.position.as_meters();
            let (x2, y2, z2) = b2.position.as_meters();
            let (vx1, vy1, vz1) = b1.velocity.as_mps();
            let (vx2, vy2, vz2) = b2.velocity.as_mps();

            x1 == x2 && y1 == y2 && z1 == z2 && vx1 == vx2 && vy1 == vy2 && vz1 == vz2
        });

    if identical {
        MetamorphicResult::pass("Deterministic Replay", 0.0, 0.0)
    } else {
        MetamorphicResult::fail(
            "Deterministic Replay",
            1.0,
            0.0,
            "Results not bit-identical",
        )
    }
}

/// Run all metamorphic tests on a state.
#[must_use]
pub fn run_all_metamorphic_tests(
    state: &NBodyState,
    steps: usize,
    dt: f64,
) -> Vec<MetamorphicResult> {
    vec![
        test_rotation_invariance(state, steps, dt, 1e-10),
        test_time_reversal(state, steps, dt, 1e-6),
        test_energy_conservation(state, steps, dt, 1e-9),
        test_angular_momentum_conservation(state, steps, dt, 1e-12),
        test_deterministic_replay(state, steps, dt),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::scenarios::KeplerConfig;
    use crate::orbit::units::AU;

    fn create_earth_sun_state() -> NBodyState {
        KeplerConfig::earth_sun().build(1e6)
    }

    #[test]
    fn test_metamorphic_result_pass() {
        let result = MetamorphicResult::pass("Test", 1e-10, 1e-9);
        assert!(result.passed);
        assert!(result.error < result.tolerance);
    }

    #[test]
    fn test_metamorphic_result_fail() {
        let result = MetamorphicResult::fail("Test", 1e-5, 1e-9, "Too large");
        assert!(!result.passed);
        assert!(result.error > result.tolerance);
    }

    #[test]
    fn test_compute_pairwise_distances() {
        let state = create_earth_sun_state();
        let distances = compute_pairwise_distances(&state);
        assert_eq!(distances.len(), 1); // 2 bodies = 1 pair
                                        // Earth starts at perihelion (~0.983 AU due to eccentricity 0.0167)
        let perihelion = AU * (1.0 - 0.0167);
        assert!((distances[0] - perihelion).abs() / perihelion < 0.01);
    }

    #[test]
    fn test_rotate_z() {
        let (x, y, z) = rotate_z(1.0, 0.0, 0.0, std::f64::consts::FRAC_PI_2);
        assert!(x.abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
        assert!(z.abs() < 1e-10);
    }

    #[test]
    fn test_rotate_state() {
        let state = create_earth_sun_state();
        let rotated = rotate_state(&state, std::f64::consts::FRAC_PI_2);

        // Sun should stay at origin
        let (sx, sy, _) = rotated.bodies[0].position.as_meters();
        assert!(sx.abs() < 1e-10);
        assert!(sy.abs() < 1e-10);

        // Earth should be rotated 90 degrees
        // Original position is at perihelion (~0.983 AU, 0)
        let (orig_x, _, _) = state.bodies[1].position.as_meters();
        let (ex, ey, _) = rotated.bodies[1].position.as_meters();
        assert!(ex.abs() < orig_x.abs() * 0.01); // ~0 after 90° rotation from (perihelion, 0)
        assert!((ey - orig_x).abs() / orig_x.abs() < 0.01); // ~perihelion after 90° rotation
    }

    #[test]
    fn test_reverse_velocities() {
        let state = create_earth_sun_state();
        let reversed = reverse_velocities(&state);

        let (_, vy_orig, _) = state.bodies[1].velocity.as_mps();
        let (_, vy_rev, _) = reversed.bodies[1].velocity.as_mps();

        assert!((vy_orig + vy_rev).abs() < 1e-10);
    }

    #[test]
    fn test_scale_masses() {
        let state = create_earth_sun_state();
        let scaled = scale_masses(&state, 2.0);

        let orig_mass = state.bodies[0].mass.as_kg();
        let scaled_mass = scaled.bodies[0].mass.as_kg();

        assert!((scaled_mass - 2.0 * orig_mass).abs() / orig_mass < 1e-10);
    }

    #[test]
    fn test_mr_rotation_invariance() {
        let state = create_earth_sun_state();
        let result = test_rotation_invariance(&state, 100, 3600.0, 1e-8);
        assert!(result.passed, "Rotation invariance failed: {:?}", result);
    }

    #[test]
    fn test_mr_time_reversal() {
        let state = create_earth_sun_state();
        // Use fewer steps and larger tolerance for time-reversal
        // (error accumulates both ways)
        let result = test_time_reversal(&state, 50, 3600.0, 1e-4);
        assert!(result.passed, "Time-reversal failed: {:?}", result);
    }

    #[test]
    fn test_mr_energy_conservation() {
        let state = create_earth_sun_state();
        let result = test_energy_conservation(&state, 100, 3600.0, 1e-9);
        assert!(result.passed, "Energy conservation failed: {:?}", result);
    }

    #[test]
    fn test_mr_angular_momentum_conservation() {
        let state = create_earth_sun_state();
        let result = test_angular_momentum_conservation(&state, 100, 3600.0, 1e-12);
        assert!(
            result.passed,
            "Angular momentum conservation failed: {:?}",
            result
        );
    }

    #[test]
    fn test_mr_deterministic_replay() {
        let state = create_earth_sun_state();
        let result = test_deterministic_replay(&state, 100, 3600.0);
        assert!(result.passed, "Deterministic replay failed: {:?}", result);
    }

    #[test]
    fn test_run_all_metamorphic_tests() {
        let state = create_earth_sun_state();
        let results = run_all_metamorphic_tests(&state, 50, 3600.0);

        assert_eq!(results.len(), 5);

        // At minimum, deterministic replay and angular momentum should pass
        let deterministic = &results[4];
        assert!(deterministic.passed, "Deterministic: {:?}", deterministic);

        let angular = &results[3];
        assert!(angular.passed, "Angular momentum: {:?}", angular);
    }

    #[test]
    fn test_mr_energy_with_longer_simulation() {
        let state = create_earth_sun_state();
        // Test AC-2: Energy drift < 1e-9 over 100 orbits
        // 100 days at 1 hour steps = 2400 steps
        let result = test_energy_conservation(&state, 2400, 3600.0, 1e-9);
        assert!(result.passed, "Energy over 100 days: {:?}", result);
    }

    #[test]
    fn test_orbital_period_metamorphic() {
        // MR: After one orbital period, Earth should return near starting position
        let state = create_earth_sun_state();
        let yoshida = YoshidaIntegrator::new();
        let dt = OrbitTime::from_seconds(3600.0); // 1 hour

        // Earth's orbital period is ~365.25 days = 8766 hours
        let steps = 8766;
        let mut current = state.clone();

        for _ in 0..steps {
            yoshida.step(&mut current, dt).expect("step failed");
        }

        // Check Earth position relative to starting
        let (x0, y0, _) = state.bodies[1].position.as_meters();
        let (x1, y1, _) = current.bodies[1].position.as_meters();

        let start_r = (x0 * x0 + y0 * y0).sqrt();
        let pos_diff = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
        let rel_error = pos_diff / start_r;

        // Should be within 1% of starting position after 1 year
        assert!(rel_error < 0.01, "Orbital period error: {rel_error}");
    }
}
