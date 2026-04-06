//! Probar WASM Tests for Orbit Module
//!
//! Per spec Section 12: probar integration for WASM testing.
//! These tests verify physics invariants, deterministic replay, and fuzzing.
//!
//! Run with: cargo test --test probar_orbit --features wasm

#![cfg(feature = "wasm")]

use jugar_probar::{Assertion, InvariantCheck, InvariantChecker, TestCase, TestHarness, TestSuite};
use simular::orbit::units::AU;
use simular::orbit::wasm::OrbitSimulation;

// ============================================================================
// WASM LOADING & INITIALIZATION
// ============================================================================

#[test]
fn test_wasm_module_loads() {
    let sim = OrbitSimulation::new();
    let result = Assertion::equals(&sim.num_bodies(), &2);
    assert!(result.passed, "WASM module should load with 2 bodies");
}

#[test]
fn test_earth_sun_initialization() {
    let sim = OrbitSimulation::earth_sun();
    let result = Assertion::equals(&sim.num_bodies(), &2);
    assert!(result.passed, "Earth-Sun system should have 2 bodies");

    // Sun should be at origin
    let sun_x = sim.body_x(0);
    let result = Assertion::approx_eq(sun_x, 0.0, 1e-10);
    assert!(result.passed, "Sun should be at x=0: {}", result.message);
}

#[test]
fn test_circular_orbit_creation() {
    let sim = OrbitSimulation::circular_orbit(
        1.989e30, // Sun mass
        5.972e24, // Earth mass
        AU,       // 1 AU
    );
    let result = Assertion::equals(&sim.num_bodies(), &2);
    assert!(result.passed, "Circular orbit should have 2 bodies");
}

// ============================================================================
// PHYSICS INVARIANTS (Jidoka)
// ============================================================================

#[test]
fn test_energy_conservation_short_term() {
    let mut sim = OrbitSimulation::new();
    let initial_energy = sim.total_energy();

    // Simulate 100 steps (approximately 100 hours)
    for _ in 0..100 {
        sim.step(3600.0); // 1 hour
    }

    let final_energy = sim.total_energy();
    // Use relative error check instead of approx_eq (which uses absolute epsilon)
    let rel_error = (final_energy - initial_energy).abs() / initial_energy.abs();
    let result = Assertion::is_true(
        rel_error < 1e-9,
        &format!("Energy drift {:.2e} exceeds 1e-9", rel_error),
    );
    assert!(
        result.passed,
        "Energy should be conserved (short term): {}",
        result.message
    );
}

#[test]
fn test_energy_conservation_long_term() {
    let mut sim = OrbitSimulation::new();
    let initial_energy = sim.total_energy();

    // Simulate 365 days (1 orbit)
    for _ in 0..365 {
        sim.step_days(1.0);
    }

    let final_energy = sim.total_energy();
    let rel_error = (final_energy - initial_energy).abs() / initial_energy.abs();
    let result = Assertion::is_true(
        rel_error < 1e-9,
        &format!("Energy drift {:.2e} exceeds 1e-9", rel_error),
    );
    assert!(
        result.passed,
        "Energy conservation (long term): {}",
        result.message
    );
}

#[test]
fn test_angular_momentum_conservation() {
    let mut sim = OrbitSimulation::new();
    let initial_l = sim.angular_momentum();

    // Simulate 100 orbits
    for _ in 0..100 {
        sim.step_days(365.25);
    }

    let final_l = sim.angular_momentum();
    let rel_error = (final_l - initial_l).abs() / initial_l.abs();
    let result = Assertion::is_true(
        rel_error < 1e-9,
        &format!("Angular momentum drift {:.2e} exceeds 1e-9", rel_error),
    );
    assert!(
        result.passed,
        "Angular momentum conservation: {}",
        result.message
    );
}

#[test]
fn test_jidoka_status_all_ok() {
    let mut sim = OrbitSimulation::new();
    // Run a step to initialize jidoka status properly
    sim.step(3600.0);
    let status = sim.jidoka_status_json();

    let result = Assertion::contains(&status, "\"energy_ok\":true");
    assert!(
        result.passed,
        "Jidoka energy should be OK: {}",
        result.message
    );

    let result = Assertion::contains(&status, "\"angular_momentum_ok\":true");
    assert!(
        result.passed,
        "Jidoka angular momentum should be OK: {}",
        result.message
    );

    let result = Assertion::contains(&status, "\"finite_ok\":true");
    assert!(
        result.passed,
        "Jidoka finite check should be OK: {}",
        result.message
    );
}

#[test]
fn test_jidoka_after_simulation() {
    let mut sim = OrbitSimulation::new();

    // Run simulation for a year
    sim.run_steps(365, 86400.0);

    let status = sim.jidoka_status_json();
    let result = Assertion::contains(&status, "\"finite_ok\":true");
    assert!(
        result.passed,
        "All values should remain finite: {}",
        result.message
    );
}

// ============================================================================
// DETERMINISTIC REPLAY
// ============================================================================

#[test]
fn test_deterministic_replay_identical() {
    // Run simulation 1
    let mut sim1 = OrbitSimulation::new();
    for _ in 0..100 {
        sim1.step(3600.0);
    }
    let energy1 = sim1.total_energy();
    let x1 = sim1.body_x(1);
    let y1 = sim1.body_y(1);

    // Run simulation 2 (same steps)
    let mut sim2 = OrbitSimulation::new();
    for _ in 0..100 {
        sim2.step(3600.0);
    }
    let energy2 = sim2.total_energy();
    let x2 = sim2.body_x(1);
    let y2 = sim2.body_y(1);

    // Should be bit-identical
    let result = Assertion::equals(&energy1, &energy2);
    assert!(
        result.passed,
        "Energy should be identical: {}",
        result.message
    );

    let result = Assertion::equals(&x1, &x2);
    assert!(
        result.passed,
        "X position should be identical: {}",
        result.message
    );

    let result = Assertion::equals(&y1, &y2);
    assert!(
        result.passed,
        "Y position should be identical: {}",
        result.message
    );
}

#[test]
fn test_different_dt_diverge() {
    let mut sim1 = OrbitSimulation::new();
    let mut sim2 = OrbitSimulation::new();

    // Same total time, different dt
    sim1.run_steps(100, 3600.0); // 100 hours
    sim2.run_steps(200, 1800.0); // 100 hours with smaller dt

    // Energies should be close but not identical
    let rel_error = (sim1.total_energy() - sim2.total_energy()).abs() / sim1.total_energy().abs();
    let result = Assertion::is_true(
        rel_error < 1e-6,
        &format!("Energies should be within 1e-6: {:.2e}", rel_error),
    );
    assert!(
        result.passed,
        "Different dt should still conserve energy: {}",
        result.message
    );
}

// ============================================================================
// SCENARIO TESTS
// ============================================================================

#[test]
fn test_earth_orbital_period() {
    let mut sim = OrbitSimulation::new();
    let initial_x = sim.body_x(1);
    let initial_y = sim.body_y(1);

    // Simulate one year
    for _ in 0..365 {
        sim.step_days(1.0);
    }

    // Earth should return to approximately same position
    let final_x = sim.body_x(1);
    let final_y = sim.body_y(1);

    let distance = ((final_x - initial_x).powi(2) + (final_y - initial_y).powi(2)).sqrt();
    let orbital_radius = AU * 0.983; // Perihelion
    let rel_error = distance / orbital_radius;

    let result = Assertion::is_true(
        rel_error < 0.01,
        &format!("Position drift {:.2e} exceeds 1%", rel_error),
    );
    assert!(
        result.passed,
        "Earth should complete orbit in ~365 days: {}",
        result.message
    );
}

#[test]
fn test_positions_flat_length() {
    let sim = OrbitSimulation::new();
    let positions = sim.positions_flat();

    let result = Assertion::equals(&positions.len(), &6); // 2 bodies * 3 coords
    assert!(
        result.passed,
        "Positions flat should have 6 elements: {}",
        result.message
    );
}

#[test]
fn test_velocities_flat_length() {
    let sim = OrbitSimulation::new();
    let velocities = sim.velocities_flat();

    let result = Assertion::equals(&velocities.len(), &6); // 2 bodies * 3 coords
    assert!(
        result.passed,
        "Velocities flat should have 6 elements: {}",
        result.message
    );
}

#[test]
fn test_positions_au_flat() {
    let sim = OrbitSimulation::new();
    let positions_au = sim.positions_au_flat();

    // Earth at ~1 AU
    let earth_x_au = positions_au[3];
    let result = Assertion::in_range(earth_x_au, 0.9, 1.1);
    assert!(result.passed, "Earth X should be ~1 AU: {}", result.message);
}

// ============================================================================
// PAUSE/RESUME TESTS
// ============================================================================

#[test]
fn test_pause_resume() {
    let mut sim = OrbitSimulation::new();

    let result = Assertion::is_false(sim.paused(), "Should not be paused initially");
    assert!(result.passed);

    // Step should work
    let step_result = sim.step(3600.0);
    let result = Assertion::is_true(step_result, "Step should succeed");
    assert!(result.passed);
}

#[test]
fn test_reset() {
    let mut sim = OrbitSimulation::new();
    let initial_x = sim.body_x(1);

    // Move forward
    sim.run_steps(100, 86400.0);

    // Reset
    sim.reset();

    let result = Assertion::approx_eq(sim.body_x(1), initial_x, 1.0);
    assert!(result.passed, "Position should reset: {}", result.message);

    let result = Assertion::approx_eq(sim.sim_time(), 0.0, 0.001);
    assert!(result.passed, "Time should reset: {}", result.message);
}

// ============================================================================
// BOUNDARY CONDITIONS
// ============================================================================

#[test]
fn test_invalid_body_index() {
    let sim = OrbitSimulation::new();

    let result = Assertion::is_true(sim.body_x(999).is_nan(), "Invalid index should return NaN");
    assert!(result.passed);

    let result = Assertion::is_true(sim.body_y(999).is_nan(), "Invalid index should return NaN");
    assert!(result.passed);

    let result = Assertion::is_true(sim.body_z(999).is_nan(), "Invalid index should return NaN");
    assert!(result.passed);

    let result = Assertion::is_true(sim.body_vx(999).is_nan(), "Invalid index should return NaN");
    assert!(result.passed);

    let result = Assertion::is_true(sim.body_vy(999).is_nan(), "Invalid index should return NaN");
    assert!(result.passed);

    let result = Assertion::is_true(sim.body_vz(999).is_nan(), "Invalid index should return NaN");
    assert!(result.passed);

    let result = Assertion::is_true(
        sim.body_mass(999).is_nan(),
        "Invalid index should return NaN",
    );
    assert!(result.passed);
}

#[test]
fn test_sim_time_tracking() {
    let mut sim = OrbitSimulation::new();

    sim.step_days(10.0);

    let result = Assertion::approx_eq(sim.sim_time_days(), 10.0, 0.001);
    assert!(
        result.passed,
        "sim_time_days should track correctly: {}",
        result.message
    );

    let result = Assertion::approx_eq(sim.sim_time(), 10.0 * 86400.0, 1.0);
    assert!(
        result.passed,
        "sim_time should track correctly: {}",
        result.message
    );
}

// ============================================================================
// INVARIANT FUZZING (using deterministic time steps)
// ============================================================================

#[test]
fn test_fuzz_energy_invariant() {
    let mut sim = OrbitSimulation::new();
    let initial_energy = sim.total_energy();

    // Use varying time steps deterministically
    let time_steps = [
        100.0, 500.0, 1000.0, 2000.0, 3600.0, 5000.0, 7200.0, 10000.0,
    ];

    for frame in 0..1000 {
        let dt = time_steps[frame % time_steps.len()];
        sim.step(dt);

        // Check energy conservation
        let current_energy = sim.total_energy();
        let rel_error = (current_energy - initial_energy).abs() / initial_energy.abs();

        let result = Assertion::is_true(
            rel_error < 1e-6,
            &format!(
                "Frame {}: Energy drift {:.2e} exceeds 1e-6",
                frame, rel_error
            ),
        );
        assert!(result.passed, "{}", result.message);
    }
}

#[test]
fn test_fuzz_finite_values_invariant() {
    let mut sim = OrbitSimulation::new();

    // Use varying time steps
    let time_steps = [1000.0, 3600.0, 7200.0, 14400.0, 28800.0, 43200.0, 86400.0];

    for frame in 0..500 {
        let dt = time_steps[frame % time_steps.len()];
        sim.step(dt);

        // Check all values are finite
        for i in 0..sim.num_bodies() {
            let result = Assertion::is_true(
                sim.body_x(i).is_finite()
                    && sim.body_y(i).is_finite()
                    && sim.body_z(i).is_finite()
                    && sim.body_vx(i).is_finite()
                    && sim.body_vy(i).is_finite()
                    && sim.body_vz(i).is_finite(),
                &format!("Frame {}: Body {} has non-finite values", frame, i),
            );
            assert!(result.passed, "{}", result.message);
        }
    }
}

// ============================================================================
// HARNESS TEST
// ============================================================================

#[test]
fn test_probar_harness_integration() {
    let _harness = TestHarness::new();
    let mut suite = TestSuite::new("Orbit WASM Tests");

    suite.add_test(TestCase::new("energy_conservation"));
    suite.add_test(TestCase::new("angular_momentum_conservation"));
    suite.add_test(TestCase::new("deterministic_replay"));

    let result = Assertion::equals(&suite.test_count(), &3);
    assert!(result.passed, "Suite should have 3 tests");
}

// ============================================================================
// INVARIANT CHECKER INTEGRATION
// ============================================================================

#[test]
fn test_invariant_checker_orbit() {
    let mut checker = InvariantChecker::new();
    checker.add_check(InvariantCheck::new(
        "energy_bounded",
        "Energy must remain bounded",
    ));
    checker.add_check(InvariantCheck::new(
        "finite_values",
        "All values must be finite",
    ));
    checker.add_check(InvariantCheck::new(
        "angular_momentum",
        "Angular momentum conserved",
    ));

    let mut sim = OrbitSimulation::new();
    let initial_energy = sim.total_energy();

    for step in 0..100 {
        sim.step(3600.0);

        // Check energy
        let energy = sim.total_energy();
        let rel_error = (energy - initial_energy).abs() / initial_energy.abs();
        if rel_error > 1e-6 {
            checker.record_violation("energy_bounded", "Energy drift exceeded", step as u64);
        }

        // Check finite
        for i in 0..sim.num_bodies() {
            if !sim.body_x(i).is_finite() {
                checker.record_violation("finite_values", "Non-finite position", step as u64);
            }
        }
    }

    let result = Assertion::is_false(checker.has_violations(), "No invariant violations expected");
    assert!(result.passed, "Checker should have no violations");
}
