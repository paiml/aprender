//! Physics Simulation Example
//!
//! Demonstrates the Verlet integration physics engine:
//! - Simple gravity simulation
//! - Energy conservation verification (Jidoka)
//! - Central force field (orbital mechanics)
//!
//! # Running
//! ```bash
//! cargo run --example physics_simulation
//! ```

use simular::domains::physics::{
    CentralForceField, EulerIntegrator, GravityField, PhysicsEngine, VerletIntegrator,
};
use simular::engine::state::{SimState, Vec3};

fn main() {
    println!("=== Simular Physics Simulation ===\n");

    // 1. Simple projectile with gravity
    println!("1. Projectile Motion (Gravity Field):");

    let mut state = SimState::new();
    state.add_body(
        1.0,                        // mass
        Vec3::new(0.0, 0.0, 100.0), // position: 100m height
        Vec3::new(10.0, 0.0, 20.0), // velocity: launch upward
    );

    let gravity = GravityField::default(); // g = -9.81 m/s² in z
    let integrator = VerletIntegrator;
    let engine = PhysicsEngine::new(gravity, integrator);

    let dt = 0.01; // 10ms timestep
    let initial_energy = state.kinetic_energy() + state.potential_energy();

    println!("   Initial position: {:?}", state.positions()[0]);
    println!("   Initial velocity: {:?}", state.velocities()[0]);
    println!("   Initial energy:   {:.4} J\n", initial_energy);

    // Simulate until projectile hits ground (z <= 0)
    let mut steps = 0;
    while state.positions()[0].z > 0.0 && steps < 10000 {
        let _ = engine.step(&mut state, dt);
        steps += 1;
    }

    let final_energy = state.kinetic_energy() + state.potential_energy();
    let energy_drift = if initial_energy.abs() > f64::EPSILON {
        ((final_energy - initial_energy) / initial_energy).abs()
    } else {
        0.0
    };

    println!("   Final position:   {:?}", state.positions()[0]);
    println!("   Final velocity:   {:?}", state.velocities()[0]);
    println!("   Steps taken:      {}", steps);
    println!("   Energy drift:     {:.2e}", energy_drift);
    println!(
        "   Symplectic:       {} (Verlet preserves energy)\n",
        energy_drift < 0.01
    );

    // 2. Compare integrators
    println!("2. Integrator Comparison:");

    fn run_free_fall(engine: &PhysicsEngine, name: &str) {
        let mut state = SimState::new();
        state.add_body(1.0, Vec3::new(0.0, 0.0, 100.0), Vec3::zero());

        let dt = 0.1; // Larger timestep to show differences

        for _ in 0..100 {
            let _ = engine.step(&mut state, dt);
        }

        // Analytical: z = z0 - 0.5*g*t² = 100 - 0.5*9.81*100 = -390.5
        let expected_z = 100.0 - 0.5 * 9.81 * 100.0;
        let error = (state.positions()[0].z - expected_z).abs();
        println!(
            "   {}: final z = {:.2}, error = {:.4}",
            name,
            state.positions()[0].z,
            error
        );
    }

    let gravity = GravityField::default();
    run_free_fall(
        &PhysicsEngine::new(gravity.clone(), VerletIntegrator),
        "Verlet  ",
    );
    run_free_fall(
        &PhysicsEngine::new(gravity.clone(), EulerIntegrator),
        "Euler   ",
    );

    // 3. Central force field (orbital mechanics)
    println!("\n3. Central Force Field (Orbital Mechanics):");

    let mut state = SimState::new();
    state.add_body(
        1.0,                      // mass
        Vec3::new(1.0, 0.0, 0.0), // Start on x-axis
        Vec3::new(0.0, 1.0, 0.0), // Circular orbit velocity
    );

    // mu = 1.0 for unit circular orbit at r=1 with v=1
    let central = CentralForceField::new(1.0, Vec3::zero());
    let engine = PhysicsEngine::new(central, VerletIntegrator);

    let dt = 0.01;
    let initial_energy = state.kinetic_energy() + state.potential_energy();
    let initial_r = state.positions()[0].magnitude();

    // One full orbit (approximately 2*pi time units)
    let orbit_steps = ((2.0 * std::f64::consts::PI) / dt) as usize;

    for _ in 0..orbit_steps {
        let _ = engine.step(&mut state, dt);
    }

    let final_energy = state.kinetic_energy() + state.potential_energy();
    let final_r = state.positions()[0].magnitude();
    let energy_drift = if initial_energy.abs() > f64::EPSILON {
        ((final_energy - initial_energy) / initial_energy).abs()
    } else {
        0.0
    };

    println!("   Initial radius: {:.6}", initial_r);
    println!("   Final radius:   {:.6}", final_r);
    println!("   Radius drift:   {:.6}", (final_r - initial_r).abs());
    println!("   Energy drift:   {:.2e}", energy_drift);
    println!("   Orbit closed:   {}", (final_r - initial_r).abs() < 0.01);

    // 4. Determinism verification
    println!("\n4. Determinism Verification:");

    fn simulate_and_get_pos() -> Vec3 {
        let mut state = SimState::new();
        state.add_body(2.5, Vec3::new(0.0, 0.0, 100.0), Vec3::new(5.0, 3.0, 10.0));

        let gravity = GravityField::default();
        let engine = PhysicsEngine::new(gravity, VerletIntegrator);

        for _ in 0..1000 {
            let _ = engine.step(&mut state, 0.001);
        }

        state.positions()[0]
    }

    let result1 = simulate_and_get_pos();
    let result2 = simulate_and_get_pos();

    println!(
        "   Run 1 position: ({:.10}, {:.10}, {:.10})",
        result1.x, result1.y, result1.z
    );
    println!(
        "   Run 2 position: ({:.10}, {:.10}, {:.10})",
        result2.x, result2.y, result2.z
    );
    println!("   Bitwise identical: {}", result1 == result2);

    println!("\n=== Simulation Complete ===");
}
