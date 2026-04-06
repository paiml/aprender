//! Simular Orbit Demo Example
//!
//! Demonstrates orbital mechanics simulation with:
//! - Type-safe units (Poka-Yoke)
//! - Yoshida symplectic integration
//! - Jidoka guards for energy/momentum conservation
//! - Heijunka time-budget scheduling
//!
//! Run with: cargo run --example orbit_demo

use simular::orbit::physics::YoshidaIntegrator;
use simular::orbit::prelude::*;
use simular::orbit::scenarios::ScenarioType;
use simular::orbit::{run_simulation, SimulationResult};

fn main() {
    println!("=== Simular Orbit Demo ===\n");

    // 1. Earth-Sun Keplerian Orbit
    println!("1. Earth-Sun Keplerian Orbit:");
    let config = KeplerConfig::earth_sun();
    println!(
        "   Semi-major axis: {:.2e} m (1 AU)",
        config.semi_major_axis
    );
    println!("   Eccentricity: {:.4}", config.eccentricity);
    println!("   Orbital period: {:.2} days", config.period() / 86400.0);
    println!(
        "   Circular velocity: {:.2} km/s",
        config.circular_velocity() / 1000.0
    );

    let state = config.build(1e6);
    println!(
        "   Initial Earth position: ({:.2e}, {:.2e}) m",
        state.bodies[1].position.as_meters().0,
        state.bodies[1].position.as_meters().1
    );
    println!();

    // 2. Symplectic Integration (Energy Conservation)
    println!("2. Yoshida 4th-Order Symplectic Integration:");
    let mut state = config.build(1e6);
    let yoshida = YoshidaIntegrator::new();
    let dt = OrbitTime::from_seconds(3600.0); // 1 hour steps

    let initial_energy = state.total_energy();
    let initial_l = state.angular_momentum_magnitude();

    // Simulate 1 year
    let steps = 365 * 24;
    for _ in 0..steps {
        if yoshida.step(&mut state, dt).is_err() {
            println!("   Integration error!");
            break;
        }
    }

    let final_energy = state.total_energy();
    let final_l = state.angular_momentum_magnitude();
    let energy_error = (final_energy - initial_energy).abs() / initial_energy.abs();
    let l_error = (final_l - initial_l).abs() / initial_l.abs();

    println!("   Simulated 1 year ({} steps)", steps);
    println!("   Energy drift: {:.2e} (target: <1e-9)", energy_error);
    println!(
        "   Angular momentum drift: {:.2e} (target: <1e-12)",
        l_error
    );
    println!(
        "   Final Earth position: ({:.2e}, {:.2e}) m",
        state.bodies[1].position.as_meters().0,
        state.bodies[1].position.as_meters().1
    );
    println!();

    // 3. Jidoka Guards
    println!("3. Jidoka Graceful Degradation:");
    let mut jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
    let state = config.build(1e6);
    jidoka.initialize(&state);

    let response = jidoka.check(&state);
    println!("   Initial check: {:?}", response);

    let status = jidoka.status();
    println!("   Energy OK: {}", status.energy_ok);
    println!("   Angular momentum OK: {}", status.angular_momentum_ok);
    println!("   Finite values OK: {}", status.finite_ok);
    println!();

    // 4. Heijunka Time-Budget Scheduler
    println!("4. Heijunka Time-Budget Scheduling:");
    let heijunka_config = HeijunkaConfig {
        frame_budget_ms: 16.67, // 60 FPS target
        physics_budget_fraction: 0.8,
        base_dt: 3600.0,
        max_substeps: 24,
        ..HeijunkaConfig::default()
    };
    let mut scheduler = HeijunkaScheduler::new(heijunka_config);
    let mut state = config.build(1e6);

    // Run 10 frames
    let mut total_sim_time = 0.0;
    for frame in 0..10 {
        if let Ok(result) = scheduler.execute_frame(&mut state) {
            total_sim_time += result.sim_time_advanced;
            if frame == 0 {
                let status = scheduler.status();
                println!("   Frame budget: {:.2} ms", status.budget_ms);
                println!("   Quality level: {:?}", status.quality);
                println!("   Utilization: {:.1}%", status.utilization * 100.0);
            }
        }
    }
    println!(
        "   10 frames simulated {:.2} hours",
        total_sim_time / 3600.0
    );
    println!();

    // 5. Full Year Simulation with run_simulation
    println!("5. Full Year Simulation:");
    let result: SimulationResult = run_simulation(
        &ScenarioType::Kepler(KeplerConfig::earth_sun()),
        365.25 * 86400.0, // 1 year
        3600.0,           // 1 hour steps
        1e6,              // softening
    );
    println!("   Steps completed: {}", result.steps);
    println!("   Warnings: {}", result.warnings);
    println!("   Paused (Jidoka): {}", result.paused);
    println!("   Energy error: {:.2e}", result.energy_error);
    println!(
        "   Angular momentum error: {:.2e}",
        result.angular_momentum_error
    );
    println!();

    // 6. Hohmann Transfer
    println!("6. Hohmann Transfer (Earth to Mars):");
    let hohmann = HohmannConfig::earth_to_mars();
    let dv1 = hohmann.delta_v1();
    let dv2 = hohmann.delta_v2();
    println!("   Departure burn (delta-v1): {:.2} km/s", dv1 / 1000.0);
    println!("   Arrival burn (delta-v2): {:.2} km/s", dv2 / 1000.0);
    println!("   Total delta-v: {:.2} km/s", (dv1 + dv2) / 1000.0);
    println!(
        "   Transfer time: {:.0} days",
        hohmann.transfer_time() / 86400.0
    );
    println!();

    // 7. Lagrange Points
    println!("7. Sun-Earth L2 Lagrange Point:");
    let l2_config = LagrangeConfig::sun_earth_l2();
    let (lx, ly, _lz) = l2_config.lagrange_position();
    let l2_distance = (lx * lx + ly * ly).sqrt();
    println!("   L2 distance from Sun: {:.4} AU", l2_distance / AU);
    println!(
        "   L2 distance from Earth: {:.2e} km",
        (l2_distance - AU) / 1000.0
    );
    println!();

    // 8. N-Body Solar System
    println!("8. Inner Solar System (N-Body):");
    let nbody = NBodyConfig::inner_solar_system();
    println!("   Bodies: {}", nbody.bodies.len());
    for body in &nbody.bodies {
        println!("   - {}", body.name);
    }

    let mut state = nbody.build(1e9);
    let initial_energy = state.total_energy();
    let dt = OrbitTime::from_seconds(86400.0); // 1 day

    for _ in 0..30 {
        if yoshida.step(&mut state, dt).is_err() {
            break;
        }
    }

    let final_energy = state.total_energy();
    let energy_error = (final_energy - initial_energy).abs() / initial_energy.abs();
    println!("   30 days simulated, energy error: {:.2e}", energy_error);
    println!();

    println!("=== Orbit Demo Complete ===");
    println!("\nToyota Way Principles Applied:");
    println!("  - Poka-Yoke: Type-safe units prevent dimensional errors");
    println!("  - Jidoka: Graceful degradation on physics violations");
    println!("  - Heijunka: Load-leveled frame delivery");
    println!("  - Mieruka: Visual status management (see TUI)");
    println!("\nRun TUI: cargo run --bin orbit-tui --features tui");
}
