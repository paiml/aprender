//! EDD Falsification Example
//!
//! This example demonstrates the Popperian falsification approach in EDD:
//! actively searching for conditions that would disprove the model.
//!
//! Run with: cargo run --example edd_falsification

use simular::edd::{
    ExperimentSeed, FalsifiableSimulation, FalsificationAction, FalsificationCriterion, ParamSpace,
    Trajectory,
};
use std::collections::HashMap;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Popperian Falsification: Active Model Refutation          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    demonstrate_param_space();
    demonstrate_trajectory();
    demonstrate_experiment_seed();
    demonstrate_falsification_search();

    println!("\n✓ Falsification demonstration completed!");
}

fn demonstrate_param_space() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Parameter Space for Falsification Search");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Define parameter space for Little's Law falsification
    let space = ParamSpace::new()
        .with_param("lambda", 1.0, 10.0) // Throughput range
        .with_param("W", 0.5, 5.0) // Cycle time range
        .with_samples(5); // 5 samples per dimension

    println!("Parameter Space:");
    for (name, (min, max)) in &space.bounds {
        println!("  {name}: [{min:.1}, {max:.1}]");
    }
    println!("  Samples per dimension: {}", space.samples_per_dim);

    // Generate grid points
    let points = space.grid_points();
    println!("\nGenerated {} test points:", points.len());
    println!("┌────────────┬────────────┐");
    println!("│   lambda   │     W      │");
    println!("├────────────┼────────────┤");
    for point in points.iter().take(10) {
        let lambda = point.get("lambda").unwrap_or(&0.0);
        let w = point.get("W").unwrap_or(&0.0);
        println!("│   {:>6.2}   │   {:>6.2}   │", lambda, w);
    }
    if points.len() > 10 {
        println!("│    ...     │    ...     │");
    }
    println!("└────────────┴────────────┘");
    println!("(Total: {} points for exhaustive search)\n", points.len());
}

fn demonstrate_trajectory() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Simulation Trajectory for Robustness Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create a trajectory for a harmonic oscillator simulation
    let mut traj = Trajectory::new(vec![
        "position".to_string(),
        "velocity".to_string(),
        "energy".to_string(),
    ]);

    // Add some sample points (simulating SHO with ω=1)
    let omega = 1.0;
    for i in 0..10 {
        let t = i as f64 * 0.5;
        let x = (omega * t).cos();
        let v = -(omega * t).sin();
        let e = 0.5 * (x * x + v * v); // Energy should be constant = 0.5
        traj.push(t, &[x, v, e]);
    }

    println!(
        "Trajectory: {} time points, {} state variables",
        traj.len(),
        traj.state_names.len()
    );
    println!("\nSimulated Harmonic Oscillator (ω=1):");
    println!("┌────────┬──────────┬──────────┬──────────┐");
    println!("│  Time  │ Position │ Velocity │  Energy  │");
    println!("├────────┼──────────┼──────────┼──────────┤");
    for i in 0..traj.len().min(6) {
        let t = traj.times[i];
        let x = traj.get_state("position", i).unwrap_or(0.0);
        let v = traj.get_state("velocity", i).unwrap_or(0.0);
        let e = traj.get_state("energy", i).unwrap_or(0.0);
        println!("│ {:>6.2} │ {:>8.4} │ {:>8.4} │ {:>8.4} │", t, x, v, e);
    }
    println!("└────────┴──────────┴──────────┴──────────┘");

    // Check energy conservation
    let e0 = traj.get_state("energy", 0).unwrap();
    let e_final = traj.get_state("energy", traj.len() - 1).unwrap();
    let drift = (e_final - e0).abs() / e0;
    println!("\nEnergy Conservation Check:");
    println!("  Initial: {e0:.6}");
    println!("  Final:   {e_final:.6}");
    println!("  Drift:   {:.2e} ({:.4}%)", drift, drift * 100.0);
    println!(
        "  Status:  {}\n",
        if drift < 1e-10 {
            "✓ CONSERVED"
        } else {
            "⚠ DRIFT DETECTED"
        }
    );
}

fn demonstrate_experiment_seed() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Deterministic Reproducibility via Seeds");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create experiment seed
    let seed = ExperimentSeed::new(42)
        .with_component("arrivals", 12345)
        .with_component("service", 67890);

    println!("Experiment Seed Configuration:");
    println!("  Master seed: {}", seed.master_seed);
    println!("  IEEE strict: {}", seed.ieee_strict);
    println!(
        "  Pre-registered components: {}",
        seed.component_seeds.len()
    );

    // Derive component seeds
    println!("\nDerived Component Seeds:");
    let components = ["arrivals", "service", "routing", "failures", "demand"];
    for comp in components {
        let derived = seed.derive_seed(comp);
        let source = if seed.component_seeds.contains_key(comp) {
            "pre-registered"
        } else {
            "derived"
        };
        println!("  {comp}: {derived} ({source})");
    }

    // Demonstrate reproducibility
    println!("\nReproducibility Check:");
    let seed1 = seed.derive_seed("unknown_component");
    let seed2 = seed.derive_seed("unknown_component");
    println!(
        "  Same component, same seed: {} == {} → {}",
        seed1,
        seed2,
        if seed1 == seed2 {
            "✓ REPRODUCIBLE"
        } else {
            "✗ FAILED"
        }
    );

    let seed_a = ExperimentSeed::new(42).derive_seed("test");
    let seed_b = ExperimentSeed::new(42).derive_seed("test");
    println!(
        "  Different instances, same master: {} == {} → {}",
        seed_a,
        seed_b,
        if seed_a == seed_b {
            "✓ REPRODUCIBLE"
        } else {
            "✗ FAILED"
        }
    );

    println!();
}

fn demonstrate_falsification_search() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Active Falsification Search");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Define a mock simulation that implements FalsifiableSimulation
    struct HarmonicOscillator {
        fail_on_high_omega: bool,
    }

    impl FalsifiableSimulation for HarmonicOscillator {
        fn falsification_criteria(&self) -> Vec<FalsificationCriterion> {
            vec![
                FalsificationCriterion::new(
                    "energy_conservation",
                    "|E(t) - E(0)| / E(0) < 1e-6",
                    FalsificationAction::RejectModel,
                ),
                FalsificationCriterion::new(
                    "bounded_motion",
                    "|x(t)| <= amplitude for all t",
                    FalsificationAction::Stop,
                ),
            ]
        }

        fn evaluate(&self, params: &HashMap<String, f64>) -> Trajectory {
            let omega = params.get("omega").copied().unwrap_or(1.0);
            let mut traj = Trajectory::new(vec!["x".to_string(), "energy".to_string()]);

            // Simulate with potential instability at high omega
            for i in 0..10 {
                let t = i as f64 * 0.1;
                let x = (omega * t).cos();
                let energy = if self.fail_on_high_omega && omega > 5.0 {
                    0.5 * (1.0 + 0.1 * t) // Introduce energy drift
                } else {
                    0.5 // Conserved
                };
                traj.push(t, &[x, energy]);
            }
            traj
        }

        fn check_criterion(&self, criterion: &str, trajectory: &Trajectory) -> f64 {
            match criterion {
                "energy_conservation" => {
                    let e0 = trajectory.get_state("energy", 0).unwrap_or(0.5);
                    let e_final = trajectory
                        .get_state("energy", trajectory.len() - 1)
                        .unwrap_or(0.5);
                    let drift = (e_final - e0).abs() / e0;
                    1e-6 - drift // Positive = satisfied, Negative = violated
                }
                "bounded_motion" => {
                    let max_x = (0..trajectory.len())
                        .filter_map(|i| trajectory.get_state("x", i))
                        .map(|x| x.abs())
                        .fold(0.0, f64::max);
                    1.0 - max_x // Should be positive if |x| <= 1
                }
                _ => f64::INFINITY,
            }
        }
    }

    // Test 1: Model that passes all tests
    println!("Test 1: Stable Harmonic Oscillator");
    let stable_sim = HarmonicOscillator {
        fail_on_high_omega: false,
    };

    let params = ParamSpace::new()
        .with_param("omega", 0.5, 10.0)
        .with_samples(5);

    let result = stable_sim.seek_falsification(&params);
    println!("  Tests performed: {}", result.tests_performed);
    println!("  Falsified: {}", result.falsified);
    println!("  Min robustness: {:.6}", result.robustness);
    println!("  Summary: {}", result.summary);

    // Test 2: Model with instability at high omega
    println!("\nTest 2: Unstable Oscillator (energy drift at high ω)");
    let unstable_sim = HarmonicOscillator {
        fail_on_high_omega: true,
    };

    let result = unstable_sim.seek_falsification(&params);
    println!("  Tests performed: {}", result.tests_performed);
    println!("  Falsified: {}", result.falsified);
    if result.falsified {
        println!("  ✗ MODEL FALSIFIED!");
        if let Some(criterion) = &result.violated_criterion {
            println!("    Violated criterion: {criterion}");
        }
        if let Some(params) = &result.falsifying_params {
            for (k, v) in params {
                println!("    Falsifying {k}: {v:.4}");
            }
        }
    }
    println!("  Min robustness: {:.6}", result.robustness);
    println!("  Summary: {}", result.summary);
}
