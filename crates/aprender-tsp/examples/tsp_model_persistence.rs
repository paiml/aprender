//! Model persistence example demonstrating .apr format.
//!
//! This example shows how to:
//! 1. Train a TSP model
//! 2. Save it to .apr format with CRC32 checksum
//! 3. Load and verify the model
//! 4. Use loaded model for solving
//!
//! # Usage
//!
//! ```bash
//! cargo run --example tsp_model_persistence
//! ```

use aprender_tsp::{
    model::{TspModelMetadata, TspParams},
    AcoSolver, Budget, TspAlgorithm, TspInstance, TspModel, TspSolver,
};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TSP Model Persistence Demo ===");
    println!();

    // Create test instance
    let coords = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ];
    let instance = TspInstance::from_coords("hexagon", coords)?;

    println!(
        "Instance: {} ({} cities)",
        instance.name, instance.dimension
    );

    // Train model
    println!("\n1. Training ACO model...");
    let start = Instant::now();
    let mut solver = AcoSolver::new()
        .with_seed(42)
        .with_alpha(1.5)
        .with_beta(3.0)
        .with_rho(0.15)
        .with_q0(0.85)
        .with_num_ants(25);

    let solution = solver.solve(&instance, Budget::Iterations(500))?;
    let training_time = start.elapsed().as_secs_f64();

    println!("   Best tour length: {:.4}", solution.length);
    println!("   Training time: {:.3}s", training_time);

    // Create model with trained parameters
    let model = TspModel::new(TspAlgorithm::Aco)
        .with_params(TspParams::Aco {
            alpha: solver.alpha,
            beta: solver.beta,
            rho: solver.rho,
            q0: solver.q0,
            num_ants: solver.num_ants,
        })
        .with_metadata(TspModelMetadata {
            trained_instances: 1,
            avg_instance_size: instance.dimension as u32,
            best_known_gap: 0.0,
            training_time_secs: training_time,
        });

    // Save model
    let model_path = Path::new("demo_model.apr");
    println!("\n2. Saving model to {:?}...", model_path);
    model.save(model_path)?;

    let file_size = std::fs::metadata(model_path)?.len();
    println!("   File size: {} bytes", file_size);
    println!("   Format: .apr v1 with CRC32 checksum");

    // Load model
    println!("\n3. Loading model from {:?}...", model_path);
    let loaded = TspModel::load(model_path)?;

    println!("   Algorithm: {:?}", loaded.algorithm);
    println!(
        "   Trained instances: {}",
        loaded.metadata.trained_instances
    );
    println!(
        "   Avg instance size: {} cities",
        loaded.metadata.avg_instance_size
    );

    // Verify parameters
    if let TspParams::Aco {
        alpha,
        beta,
        rho,
        q0,
        num_ants,
    } = &loaded.params
    {
        println!("\n4. Loaded parameters:");
        println!("   alpha: {:.2}", alpha);
        println!("   beta: {:.2}", beta);
        println!("   rho: {:.2}", rho);
        println!("   q0: {:.2}", q0);
        println!("   num_ants: {}", num_ants);
    }

    // Use loaded model for solving
    println!("\n5. Solving with loaded model...");
    let mut loaded_solver = AcoSolver::new().with_seed(42);
    if let TspParams::Aco {
        alpha,
        beta,
        rho,
        q0,
        num_ants,
    } = loaded.params
    {
        loaded_solver = loaded_solver
            .with_alpha(alpha)
            .with_beta(beta)
            .with_rho(rho)
            .with_q0(q0)
            .with_num_ants(num_ants);
    }

    let loaded_solution = loaded_solver.solve(&instance, Budget::Iterations(500))?;
    println!("   Tour length: {:.4}", loaded_solution.length);

    // Verify determinism
    assert!(
        (solution.length - loaded_solution.length).abs() < 1e-10,
        "Loaded model should produce identical results"
    );
    println!("\n   Determinism verified: original == loaded");

    // Cleanup
    std::fs::remove_file(model_path)?;
    println!("\n6. Cleaned up temporary model file.");

    println!("\nDemo complete!");
    Ok(())
}
