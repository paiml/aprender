//! aprender-tsp CLI: Local TSP optimization with .apr models.

use aprender_tsp::{
    model::{TspModelMetadata, TspParams},
    AcoSolver, Budget, GaSolver, HybridSolver, TabuSolver, TspAlgorithm, TspInstance, TspModel,
    TspSolver,
};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "aprender-tsp")]
#[command(about = "Local TSP optimization with personalized .apr models")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a TSP model from problem instances
    Train {
        /// Input instance files (TSPLIB or CSV format)
        #[arg(required = true)]
        instances: Vec<PathBuf>,

        /// Algorithm to use
        #[arg(short, long, default_value = "aco")]
        algorithm: String,

        /// Output model file
        #[arg(short, long, default_value = "model.apr")]
        output: PathBuf,

        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: usize,

        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Solve a TSP instance using a trained model
    Solve {
        /// Instance file to solve
        instance: PathBuf,

        /// Trained model file
        #[arg(short, long)]
        model: PathBuf,

        /// Output solution file (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Maximum iterations (overrides model default)
        #[arg(long)]
        iterations: Option<usize>,
    },

    /// Evaluate model against instances
    Benchmark {
        /// Model file
        model: PathBuf,

        /// Instance files to benchmark
        #[arg(long)]
        instances: Vec<PathBuf>,
    },

    /// Display model information
    Info {
        /// Model file
        model: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Train {
            instances,
            algorithm,
            output,
            iterations,
            seed,
        } => cmd_train(&instances, &algorithm, &output, iterations, seed),
        Commands::Solve {
            instance,
            model,
            output,
            iterations,
        } => cmd_solve(&instance, &model, output.as_deref(), iterations),
        Commands::Benchmark { model, instances } => cmd_benchmark(&model, &instances),
        Commands::Info { model } => cmd_info(&model),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
fn cmd_train(
    instances: &[PathBuf],
    algorithm: &str,
    output: &Path,
    iterations: usize,
    seed: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let algo =
        TspAlgorithm::parse(algorithm).ok_or_else(|| format!("Unknown algorithm: {algorithm}"))?;

    println!("Training TSP Model");
    println!("==================");
    println!("Algorithm:    {}", algo.as_str().to_uppercase());
    println!("Instances:    {}", instances.len());
    println!("Iterations:   {iterations}");
    println!();

    // Load instances
    let mut loaded_instances = Vec::new();
    let mut total_cities = 0;

    for path in instances {
        let instance = TspInstance::load(path)?;
        println!(
            "  Loaded: {} ({} cities)",
            instance.name, instance.dimension
        );
        total_cities += instance.dimension;
        loaded_instances.push(instance);
    }

    let avg_size = if loaded_instances.is_empty() {
        0
    } else {
        total_cities / loaded_instances.len()
    };

    println!();

    // Train on each instance and average parameters
    let start = Instant::now();
    let mut best_gap = f64::INFINITY;

    // Create solver based on algorithm
    let params = match algo {
        TspAlgorithm::Aco => {
            let mut solver = AcoSolver::new();
            if let Some(s) = seed {
                solver = solver.with_seed(s);
            }

            for instance in &loaded_instances {
                let solution = solver.solve(instance, Budget::Iterations(iterations))?;
                print_progress(&instance.name, solution.length);

                if let Some(optimal) = instance.best_known {
                    let gap = (solution.length - optimal) / optimal * 100.0;
                    if gap < best_gap {
                        best_gap = gap;
                    }
                }
            }

            TspParams::Aco {
                alpha: solver.alpha,
                beta: solver.beta,
                rho: solver.rho,
                q0: solver.q0,
                num_ants: solver.num_ants,
            }
        }
        TspAlgorithm::Tabu => {
            let mut solver = TabuSolver::new();
            if let Some(s) = seed {
                solver = solver.with_seed(s);
            }

            for instance in &loaded_instances {
                let solution = solver.solve(instance, Budget::Iterations(iterations))?;
                print_progress(&instance.name, solution.length);
            }

            TspParams::Tabu {
                tenure: solver.tenure,
                max_neighbors: solver.max_neighbors,
            }
        }
        TspAlgorithm::Ga => {
            let mut solver = GaSolver::new();
            if let Some(s) = seed {
                solver = solver.with_seed(s);
            }

            for instance in &loaded_instances {
                let solution = solver.solve(instance, Budget::Iterations(iterations))?;
                print_progress(&instance.name, solution.length);
            }

            TspParams::Ga {
                population_size: solver.population_size,
                crossover_rate: solver.crossover_rate,
                mutation_rate: solver.mutation_rate,
            }
        }
        TspAlgorithm::Hybrid => {
            let mut solver = HybridSolver::new();
            if let Some(s) = seed {
                solver = solver.with_seed(s);
            }

            for instance in &loaded_instances {
                let solution = solver.solve(instance, Budget::Iterations(iterations))?;
                print_progress(&instance.name, solution.length);
            }

            TspParams::Hybrid {
                ga_fraction: solver.ga_fraction,
                tabu_fraction: solver.tabu_fraction,
                aco_fraction: solver.aco_fraction,
            }
        }
    };

    let elapsed = start.elapsed();

    // Create and save model
    let metadata = TspModelMetadata {
        trained_instances: loaded_instances.len() as u32,
        avg_instance_size: avg_size as u32,
        best_known_gap: if best_gap.is_finite() { best_gap } else { 0.0 },
        training_time_secs: elapsed.as_secs_f64(),
    };

    let model = TspModel::new(algo)
        .with_params(params)
        .with_metadata(metadata);

    model.save(output)?;

    println!();
    println!("Training Complete");
    println!("-----------------");
    println!("Training time:     {:.2}s", elapsed.as_secs_f64());
    println!(
        "Model saved:       {} ({} bytes)",
        output.display(),
        std::fs::metadata(output)?.len()
    );

    Ok(())
}

fn print_progress(name: &str, length: f64) {
    println!("  {name}: best tour = {length:.2}");
}

fn cmd_solve(
    instance_path: &Path,
    model_path: &Path,
    output: Option<&Path>,
    iterations: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = TspModel::load(model_path)?;
    let instance = TspInstance::load(instance_path)?;

    println!("Solving TSP Instance");
    println!("====================");
    println!(
        "Instance:     {} ({} cities)",
        instance.name, instance.dimension
    );
    println!(
        "Model:        {} ({:?})",
        model_path.display(),
        model.algorithm
    );
    println!();

    let iters = iterations.unwrap_or(1000);
    let start = Instant::now();

    // Create solver from model parameters
    let solution = match &model.params {
        TspParams::Aco {
            alpha,
            beta,
            rho,
            q0,
            num_ants,
        } => {
            let mut solver = AcoSolver::new()
                .with_alpha(*alpha)
                .with_beta(*beta)
                .with_rho(*rho)
                .with_q0(*q0)
                .with_num_ants(*num_ants);
            solver.solve(&instance, Budget::Iterations(iters))?
        }
        TspParams::Tabu {
            tenure,
            max_neighbors,
        } => {
            let mut solver = TabuSolver::new()
                .with_tenure(*tenure)
                .with_max_neighbors(*max_neighbors);
            solver.solve(&instance, Budget::Iterations(iters))?
        }
        TspParams::Ga {
            population_size,
            crossover_rate,
            mutation_rate,
        } => {
            let mut solver = GaSolver::new()
                .with_population_size(*population_size)
                .with_crossover_rate(*crossover_rate)
                .with_mutation_rate(*mutation_rate);
            solver.solve(&instance, Budget::Iterations(iters))?
        }
        TspParams::Hybrid {
            ga_fraction,
            tabu_fraction,
            aco_fraction,
        } => {
            let mut solver = HybridSolver::new()
                .with_ga_fraction(*ga_fraction)
                .with_tabu_fraction(*tabu_fraction)
                .with_aco_fraction(*aco_fraction);
            solver.solve(&instance, Budget::Iterations(iters))?
        }
    };

    let elapsed = start.elapsed();

    println!("Solution Found");
    println!("--------------");
    println!("Tour length:      {:.2}", solution.length);
    println!("Computation time: {:.3}s", elapsed.as_secs_f64());
    println!(
        "Tour: {} -> ... -> {}",
        solution.tour.first().unwrap_or(&0),
        solution.tour.last().unwrap_or(&0)
    );

    // Output to file if requested
    if let Some(out_path) = output {
        let json = format!(
            r#"{{"tour": {:?}, "length": {}, "evaluations": {}}}"#,
            solution.tour, solution.length, solution.evaluations
        );
        std::fs::write(out_path, json)?;
        println!("\nOutput: {}", out_path.display());
    }

    Ok(())
}

fn cmd_benchmark(
    model_path: &Path,
    instances: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    let model = TspModel::load(model_path)?;

    println!("Benchmark Results");
    println!("=================");
    println!(
        "Model: {} ({:?}, trained on {} instances)",
        model_path.display(),
        model.algorithm,
        model.metadata.trained_instances
    );
    println!();
    println!(
        "{:<15} {:>6} {:>10} {:>10} {:>8} {:>10}",
        "Instance", "Size", "Optimal", "Found", "Gap", "Tier"
    );
    println!("{}", "-".repeat(65));

    for path in instances {
        let instance = TspInstance::load(path)?;

        // Solve with model
        let solution = match &model.params {
            TspParams::Aco {
                alpha,
                beta,
                rho,
                q0,
                num_ants,
            } => {
                let mut solver = AcoSolver::new()
                    .with_alpha(*alpha)
                    .with_beta(*beta)
                    .with_rho(*rho)
                    .with_q0(*q0)
                    .with_num_ants(*num_ants);
                solver.solve(&instance, Budget::Iterations(500))?
            }
            TspParams::Tabu {
                tenure,
                max_neighbors,
            } => {
                let mut solver = TabuSolver::new()
                    .with_tenure(*tenure)
                    .with_max_neighbors(*max_neighbors);
                solver.solve(&instance, Budget::Iterations(500))?
            }
            TspParams::Ga {
                population_size,
                crossover_rate,
                mutation_rate,
            } => {
                let mut solver = GaSolver::new()
                    .with_population_size(*population_size)
                    .with_crossover_rate(*crossover_rate)
                    .with_mutation_rate(*mutation_rate);
                solver.solve(&instance, Budget::Iterations(500))?
            }
            TspParams::Hybrid {
                ga_fraction,
                tabu_fraction,
                aco_fraction,
            } => {
                let mut solver = HybridSolver::new()
                    .with_ga_fraction(*ga_fraction)
                    .with_tabu_fraction(*tabu_fraction)
                    .with_aco_fraction(*aco_fraction);
                solver.solve(&instance, Budget::Iterations(500))?
            }
        };

        let optimal_str = instance
            .best_known
            .map_or("-".to_string(), |o| format!("{o:.0}"));
        let gap_str = instance.best_known.map_or("-".to_string(), |o| {
            let gap = (solution.length - o) / o * 100.0;
            format!("{gap:.2}%")
        });
        let tier = instance.best_known.map_or("N/A".to_string(), |o| {
            let gap = (solution.length - o) / o * 100.0;
            aprender_tsp::SolutionTier::from_gap(gap)
                .as_str()
                .to_string()
        });

        println!(
            "{:<15} {:>6} {:>10} {:>10.0} {:>8} {:>10}",
            instance.name, instance.dimension, optimal_str, solution.length, gap_str, tier
        );
    }

    Ok(())
}

fn cmd_info(model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let model = TspModel::load(model_path)?;

    println!("Model Information");
    println!("=================");
    println!("File:             {}", model_path.display());
    println!("Algorithm:        {:?}", model.algorithm);
    println!();

    println!("Training Metadata");
    println!("-----------------");
    println!("Trained instances: {}", model.metadata.trained_instances);
    println!(
        "Avg instance size: {} cities",
        model.metadata.avg_instance_size
    );
    println!("Best known gap:    {:.2}%", model.metadata.best_known_gap);
    println!(
        "Training time:     {:.2}s",
        model.metadata.training_time_secs
    );
    println!();

    println!("Parameters");
    println!("----------");
    match &model.params {
        TspParams::Aco {
            alpha,
            beta,
            rho,
            q0,
            num_ants,
        } => {
            println!("alpha (pheromone):  {alpha:.2}");
            println!("beta (heuristic):   {beta:.2}");
            println!("rho (evaporation):  {rho:.2}");
            println!("q0 (exploitation):  {q0:.2}");
            println!("num_ants:           {num_ants}");
        }
        TspParams::Tabu {
            tenure,
            max_neighbors,
        } => {
            println!("tenure:         {tenure}");
            println!("max_neighbors:  {max_neighbors}");
        }
        TspParams::Ga {
            population_size,
            crossover_rate,
            mutation_rate,
        } => {
            println!("population_size:  {population_size}");
            println!("crossover_rate:   {crossover_rate:.2}");
            println!("mutation_rate:    {mutation_rate:.2}");
        }
        TspParams::Hybrid {
            ga_fraction,
            tabu_fraction,
            aco_fraction,
        } => {
            println!("ga_fraction:    {ga_fraction:.2}");
            println!("tabu_fraction:  {tabu_fraction:.2}");
            println!("aco_fraction:   {aco_fraction:.2}");
        }
    }

    Ok(())
}
