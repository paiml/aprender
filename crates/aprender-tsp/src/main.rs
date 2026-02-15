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

fn cmd_train(
    instances: &[PathBuf],
    algorithm: &str,
    output: &Path,
    iterations: usize,
    seed: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let algo =
        TspAlgorithm::parse(algorithm).ok_or_else(|| format!("Unknown algorithm: {algorithm}"))?;

    print_train_header(algo, instances.len(), iterations);

    let (loaded_instances, avg_size) = load_instances(instances)?;

    let start = Instant::now();
    let params = train_with_algorithm(algo, &loaded_instances, iterations, seed)?;
    let elapsed = start.elapsed();

    save_trained_model(algo, params, &loaded_instances, avg_size, elapsed, output)?;

    Ok(())
}

fn print_train_header(algo: TspAlgorithm, num_instances: usize, iterations: usize) {
    println!("Training TSP Model");
    println!("==================");
    println!("Algorithm:    {}", algo.as_str().to_uppercase());
    println!("Instances:    {num_instances}");
    println!("Iterations:   {iterations}");
    println!();
}

fn load_instances(
    instances: &[PathBuf],
) -> Result<(Vec<TspInstance>, usize), Box<dyn std::error::Error>> {
    let mut loaded = Vec::new();
    let mut total_cities = 0;

    for path in instances {
        let instance = TspInstance::load(path)?;
        println!(
            "  Loaded: {} ({} cities)",
            instance.name, instance.dimension
        );
        total_cities += instance.dimension;
        loaded.push(instance);
    }

    let avg_size = if loaded.is_empty() {
        0
    } else {
        total_cities / loaded.len()
    };
    println!();
    Ok((loaded, avg_size))
}

fn train_with_algorithm(
    algo: TspAlgorithm,
    instances: &[TspInstance],
    iterations: usize,
    seed: Option<u64>,
) -> Result<TspParams, Box<dyn std::error::Error>> {
    let params = match algo {
        TspAlgorithm::Aco => train_aco(instances, iterations, seed)?,
        TspAlgorithm::Tabu => train_tabu(instances, iterations, seed)?,
        TspAlgorithm::Ga => train_ga(instances, iterations, seed)?,
        TspAlgorithm::Hybrid => train_hybrid(instances, iterations, seed)?,
    };
    Ok(params)
}

fn train_aco(
    instances: &[TspInstance],
    iterations: usize,
    seed: Option<u64>,
) -> Result<TspParams, Box<dyn std::error::Error>> {
    let mut solver = AcoSolver::new();
    if let Some(s) = seed {
        solver = solver.with_seed(s);
    }
    run_solver(&mut solver, instances, iterations)?;
    Ok(TspParams::Aco {
        alpha: solver.alpha,
        beta: solver.beta,
        rho: solver.rho,
        q0: solver.q0,
        num_ants: solver.num_ants,
    })
}

fn train_tabu(
    instances: &[TspInstance],
    iterations: usize,
    seed: Option<u64>,
) -> Result<TspParams, Box<dyn std::error::Error>> {
    let mut solver = TabuSolver::new();
    if let Some(s) = seed {
        solver = solver.with_seed(s);
    }
    run_solver(&mut solver, instances, iterations)?;
    Ok(TspParams::Tabu {
        tenure: solver.tenure,
        max_neighbors: solver.max_neighbors,
    })
}

fn train_ga(
    instances: &[TspInstance],
    iterations: usize,
    seed: Option<u64>,
) -> Result<TspParams, Box<dyn std::error::Error>> {
    let mut solver = GaSolver::new();
    if let Some(s) = seed {
        solver = solver.with_seed(s);
    }
    run_solver(&mut solver, instances, iterations)?;
    Ok(TspParams::Ga {
        population_size: solver.population_size,
        crossover_rate: solver.crossover_rate,
        mutation_rate: solver.mutation_rate,
    })
}

fn train_hybrid(
    instances: &[TspInstance],
    iterations: usize,
    seed: Option<u64>,
) -> Result<TspParams, Box<dyn std::error::Error>> {
    let mut solver = HybridSolver::new();
    if let Some(s) = seed {
        solver = solver.with_seed(s);
    }
    run_solver(&mut solver, instances, iterations)?;
    Ok(TspParams::Hybrid {
        ga_fraction: solver.ga_fraction,
        tabu_fraction: solver.tabu_fraction,
        aco_fraction: solver.aco_fraction,
    })
}

fn run_solver<S: TspSolver>(
    solver: &mut S,
    instances: &[TspInstance],
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    for instance in instances {
        let solution = solver.solve(instance, Budget::Iterations(iterations))?;
        print_progress(&instance.name, solution.length);
    }
    Ok(())
}

fn save_trained_model(
    algo: TspAlgorithm,
    params: TspParams,
    instances: &[TspInstance],
    avg_size: usize,
    elapsed: std::time::Duration,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let metadata = TspModelMetadata {
        trained_instances: instances.len() as u32,
        avg_instance_size: avg_size as u32,
        best_known_gap: 0.0,
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

include!("main_part_02.rs");
