
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
