//! entrenar-bench CLI entry point.

use clap::{Parser, Subcommand};
use entrenar_bench::{
    cost::{generate_sample_points, Constraints, CostModel, CostPerformanceAnalysis},
    strategies::{compare, DistillStrategy},
    sweep::{SweepConfig, Sweeper},
};
use entrenar_common::cli::{styles, CommonArgs};

#[derive(Parser)]
#[command(name = "entrenar-bench")]
#[command(about = "Distillation benchmarking and hyperparameter sweep tool")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Sweep temperature hyperparameter
    Temperature {
        /// Start of range
        #[arg(long, default_value = "1.0")]
        start: f32,

        /// End of range
        #[arg(long, default_value = "8.0")]
        end: f32,

        /// Step size
        #[arg(long, default_value = "0.5")]
        step: f32,

        /// Runs per configuration
        #[arg(long, default_value = "3")]
        runs: usize,
    },

    /// Sweep alpha hyperparameter
    Alpha {
        /// Start of range
        #[arg(long, default_value = "0.1")]
        start: f32,

        /// End of range
        #[arg(long, default_value = "0.9")]
        end: f32,

        /// Step size
        #[arg(long, default_value = "0.1")]
        step: f32,

        /// Runs per configuration
        #[arg(long, default_value = "3")]
        runs: usize,
    },

    /// Compare distillation strategies
    Compare {
        /// Strategies to compare (kd, progressive, attention, combined, all)
        #[arg(long, value_delimiter = ',', default_value = "all")]
        strategies: Vec<String>,

        /// Runs per strategy
        #[arg(long, default_value = "5")]
        runs: usize,
    },

    /// Run ablation study
    Ablation {
        /// Base configuration file
        #[arg(short, long)]
        config: Option<std::path::PathBuf>,
    },

    /// Analyze cost vs performance trade-offs
    CostPerformance {
        /// GPU type for cost calculation
        #[arg(long, default_value = "a100-80gb")]
        gpu: String,

        /// Path to benchmark results file (JSON)
        #[arg(long)]
        results: Option<std::path::PathBuf>,
    },

    /// Recommend configurations based on constraints
    Recommend {
        /// Maximum GPU-hours
        #[arg(long)]
        max_gpu_hours: Option<f64>,

        /// Maximum cost in USD
        #[arg(long)]
        max_cost: Option<f64>,

        /// Minimum accuracy required (0.0 - 1.0)
        #[arg(long)]
        min_accuracy: Option<f64>,

        /// Maximum memory in GB
        #[arg(long)]
        max_memory: Option<f64>,

        /// GPU type for cost calculation
        #[arg(long, default_value = "a100-80gb")]
        gpu: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let config = cli.common.to_cli();

    let result = match cli.command {
        Commands::Temperature {
            start,
            end,
            step,
            runs,
        } => temperature_command(start, end, step, runs, &config),
        Commands::Alpha {
            start,
            end,
            step,
            runs,
        } => alpha_command(start, end, step, runs, &config),
        Commands::Compare { strategies, runs } => compare_command(&strategies, runs, &config),
        Commands::Ablation { config: cfg_path } => ablation_command(cfg_path.as_deref(), &config),
        Commands::CostPerformance { gpu, results } => {
            cost_performance_command(&gpu, results.as_deref(), &config)
        }
        Commands::Recommend {
            max_gpu_hours,
            max_cost,
            min_accuracy,
            max_memory,
            gpu,
        } => recommend_command(
            max_gpu_hours,
            max_cost,
            min_accuracy,
            max_memory,
            &gpu,
            &config,
        ),
    };

    if let Err(e) = result {
        if !config.is_quiet() {
            eprintln!("{}", styles::error(&e.to_string()));
        }
        std::process::exit(1);
    }
}

fn temperature_command(
    start: f32,
    end: f32,
    step: f32,
    runs: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !cli.is_quiet() {
        println!("{}", styles::header("Temperature Sweep"));
        println!("Range: {start:.1} to {end:.1}, step {step:.1}, {runs} runs per point\n");
    }

    let config = SweepConfig::temperature(start..end, step).with_runs(runs);
    let sweeper = Sweeper::new(config);
    let result = sweeper.run()?;

    if cli.format == entrenar_common::OutputFormat::Json {
        let json: Vec<_> = result
            .data_points
            .iter()
            .map(|p| {
                serde_json::json!({
                    "value": p.parameter_value,
                    "loss": p.mean_loss,
                    "loss_std": p.std_loss,
                    "accuracy": p.mean_accuracy,
                    "accuracy_std": p.std_accuracy,
                })
            })
            .collect();
        if let Ok(json_str) = serde_json::to_string_pretty(&json) {
            println!("{json_str}");
        }
    } else {
        println!("{}", result.to_table());
    }

    Ok(())
}

fn alpha_command(
    start: f32,
    end: f32,
    step: f32,
    runs: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !cli.is_quiet() {
        println!("{}", styles::header("Alpha Sweep"));
        println!("Range: {start:.1} to {end:.1}, step {step:.1}, {runs} runs per point\n");
    }

    let config = SweepConfig::alpha(start..end, step).with_runs(runs);
    let sweeper = Sweeper::new(config);
    let result = sweeper.run()?;

    if cli.format == entrenar_common::OutputFormat::Json {
        let json: Vec<_> = result
            .data_points
            .iter()
            .map(|p| {
                serde_json::json!({
                    "value": p.parameter_value,
                    "loss": p.mean_loss,
                    "loss_std": p.std_loss,
                    "accuracy": p.mean_accuracy,
                    "accuracy_std": p.std_accuracy,
                })
            })
            .collect();
        if let Ok(json_str) = serde_json::to_string_pretty(&json) {
            println!("{json_str}");
        }
    } else {
        println!("{}", result.to_table());
    }

    Ok(())
}

fn compare_command(
    strategy_names: &[String],
    _runs: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let strategies: Vec<DistillStrategy> = if strategy_names.iter().any(|s| s == "all") {
        vec![
            DistillStrategy::kd_only(),
            DistillStrategy::progressive(),
            DistillStrategy::attention(),
            DistillStrategy::combined(),
        ]
    } else {
        strategy_names
            .iter()
            .filter_map(|name| match name.to_lowercase().as_str() {
                "kd" | "kd-only" | "kdonly" => Some(DistillStrategy::kd_only()),
                "progressive" | "prog" => Some(DistillStrategy::progressive()),
                "attention" | "attn" => Some(DistillStrategy::attention()),
                "combined" | "all" => Some(DistillStrategy::combined()),
                _ => None,
            })
            .collect()
    };

    if strategies.is_empty() {
        return Err(entrenar_common::EntrenarError::ConfigValue {
            field: "strategies".into(),
            message: "No valid strategies specified".into(),
            suggestion: "Use: kd, progressive, attention, combined, all".into(),
        });
    }

    if !cli.is_quiet() {
        println!("{}", styles::header("Strategy Comparison"));
        println!("Comparing {} strategies\n", strategies.len());
    }

    let comparison = compare(&strategies)?;

    if cli.format == entrenar_common::OutputFormat::Json {
        let json = serde_json::json!({
            "results": comparison.results.iter().map(|r| {
                serde_json::json!({
                    "strategy": r.name,
                    "loss": r.mean_loss,
                    "loss_std": r.std_loss,
                    "accuracy": r.mean_accuracy,
                    "accuracy_std": r.std_accuracy,
                    "time_hours": r.mean_time_hours,
                })
            }).collect::<Vec<_>>(),
            "best_by_loss": comparison.best_by_loss,
            "best_by_accuracy": comparison.best_by_accuracy,
        });
        if let Ok(json_str) = serde_json::to_string_pretty(&json) {
            println!("{json_str}");
        }
    } else {
        println!("{}", comparison.to_table());

        if let Some(best) = &comparison.best_by_accuracy {
            println!(
                "\n{}",
                styles::success(&format!("Recommendation: {best} for best accuracy"))
            );
        }
    }

    Ok(())
}

fn ablation_command(
    _config_path: Option<&std::path::Path>,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    if !cli.is_quiet() {
        println!("{}", styles::header("Ablation Study"));
        println!("Testing contribution of each component...\n");
    }

    // Run ablation by progressively adding components
    let ablations = [
        (
            "Baseline (CE only)",
            DistillStrategy::KDOnly {
                temperature: 1.0,
                alpha: 0.0, // No KD, just CE
            },
        ),
        (
            "+ KD (T=4)",
            DistillStrategy::KDOnly {
                temperature: 4.0,
                alpha: 0.7,
            },
        ),
        (
            "+ Progressive",
            DistillStrategy::Progressive {
                temperature: 4.0,
                alpha: 0.7,
                layer_weight: 0.3,
            },
        ),
        (
            "+ Attention",
            DistillStrategy::Combined {
                temperature: 4.0,
                alpha: 0.7,
                layer_weight: 0.3,
                attention_weight: 0.1,
            },
        ),
    ];

    let strategies: Vec<DistillStrategy> = ablations.iter().map(|(_, s)| s.clone()).collect();
    let comparison = compare(&strategies)?;

    // Custom output for ablation
    println!("Ablation Results:");
    println!("┌─────────────────────┬────────────┬────────────┬────────────┐");
    println!("│ Configuration       │ Loss       │ Δ Loss     │ Accuracy   │");
    println!("├─────────────────────┼────────────┼────────────┼────────────┤");

    let mut prev_loss = None;
    for (i, (name, _)) in ablations.iter().enumerate() {
        let result = &comparison.results[i];
        let delta = prev_loss
            .map(|p: f64| result.mean_loss - p)
            .map_or_else(|| "-".to_string(), |d| format!("{d:+.4}"));

        println!(
            "│ {:19} │ {:>10.4} │ {:>10} │ {:>9.1}% │",
            name,
            result.mean_loss,
            delta,
            result.mean_accuracy * 100.0
        );

        prev_loss = Some(result.mean_loss);
    }

    println!("└─────────────────────┴────────────┴────────────┴────────────┘");

    Ok(())
}

fn cost_performance_command(
    gpu: &str,
    _results_path: Option<&std::path::Path>,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    // Parse GPU type
    let cost_model = parse_gpu_model(gpu)?;

    if !cli.is_quiet() {
        println!("{}", styles::header("Cost-Performance Analysis"));
        println!(
            "GPU: {} (${:.2}/hour)\n",
            cost_model.gpu_type, cost_model.cost_per_hour
        );
    }

    // Generate sample data points (in a real scenario, load from results file)
    let points = generate_sample_points(&cost_model);
    let analysis = CostPerformanceAnalysis::from_points(points);

    if cli.format == entrenar_common::OutputFormat::Json {
        let json = serde_json::json!({
            "gpu": cost_model.gpu_type,
            "cost_per_hour": cost_model.cost_per_hour,
            "points": analysis.points,
            "pareto_frontier": analysis.pareto_frontier,
            "best_accuracy": analysis.best_accuracy,
            "best_efficiency": analysis.best_efficiency,
            "lowest_cost": analysis.lowest_cost,
        });
        if let Ok(json_str) = serde_json::to_string_pretty(&json) {
            println!("{json_str}");
        }
    } else {
        println!("{}", analysis.to_table());

        if let Some(best) = &analysis.best_accuracy {
            println!(
                "{}",
                styles::info(&format!(
                    "Best accuracy: {} ({:.1}%)",
                    best.name,
                    best.accuracy * 100.0
                ))
            );
        }

        if let Some(best) = &analysis.best_efficiency {
            let efficiency = best.accuracy / best.cost_usd;
            println!(
                "{}",
                styles::info(&format!(
                    "Best efficiency: {} ({:.4}% per $)",
                    best.name,
                    efficiency * 100.0
                ))
            );
        }

        println!("\nPareto-optimal configurations:");
        for point in &analysis.pareto_frontier {
            println!(
                "  • {} - ${:.2}, {:.1}% accuracy",
                point.name,
                point.cost_usd,
                point.accuracy * 100.0
            );
        }
    }

    Ok(())
}

/// Print constraint summary to stdout.
fn print_constraints(
    max_gpu_hours: Option<f64>,
    max_cost: Option<f64>,
    min_accuracy: Option<f64>,
    max_memory: Option<f64>,
) {
    println!("Constraints:");

    let constraint_lines: Vec<String> = [
        max_gpu_hours.map(|h| format!("  \u{2022} Max GPU-hours: {h}")),
        max_cost.map(|c| format!("  \u{2022} Max cost: ${c}")),
        min_accuracy.map(|a| format!("  \u{2022} Min accuracy: {:.1}%", a * 100.0)),
        max_memory.map(|m| format!("  \u{2022} Max memory: {m} GB")),
    ]
    .into_iter()
    .flatten()
    .collect();

    if constraint_lines.is_empty() {
        println!("  (none specified - showing all recommendations)");
    } else {
        for line in &constraint_lines {
            println!("{line}");
        }
    }
    println!();
}

/// Build a `Constraints` value from optional fields.
fn build_constraints(
    max_gpu_hours: Option<f64>,
    max_cost: Option<f64>,
    min_accuracy: Option<f64>,
    max_memory: Option<f64>,
) -> Constraints {
    let mut constraints = Constraints::new();
    if let Some(h) = max_gpu_hours {
        constraints = constraints.with_max_gpu_hours(h);
    }
    if let Some(c) = max_cost {
        constraints = constraints.with_max_cost(c);
    }
    if let Some(a) = min_accuracy {
        constraints = constraints.with_min_accuracy(a);
    }
    if let Some(m) = max_memory {
        constraints = constraints.with_max_memory(m);
    }
    constraints
}

/// Print human-readable recommendation output (non-JSON).
fn print_recommendations(recommendations: &[entrenar_bench::cost::Recommendation]) {
    if recommendations.is_empty() {
        println!(
            "{}",
            styles::warning("No configurations match the specified constraints.")
        );
        println!("\nTry relaxing your constraints:");
        println!("  \u{2022} Increase max-cost or max-gpu-hours");
        println!("  \u{2022} Decrease min-accuracy");
        println!("  \u{2022} Increase max-memory");
        return;
    }

    println!("Recommendations:\n");
    for (i, rec) in recommendations.iter().enumerate() {
        let bullet = if i == 0 { "\u{2605}" } else { "\u{2022}" };
        println!("{bullet} {} ({})", rec.point.name, rec.reason);
        println!("    GPU hours: {:.1}", rec.point.gpu_hours);
        println!("    Cost: ${:.2}", rec.point.cost_usd);
        println!("    Accuracy: {:.1}%", rec.point.accuracy * 100.0);
        println!("    Memory: {:.0} GB", rec.point.memory_gb);
        print_optional_config(&rec.point.config);
        println!();
    }

    if let Some(top) = recommendations.first() {
        println!(
            "{}",
            styles::success(&format!("Top recommendation: {}", top.point.name))
        );
    }
}

/// Print optional configuration fields (LoRA rank, quantization bits, temperature).
fn print_optional_config(config: &entrenar_bench::cost::ConfigParams) {
    if let Some(rank) = config.lora_rank {
        println!("    LoRA rank: {rank}");
    }
    if let Some(bits) = config.quant_bits {
        println!("    Quantization: {bits}-bit");
    }
    if let Some(temp) = config.temperature {
        println!("    Temperature: {temp}");
    }
}

fn recommend_command(
    max_gpu_hours: Option<f64>,
    max_cost: Option<f64>,
    min_accuracy: Option<f64>,
    max_memory: Option<f64>,
    gpu: &str,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let cost_model = parse_gpu_model(gpu)?;

    if !cli.is_quiet() {
        println!("{}", styles::header("Configuration Recommendation"));
        println!(
            "GPU: {} (${:.2}/hour)\n",
            cost_model.gpu_type, cost_model.cost_per_hour
        );
        print_constraints(max_gpu_hours, max_cost, min_accuracy, max_memory);
    }

    let constraints = build_constraints(max_gpu_hours, max_cost, min_accuracy, max_memory);
    let points = generate_sample_points(&cost_model);
    let analysis = CostPerformanceAnalysis::from_points(points);
    let recommendations = analysis.recommend(&constraints);

    if cli.format == entrenar_common::OutputFormat::Json {
        let json = serde_json::json!({
            "constraints": {
                "max_gpu_hours": max_gpu_hours,
                "max_cost": max_cost,
                "min_accuracy": min_accuracy,
                "max_memory": max_memory,
            },
            "recommendations": recommendations,
        });
        if let Ok(json_str) = serde_json::to_string_pretty(&json) {
            println!("{json_str}");
        }
    } else {
        print_recommendations(&recommendations);
    }

    Ok(())
}

fn parse_gpu_model(gpu: &str) -> entrenar_common::Result<CostModel> {
    match gpu.to_lowercase().as_str() {
        "a100-80gb" | "a100_80gb" => Ok(CostModel::a100_80gb()),
        "a100-40gb" | "a100_40gb" => Ok(CostModel::a100_40gb()),
        "v100" => Ok(CostModel::v100()),
        "t4" => Ok(CostModel::t4()),
        _ => Err(entrenar_common::EntrenarError::ConfigValue {
            field: "gpu".into(),
            message: format!("Unknown GPU type: {gpu}"),
            suggestion: "Use: a100-80gb, a100-40gb, v100, t4".into(),
        }),
    }
}
