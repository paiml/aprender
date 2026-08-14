//! Command implementations for the distillation benchmark tool.
//!
//! These functions were the body of the `aprender-train-bench` binary's
//! `main.rs`. That binary is gone (APR-MONO Rule 1: `apr` is the only
//! user-facing binary); the capability is reachable as `apr train bench <verb>`,
//! which calls exactly these entry points. Nothing was reimplemented in the CLI
//! layer — `apr` parses arguments and delegates here.

// The `serde_json::json!` macro expands to code containing `.unwrap()`, which
// trips clippy::disallowed_methods at the macro invocation site even though no
// author-written unwrap exists. Scope the allow to this presentation module.
#![allow(clippy::disallowed_methods)]

use crate::cost::{generate_sample_points, Constraints, CostModel, CostPerformanceAnalysis};
use crate::strategies::{compare, DistillStrategy};
use crate::sweep::{SweepConfig, Sweeper};
use entrenar_common::cli::styles;

/// Sweep the distillation temperature hyperparameter over `start..end`.
///
/// # Errors
///
/// Propagates sweep failures from [`Sweeper::run`].
pub fn run_temperature(
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

/// Sweep the KD/CE mixing weight (alpha) over `start..end`.
///
/// # Errors
///
/// Propagates sweep failures from [`Sweeper::run`].
pub fn run_alpha(
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

/// Resolve strategy names to [`DistillStrategy`] values.
///
/// `"all"` anywhere in the list expands to every strategy. Unknown names are
/// dropped; an all-unknown list yields an empty vector, which
/// [`run_compare`] turns into a refusal.
#[must_use]
pub fn resolve_strategies(strategy_names: &[String]) -> Vec<DistillStrategy> {
    if strategy_names.iter().any(|s| s == "all") {
        return vec![
            DistillStrategy::kd_only(),
            DistillStrategy::progressive(),
            DistillStrategy::attention(),
            DistillStrategy::combined(),
        ];
    }

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
}

/// Compare distillation strategies head to head.
///
/// `runs` is accepted for CLI compatibility and deliberately unused: the
/// comparison harness is deterministic, so repeat runs produce identical
/// numbers. This matches the pre-migration binary exactly.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::ConfigValue`] when no name in
/// `strategy_names` resolves to a known strategy.
pub fn run_compare(
    strategy_names: &[String],
    runs: usize,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let _ = runs;
    let strategies = resolve_strategies(strategy_names);

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

/// Run the ablation study: baseline, +KD, +progressive, +attention.
///
/// `config_path` is accepted for CLI compatibility and deliberately unused —
/// the ablation ladder is fixed in code. This matches the pre-migration binary.
///
/// # Errors
///
/// Propagates comparison failures from [`compare`].
pub fn run_ablation(
    config_path: Option<&std::path::Path>,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let _ = config_path;
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

/// Analyse the cost/performance frontier for a GPU type.
///
/// `results_path` is accepted for CLI compatibility and deliberately unused —
/// the analysis runs on generated sample points. This matches the
/// pre-migration binary.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::ConfigValue`] for an unknown GPU.
pub fn run_cost_performance(
    gpu: &str,
    results_path: Option<&std::path::Path>,
    cli: &entrenar_common::Cli,
) -> entrenar_common::Result<()> {
    let _ = results_path;
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
#[must_use]
pub fn build_constraints(
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
fn print_recommendations(recommendations: &[crate::cost::Recommendation]) {
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
fn print_optional_config(config: &crate::cost::ConfigParams) {
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

/// Recommend configurations that satisfy the given budget constraints.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::ConfigValue`] for an unknown GPU.
pub fn run_recommend(
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

/// Map a GPU name to its cost model.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::ConfigValue`] when `gpu` is not
/// one of `a100-80gb`, `a100-40gb`, `v100`, `t4` (case-insensitive, `_` or `-`).
pub fn parse_gpu_model(gpu: &str) -> entrenar_common::Result<CostModel> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gpu_model_accepts_every_documented_name() {
        for name in [
            "a100-80gb",
            "a100_80gb",
            "A100-80GB",
            "a100-40gb",
            "a100_40gb",
            "v100",
            "V100",
            "t4",
            "T4",
        ] {
            assert!(
                parse_gpu_model(name).is_ok(),
                "documented GPU name {name} must resolve to a cost model"
            );
        }
    }

    #[test]
    fn parse_gpu_model_refuses_unknown_gpu() {
        // Asserting is_ok() on invalid input would lock the defect in; assert
        // the refusal, and that it names the offending value.
        let err = match parse_gpu_model("h100") {
            Ok(_) => panic!("parse_gpu_model must refuse an unknown GPU name"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("h100"),
            "refusal must quote the rejected value, got: {err}"
        );
    }

    #[test]
    fn resolve_strategies_all_expands_to_four() {
        let all = resolve_strategies(&["all".to_string()]);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn resolve_strategies_accepts_every_alias() {
        for alias in [
            "kd",
            "kd-only",
            "kdonly",
            "progressive",
            "prog",
            "attention",
            "attn",
            "combined",
        ] {
            let resolved = resolve_strategies(&[alias.to_string()]);
            assert_eq!(
                resolved.len(),
                1,
                "alias {alias} must resolve to exactly one strategy"
            );
        }
    }

    #[test]
    fn run_compare_refuses_when_no_name_resolves() {
        let cli = entrenar_common::Cli::new().with_verbosity(0);
        let err = match run_compare(&["nonsense".to_string()], 5, &cli) {
            Ok(()) => panic!("run_compare must refuse a strategy list that resolves to nothing"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("No valid strategies specified"),
            "refusal must explain the empty strategy set, got: {err}"
        );
    }

    #[test]
    fn build_constraints_carries_every_bound() {
        let c = build_constraints(Some(10.0), Some(25.0), Some(0.9), Some(40.0));
        assert_eq!(c.max_gpu_hours, Some(10.0));
        assert_eq!(c.max_cost_usd, Some(25.0));
        assert_eq!(c.min_accuracy, Some(0.9));
        assert_eq!(c.max_memory_gb, Some(40.0));
    }
}
