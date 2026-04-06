//! Pacha CLI - Model, Data and Recipe Registry

// Clippy configuration for binary
#![allow(clippy::double_ended_iterator_last)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::map_unwrap_or)]

use clap::{Parser, Subcommand};
use pacha::prelude::*;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "pacha")]
#[command(author, version, about = "Model, Data and Recipe Registry", long_about = None)]
struct Cli {
    /// Registry path (default: ~/.pacha)
    #[arg(long, global = true)]
    registry: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Model registry operations
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
    /// Dataset registry operations
    Data {
        #[command(subcommand)]
        action: DataAction,
    },
    /// Recipe registry operations
    Recipe {
        #[command(subcommand)]
        action: RecipeAction,
    },
    /// Experiment run operations
    Run {
        #[command(subcommand)]
        action: RunAction,
    },
    /// Show registry statistics
    Stats,
    /// Initialize a new registry
    Init,
}

#[derive(Subcommand)]
enum ModelAction {
    /// Register a new model
    Register {
        /// Model name
        name: String,
        /// Path to artifact file
        artifact: PathBuf,
        /// Model version (e.g., 1.0.0)
        #[arg(long, short)]
        version: String,
        /// Model description
        #[arg(long, short)]
        description: Option<String>,
    },
    /// List models
    List {
        /// Model name (optional, lists versions if provided)
        name: Option<String>,
    },
    /// Get model details
    Get {
        /// Model name
        name: String,
        /// Model version
        #[arg(long, short)]
        version: String,
    },
    /// Download model artifact
    Download {
        /// Model name
        name: String,
        /// Model version
        #[arg(long, short)]
        version: String,
        /// Output path
        #[arg(long, short)]
        output: PathBuf,
    },
    /// Show model lineage
    Lineage {
        /// Model name
        name: String,
        /// Model version
        #[arg(long, short)]
        version: String,
    },
    /// Transition model stage
    Stage {
        /// Model name
        name: String,
        /// Model version
        #[arg(long, short)]
        version: String,
        /// Target stage (development, staging, production, archived)
        #[arg(long, short)]
        target: String,
    },
}

#[derive(Subcommand)]
enum DataAction {
    /// Register a new dataset
    Register {
        /// Dataset name
        name: String,
        /// Path to data file
        data: PathBuf,
        /// Dataset version (e.g., 1.0.0)
        #[arg(long, short)]
        version: String,
        /// Dataset purpose/description
        #[arg(long, short)]
        purpose: Option<String>,
    },
    /// List datasets
    List {
        /// Dataset name (optional)
        name: Option<String>,
    },
    /// Get dataset details
    Get {
        /// Dataset name
        name: String,
        /// Dataset version
        #[arg(long, short)]
        version: String,
    },
    /// Download dataset
    Download {
        /// Dataset name
        name: String,
        /// Dataset version
        #[arg(long, short)]
        version: String,
        /// Output path
        #[arg(long, short)]
        output: PathBuf,
    },
}

#[derive(Subcommand)]
enum RecipeAction {
    /// Register a new recipe
    Register {
        /// Path to recipe TOML file
        recipe: PathBuf,
    },
    /// List recipes
    List {
        /// Recipe name (optional)
        name: Option<String>,
    },
    /// Get recipe details
    Get {
        /// Recipe name
        name: String,
        /// Recipe version
        #[arg(long, short)]
        version: String,
    },
    /// Validate a recipe
    Validate {
        /// Recipe name
        name: String,
        /// Recipe version
        #[arg(long, short)]
        version: String,
    },
}

#[derive(Subcommand)]
enum RunAction {
    /// List experiment runs
    List {
        /// Recipe name
        recipe: String,
        /// Recipe version
        #[arg(long, short)]
        version: String,
    },
    /// Get run details
    Get {
        /// Run ID
        id: String,
    },
    /// Compare runs
    Compare {
        /// Run IDs to compare
        ids: Vec<String>,
    },
    /// Find best run by metric
    Best {
        /// Recipe name
        recipe: String,
        /// Recipe version
        #[arg(long, short)]
        version: String,
        /// Metric name
        #[arg(long, short)]
        metric: String,
        /// Minimize metric (default: maximize)
        #[arg(long)]
        minimize: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("Error: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

fn run(cli: Cli) -> pacha::Result<()> {
    let config = cli.registry.map(RegistryConfig::new).unwrap_or_default();

    match cli.command {
        Commands::Init => {
            let registry = Registry::open(config)?;
            println!("Registry initialized at: {}", registry.config().base_path.display());
        }
        Commands::Stats => {
            let registry = Registry::open(config)?;
            let stats = registry.storage_stats()?;
            println!("Registry Statistics:");
            println!("  Models:   {}", stats.model_count);
            println!("  Datasets: {}", stats.dataset_count);
            println!("  Recipes:  {}", stats.recipe_count);
            println!("  Objects:  {}", stats.object_count);
            println!("  Size:     {} bytes", stats.total_size_bytes);
        }
        Commands::Model { action } => handle_model(config, action)?,
        Commands::Data { action } => handle_data(config, action)?,
        Commands::Recipe { action } => handle_recipe(config, action)?,
        Commands::Run { action } => handle_run(config, action)?,
    }

    Ok(())
}

fn handle_model(config: RegistryConfig, action: ModelAction) -> pacha::Result<()> {
    let registry = Registry::open(config)?;

    match action {
        ModelAction::Register { name, artifact, version, description } => {
            model_register(&registry, &name, &artifact, &version, description)?;
        }
        ModelAction::List { name } => model_list(&registry, name.as_deref())?,
        ModelAction::Get { name, version } => model_get(&registry, &name, &version)?,
        ModelAction::Download { name, version, output } => {
            model_download(&registry, &name, &version, &output)?;
        }
        ModelAction::Lineage { name, version } => {
            handle_model_lineage(&registry, &name, &version)?;
        }
        ModelAction::Stage { name, version, target } => {
            model_stage(&registry, &name, &version, &target)?;
        }
    }

    Ok(())
}

fn model_register(
    registry: &Registry,
    name: &str,
    artifact: &PathBuf,
    version: &str,
    description: Option<String>,
) -> pacha::Result<()> {
    let version: ModelVersion = version.parse()?;
    let data = std::fs::read(artifact)?;
    let card = ModelCard::new(description.unwrap_or_default());
    let id = registry.register_model(name, &version, &data, card)?;
    println!("Registered model: {name}:{version} ({id})");
    Ok(())
}

fn model_list(registry: &Registry, name: Option<&str>) -> pacha::Result<()> {
    if let Some(name) = name {
        let versions = registry.list_model_versions(name)?;
        println!("Versions of '{name}':");
        for v in versions {
            println!("  {v}");
        }
    } else {
        let models = registry.list_models()?;
        println!("Models:");
        for m in models {
            println!("  {m}");
        }
    }
    Ok(())
}

fn model_get(registry: &Registry, name: &str, version: &str) -> pacha::Result<()> {
    let version: ModelVersion = version.parse()?;
    let model = registry.get_model(name, &version)?;
    println!("Model: {}:{}", model.name, model.version);
    println!("  ID:          {}", model.id);
    println!("  Stage:       {}", model.stage);
    println!("  Created:     {}", model.created_at);
    println!("  Description: {}", model.card.description);
    println!("  Size:        {} bytes", model.content_address.size());
    println!("  Hash:        {}", model.content_address.hash_hex());
    if !model.card.metrics.is_empty() {
        println!("  Metrics:");
        for (k, v) in &model.card.metrics {
            println!("    {k}: {v}");
        }
    }
    Ok(())
}

fn model_download(
    registry: &Registry,
    name: &str,
    version: &str,
    output: &PathBuf,
) -> pacha::Result<()> {
    let version: ModelVersion = version.parse()?;
    let data = registry.get_model_artifact(name, &version)?;
    std::fs::write(output, &data)?;
    println!("Downloaded {name}:{version} to {}", output.display());
    Ok(())
}

fn model_stage(registry: &Registry, name: &str, version: &str, target: &str) -> pacha::Result<()> {
    let version: ModelVersion = version.parse()?;
    let target_stage: ModelStage = target.parse()?;
    registry.transition_model_stage(name, &version, target_stage)?;
    println!("Transitioned {name}:{version} to {target_stage}");
    Ok(())
}

fn handle_model_lineage(registry: &Registry, name: &str, version: &str) -> pacha::Result<()> {
    let version: ModelVersion = version.parse()?;
    let model = registry.get_model(name, &version)?;
    let lineage = registry.get_model_lineage(&model.id)?;

    println!("Lineage for {name}:{version}");
    println!("{}", "=".repeat(40));
    println!();

    if lineage.node_count() == 0 {
        println!("No lineage information available.");
        return Ok(());
    }

    let target_idx = lineage.find_node(&model.id);

    println!("Models ({} total):", lineage.node_count());
    for (idx, node) in lineage.nodes.iter().enumerate() {
        let marker = if Some(idx) == target_idx { " <-- current" } else { "" };
        println!("  [{}] {}:{}{}", idx, node.model_name, node.model_version, marker);
    }
    println!();

    if lineage.edge_count() == 0 {
        println!("No derivation relationships recorded.");
        return Ok(());
    }

    print_derivation_history(&lineage);
    if let Some(idx) = target_idx {
        print_relatives(&lineage, idx, name, &version);
    }

    Ok(())
}

fn print_derivation_history(lineage: &pacha::lineage::LineageGraph) {
    println!("Derivation History ({} relationships):", lineage.edge_count());
    for edge in &lineage.edges {
        let from = &lineage.nodes[edge.from_idx];
        let to = &lineage.nodes[edge.to_idx];
        let edge_type = match &edge.edge {
            pacha::lineage::ModelLineageEdge::FineTuned { .. } => "fine-tuned from".to_string(),
            pacha::lineage::ModelLineageEdge::Distilled { .. } => "distilled from".to_string(),
            pacha::lineage::ModelLineageEdge::Merged { .. } => "merged from".to_string(),
            pacha::lineage::ModelLineageEdge::Quantized { quantization, .. } => {
                format!("quantized ({quantization}) from")
            }
            pacha::lineage::ModelLineageEdge::Pruned { sparsity, .. } => {
                format!("pruned ({:.0}% sparse) from", sparsity * 100.0)
            }
        };
        println!(
            "  {}:{} --> {} --> {}:{}",
            from.model_name, from.model_version, edge_type, to.model_name, to.model_version
        );
    }
    println!();
}

fn print_relatives(
    lineage: &pacha::lineage::LineageGraph,
    idx: usize,
    name: &str,
    version: &ModelVersion,
) {
    let ancestors = lineage.ancestors(idx);
    if !ancestors.is_empty() {
        println!("Direct ancestors of {name}:{version}:");
        for a_idx in ancestors {
            let a = &lineage.nodes[a_idx];
            println!("  - {}:{}", a.model_name, a.model_version);
        }
        println!();
    }

    let descendants = lineage.descendants(idx);
    if !descendants.is_empty() {
        println!("Derived models from {name}:{version}:");
        for d_idx in descendants {
            let d = &lineage.nodes[d_idx];
            println!("  - {}:{}", d.model_name, d.model_version);
        }
    }
}

fn handle_data(config: RegistryConfig, action: DataAction) -> pacha::Result<()> {
    let registry = Registry::open(config)?;

    match action {
        DataAction::Register { name, data, version, purpose } => {
            data_register(&registry, &name, &data, &version, purpose)?;
        }
        DataAction::List { name } => data_list(&registry, name.as_deref())?,
        DataAction::Get { name, version } => data_get(&registry, &name, &version)?,
        DataAction::Download { name, version, output } => {
            data_download(&registry, &name, &version, &output)?;
        }
    }

    Ok(())
}

fn data_register(
    registry: &Registry,
    name: &str,
    data: &PathBuf,
    version: &str,
    purpose: Option<String>,
) -> pacha::Result<()> {
    let version: DatasetVersion = version.parse()?;
    let content = std::fs::read(data)?;
    let datasheet = Datasheet::new(purpose.unwrap_or_default());
    let id = registry.register_dataset(name, &version, &content, datasheet)?;
    println!("Registered dataset: {name}:{version} ({id})");
    Ok(())
}

fn data_list(registry: &Registry, name: Option<&str>) -> pacha::Result<()> {
    if let Some(dataset_name) = name {
        let versions = registry.list_dataset_versions(dataset_name)?;
        if versions.is_empty() {
            println!("No versions found for dataset: {dataset_name}");
        } else {
            println!("Versions of {dataset_name}:");
            for v in versions {
                println!("  {v}");
            }
        }
    } else {
        let datasets = registry.list_datasets()?;
        println!("Datasets:");
        for d in datasets {
            println!("  {d}");
        }
    }
    Ok(())
}

fn data_get(registry: &Registry, name: &str, version: &str) -> pacha::Result<()> {
    let version: DatasetVersion = version.parse()?;
    let dataset = registry.get_dataset(name, &version)?;
    println!("Dataset: {}:{}", dataset.name, dataset.version);
    println!("  ID:      {}", dataset.id);
    println!("  Created: {}", dataset.created_at);
    println!("  Purpose: {}", dataset.datasheet.purpose);
    println!("  Size:    {} bytes", dataset.content_address.size());
    println!("  Hash:    {}", dataset.content_address.hash_hex());
    Ok(())
}

fn data_download(
    registry: &Registry,
    name: &str,
    version: &str,
    output: &PathBuf,
) -> pacha::Result<()> {
    let version: DatasetVersion = version.parse()?;
    let data = registry.get_dataset_data(name, &version)?;
    std::fs::write(output, &data)?;
    println!("Downloaded {name}:{version} to {}", output.display());
    Ok(())
}

fn handle_recipe(config: RegistryConfig, action: RecipeAction) -> pacha::Result<()> {
    let registry = Registry::open(config)?;

    match action {
        RecipeAction::Register { recipe: path } => {
            let content = std::fs::read_to_string(&path)?;
            let recipe: TrainingRecipe = toml::from_str(&content)?;
            let id = registry.register_recipe(&recipe)?;
            println!("Registered recipe: {}:{} ({})", recipe.name, recipe.version, id);
        }
        RecipeAction::List { name } => {
            handle_recipe_list(&registry, name.as_deref())?;
        }
        RecipeAction::Get { name, version } => {
            handle_recipe_get(&registry, &name, &version)?;
        }
        RecipeAction::Validate { name, version } => {
            handle_recipe_validate(&registry, &name, &version)?;
        }
    }

    Ok(())
}

fn handle_recipe_list(registry: &Registry, name: Option<&str>) -> pacha::Result<()> {
    if let Some(recipe_name) = name {
        let versions = registry.list_recipe_versions(recipe_name)?;
        if versions.is_empty() {
            println!("No versions found for recipe: {recipe_name}");
        } else {
            println!("Versions of {recipe_name}:");
            for v in versions {
                println!("  {v}");
            }
        }
    } else {
        let recipes = registry.list_recipes()?;
        println!("Recipes:");
        for r in recipes {
            println!("  {r}");
        }
    }
    Ok(())
}

fn handle_recipe_get(registry: &Registry, name: &str, version: &str) -> pacha::Result<()> {
    let version: RecipeVersion = version.parse()?;
    let recipe = registry.get_recipe(name, &version)?;
    println!("Recipe: {}:{}", recipe.name, recipe.version);
    println!("  ID:          {}", recipe.id);
    println!("  Description: {}", recipe.description);
    println!("  Created:     {}", recipe.created_at);
    println!("  Hyperparameters:");
    println!("    Learning rate: {}", recipe.hyperparameters.learning_rate);
    println!("    Batch size:    {}", recipe.hyperparameters.batch_size);
    println!("    Epochs:        {}", recipe.hyperparameters.epochs);
    Ok(())
}

fn handle_recipe_validate(registry: &Registry, name: &str, version: &str) -> pacha::Result<()> {
    let version: RecipeVersion = version.parse()?;
    let recipe = registry.get_recipe(name, &version)?;
    println!("Validating recipe: {name}:{version}...");
    if recipe.hyperparameters.learning_rate <= 0.0 {
        println!("  Warning: learning_rate should be positive");
    }
    if recipe.hyperparameters.batch_size == 0 {
        println!("  Warning: batch_size should be > 0");
    }
    println!("  Validation complete");
    Ok(())
}

fn handle_run(config: RegistryConfig, action: RunAction) -> pacha::Result<()> {
    let registry = Registry::open(config)?;

    match action {
        RunAction::List { recipe, version } => {
            handle_run_list(&registry, &recipe, &version)?;
        }
        RunAction::Get { id } => {
            handle_run_get(&registry, &id)?;
        }
        RunAction::Compare { ids } => {
            handle_run_compare(&registry, &ids)?;
        }
        RunAction::Best { recipe, version, metric, minimize } => {
            handle_run_best(&registry, &recipe, &version, &metric, minimize)?;
        }
    }

    Ok(())
}

fn handle_run_list(registry: &Registry, recipe: &str, version: &str) -> pacha::Result<()> {
    let version: RecipeVersion = version.parse()?;
    let recipe_ref = RecipeReference::new(recipe.to_string(), version);
    let runs = registry.list_runs(&recipe_ref)?;
    println!("Runs for {recipe_ref}:");
    for run in runs {
        println!(
            "  {} - {} ({})",
            run.run_id,
            run.status,
            run.started_at.format("%Y-%m-%d %H:%M:%S")
        );
    }
    Ok(())
}

fn handle_run_get(registry: &Registry, id: &str) -> pacha::Result<()> {
    let run_id: RunId =
        id.parse().map_err(|_| pacha::PachaError::Validation("invalid run id".to_string()))?;
    let run = registry.get_run(&run_id)?;
    println!("Run: {}", run.run_id);
    println!("  Status:  {}", run.status);
    println!("  Started: {}", run.started_at);
    if let Some(finished) = run.finished_at {
        println!("  Finished: {finished}");
    }
    if !run.metrics.is_empty() {
        println!("  Final metrics:");
        let mut latest: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        for m in &run.metrics {
            latest.insert(&m.name, m.value);
        }
        for (k, v) in latest {
            println!("    {k}: {v}");
        }
    }
    Ok(())
}

fn handle_run_compare(registry: &Registry, ids: &[String]) -> pacha::Result<()> {
    if ids.len() < 2 {
        return Err(pacha::PachaError::Validation(
            "Need at least 2 run IDs to compare.".to_string(),
        )
        .into());
    }

    let mut runs = Vec::new();
    for id in ids {
        let run_id: RunId =
            id.parse().map_err(|_| pacha::PachaError::Validation("invalid run id".to_string()))?;
        let run = registry.get_run(&run_id)?;
        runs.push(run);
    }

    let metric_names = collect_metric_names(&runs);

    print_comparison_header(&runs);
    print_comparison_rows(&runs, &metric_names);
    print_best_values(&runs, &metric_names);

    Ok(())
}

fn collect_metric_names(runs: &[pacha::experiment::ExperimentRun]) -> Vec<String> {
    let mut all_metrics: std::collections::HashSet<String> = std::collections::HashSet::new();
    for run in runs {
        for m in &run.metrics {
            all_metrics.insert(m.name.clone());
        }
    }
    let mut names: Vec<_> = all_metrics.into_iter().collect();
    names.sort();
    names
}

fn print_comparison_header(runs: &[pacha::experiment::ExperimentRun]) {
    println!("Run Comparison");
    println!("{}", "=".repeat(60));
    println!();

    print!("{:<20}", "Metric");
    for (i, _) in runs.iter().enumerate() {
        print!(" {:>15}", format!("Run {}", i + 1));
    }
    println!();
    println!("{}", "-".repeat(20 + runs.len() * 16));

    print!("{:<20}", "ID (short)");
    for run in runs {
        let short_id = run.run_id.to_string().chars().take(8).collect::<String>();
        print!(" {:>15}", short_id);
    }
    println!();

    print!("{:<20}", "Status");
    for run in runs {
        print!(" {:>15}", run.status);
    }
    println!();

    print!("{:<20}", "Duration");
    for run in runs {
        let duration = if let Some(finished) = run.finished_at {
            let secs = (finished - run.started_at).num_seconds();
            format!("{}s", secs)
        } else {
            "ongoing".to_string()
        };
        print!(" {:>15}", duration);
    }
    println!();
    println!("{}", "-".repeat(20 + runs.len() * 16));
}

fn print_comparison_rows(runs: &[pacha::experiment::ExperimentRun], metric_names: &[String]) {
    for metric_name in metric_names {
        print!("{:<20}", metric_name);
        for run in runs {
            let value = run
                .metrics
                .iter()
                .filter(|m| &m.name == metric_name)
                .last()
                .map(|m| format!("{:.4}", m.value))
                .unwrap_or_else(|| "-".to_string());
            print!(" {:>15}", value);
        }
        println!();
    }

    if metric_names.is_empty() {
        println!("(no metrics recorded)");
    }
    println!();
}

fn print_best_values(runs: &[pacha::experiment::ExperimentRun], metric_names: &[String]) {
    if metric_names.is_empty() {
        return;
    }
    println!("Best values (assuming higher is better, except for 'loss'):");
    for metric_name in metric_names {
        let values: Vec<Option<f64>> = runs
            .iter()
            .map(|run| {
                run.metrics.iter().filter(|m| &m.name == metric_name).last().map(|m| m.value)
            })
            .collect();

        let lower_is_better = metric_name.contains("loss") || metric_name.contains("error");
        let best_idx = values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|val| (i, val)))
            .min_by(|a, b| {
                if lower_is_better {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                }
            })
            .map(|(i, _)| i);

        if let Some(idx) = best_idx {
            if let Some(val) = values[idx] {
                println!("  {}: Run {} ({:.4})", metric_name, idx + 1, val);
            }
        }
    }
}

fn handle_run_best(
    registry: &Registry,
    recipe: &str,
    version: &str,
    metric: &str,
    minimize: bool,
) -> pacha::Result<()> {
    let version: RecipeVersion = version.parse()?;
    let recipe_ref = RecipeReference::new(recipe.to_string(), version);
    let runs = registry.list_runs(&recipe_ref)?;

    let best = runs
        .iter()
        .filter(|r| r.status == RunStatus::Completed)
        .filter_map(|r| r.get_metric(metric).map(|v| (r, v)))
        .min_by(|(_, a), (_, b)| {
            if minimize {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

    if let Some((run, value)) = best {
        println!("Best run by {metric}:");
        println!("  ID:    {}", run.run_id);
        println!("  Value: {value}");
    } else {
        println!("No completed runs found with metric '{metric}'");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse() {
        let cli = Cli::try_parse_from(["pacha", "stats"]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_model_register() {
        let cli = Cli::try_parse_from([
            "pacha",
            "model",
            "register",
            "test-model",
            "model.apr",
            "-v",
            "1.0.0",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_model_list() {
        let cli = Cli::try_parse_from(["pacha", "model", "list"]);
        assert!(cli.is_ok());
    }
}
