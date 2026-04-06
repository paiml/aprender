//! CLI command handlers.
//!
//! This module contains the business logic for CLI commands,
//! separated from argument parsing for testability.

use crate::prelude::*;
use std::fmt::Write;
use std::path::Path;

/// Handle model commands.
pub fn handle_model_register(
    registry: &Registry,
    name: &str,
    artifact: &Path,
    version: &ModelVersion,
    description: Option<&str>,
) -> Result<ModelId> {
    let data = std::fs::read(artifact)?;
    let card = ModelCard::new(description.unwrap_or_default());
    registry.register_model(name, version, &data, card)
}

/// Format model info for display.
pub fn format_model_info(model: &Model) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Model: {}:{}", model.name, model.version);
    let _ = writeln!(out, "  ID:          {}", model.id);
    let _ = writeln!(out, "  Stage:       {}", model.stage);
    let _ = writeln!(out, "  Created:     {}", model.created_at);
    let _ = writeln!(out, "  Description: {}", model.card.description);
    let _ = writeln!(out, "  Size:        {} bytes", model.content_address.size());
    let _ = writeln!(out, "  Hash:        {}", model.content_address.hash_hex());
    if !model.card.metrics.is_empty() {
        out.push_str("  Metrics:\n");
        for (k, v) in &model.card.metrics {
            let _ = writeln!(out, "    {k}: {v}");
        }
    }
    out
}

/// Handle dataset commands.
pub fn handle_data_register(
    registry: &Registry,
    name: &str,
    data_path: &Path,
    version: &DatasetVersion,
    purpose: Option<&str>,
) -> Result<DatasetId> {
    let content = std::fs::read(data_path)?;
    let datasheet = Datasheet::new(purpose.unwrap_or_default());
    registry.register_dataset(name, version, &content, datasheet)
}

/// Format dataset info for display.
pub fn format_dataset_info(dataset: &Dataset) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Dataset: {}:{}", dataset.name, dataset.version);
    let _ = writeln!(out, "  ID:      {}", dataset.id);
    let _ = writeln!(out, "  Created: {}", dataset.created_at);
    let _ = writeln!(out, "  Purpose: {}", dataset.datasheet.purpose);
    let _ = writeln!(out, "  Size:    {} bytes", dataset.content_address.size());
    let _ = writeln!(out, "  Hash:    {}", dataset.content_address.hash_hex());
    out
}

/// Format recipe info for display.
pub fn format_recipe_info(recipe: &TrainingRecipe) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Recipe: {}:{}", recipe.name, recipe.version);
    let _ = writeln!(out, "  ID:          {}", recipe.id);
    let _ = writeln!(out, "  Description: {}", recipe.description);
    let _ = writeln!(out, "  Created:     {}", recipe.created_at);
    out.push_str("  Hyperparameters:\n");
    let _ = writeln!(out, "    Learning rate: {}", recipe.hyperparameters.learning_rate);
    let _ = writeln!(out, "    Batch size:    {}", recipe.hyperparameters.batch_size);
    let _ = writeln!(out, "    Epochs:        {}", recipe.hyperparameters.epochs);
    out
}

/// Format storage stats for display.
pub fn format_stats(stats: &StorageStats) -> String {
    let mut out = String::new();
    out.push_str("Registry Statistics:\n");
    let _ = writeln!(out, "  Models:   {}", stats.model_count);
    let _ = writeln!(out, "  Datasets: {}", stats.dataset_count);
    let _ = writeln!(out, "  Recipes:  {}", stats.recipe_count);
    let _ = writeln!(out, "  Objects:  {}", stats.object_count);
    let _ = writeln!(out, "  Size:     {} bytes", stats.total_size_bytes);
    out
}

/// Format run info for display.
pub fn format_run_info(run: &ExperimentRun) -> String {
    contract_pre_display_format!();
    let mut out = String::new();
    let _ = writeln!(out, "Run: {}", run.run_id);
    let _ = writeln!(out, "  Status:  {}", run.status);
    let _ = writeln!(out, "  Started: {}", run.started_at);
    if let Some(finished) = run.finished_at {
        let _ = writeln!(out, "  Finished: {finished}");
    }
    if !run.metrics.is_empty() {
        out.push_str("  Final metrics:\n");
        let mut latest: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        for m in &run.metrics {
            latest.insert(&m.name, m.value);
        }
        for (k, v) in latest {
            let _ = writeln!(out, "    {k}: {v}");
        }
    }
    out
}

/// Find best run by metric.
pub fn find_best_run<'a>(
    runs: &'a [ExperimentRun],
    metric: &str,
    minimize: bool,
) -> Option<(&'a ExperimentRun, f64)> {
    contract_pre_configuration!(runs);
    runs.iter()
        .filter(|r| r.status == RunStatus::Completed)
        .filter_map(|r| r.get_metric(metric).map(|v| (r, v)))
        .min_by(|(_, a), (_, b)| {
            if minimize {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, Registry) {
        let dir = TempDir::new().unwrap();
        let config = RegistryConfig::new(dir.path());
        let registry = Registry::open(config).unwrap();
        (dir, registry)
    }

    #[test]
    fn test_handle_model_register() {
        let (dir, registry) = setup();
        let artifact = dir.path().join("model.bin");
        std::fs::write(&artifact, b"weights").unwrap();

        let id = handle_model_register(
            &registry,
            "test",
            &artifact,
            &ModelVersion::new(1, 0, 0),
            Some("Test model"),
        )
        .unwrap();

        assert!(!id.to_string().is_empty());
    }

    #[test]
    fn test_handle_data_register() {
        let (dir, registry) = setup();
        let data = dir.path().join("data.csv");
        std::fs::write(&data, b"a,b,c").unwrap();

        let id = handle_data_register(
            &registry,
            "test-data",
            &data,
            &DatasetVersion::new(1, 0, 0),
            Some("Test data"),
        )
        .unwrap();

        assert!(!id.to_string().is_empty());
    }

    #[test]
    fn test_format_stats() {
        let stats = StorageStats {
            model_count: 5,
            dataset_count: 3,
            recipe_count: 2,
            object_count: 10,
            total_size_bytes: 1024,
        };
        let out = format_stats(&stats);
        assert!(out.contains("Models:   5"));
        assert!(out.contains("Datasets: 3"));
    }

    #[test]
    fn test_find_best_run_maximize() {
        let runs = vec![
            create_run_with_metric("auc", 0.8),
            create_run_with_metric("auc", 0.95),
            create_run_with_metric("auc", 0.85),
        ];
        let best = find_best_run(&runs, "auc", false);
        assert!(best.is_some());
        assert!((best.unwrap().1 - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_find_best_run_minimize() {
        let runs = vec![
            create_run_with_metric("loss", 0.5),
            create_run_with_metric("loss", 0.1),
            create_run_with_metric("loss", 0.3),
        ];
        let best = find_best_run(&runs, "loss", true);
        assert!(best.is_some());
        assert!((best.unwrap().1 - 0.1).abs() < 1e-9);
    }

    fn create_run_with_metric(name: &str, value: f64) -> ExperimentRun {
        let mut run = ExperimentRun::new(Hyperparameters::default());
        run.log_metric(name, value, 0);
        run.complete();
        run
    }

    #[test]
    fn test_format_model_info() {
        let (dir, registry) = setup();
        let artifact = dir.path().join("m.bin");
        std::fs::write(&artifact, b"data").unwrap();

        let card = ModelCard::builder().description("Test").metrics([("acc", 0.9)]).build();
        registry.register_model("fmt-test", &ModelVersion::new(1, 0, 0), b"data", card).unwrap();

        let model = registry.get_model("fmt-test", &ModelVersion::new(1, 0, 0)).unwrap();
        let out = format_model_info(&model);
        assert!(out.contains("fmt-test:1.0.0"));
        assert!(out.contains("Stage:"));
        assert!(out.contains("acc: 0.9"));
    }

    #[test]
    fn test_format_dataset_info() {
        let (_dir, registry) = setup();
        let datasheet = Datasheet::new("Test purpose");
        registry
            .register_dataset("fmt-data", &DatasetVersion::new(1, 0, 0), b"csv", datasheet)
            .unwrap();

        let ds = registry.get_dataset("fmt-data", &DatasetVersion::new(1, 0, 0)).unwrap();
        let out = format_dataset_info(&ds);
        assert!(out.contains("fmt-data:1.0.0"));
        assert!(out.contains("Purpose: Test purpose"));
    }

    #[test]
    fn test_format_recipe_info() {
        let (_dir, registry) = setup();
        let recipe = TrainingRecipe::builder()
            .name("fmt-recipe")
            .version(RecipeVersion::new(1, 0, 0))
            .description("Test recipe")
            .hyperparameters(
                Hyperparameters::builder().learning_rate(0.01).batch_size(64).epochs(5).build(),
            )
            .build();
        registry.register_recipe(&recipe).unwrap();

        let r = registry.get_recipe("fmt-recipe", &RecipeVersion::new(1, 0, 0)).unwrap();
        let out = format_recipe_info(&r);
        assert!(out.contains("fmt-recipe:1.0.0"));
        assert!(out.contains("Batch size:    64"));
    }

    #[test]
    fn test_format_run_info() {
        let mut run = ExperimentRun::new(Hyperparameters::default());
        run.log_metric("loss", 0.5, 0);
        run.log_metric("loss", 0.2, 100);
        run.complete();

        let out = format_run_info(&run);
        assert!(out.contains("Status:  completed"));
        assert!(out.contains("loss: 0.2"));
    }

    #[test]
    fn test_find_best_run_no_matches() {
        let runs = vec![create_run_with_metric("auc", 0.8)];
        let best = find_best_run(&runs, "nonexistent", false);
        assert!(best.is_none());
    }

    #[test]
    fn test_find_best_run_empty() {
        let runs: Vec<ExperimentRun> = vec![];
        let best = find_best_run(&runs, "auc", false);
        assert!(best.is_none());
    }
}
