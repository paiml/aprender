//! Federated split coordination CLI commands.

use std::path::{Path, PathBuf};

use clap::Subcommand;

use super::basic::load_dataset;
use crate::{
    federated::{
        FederatedSplitCoordinator, FederatedSplitStrategy, NodeSplitInstruction, NodeSplitManifest,
    },
    split::DatasetSplit,
    Dataset,
};

/// Federated split coordination commands.
#[derive(Subcommand)]
pub enum FedCommands {
    /// Generate a manifest from local dataset (runs on each node)
    Manifest {
        /// Input dataset file
        input: PathBuf,
        /// Output manifest file
        #[arg(short, long)]
        output: PathBuf,
        /// Unique node identifier
        #[arg(short, long)]
        node_id: String,
        /// Training set ratio
        #[arg(short = 'r', long, default_value = "0.8")]
        train_ratio: f64,
        /// Random seed for reproducibility
        #[arg(short, long, default_value = "42")]
        seed: u64,
        /// Output format (json, binary)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Create a split plan from manifests (runs on coordinator)
    Plan {
        /// Manifest files from all nodes
        #[arg(required = true)]
        manifests: Vec<PathBuf>,
        /// Output plan file
        #[arg(short, long)]
        output: PathBuf,
        /// Split strategy (local, proportional, stratified)
        #[arg(short, long, default_value = "local")]
        strategy: String,
        /// Training set ratio
        #[arg(short = 'r', long, default_value = "0.8")]
        train_ratio: f64,
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Column for stratification (required for stratified strategy)
        #[arg(long)]
        stratify_column: Option<String>,
        /// Output format (json, binary)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Execute local split based on plan (runs on each node)
    Split {
        /// Input dataset file
        input: PathBuf,
        /// Split plan file
        #[arg(short, long)]
        plan: PathBuf,
        /// This node's ID
        #[arg(short, long)]
        node_id: String,
        /// Output training set file
        #[arg(long)]
        train_output: PathBuf,
        /// Output test set file
        #[arg(long)]
        test_output: PathBuf,
        /// Output validation set file (optional)
        #[arg(long)]
        validation_output: Option<PathBuf>,
    },
    /// Verify global split quality from manifests (runs on coordinator)
    Verify {
        /// Manifest files from all nodes
        #[arg(required = true)]
        manifests: Vec<PathBuf>,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
}

/// Parse federated split strategy from string.
pub(crate) fn parse_fed_strategy(
    strategy: &str,
    train_ratio: f64,
    seed: u64,
    stratify_column: Option<&str>,
) -> Option<FederatedSplitStrategy> {
    match strategy.to_lowercase().as_str() {
        "local" | "local-seed" => Some(FederatedSplitStrategy::LocalWithSeed { seed, train_ratio }),
        "proportional" | "iid" => Some(FederatedSplitStrategy::ProportionalIID { train_ratio }),
        "stratified" => {
            let column = stratify_column.unwrap_or("label").to_string();
            Some(FederatedSplitStrategy::GlobalStratified {
                label_column: column,
                target_distribution: std::collections::HashMap::new(),
            })
        }
        _ => None,
    }
}

/// Generate a manifest from local dataset.
pub(crate) fn cmd_fed_manifest(
    input: &Path,
    output: &Path,
    node_id: &str,
    train_ratio: f64,
    seed: u64,
    format: &str,
) -> crate::Result<()> {
    let dataset = load_dataset(input)?;

    // Create a split to generate the manifest
    let split =
        DatasetSplit::from_ratios(&dataset, train_ratio, 1.0 - train_ratio, None, Some(seed))?;

    let manifest = NodeSplitManifest::from_split(node_id, &split);

    match format {
        "binary" | "bin" => {
            let bytes =
                rmp_serde::to_vec(&manifest).map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, bytes).map_err(|e| crate::Error::io(e, output))?;
        }
        _ => {
            let json = serde_json::to_string_pretty(&manifest)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, json).map_err(|e| crate::Error::io(e, output))?;
        }
    }

    println!(
        "Created manifest for node '{}' ({} rows) -> {}",
        node_id,
        dataset.len(),
        output.display()
    );

    Ok(())
}

/// Load a manifest from a file.
pub(crate) fn load_manifest(path: &Path) -> crate::Result<NodeSplitManifest> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "bin" | "binary" => {
            let bytes = std::fs::read(path).map_err(|e| crate::Error::io(e, path))?;
            rmp_serde::from_slice(&bytes)
                .map_err(|e| crate::Error::Format(format!("Invalid manifest binary: {}", e)))
        }
        _ => {
            let json = std::fs::read_to_string(path).map_err(|e| crate::Error::io(e, path))?;
            serde_json::from_str(&json)
                .map_err(|e| crate::Error::Format(format!("Invalid manifest JSON: {}", e)))
        }
    }
}

/// Create a split plan from manifests.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_fed_plan(
    manifests: &[PathBuf],
    output: &Path,
    strategy: &str,
    train_ratio: f64,
    seed: u64,
    stratify_column: Option<&str>,
    format: &str,
) -> crate::Result<()> {
    if manifests.is_empty() {
        return Err(crate::Error::invalid_config("No manifests provided"));
    }

    let loaded: Vec<NodeSplitManifest> = manifests
        .iter()
        .map(|p| load_manifest(p))
        .collect::<Result<Vec<_>, _>>()?;

    let strategy =
        parse_fed_strategy(strategy, train_ratio, seed, stratify_column).ok_or_else(|| {
            crate::Error::invalid_config(format!(
                "Unknown strategy: {}. Use 'local', 'proportional', or 'stratified'",
                strategy
            ))
        })?;

    let coordinator = FederatedSplitCoordinator::new(strategy);
    let instructions = coordinator.compute_split_plan(&loaded)?;

    match format {
        "binary" | "bin" => {
            let bytes = rmp_serde::to_vec(&instructions)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, bytes).map_err(|e| crate::Error::io(e, output))?;
        }
        _ => {
            let json = serde_json::to_string_pretty(&instructions)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, json).map_err(|e| crate::Error::io(e, output))?;
        }
    }

    println!(
        "Created split plan for {} nodes -> {}",
        instructions.len(),
        output.display()
    );

    Ok(())
}

/// Load a split plan from a file.
pub(crate) fn load_plan(path: &Path) -> crate::Result<Vec<NodeSplitInstruction>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "bin" | "binary" => {
            let bytes = std::fs::read(path).map_err(|e| crate::Error::io(e, path))?;
            rmp_serde::from_slice(&bytes)
                .map_err(|e| crate::Error::Format(format!("Invalid plan binary: {}", e)))
        }
        _ => {
            let json = std::fs::read_to_string(path).map_err(|e| crate::Error::io(e, path))?;
            serde_json::from_str(&json)
                .map_err(|e| crate::Error::Format(format!("Invalid plan JSON: {}", e)))
        }
    }
}

/// Execute local split based on plan.
pub(crate) fn cmd_fed_split(
    input: &Path,
    plan: &Path,
    node_id: &str,
    train_output: &Path,
    test_output: &Path,
    validation_output: Option<&PathBuf>,
) -> crate::Result<()> {
    let dataset = load_dataset(input)?;
    let instructions = load_plan(plan)?;

    // Find instruction for this node
    let instruction = instructions
        .iter()
        .find(|i| i.node_id == node_id)
        .ok_or_else(|| {
            crate::Error::invalid_config(format!(
                "No instruction found for node '{}' in plan",
                node_id
            ))
        })?;

    // Execute the split
    let split = FederatedSplitCoordinator::execute_local_split(&dataset, instruction)?;

    // Save outputs
    split.train.to_parquet(train_output)?;
    split.test.to_parquet(test_output)?;

    if let (Some(val_output), Some(val_data)) = (validation_output, &split.validation) {
        val_data.to_parquet(val_output)?;
    }

    println!(
        "Split executed for node '{}': {} train, {} test{}",
        node_id,
        split.train.len(),
        split.test.len(),
        split
            .validation
            .as_ref()
            .map_or(String::new(), |v| format!(", {} validation", v.len()))
    );

    Ok(())
}

/// Verify global split quality from manifests.
pub(crate) fn cmd_fed_verify(manifests: &[PathBuf], format: &str) -> crate::Result<()> {
    if manifests.is_empty() {
        return Err(crate::Error::invalid_config("No manifests provided"));
    }

    let loaded: Vec<NodeSplitManifest> = manifests
        .iter()
        .map(|p| load_manifest(p))
        .collect::<Result<Vec<_>, _>>()?;

    let report = FederatedSplitCoordinator::verify_global_split(&loaded)?;

    if format == "json" {
        let json = serde_json::json!({
            "total_rows": report.total_rows,
            "total_train_rows": report.total_train_rows,
            "total_test_rows": report.total_test_rows,
            "total_validation_rows": report.total_validation_rows,
            "effective_train_ratio": report.effective_train_ratio,
            "effective_test_ratio": report.effective_test_ratio,
            "effective_validation_ratio": report.effective_validation_ratio,
            "quality_passed": report.quality_passed,
            "issues": report.issues.iter().map(|i| format!("{:?}", i)).collect::<Vec<_>>(),
            "node_summaries": report.node_summaries.iter().map(|n| {
                serde_json::json!({
                    "node_id": n.node_id,
                    "contribution_ratio": n.contribution_ratio,
                    "train_ratio": n.train_ratio,
                    "test_ratio": n.test_ratio,
                })
            }).collect::<Vec<_>>()
        });

        let json_str =
            serde_json::to_string_pretty(&json).map_err(|e| crate::Error::Format(e.to_string()))?;
        println!("{}", json_str);
    } else {
        // Text format
        println!("Federated Split Verification");
        println!("============================");
        println!();
        println!("Global Statistics:");
        println!("  Total rows:        {}", report.total_rows);
        println!(
            "  Train rows:        {} ({:.1}%)",
            report.total_train_rows,
            report.effective_train_ratio * 100.0
        );
        println!(
            "  Test rows:         {} ({:.1}%)",
            report.total_test_rows,
            report.effective_test_ratio * 100.0
        );
        if let Some(val) = report.total_validation_rows {
            println!(
                "  Validation rows:   {} ({:.1}%)",
                val,
                report.effective_validation_ratio.unwrap_or(0.0) * 100.0
            );
        }
        println!();

        println!("Node Summaries:");
        println!(
            "{:<15} {:>12} {:>10} {:>10}",
            "NODE", "CONTRIBUTION", "TRAIN", "TEST"
        );
        println!("{}", "-".repeat(50));

        for summary in &report.node_summaries {
            println!(
                "{:<15} {:>11.1}% {:>9.1}% {:>9.1}%",
                summary.node_id,
                summary.contribution_ratio * 100.0,
                summary.train_ratio * 100.0,
                summary.test_ratio * 100.0
            );
        }

        println!();
        if report.quality_passed {
            println!("\u{2713} Quality check passed");
        } else {
            println!("\u{26A0} Quality issues detected:");
            for issue in &report.issues {
                println!("  - {:?}", issue);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::redundant_clone,
    clippy::cast_lossless,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::needless_late_init,
    clippy::redundant_pattern_matching
)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;
    use crate::ArrowDataset;

    fn create_test_parquet(path: &Path, rows: usize) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));
    }

    #[test]
    fn test_parse_fed_strategy() {
        assert!(matches!(
            parse_fed_strategy("local", 0.8, 42, None),
            Some(_)
        ));
        assert!(matches!(
            parse_fed_strategy("proportional", 0.8, 42, None),
            Some(_)
        ));
        assert!(matches!(
            parse_fed_strategy("stratified", 0.8, 42, Some("label")),
            Some(_)
        ));
        assert!(parse_fed_strategy("invalid", 0.8, 42, None).is_none());
    }

    #[test]
    fn test_cmd_fed_manifest_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        create_test_parquet(&data_path, 100);

        let result = cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json");
        assert!(result.is_ok());
        assert!(manifest_path.exists());

        // Verify manifest contents
        let content = std::fs::read_to_string(&manifest_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert_eq!(
            parsed.get("node_id").and_then(|v| v.as_str()),
            Some("node-1")
        );
        assert_eq!(parsed.get("total_rows").and_then(|v| v.as_u64()), Some(100));
    }

    #[test]
    fn test_cmd_fed_manifest_binary() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.bin");
        create_test_parquet(&data_path, 50);

        let result = cmd_fed_manifest(&data_path, &manifest_path, "node-2", 0.8, 42, "binary");
        assert!(result.is_ok());
        assert!(manifest_path.exists());
    }

    #[test]
    fn test_cmd_fed_plan_local_strategy() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        // Create manifests for two nodes
        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let manifest1 = temp_dir.path().join("manifest1.json");
        let manifest2 = temp_dir.path().join("manifest2.json");
        let plan_path = temp_dir.path().join("plan.json");

        create_test_parquet(&data1, 100);
        create_test_parquet(&data2, 150);

        cmd_fed_manifest(&data1, &manifest1, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest1"));
        cmd_fed_manifest(&data2, &manifest2, "node-2", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest2"));

        let manifests = vec![manifest1.clone(), manifest2.clone()];
        let result = cmd_fed_plan(&manifests, &plan_path, "local", 0.8, 42, None, "json");
        assert!(result.is_ok());
        assert!(plan_path.exists());

        // Verify plan contents
        let content = std::fs::read_to_string(&plan_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        let instructions = parsed.as_array();
        assert!(instructions.is_some());
        assert_eq!(instructions.map(|a| a.len()), Some(2));
    }

    #[test]
    fn test_cmd_fed_plan_empty_manifests_fails() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let plan_path = temp_dir.path().join("plan.json");

        let manifests: Vec<PathBuf> = vec![];
        let result = cmd_fed_plan(&manifests, &plan_path, "local", 0.8, 42, None, "json");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_split_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");
        let train_path = temp_dir.path().join("train.parquet");
        let test_path = temp_dir.path().join("test.parquet");

        create_test_parquet(&data_path, 100);

        // Create manifest
        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        // Create plan
        let manifests = vec![manifest_path.clone()];
        cmd_fed_plan(&manifests, &plan_path, "local", 0.8, 42, None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create plan"));

        // Execute split
        let result = cmd_fed_split(
            &data_path,
            &plan_path,
            "node-1",
            &train_path,
            &test_path,
            None,
        );
        assert!(result.is_ok());
        assert!(train_path.exists());
        assert!(test_path.exists());

        // Verify split sizes
        let train_ds = ArrowDataset::from_parquet(&train_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load train"));
        let test_ds = ArrowDataset::from_parquet(&test_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load test"));

        assert!(train_ds.len() > 0);
        assert!(test_ds.len() > 0);
        assert_eq!(train_ds.len() + test_ds.len(), 100);
    }

    #[test]
    fn test_cmd_fed_split_node_not_found() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");
        let train_path = temp_dir.path().join("train.parquet");
        let test_path = temp_dir.path().join("test.parquet");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        cmd_fed_plan(&manifests, &plan_path, "local", 0.8, 42, None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create plan"));

        // Try to split with wrong node ID
        let result = cmd_fed_split(
            &data_path,
            &plan_path,
            "wrong-node",
            &train_path,
            &test_path,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_verify_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let manifest1 = temp_dir.path().join("manifest1.json");
        let manifest2 = temp_dir.path().join("manifest2.json");

        create_test_parquet(&data1, 100);
        create_test_parquet(&data2, 150);

        cmd_fed_manifest(&data1, &manifest1, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest1"));
        cmd_fed_manifest(&data2, &manifest2, "node-2", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest2"));

        let manifests = vec![manifest1.clone(), manifest2.clone()];
        let result = cmd_fed_verify(&manifests, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_fed_verify_json_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_verify(&manifests, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_fed_verify_empty_manifests_fails() {
        let manifests: Vec<PathBuf> = vec![];
        let result = cmd_fed_verify(&manifests, "text");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_plan_proportional_strategy() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_plan(
            &manifests,
            &plan_path,
            "proportional",
            0.7,
            42,
            None,
            "json",
        );
        assert!(result.is_ok());

        // Verify train ratio is 0.7
        let content = std::fs::read_to_string(&plan_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        let instructions = parsed
            .as_array()
            .unwrap_or_else(|| panic!("Should be array"));
        let train_ratio = instructions[0].get("train_ratio").and_then(|v| v.as_f64());
        assert!((train_ratio.unwrap_or(0.0) - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_load_manifest_invalid_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let manifest_path = temp_dir.path().join("invalid.json");

        std::fs::write(&manifest_path, "not valid json")
            .ok()
            .unwrap_or_else(|| panic!("Should write file"));

        let result = load_manifest(&manifest_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_plan_invalid_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let plan_path = temp_dir.path().join("invalid.json");

        std::fs::write(&plan_path, "{ broken }")
            .ok()
            .unwrap_or_else(|| panic!("Should write file"));

        let result = load_plan(&plan_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_plan_invalid_strategy() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_plan(
            &manifests,
            &plan_path,
            "invalid_strategy",
            0.8,
            42,
            None,
            "json",
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_fed_plan_stratified_strategy() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("data.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");
        let plan_path = temp_dir.path().join("plan.json");

        create_test_parquet(&data_path, 100);

        cmd_fed_manifest(&data_path, &manifest_path, "node-1", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_plan(
            &manifests,
            &plan_path,
            "stratified",
            0.8,
            42,
            Some("name"),
            "json",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_fed_verify_with_quality_issues() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data_path = temp_dir.path().join("small.parquet");
        let manifest_path = temp_dir.path().join("manifest.json");

        create_test_parquet(&data_path, 15);

        cmd_fed_manifest(&data_path, &manifest_path, "small-node", 0.8, 42, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create manifest"));

        let manifests = vec![manifest_path.clone()];
        let result = cmd_fed_verify(&manifests, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_fed_strategy_iid() {
        let result = parse_fed_strategy("iid", 0.8, 42, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_fed_strategy_proportional() {
        let result = parse_fed_strategy("proportional", 0.8, 42, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_fed_strategy_local() {
        let result = parse_fed_strategy("local", 0.8, 42, None);
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_fed_strategy_stratified() {
        let result = parse_fed_strategy("stratified", 0.8, 42, Some("label"));
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_fed_strategy_unknown() {
        assert!(parse_fed_strategy("invalid", 0.8, 42, None).is_none());
    }

    #[test]
    fn test_cmd_fed_plan() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let manifest1 = temp_dir.path().join("node1.json");
        let manifest2 = temp_dir.path().join("node2.json");
        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let output = temp_dir.path().join("plan.json");

        create_test_parquet(&data1, 50);
        create_test_parquet(&data2, 50);

        cmd_fed_manifest(&data1, &manifest1, "node1", 0.8, 42, "json").unwrap();
        cmd_fed_manifest(&data2, &manifest2, "node2", 0.8, 42, "json").unwrap();

        let manifests = vec![manifest1, manifest2];
        let result = cmd_fed_plan(&manifests, &output, "iid", 0.8, 42, None, "json");
        assert!(result.is_ok());
    }
}
