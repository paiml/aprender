//! Registry CLI commands for dataset sharing and discovery.

use std::path::{Path, PathBuf};

use clap::Subcommand;

use super::basic::load_dataset;
use crate::{
    backend::LocalBackend,
    registry::{DatasetMetadata, Registry},
    Dataset,
};

/// Registry commands for dataset sharing and discovery.
#[derive(Subcommand)]
pub enum RegistryCommands {
    /// Initialize a new registry
    Init {
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
    /// List all datasets in a registry
    List {
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
    /// Push (publish) a dataset to the registry
    Push {
        /// Path to the dataset file (parquet)
        input: PathBuf,
        /// Dataset name in the registry
        #[arg(short, long)]
        name: String,
        /// Dataset version (semver)
        #[arg(short, long, default_value = "1.0.0")]
        version: String,
        /// Description of the dataset
        #[arg(short, long, default_value = "")]
        description: String,
        /// License identifier (e.g., MIT, Apache-2.0)
        #[arg(short, long, default_value = "")]
        license: String,
        /// Tags for the dataset (comma-separated)
        #[arg(short, long, default_value = "")]
        tags: String,
        /// Path to registry directory
        #[arg(long, default_value = ".alimentar")]
        registry: PathBuf,
    },
    /// Pull (download) a dataset from the registry
    Pull {
        /// Dataset name
        name: String,
        /// Output path for the dataset
        #[arg(short, long)]
        output: PathBuf,
        /// Specific version to pull (defaults to latest)
        #[arg(short, long)]
        version: Option<String>,
        /// Path to registry directory
        #[arg(long, default_value = ".alimentar")]
        registry: PathBuf,
    },
    /// Search datasets by name or description
    Search {
        /// Search query
        query: String,
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
    /// Show detailed info about a specific dataset
    ShowInfo {
        /// Dataset name
        name: String,
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
    /// Delete a dataset version from the registry
    Delete {
        /// Dataset name
        name: String,
        /// Version to delete
        #[arg(short, long)]
        version: String,
        /// Path to registry directory
        #[arg(short, long, default_value = ".alimentar")]
        path: PathBuf,
    },
}

/// Create a registry with the given path.
pub(crate) fn create_registry(path: &Path) -> crate::Result<Registry> {
    // Ensure directory exists
    if !path.exists() {
        std::fs::create_dir_all(path).map_err(|e| crate::Error::io(e, path))?;
    }
    let backend = LocalBackend::new(path)?;
    Ok(Registry::new(Box::new(backend)))
}

/// Initialize a new registry.
pub(crate) fn cmd_registry_init(path: &Path) -> crate::Result<()> {
    let registry = create_registry(path)?;
    registry.init()?;
    println!("Initialized registry at: {}", path.display());
    Ok(())
}

/// List all datasets in a registry.
pub(crate) fn cmd_registry_list(path: &Path) -> crate::Result<()> {
    let registry = create_registry(path)?;
    let datasets = registry.list()?;

    if datasets.is_empty() {
        println!("No datasets in registry.");
        return Ok(());
    }

    println!("Datasets in registry:\n");
    println!(
        "{:<25} {:<12} {:<10} {:<15} DESCRIPTION",
        "NAME", "LATEST", "VERSIONS", "ROWS"
    );
    println!("{}", "-".repeat(80));

    for ds in datasets {
        let desc = if ds.metadata.description.len() > 30 {
            format!("{}...", &ds.metadata.description[..27])
        } else {
            ds.metadata.description.clone()
        };
        println!(
            "{:<25} {:<12} {:<10} {:<15} {}",
            ds.name,
            ds.latest,
            ds.versions.len(),
            ds.num_rows,
            desc
        );
    }

    Ok(())
}

/// Push (publish) a dataset to the registry.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_registry_push(
    input: &Path,
    name: &str,
    version: &str,
    description: &str,
    license: &str,
    tags: &str,
    registry_path: &Path,
) -> crate::Result<()> {
    let registry = create_registry(registry_path)?;

    // Initialize if needed
    registry.init()?;

    // Load the dataset
    let dataset = load_dataset(input)?;

    // Parse tags
    let tag_list: Vec<String> = if tags.is_empty() {
        Vec::new()
    } else {
        tags.split(',').map(|s| s.trim().to_string()).collect()
    };

    // Create metadata
    let metadata = DatasetMetadata {
        description: description.to_string(),
        license: license.to_string(),
        tags: tag_list,
        source: Some(input.display().to_string()),
        citation: None,
        sha256: None, // Computed during save, not at publish time
    };

    // Publish
    registry.publish(name, version, &dataset, metadata)?;

    println!(
        "Published {}@{} ({} rows) to registry",
        name,
        version,
        dataset.len()
    );

    Ok(())
}

/// Pull (download) a dataset from the registry.
pub(crate) fn cmd_registry_pull(
    name: &str,
    output: &Path,
    version: Option<&str>,
    registry_path: &Path,
) -> crate::Result<()> {
    let registry = create_registry(registry_path)?;

    // Pull the dataset
    let dataset = registry.pull(name, version)?;

    // Save to output
    dataset.to_parquet(output)?;

    let ver = version.unwrap_or("latest");
    println!(
        "Pulled {}@{} ({} rows) to {}",
        name,
        ver,
        dataset.len(),
        output.display()
    );

    Ok(())
}

/// Search datasets by name or description.
pub(crate) fn cmd_registry_search(query: &str, path: &Path) -> crate::Result<()> {
    let registry = create_registry(path)?;
    let results = registry.search(query)?;

    if results.is_empty() {
        println!("No datasets found matching '{}'", query);
        return Ok(());
    }

    println!("Search results for '{}':\n", query);
    println!("{:<25} {:<12} {:<10} DESCRIPTION", "NAME", "LATEST", "ROWS");
    println!("{}", "-".repeat(70));

    for ds in results {
        let desc = if ds.metadata.description.len() > 30 {
            format!("{}...", &ds.metadata.description[..27])
        } else {
            ds.metadata.description.clone()
        };
        println!(
            "{:<25} {:<12} {:<10} {}",
            ds.name, ds.latest, ds.num_rows, desc
        );
    }

    Ok(())
}

/// Show detailed info about a specific dataset.
pub(crate) fn cmd_registry_show_info(name: &str, path: &Path) -> crate::Result<()> {
    let registry = create_registry(path)?;
    let info = registry.get_info(name)?;

    println!("Dataset: {}", info.name);
    println!("Latest: {}", info.latest);
    println!("Versions: {}", info.versions.join(", "));
    println!("Rows: {}", info.num_rows);
    println!("Size: {} bytes", info.size_bytes);
    println!();
    println!("Description: {}", info.metadata.description);
    println!("License: {}", info.metadata.license);
    println!("Tags: {}", info.metadata.tags.join(", "));

    if let Some(source) = &info.metadata.source {
        println!("Source: {}", source);
    }
    if let Some(citation) = &info.metadata.citation {
        println!("Citation: {}", citation);
    }

    println!();
    println!("Schema:");
    if let Some(fields) = info.schema.get("fields").and_then(|f| f.as_array()) {
        for field in fields {
            let name = field.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            let dtype = field
                .get("data_type")
                .and_then(|d| d.as_str())
                .unwrap_or("?");
            let nullable = field
                .get("nullable")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true);
            let null_str = if nullable { "nullable" } else { "not null" };
            println!("  - {} ({}) [{}]", name, dtype, null_str);
        }
    }

    Ok(())
}

/// Delete a dataset version from the registry.
pub(crate) fn cmd_registry_delete(name: &str, version: &str, path: &Path) -> crate::Result<()> {
    let registry = create_registry(path)?;
    registry.delete(name, version)?;
    println!("Deleted {}@{} from registry", name, version);
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
    fn test_cmd_registry_init() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        let result = cmd_registry_init(&registry_path);
        assert!(result.is_ok());
        assert!(registry_path.exists());
    }

    #[test]
    fn test_cmd_registry_list_empty() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        // Init first
        cmd_registry_init(&registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        let result = cmd_registry_list(&registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_push_and_pull() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");
        let output = temp_dir.path().join("pulled.parquet");

        // Create test data
        create_test_parquet(&input, 25);

        // Push
        let result = cmd_registry_push(
            &input,
            "test-dataset",
            "1.0.0",
            "A test dataset",
            "MIT",
            "test,example",
            &registry_path,
        );
        assert!(result.is_ok());

        // List should show the dataset
        let result = cmd_registry_list(&registry_path);
        assert!(result.is_ok());

        // Pull
        let result = cmd_registry_pull("test-dataset", &output, Some("1.0.0"), &registry_path);
        assert!(result.is_ok());
        assert!(output.exists());

        // Verify data
        let original = ArrowDataset::from_parquet(&input)
            .ok()
            .unwrap_or_else(|| panic!("Should load original"));
        let pulled = ArrowDataset::from_parquet(&output)
            .ok()
            .unwrap_or_else(|| panic!("Should load pulled"));
        assert_eq!(original.len(), pulled.len());
    }

    #[test]
    fn test_cmd_registry_search() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        // Push with description
        cmd_registry_push(
            &input,
            "ml-dataset",
            "1.0.0",
            "Machine learning training data",
            "Apache-2.0",
            "ml,training",
            &registry_path,
        )
        .ok()
        .unwrap_or_else(|| panic!("Should push"));

        // Search by name
        let result = cmd_registry_search("ml", &registry_path);
        assert!(result.is_ok());

        // Search by description
        let result = cmd_registry_search("machine", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_show_info() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        cmd_registry_push(
            &input,
            "info-test",
            "1.0.0",
            "Test description",
            "MIT",
            "test",
            &registry_path,
        )
        .ok()
        .unwrap_or_else(|| panic!("Should push"));

        let result = cmd_registry_show_info("info-test", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_delete() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        // Push
        cmd_registry_push(
            &input,
            "delete-test",
            "1.0.0",
            "Will be deleted",
            "",
            "",
            &registry_path,
        )
        .ok()
        .unwrap_or_else(|| panic!("Should push"));

        // Delete
        let result = cmd_registry_delete("delete-test", "1.0.0", &registry_path);
        assert!(result.is_ok());

        // Should no longer exist
        let result = cmd_registry_show_info("delete-test", &registry_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_registry_pull_latest() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input1 = temp_dir.path().join("v1.parquet");
        let input2 = temp_dir.path().join("v2.parquet");
        let output = temp_dir.path().join("pulled.parquet");

        create_test_parquet(&input1, 10);
        create_test_parquet(&input2, 20);

        // Push v1
        cmd_registry_push(&input1, "versioned", "1.0.0", "V1", "", "", &registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should push v1"));

        // Push v2
        cmd_registry_push(&input2, "versioned", "2.0.0", "V2", "", "", &registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should push v2"));

        // Pull latest (no version specified)
        let result = cmd_registry_pull("versioned", &output, None, &registry_path);
        assert!(result.is_ok());

        // Should be v2 (20 rows)
        let pulled = ArrowDataset::from_parquet(&output)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        assert_eq!(pulled.len(), 20);
    }

    #[test]
    fn test_cmd_registry_search_no_results() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        // Init registry
        cmd_registry_init(&registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        // Search for something that doesn't exist
        let result = cmd_registry_search("nonexistent-dataset-xyz", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_push_with_long_description() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        // Push with a very long description
        let long_desc = "This is a very long description that exceeds thirty characters and will be truncated in the list view";
        let result = cmd_registry_push(
            &input,
            "long-desc-test",
            "1.0.0",
            long_desc,
            "MIT",
            "",
            &registry_path,
        );
        assert!(result.is_ok());

        // List should truncate the description
        let result = cmd_registry_list(&registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_show_info_with_all_metadata() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let input = temp_dir.path().join("data.parquet");

        create_test_parquet(&input, 10);

        // Push with all metadata
        cmd_registry_push(
            &input,
            "full-metadata",
            "1.0.0",
            "Full metadata test",
            "Apache-2.0",
            "test,metadata,full",
            &registry_path,
        )
        .ok()
        .unwrap_or_else(|| panic!("Should push"));

        let result = cmd_registry_show_info("full-metadata", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_registry_new_directory() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("new_registry_dir");

        // Directory doesn't exist yet
        assert!(!registry_path.exists());

        let result = create_registry(&registry_path);
        assert!(result.is_ok());

        // Directory should now exist
        assert!(registry_path.exists());
    }

    #[test]
    fn test_cmd_registry_delete_nonexistent() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        // Init registry
        cmd_registry_init(&registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        // Try to delete something that doesn't exist
        let result = cmd_registry_delete("nonexistent", "1.0.0", &registry_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_registry_pull_nonexistent() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let output = temp_dir.path().join("output.parquet");

        // Init registry
        cmd_registry_init(&registry_path)
            .ok()
            .unwrap_or_else(|| panic!("Should init"));

        // Try to pull something that doesn't exist
        let result = cmd_registry_pull("nonexistent", &output, None, &registry_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_registry_search_with_data() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let data_path = temp_dir.path().join("data.parquet");

        create_test_parquet(&data_path, 20);

        // Push a dataset first
        cmd_registry_push(
            &data_path,
            "searchable-data",
            "1.0.0",
            "Dataset for search test",
            "MIT",
            "search,test",
            &registry_path,
        )
        .unwrap();

        let result = cmd_registry_search("search", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_search_empty_results() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        cmd_registry_init(&registry_path).unwrap();

        let result = cmd_registry_search("nonexistent", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_show_info_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let data_path = temp_dir.path().join("data.parquet");

        create_test_parquet(&data_path, 20);

        cmd_registry_push(
            &data_path,
            "info-dataset",
            "1.0.0",
            "Dataset for info test",
            "Apache-2.0",
            "info,test",
            &registry_path,
        )
        .unwrap();

        let result = cmd_registry_show_info("info-dataset", &registry_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_registry_show_info_not_found() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");

        cmd_registry_init(&registry_path).unwrap();

        let result = cmd_registry_show_info("nonexistent", &registry_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_registry_delete_existing() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let registry_path = temp_dir.path().join("registry");
        let data_path = temp_dir.path().join("data.parquet");

        create_test_parquet(&data_path, 20);

        cmd_registry_push(
            &data_path,
            "delete-test",
            "1.0.0",
            "Dataset to delete",
            "MIT",
            "delete,test",
            &registry_path,
        )
        .unwrap();

        let result = cmd_registry_delete("delete-test", "1.0.0", &registry_path);
        assert!(result.is_ok());
    }
}
