#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::needless_late_init,
    clippy::redundant_clone,
    clippy::doc_markdown,
    clippy::unnecessary_debug_formatting
)]
//! Registry Publish/Pull Example
//!
//! Demonstrates dataset registry operations:
//! - Publishing datasets with metadata
//! - Listing available datasets
//! - Searching by name and tags
//! - Pulling specific versions
//! - Dataset versioning
//!
//! Run with: cargo run --example registry_publish

use std::sync::Arc;

use alimentar::{
    backend::MemoryBackend,
    registry::{DatasetMetadata, Registry},
    ArrowDataset, Dataset,
};
use arrow::{
    array::{Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn create_iris_dataset() -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("sepal_length", DataType::Float64, false),
        Field::new("sepal_width", DataType::Float64, false),
        Field::new("petal_length", DataType::Float64, false),
        Field::new("petal_width", DataType::Float64, false),
        Field::new("species", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![5.1, 4.9, 4.7, 7.0, 6.4, 6.9])),
            Arc::new(Float64Array::from(vec![3.5, 3.0, 3.2, 3.2, 3.2, 3.1])),
            Arc::new(Float64Array::from(vec![1.4, 1.4, 1.3, 4.7, 4.5, 4.9])),
            Arc::new(Float64Array::from(vec![0.2, 0.2, 0.2, 1.4, 1.5, 1.5])),
            Arc::new(StringArray::from(vec![
                "setosa",
                "setosa",
                "setosa",
                "versicolor",
                "versicolor",
                "versicolor",
            ])),
        ],
    )?;

    ArrowDataset::from_batch(batch)
}

fn create_mnist_dataset() -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("pixel_0", DataType::Int32, false),
        Field::new("pixel_1", DataType::Int32, false),
        Field::new("label", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![0, 128, 255, 64, 192])),
            Arc::new(Int32Array::from(vec![255, 64, 0, 128, 32])),
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])),
        ],
    )?;

    ArrowDataset::from_batch(batch)
}

fn main() -> alimentar::Result<()> {
    println!("=== Alimentar Registry Example ===\n");

    // Create an in-memory registry (could also use LocalBackend for persistence)
    let backend = MemoryBackend::new();
    let registry = Registry::new(Box::new(backend));

    // Initialize the registry
    registry.init()?;
    println!("Registry initialized");

    // 1. Publish Iris dataset
    println!("\n1. Publishing Iris dataset v1.0.0");
    let iris = create_iris_dataset()?;
    let iris_metadata = DatasetMetadata {
        description: "Classic Iris flower classification dataset".to_string(),
        license: "CC0-1.0".to_string(),
        tags: vec![
            "classification".to_string(),
            "tabular".to_string(),
            "flowers".to_string(),
            "ml-basics".to_string(),
        ],
        source: Some("UCI Machine Learning Repository".to_string()),
        citation: Some("Fisher, R.A. (1936)".to_string()),
        sha256: None,
    };

    registry.publish("iris", "1.0.0", &iris, iris_metadata)?;
    println!("   Published: iris v1.0.0 ({} rows)", iris.len());

    // 2. Publish MNIST dataset
    println!("\n2. Publishing MNIST dataset v1.0.0");
    let mnist = create_mnist_dataset()?;
    let mnist_metadata = DatasetMetadata {
        description: "Handwritten digit recognition dataset (sample)".to_string(),
        license: "CC-BY-SA-3.0".to_string(),
        tags: vec![
            "classification".to_string(),
            "images".to_string(),
            "digits".to_string(),
            "computer-vision".to_string(),
        ],
        source: Some("Yann LeCun".to_string()),
        citation: None,
        sha256: None,
    };

    registry.publish("mnist", "1.0.0", &mnist, mnist_metadata)?;
    println!("   Published: mnist v1.0.0 ({} rows)", mnist.len());

    // 3. Publish a new version of Iris
    println!("\n3. Publishing Iris v2.0.0 (more samples)");
    let iris_v2_metadata = DatasetMetadata {
        description: "Iris dataset with extended samples".to_string(),
        license: "CC0-1.0".to_string(),
        tags: vec![
            "classification".to_string(),
            "tabular".to_string(),
            "flowers".to_string(),
        ],
        source: Some("UCI Machine Learning Repository".to_string()),
        citation: Some("Fisher, R.A. (1936)".to_string()),
        sha256: None,
    };

    registry.publish("iris", "2.0.0", &iris, iris_v2_metadata)?;
    println!("   Published: iris v2.0.0");

    // 4. List all datasets
    println!("\n4. Listing all datasets");
    let datasets = registry.list()?;
    for ds in &datasets {
        println!(
            "   - {} (versions: {:?}, latest: {})",
            ds.name, ds.versions, ds.latest
        );
        println!("     Description: {}", ds.metadata.description);
        println!("     Tags: {:?}", ds.metadata.tags);
    }

    // 5. Search by name
    println!("\n5. Searching for 'iris'");
    let results = registry.search("iris")?;
    println!("   Found {} result(s)", results.len());
    for ds in &results {
        println!("   - {}: {}", ds.name, ds.metadata.description);
    }

    // 6. Search by tags
    println!("\n6. Searching by tag 'classification'");
    let results = registry.search_tags(&["classification"])?;
    println!(
        "   Found {} dataset(s) with 'classification' tag",
        results.len()
    );
    for ds in &results {
        println!("   - {}", ds.name);
    }

    // 7. Get dataset info
    println!("\n7. Getting info for 'iris'");
    let info = registry.get_info("iris")?;
    println!("   Name: {}", info.name);
    println!("   Versions: {:?}", info.versions);
    println!("   Latest: {}", info.latest);
    println!("   Size: {} bytes", info.size_bytes);
    println!("   Rows: {}", info.num_rows);
    println!("   License: {}", info.metadata.license);

    // 8. Pull latest version
    println!("\n8. Pulling iris (latest)");
    let pulled = registry.pull("iris", None)?;
    println!("   Pulled {} rows", pulled.len());

    // 9. Pull specific version
    println!("\n9. Pulling iris v1.0.0 specifically");
    let pulled_v1 = registry.pull("iris", Some("1.0.0"))?;
    println!("   Pulled {} rows (v1.0.0)", pulled_v1.len());

    // 10. Search by description
    println!("\n10. Searching for 'digit' in descriptions");
    let results = registry.search("digit")?;
    println!("   Found {} dataset(s)", results.len());
    for ds in &results {
        println!("   - {}: {}", ds.name, ds.metadata.description);
    }

    // 11. Delete a version
    println!("\n11. Deleting iris v1.0.0");
    registry.delete("iris", "1.0.0")?;
    println!("   Deleted iris v1.0.0");

    let info = registry.get_info("iris")?;
    println!("   Remaining versions: {:?}", info.versions);

    println!("\n=== Example Complete ===");
    Ok(())
}
