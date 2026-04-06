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
//! Basic Data Loading Example
//!
//! Demonstrates loading datasets from various file formats:
//! - CSV files
//! - JSON files
//! - Parquet files
//!
//! Run with: cargo run --example basic_loading

use std::sync::Arc;

use alimentar::{ArrowDataset, CsvOptions, Dataset, JsonOptions};
use arrow::{
    array::{Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn main() -> alimentar::Result<()> {
    println!("=== Alimentar Basic Loading Example ===\n");

    // Create sample data for demonstrations
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "alpha", "beta", "gamma", "delta", "epsilon",
            ])),
            Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3, 4.4, 5.5])),
        ],
    )?;

    // 1. Create dataset from RecordBatch
    println!("1. Creating dataset from RecordBatch");
    let dataset = ArrowDataset::from_batch(batch)?;
    println!("   Dataset has {} rows", dataset.len());
    println!("   Schema: {:?}", dataset.schema());

    // 2. Save to various formats
    let temp_dir = std::env::temp_dir();

    // Save as Parquet
    let parquet_path = temp_dir.join("example_data.parquet");
    dataset.to_parquet(&parquet_path)?;
    println!("\n2. Saved to Parquet: {:?}", parquet_path);

    // Save as CSV
    let csv_path = temp_dir.join("example_data.csv");
    dataset.to_csv(&csv_path)?;
    println!("   Saved to CSV: {:?}", csv_path);

    // Save as JSON
    let json_path = temp_dir.join("example_data.json");
    dataset.to_json(&json_path)?;
    println!("   Saved to JSON: {:?}", json_path);

    // 3. Load from Parquet
    println!("\n3. Loading from Parquet");
    let loaded_parquet = ArrowDataset::from_parquet(&parquet_path)?;
    println!("   Loaded {} rows from Parquet", loaded_parquet.len());

    // 4. Load from CSV
    println!("\n4. Loading from CSV");
    let loaded_csv = ArrowDataset::from_csv(&csv_path)?;
    println!("   Loaded {} rows from CSV", loaded_csv.len());

    // 5. Load from CSV with options
    println!("\n5. Loading CSV with custom options");
    let csv_options = CsvOptions::default()
        .with_header(true)
        .with_delimiter(b',')
        .with_batch_size(1000);
    let loaded_csv_opts = ArrowDataset::from_csv_with_options(&csv_path, csv_options)?;
    println!(
        "   Loaded {} rows with custom options",
        loaded_csv_opts.len()
    );

    // 6. Load from JSON
    println!("\n6. Loading from JSON");
    let loaded_json = ArrowDataset::from_json(&json_path)?;
    println!("   Loaded {} rows from JSON", loaded_json.len());

    // 7. Load JSON with options
    println!("\n7. Loading JSON with custom options");
    let json_options = JsonOptions::default().with_batch_size(500);
    let loaded_json_opts = ArrowDataset::from_json_with_options(&json_path, json_options)?;
    println!(
        "   Loaded {} rows with custom options",
        loaded_json_opts.len()
    );

    // 8. Iterate over batches
    println!("\n8. Iterating over batches");
    for (i, batch) in dataset.iter().enumerate() {
        println!("   Batch {}: {} rows", i, batch.num_rows());
    }

    // 9. Access individual rows
    println!("\n9. Accessing individual rows");
    if let Some(row) = dataset.get(0) {
        println!("   First row: {} columns", row.num_columns());
    }

    // Cleanup
    let _ = std::fs::remove_file(&parquet_path);
    let _ = std::fs::remove_file(&csv_path);
    let _ = std::fs::remove_file(&json_path);

    println!("\n=== Example Complete ===");
    Ok(())
}
