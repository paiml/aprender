//! Basic Trueno-DB usage: Storage engine and morsel iteration
//!
//! This example demonstrates:
//! - Loading Parquet files
//! - Morsel-based iteration (128MB chunks for out-of-core execution)
//! - Append-only OLAP write pattern
//!
//! Run with: cargo run --example `basic_usage`

use arrow::array::{Float32Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use trueno_db::storage::StorageEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Trueno-DB Basic Usage Example ===\n");

    // Create sample data (1M rows)
    println!("Creating sample dataset (1M rows)...");
    let batch = create_sample_batch(1_000_000)?;
    println!("  ✓ Created batch: {} rows, {} columns", batch.num_rows(), batch.num_columns());
    println!("  ✓ Memory size: {:.2} MB\n", batch.get_array_memory_size() as f64 / 1_048_576.0);

    // Initialize storage engine
    println!("Initializing storage engine...");
    let mut storage = StorageEngine::new(vec![]);
    println!("  ✓ Storage engine initialized\n");

    // Append batch (OLAP pattern)
    println!("Appending batch to storage (OLAP append-only pattern)...");
    storage.append_batch(batch)?;
    println!("  ✓ Batch appended successfully");
    println!("  ✓ Total batches in storage: {}\n", storage.batches().len());

    // Demonstrate morsel iteration (128MB chunks for out-of-core execution)
    println!("Iterating over morsels (128MB chunks):");
    println!("  This prevents GPU VRAM OOM by processing data in chunks\n");

    let mut total_rows = 0;
    let mut morsel_count = 0;

    for morsel in storage.morsels() {
        morsel_count += 1;
        let rows = morsel.num_rows();
        let size_mb = morsel.get_array_memory_size() as f64 / 1_048_576.0;
        total_rows += rows;

        if morsel_count <= 3 || morsel_count % 10 == 0 {
            println!("  Morsel #{morsel_count}: {rows} rows, {size_mb:.2} MB");
        }
    }

    println!("\n  ✓ Total morsels: {morsel_count}");
    println!("  ✓ Total rows processed: {total_rows}");
    println!("  ✓ All data accounted for: {}\n", total_rows == 1_000_000);

    // Show schema information
    println!("Schema information:");
    if let Some(first_batch) = storage.batches().first() {
        let schema = first_batch.schema();
        for field in schema.fields() {
            println!("  - {}: {:?}", field.name(), field.data_type());
        }
    }
    println!();

    // Demonstrate OLAP vs OLTP
    println!("=== OLAP vs OLTP Design ===");
    println!("✓ Supported: append_batch() - Bulk append (OLAP pattern)");
    println!("✗ Not supported: update_row() - Random updates (OLTP pattern)");
    println!("\nRationale:");
    println!("  Columnar storage optimizes for bulk reads, not random writes");
    println!("  Single-row update cost: O(N) (rewrite entire column)");
    println!("  Batch append cost: O(1) (append to new partition)\n");

    Ok(())
}

fn create_sample_batch(num_rows: usize) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let schema = Schema::new(vec![
        Field::new("user_id", DataType::Int32, false),
        Field::new("score", DataType::Float32, false),
        Field::new("category", DataType::Utf8, false),
    ]);

    let user_ids: Vec<i32> = (0..num_rows).map(|i| i as i32).collect();
    let scores: Vec<f32> = (0..num_rows).map(|i| (i as f32 * 1.5) % 100.0).collect();
    let categories: Vec<String> = (0..num_rows).map(|i| format!("category_{}", i % 10)).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(user_ids)),
            Arc::new(Float32Array::from(scores)),
            Arc::new(StringArray::from(categories)),
        ],
    )?;

    Ok(batch)
}
