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
//! DataLoader Batching Example
//!
//! Demonstrates using DataLoader for ML-style batched iteration:
//! - Configurable batch sizes
//! - Shuffling with reproducible seeds
//! - Drop last incomplete batch option
//!
//! Run with: cargo run --example dataloader_batching --features shuffle

use std::sync::Arc;

use alimentar::{ArrowDataset, DataLoader, Dataset};
use arrow::{
    array::{Float64Array, Int32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn create_sample_dataset(num_samples: usize) -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("feature_1", DataType::Float64, false),
        Field::new("feature_2", DataType::Float64, false),
        Field::new("label", DataType::Int32, false),
    ]));

    let feature_1: Vec<f64> = (0..num_samples).map(|i| i as f64 * 0.1).collect();
    let feature_2: Vec<f64> = (0..num_samples).map(|i| (i as f64).sin()).collect();
    let labels: Vec<i32> = (0..num_samples).map(|i| (i % 3) as i32).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(feature_1)),
            Arc::new(Float64Array::from(feature_2)),
            Arc::new(Int32Array::from(labels)),
        ],
    )?;

    ArrowDataset::from_batch(batch)
}

fn main() -> alimentar::Result<()> {
    println!("=== Alimentar DataLoader Example ===\n");

    // Create a dataset with 100 samples
    let dataset = create_sample_dataset(100)?;
    println!("Created dataset with {} samples", dataset.len());

    // 1. Basic DataLoader with batch size
    println!("\n1. Basic DataLoader (batch_size=32)");
    let loader = DataLoader::new(dataset.clone()).batch_size(32);

    let mut batch_count = 0;
    let mut total_rows = 0;
    for batch in loader {
        batch_count += 1;
        total_rows += batch.num_rows();
        println!("   Batch {}: {} rows", batch_count, batch.num_rows());
    }
    println!("   Total: {} batches, {} rows", batch_count, total_rows);

    // 2. DataLoader with drop_last
    println!("\n2. DataLoader with drop_last=true (batch_size=32)");
    let loader = DataLoader::new(dataset.clone())
        .batch_size(32)
        .drop_last(true);

    let mut batch_count = 0;
    let mut total_rows = 0;
    for batch in loader {
        batch_count += 1;
        total_rows += batch.num_rows();
        println!("   Batch {}: {} rows", batch_count, batch.num_rows());
    }
    println!(
        "   Total: {} batches, {} rows (dropped incomplete last batch)",
        batch_count, total_rows
    );

    // 3. DataLoader with shuffling (requires shuffle feature)
    #[cfg(feature = "shuffle")]
    {
        println!("\n3. Shuffled DataLoader (seed=42)");
        let loader = DataLoader::new(dataset.clone())
            .batch_size(32)
            .shuffle(true)
            .seed(42);

        println!("   First epoch:");
        let first_epoch: Vec<_> = loader.into_iter().collect();
        for (i, batch) in first_epoch.iter().enumerate() {
            println!("   Batch {}: {} rows", i + 1, batch.num_rows());
        }

        // Same seed produces same order
        println!("\n   Second epoch with same seed (should be identical):");
        let loader2 = DataLoader::new(dataset.clone())
            .batch_size(32)
            .shuffle(true)
            .seed(42);

        for (i, batch) in loader2.into_iter().enumerate() {
            println!("   Batch {}: {} rows", i + 1, batch.num_rows());
        }

        // Different seed produces different order
        println!("\n4. Different seed produces different order");
        let loader3 = DataLoader::new(dataset.clone())
            .batch_size(32)
            .shuffle(true)
            .seed(123);

        for (i, batch) in loader3.into_iter().take(2).enumerate() {
            println!("   Batch {}: {} rows", i + 1, batch.num_rows());
        }
    }

    #[cfg(not(feature = "shuffle"))]
    {
        println!("\n3. Shuffling requires --features shuffle");
        println!("   Run: cargo run --example dataloader_batching --features shuffle");
    }

    // 4. Using num_batches for planning
    println!("\n5. Pre-computing number of batches");
    let loader = DataLoader::new(dataset.clone()).batch_size(32);
    let num_batches = loader.num_batches();
    println!("   DataLoader will produce {} batches", num_batches);

    // 5. Training loop simulation
    println!("\n6. Simulated training loop");
    let epochs = 3;
    let batch_size = 25;

    for epoch in 0..epochs {
        let loader = DataLoader::new(dataset.clone()).batch_size(batch_size);
        let mut epoch_loss = 0.0;

        for (batch_idx, batch) in loader.into_iter().enumerate() {
            // Simulate training step
            let batch_loss = 1.0 / (epoch as f64 + batch_idx as f64 + 1.0);
            epoch_loss += batch_loss;

            // Progress would be shown here in real training
            let _ = batch.num_rows(); // Use the batch
        }

        println!("   Epoch {}: avg_loss = {:.4}", epoch + 1, epoch_loss / 4.0);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
