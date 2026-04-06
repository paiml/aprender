#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::redundant_clone,
    clippy::use_debug,
    clippy::unnecessary_debug_formatting
)]
//! Streaming Large Datasets Example
//!
//! Demonstrates `StreamingDataset` for memory-efficient processing:
//! - Lazy loading from Parquet files
//! - Configurable buffer sizes
//! - Prefetching for reduced latency
//! - Chaining multiple data sources
//! - Memory source for testing
//!
//! Run with: cargo run --example streaming_large

use std::sync::Arc;

use alimentar::{
    streaming::{ChainedSource, MemorySource, StreamingDataset},
    ArrowDataset,
};
use arrow::{
    array::{Float64Array, Int32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn create_large_batch(start: i32, count: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let ids: Vec<i32> = (start..start + count as i32).collect();
    let values: Vec<f64> = ids.iter().map(|&i| i as f64 * 0.1).collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("batch creation failed")
}

fn main() -> alimentar::Result<()> {
    println!("=== Alimentar Streaming Dataset Example ===\n");

    // 1. Create StreamingDataset from MemorySource
    println!("1. Streaming from MemorySource");
    let batches = vec![
        create_large_batch(0, 1000),
        create_large_batch(1000, 1000),
        create_large_batch(2000, 1000),
    ];

    let source = MemorySource::new(batches.clone())?;
    let dataset = StreamingDataset::new(Box::new(source), 4);

    println!("   Schema: {:?}", dataset.schema());
    if let Some(hint) = dataset.size_hint() {
        println!("   Size hint: {} rows", hint);
    }

    let mut total_rows = 0;
    let mut batch_count = 0;
    for batch in dataset {
        total_rows += batch.num_rows();
        batch_count += 1;
    }
    println!(
        "   Processed {} batches, {} total rows",
        batch_count, total_rows
    );

    // 2. With prefetch for reduced latency
    println!("\n2. StreamingDataset with prefetch=2");
    let source = MemorySource::new(batches.clone())?;
    let dataset = StreamingDataset::new(Box::new(source), 4).prefetch(2);

    let mut batch_count = 0;
    for batch in dataset {
        batch_count += 1;
        if batch_count <= 3 {
            println!("   Batch {}: {} rows", batch_count, batch.num_rows());
        }
    }
    println!("   ... ({} batches total)", batch_count);

    // 3. Streaming from Parquet file
    println!("\n3. Streaming from Parquet file");

    // Create a sample parquet file first
    let temp_dir = std::env::temp_dir();
    let parquet_path = temp_dir.join("streaming_example.parquet");

    let sample_dataset = ArrowDataset::from_batch(create_large_batch(0, 5000))?;
    sample_dataset.to_parquet(&parquet_path)?;
    println!("   Created sample Parquet file: {:?}", parquet_path);

    // Stream from the parquet file
    let streaming = StreamingDataset::from_parquet(&parquet_path, 512)?;
    println!("   Batch size: 512 rows");

    if let Some(hint) = streaming.size_hint() {
        println!("   Total rows (from size_hint): {}", hint);
    }

    let mut batch_count = 0;
    let mut total_rows = 0;
    for batch in streaming {
        total_rows += batch.num_rows();
        batch_count += 1;
    }
    println!("   Streamed {} batches, {} rows", batch_count, total_rows);

    // 4. Chaining multiple sources
    println!("\n4. Chaining multiple data sources");

    let source1 = MemorySource::new(vec![create_large_batch(0, 500)])?;
    let source2 = MemorySource::new(vec![create_large_batch(500, 500)])?;
    let source3 = MemorySource::new(vec![create_large_batch(1000, 500)])?;

    let chained = ChainedSource::new(vec![
        Box::new(source1),
        Box::new(source2),
        Box::new(source3),
    ])?;

    let dataset = StreamingDataset::new(Box::new(chained), 4);
    println!("   Chained 3 sources");

    let mut total_rows = 0;
    for batch in dataset {
        total_rows += batch.num_rows();
    }
    println!("   Total rows from chain: {}", total_rows);

    // 5. Processing large datasets in chunks
    println!("\n5. Processing large data in streaming fashion");

    let large_batches: Vec<_> = (0..10)
        .map(|i| create_large_batch(i * 1000, 1000))
        .collect();

    let source = MemorySource::new(large_batches)?;
    let dataset = StreamingDataset::new(Box::new(source), 2).prefetch(2);

    println!("   Simulating ML preprocessing pipeline...");
    let mut processed_count = 0;
    let mut sum = 0.0;

    for batch in dataset {
        // Simulate some processing
        if let Some(values) = batch.column(1).as_any().downcast_ref::<Float64Array>() {
            for val in values.iter().flatten() {
                sum += val;
                processed_count += 1;
            }
        }

        // In real scenario, you might:
        // - Apply transforms
        // - Send to GPU
        // - Write to output file
    }

    println!("   Processed {} values", processed_count);
    println!("   Mean value: {:.4}", sum / processed_count as f64);

    // 6. Memory-efficient iteration pattern
    println!("\n6. Memory-efficient iteration (constant memory)");

    // This pattern processes data without loading everything into memory
    let batches = vec![
        create_large_batch(0, 2000),
        create_large_batch(2000, 2000),
        create_large_batch(4000, 2000),
    ];

    let source = MemorySource::new(batches)?;
    let dataset = StreamingDataset::new(Box::new(source), 1); // buffer_size=1

    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;

    for batch in dataset {
        if let Some(values) = batch.column(1).as_any().downcast_ref::<Float64Array>() {
            for val in values.iter().flatten() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }
    }

    println!("   Min value: {:.4}", min_val);
    println!("   Max value: {:.4}", max_val);
    println!("   (Only 1 batch in memory at a time)");

    // Cleanup
    let _ = std::fs::remove_file(&parquet_path);

    println!("\n=== Example Complete ===");
    Ok(())
}
