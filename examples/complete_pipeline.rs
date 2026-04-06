//! Complete Trueno-DB Pipeline: Storage → Backend Selection → Top-K
//!
//! This example demonstrates the complete workflow:
//! 1. Load data with `StorageEngine`
//! 2. Process with morsel iteration
//! 3. Apply backend selection (GPU vs SIMD)
//! 4. Execute Top-K selection
//!
//! Run with: cargo run --example `complete_pipeline` --release

use arrow::array::{Float64Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use std::time::Instant;
use trueno_db::backend::BackendDispatcher;
use trueno_db::storage::StorageEngine;
use trueno_db::topk::{SortOrder, TopKSelection};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         TRUENO-DB COMPLETE PIPELINE DEMONSTRATION           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Create and load data
    println!("┌─ STEP 1: DATA LOADING ─────────────────────────────────────┐");
    let start = Instant::now();
    let batch = create_sample_data(5_000_000)?;
    let load_time = start.elapsed();

    println!("│ Dataset created:");
    println!("│   Rows: {}", batch.num_rows());
    println!("│   Columns: {}", batch.num_columns());
    println!("│   Memory: {:.2} MB", batch.get_array_memory_size() as f64 / 1_048_576.0);
    println!("│   Time: {load_time:?}");

    let schema = batch.schema();
    println!("│ Schema:");
    for field in schema.fields() {
        println!("│   - {}: {:?}", field.name(), field.data_type());
    }
    println!("└────────────────────────────────────────────────────────────┘\n");

    // Step 2: Initialize storage engine
    println!("┌─ STEP 2: STORAGE ENGINE ───────────────────────────────────┐");
    let mut storage = StorageEngine::new(vec![]);
    storage.append_batch(batch.clone())?;

    println!("│ Storage engine initialized:");
    println!("│   Pattern: OLAP (append-only)");
    println!("│   Batches: {}", storage.batches().len());
    println!(
        "│   Total rows: {}",
        storage.batches().iter().map(arrow::array::RecordBatch::num_rows).sum::<usize>()
    );
    println!("└────────────────────────────────────────────────────────────┘\n");

    // Step 3: Morsel iteration
    println!("┌─ STEP 3: MORSEL ITERATION (Out-of-Core) ──────────────────┐");
    println!("│ Morsel size: 128 MB chunks");
    println!("│ Purpose: Prevent GPU VRAM OOM\n│");

    let mut morsel_count = 0;
    let mut total_morsel_rows = 0;

    for morsel in storage.morsels() {
        morsel_count += 1;
        total_morsel_rows += morsel.num_rows();

        if morsel_count <= 3 {
            let size_mb = morsel.get_array_memory_size() as f64 / 1_048_576.0;
            println!("│   Morsel #{}: {} rows, {:.2} MB", morsel_count, morsel.num_rows(), size_mb);
        }
    }

    if morsel_count > 3 {
        println!("│   ... ({} more morsels)", morsel_count - 3);
    }

    println!("│");
    println!("│ Summary:");
    println!("│   Total morsels: {morsel_count}");
    println!("│   Total rows: {total_morsel_rows}");
    println!("│   Integrity check: {} ✓", total_morsel_rows == batch.num_rows());
    println!("└────────────────────────────────────────────────────────────┘\n");

    // Step 4: Backend selection
    println!("┌─ STEP 4: BACKEND SELECTION (Cost-Based) ──────────────────┐");
    let data_bytes = batch.get_array_memory_size();
    let estimated_flops = batch.num_rows() as f64 * 10.0; // Simple aggregation

    let pcie_transfer_ms = (data_bytes as f64 / (32.0 * 1_000_000_000.0)) * 1000.0;
    let gpu_compute_ms = (estimated_flops / (100.0 * 1_000_000_000.0)) * 1000.0;

    let backend = BackendDispatcher::select(data_bytes, estimated_flops);

    println!("│ Cost model:");
    println!("│   Data size: {:.2} MB", data_bytes as f64 / 1_048_576.0);
    println!("│   Estimated FLOPs: {estimated_flops:.0}");
    println!("│   PCIe transfer time: {pcie_transfer_ms:.3} ms");
    println!("│   GPU compute time: {gpu_compute_ms:.3} ms");
    println!("│   Ratio: {:.2}x (compute/transfer)", gpu_compute_ms / pcie_transfer_ms);
    println!("│");
    println!("│ Decision:");
    println!("│   Selected backend: {backend:?}");

    match backend {
        trueno_db::Backend::Gpu => {
            println!("│   Rationale: Compute > 5x transfer (GPU efficient)");
            println!("│   Note: GPU kernels in Phase 2, falling back to SIMD");
        }
        trueno_db::Backend::Simd => {
            println!("│   Rationale: Transfer overhead too high");
            println!("│   SIMD features: AVX-512/AVX2/SSE2 (auto-detect)");
        }
        _ => {}
    }
    println!("└────────────────────────────────────────────────────────────┘\n");

    // Step 5: Top-K selection
    println!("┌─ STEP 5: TOP-K SELECTION ──────────────────────────────────┐");
    println!("│ Algorithm: O(N log K) heap-based selection");
    println!("│");

    // Top 10 highest scores
    let start = Instant::now();
    let top10_high = batch.top_k(2, 10, SortOrder::Descending)?;
    let topk_time = start.elapsed();

    println!("│ Top-10 Highest Scores:");
    let score_col = top10_high
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Example should work with valid test data");
    let id_col = top10_high
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Example should work with valid test data");
    let user_col = top10_high
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Example should work with valid test data");

    for i in 0..top10_high.num_rows().min(5) {
        println!(
            "│   #{}: user_id={}, username={}, score={:.2}",
            i + 1,
            id_col.value(i),
            user_col.value(i),
            score_col.value(i)
        );
    }
    if top10_high.num_rows() > 5 {
        println!("│   ... ({} more)", top10_high.num_rows() - 5);
    }

    println!("│");
    println!("│ Performance:");
    println!("│   Time: {topk_time:?}");
    println!(
        "│   Throughput: {:.2} M rows/sec",
        batch.num_rows() as f64 / 1_000_000.0 / topk_time.as_secs_f64()
    );
    println!("└────────────────────────────────────────────────────────────┘\n");

    // Summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    PIPELINE SUMMARY                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("✓ Data loaded: 5M rows ({:.2} MB)", data_bytes as f64 / 1_048_576.0);
    println!("✓ Storage: OLAP append-only pattern");
    println!("✓ Morsels: {morsel_count} chunks (out-of-core execution)");
    println!("✓ Backend: {backend:?} (cost-based selection)");
    println!("✓ Top-K: 10 results in {topk_time:?}");
    println!();
    println!("Phase 1 MVP Features Demonstrated:");
    println!("  1. Arrow/Parquet storage engine");
    println!("  2. Morsel-based iteration (128 MB chunks)");
    println!("  3. OLAP write pattern (append_batch)");
    println!("  4. Backend dispatcher (GPU vs SIMD selection)");
    println!("  5. Top-K selection (heap-based algorithm)");
    println!("  6. SIMD integration (via trueno crate)");
    println!();
    println!("Phase 2 Roadmap (GPU Kernels):");
    println!("  - Actual wgpu compute shaders");
    println!("  - GPU device initialization");
    println!("  - PCIe bandwidth runtime calibration");
    println!("  - Multi-GPU data partitioning");
    println!();

    Ok(())
}

fn create_sample_data(num_rows: usize) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    use rand::Rng;

    let schema = Schema::new(vec![
        Field::new("user_id", DataType::Int32, false),
        Field::new("username", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
    ]);

    let mut rng = rand::thread_rng();

    let user_ids: Vec<i32> = (0..num_rows).map(|i| i as i32).collect();
    let usernames: Vec<String> = (0..num_rows).map(|i| format!("user_{i}")).collect();
    let scores: Vec<f64> = (0..num_rows).map(|_| rng.gen_range(0.0..1000.0)).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(user_ids)),
            Arc::new(StringArray::from(usernames)),
            Arc::new(Float64Array::from(scores)),
        ],
    )?;

    Ok(batch)
}
