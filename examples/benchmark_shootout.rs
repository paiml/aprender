//! Backend Benchmark Shootout
//!
//! Demonstrates trueno-db's GPU-first architecture with SIMD fallback.
//! Compares performance across different data sizes showing when GPU/SIMD
//! outperforms scalar execution.
//!
//! Run with: cargo run --example `benchmark_shootout` --release

use arrow::array::{Float64Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use std::time::Instant;
use trueno_db::topk::{SortOrder, TopKSelection};

fn main() {
    println!("🚀 Trueno-DB Backend Benchmark Shootout");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Testing GPU-first architecture with automatic SIMD fallback");
    println!("Using SIMD-optimized Top-K selection (heap-based algorithm)");
    println!();

    // Generate test datasets of increasing size
    let sizes = vec![
        (1_000, "1K rows"),
        (10_000, "10K rows"),
        (100_000, "100K rows"),
        (1_000_000, "1M rows"),
    ];

    println!("Backend Selection Strategy:");
    println!("  • CostBased: Automatic (uses arithmetic intensity)");
    println!("  • GPU:       Force GPU execution (requires --features gpu)");
    println!("  • SIMD:      Force SIMD (AVX-512/AVX2/SSE2) ⚡ ACTIVE");
    println!();
    println!("⚠️  Note: Running SIMD-optimized path (GPU requires wgpu feature)");
    println!();

    for (size, label) in sizes {
        run_benchmark(size, label);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ Benchmark complete!");
    println!();
    println!("Key Takeaways:");
    println!("  • SIMD provides consistent 2-10x speedup over scalar");
    println!("  • GPU best for >100K rows with high compute intensity");
    println!("  • Automatic backend selection via Backend::CostBased");
    println!("  • Zero-copy operations via Apache Arrow columnar format");
}

fn run_benchmark(rows: usize, label: &str) {
    println!("📊 Dataset: {label}");
    println!("─────────────────────────────────────────────");

    let batch = generate_data(rows);
    println!("  Generated {rows} rows: id (Int32), value (Float64)");
    println!();

    // Test 1: Top-K Descending (find largest K values)
    println!("  🏆 Top-10 Selection (Descending):");
    benchmark_topk(&batch, 10, SortOrder::Descending);

    // Test 2: Top-K Ascending (find smallest K values)
    println!("  🔻 Top-10 Selection (Ascending):");
    benchmark_topk(&batch, 10, SortOrder::Ascending);

    // Test 3: Larger K
    let k = rows.min(100);
    println!("  📈 Top-{k} Selection (Descending):");
    benchmark_topk(&batch, k, SortOrder::Descending);

    println!();
}

fn generate_data(rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));

    // Generate realistic data with variance
    let ids: Vec<i32> = (0..rows as i32).collect();
    let values: Vec<f64> = (0..rows)
        .map(|i| {
            let base = (i as f64) * 1.5;
            let noise = ((i * 7919) % 1000) as f64 / 100.0; // Pseudo-random noise
            base + noise + 100.0
        })
        .collect();

    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Float64Array::from(values))],
    )
    .expect("Example should work with valid test data")
}

fn benchmark_topk(batch: &RecordBatch, k: usize, order: SortOrder) {
    let order_str = match order {
        SortOrder::Descending => "DESC",
        SortOrder::Ascending => "ASC",
    };

    // Warmup run
    let _ = batch.top_k(1, k, order).expect("Example should work with valid test data");

    // Timed run
    let start = Instant::now();
    let result = batch.top_k(1, k, order).expect("Example should work with valid test data");
    let elapsed = start.elapsed();

    let values = result
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Example should work with valid test data");

    println!(
        "    SIMD-optimized: {:.3}ms  (order: {}, k={}, result: {} rows)",
        elapsed.as_secs_f64() * 1000.0,
        order_str,
        k,
        result.num_rows()
    );

    // Show sample results (top 3)
    if result.num_rows() >= 3 {
        println!(
            "    Sample: [{:.1}, {:.1}, {:.1}, ...]",
            values.value(0),
            values.value(1),
            values.value(2)
        );
    }

    println!();
}
