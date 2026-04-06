//! GPU-Accelerated Sales Analytics
//!
//! This example demonstrates a realistic database scenario:
//! Analyzing sales transactions with GPU acceleration.
//!
//! Dataset: 500,000 sales transactions
//! Query: Find top sales, filter by region, compute statistics
//!
//! Run with: cargo run --example gpu_sales_analytics --features gpu --release

use arrow::array::Int32Array;
use rand::Rng;
use std::time::Instant;
use trueno_db::gpu::GpuEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU-Accelerated Sales Analytics Dashboard ===\n");

    // Initialize GPU
    println!("🔧 Initializing GPU compute engine...");
    let start = Instant::now();
    let gpu = match GpuEngine::new().await {
        Ok(engine) => {
            println!("✅ GPU initialized in {:?}", start.elapsed());
            engine
        }
        Err(e) => {
            eprintln!("❌ GPU not available: {}", e);
            eprintln!("   Falling back to CPU would occur in production");
            return Ok(());
        }
    };
    println!();

    // Generate synthetic sales data
    println!("📊 Generating sales dataset (500,000 transactions)...");
    let mut rng = rand::thread_rng();
    let num_transactions = 500_000;

    // Sales amounts: $1 to $1000
    let sales_data: Vec<i32> = (0..num_transactions).map(|_| rng.gen_range(1..=1000)).collect();

    println!("   Generated {} transactions", sales_data.len());
    println!("   Amount range: $1 - $1,000 per transaction");
    println!();

    // Query 1: Total Sales Revenue
    println!("=== Query 1: Total Sales Revenue ===");
    println!("SQL: SELECT SUM(amount) FROM sales");

    let sales_array = Int32Array::from(sales_data.clone());
    let start = Instant::now();
    let total_revenue = gpu.sum_i32(&sales_array).await?;
    let gpu_time = start.elapsed();

    println!("  GPU Execution Time: {:?}", gpu_time);
    println!("  Total Revenue: ${}", total_revenue);
    println!("  Transactions: {}", num_transactions);
    println!("  Average: ${:.2}", total_revenue as f64 / num_transactions as f64);
    println!();

    // Query 2: Find Minimum Sale
    println!("=== Query 2: Minimum Sale Amount ===");
    println!("SQL: SELECT MIN(amount) FROM sales");

    let start = Instant::now();
    let min_sale = gpu.min_i32(&sales_array).await?;
    let gpu_time = start.elapsed();

    println!("  GPU Execution Time: {:?}", gpu_time);
    println!("  Minimum Sale: ${}", min_sale);
    println!();

    // Query 3: Find Maximum Sale
    println!("=== Query 3: Maximum Sale Amount ===");
    println!("SQL: SELECT MAX(amount) FROM sales");

    let start = Instant::now();
    let max_sale = gpu.max_i32(&sales_array).await?;
    let gpu_time = start.elapsed();

    println!("  GPU Execution Time: {:?}", gpu_time);
    println!("  Maximum Sale: ${}", max_sale);
    println!();

    // Query 4: High-Value Transactions
    println!("=== Query 4: High-Value Transactions (>$500) ===");
    println!("SQL: SELECT SUM(amount) FROM sales WHERE amount > 500");

    let start = Instant::now();
    let high_value_sum = gpu.fused_filter_sum(&sales_array, 500, "gt").await?;
    let gpu_time = start.elapsed();

    let high_value_count = sales_data.iter().filter(|&&x| x > 500).count();

    println!("  GPU Execution Time: {:?}", gpu_time);
    println!("  Total High-Value Revenue: ${}", high_value_sum);
    println!("  High-Value Transaction Count: {}", high_value_count);
    println!("  Percentage: {:.1}%", (high_value_count as f64 / num_transactions as f64) * 100.0);
    println!("  🎯 Toyota Way: Kernel fusion (filter+sum in single GPU pass)");
    println!();

    // Query 5: Low-Value Transactions
    println!("=== Query 5: Low-Value Transactions (≤$100) ===");
    println!("SQL: SELECT SUM(amount) FROM sales WHERE amount <= 100");

    let start = Instant::now();
    let low_value_sum = gpu.fused_filter_sum(&sales_array, 100, "lte").await?;
    let gpu_time = start.elapsed();

    let low_value_count = sales_data.iter().filter(|&&x| x <= 100).count();

    println!("  GPU Execution Time: {:?}", gpu_time);
    println!("  Total Low-Value Revenue: ${}", low_value_sum);
    println!("  Low-Value Transaction Count: {}", low_value_count);
    println!("  Percentage: {:.1}%", (low_value_count as f64 / num_transactions as f64) * 100.0);
    println!();

    // Query 6: Mid-Range Transactions
    println!("=== Query 6: Mid-Range Transactions ($250-$750) ===");
    println!("SQL: SELECT COUNT(*) FROM sales WHERE amount >= 250 AND amount <= 750");

    // For this we'd need a compound filter, so we'll use CPU for now
    let mid_range_count = sales_data.iter().filter(|&&x| (250..=750).contains(&x)).count();
    let mid_range_sum: i64 =
        sales_data.iter().filter(|&&x| (250..=750).contains(&x)).map(|&x| x as i64).sum();

    println!("  Mid-Range Transaction Count: {}", mid_range_count);
    println!("  Mid-Range Revenue: ${}", mid_range_sum);
    println!("  Percentage: {:.1}%", (mid_range_count as f64 / num_transactions as f64) * 100.0);
    println!("  Note: Compound filters will use GPU in Phase 2");
    println!();

    // Performance Summary
    println!("=== Performance Summary ===");
    println!(
        "  Dataset Size: {} transactions ({:.1} MB)",
        num_transactions,
        (num_transactions * 4) as f64 / 1_048_576.0
    );
    println!("  GPU Backend: wgpu (Vulkan/Metal/DX12)");
    println!("  Workgroup Size: 256 threads");
    println!("  Memory Model: Zero-copy Arrow columnar format");
    println!();

    println!("=== Dashboard Insights ===");
    println!("  📈 Total Revenue: ${}", total_revenue);
    println!(
        "  💎 High-Value (>$500): ${} ({:.1}%)",
        high_value_sum,
        (high_value_count as f64 / num_transactions as f64) * 100.0
    );
    println!(
        "  💰 Mid-Range ($250-$750): ${} ({:.1}%)",
        mid_range_sum,
        (mid_range_count as f64 / num_transactions as f64) * 100.0
    );
    println!(
        "  📊 Low-Value (≤$100): ${} ({:.1}%)",
        low_value_sum,
        (low_value_count as f64 / num_transactions as f64) * 100.0
    );
    println!("  🔽 Min Transaction: ${}", min_sale);
    println!("  🔼 Max Transaction: ${}", max_sale);
    println!();

    println!("🎉 Sales analytics complete!");
    println!("   GPU provided real-time aggregations across 500K transactions");

    Ok(())
}
