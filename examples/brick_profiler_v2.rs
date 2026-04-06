#![allow(clippy::disallowed_methods)]
//! PAR-200: BrickProfiler v2 Demo
//!
//! Run with: cargo run --example brick_profiler_v2

use std::thread::sleep;
use std::time::Duration;
use trueno::{BrickCategory, BrickId, BrickProfiler, SyncMode};

fn main() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    println!("=== PAR-200: BrickProfiler v2 Demo ===\n");

    // Simulate a transformer forward pass (3 layers)
    for _layer in 0..3 {
        // RmsNorm (Norm category)
        let t = profiler.start_brick(BrickId::RmsNorm);
        sleep(Duration::from_micros(50));
        profiler.stop_brick(t, 1);

        // QKV Projection (Attention category)
        let t = profiler.start_brick(BrickId::QkvProjection);
        sleep(Duration::from_micros(200));
        profiler.stop_brick(t, 1);

        // Attention Score
        let t = profiler.start_brick(BrickId::AttentionScore);
        sleep(Duration::from_micros(150));
        profiler.stop_brick(t, 1);

        // FFN Gate (FFN category)
        let t = profiler.start_brick(BrickId::GateProjection);
        sleep(Duration::from_micros(300));
        profiler.stop_brick(t, 1);

        // FFN Down
        let t = profiler.start_brick(BrickId::DownProjection);
        sleep(Duration::from_micros(300));
        profiler.stop_brick(t, 1);
    }

    // Print per-brick stats
    println!("Per-Brick Timing:");
    println!("{:20} {:>10} {:>10} {:>8}", "Brick", "Avg (µs)", "Total (µs)", "Count");
    println!("{}", "-".repeat(52));

    for brick_id in [
        BrickId::RmsNorm,
        BrickId::QkvProjection,
        BrickId::AttentionScore,
        BrickId::GateProjection,
        BrickId::DownProjection,
    ] {
        let stats = profiler.brick_stats(brick_id);
        if stats.count > 0 {
            println!(
                "{:20} {:>10.1} {:>10.1} {:>8}",
                brick_id.name(),
                stats.avg_us(),
                stats.total_ns as f64 / 1000.0,
                stats.count
            );
        }
    }

    // Print category breakdown
    println!("\nCategory Breakdown:");
    println!("{:12} {:>10} {:>8} {:>10}", "Category", "Avg (µs)", "Pct", "Samples");
    println!("{}", "-".repeat(44));

    let cats = profiler.category_stats();
    let total = profiler.total_ns();

    for cat in [BrickCategory::Norm, BrickCategory::Attention, BrickCategory::Ffn] {
        let cs = &cats[cat as usize];
        if cs.count > 0 {
            println!(
                "{:12} {:>10.1} {:>7.1}% {:>10}",
                cat.name(),
                cs.avg_us(),
                cs.percentage(total),
                cs.count
            );
        }
    }

    println!(
        "\nTotal: {} tokens, {:.1}µs, {:.0} tok/s",
        profiler.total_tokens(),
        profiler.total_ns() as f64 / 1000.0,
        profiler.total_throughput()
    );

    // Demo deferred sync mode
    println!("\n=== Deferred Sync Mode Demo ===\n");
    profiler.reset();
    profiler.set_sync_mode(SyncMode::Deferred);
    profiler.reset_epoch();

    // Record without immediate sync (simulates GPU async ops)
    let s1 = profiler.elapsed_ns();
    sleep(Duration::from_micros(100));
    profiler.record_deferred(BrickId::RmsNorm, s1, 1);

    let s2 = profiler.elapsed_ns();
    sleep(Duration::from_micros(200));
    profiler.record_deferred(BrickId::QkvProjection, s2, 1);

    println!("Pending measurements: {}", profiler.pending_count());

    // Simulate GPU sync point
    let end = profiler.elapsed_ns();
    profiler.finalize(end);

    println!(
        "After finalize: {} measurements applied",
        profiler.brick_stats(BrickId::RmsNorm).count
            + profiler.brick_stats(BrickId::QkvProjection).count
    );

    println!("\n✓ PAR-200 BrickProfiler v2 working correctly");
}
