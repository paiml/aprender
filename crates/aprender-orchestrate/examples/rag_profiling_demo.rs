//! RAG Profiling Demo
//!
//! Demonstrates the RAG query optimization and profiling infrastructure.
//!
//! Run with: cargo run --example rag_profiling_demo

use batuta::oracle::rag::profiling::{span, Counter, Histogram, GLOBAL_METRICS};
use std::time::Duration;

fn main() {
    println!("=== RAG Profiling Demo ===\n");

    // 1. Histogram usage for latency tracking
    println!("1. Histogram for latency tracking");
    let histogram = Histogram::new();

    // Simulate some query latencies
    for latency_ms in [12, 15, 8, 22, 11, 9, 18, 14] {
        histogram.observe(Duration::from_millis(latency_ms));
    }

    println!("   Observations: {}", histogram.count());
    println!("   Sum: {:.2}ms", histogram.sum_ms());
    println!("   p50: {:.2}ms", histogram.percentile(50.0));
    println!("   p90: {:.2}ms", histogram.percentile(90.0));
    println!("   p99: {:.2}ms", histogram.percentile(99.0));

    // 2. Counter usage for cache statistics
    println!("\n2. Counter for cache statistics");
    let hits = Counter::new();
    let misses = Counter::new();

    // Simulate cache behavior
    hits.inc_by(45);
    misses.inc_by(15);

    let hit_rate = hits.get() as f64 / (hits.get() + misses.get()) as f64;
    println!("   Cache hits: {}", hits.get());
    println!("   Cache misses: {}", misses.get());
    println!("   Hit rate: {:.1}%", hit_rate * 100.0);

    // 3. Timed spans for profiling
    println!("\n3. Timed span instrumentation");
    {
        let _span = span("bm25_search");
        // Simulate BM25 search work
        std::thread::sleep(Duration::from_millis(5));
    }
    {
        let _span = span("tfidf_search");
        // Simulate TF-IDF search work
        std::thread::sleep(Duration::from_millis(3));
    }
    {
        let _span = span("rrf_fusion");
        // Simulate RRF fusion work
        std::thread::sleep(Duration::from_millis(1));
    }

    let summary = GLOBAL_METRICS.summary();
    println!("   Recorded spans:");
    for (name, stats) in &summary.spans {
        println!("     {}: {:.2}ms (count: {})", name, stats.total_us as f64 / 1000.0, stats.count);
    }

    // 4. Global metrics
    println!("\n4. Global metrics overview");
    println!("   Cache hit rate: {:.1}%", GLOBAL_METRICS.cache_hit_rate() * 100.0);

    // 5. Demonstrating profiling workflow
    println!("\n5. Typical profiling workflow:");
    println!("   a. Create spans around operations");
    println!("   b. Spans auto-record duration on drop");
    println!("   c. Query GLOBAL_METRICS.summary() for stats");
    println!("   d. Use --rag-profile flag for CLI output");

    println!("\n=== Demo Complete ===");
    println!("Use 'batuta oracle --rag \"query\" --rag-profile' for CLI profiling.");
}
