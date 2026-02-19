#![allow(clippy::disallowed_methods)]
//! Demo: Model Bundling with Renacer Tracing
//!
//! This example demonstrates the bundle module and can be traced with renacer
//! to see the actual system calls for file I/O and memory operations.
//!
//! Run with tracing:
//!   cargo build --example bundle_trace_demo
//!   renacer -s -e trace=file -T -- ./target/debug/examples/bundle_trace_demo

use aprender::bundle::{BundleBuilder, BundleConfig, ModelBundle, PagedBundle, PagingConfig};
use std::path::Path;

fn main() {
    println!("=== Model Bundling and Memory Paging Demo ===\n");

    let bundle_path = "/tmp/demo_bundle.apbundle";

    // Step 1: Create a bundle with multiple models
    println!("1. Creating bundle with 3 models...");
    create_bundle(bundle_path);

    // Step 2: Load entire bundle into memory
    println!("\n2. Loading bundle into memory...");
    load_bundle_full(bundle_path);

    // Step 3: Load with memory paging
    println!("\n3. Loading with memory paging (limited to 1KB)...");
    load_bundle_paged(bundle_path);

    // Cleanup
    let _ = std::fs::remove_file(bundle_path);
    println!("\n=== Demo Complete ===");
}

fn create_bundle(path: &str) {
    // Create some fake model weights
    let encoder_weights = vec![1u8; 500]; // 500 bytes
    let decoder_weights = vec![2u8; 500]; // 500 bytes
    let classifier_weights = vec![3u8; 300]; // 300 bytes

    println!("   - Encoder: {} bytes", encoder_weights.len());
    println!("   - Decoder: {} bytes", decoder_weights.len());
    println!("   - Classifier: {} bytes", classifier_weights.len());

    let bundle = BundleBuilder::new(path)
        .with_config(BundleConfig::new().with_compression(false))
        .add_model("encoder", encoder_weights)
        .add_model("decoder", decoder_weights)
        .add_model("classifier", classifier_weights)
        .build()
        .expect("Failed to create bundle");

    println!("   Bundle created with {} models", bundle.len());
    println!("   Total size: {} bytes", bundle.total_size());
}

fn load_bundle_full(path: &str) {
    let bundle = ModelBundle::load(Path::new(path)).expect("Failed to load bundle");

    println!("   Loaded {} models:", bundle.len());
    for name in bundle.model_names() {
        if let Some(data) = bundle.get_model(name) {
            println!("   - {}: {} bytes", name, data.len());
        }
    }
}

fn load_bundle_paged(path: &str) {
    // Use tiny memory limit to force paging behavior
    let config = PagingConfig::new()
        .with_max_memory(1024) // 1KB limit
        .with_prefetch(false);

    let mut bundle =
        PagedBundle::open(Path::new(path), config).expect("Failed to open paged bundle");

    println!("   Memory limit: {} bytes", bundle.config().max_memory);
    println!("   Initially cached: {} models", bundle.cached_count());

    // Access encoder - should load it
    println!("\n   Accessing encoder...");
    let encoder = bundle.get_model("encoder").expect("Failed to get encoder");
    println!("   - Loaded encoder: {} bytes", encoder.len());
    println!(
        "   - Cached: {}, Memory used: {} bytes",
        bundle.cached_count(),
        bundle.memory_used()
    );

    // Access decoder - should evict encoder (1024 limit)
    println!("\n   Accessing decoder...");
    let decoder = bundle.get_model("decoder").expect("Failed to get decoder");
    println!("   - Loaded decoder: {} bytes", decoder.len());
    println!(
        "   - Cached: {}, Memory used: {} bytes",
        bundle.cached_count(),
        bundle.memory_used()
    );

    // Access classifier - should evict decoder
    println!("\n   Accessing classifier...");
    let classifier = bundle
        .get_model("classifier")
        .expect("Failed to get classifier");
    println!("   - Loaded classifier: {} bytes", classifier.len());
    println!(
        "   - Cached: {}, Memory used: {} bytes",
        bundle.cached_count(),
        bundle.memory_used()
    );

    // Show paging statistics
    let stats = bundle.stats();
    println!("\n   Paging Statistics:");
    println!("   - Hits: {}", stats.hits);
    println!("   - Misses: {}", stats.misses);
    println!("   - Evictions: {}", stats.evictions);
    println!("   - Hit rate: {:.1}%", stats.hit_rate() * 100.0);
    println!("   - Total bytes loaded: {}", stats.bytes_loaded);
}
