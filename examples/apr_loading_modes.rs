#![allow(clippy::disallowed_methods)]
//! APR Loading Modes Example
//!
//! Demonstrates the loading subsystem for .apr model files with different
//! deployment targets (embedded, server, WASM) following Toyota Way principles:
//!
//! - **Heijunka**: Level resource demands during model initialization
//! - **Jidoka**: Quality built-in with verification at each layer
//! - **Poka-yoke**: Error-proofing via type-safe APIs
//!
//! Run with: `cargo run --example apr_loading_modes`

use aprender::loading::{
    platforms, Backend, BufferPool, LoadConfig, LoadResult, LoadingMode, PlatformSpecs,
    VerificationLevel,
};
use std::sync::Arc;
use std::time::Duration;

fn main() {
    println!("=== APR Loading Modes Demo ===\n");

    // Part 1: Loading Modes
    loading_modes_demo();

    // Part 2: Verification Levels
    verification_levels_demo();

    // Part 3: Deployment Configurations
    deployment_configs_demo();

    // Part 4: Buffer Pools
    buffer_pool_demo();

    // Part 5: WCET (Worst-Case Execution Time)
    wcet_demo();

    println!("\n=== Loading Modes Demo Complete! ===");
}

fn loading_modes_demo() {
    println!("--- Part 1: Loading Modes ---\n");

    let modes = [
        LoadingMode::Eager,
        LoadingMode::MappedDemand,
        LoadingMode::Streaming,
        LoadingMode::LazySection,
    ];

    println!(
        "{:<15} {:<40} {:>10} {:>12}",
        "Mode", "Description", "Zero-Copy", "Deterministic"
    );
    println!("{}", "-".repeat(80));

    for mode in &modes {
        println!(
            "{:<15} {:<40} {:>10} {:>12}",
            format!("{:?}", mode),
            mode.description(),
            if mode.supports_zero_copy() {
                "Yes"
            } else {
                "No"
            },
            if mode.is_deterministic() { "Yes" } else { "No" },
        );
    }

    // Memory budget-based mode selection
    println!("\nAutomatic mode selection based on memory budget:");
    let test_cases = [
        (200, 100, "2x model size"),
        (100, 100, "1x model size"),
        (512 * 1024, 1024 * 1024, "50% of model"),
        (64 * 1024, 1024 * 1024, "6% of model"),
    ];

    for (budget, model_size, description) in test_cases {
        let mode = LoadingMode::for_memory_budget(budget, model_size);
        println!("  Budget: {} ({}) -> {:?}", budget, description, mode);
    }
    println!();
}

fn verification_levels_demo() {
    println!("--- Part 2: Verification Levels ---\n");

    let levels = [
        VerificationLevel::UnsafeSkip,
        VerificationLevel::ChecksumOnly,
        VerificationLevel::Standard,
        VerificationLevel::Paranoid,
    ];

    println!(
        "{:<15} {:<10} {:<12} {:<10} {:<10}",
        "Level", "Checksum", "Signature", "ASIL", "DAL"
    );
    println!("{}", "-".repeat(60));

    for level in &levels {
        println!(
            "{:<15} {:<10} {:<12} {:<10} {:<10}",
            format!("{:?}", level),
            if level.verifies_checksum() {
                "Yes"
            } else {
                "No"
            },
            if level.verifies_signature() {
                "Yes"
            } else {
                "No"
            },
            level.asil_level(),
            level.dal_level(),
        );
    }
    println!();
}

fn deployment_configs_demo() {
    println!("--- Part 3: Deployment Configurations ---\n");

    // Embedded configuration (automotive ECU)
    let embedded = LoadConfig::embedded(1024 * 1024);
    println!("Embedded Configuration (1MB budget):");
    println!("  Mode: {:?}", embedded.mode);
    println!(
        "  Verification: {:?} ({})",
        embedded.verification,
        embedded.verification.asil_level()
    );
    println!("  Backend: {:?}", embedded.backend);
    println!("  Max Memory: {:?}", embedded.max_memory_bytes);
    println!("  Time Budget: {:?}", embedded.time_budget);

    // Server configuration
    let server = LoadConfig::server();
    println!("\nServer Configuration:");
    println!("  Mode: {:?}", server.mode);
    println!("  Verification: {:?}", server.verification);
    println!(
        "  Backend: {:?} (SIMD: {})",
        server.backend,
        server.backend.supports_simd()
    );
    println!("  Max Memory: {:?}", server.max_memory_bytes);

    // WASM configuration
    let wasm = LoadConfig::wasm();
    println!("\nWASM Configuration:");
    println!("  Mode: {:?}", wasm.mode);
    println!("  Verification: {:?}", wasm.verification);
    println!("  Backend: {:?}", wasm.backend);
    println!("  Max Memory: {:?}", wasm.max_memory_bytes);
    println!("  Streaming: {}", wasm.streaming);
    println!("  Ring Buffer: {} KB", wasm.ring_buffer_size / 1024);

    // Custom builder pattern
    let custom = LoadConfig::new()
        .with_mode(LoadingMode::Streaming)
        .with_max_memory(512 * 1024)
        .with_verification(VerificationLevel::Paranoid)
        .with_backend(Backend::CpuSimd)
        .with_time_budget(Duration::from_millis(50))
        .with_streaming(128 * 1024);

    println!("\nCustom Configuration (builder pattern):");
    println!("  Mode: {:?}", custom.mode);
    println!("  Verification: {:?}", custom.verification);
    println!("  Time Budget: {:?}", custom.time_budget);
    println!();
}

fn buffer_pool_demo() {
    println!("--- Part 4: Buffer Pools ---\n");

    // Create buffer pool for deterministic allocation
    let pool = BufferPool::new(4, 64 * 1024); // 4 buffers, 64KB each

    println!("Buffer Pool (deterministic allocation):");
    println!("  Buffer Count: {}", pool.total_count());
    println!("  Buffer Size: {} KB", pool.buffer_size() / 1024);
    println!("  Free Buffers: {}", pool.free_count());
    println!("  Total Memory: {} KB", pool.total_memory() / 1024);

    // Use pool in config
    let config = LoadConfig::new()
        .with_buffer_pool(Arc::new(pool))
        .with_mode(LoadingMode::Streaming);

    println!("\nConfig with Buffer Pool:");
    println!("  Has pool: {}", config.buffer_pool.is_some());
    println!();
}

fn wcet_demo() {
    println!("--- Part 5: WCET (Worst-Case Execution Time) ---\n");

    println!("Platform Specifications:");
    println!(
        "{:<20} {:>12} {:>12} {:>12}",
        "Platform", "Read MB/s", "Decomp MB/s", "Ed25519 µs"
    );
    println!("{}", "-".repeat(60));

    let platforms_list: [(&str, &PlatformSpecs); 3] = [
        ("Automotive S32G", &platforms::AUTOMOTIVE_S32G),
        ("Aerospace RAD750", &platforms::AEROSPACE_RAD750),
        ("Edge (RPi 4)", &platforms::EDGE_RPI4),
    ];

    for (name, specs) in &platforms_list {
        println!(
            "{:<20} {:>12.1} {:>12.1} {:>12.0}",
            name, specs.min_read_speed_mbps, specs.min_decomp_speed_mbps, specs.ed25519_verify_us,
        );
    }

    // Calculate WCET for different model sizes
    println!("\nWCET Estimates (10MB model, Zstd compressed):");

    // Simulate a model header
    let model_compressed_size = 5 * 1024 * 1024; // 5MB compressed
    let model_uncompressed_size = 10 * 1024 * 1024; // 10MB uncompressed

    for (name, specs) in &platforms_list {
        // Simplified WCET calculation (actual uses HeaderInfo)
        let read_time_ms =
            (model_compressed_size as f64 / (specs.min_read_speed_mbps * 1024.0 * 1024.0)) * 1000.0;
        let decomp_time_ms = (model_uncompressed_size as f64
            / (specs.min_decomp_speed_mbps * 1024.0 * 1024.0))
            * 1000.0;
        let verify_time_ms = specs.ed25519_verify_us / 1000.0;
        let total_ms = (read_time_ms + decomp_time_ms + verify_time_ms) * 1.1; // 10% safety margin

        println!(
            "  {:<20}: ~{:.1}ms (read: {:.1}ms, decomp: {:.1}ms, verify: {:.2}ms)",
            name, total_ms, read_time_ms, decomp_time_ms, verify_time_ms
        );
    }

    // Load result example
    println!("\nLoad Result Example:");
    let result = LoadResult::new(Duration::from_millis(45), 10 * 1024 * 1024);
    println!("  Load Time: {:?}", result.load_time);
    println!("  Memory Used: {} MB", result.memory_used / (1024 * 1024));
    println!("  Throughput: {:.1} MB/s", result.throughput_mbps());
    println!();
}
