//! Memory-Mapped APR Model Loading Example
//!
//! Demonstrates zero-copy model loading using mmap to reduce zram pressure
//! when loading large models for GPU inference.
//!
//! ## Problem
//!
//! When loading multi-GB model files with `fs::read()`:
//! 1. Entire file is copied into userspace heap
//! 2. Data transferred to GPU VRAM
//! 3. CPU copy sits idle, gets compressed to zram
//! 4. zram holds stale data indefinitely (e.g., 47GB using 10GB real RAM)
//!
//! ## Solution
//!
//! Memory-mapped loading with `release_cpu_pages()`:
//! 1. File is mmap'd (zero-copy, kernel manages pages)
//! 2. Data transferred to GPU VRAM
//! 3. Call `release_cpu_pages()` → kernel drops pages immediately
//! 4. Pages re-fault from disk if needed (not zram)
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example apr_mmap_loading
//! cargo run --example apr_mmap_loading -- /path/to/model.apr
//! ```
//!
//! ## References
//!
//! - Didona et al. (2022): mmap vs read() achieves 2.3x throughput
//! - See docs/model-loading.md for full design rationale

use std::env;
use std::time::Instant;

use realizar::apr::{AprV2Model, HEADER_SIZE, MAGIC};
use realizar::error::Result;

fn main() -> Result<()> {
    println!("=== Memory-Mapped APR Loading Demo ===\n");

    // Check for model path argument
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        // Load real model
        let model_path = &args[1];
        demo_real_model(model_path)?;
    } else {
        // Demo with synthetic model
        demo_synthetic_model()?;
    }

    println!("\n=== Demo Complete ===");
    Ok(())
}

/// Demonstrate mmap loading with a real model file
fn demo_real_model(path: &str) -> Result<()> {
    println!("Loading model: {}\n", path);

    // Time the load
    let start = Instant::now();
    let model = AprV2Model::load(path)?;
    let load_time = start.elapsed();

    println!("Model Info:");
    println!("  Tensors: {}", model.tensor_count());
    println!("  Parameters: ~{}", model.estimated_parameters());
    println!("  Load time: {:?}", load_time);
    println!("  Using mmap: {}", model.is_mmap());
    println!();

    // Show tensor names (first 10)
    let names = model.tensor_names();
    println!("Tensors (first 10 of {}):", names.len());
    for name in names.iter().take(10) {
        println!("  - {}", name);
    }
    if names.len() > 10 {
        println!("  ... and {} more", names.len() - 10);
    }
    println!();

    // Simulate GPU transfer
    println!("Simulating GPU transfer...");
    let transfer_start = Instant::now();

    for name in model.tensor_names() {
        // In real code: cuda::memcpy_htod(gpu_ptr, bytes)
        let _bytes = model.get_tensor_bytes(name)?;
    }

    let transfer_time = transfer_start.elapsed();
    println!("  Transfer time: {:?}", transfer_time);
    println!();

    // Release CPU pages (the key optimization!)
    #[cfg(unix)]
    {
        println!("Releasing CPU pages (madvise MADV_DONTNEED)...");
        let release_start = Instant::now();
        model.release_cpu_pages()?;
        let release_time = release_start.elapsed();
        println!("  Release time: {:?}", release_time);
        println!();
        println!("  Pages are now backed by file, not zram.");
        println!("  Re-access will fault from disk (cheap for SSDs).");
    }

    Ok(())
}

/// Demonstrate with a synthetic model (no file required)
fn demo_synthetic_model() -> Result<()> {
    println!("No model path provided. Creating synthetic demo.\n");

    // Create a minimal APR v2 file in memory
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // version major
    data[5] = 0; // version minor
    data[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags = 0 (uncompressed)
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
    data[24..32].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // data_offset

    // Load from bytes (uses heap, not mmap)
    let model = AprV2Model::from_bytes(data)?;

    println!("Synthetic Model Info:");
    println!("  Tensors: {}", model.tensor_count());
    println!(
        "  Using mmap: {} (from_bytes always uses heap)",
        model.is_mmap()
    );
    println!();

    // Demonstrate mmap detection
    println!("Loading Strategy:");
    println!("  - AprV2Model::load(path)     → mmap for uncompressed files");
    println!("  - AprV2Model::from_bytes(v)  → heap (for network/compressed)");
    println!();

    // Show API
    println!("Key APIs for Memory Management:");
    println!();
    println!("  // Load with mmap (zero-copy)");
    println!("  let model = AprV2Model::load(\"model.apr\")?;");
    println!();
    println!("  // Check if using mmap");
    println!("  if model.is_mmap() {{");
    println!("      println!(\"Using memory-mapped I/O\");");
    println!("  }}");
    println!();
    println!("  // Transfer tensors to GPU");
    println!("  for name in model.tensor_names() {{");
    println!("      let bytes = model.get_tensor_bytes(&name)?;");
    println!("      cuda::memcpy_htod(gpu_ptr, bytes);");
    println!("  }}");
    println!();
    println!("  // Release CPU pages (Unix only)");
    println!("  #[cfg(unix)]");
    println!("  model.release_cpu_pages()?;");
    println!();
    println!("  // Pages now backed by file, not zram!");

    // Show zram monitoring commands
    println!();
    println!("Monitor zram usage:");
    println!("  cat /sys/block/zram0/mm_stat  # Raw stats");
    println!("  zramctl                        # Human-readable");

    Ok(())
}
