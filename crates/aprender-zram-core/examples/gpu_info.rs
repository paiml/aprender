//! GPU detection and information example.
//!
//! Run with: `cargo run --example gpu_info --features cuda`
//!
//! This example demonstrates:
//! - CUDA device detection
//! - GPU capability querying
//! - Backend selection logic
//! - PCIe 5x rule evaluation

use trueno_zram_core::gpu::{
    gpu_available, meets_pcie_rule, select_backend, GpuBackend, GpuDeviceInfo, GPU_MIN_BATCH_SIZE,
};
use trueno_zram_core::PAGE_SIZE;

#[cfg(feature = "cuda")]
use trueno_zram_core::gpu::GpuCompressor;
#[cfg(feature = "cuda")]
use trueno_zram_core::Algorithm;

fn main() {
    println!("trueno-zram GPU Information");
    println!("============================\n");

    // 1. Check GPU availability
    println!("1. GPU Availability");
    println!("   ----------------");
    let available = gpu_available();
    println!("   GPU available: {available}");

    #[cfg(feature = "cuda")]
    if available {
        print_cuda_info();
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("   (CUDA feature not enabled - run with --features cuda)");
    }

    println!();

    // 2. Backend selection logic
    println!("2. Backend Selection Logic");
    println!("   -----------------------");
    println!("   GPU_MIN_BATCH_SIZE: {GPU_MIN_BATCH_SIZE} pages");
    println!("   PAGE_SIZE: {PAGE_SIZE} bytes\n");

    let test_sizes = [1, 10, 100, 500, 1000, 5000, 10000];
    println!("   {:>8} {:>15} {:>15}", "Batch", "No GPU", "With GPU");
    println!("   {}", "-".repeat(42));

    for size in test_sizes {
        let no_gpu = select_backend(size, false);
        let with_gpu = select_backend(size, true);
        println!(
            "   {:>8} {:>15} {:>15}",
            size,
            format!("{no_gpu:?}"),
            format!("{with_gpu:?}")
        );
    }

    println!();

    // 3. PCIe 5x Rule
    println!("3. PCIe 5x Rule Evaluation");
    println!("   -----------------------");
    println!("   GPU offload beneficial when: T_cpu > 5 * (T_transfer + T_gpu)\n");

    let scenarios = [
        (1000, 25.0, 100.0, "1K pages, PCIe 4.0, 100 GB/s GPU"),
        (10000, 25.0, 100.0, "10K pages, PCIe 4.0, 100 GB/s GPU"),
        (100000, 64.0, 500.0, "100K pages, PCIe 5.0, 500 GB/s GPU"),
        (1000000, 64.0, 500.0, "1M pages, PCIe 5.0, 500 GB/s GPU"),
    ];

    for (pages, pcie, gpu_tput, desc) in scenarios {
        let beneficial = meets_pcie_rule(pages, pcie, gpu_tput);
        let data_mb = (pages * PAGE_SIZE) / (1024 * 1024);
        println!(
            "   {} ({} MB): {}",
            desc,
            data_mb,
            if beneficial {
                "GPU beneficial"
            } else {
                "CPU preferred"
            }
        );
    }

    println!();

    // 4. Simulated device info
    println!("4. Example Device Configurations");
    println!("   ------------------------------");

    let devices = [
        GpuDeviceInfo {
            index: 0,
            name: "RTX 4090 (Ada Lovelace)".to_string(),
            total_memory: 24 * 1024 * 1024 * 1024,
            l2_cache_size: 72 * 1024 * 1024,
            compute_capability: (8, 9),
            backend: GpuBackend::Cuda,
        },
        GpuDeviceInfo {
            index: 0,
            name: "A100 (Ampere)".to_string(),
            total_memory: 80 * 1024 * 1024 * 1024,
            l2_cache_size: 40 * 1024 * 1024,
            compute_capability: (8, 0),
            backend: GpuBackend::Cuda,
        },
        GpuDeviceInfo {
            index: 0,
            name: "H100 (Hopper)".to_string(),
            total_memory: 80 * 1024 * 1024 * 1024,
            l2_cache_size: 50 * 1024 * 1024,
            compute_capability: (9, 0),
            backend: GpuBackend::Cuda,
        },
    ];

    for dev in &devices {
        println!("\n   {}", dev.name);
        println!(
            "     SM: {}.{}",
            dev.compute_capability.0, dev.compute_capability.1
        );
        println!(
            "     Memory: {} GB",
            dev.total_memory / (1024 * 1024 * 1024)
        );
        println!("     L2 Cache: {} MB", dev.l2_cache_size / (1024 * 1024));
        println!("     Optimal batch: {} pages", dev.optimal_batch_size());
        println!("     Supported: {}", dev.is_supported());
    }

    println!();
}

#[cfg(feature = "cuda")]
fn print_cuda_info() {
    println!("\n   CUDA Device Information:");

    match GpuCompressor::new(0, Algorithm::Lz4) {
        Ok(compressor) => {
            let dev = compressor.device();
            println!("     Device: {}", dev.name);
            println!(
                "     Compute Capability: SM {}.{}",
                dev.compute_capability.0, dev.compute_capability.1
            );
            println!("     Memory: {} MB", dev.total_memory / (1024 * 1024));
            println!("     L2 Cache: {} KB", dev.l2_cache_size / 1024);
            println!("     Optimal batch: {} pages", compressor.batch_size());
            println!("     Supported: {}", dev.is_supported());
        }
        Err(e) => {
            println!("     Error: {e}");
        }
    }
}
