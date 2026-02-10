//! CUDA and GPU Backend Configuration Example
//!
//! This example demonstrates how to configure aprender for different
//! compute backends: CPU SIMD, GPU (wgpu/WebGPU), and NVIDIA CUDA.
//!
//! # Features Required
//!
//! - `gpu`: Enable GPU acceleration via trueno wgpu backend
//! - `cuda`: Enable NVIDIA CUDA monitoring via trueno-gpu
//!
//! # Run Commands
//!
//! ```bash
//! # Run with default (CPU SIMD) backend
//! cargo run --example cuda_backend
//!
//! # Run with GPU feature
//! cargo run --example cuda_backend --features gpu
//!
//! # Run with CUDA feature (requires NVIDIA driver)
//! cargo run --example cuda_backend --features cuda
//! ```

use aprender::loading::{Backend, LoadConfig, LoadingMode, VerificationLevel};

fn main() {
    println!("=== Aprender Backend Configuration Demo ===\n");

    // 1. Default CPU SIMD backend
    demonstrate_cpu_simd();

    // 2. GPU (wgpu/WebGPU) backend
    demonstrate_gpu();

    // 3. NVIDIA CUDA backend
    demonstrate_cuda();

    // 4. Backend comparison
    compare_backends();

    // 5. Custom configurations
    demonstrate_custom_configs();

    println!("\n=== Demo Complete ===");
}

fn demonstrate_cpu_simd() {
    println!("1. CPU SIMD Backend (Default)");
    println!("   -------------------------");

    let config = LoadConfig::default();
    println!("   Backend: {:?}", config.backend);
    println!("   Supports SIMD: {}", config.backend.supports_simd());
    println!(
        "   GPU Accelerated: {}",
        config.backend.is_gpu_accelerated()
    );
    println!(
        "   Requires NVIDIA: {}",
        config.backend.requires_nvidia_driver()
    );
    println!("   Requires std: {}", config.backend.requires_std());
    println!();

    // Server configuration uses CPU SIMD by default
    let server_config = LoadConfig::server();
    println!("   Server preset:");
    println!("   - Backend: {:?}", server_config.backend);
    println!("   - Mode: {:?}", server_config.mode);
    println!("   - Verification: {:?}", server_config.verification);
    println!();
}

fn demonstrate_gpu() {
    println!("2. GPU Backend (wgpu/WebGPU)");
    println!("   -------------------------");

    let config = LoadConfig::gpu();
    println!("   Backend: {:?}", config.backend);
    println!("   Supports SIMD: {}", config.backend.supports_simd());
    println!(
        "   GPU Accelerated: {}",
        config.backend.is_gpu_accelerated()
    );
    println!(
        "   Requires NVIDIA: {}",
        config.backend.requires_nvidia_driver()
    );
    println!("   Mode: {:?}", config.mode);

    println!();
    println!("   Note: GPU backend uses wgpu for cross-platform GPU compute.");
    println!("   Works on: Vulkan, Metal, DX12, WebGPU (browser)");
    println!();
}

fn demonstrate_cuda() {
    println!("3. NVIDIA CUDA Backend");
    println!("   --------------------");

    let config = LoadConfig::cuda();
    println!("   Backend: {:?}", config.backend);
    println!("   Supports SIMD: {}", config.backend.supports_simd());
    println!(
        "   GPU Accelerated: {}",
        config.backend.is_gpu_accelerated()
    );
    println!(
        "   Requires NVIDIA: {}",
        config.backend.requires_nvidia_driver()
    );
    println!("   Mode: {:?}", config.mode);

    println!();
    println!("   Note: CUDA backend requires:");
    println!("   - NVIDIA GPU hardware");
    println!("   - NVIDIA driver installed");
    println!("   - `cuda` feature enabled in Cargo.toml");
    println!();

    // Check if CUDA is available at runtime
    #[cfg(feature = "cuda")]
    {
        println!("   CUDA feature is ENABLED");
        // In a real scenario, you'd check for NVIDIA driver here
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("   CUDA feature is NOT enabled");
        println!("   Run with: cargo run --example cuda_backend --features cuda");
    }
    println!();
}

fn compare_backends() {
    println!("4. Backend Comparison");
    println!("   ------------------");
    println!();
    println!("   | Backend   | SIMD | GPU Accel | NVIDIA Req | std Req |");
    println!("   |-----------|------|-----------|------------|---------|");

    let backends = [
        Backend::CpuSimd,
        Backend::Gpu,
        Backend::Cuda,
        Backend::Wasm,
        Backend::Embedded,
    ];

    for backend in backends {
        println!(
            "   | {:9} | {:4} | {:9} | {:10} | {:7} |",
            format!("{:?}", backend),
            if backend.supports_simd() { "Yes" } else { "No" },
            if backend.is_gpu_accelerated() {
                "Yes"
            } else {
                "No"
            },
            if backend.requires_nvidia_driver() {
                "Yes"
            } else {
                "No"
            },
            if backend.requires_std() { "Yes" } else { "No" },
        );
    }
    println!();
}

fn demonstrate_custom_configs() {
    println!("5. Custom Backend Configurations");
    println!("   -----------------------------");
    println!();

    // High-performance CUDA config with paranoid verification
    let cuda_paranoid = LoadConfig::new()
        .with_backend(Backend::Cuda)
        .with_mode(LoadingMode::Eager)
        .with_verification(VerificationLevel::Paranoid)
        .with_max_memory(4 * 1024 * 1024 * 1024); // 4GB

    println!("   CUDA with Paranoid Verification:");
    println!("   - Backend: {:?}", cuda_paranoid.backend);
    println!("   - Mode: {:?}", cuda_paranoid.mode);
    println!("   - Verification: {:?}", cuda_paranoid.verification);
    println!(
        "   - Max Memory: {} GB",
        cuda_paranoid
            .max_memory_bytes
            .expect("max memory was configured")
            / (1024 * 1024 * 1024)
    );
    println!();

    // GPU streaming config for large models
    let gpu_streaming = LoadConfig::new()
        .with_backend(Backend::Gpu)
        .with_mode(LoadingMode::Streaming)
        .with_streaming(2 * 1024 * 1024); // 2MB ring buffer

    println!("   GPU with Streaming Mode:");
    println!("   - Backend: {:?}", gpu_streaming.backend);
    println!("   - Mode: {:?}", gpu_streaming.mode);
    println!("   - Streaming: {}", gpu_streaming.streaming);
    println!(
        "   - Ring Buffer: {} MB",
        gpu_streaming.ring_buffer_size / (1024 * 1024)
    );
    println!();

    // Embedded config (no GPU, minimal footprint)
    let embedded = LoadConfig::embedded(64 * 1024); // 64KB
    println!("   Embedded Configuration:");
    println!("   - Backend: {:?}", embedded.backend);
    println!("   - Mode: {:?}", embedded.mode);
    println!("   - Verification: {:?}", embedded.verification);
    println!(
        "   - Max Memory: {} KB",
        embedded
            .max_memory_bytes
            .expect("embedded max memory was configured")
            / 1024
    );
    println!();

    // WASM config (browser deployment)
    let wasm = LoadConfig::wasm();
    println!("   WASM Configuration:");
    println!("   - Backend: {:?}", wasm.backend);
    println!("   - Mode: {:?}", wasm.mode);
    println!("   - Streaming: {}", wasm.streaming);
    println!(
        "   - Max Memory: {} MB",
        wasm.max_memory_bytes
            .expect("WASM max memory was configured")
            / (1024 * 1024)
    );
}
