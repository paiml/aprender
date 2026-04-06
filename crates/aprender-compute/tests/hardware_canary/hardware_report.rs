// ============================================================================
// HARDWARE REPORT
// ============================================================================

/// Generate a hardware capability report for debugging
#[test]
fn hardware_capability_report() {
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  HARDWARE CAPABILITY REPORT");
    println!("═══════════════════════════════════════════════════════════");

    // CPU Architecture
    println!("\n📦 CPU Architecture:");
    #[cfg(target_arch = "x86_64")]
    {
        println!("   Arch: x86_64");
        use std::arch::is_x86_feature_detected;
        println!("   SSE2:    {}", is_x86_feature_detected!("sse2"));
        println!("   AVX:     {}", is_x86_feature_detected!("avx"));
        println!("   AVX2:    {}", is_x86_feature_detected!("avx2"));
        println!("   FMA:     {}", is_x86_feature_detected!("fma"));
        println!("   AVX-512F: {}", is_x86_feature_detected!("avx512f"));
    }
    #[cfg(target_arch = "aarch64")]
    println!("   Arch: aarch64 (ARM64 NEON)");
    #[cfg(target_arch = "wasm32")]
    println!("   Arch: wasm32");

    // Backend selection
    let backend = trueno::Backend::select_best();
    println!("\n🔧 Selected Backend: {:?}", backend);

    // GPU status
    println!("\n🖥️  GPU Status:");
    #[cfg(feature = "cuda")]
    {
        match trueno_gpu::driver::CudaContext::new(0) {
            Ok(ctx) => println!("   CUDA: Available (device {})", ctx.device()),
            Err(e) => println!("   CUDA: Not available ({:?})", e),
        }
    }
    #[cfg(not(feature = "cuda"))]
    println!("   CUDA: Feature not enabled");

    println!("\n═══════════════════════════════════════════════════════════\n");
}
