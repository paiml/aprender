// ============================================================================
// GPU CANARY TESTS (CUDA Integration)
// ============================================================================

/// GPU Integration "Hello World": Basic vector copy to/from GPU
///
/// This test verifies CUDA is working by:
/// 1. Uploading a vector to GPU memory
/// 2. Downloading it back to host
/// 3. Verifying the data is unchanged
///
/// **If this test fails:**
/// - CUDA drivers may not be installed
/// - GPU may not be available
/// - `--all-features` may not be set
#[test]
#[cfg(feature = "cuda")]
fn canary_gpu_vector_roundtrip() {
    use trueno_gpu::driver::{CudaContext, CudaStream, GpuBuffer};

    // Try to create CUDA context
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            // Gracefully skip if CUDA is not available (CI environments, etc.)
            eprintln!(
                "⚠️  GPU CANARY SKIPPED: CUDA context creation failed: {:?}\n\
                 Note: If you expect CUDA to work, check: nvidia-smi",
                e
            );
            return;
        }
    };

    // Create stream for async operations
    let stream = CudaStream::new(&ctx).expect("CUDA stream creation failed");

    // Test data: recognizable pattern to verify integrity
    let test_data: Vec<f32> =
        vec![1.0, 2.0, 3.0, 4.0, 0.5, -0.5, std::f32::consts::PI, std::f32::consts::E];
    let n = test_data.len();

    // Allocate GPU buffer and upload
    let mut gpu_buffer: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, n).expect("GPU buffer allocation failed");
    gpu_buffer.copy_from_host(&test_data).expect("Host→GPU copy failed");

    // Download back to host
    let mut result = vec![0.0f32; n];
    gpu_buffer.copy_to_host(&mut result).expect("GPU→Host copy failed");

    // Synchronize to ensure all operations complete
    stream.synchronize().expect("Stream sync failed");

    // Verify data integrity
    for (i, (&expected, &actual)) in test_data.iter().zip(result.iter()).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-6,
            "GPU CANARY FAILED: Data mismatch at index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    println!("✅ GPU CANARY PASSED: Vector roundtrip successful ({} elements)", n);
}

/// GPU Canary: Verify CUDA device properties are accessible
#[test]
#[cfg(feature = "cuda")]
fn canary_gpu_device_info() {
    use trueno_gpu::driver::CudaContext;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("⚠️  GPU CANARY SKIPPED: No CUDA device available");
            return;
        }
    };

    // If we got here, CUDA is working
    println!("✅ GPU CANARY PASSED: CUDA device {} accessible", ctx.device());
}

/// GPU Canary: Basic kernel execution test
#[test]
#[cfg(feature = "cuda")]
fn canary_gpu_kernel_execution() {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::ptx::{PtxArithmetic, PtxControl, PtxKernel, PtxModule, PtxReg, PtxType};

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("⚠️  GPU CANARY SKIPPED: No CUDA device available");
            return;
        }
    };

    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    // Simple kernel: write thread ID to output
    let kernel = PtxKernel::new("canary_kernel").param(PtxType::U64, "output").build(|ctx| {
        let out_ptr = ctx.load_param_u64("output");
        let tid = ctx.special_reg(PtxReg::TidX);

        // Compute output address: output + tid * 4
        let offset = ctx.mul_u32(tid, 4); // mul_u32 takes (VirtualReg, u32)
        let offset64 = ctx.cvt_u64_u32(offset);
        let addr = ctx.add_u64(out_ptr, offset64);

        // Write tid to output[tid]
        ctx.st_global_u32(addr, tid);
        ctx.ret();
    });

    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_70") // Safe minimum for modern GPUs
        .address_size(64)
        .add_kernel(kernel)
        .emit();

    // Compile and load module
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX compilation failed");

    // Allocate output buffer
    let mut output: GpuBuffer<u32> = GpuBuffer::new(&ctx, 32).expect("Buffer allocation");
    output.copy_from_host(&[0u32; 32]).expect("Buffer init");

    // Launch kernel with 32 threads
    let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };

    let mut args: [*mut c_void; 1] = [output.as_kernel_arg()];

    unsafe {
        stream
            .launch_kernel(&mut module, "canary_kernel", &config, &mut args)
            .expect("Kernel launch failed");
    }

    stream.synchronize().expect("Sync failed");

    // Verify output
    let mut result = vec![0u32; 32];
    output.copy_to_host(&mut result).expect("Copy to host");

    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val, i as u32,
            "GPU CANARY FAILED: Kernel output mismatch at {}: expected {}, got {}",
            i, i, val
        );
    }

    println!("✅ GPU CANARY PASSED: PTX kernel execution successful");
}

/// Titan Duel: Numerical Parity Test - CPU vs GPU GEMM
///
/// This is the "Duel of the Titans" - verifying that GPU GEMM produces
/// mathematically identical results to the CPU reference implementation.
///
/// If this test fails, the GPU kernel is a "successful hallucination" -
/// it runs without crashing but produces incorrect results.
#[test]
#[cfg(feature = "cuda")]
fn titan_duel_numerical_parity() {
    use std::ffi::c_void;
    use trueno::blis::gemm_reference;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::{GemmKernel, Kernel};

    const N: usize = 128; // 128x128 matrix
    const EPSILON: f32 = 1e-4; // Tolerance for FP32 accumulation differences

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("⚠️  TITAN DUEL SKIPPED: No CUDA device available");
            return;
        }
    };

    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    // Generate deterministic test data (not random for reproducibility)
    let mut a = vec![0.0f32; N * N];
    let mut b = vec![0.0f32; N * N];
    for i in 0..N {
        for j in 0..N {
            // Simple pattern: a[i,j] = (i + j) % 10 / 10.0
            a[i * N + j] = ((i + j) % 10) as f32 / 10.0;
            b[i * N + j] = ((i * 2 + j) % 10) as f32 / 10.0;
        }
    }

    // ============================================================
    // CPU Reference: gemm_reference (gold standard)
    // ============================================================
    let mut c_cpu = vec![0.0f32; N * N];
    gemm_reference(N, N, N, &a, &b, &mut c_cpu).expect("CPU GEMM failed");

    // ============================================================
    // GPU: Generate PTX and execute
    // ============================================================
    let kernel = GemmKernel::naive(N as u32, N as u32, N as u32);
    let ptx = kernel.emit_ptx();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX compilation failed");

    // Upload matrices to GPU
    let mut gpu_a: GpuBuffer<f32> = GpuBuffer::new(&ctx, N * N).expect("Buffer A");
    let mut gpu_b: GpuBuffer<f32> = GpuBuffer::new(&ctx, N * N).expect("Buffer B");
    let mut gpu_c: GpuBuffer<f32> = GpuBuffer::new(&ctx, N * N).expect("Buffer C");

    gpu_a.copy_from_host(&a).expect("Copy A");
    gpu_b.copy_from_host(&b).expect("Copy B");
    let zeros = [0.0f32; N * N];
    gpu_c.copy_from_host(&zeros).expect("Copy C");

    // Launch kernel
    let block_size = 16;
    let grid_size = N.div_ceil(block_size);
    let config = LaunchConfig {
        grid: (grid_size as u32, grid_size as u32, 1),
        block: (block_size as u32, block_size as u32, 1),
        shared_mem: 0,
    };

    let m = N as u32;
    let n = N as u32;
    let k = N as u32;

    let mut args: [*mut c_void; 6] = [
        gpu_a.as_kernel_arg(),
        gpu_b.as_kernel_arg(),
        gpu_c.as_kernel_arg(),
        &m as *const u32 as *mut c_void,
        &n as *const u32 as *mut c_void,
        &k as *const u32 as *mut c_void,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, "gemm_naive", &config, &mut args)
            .expect("Kernel launch failed");
    }

    stream.synchronize().expect("Sync failed");

    // Download result
    let mut c_gpu = vec![0.0f32; N * N];
    gpu_c.copy_to_host(&mut c_gpu).expect("Copy result");

    // ============================================================
    // PARITY CHECK: Compare CPU vs GPU
    // ============================================================
    let mut max_diff: f32 = 0.0;
    let mut diff_count = 0;

    for i in 0..N * N {
        let diff = (c_cpu[i] - c_gpu[i]).abs();
        if diff > EPSILON {
            diff_count += 1;
            if diff_count <= 5 {
                println!(
                    "  Mismatch at [{}]: CPU={:.6}, GPU={:.6}, diff={:.6}",
                    i, c_cpu[i], c_gpu[i], diff
                );
            }
        }
        max_diff = max_diff.max(diff);
    }

    if diff_count > 0 {
        panic!(
            "\n\
            ╔══════════════════════════════════════════════════════════════════════════════╗\n\
            ║  TITAN DUEL FAILED: GPU NUMERICAL PARITY VIOLATION                           ║\n\
            ╠══════════════════════════════════════════════════════════════════════════════╣\n\
            ║  The GPU GEMM kernel produces mathematically incorrect results.              ║\n\
            ║  This is a 'successful hallucination' - it runs but lies.                    ║\n\
            ║                                                                              ║\n\
            ║  Matrix size: {}x{}                                                         ║\n\
            ║  Max difference: {:.6}                                                      ║\n\
            ║  Elements failing (>{:.0e}): {} / {}                                       ║\n\
            ║                                                                              ║\n\
            ║  FIX: Review GPU kernel PTX for accumulation order or precision issues.      ║\n\
            ╚══════════════════════════════════════════════════════════════════════════════╝\n",
            N,
            N,
            max_diff,
            EPSILON,
            diff_count,
            N * N
        );
    }

    println!(
        "✅ TITAN DUEL PASSED: CPU-GPU parity verified (max diff: {:.2e}, {}x{} matrix)",
        max_diff, N, N
    );
}

// ============================================================================
// NON-CUDA STUBS (when CUDA feature is disabled)
// ============================================================================

#[cfg(not(feature = "cuda"))]
mod cuda_stubs {
    #[test]
    fn canary_gpu_vector_roundtrip() {
        println!("⚠️  GPU CANARY SKIPPED: CUDA feature not enabled");
        println!("   Run with: cargo test --test hardware_canary --all-features");
    }

    #[test]
    fn canary_gpu_device_info() {
        println!("⚠️  GPU CANARY SKIPPED: CUDA feature not enabled");
    }

    #[test]
    fn canary_gpu_kernel_execution() {
        println!("⚠️  GPU CANARY SKIPPED: CUDA feature not enabled");
    }

    #[test]
    fn titan_duel_numerical_parity() {
        println!("⚠️  TITAN DUEL SKIPPED: CUDA feature not enabled");
        println!("   Run with: cargo test --test hardware_canary --all-features");
    }
}
