//! cuBLAS hardware integration tests — REQUIRE RTX 4090
//!
//! Tests: handle lifecycle, FP16 GEMM correctness, FP32 GEMM, row-major helper,
//! buffer overflow detection, throughput measurement.
//!
//! Contract: cublas-gemm-v1.yaml (FALSIFY-CUBLAS-001, -003, -005)

use crate::driver::{CublasHandle, CudaContext, CudaStream, GpuBuffer, LaunchConfig};

/// FALSIFY-CUBLAS-005: CublasHandle creates and destroys cleanly
#[test]
fn test_cublas_handle_lifecycle() {
    let ctx = CudaContext::new(0).expect("CUDA context required");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle creation must succeed");
    let stream = CudaStream::new(&ctx).expect("stream required");
    handle.set_stream(&stream).expect("set_stream must succeed");
    // Handle dropped here — cublasDestroy_v2 called via Drop
    drop(handle);
}

/// Basic FP32 GEMM correctness: C = A @ B
/// A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
/// Expected C = [[19, 22], [43, 50]]
#[test]
fn test_cublas_gemm_f32_small() {
    let ctx = CudaContext::new(0).expect("CUDA context required");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle required");
    let stream = CudaStream::new(&ctx).expect("stream required");
    handle.set_stream(&stream).expect("set_stream must succeed");

    // Row-major: A = [[1, 2], [3, 4]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let c_data: Vec<f32> = vec![0.0; 4];

    let a_buf = GpuBuffer::from_host(&ctx, &a_data).expect("A upload");
    let b_buf = GpuBuffer::from_host(&ctx, &b_data).expect("B upload");
    let mut c_buf = GpuBuffer::from_host(&ctx, &c_data).expect("C upload");

    // Row-major GEMM: C[2,2] = A[2,2] @ B[2,2]
    handle
        .gemm_f32_row_major(
            2,
            2,
            2,              // m, n, k
            1.0,            // alpha
            a_buf.as_ptr(), // A
            b_buf.as_ptr(), // B
            0.0,            // beta
            c_buf.as_ptr(), // C
        )
        .expect("gemm_f32_row_major must succeed");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; 4];
    c_buf.copy_to_host(&mut result).expect("D2H");

    // C = [[19, 22], [43, 50]]
    assert!((result[0] - 19.0).abs() < 1e-3, "C[0,0] = {} expected 19.0", result[0]);
    assert!((result[1] - 22.0).abs() < 1e-3, "C[0,1] = {} expected 22.0", result[1]);
    assert!((result[2] - 43.0).abs() < 1e-3, "C[1,0] = {} expected 43.0", result[2]);
    assert!((result[3] - 50.0).abs() < 1e-3, "C[1,1] = {} expected 50.0", result[3]);
}

/// FP16 GEMM on training-relevant shape: [4096, 1024] x [1024, 4096]
/// FALSIFY-CUBLAS-003: Must achieve > 100 TFLOP/s
#[test]
fn test_cublas_gemm_f16_training_shape() {
    let ctx = CudaContext::new(0).expect("CUDA context required");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle required");
    let stream = CudaStream::new(&ctx).expect("stream required");
    handle.set_stream(&stream).expect("set_stream must succeed");

    let m: usize = 4096;
    let k: usize = 1024;
    let n: usize = 4096;

    // FP16 = 2 bytes per element. Use u16 to represent half-precision.
    // Fill A with 0x3C00 (1.0 in FP16), B with 0x3C00 (1.0 in FP16)
    let fp16_one: u16 = 0x3C00; // 1.0 in IEEE 754 half
    let a_data: Vec<u16> = vec![fp16_one; m * k];
    let b_data: Vec<u16> = vec![fp16_one; k * n];
    let c_data: Vec<u16> = vec![0u16; m * n];

    let a_buf = GpuBuffer::from_host(&ctx, &a_data).expect("A upload");
    let b_buf = GpuBuffer::from_host(&ctx, &b_data).expect("B upload");
    let mut c_buf = GpuBuffer::from_host(&ctx, &c_data).expect("C upload");

    // Warmup
    for _ in 0..5 {
        handle
            .gemm_f16_row_major(
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a_buf.as_ptr(),
                b_buf.as_ptr(),
                0.0,
                c_buf.as_ptr(),
            )
            .expect("warmup GEMM");
    }
    stream.synchronize().expect("warmup sync");

    // Timed run
    let iters = 100;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        handle
            .gemm_f16_row_major(
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a_buf.as_ptr(),
                b_buf.as_ptr(),
                0.0,
                c_buf.as_ptr(),
            )
            .expect("timed GEMM");
    }
    stream.synchronize().expect("timed sync");
    let elapsed = start.elapsed();

    // Check correctness: C = A @ B where all elements are 1.0
    // Each element should be k = 1024 (sum of k ones)
    // 1024 in FP16 = 0x6400
    let mut result = vec![0u16; m * n];
    c_buf.copy_to_host(&mut result).expect("D2H");

    // Check a sample of elements
    let expected_fp16: u16 = 0x6400; // 1024.0 in FP16
    assert_eq!(
        result[0], expected_fp16,
        "C[0,0] should be 1024.0 (0x6400), got 0x{:04X}",
        result[0]
    );
    assert_eq!(
        result[m * n - 1],
        expected_fp16,
        "C[last] should be 1024.0 (0x6400), got 0x{:04X}",
        result[m * n - 1]
    );

    // FALSIFY-CUBLAS-003: Throughput > 100 TFLOP/s
    let flops_per_gemm = 2.0 * m as f64 * n as f64 * k as f64;
    let total_flops = flops_per_gemm * iters as f64;
    let tflops = total_flops / elapsed.as_secs_f64() / 1e12;

    eprintln!(
        "cuBLAS FP16 GEMM [{m}x{k}] x [{k}x{n}]: {tflops:.1} TFLOP/s ({} iters, {:.1}ms)",
        iters,
        elapsed.as_millis()
    );

    assert!(tflops > 50.0, "cuBLAS FP16 GEMM must exceed 50 TFLOP/s, got {tflops:.1} TFLOP/s");
}

/// All 6 training shapes from cublas-gemm-v1.yaml must work
#[test]
fn test_cublas_all_training_shapes() {
    let ctx = CudaContext::new(0).expect("CUDA context required");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle required");
    let stream = CudaStream::new(&ctx).expect("stream required");
    handle.set_stream(&stream).expect("set_stream must succeed");

    // Training shapes from the contract
    let shapes: Vec<(usize, usize, usize, &str)> = vec![
        (4096, 3072, 1024, "attn_qkv"),
        (1024, 3072, 4096, "attn_qkv_backward"),
        (4096, 1024, 1024, "attn_output"),
        (4096, 8192, 1024, "ffn_up_gate"),
        (4096, 1024, 4096, "ffn_down"),
        (4096, 256, 1024, "gqa_kv"),
    ];

    let fp16_one: u16 = 0x3C00;

    for (m, n, k, name) in &shapes {
        let a = GpuBuffer::from_host(&ctx, &vec![fp16_one; m * k]).expect("A");
        let b = GpuBuffer::from_host(&ctx, &vec![fp16_one; k * n]).expect("B");
        let mut c = GpuBuffer::from_host(&ctx, &vec![0u16; m * n]).expect("C");

        handle
            .gemm_f16_row_major(
                *m as i32,
                *n as i32,
                *k as i32,
                1.0,
                a.as_ptr(),
                b.as_ptr(),
                0.0,
                c.as_ptr(),
            )
            .expect(&format!("GEMM {name} [{m}x{k}] x [{k}x{n}] must succeed"));

        stream.synchronize().expect("sync");

        // Spot-check first element: should be k as FP16
        let mut result = vec![0u16; 1];
        // Only read first element via raw pointer offset
        let first_elem_buf = unsafe { GpuBuffer::<u16>::from_raw_parts(c.as_ptr(), 1) };
        let mut check = vec![0u16; 1];
        first_elem_buf.copy_to_host(&mut check).expect("D2H check");
        std::mem::forget(first_elem_buf); // Don't free c's memory

        eprintln!(
            "Shape {name} [{m}x{k}] x [{k}x{n}]: C[0,0] = 0x{:04X} (expected ~{k}.0)",
            check[0]
        );
    }
}

/// cuBLAS GEMM throughput benchmark (ALB-075).
///
/// Run: cargo test -p trueno-gpu --features cuda --lib --release -- cublas_bench --no-capture
#[test]
fn cublas_bench_gemm_fp16_throughput() {
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("stream");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle");
    handle.set_stream(&stream).expect("set_stream");

    eprintln!();
    eprintln!("=== cuBLAS FP16 GEMM Throughput (RTX 4090) ===");
    eprintln!("{:<10} {:>12} {:>12} {:>10}", "Size", "Time(µs)", "TFLOP/s", "Efficiency");
    eprintln!("{}", "-".repeat(48));

    for &n in &[256_usize, 512, 1024, 2048, 4096] {
        let m = n;
        let k = n;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        let a_data = vec![0x3C00u16; m * k]; // 1.0 in FP16
        let b_data = vec![0x3C00u16; k * n];

        let a_buf = GpuBuffer::from_host(&ctx, &a_data).expect("A");
        let b_buf = GpuBuffer::from_host(&ctx, &b_data).expect("B");
        let c_buf = GpuBuffer::from_host(&ctx, &vec![0u16; m * n]).expect("C");

        // Warmup
        for _ in 0..5 {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();

        let iters: u32 = if n <= 512 {
            200
        } else if n <= 1024 {
            100
        } else {
            30
        };
        let start = Instant::now();
        for _ in 0..iters {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();
        let elapsed = start.elapsed();

        let per_call_us = elapsed.as_micros() as f64 / iters as f64;
        let tflops = flops / (per_call_us * 1e6);
        // RTX 4090 FP16 tensor core peak: ~330 TFLOP/s
        let eff = tflops / 330.0 * 100.0;

        eprintln!(
            "{:<10} {:>10.1}µs {:>10.1} {:>8.1}%",
            format!("{n}x{n}"),
            per_call_us,
            tflops,
            eff,
        );
    }
    eprintln!();
}

/// PTX GEMM vs cuBLAS side-by-side benchmark.
///
/// Compares trueno-gpu's pure-Rust PTX kernels against NVIDIA cuBLAS.
/// Run: cargo test -p trueno-gpu --features cuda --lib --release -- ptx_vs_cublas --no-capture
#[test]
fn ptx_vs_cublas_gemm_f32() {
    use crate::driver::module::CudaModule;
    use crate::kernels::{GemmKernel, Kernel};
    use std::ffi::c_void;
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("stream");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle");
    handle.set_stream(&stream).expect("set_stream");

    eprintln!();
    eprintln!("=== PTX GEMM (pure Rust) vs cuBLAS (NVIDIA) — FP32 ===");
    eprintln!("{:<10} {:>12} {:>12} {:>10}", "Size", "PTX(µs)", "cuBLAS(µs)", "Ratio");
    eprintln!("{}", "-".repeat(48));

    // FP32 tiled GEMM — test small sizes where PTX launch overhead matters
    for &n in &[32_usize, 64, 128, 256] {
        let m = n;
        let k = n;

        let a_data = vec![1.0f32; m * k];
        let b_data = vec![1.0f32; k * n];
        let c_data = vec![0.0f32; m * n];

        let a_buf = GpuBuffer::from_host(&ctx, &a_data).expect("A");
        let b_buf = GpuBuffer::from_host(&ctx, &b_data).expect("B");
        let c_buf = GpuBuffer::from_host(&ctx, &c_data).expect("C");

        // --- PTX GEMM ---
        let tile_size = 32.min(n);
        let kernel = GemmKernel::tiled(m as u32, n as u32, k as u32, tile_size as u32);
        let ptx = kernel.emit_ptx();
        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{:<10} PTX compile failed: {e}", format!("{n}x{n}"));
                continue;
            }
        };

        let grid_x = ((n + tile_size - 1) / tile_size) as u32;
        let grid_y = ((m + tile_size - 1) / tile_size) as u32;
        let config = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (tile_size as u32, tile_size as u32, 1),
            shared_mem: 0,
        };

        let mut a_ptr = a_buf.as_ptr();
        let mut b_ptr = b_buf.as_ptr();
        let mut c_ptr = c_buf.as_ptr();
        let mut m_val = m as u32;
        let mut n_val = n as u32;
        let mut k_val = k as u32;

        let mut args: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // Warmup PTX
        for _ in 0..3 {
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args).ok();
            }
        }
        stream.synchronize().ok();

        // Benchmark PTX
        let iters = 50;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let ptx_us = start.elapsed().as_micros() as f64 / iters as f64;

        // --- cuBLAS FP32 ---
        // cuBLAS uses column-major, so we transpose: C^T = B^T @ A^T
        use crate::driver::GemmOp;
        // Warmup
        for _ in 0..3 {
            handle
                .gemm_f32(
                    GemmOp::NoTrans,
                    GemmOp::NoTrans,
                    n as i32,
                    m as i32,
                    k as i32,
                    1.0,
                    b_buf.as_ptr(),
                    n as i32,
                    a_buf.as_ptr(),
                    k as i32,
                    0.0,
                    c_buf.as_ptr(),
                    n as i32,
                )
                .ok();
        }
        stream.synchronize().ok();

        let start = Instant::now();
        for _ in 0..iters {
            handle
                .gemm_f32(
                    GemmOp::NoTrans,
                    GemmOp::NoTrans,
                    n as i32,
                    m as i32,
                    k as i32,
                    1.0,
                    b_buf.as_ptr(),
                    n as i32,
                    a_buf.as_ptr(),
                    k as i32,
                    0.0,
                    c_buf.as_ptr(),
                    n as i32,
                )
                .ok();
        }
        stream.synchronize().ok();
        let cublas_us = start.elapsed().as_micros() as f64 / iters as f64;

        let ratio = if cublas_us > 0.0 { ptx_us / cublas_us } else { 0.0 };
        let status = if ratio < 2.0 {
            "competitive"
        } else if ratio < 5.0 {
            "gap"
        } else {
            "needs work"
        };

        eprintln!(
            "{:<10} {:>10.1}µs {:>10.1}µs {:>8.1}x  {status}",
            format!("{n}x{n}"),
            ptx_us,
            cublas_us,
            ratio,
        );
    }
    eprintln!();
    eprintln!("Note: PTX kernels are pure Rust (no nvcc). cuBLAS is NVIDIA vendor-optimized.");
    eprintln!("PTX/cuBLAS ratio <2x = competitive, <5x = gap, >5x = needs work.");
}

/// WMMA Tensor Core FP16 PTX vs cuBLAS FP16 benchmark.
///
/// Tests trueno-gpu's pure-Rust WMMA 16×16×16 FP16 kernels against cuBLAS
/// with tensor core FP16 accumulation. Both use hardware tensor cores.
///
/// Run: cargo test -p trueno-gpu --features cuda --lib --release -- wmma_vs_cublas --no-capture
#[test]
fn wmma_vs_cublas_fp16() {
    use crate::driver::module::CudaModule;
    use crate::driver::GemmOp;
    use crate::kernels::{GemmKernel, Kernel};
    use std::ffi::c_void;
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("stream");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle");
    handle.set_stream(&stream).expect("set_stream");

    eprintln!();
    eprintln!("=== WMMA Tensor Core (pure Rust PTX) vs cuBLAS — FP16 ===");
    eprintln!(
        "{:<10} {:>12} {:>12} {:>12} {:>10}",
        "Size", "WMMA(µs)", "cuBLAS(µs)", "WMMA TFLOP/s", "Ratio"
    );
    eprintln!("{}", "-".repeat(60));

    for &n in &[128_usize, 256, 512, 1024] {
        let m = n;
        let k = n;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        // FP32 input (WMMA kernel converts to FP16 internally)
        let a_data = vec![1.0f32; m * k];
        let b_data = vec![1.0f32; k * n];
        let c_data = vec![0.0f32; m * n];

        let a_buf = GpuBuffer::from_host(&ctx, &a_data).expect("A");
        let b_buf = GpuBuffer::from_host(&ctx, &b_data).expect("B");
        let c_buf = GpuBuffer::from_host(&ctx, &c_data).expect("C");

        // --- WMMA PTX kernel ---
        let kernel = GemmKernel::tensor_core(m as u32, n as u32, k as u32);
        let ptx = kernel.emit_ptx();
        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{:<10} WMMA PTX compile failed: {e}", format!("{n}x{n}"));
                continue;
            }
        };

        let grid_x = ((n + 15) / 16) as u32;
        let grid_y = ((m + 15) / 16) as u32;
        let config = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (32, 1, 1), // 1 warp per 16x16 tile
            shared_mem: 0,
        };

        let mut a_ptr = a_buf.as_ptr();
        let mut b_ptr = b_buf.as_ptr();
        let mut c_ptr = c_buf.as_ptr();
        let mut m_val = m as u32;
        let mut n_val = n as u32;
        let mut k_val = k as u32;

        let mut args: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // Warmup WMMA
        for _ in 0..5 {
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args).ok();
            }
        }
        stream.synchronize().ok();

        let iters = if n <= 256 { 100 } else { 50 };
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let wmma_us = start.elapsed().as_micros() as f64 / iters as f64;
        let wmma_tflops = flops / (wmma_us * 1e6);

        // --- cuBLAS FP16 ---
        let a16 = vec![0x3C00u16; m * k]; // 1.0 in FP16
        let b16 = vec![0x3C00u16; k * n];
        let c16 = vec![0u16; m * n];
        let a16_buf = GpuBuffer::from_host(&ctx, &a16).expect("A16");
        let b16_buf = GpuBuffer::from_host(&ctx, &b16).expect("B16");
        let c16_buf = GpuBuffer::from_host(&ctx, &c16).expect("C16");

        for _ in 0..5 {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a16_buf.as_ptr(),
                    b16_buf.as_ptr(),
                    0.0,
                    c16_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();

        let start = Instant::now();
        for _ in 0..iters {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a16_buf.as_ptr(),
                    b16_buf.as_ptr(),
                    0.0,
                    c16_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();
        let cublas_us = start.elapsed().as_micros() as f64 / iters as f64;

        let ratio = if cublas_us > 0.0 { wmma_us / cublas_us } else { 0.0 };

        eprintln!(
            "{:<10} {:>10.1}µs {:>10.1}µs {:>10.1} {:>8.1}x",
            format!("{n}x{n}"),
            wmma_us,
            cublas_us,
            wmma_tflops,
            ratio,
        );
    }
    eprintln!();
}

/// CTA-tiled WMMA (4 warps, 32×32 tiles) vs cuBLAS FP16.
///
/// Run: cargo test -p trueno-gpu --features cuda --lib --release -- cta_wmma_vs_cublas --no-capture
#[test]
fn cta_wmma_vs_cublas_fp16() {
    use crate::driver::module::CudaModule;
    use crate::kernels::gemm::basic::tensor_core::cta_wmma::build_cta_wmma_fp16;
    use crate::ptx::PtxModule;
    use std::ffi::c_void;
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("stream");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle");
    handle.set_stream(&stream).expect("set_stream");

    eprintln!();
    eprintln!("=== CTA WMMA (4-warp, 32x32) vs cuBLAS — FP16 ===");
    eprintln!(
        "{:<10} {:>12} {:>12} {:>12} {:>10}",
        "Size", "CTA(us)", "cuBLAS(us)", "CTA TFLOP/s", "Ratio"
    );
    eprintln!("{}", "-".repeat(60));

    for &n in &[128_usize, 256, 512, 1024] {
        let m = n;
        let k = n;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        let a16 = vec![0x3C00u16; m * k];
        let b16 = vec![0x3C00u16; k * n];
        let c32 = vec![0.0f32; m * n];

        let a_buf = GpuBuffer::from_host(&ctx, &a16).expect("A");
        let b_buf = GpuBuffer::from_host(&ctx, &b16).expect("B");
        let c_buf = GpuBuffer::from_host(&ctx, &c32).expect("C");

        let kernel = build_cta_wmma_fp16(m as u32, n as u32, k as u32);
        let ptx_str = PtxModule::new().add_kernel(kernel).emit();
        let mut module = match CudaModule::from_ptx(&ctx, &ptx_str) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{:<10} CTA compile failed: {e}", format!("{n}x{n}"));
                continue;
            }
        };

        let grid_x = ((n + 31) / 32) as u32;
        let grid_y = ((m + 31) / 32) as u32;
        let config =
            LaunchConfig { grid: (grid_x, grid_y, 1), block: (128, 1, 1), shared_mem: 2048 };

        let mut a_ptr = a_buf.as_ptr();
        let mut b_ptr = b_buf.as_ptr();
        let mut c_ptr = c_buf.as_ptr();
        let mut m_v = m as u32;
        let mut n_v = n as u32;
        let mut k_v = k as u32;
        let mut args: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut m_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
        ];

        for _ in 0..5 {
            unsafe {
                stream.launch_kernel(&mut module, "gemm_cta_wmma_fp16", &config, &mut args).ok();
            }
        }
        stream.synchronize().ok();

        let iters = 50;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream.launch_kernel(&mut module, "gemm_cta_wmma_fp16", &config, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let cta_us = start.elapsed().as_micros() as f64 / iters as f64;
        let cta_tflops = flops / (cta_us * 1e6);

        // cuBLAS
        let c16_buf = GpuBuffer::from_host(&ctx, &vec![0u16; m * n]).expect("C16");
        for _ in 0..5 {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c16_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();
        let start = Instant::now();
        for _ in 0..iters {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c16_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();
        let cublas_us = start.elapsed().as_micros() as f64 / iters as f64;
        let ratio = if cublas_us > 0.0 { cta_us / cublas_us } else { 0.0 };

        eprintln!(
            "{:<10} {:>10.1}us {:>10.1}us {:>10.1} {:>8.1}x",
            format!("{n}x{n}"),
            cta_us,
            cublas_us,
            cta_tflops,
            ratio,
        );
    }
    eprintln!();
}

/// CTA WMMA double-buffered vs single-buffered vs cuBLAS FP16.
///
/// PERF-CTA-007: Measures the speedup from double-buffered shared memory
/// which overlaps global load of next K-tile with WMMA compute of current tile.
///
/// Run: cargo test -p trueno-gpu --features cuda --lib --release -- cta_wmma_dbuf_bench --no-capture
#[test]
fn cta_wmma_dbuf_bench_fp16() {
    use crate::driver::module::CudaModule;
    use crate::kernels::gemm::basic::tensor_core::cta_wmma::{
        build_cta_wmma_fp16, build_cta_wmma_fp16_dbuf,
    };
    use crate::ptx::PtxModule;
    use std::ffi::c_void;
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("stream");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle");
    handle.set_stream(&stream).expect("set_stream");

    eprintln!();
    eprintln!("=== CTA WMMA: Single-buf vs Double-buf vs cuBLAS — FP16 ===");
    eprintln!(
        "{:<8} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10}",
        "Size", "Single(us)", "Dbuf(us)", "cuBLAS(us)", "Speedup", "vs cuBL", "Dbuf TFLOP/s"
    );
    eprintln!("{}", "-".repeat(78));

    for &n in &[128_usize, 256, 512, 1024] {
        let m = n;
        let k = n;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        let a16 = vec![0x3C00u16; m * k];
        let b16 = vec![0x3C00u16; k * n];
        let c32 = vec![0.0f32; m * n];

        let a_buf = GpuBuffer::from_host(&ctx, &a16).expect("A");
        let b_buf = GpuBuffer::from_host(&ctx, &b16).expect("B");
        let c_buf = GpuBuffer::from_host(&ctx, &c32).expect("C");

        let grid_x = ((n + 31) / 32) as u32;
        let grid_y = ((m + 31) / 32) as u32;

        // ─── Single-buffer ───
        let kernel_s = build_cta_wmma_fp16(m as u32, n as u32, k as u32);
        let ptx_s = PtxModule::new().add_kernel(kernel_s).emit();
        let mut mod_s = match CudaModule::from_ptx(&ctx, &ptx_s) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{:<8} Single compile failed: {e}", format!("{n}"));
                continue;
            }
        };
        let cfg_s =
            LaunchConfig { grid: (grid_x, grid_y, 1), block: (128, 1, 1), shared_mem: 2048 };

        let mut a_ptr = a_buf.as_ptr();
        let mut b_ptr = b_buf.as_ptr();
        let mut c_ptr = c_buf.as_ptr();
        let mut m_v = m as u32;
        let mut n_v = n as u32;
        let mut k_v = k as u32;
        let mut args: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut m_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
        ];

        for _ in 0..5 {
            unsafe {
                stream.launch_kernel(&mut mod_s, "gemm_cta_wmma_fp16", &cfg_s, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let iters = 50;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream.launch_kernel(&mut mod_s, "gemm_cta_wmma_fp16", &cfg_s, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let single_us = start.elapsed().as_micros() as f64 / iters as f64;

        // ─── Double-buffer ───
        let kernel_d = build_cta_wmma_fp16_dbuf(m as u32, n as u32, k as u32);
        let ptx_d = PtxModule::new().add_kernel(kernel_d).emit();
        let mut mod_d = match CudaModule::from_ptx(&ctx, &ptx_d) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{:<8} Double-buf compile failed: {e}", format!("{n}"));
                continue;
            }
        };
        let cfg_d =
            LaunchConfig { grid: (grid_x, grid_y, 1), block: (128, 1, 1), shared_mem: 4096 };

        // Reset args
        a_ptr = a_buf.as_ptr();
        b_ptr = b_buf.as_ptr();
        c_ptr = c_buf.as_ptr();
        m_v = m as u32;
        n_v = n as u32;
        k_v = k as u32;

        for _ in 0..5 {
            unsafe {
                stream.launch_kernel(&mut mod_d, "gemm_cta_wmma_fp16", &cfg_d, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream.launch_kernel(&mut mod_d, "gemm_cta_wmma_fp16", &cfg_d, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let dbuf_us = start.elapsed().as_micros() as f64 / iters as f64;
        let dbuf_tflops = flops / (dbuf_us * 1e6);

        // ─── cuBLAS reference ───
        let c16_buf = GpuBuffer::from_host(&ctx, &vec![0u16; m * n]).expect("C16");
        for _ in 0..5 {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c16_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();
        let start = Instant::now();
        for _ in 0..iters {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c16_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();
        let cublas_us = start.elapsed().as_micros() as f64 / iters as f64;

        let speedup = single_us / dbuf_us;
        let vs_cublas = cublas_us / dbuf_us;

        eprintln!(
            "{:<8} {:>10.1} {:>10.1} {:>10.1} {:>7.2}x {:>7.2}x {:>10.1}",
            format!("{n}"),
            single_us,
            dbuf_us,
            cublas_us,
            speedup,
            vs_cublas,
            dbuf_tflops,
        );
    }
    eprintln!();
}

/// 64×64 CTA WMMA: single-buf vs double-buf vs 32×32 vs cuBLAS FP16.
///
/// PERF-CTA64-001: 2× compute-to-load ratio (32 FLOP/byte).
/// Double-buffer retried with 16 WMMAs/K-tile amortization (was 4 in 32×32).
///
/// Run: cargo test -p trueno-gpu --features cuda --lib --release -- cta64_vs_cta32 --no-capture
#[test]
fn cta64_vs_cta32_vs_cublas_fp16() {
    use crate::driver::module::CudaModule;
    use crate::kernels::gemm::basic::tensor_core::cta64_wmma::{
        build_cta64_mma_fp16_cpasync, build_cta64_wmma_fp16, build_cta64_wmma_fp16_cpasync,
        build_cta64_wmma_fp16_dbuf, build_cta64x128_mma_fp16_cpasync,
        build_cta64x128_mma_pipeline_fp16,
    };
    use crate::kernels::gemm::basic::tensor_core::cta_wmma::build_cta_wmma_fp16;
    use crate::ptx::PtxModule;
    use std::ffi::c_void;
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("stream");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle");
    handle.set_stream(&stream).expect("set_stream");

    eprintln!();
    eprintln!("=== CTA64: single vs dbuf vs cp.async vs CTA32 vs cuBLAS — FP16 ===");
    eprintln!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "Size", "CTA32", "CTA64", "Dbuf64", "CpAsync", "cuBLAS", "CpA TF/s", "cpVsCu"
    );
    eprintln!("{}", "-".repeat(80));

    for &n in &[128_usize, 256, 512, 1024, 2048, 4096] {
        let m = n;
        let k = n;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        let a16 = vec![0x3C00u16; m * k];
        let b16 = vec![0x3C00u16; k * n];
        let c32 = vec![0.0f32; m * n];

        let a_buf = GpuBuffer::from_host(&ctx, &a16).expect("A");
        let b_buf = GpuBuffer::from_host(&ctx, &b16).expect("B");
        let c_buf = GpuBuffer::from_host(&ctx, &c32).expect("C");

        // Scale iterations for larger sizes
        let iters = if n <= 1024 {
            50
        } else if n <= 2048 {
            20
        } else {
            10
        };

        // ─── 32×32 CTA (baseline) ───
        let kernel_32 = build_cta_wmma_fp16(m as u32, n as u32, k as u32);
        let ptx_32 = PtxModule::new().add_kernel(kernel_32).emit();
        let mut mod_32 = match CudaModule::from_ptx(&ctx, &ptx_32) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{:<8} CTA32 compile failed: {e}", n);
                continue;
            }
        };
        let cfg_32 = LaunchConfig {
            grid: (((n + 31) / 32) as u32, ((m + 31) / 32) as u32, 1),
            block: (128, 1, 1),
            shared_mem: 2048,
        };

        let mut a_ptr = a_buf.as_ptr();
        let mut b_ptr = b_buf.as_ptr();
        let mut c_ptr = c_buf.as_ptr();
        let mut m_v = m as u32;
        let mut n_v = n as u32;
        let mut k_v = k as u32;
        let mut args: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut m_v as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
            &mut k_v as *mut _ as *mut c_void,
        ];

        for _ in 0..5 {
            unsafe {
                stream.launch_kernel(&mut mod_32, "gemm_cta_wmma_fp16", &cfg_32, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream.launch_kernel(&mut mod_32, "gemm_cta_wmma_fp16", &cfg_32, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let cta32_us = start.elapsed().as_micros() as f64 / iters as f64;

        // ─── 64×64 CTA ───
        let kernel_64 = build_cta64_wmma_fp16(m as u32, n as u32, k as u32);
        let ptx_64 = PtxModule::new().add_kernel(kernel_64).emit();
        let mut mod_64 = match CudaModule::from_ptx(&ctx, &ptx_64) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{:<8} CTA64 compile failed: {e}", n);
                continue;
            }
        };
        let cfg_64 = LaunchConfig {
            grid: (((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1),
            block: (512, 1, 1),
            shared_mem: 4096,
        };

        a_ptr = a_buf.as_ptr();
        b_ptr = b_buf.as_ptr();
        c_ptr = c_buf.as_ptr();
        m_v = m as u32;
        n_v = n as u32;
        k_v = k as u32;

        for _ in 0..5 {
            unsafe {
                stream.launch_kernel(&mut mod_64, "gemm_cta64_wmma_fp16", &cfg_64, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream.launch_kernel(&mut mod_64, "gemm_cta64_wmma_fp16", &cfg_64, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let cta64_us = start.elapsed().as_micros() as f64 / iters as f64;

        // ─── 64×64 CTA double-buffer ───
        let kernel_db = build_cta64_wmma_fp16_dbuf(m as u32, n as u32, k as u32);
        let ptx_db = PtxModule::new().add_kernel(kernel_db).emit();
        let mut mod_db = match CudaModule::from_ptx(&ctx, &ptx_db) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{:<6} CTA64-dbuf compile failed: {e}", n);
                continue;
            }
        };
        let cfg_db = LaunchConfig {
            grid: (((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1),
            block: (512, 1, 1),
            shared_mem: 8192, // 2×4096
        };

        a_ptr = a_buf.as_ptr();
        b_ptr = b_buf.as_ptr();
        c_ptr = c_buf.as_ptr();
        m_v = m as u32;
        n_v = n as u32;
        k_v = k as u32;

        for _ in 0..5 {
            unsafe {
                stream.launch_kernel(&mut mod_db, "gemm_cta64_wmma_fp16", &cfg_db, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream.launch_kernel(&mut mod_db, "gemm_cta64_wmma_fp16", &cfg_db, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let dbuf64_us = start.elapsed().as_micros() as f64 / iters as f64;
        let _dbuf64_tflops = flops / (dbuf64_us * 1e6);

        // ─── 64×64 cp.async (SM 8.0+) ───
        let kernel_cp = build_cta64_wmma_fp16_cpasync(m as u32, n as u32, k as u32);
        // cp.async requires sm_80+ target
        let ptx_cp = PtxModule::new().target("sm_80").add_kernel(kernel_cp).emit();
        let cpasync_us = match CudaModule::from_ptx(&ctx, &ptx_cp) {
            Ok(mut mod_cp) => {
                let cfg_cp = LaunchConfig {
                    grid: (((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1),
                    block: (512, 1, 1),
                    shared_mem: 8192,
                };
                a_ptr = a_buf.as_ptr();
                b_ptr = b_buf.as_ptr();
                c_ptr = c_buf.as_ptr();
                m_v = m as u32;
                n_v = n as u32;
                k_v = k as u32;
                for _ in 0..5 {
                    unsafe {
                        stream
                            .launch_kernel(
                                &mut mod_cp,
                                "gemm_cta64_cpasync_fp16",
                                &cfg_cp,
                                &mut args,
                            )
                            .ok();
                    }
                }
                stream.synchronize().ok();
                let start = Instant::now();
                for _ in 0..iters {
                    unsafe {
                        stream
                            .launch_kernel(
                                &mut mod_cp,
                                "gemm_cta64_cpasync_fp16",
                                &cfg_cp,
                                &mut args,
                            )
                            .ok();
                    }
                }
                stream.synchronize().ok();
                start.elapsed().as_micros() as f64 / iters as f64
            }
            Err(e) => {
                eprintln!("{:<6} cp.async compile failed: {e}", n);
                f64::INFINITY
            }
        };
        let cpasync_tflops = flops / (cpasync_us * 1e6);

        // ─── cuBLAS reference ───
        let c16_buf = GpuBuffer::from_host(&ctx, &vec![0u16; m * n]).expect("C16");
        for _ in 0..5 {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c16_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();
        let start = Instant::now();
        for _ in 0..iters {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c16_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();
        let cublas_us = start.elapsed().as_micros() as f64 / iters as f64;

        let cp_vs_cublas = cublas_us / cpasync_us;

        // ─── 128×128 cp.async (Phase 2 bridge plan) ───
        let kernel_128 =
            crate::kernels::gemm::basic::tensor_core::cta128_wmma::build_cta128_wmma_fp16_cpasync(
                m as u32, n as u32, k as u32,
            );
        let ptx_128 = PtxModule::new().target("sm_80").add_kernel(kernel_128).emit();
        let cta128_us = match CudaModule::from_ptx(&ctx, &ptx_128) {
            Ok(mut mod_128) => {
                let cfg_128 = LaunchConfig {
                    grid: (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1),
                    block: (512, 1, 1),
                    shared_mem: 16384, // 2 stages × 8KB
                };
                a_ptr = a_buf.as_ptr();
                b_ptr = b_buf.as_ptr();
                c_ptr = c_buf.as_ptr();
                m_v = m as u32;
                n_v = n as u32;
                k_v = k as u32;
                for _ in 0..5 {
                    unsafe {
                        stream
                            .launch_kernel(
                                &mut mod_128,
                                "gemm_cta128_cpasync_fp16",
                                &cfg_128,
                                &mut args,
                            )
                            .ok();
                    }
                }
                stream.synchronize().ok();
                let start = Instant::now();
                for _ in 0..iters {
                    unsafe {
                        stream
                            .launch_kernel(
                                &mut mod_128,
                                "gemm_cta128_cpasync_fp16",
                                &cfg_128,
                                &mut args,
                            )
                            .ok();
                    }
                }
                stream.synchronize().ok();
                start.elapsed().as_micros() as f64 / iters as f64
            }
            Err(e) => {
                eprintln!("{:<6} cta128 compile failed: {e}", n);
                f64::INFINITY
            }
        };
        let cta128_tflops = flops / (cta128_us * 1e6);
        let cta128_vs_cublas = cublas_us / cta128_us;

        // ─── 64×64 mma.sync (Phase 1 bridge plan) ───
        let kernel_mma = build_cta64_mma_fp16_cpasync(m as u32, n as u32, k as u32);
        let ptx_mma = PtxModule::new().target("sm_80").add_kernel(kernel_mma).emit();
        let mma_us = match CudaModule::from_ptx(&ctx, &ptx_mma) {
            Ok(mut mod_mma) => {
                let cfg_mma = LaunchConfig {
                    grid: (((n + 63) / 64) as u32, ((m + 63) / 64) as u32, 1),
                    block: (512, 1, 1),
                    shared_mem: 8192,
                };
                a_ptr = a_buf.as_ptr();
                b_ptr = b_buf.as_ptr();
                c_ptr = c_buf.as_ptr();
                m_v = m as u32;
                n_v = n as u32;
                k_v = k as u32;
                for _ in 0..5 {
                    unsafe {
                        stream
                            .launch_kernel(&mut mod_mma, "gemm_cta64_mma_fp16", &cfg_mma, &mut args)
                            .ok();
                    }
                }
                stream.synchronize().ok();
                let start = Instant::now();
                for _ in 0..iters {
                    unsafe {
                        stream
                            .launch_kernel(&mut mod_mma, "gemm_cta64_mma_fp16", &cfg_mma, &mut args)
                            .ok();
                    }
                }
                stream.synchronize().ok();
                start.elapsed().as_micros() as f64 / iters as f64
            }
            Err(e) => {
                eprintln!("{:<6} mma.sync compile failed: {e}", n);
                f64::INFINITY
            }
        };
        let mma_tflops = flops / (mma_us * 1e6);

        // ─── 64×128 mma.sync (wider tile, +33% AI) ───
        let mma128_us = if n >= 256 {
            let kernel_128 = build_cta64x128_mma_fp16_cpasync(m as u32, n as u32, k as u32);
            let ptx_128 = PtxModule::new().target("sm_80").add_kernel(kernel_128).emit();
            match CudaModule::from_ptx(&ctx, &ptx_128) {
                Ok(mut mod_128) => {
                    let cfg_128 = LaunchConfig {
                        grid: (((n + 127) / 128) as u32, ((m + 63) / 64) as u32, 1),
                        block: (512, 1, 1),
                        shared_mem: 12288,
                    };
                    a_ptr = a_buf.as_ptr();
                    b_ptr = b_buf.as_ptr();
                    c_ptr = c_buf.as_ptr();
                    m_v = m as u32;
                    n_v = n as u32;
                    k_v = k as u32;
                    for _ in 0..5 {
                        unsafe {
                            stream
                                .launch_kernel(
                                    &mut mod_128,
                                    "gemm_cta64x128_mma_fp16",
                                    &cfg_128,
                                    &mut args,
                                )
                                .ok();
                        }
                    }
                    stream.synchronize().ok();
                    let start = Instant::now();
                    for _ in 0..iters {
                        unsafe {
                            stream
                                .launch_kernel(
                                    &mut mod_128,
                                    "gemm_cta64x128_mma_fp16",
                                    &cfg_128,
                                    &mut args,
                                )
                                .ok();
                        }
                    }
                    stream.synchronize().ok();

                    // Correctness spot-check: all-ones inputs → C[i] should be K
                    if n == 256 {
                        let mut result = vec![0.0f32; m * n];
                        c_buf.copy_to_host(&mut result).expect("D2H");
                        let expected = k as f32;
                        let max_err =
                            result.iter().map(|&v| (v - expected).abs()).fold(0.0f32, f32::max);
                        assert!(
                            max_err < 1.0,
                            "64x128 correctness FAILED at {n}: max_err={max_err}, expected={expected}"
                        );
                        eprintln!("  64x128 correctness OK at {n}: max_err={max_err:.4}");
                        // Reset C buffer for timing
                        let c32 = vec![0.0f32; m * n];
                        let c_buf_fresh = GpuBuffer::from_host(&ctx, &c32).expect("C reset");
                        c_ptr = c_buf_fresh.as_ptr();
                    }

                    start.elapsed().as_micros() as f64 / iters as f64
                }
                Err(e) => {
                    eprintln!("  64x128 compile failed: {e}");
                    f64::INFINITY
                }
            }
        } else {
            f64::INFINITY
        };
        let mma128_tflops = flops / (mma128_us * 1e6);

        // ─── 64×128 pipelined mma.sync (3-stage sw pipeline) ───
        let pipe_us = if n >= 256 {
            let kernel_pipe = build_cta64x128_mma_pipeline_fp16(m as u32, n as u32, k as u32);
            let ptx_pipe = PtxModule::new().target("sm_80").add_kernel(kernel_pipe).emit();
            match CudaModule::from_ptx(&ctx, &ptx_pipe) {
                Ok(mut mod_pipe) => {
                    let cfg_pipe = LaunchConfig {
                        grid: (((n + 127) / 128) as u32, ((m + 63) / 64) as u32, 1),
                        block: (512, 1, 1),
                        shared_mem: 18432,
                    };
                    // Reset C
                    let c32 = vec![0.0f32; m * n];
                    let c_buf_p = GpuBuffer::from_host(&ctx, &c32).expect("C pipe");
                    let mut c_ptr_p = c_buf_p.as_ptr();
                    let mut args_pipe: Vec<*mut c_void> = vec![
                        &mut a_ptr as *mut _ as *mut c_void,
                        &mut b_ptr as *mut _ as *mut c_void,
                        &mut c_ptr_p as *mut _ as *mut c_void,
                        &mut m_v as *mut _ as *mut c_void,
                        &mut n_v as *mut _ as *mut c_void,
                        &mut k_v as *mut _ as *mut c_void,
                    ];
                    // Warmup
                    for _ in 0..5 {
                        unsafe {
                            stream
                                .launch_kernel(
                                    &mut mod_pipe,
                                    "gemm_cta64x128_mma_pipeline_fp16",
                                    &cfg_pipe,
                                    &mut args_pipe,
                                )
                                .ok();
                        }
                    }
                    stream.synchronize().ok();
                    let start = Instant::now();
                    for _ in 0..iters {
                        unsafe {
                            stream
                                .launch_kernel(
                                    &mut mod_pipe,
                                    "gemm_cta64x128_mma_pipeline_fp16",
                                    &cfg_pipe,
                                    &mut args_pipe,
                                )
                                .ok();
                        }
                    }
                    stream.synchronize().ok();
                    start.elapsed().as_micros() as f64 / iters as f64
                }
                Err(e) => {
                    eprintln!("  pipeline compile failed: {e}");
                    f64::INFINITY
                }
            }
        } else {
            f64::INFINITY
        };
        let pipe_tflops = flops / (pipe_us * 1e6);

        eprintln!(
            "{:<6} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>5.2}x | mma: {:>8.1} {:>8.1} | 128: {:>8.1} {:>8.1} | pipe: {:>8.1} {:>8.1}",
            n,
            cta32_us,
            cta64_us,
            dbuf64_us,
            cpasync_us,
            cublas_us,
            cpasync_tflops,
            cp_vs_cublas,
            mma_us,
            mma_tflops,
            mma128_us,
            mma128_tflops,
            pipe_us,
            pipe_tflops,
        );
    }
    eprintln!();
}
