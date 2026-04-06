//! Whisper-scale CUDA tests for batched attention pipeline.

#![allow(unused_imports)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{
    BatchedGemmKernel, BatchedTransposeKernel, InterleavedToBatchedKernel, Kernel,
};

use super::cpu_references::*;

/// Test 7: InterleavedToBatched at Whisper scale (seq_len=1500, n_heads=6, head_dim=64)
#[test]
#[cfg(feature = "cuda")]
fn test_interleaved_to_batched_whisper_scale() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let seq_len = 1500u32;
    let n_heads = 6u32;
    let head_dim = 64u32;
    let d_model = n_heads * head_dim;
    let total = seq_len * d_model;

    let input: Vec<f32> = (0..total).map(|i| (i % 1000) as f32 * 0.001).collect();
    let expected =
        cpu_interleaved_to_batched(&input, seq_len as usize, n_heads as usize, head_dim as usize);

    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total as usize).expect("Alloc failed");

    let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads = 256u32;
    let blocks = (total + threads - 1) / threads;
    let config = LaunchConfig { grid: (blocks, 1, 1), block: (threads, 1, 1), shared_mem: 0 };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> =
        vec![std::ptr::addr_of!(input_ptr) as *mut _, std::ptr::addr_of!(output_ptr) as *mut _];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; total as usize];
    output_buf.copy_to_host(&mut output).expect("Download failed");

    let mut mismatches = 0;
    let check_indices = [0, 1000, 10000, 100000, 500000, total as usize - 1];
    for &i in &check_indices {
        if i < total as usize {
            let delta = (output[i] - expected[i]).abs();
            if delta > 1e-5 {
                eprintln!("Mismatch at {}: GPU={} vs CPU={}", i, output[i], expected[i]);
                mismatches += 1;
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "InterleavedToBatched at Whisper scale has {} mismatches",
        mismatches
    );
    eprintln!("InterleavedToBatched Whisper scale test PASSED");
}

/// Test 8: BatchedTranspose at Whisper scale
#[test]
#[cfg(feature = "cuda")]
fn test_batched_transpose_whisper_scale() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let batch = 6u32;
    let rows = 1500u32;
    let cols = 64u32;
    let total = batch * rows * cols;

    let input: Vec<f32> = (0..total).map(|i| (i % 1000) as f32 * 0.001).collect();
    let expected = cpu_batched_transpose(&input, batch as usize, rows as usize, cols as usize);

    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total as usize).expect("Alloc failed");

    let kernel = BatchedTransposeKernel::new(batch, rows, cols);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads = 256u32;
    let elems_per_batch = rows * cols;
    let blocks_x = (elems_per_batch + threads - 1) / threads;
    let config = LaunchConfig { grid: (blocks_x, 1, batch), block: (threads, 1, 1), shared_mem: 0 };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(batch) as *mut _,
        std::ptr::addr_of!(rows) as *mut _,
        std::ptr::addr_of!(cols) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; total as usize];
    output_buf.copy_to_host(&mut output).expect("Download failed");

    let mut mismatches = 0;
    let check_indices = [0, 1000, 10000, 100000, 500000, total as usize - 1];
    for &i in &check_indices {
        if i < total as usize {
            let delta = (output[i] - expected[i]).abs();
            if delta > 1e-5 {
                eprintln!("Mismatch at {}: GPU={} vs CPU={}", i, output[i], expected[i]);
                mismatches += 1;
            }
        }
    }

    assert_eq!(mismatches, 0, "BatchedTranspose at Whisper scale has {} mismatches", mismatches);
    eprintln!("BatchedTranspose Whisper scale test PASSED");
}

/// Test 9: BatchedGemm Q @ K^T at Whisper scale
#[test]
#[cfg(feature = "cuda")]
fn test_batched_gemm_qkt_whisper_scale() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let batch = 6u32;
    let m = 1500u32;
    let n = 1500u32;
    let k = 64u32;

    let a: Vec<f32> = (0..batch * m * k).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();

    eprintln!("Computing CPU reference for GEMM [6, 1500, 64] @ [6, 64, 1500]...");
    let expected = cpu_batched_gemm(&a, &b, batch as usize, m as usize, n as usize, k as usize);
    eprintln!("CPU reference computed. Sample values: {:?}", &expected[..5]);

    let a_buf = GpuBuffer::from_host(&ctx, &a).expect("Upload A failed");
    let b_buf = GpuBuffer::from_host(&ctx, &b).expect("Upload B failed");
    let c_buf: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, (batch * m * n) as usize).expect("Alloc C failed");

    let kernel = BatchedGemmKernel::naive(batch, m, n, k);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads_per_block = 16u32;
    let blocks_x = (n + threads_per_block - 1) / threads_per_block;
    let blocks_y = (m + threads_per_block - 1) / threads_per_block;
    let config = LaunchConfig {
        grid: (blocks_x, blocks_y, batch),
        block: (threads_per_block, threads_per_block, 1),
        shared_mem: 0,
    };

    let a_ptr = a_buf.as_ptr();
    let b_ptr = b_buf.as_ptr();
    let c_ptr = c_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(a_ptr) as *mut _,
        std::ptr::addr_of!(b_ptr) as *mut _,
        std::ptr::addr_of!(c_ptr) as *mut _,
        std::ptr::addr_of!(batch) as *mut _,
        std::ptr::addr_of!(m) as *mut _,
        std::ptr::addr_of!(n) as *mut _,
        std::ptr::addr_of!(k) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; (batch * m * n) as usize];
    c_buf.copy_to_host(&mut output).expect("Download failed");

    eprintln!("GPU output sample: {:?}", &output[..5]);

    let mut mismatches = 0;
    let total = (batch * m * n) as usize;
    let check_indices = [0, 1000, 10000, 100000, 1000000, total - 1];
    for &i in &check_indices {
        if i < total {
            let delta = (output[i] - expected[i]).abs();
            let rel_delta =
                if expected[i].abs() > 1e-6 { delta / expected[i].abs() } else { delta };
            if rel_delta > 0.01 && delta > 1e-4 {
                eprintln!(
                    "Mismatch at {}: GPU={} vs CPU={}, delta={}, rel={}",
                    i, output[i], expected[i], delta, rel_delta
                );
                mismatches += 1;
            }
        }
    }

    assert_eq!(mismatches, 0, "BatchedGemm Q@K^T at Whisper scale has {} mismatches", mismatches);
    eprintln!("BatchedGemm Q@K^T Whisper scale test PASSED");
}

// Stub tests for non-CUDA builds
#[cfg(not(feature = "cuda"))]
#[test]
fn test_interleaved_to_batched_whisper_scale() {}
#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_transpose_whisper_scale() {}
#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_gemm_qkt_whisper_scale() {}
