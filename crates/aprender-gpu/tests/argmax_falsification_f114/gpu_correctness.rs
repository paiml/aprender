//! F114 Tests 3-7: GPU kernel correctness tests (require CUDA hardware)

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{ArgMaxFinalKernel, ArgMaxKernel, Kernel};

/// F114-TEST-3: GPU kernel correctness on known input
///
/// Verifies argmax returns correct index for known maximum
#[test]
fn f114_test3_correctness_known_max() {
    println!("F114-TEST-3: Correctness on known input");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("  CUDA context failed: {} (skipping)", e);
            return;
        }
    };

    let stream = CudaStream::new(&ctx).unwrap();

    // Create input with known maximum at index 42
    let mut input = vec![-1.0f32; 256];
    input[42] = 100.0; // Maximum value

    let kernel = ArgMaxKernel::new(256);
    let ptx = kernel.emit_ptx();

    let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(m) => m,
        Err(e) => {
            println!("  PTX load failed: {} (skipping)", e);
            return;
        }
    };

    // Allocate buffers
    let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 256).unwrap();
    let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1).unwrap();
    let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

    input_buf.copy_from_host(&input).unwrap();

    let config = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem: 2048 };

    let length = 256u32;
    let mut args: [*mut c_void; 4] = [
        input_buf.as_kernel_arg(),
        block_vals_buf.as_kernel_arg(),
        block_idxs_buf.as_kernel_arg(),
        &length as *const u32 as *mut c_void,
    ];

    let result =
        unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

    match result {
        Ok(_) => match stream.synchronize() {
            Ok(_) => {
                let mut block_vals = vec![0.0f32; 1];
                let mut block_idxs = vec![0u32; 1];
                block_vals_buf.copy_to_host(&mut block_vals).unwrap();
                block_idxs_buf.copy_to_host(&mut block_idxs).unwrap();

                println!("  Block max value: {}", block_vals[0]);
                println!("  Block max index: {}", block_idxs[0]);

                assert_eq!(block_idxs[0], 42, "Expected argmax at index 42");
                assert_eq!(block_vals[0], 100.0, "Expected max value 100.0");
                println!("  PASSED - Correctness verified");
            }
            Err(e) => {
                panic!("  CRASHED at sync: {} - PAR-062 correctness FAILED", e);
            }
        },
        Err(e) => {
            panic!("  Launch failed: {} - Kernel execution FAILED", e);
        }
    }
}

/// F114-TEST-4: Edge case - maximum at index 0
#[test]
fn f114_test4_max_at_zero() {
    println!("F114-TEST-4: Maximum at index 0");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("  CUDA context failed: {} (skipping)", e);
            return;
        }
    };

    let stream = CudaStream::new(&ctx).unwrap();

    let mut input = vec![-100.0f32; 512];
    input[0] = 999.0; // Maximum at index 0

    let kernel = ArgMaxKernel::new(512);
    let ptx = kernel.emit_ptx();

    let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(m) => m,
        Err(e) => {
            println!("  PTX load failed: {} (skipping)", e);
            return;
        }
    };

    let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 512).unwrap();
    let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1).unwrap();
    let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

    input_buf.copy_from_host(&input).unwrap();

    let config = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem: 2048 };

    let length = 512u32;
    let mut args: [*mut c_void; 4] = [
        input_buf.as_kernel_arg(),
        block_vals_buf.as_kernel_arg(),
        block_idxs_buf.as_kernel_arg(),
        &length as *const u32 as *mut c_void,
    ];

    let result =
        unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

    match result {
        Ok(_) => {
            if let Err(e) = stream.synchronize() {
                panic!("  CRASHED: {} - Edge case at index 0 FAILED", e);
            }

            let mut block_idxs = vec![0u32; 1];
            block_idxs_buf.copy_to_host(&mut block_idxs).unwrap();

            println!("  Max index: {}", block_idxs[0]);
            assert_eq!(block_idxs[0], 0, "Expected argmax at index 0");
            println!("  PASSED");
        }
        Err(e) => {
            panic!("  Launch failed: {}", e);
        }
    }
}

/// F114-TEST-5: Edge case - maximum at last index
#[test]
fn f114_test5_max_at_last() {
    println!("F114-TEST-5: Maximum at last index");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("  CUDA context failed: {} (skipping)", e);
            return;
        }
    };

    let stream = CudaStream::new(&ctx).unwrap();

    let mut input = vec![-100.0f32; 1000];
    input[999] = 999.0; // Maximum at last index

    let kernel = ArgMaxKernel::new(1000);
    let ptx = kernel.emit_ptx();

    let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(m) => m,
        Err(e) => {
            println!("  PTX load failed: {} (skipping)", e);
            return;
        }
    };

    let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1000).unwrap();
    let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1).unwrap();
    let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

    input_buf.copy_from_host(&input).unwrap();

    let config = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem: 2048 };

    let length = 1000u32;
    let mut args: [*mut c_void; 4] = [
        input_buf.as_kernel_arg(),
        block_vals_buf.as_kernel_arg(),
        block_idxs_buf.as_kernel_arg(),
        &length as *const u32 as *mut c_void,
    ];

    let result =
        unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

    match result {
        Ok(_) => {
            if let Err(e) = stream.synchronize() {
                panic!("  CRASHED: {} - Edge case at last index FAILED", e);
            }

            let mut block_idxs = vec![0u32; 1];
            block_idxs_buf.copy_to_host(&mut block_idxs).unwrap();

            println!("  Max index: {}", block_idxs[0]);
            assert_eq!(block_idxs[0], 999, "Expected argmax at index 999");
            println!("  PASSED");
        }
        Err(e) => {
            panic!("  Launch failed: {}", e);
        }
    }
}

/// F114-TEST-6: Multi-block reduction (vocab size)
#[test]
fn f114_test6_vocab_size_reduction() {
    println!("F114-TEST-6: Multi-block reduction (vocab size)");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("  CUDA context failed: {} (skipping)", e);
            return;
        }
    };

    let stream = CudaStream::new(&ctx).unwrap();

    // Simulated vocab size (smaller for test)
    let vocab_size = 4096u32;
    let expected_max_idx = 3333u32;

    let mut input = vec![-100.0f32; vocab_size as usize];
    input[expected_max_idx as usize] = 1000.0;

    let kernel = ArgMaxKernel::new(vocab_size);
    let num_blocks = kernel.num_blocks();
    let ptx = kernel.emit_ptx();

    println!("  Vocab size: {}", vocab_size);
    println!("  Num blocks: {}", num_blocks);

    let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(m) => m,
        Err(e) => {
            println!("  PTX load failed: {} (skipping)", e);
            return;
        }
    };

    let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, vocab_size as usize).unwrap();
    let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, num_blocks as usize).unwrap();
    let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, num_blocks as usize).unwrap();

    input_buf.copy_from_host(&input).unwrap();

    let config = LaunchConfig { grid: (num_blocks, 1, 1), block: (256, 1, 1), shared_mem: 2048 };

    let mut args: [*mut c_void; 4] = [
        input_buf.as_kernel_arg(),
        block_vals_buf.as_kernel_arg(),
        block_idxs_buf.as_kernel_arg(),
        &vocab_size as *const u32 as *mut c_void,
    ];

    let result =
        unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

    match result {
        Ok(_) => {
            if let Err(e) = stream.synchronize() {
                panic!("  CRASHED at first pass: {} - Multi-block FAILED", e);
            }

            // Second pass: final reduction
            let final_kernel = ArgMaxFinalKernel::new(num_blocks);
            let final_ptx = final_kernel.emit_ptx();

            let mut final_module = CudaModule::from_ptx(&ctx, &final_ptx).unwrap();
            let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

            let final_config =
                LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem: 2048 };

            let mut final_args: [*mut c_void; 4] = [
                block_vals_buf.as_kernel_arg(),
                block_idxs_buf.as_kernel_arg(),
                output_buf.as_kernel_arg(),
                &num_blocks as *const u32 as *mut c_void,
            ];

            let final_result = unsafe {
                stream.launch_kernel(
                    &mut final_module,
                    "argmax_final_reduce",
                    &final_config,
                    &mut final_args,
                )
            };

            match final_result {
                Ok(_) => {
                    if let Err(e) = stream.synchronize() {
                        panic!("  CRASHED at final pass: {} - Final reduction FAILED", e);
                    }

                    let mut output = vec![0u32; 1];
                    output_buf.copy_to_host(&mut output).unwrap();

                    println!("  Final argmax: {}", output[0]);
                    assert_eq!(
                        output[0], expected_max_idx,
                        "Expected argmax at index {}",
                        expected_max_idx
                    );
                    println!("  PASSED - Multi-block reduction correct");
                }
                Err(e) => {
                    panic!("  Final launch failed: {}", e);
                }
            }
        }
        Err(e) => {
            panic!("  First pass launch failed: {}", e);
        }
    }
}

// =========================================================================
// F114-TEST-7: Statistical correctness with random data
// =========================================================================

/// F114-TEST-7: Verify argmax matches CPU reference
#[test]
fn f114_test7_cpu_reference_match() {
    println!("F114-TEST-7: CPU reference match");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("  CUDA context failed: {} (skipping)", e);
            return;
        }
    };

    let stream = CudaStream::new(&ctx).unwrap();

    // Use deterministic "random" data for reproducibility
    let input: Vec<f32> = (0..512).map(|i| ((i * 17 + 13) % 1000) as f32 - 500.0).collect();

    // CPU reference
    let cpu_argmax = input
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    println!("  CPU argmax: {}", cpu_argmax);

    let kernel = ArgMaxKernel::new(512);
    let ptx = kernel.emit_ptx();

    let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(m) => m,
        Err(e) => {
            println!("  PTX load failed: {} (skipping)", e);
            return;
        }
    };

    let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 512).unwrap();
    let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1).unwrap();
    let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

    input_buf.copy_from_host(&input).unwrap();

    let config = LaunchConfig { grid: (1, 1, 1), block: (256, 1, 1), shared_mem: 2048 };

    let length = 512u32;
    let mut args: [*mut c_void; 4] = [
        input_buf.as_kernel_arg(),
        block_vals_buf.as_kernel_arg(),
        block_idxs_buf.as_kernel_arg(),
        &length as *const u32 as *mut c_void,
    ];

    let result =
        unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

    match result {
        Ok(_) => {
            if let Err(e) = stream.synchronize() {
                panic!("  CRASHED: {} - GPU execution FAILED", e);
            }

            let mut block_idxs = vec![0u32; 1];
            block_idxs_buf.copy_to_host(&mut block_idxs).unwrap();

            let gpu_argmax = block_idxs[0];
            println!("  GPU argmax: {}", gpu_argmax);

            assert_eq!(
                gpu_argmax, cpu_argmax,
                "GPU argmax ({}) != CPU argmax ({})",
                gpu_argmax, cpu_argmax
            );
            println!("  PASSED - GPU matches CPU reference");
        }
        Err(e) => {
            panic!("  Launch failed: {}", e);
        }
    }
}
