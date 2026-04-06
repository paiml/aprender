//! Minimal debug tests: basic debug buffer, shared memory, and global memory.

use super::common::*;

/// FKR-101-MINIMAL: Absolute minimal debug buffer test
#[test]
fn fkr_101_minimal_debug_test() {
    if !cuda_available() {
        eprintln!("FKR-101 SKIPPED: No CUDA device available");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    // Build minimal kernel that just writes a debug marker
    let kernel = PtxKernel::new("minimal_debug").param(PtxType::U64, "debug_buf").build(|ctx| {
        let debug_ptr = ctx.load_param_u64("debug_buf");
        let tid = ctx.special_reg(PtxReg::TidX);
        let zero = ctx.mov_u32_imm(0);
        let is_t0 = ctx.setp_eq_u32(tid, zero);
        ctx.branch_if_not(is_t0, "L_end");

        // Write a marker
        ctx.emit_debug_marker(debug_ptr, 0xCAFEBABE);

        ctx.label("L_end");
        ctx.ret();
    });

    let ptx =
        PtxModule::new().version(8, 0).target("sm_89").address_size(64).add_kernel(kernel).emit();

    println!("=== Minimal Debug PTX ===\n{}", ptx);

    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 64]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig { grid: (1, 1, 1), block: (32, 1, 1), shared_mem: 0 };

    let mut args: [*mut c_void; 1] = [debug_buf.as_kernel_arg()];

    unsafe {
        stream
            .launch_kernel(&mut module, "minimal_debug", &config, &mut args)
            .expect("Kernel launch");
    }

    stream.synchronize().expect("Sync");

    let mut output = vec![0u32; 64];
    debug_buf.copy_to_host(&mut output).unwrap();

    println!("Counter: {}", output[0]);
    println!("Marker: 0x{:08X}", output[1]);

    assert_eq!(output[0], 1, "Should have 1 marker");
    assert_eq!(output[1], 0xCAFEBABE, "Marker should be 0xCAFEBABE");
    println!("Minimal debug test PASSED!");
}

/// FKR-101-SMEM: Test debug with shared memory
#[test]
fn fkr_101_smem_debug_test() {
    if !cuda_available() {
        eprintln!("SKIPPED: No CUDA");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    const SMEM_SIZE: usize = 12544 * 3; // Same as instrumented kernel

    // Build kernel with shared memory
    let kernel = PtxKernel::new("smem_debug")
        .param(PtxType::U64, "debug_buf")
        .shared_memory(SMEM_SIZE)
        .build(|ctx| {
            let debug_ptr = ctx.load_param_u64("debug_buf");
            let tid = ctx.special_reg(PtxReg::TidX);
            let zero = ctx.mov_u32_imm(0);
            let is_t0 = ctx.setp_eq_u32(tid, zero);
            ctx.branch_if_not(is_t0, "L_end");

            // Write marker 1
            ctx.emit_debug_marker(debug_ptr, 0x11111111);

            // Write to shared memory
            let addr_0 = ctx.mov_u32_imm(0);
            let val = ctx.mov_u32_imm(0xDEADBEEF);
            ctx.st_shared_u32(addr_0, val);

            // Write marker 2
            ctx.emit_debug_marker(debug_ptr, 0x22222222);

            // Read from shared memory
            let addr_0_2 = ctx.mov_u32_imm(0);
            let _read_val = ctx.ld_shared_u32(addr_0_2);

            // Write marker 3
            ctx.emit_debug_marker(debug_ptr, 0x33333333);

            ctx.label("L_end");
            ctx.ret();
        });

    let ptx =
        PtxModule::new().version(8, 0).target("sm_89").address_size(64).add_kernel(kernel).emit();

    println!("=== SMEM Debug PTX ===\n{}", ptx);

    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 64]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (96, 1, 1), // 3 warps, same as instrumented
        shared_mem: 0,     // Static shared, not dynamic
    };

    let mut args: [*mut c_void; 1] = [debug_buf.as_kernel_arg()];

    println!("Launching kernel...");
    unsafe {
        stream.launch_kernel(&mut module, "smem_debug", &config, &mut args).expect("Kernel launch");
    }

    let sync_result = stream.synchronize();

    let mut output = vec![0u32; 64];
    debug_buf.copy_to_host(&mut output).unwrap();

    println!("Counter: {}", output[0]);
    for i in 0..output[0].min(10) as usize {
        println!("  Marker {}: 0x{:08X}", i, output[i + 1]);
    }

    if let Err(e) = sync_result {
        panic!("SMEM test crashed: {:?}", e);
    }

    assert_eq!(output[0], 3, "Should have 3 markers");
    println!("SMEM debug test PASSED!");
}

/// FKR-101-GLOBAL: Test debug with global memory read
#[test]
fn fkr_101_global_debug_test() {
    if !cuda_available() {
        eprintln!("SKIPPED: No CUDA");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    const SMEM_SIZE: usize = 12544 * 3;

    // Build kernel with global memory access
    let kernel = PtxKernel::new("global_debug")
        .param(PtxType::U64, "input_buf")
        .param(PtxType::U64, "debug_buf")
        .shared_memory(SMEM_SIZE)
        .build(|ctx| {
            let input_ptr = ctx.load_param_u64("input_buf");
            let debug_ptr = ctx.load_param_u64("debug_buf");
            let tid = ctx.special_reg(PtxReg::TidX);
            let zero = ctx.mov_u32_imm(0);
            let is_t0 = ctx.setp_eq_u32(tid, zero);
            ctx.branch_if_not(is_t0, "L_end");

            // Write marker 1
            ctx.emit_debug_marker(debug_ptr, 0x11111111);

            // Read from global memory
            let val = ctx.ld_global_u32(input_ptr);

            // Write marker 2
            ctx.emit_debug_marker(debug_ptr, 0x22222222);

            // Write to shared memory
            let addr_0 = ctx.mov_u32_imm(0);
            ctx.st_shared_u32(addr_0, val);

            // Write marker 3
            ctx.emit_debug_marker(debug_ptr, 0x33333333);

            // Read back from shared memory
            let addr_0_2 = ctx.mov_u32_imm(0);
            let _read_val = ctx.ld_shared_u32(addr_0_2);

            // Write marker 4
            ctx.emit_debug_marker(debug_ptr, 0x44444444);

            ctx.label("L_end");
            ctx.ret();
        });

    let ptx =
        PtxModule::new().version(8, 0).target("sm_89").address_size(64).add_kernel(kernel).emit();

    println!("=== Global Debug PTX ===\n{}", ptx);

    // Create input buffer with test data
    let mut input_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1024).unwrap();
    input_buf.copy_from_host(&vec![0x12345678u32; 1024]).unwrap();

    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 64]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig { grid: (1, 1, 1), block: (96, 1, 1), shared_mem: 0 };

    let mut args: [*mut c_void; 2] = [input_buf.as_kernel_arg(), debug_buf.as_kernel_arg()];

    println!("Launching kernel...");
    unsafe {
        stream
            .launch_kernel(&mut module, "global_debug", &config, &mut args)
            .expect("Kernel launch");
    }

    let sync_result = stream.synchronize();

    let mut output = vec![0u32; 64];
    debug_buf.copy_to_host(&mut output).unwrap();

    println!("Counter: {}", output[0]);
    for i in 0..output[0].min(10) as usize {
        println!("  Marker {}: 0x{:08X}", i, output[i + 1]);
    }

    if let Err(e) = sync_result {
        panic!("Global test crashed: {:?}", e);
    }

    assert_eq!(output[0], 4, "Should have 4 markers");
    println!("Global debug test PASSED!");
}
