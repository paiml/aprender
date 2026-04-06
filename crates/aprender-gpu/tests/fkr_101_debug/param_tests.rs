//! Parameter and load loop debug tests: 5-param, loadloop, and barebones.

use super::common::*;

/// FKR-101-5PARAM: Test debug with 5 parameters (same as instrumented)
#[test]
fn fkr_101_5param_debug_test() {
    if !cuda_available() {
        eprintln!("SKIPPED: No CUDA");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    const SMEM_SIZE: usize = 12544 * 3;

    // Build kernel with all 5 parameters
    let kernel = PtxKernel::new("five_param_debug")
        .param(PtxType::U64, "input_batch")
        .param(PtxType::U64, "output_batch")
        .param(PtxType::U64, "output_sizes")
        .param(PtxType::U64, "debug_buf")
        .param(PtxType::U32, "batch_size")
        .shared_memory(SMEM_SIZE)
        .build(|ctx| {
            let _input_ptr = ctx.load_param_u64("input_batch");
            let _output_ptr = ctx.load_param_u64("output_batch");
            let _sizes_ptr = ctx.load_param_u64("output_sizes");
            let debug_ptr = ctx.load_param_u64("debug_buf");
            let batch_size = ctx.load_param_u32("batch_size");

            let tid = ctx.special_reg(PtxReg::TidX);
            let bid = ctx.special_reg(PtxReg::CtaIdX);
            let warp_id = ctx.shr_u32_imm(tid, 5);
            let lane_mask = ctx.mov_u32_imm(31);
            let lane_id = ctx.and_u32(tid, lane_mask);

            // Only lane 0 of each warp processes
            let zero_check = ctx.mov_u32_imm(0);
            let is_leader = ctx.setp_eq_u32(lane_id, zero_check);
            ctx.branch_if_not(is_leader, "L_not_leader");

            // Compute page index
            let warps_per_block = ctx.mov_u32_imm(3);
            let block_offset = ctx.mul_lo_u32(bid, warps_per_block);
            let page_idx = ctx.add_u32_reg(block_offset, warp_id);

            // Bounds check
            let out_of_bounds = ctx.setp_ge_u32(page_idx, batch_size);
            ctx.branch_if(out_of_bounds, "L_not_leader");

            // FIRST marker - if we get here, bounds check passed
            ctx.emit_debug_marker(debug_ptr, 0xAA000000);

            // Compute warp offset
            let warp_size = ctx.mov_u32_imm(12544);
            let warp_off = ctx.mul_lo_u32(warp_id, warp_size);

            // Second marker
            ctx.emit_debug_marker(debug_ptr, 0xAA000001);

            // Write to shared memory
            let val = ctx.mov_u32_imm(0xDEADBEEF);
            ctx.st_shared_u32(warp_off, val);

            // Third marker
            ctx.emit_debug_marker(debug_ptr, 0xAA000002);

            ctx.label("L_not_leader");
            ctx.ret();
        });

    let ptx =
        PtxModule::new().version(8, 0).target("sm_89").address_size(64).add_kernel(kernel).emit();

    println!("=== 5-Param Debug PTX ===\n{}", ptx);

    // Allocate buffers
    let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4096).unwrap();
    input_buf.copy_from_host(&vec![0u8; 4096]).unwrap();

    let mut output_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4352).unwrap();
    let mut sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 64]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig { grid: (1, 1, 1), block: (96, 1, 1), shared_mem: 0 };

    let batch_size: u32 = 1; // Only process 1 page
    let mut args: [*mut c_void; 5] = [
        input_buf.as_kernel_arg(),
        output_buf.as_kernel_arg(),
        sizes_buf.as_kernel_arg(),
        debug_buf.as_kernel_arg(),
        &batch_size as *const u32 as *mut c_void,
    ];

    println!("Launching 5-param kernel with batch_size={}...", batch_size);
    unsafe {
        stream
            .launch_kernel(&mut module, "five_param_debug", &config, &mut args)
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
        panic!("5-param test crashed: {:?}", e);
    }

    assert_eq!(output[0], 3, "Should have 3 markers (1 warp processed)");
    println!("5-param debug test PASSED!");
}

/// FKR-101-LOADLOOP: Test debug with data loading loop
#[test]
fn fkr_101_loadloop_debug_test() {
    if !cuda_available() {
        eprintln!("SKIPPED: No CUDA");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    const SMEM_SIZE: usize = 12544 * 3;
    const PAGE_SIZE_VAL: u32 = 4096;

    // Build kernel with load loop
    let kernel = PtxKernel::new("loadloop_debug")
        .param(PtxType::U64, "input_batch")
        .param(PtxType::U64, "debug_buf")
        .param(PtxType::U32, "batch_size")
        .shared_memory(SMEM_SIZE)
        .build(|ctx| {
            let input_ptr = ctx.load_param_u64("input_batch");
            let debug_ptr = ctx.load_param_u64("debug_buf");
            let batch_size = ctx.load_param_u32("batch_size");

            let tid = ctx.special_reg(PtxReg::TidX);
            let bid = ctx.special_reg(PtxReg::CtaIdX);
            let warp_id = ctx.shr_u32_imm(tid, 5);
            let lane_mask = ctx.mov_u32_imm(31);
            let lane_id = ctx.and_u32(tid, lane_mask);

            // Only lane 0 of each warp processes
            let zero_check = ctx.mov_u32_imm(0);
            let is_leader = ctx.setp_eq_u32(lane_id, zero_check);
            ctx.branch_if_not(is_leader, "L_not_leader");

            // Compute page index
            let warps_per_block = ctx.mov_u32_imm(3);
            let block_offset = ctx.mul_lo_u32(bid, warps_per_block);
            let page_idx = ctx.add_u32_reg(block_offset, warp_id);

            // Bounds check
            let out_of_bounds = ctx.setp_ge_u32(page_idx, batch_size);
            ctx.branch_if(out_of_bounds, "L_not_leader");

            ctx.emit_debug_marker(debug_ptr, 0xAA000000);

            // Compute warp offset
            let warp_size = ctx.mov_u32_imm(12544);
            let warp_off = ctx.mul_lo_u32(warp_id, warp_size);

            ctx.emit_debug_marker(debug_ptr, 0xAA000001);

            // Compute input page pointer
            let page_size_val = ctx.mov_u32_imm(PAGE_SIZE_VAL);
            let page_offset = ctx.mul_lo_u32(page_idx, page_size_val);
            let page_offset_64 = ctx.cvt_u64_u32(page_offset);
            let input_page_ptr = ctx.add_u64(input_ptr, page_offset_64);

            ctx.emit_debug_marker(debug_ptr, 0xAA000002);

            // Initialize load index
            let zero_val = ctx.mov_u32_imm(0);
            ctx.st_shared_u32(warp_off, zero_val);

            ctx.emit_debug_marker(debug_ptr, 0xAA000003);

            // Load loop - limited to 10 iterations for debugging
            ctx.label("L_load_loop");
            ctx.emit_debug_marker(debug_ptr, 0xBB000000);

            let idx = ctx.ld_shared_u32(warp_off);

            // EARLY EXIT: Stop after 10 iterations
            let ten = ctx.mov_u32_imm(40); // 10 u32s = 40 bytes
            let early_exit = ctx.setp_ge_u32(idx, ten);
            ctx.branch_if(early_exit, "L_load_done");

            let load_done = ctx.setp_ge_u32(idx, page_size_val);
            ctx.branch_if(load_done, "L_load_done");

            ctx.emit_debug_marker(debug_ptr, 0xBB000001);

            let idx_64 = ctx.cvt_u64_u32(idx);
            let src_addr = ctx.add_u64(input_page_ptr, idx_64);
            let val = ctx.ld_global_u32(src_addr);

            ctx.emit_debug_marker(debug_ptr, 0xBB000002);

            let dst_off = ctx.add_u32_reg(warp_off, idx);
            ctx.st_shared_u32(dst_off, val);

            ctx.emit_debug_marker(debug_ptr, 0xBB000003);

            let four = ctx.mov_u32_imm(4);
            let idx_next = ctx.add_u32_reg(idx, four);
            ctx.st_shared_u32(warp_off, idx_next);

            ctx.branch("L_load_loop");

            ctx.label("L_load_done");
            ctx.emit_debug_marker(debug_ptr, 0xCC000000);

            ctx.label("L_not_leader");
            ctx.ret();
        });

    let ptx =
        PtxModule::new().version(8, 0).target("sm_89").address_size(64).add_kernel(kernel).emit();

    println!("=== LoadLoop Debug PTX ===\n{}", ptx);

    // Allocate buffers
    let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4096).unwrap();
    input_buf.copy_from_host(&(0..4096u32).map(|i| (i % 256) as u8).collect::<Vec<_>>()).unwrap();

    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 256).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 256]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig { grid: (1, 1, 1), block: (96, 1, 1), shared_mem: 0 };

    let batch_size: u32 = 1;
    let mut args: [*mut c_void; 3] = [
        input_buf.as_kernel_arg(),
        debug_buf.as_kernel_arg(),
        &batch_size as *const u32 as *mut c_void,
    ];

    println!("Launching loadloop kernel...");
    unsafe {
        stream
            .launch_kernel(&mut module, "loadloop_debug", &config, &mut args)
            .expect("Kernel launch");
    }

    let sync_result = stream.synchronize();

    let mut output = vec![0u32; 256];
    debug_buf.copy_to_host(&mut output).unwrap();

    println!("Counter: {}", output[0]);
    for i in 0..output[0].min(50) as usize {
        let m = output[i + 1];
        let name = match m {
            0xAA000000 => "BOUNDS_PASS",
            0xAA000001 => "WARP_OFF",
            0xAA000002 => "INPUT_PTR",
            0xAA000003 => "INIT_IDX",
            0xBB000000 => "LOOP_ITER",
            0xBB000001 => "BEFORE_LOAD",
            0xBB000002 => "AFTER_LOAD",
            0xBB000003 => "AFTER_STORE",
            0xCC000000 => "LOAD_DONE",
            _ => "UNKNOWN",
        };
        println!("  [{:2}] 0x{:08X} ({})", i, m, name);
    }

    if let Err(e) = sync_result {
        panic!("LoadLoop test crashed: {:?}", e);
    }

    assert!(output[0] >= 5, "Should have at least 5 markers");
    println!("LoadLoop debug test PASSED!");
}

/// FKR-101-BAREBONES: Absolute minimum test matching failing kernel structure
#[test]
fn fkr_101_barebones_test() {
    if !cuda_available() {
        eprintln!("SKIPPED: No CUDA");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    // Same SMEM size as failing kernel
    const SMEM_SIZE: usize = 12544 * 3; // 37632

    // Build kernel with EXACT same parameters as failing kernel
    let kernel = PtxKernel::new("barebones")
        .param(PtxType::U64, "input_batch")
        .param(PtxType::U64, "output_batch")
        .param(PtxType::U64, "output_sizes")
        .param(PtxType::U64, "debug_buf")
        .param(PtxType::U32, "batch_size")
        .shared_memory(SMEM_SIZE)
        .build(|ctx| {
            let _input_ptr = ctx.load_param_u64("input_batch");
            let _output_ptr = ctx.load_param_u64("output_batch");
            let _sizes_ptr = ctx.load_param_u64("output_sizes");
            let debug_ptr = ctx.load_param_u64("debug_buf");
            let batch_size = ctx.load_param_u32("batch_size");

            let tid = ctx.special_reg(PtxReg::TidX);
            let bid = ctx.special_reg(PtxReg::CtaIdX);
            let warp_id = ctx.shr_u32_imm(tid, 5);
            let lane_mask = ctx.mov_u32_imm(31);
            let lane_id = ctx.and_u32(tid, lane_mask);

            let zero_check = ctx.mov_u32_imm(0);
            let is_leader = ctx.setp_eq_u32(lane_id, zero_check);
            ctx.branch_if_not(is_leader, "L_end");

            let warps_per_block = ctx.mov_u32_imm(3);
            let block_offset = ctx.mul_lo_u32(bid, warps_per_block);
            let page_idx = ctx.add_u32_reg(block_offset, warp_id);

            let out_of_bounds = ctx.setp_ge_u32(page_idx, batch_size);
            ctx.branch_if(out_of_bounds, "L_end");

            // Write one marker
            ctx.emit_debug_marker(debug_ptr, 0xDEADBEEF);

            ctx.label("L_end");
            ctx.ret();
        });

    let ptx =
        PtxModule::new().version(8, 0).target("sm_89").address_size(64).add_kernel(kernel).emit();

    println!("=== Barebones PTX ===\n{}", ptx);

    // Allocate SAME buffers as failing test
    let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4096).unwrap();
    let mut output_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4352).unwrap();
    let mut sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1024).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 1024]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    // SAME config as failing test
    let config = LaunchConfig { grid: (1, 1, 1), block: (96, 1, 1), shared_mem: 0 };

    let batch_size: u32 = 1;
    let mut args: [*mut c_void; 5] = [
        input_buf.as_kernel_arg(),
        output_buf.as_kernel_arg(),
        sizes_buf.as_kernel_arg(),
        debug_buf.as_kernel_arg(),
        &batch_size as *const u32 as *mut c_void,
    ];

    println!("Launching barebones kernel...");
    unsafe {
        stream.launch_kernel(&mut module, "barebones", &config, &mut args).expect("Kernel launch");
    }

    let sync_result = stream.synchronize();

    let mut output = vec![0u32; 1024];
    debug_buf.copy_to_host(&mut output).unwrap();

    println!("Counter: {}", output[0]);
    if output[0] > 0 {
        println!("First marker: 0x{:08X}", output[1]);
    }

    if let Err(e) = sync_result {
        panic!("Barebones test crashed: {:?}", e);
    }

    assert_eq!(output[0], 1, "Should have exactly 1 marker");
    assert_eq!(output[1], 0xDEADBEEF, "Marker should be 0xDEADBEEF");
    println!("Barebones test PASSED!");
}
