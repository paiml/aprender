//! FKR-101 membar workaround (LVB-003) and fullload tests.

use super::super::common::*;

/// FKR-101-MEMBAR: Test membar.cta workaround for ComputedAddrFromLoaded bug (LVB-003)
///
/// Tests if inserting membar.cta before/after the problematic store helps.
#[test]
fn fkr_101_membar_workaround_test() {
    if !cuda_available() {
        eprintln!("SKIPPED: No CUDA");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    const SMEM_SIZE: usize = 12544 * 3;
    const PAGE_SIZE_VAL: u32 = 4096;
    const STATE_OFF: u32 = PAGE_SIZE_VAL + 8192 + 128 + 4;

    let kernel = PtxKernel::new("membar_workaround")
        .param(PtxType::U64, "debug_buf")
        .param(PtxType::U32, "batch_size")
        .shared_memory(SMEM_SIZE)
        .build(|ctx| {
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

            ctx.emit_debug_marker(debug_ptr, 0xAA000000);

            let warp_smem_size = ctx.mov_u32_imm(12544);
            let warp_off = ctx.mul_lo_u32(warp_id, warp_smem_size);

            // State offset
            let state_off_val = ctx.mov_u32_imm(STATE_OFF);
            let state_off = ctx.add_u32_reg(warp_off, state_off_val);

            // Initialize state
            let zero_val = ctx.mov_u32_imm(0);
            ctx.st_shared_u32(state_off, zero_val);

            ctx.emit_debug_marker(debug_ptr, 0xAA000001);

            // Load in_pos from shared (this is the "loaded value")
            let in_pos = ctx.ld_shared_u32(state_off);

            ctx.emit_debug_marker(debug_ptr, 0xAA000002);

            // Compute address using in_pos (loaded value)
            let computed_addr = ctx.add_u32_reg(warp_off, in_pos);

            // ====== LVB-003: Try membar.cta before store ======
            ctx.membar_cta();

            ctx.emit_debug_marker(debug_ptr, 0xBB000000);

            // Store constant to computed address - THIS IS THE CRASH POINT
            let test_constant = ctx.mov_u32_imm(0xCAFEBABE);
            ctx.st_shared_u32(computed_addr, test_constant);

            // ====== Try membar.cta after store ======
            ctx.membar_cta();

            ctx.emit_debug_marker(debug_ptr, 0xBB000001);

            ctx.label("L_end");
            ctx.ret();
        });

    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel)
        .emit();

    println!("=== Membar Workaround PTX ===\n{}", ptx);

    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 64]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (96, 1, 1),
        shared_mem: 0,
    };

    let batch_size: u32 = 1;
    let mut args: [*mut c_void; 2] = [
        debug_buf.as_kernel_arg(),
        &batch_size as *const u32 as *mut c_void,
    ];

    println!("Launching membar workaround kernel...");
    unsafe {
        stream
            .launch_kernel(&mut module, "membar_workaround", &config, &mut args)
            .expect("Kernel launch");
    }

    let sync_result = stream.synchronize();

    let mut output = vec![0u32; 64];
    debug_buf.copy_to_host(&mut output).unwrap();

    println!("Counter: {}", output[0]);
    for i in 0..output[0].min(10) as usize {
        let m = output[i + 1];
        let name = match m {
            0xAA000000 => "BOUNDS_OK",
            0xAA000001 => "STATE_INIT",
            0xAA000002 => "LOADED_IN_POS",
            0xBB000000 => "BEFORE_STORE",
            0xBB000001 => "AFTER_STORE",
            _ => "UNKNOWN",
        };
        println!("  [{:2}] 0x{:08X} ({})", i, m, name);
    }

    if let Err(e) = sync_result {
        println!("Membar workaround FAILED: {:?}", e);
        println!("LVB-003: membar.cta does NOT fix the ComputedAddrFromLoaded bug");
        panic!("Membar workaround crashed: {:?}", e);
    }

    assert_eq!(
        output[0], 5,
        "Should have 5 markers (membar workaround succeeded!)"
    );
    println!("LVB-003: membar.cta WORKS! Test PASSED!");
}

/// FKR-101-FULLLOAD: Test with full load loop but no compress loop
#[test]
fn fkr_101_fullload_test() {
    if !cuda_available() {
        eprintln!("SKIPPED: No CUDA");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    const SMEM_SIZE: usize = 12544 * 3;
    const PAGE_SIZE_VAL: u32 = 4096;
    const HASH_TABLE_SIZE: u32 = 8192;

    let kernel = PtxKernel::new("fullload")
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

            let zero_check = ctx.mov_u32_imm(0);
            let is_leader = ctx.setp_eq_u32(lane_id, zero_check);
            ctx.branch_if_not(is_leader, "L_end");

            let warps_per_block = ctx.mov_u32_imm(3);
            let block_offset = ctx.mul_lo_u32(bid, warps_per_block);
            let page_idx = ctx.add_u32_reg(block_offset, warp_id);

            let out_of_bounds = ctx.setp_ge_u32(page_idx, batch_size);
            ctx.branch_if(out_of_bounds, "L_end");

            ctx.emit_debug_marker(debug_ptr, 0x11111111); // START

            let warp_size = ctx.mov_u32_imm(12544);
            let warp_off = ctx.mul_lo_u32(warp_id, warp_size);

            let page_size_val = ctx.mov_u32_imm(PAGE_SIZE_VAL);
            let page_offset = ctx.mul_lo_u32(page_idx, page_size_val);
            let page_offset_64 = ctx.cvt_u64_u32(page_offset);
            let input_page_ptr = ctx.add_u64(input_ptr, page_offset_64);

            // Full load loop (4096 bytes = 1024 iterations)
            let zero_val = ctx.mov_u32_imm(0);
            ctx.st_shared_u32(warp_off, zero_val);

            ctx.label("L_load_loop");
            let idx = ctx.ld_shared_u32(warp_off);
            let load_done = ctx.setp_ge_u32(idx, page_size_val);
            ctx.branch_if(load_done, "L_load_done");

            let idx_64 = ctx.cvt_u64_u32(idx);
            let src_addr = ctx.add_u64(input_page_ptr, idx_64);
            let val = ctx.ld_global_u32(src_addr);
            let dst_off = ctx.add_u32_reg(warp_off, idx);
            ctx.st_shared_u32(dst_off, val);

            let four = ctx.mov_u32_imm(4);
            let idx_next = ctx.add_u32_reg(idx, four);
            ctx.st_shared_u32(warp_off, idx_next);
            ctx.branch("L_load_loop");

            ctx.label("L_load_done");
            ctx.emit_debug_marker(debug_ptr, 0x22222222); // LOAD DONE

            // Full hash init loop (8192 bytes = 2048 iterations)
            let hash_base_off = ctx.add_u32(warp_off, 4096);
            ctx.st_shared_u32(warp_off, zero_val);
            let invalid_marker = ctx.mov_u32_imm(0xFFFFFFFF);
            let hash_size = ctx.mov_u32_imm(HASH_TABLE_SIZE);

            ctx.label("L_hash_init");
            let h_idx = ctx.ld_shared_u32(warp_off);
            let init_done = ctx.setp_ge_u32(h_idx, hash_size);
            ctx.branch_if(init_done, "L_hash_done");

            let hash_off = ctx.add_u32_reg(hash_base_off, h_idx);
            ctx.st_shared_u32(hash_off, invalid_marker);

            let h_next = ctx.add_u32_reg(h_idx, four);
            ctx.st_shared_u32(warp_off, h_next);
            ctx.branch("L_hash_init");

            ctx.label("L_hash_done");
            ctx.emit_debug_marker(debug_ptr, 0x33333333); // HASH DONE

            // Initialize compression state
            let state_off = ctx.add_u32(warp_off, 12420); // STATE_OFFSET
            let out_pos_state_off = ctx.add_u32(state_off, 4);
            let anchor_state_off = ctx.add_u32(state_off, 8);

            ctx.st_shared_u32(state_off, zero_val); // in_pos = 0
            ctx.st_shared_u32(out_pos_state_off, zero_val); // out_pos = 0
            ctx.st_shared_u32(anchor_state_off, zero_val); // anchor = 0

            ctx.emit_debug_marker(debug_ptr, 0x44444444); // STATE INIT DONE

            // Simplified compress loop (just 5 iterations)
            let limit = ctx.mov_u32_imm(5);

            ctx.label("L_compress_loop");
            ctx.emit_debug_marker(debug_ptr, 0x55555555); // COMPRESS ITER

            let in_pos = ctx.ld_shared_u32(state_off);
            let at_limit = ctx.setp_ge_u32(in_pos, limit);
            ctx.branch_if(at_limit, "L_done");

            // Just increment in_pos
            let one = ctx.mov_u32_imm(1);
            let in_pos_next = ctx.add_u32_reg(in_pos, one);
            ctx.st_shared_u32(state_off, in_pos_next);
            ctx.branch("L_compress_loop");

            ctx.label("L_done");
            ctx.emit_debug_marker(debug_ptr, 0x66666666); // ALL DONE

            ctx.label("L_end");
            ctx.ret();
        });

    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel)
        .emit();

    println!("=== FullLoad PTX (first 50 lines) ===");
    for (i, line) in ptx.lines().take(50).enumerate() {
        println!("{:4}: {}", i + 1, line);
    }

    let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4096).unwrap();
    input_buf
        .copy_from_host(&(0..4096u32).map(|i| (i % 256) as u8).collect::<Vec<_>>())
        .unwrap();

    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 64]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (96, 1, 1),
        shared_mem: 0,
    };

    let batch_size: u32 = 1;
    let mut args: [*mut c_void; 3] = [
        input_buf.as_kernel_arg(),
        debug_buf.as_kernel_arg(),
        &batch_size as *const u32 as *mut c_void,
    ];

    println!("Launching fullload kernel...");
    unsafe {
        stream
            .launch_kernel(&mut module, "fullload", &config, &mut args)
            .expect("Launch");
    }

    let sync_result = stream.synchronize();

    let mut output = vec![0u32; 64];
    debug_buf.copy_to_host(&mut output).unwrap();

    println!("Counter: {}", output[0]);
    for i in 0..output[0].min(20) as usize {
        let m = output[i + 1];
        let name = match m {
            0x11111111 => "START",
            0x22222222 => "LOAD_DONE",
            0x33333333 => "HASH_DONE",
            0x44444444 => "STATE_INIT",
            0x55555555 => "COMPRESS_ITER",
            0x66666666 => "ALL_DONE",
            _ => "UNKNOWN",
        };
        println!("  [{:2}] 0x{:08X} ({})", i, m, name);
    }

    if let Err(e) = sync_result {
        panic!("FullLoad test crashed: {:?}", e);
    }

    // Expected: START, LOAD_DONE, HASH_DONE, STATE_INIT, 6x COMPRESS_ITER, ALL_DONE = 10 markers
    assert!(output[0] >= 9, "Should have at least 9 markers");
    println!("FullLoad test PASSED!");
}
