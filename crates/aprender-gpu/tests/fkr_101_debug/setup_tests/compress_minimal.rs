//! FKR-101-COMPRESS-MINIMAL: Setup + minimal compression loop (no match logic)

use super::super::common::*;

/// FKR-101-COMPRESS-MINIMAL: Setup + minimal compression loop (no match logic)
/// NOTE: This test exercises the LVB-003/F082 bug pattern - it intentionally
/// triggers a kernel crash when the compression loop runs.
#[test]
#[ignore = "LVB-003/F082: Compression loop triggers CUDA_ERROR_INVALID_ADDRESS_SPACE"]
fn fkr_101_compress_minimal_test() {
    if !cuda_available() {
        eprintln!("SKIPPED: No CUDA");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    const SMEM_SIZE: usize = 12544 * 3;
    const PAGE_SIZE_VAL: u32 = 4096;
    const HASH_TABLE_SIZE: u32 = 8192;
    const STATE_OFF: u32 = PAGE_SIZE_VAL + 8192 + 128 + 4;
    const HASH_TABLE_OFF: u32 = PAGE_SIZE_VAL;

    let kernel = PtxKernel::new("compress_minimal")
        .param(PtxType::U64, "input_batch")
        .param(PtxType::U64, "output_batch")
        .param(PtxType::U64, "output_sizes")
        .param(PtxType::U64, "debug_buf")
        .param(PtxType::U32, "batch_size")
        .shared_memory(SMEM_SIZE)
        .build(|ctx| {
            // Setup (same as setup_only)
            let input_ptr = ctx.load_param_u64("input_batch");
            let output_ptr = ctx.load_param_u64("output_batch");
            let sizes_ptr = ctx.load_param_u64("output_sizes");
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

            // Load loop
            let page_size_val = ctx.mov_u32_imm(PAGE_SIZE_VAL);
            let page_offset = ctx.mul_lo_u32(page_idx, page_size_val);
            let page_offset_64 = ctx.cvt_u64_u32(page_offset);
            let input_page_ptr = ctx.add_u64(input_ptr, page_offset_64);

            let zero_val = ctx.mov_u32_imm(0);
            let four = ctx.mov_u32_imm(4);
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

            let idx_next = ctx.add_u32_reg(idx, four);
            ctx.st_shared_u32(warp_off, idx_next);
            ctx.branch("L_load_loop");

            ctx.label("L_load_done");
            ctx.emit_debug_marker(debug_ptr, 0xAA000001);

            // Hash init
            let hash_base_off_val = ctx.mov_u32_imm(HASH_TABLE_OFF);
            let hash_base_off = ctx.add_u32_reg(warp_off, hash_base_off_val);
            ctx.st_shared_u32(warp_off, zero_val);
            let invalid_marker = ctx.mov_u32_imm(0xFFFFFFFF);
            let hash_table_size = ctx.mov_u32_imm(HASH_TABLE_SIZE);

            ctx.label("L_hash_init");
            let h_idx = ctx.ld_shared_u32(warp_off);
            let init_done = ctx.setp_ge_u32(h_idx, hash_table_size);
            ctx.branch_if(init_done, "L_hash_done");

            let hash_off = ctx.add_u32_reg(hash_base_off, h_idx);
            ctx.st_shared_u32(hash_off, invalid_marker);

            let h_next = ctx.add_u32_reg(h_idx, four);
            ctx.st_shared_u32(warp_off, h_next);
            ctx.branch("L_hash_init");

            ctx.label("L_hash_done");
            ctx.emit_debug_marker(debug_ptr, 0xAA000002);

            // State init
            let state_off_val = ctx.mov_u32_imm(STATE_OFF);
            let state_off = ctx.add_u32_reg(warp_off, state_off_val);
            let four_imm = ctx.mov_u32_imm(4);
            let eight_imm = ctx.mov_u32_imm(8);
            let out_pos_state_off = ctx.add_u32_reg(state_off, four_imm);
            let anchor_state_off = ctx.add_u32_reg(state_off, eight_imm);

            ctx.st_shared_u32(state_off, zero_val);
            ctx.st_shared_u32(out_pos_state_off, zero_val);
            ctx.st_shared_u32(anchor_state_off, zero_val);

            // Compression constants
            let limit = ctx.mov_u32_imm(PAGE_SIZE_VAL - 12);
            let lz4_prime = ctx.mov_u32_imm(0x9E3779B1);
            let hash_shift_val = ctx.mov_u32_imm(21);
            let hash_mask = ctx.mov_u32_imm(0x7FF);
            let max_iters = ctx.mov_u32_imm(10);
            let twelve_imm = ctx.mov_u32_imm(12);
            let iter_counter_off = ctx.add_u32_reg(state_off, twelve_imm);
            ctx.st_shared_u32(iter_counter_off, zero_val);

            ctx.emit_debug_marker(debug_ptr, 0xAA000003);

            // MINIMAL compression loop - NO hash at all, just iterate
            ctx.label("L_compress_loop");
            ctx.emit_debug_marker(debug_ptr, 0xBB000000);

            // Check iteration limit
            let iters = ctx.ld_shared_u32(iter_counter_off);
            let too_many = ctx.setp_ge_u32(iters, max_iters);
            ctx.branch_if(too_many, "L_done");
            let one = ctx.mov_u32_imm(1);
            let iters_next = ctx.add_u32_reg(iters, one);
            ctx.membar_cta(); // Before storing value computed from loaded
            ctx.st_shared_u32(iter_counter_off, iters_next);

            // Load state (just in_pos for now)
            let in_pos = ctx.ld_shared_u32(state_off);

            // Bounds check
            let at_limit = ctx.setp_ge_u32(in_pos, limit);
            ctx.branch_if(at_limit, "L_done");

            // TEST: Use in_pos (loaded value) in address computation

            // Compute address using in_pos (which is loaded from shared memory)
            let computed_addr = ctx.add_u32_reg(warp_off, in_pos); // warp_off + in_pos

            // LVB-003: membar.cta BEFORE store to address computed from loaded value
            ctx.membar_cta();

            // Write constant to computed address
            let test_constant = ctx.mov_u32_imm(0xCAFEBABE);
            ctx.st_shared_u32(computed_addr, test_constant);

            // Load from computed address (this works fine)
            let _loaded_val = ctx.ld_shared_u32(computed_addr);

            // Just emit a constant marker
            ctx.emit_debug_marker(debug_ptr, 0xDD000000);

            // Suppress unused
            let _ = hash_mask;
            let _ = hash_base_off;

            ctx.emit_debug_marker(debug_ptr, 0xDD000001);

            // Suppress unused
            let _ = lz4_prime;
            let _ = hash_shift_val;

            // Just advance position
            let in_pos_next = ctx.add_u32_reg(in_pos, one);
            ctx.membar_cta(); // Before storing value computed from loaded
            ctx.st_shared_u32(state_off, in_pos_next);
            ctx.branch("L_compress_loop");

            ctx.label("L_done");
            ctx.emit_debug_marker(debug_ptr, 0xCC000000);

            // Store output size
            let final_out_pos = ctx.ld_shared_u32(out_pos_state_off);
            let page_idx_64 = ctx.cvt_u64_u32(page_idx);
            let four_64 = ctx.mov_u64_imm(4);
            let size_offset = ctx.mul_u64_reg(page_idx_64, four_64);
            let size_addr = ctx.add_u64(sizes_ptr, size_offset);
            ctx.st_global_u32(size_addr, final_out_pos);

            // Suppress unused warnings
            let _ = output_ptr;
            let _ = invalid_marker;

            ctx.label("L_end");
            ctx.ret();
        });

    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel)
        .emit();

    println!("=== Compress-Minimal PTX ===");
    for (i, line) in ptx.lines().enumerate() {
        println!("{:4}: {}", i + 1, line);
    }

    let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4096).unwrap();
    input_buf
        .copy_from_host(&(0..4096u32).map(|i| (i % 256) as u8).collect::<Vec<_>>())
        .unwrap();

    let mut output_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4352).unwrap();
    let mut sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 64]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (96, 1, 1),
        shared_mem: 0,
    };

    let batch_size: u32 = 1;
    let mut args: [*mut c_void; 5] = [
        input_buf.as_kernel_arg(),
        output_buf.as_kernel_arg(),
        sizes_buf.as_kernel_arg(),
        debug_buf.as_kernel_arg(),
        &batch_size as *const u32 as *mut c_void,
    ];

    println!("Launching compress-minimal kernel...");
    unsafe {
        stream
            .launch_kernel(&mut module, "compress_minimal", &config, &mut args)
            .expect("Launch");
    }

    let sync_result = stream.synchronize();

    // Check sync_result FIRST before copy_to_host
    if let Err(e) = &sync_result {
        eprintln!("Kernel sync error: {:?}", e);
    }

    let mut output = vec![0u32; 64];
    if let Err(e) = debug_buf.copy_to_host(&mut output) {
        eprintln!("Copy to host failed: {:?}", e);
        if sync_result.is_err() {
            panic!(
                "Compress-minimal test crashed during kernel: {:?}",
                sync_result.unwrap_err()
            );
        }
        panic!("Compress-minimal test crashed during copy: {:?}", e);
    }

    println!("Counter: {}", output[0]);
    for i in 0..output[0].min(20) as usize {
        let m = output[i + 1];
        let name = match m {
            0xAA000000 => "BOUNDS_OK",
            0xAA000001 => "LOAD_DONE",
            0xAA000002 => "HASH_DONE",
            0xAA000003 => "COMPRESS_START",
            0xBB000000 => "LOOP_ITER",
            0xCC000000 => "ALL_DONE",
            _ => "UNKNOWN",
        };
        println!("  [{:2}] 0x{:08X} ({})", i, m, name);
    }

    if let Err(e) = sync_result {
        panic!("Compress-minimal test crashed: {:?}", e);
    }

    // Expected: BOUNDS_OK, LOAD_DONE, HASH_DONE, COMPRESS_START, 10x LOOP_ITER, ALL_DONE = 14 markers
    assert!(output[0] >= 14, "Should have at least 14 markers");
    println!("Compress-minimal test PASSED!");
}
