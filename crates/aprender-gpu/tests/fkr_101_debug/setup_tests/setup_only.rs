//! FKR-101-SETUP-ONLY: Test with exact same setup as failing kernel, but simple exit

use super::super::common::*;

/// FKR-101-SETUP-ONLY: Test with exact same setup as failing kernel, but simple exit
/// This tests if the crash is in setup or in compression loop
#[test]
fn fkr_101_setup_only_test() {
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

    let kernel = PtxKernel::new("setup_only")
        .param(PtxType::U64, "input_batch")
        .param(PtxType::U64, "output_batch")
        .param(PtxType::U64, "output_sizes")
        .param(PtxType::U64, "debug_buf")
        .param(PtxType::U32, "batch_size")
        .shared_memory(SMEM_SIZE)
        .build(|ctx| {
            // EXACT SAME SETUP as build_instrumented_compress_kernel()
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

            ctx.emit_debug_marker(debug_ptr, 0xAA000000); // BOUNDS OK

            // Use mul_u32 with immediate like failing kernel
            let warp_smem_size = ctx.mov_u32_imm(12544);
            let warp_off = ctx.mul_lo_u32(warp_id, warp_smem_size);

            ctx.emit_debug_marker(debug_ptr, 0xAA000001); // WARP_OFF

            // Load loop
            let page_size_val = ctx.mov_u32_imm(PAGE_SIZE_VAL);
            let page_offset = ctx.mul_lo_u32(page_idx, page_size_val);
            let page_offset_64 = ctx.cvt_u64_u32(page_offset);
            let input_page_ptr = ctx.add_u64(input_ptr, page_offset_64);

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
            ctx.emit_debug_marker(debug_ptr, 0xAA000002); // LOAD DONE

            // Hash init loop
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
            ctx.emit_debug_marker(debug_ptr, 0xAA000003); // HASH DONE

            // State init (like failing kernel)
            let state_off_val = ctx.mov_u32_imm(STATE_OFF);
            let state_off = ctx.add_u32_reg(warp_off, state_off_val);
            let four_imm = ctx.mov_u32_imm(4);
            let eight_imm = ctx.mov_u32_imm(8);
            let out_pos_state_off = ctx.add_u32_reg(state_off, four_imm);
            let anchor_state_off = ctx.add_u32_reg(state_off, eight_imm);

            ctx.st_shared_u32(state_off, zero_val);
            ctx.st_shared_u32(out_pos_state_off, zero_val);
            ctx.st_shared_u32(anchor_state_off, zero_val);

            // Output setup (like failing kernel)
            let output_size = ctx.mov_u32_imm(4352);
            let output_offset = ctx.mul_lo_u32(page_idx, output_size);
            let output_offset_64 = ctx.cvt_u64_u32(output_offset);
            let _output_page_ptr = ctx.add_u64(output_ptr, output_offset_64);

            // Initialize all the constants used in compress loop
            let limit = ctx.mov_u32_imm(PAGE_SIZE_VAL - 12);
            let lz4_prime = ctx.mov_u32_imm(0x9E3779B1);
            let hash_shift = ctx.mov_u32_imm(21);
            let hash_mask = ctx.mov_u32_imm(0x7FF);

            // Just write final output and exit (no compress loop)
            let final_pos = ctx.mov_u32_imm(0);
            let page_idx_64 = ctx.cvt_u64_u32(page_idx);
            let four_64 = ctx.mov_u64_imm(4);
            let size_offset = ctx.mul_u64_reg(page_idx_64, four_64);
            let size_addr = ctx.add_u64(sizes_ptr, size_offset);
            ctx.st_global_u32(size_addr, final_pos);

            ctx.emit_debug_marker(debug_ptr, 0xAA000004); // ALL DONE

            // Suppress unused warnings by referencing
            let _suppress = limit;
            let _suppress2 = lz4_prime;
            let _suppress3 = hash_shift;
            let _suppress4 = hash_mask;
            let _suppress5 = invalid_marker;

            ctx.label("L_end");
            ctx.ret();
        });

    let ptx =
        PtxModule::new().version(8, 0).target("sm_89").address_size(64).add_kernel(kernel).emit();

    println!("=== Setup-Only PTX ===");
    for (i, line) in ptx.lines().take(50).enumerate() {
        println!("{:4}: {}", i + 1, line);
    }

    let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4096).unwrap();
    input_buf.copy_from_host(&(0..4096u32).map(|i| (i % 256) as u8).collect::<Vec<_>>()).unwrap();

    let mut output_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4352).unwrap();
    let mut sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 64).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 64]).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig { grid: (1, 1, 1), block: (96, 1, 1), shared_mem: 0 };

    let batch_size: u32 = 1;
    let mut args: [*mut c_void; 5] = [
        input_buf.as_kernel_arg(),
        output_buf.as_kernel_arg(),
        sizes_buf.as_kernel_arg(),
        debug_buf.as_kernel_arg(),
        &batch_size as *const u32 as *mut c_void,
    ];

    println!("Launching setup-only kernel...");
    unsafe {
        stream.launch_kernel(&mut module, "setup_only", &config, &mut args).expect("Launch");
    }

    let sync_result = stream.synchronize();

    let mut output = vec![0u32; 64];
    debug_buf.copy_to_host(&mut output).unwrap();

    println!("Counter: {}", output[0]);
    for i in 0..output[0].min(10) as usize {
        let m = output[i + 1];
        let name = match m {
            0xAA000000 => "BOUNDS_OK",
            0xAA000001 => "WARP_OFF",
            0xAA000002 => "LOAD_DONE",
            0xAA000003 => "HASH_DONE",
            0xAA000004 => "ALL_DONE",
            _ => "UNKNOWN",
        };
        println!("  [{:2}] 0x{:08X} ({})", i, m, name);
    }

    if let Err(e) = sync_result {
        panic!("Setup-only test crashed: {:?}", e);
    }

    assert_eq!(output[0], 5, "Should have 5 markers");
    println!("Setup-only test PASSED!");
}
