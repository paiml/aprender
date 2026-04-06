//! Instrumented compression kernel builder and debug crash test (FKR-101).

use super::super::common::*;

/// Build a minimal instrumented compression loop kernel
fn build_instrumented_compress_kernel() -> String {
    let kernel = PtxKernel::new("instrumented_compress")
        .param(PtxType::U64, "input_batch")
        .param(PtxType::U64, "output_batch")
        .param(PtxType::U64, "output_sizes")
        .param(PtxType::U64, "debug_buf")
        .param(PtxType::U32, "batch_size")
        .shared_memory((WARP_SMEM_SIZE * 3) as usize) // 3 warps
        .build(|ctx| {
            // Load parameters
            let input_ptr = ctx.load_param_u64("input_batch");
            let output_ptr = ctx.load_param_u64("output_batch");
            let sizes_ptr = ctx.load_param_u64("output_sizes");
            let debug_ptr = ctx.load_param_u64("debug_buf");
            let batch_size = ctx.load_param_u32("batch_size");

            // Thread identification
            let tid = ctx.special_reg(PtxReg::TidX);
            let bid = ctx.special_reg(PtxReg::CtaIdX);
            let warp_id = ctx.shr_u32_imm(tid, 5);
            let lane_mask = ctx.mov_u32_imm(31);
            let lane_id = ctx.and_u32(tid, lane_mask);

            // Only lane 0 of each warp processes
            let zero_check = ctx.mov_u32_imm(0);
            let is_leader = ctx.setp_eq_u32(lane_id, zero_check);
            ctx.branch_if_not(is_leader, "L_not_leader");

            // Compute page index = bid * warps_per_block + warp_id
            let warps_per_block = ctx.mov_u32_imm(3);
            let block_offset = ctx.mul_lo_u32(bid, warps_per_block);
            let page_idx = ctx.add_u32_reg(block_offset, warp_id);

            // Bounds check
            let out_of_bounds = ctx.setp_ge_u32(page_idx, batch_size);
            ctx.branch_if(out_of_bounds, "L_not_leader");

            // DEBUG: After bounds check passed - VERY FIRST marker
            ctx.emit_debug_marker(debug_ptr, 0xAA000000);

            // Compute warp's shared memory offset
            let warp_off = ctx.mul_u32(warp_id, WARP_SMEM_SIZE);

            // DEBUG: After warp_off computed
            ctx.emit_debug_marker(debug_ptr, 0xAA000001);

            // Load page data from global to shared memory
            let page_size_val = ctx.mov_u32_imm(PAGE_SIZE);
            let page_offset = ctx.mul_lo_u32(page_idx, page_size_val);
            let page_offset_64 = ctx.cvt_u64_u32(page_offset);
            let input_page_ptr = ctx.add_u64(input_ptr, page_offset_64);

            // Simple sequential load (leader thread loads all)
            let load_idx = ctx.mov_u32_imm(0);
            ctx.st_shared_u32(warp_off, load_idx);

            ctx.label("L_load_loop");
            // REMOVED: In-loop marker was causing buffer overflow (1024 iterations)
            let idx = ctx.ld_shared_u32(warp_off);
            let load_done = ctx.setp_ge_u32(idx, page_size_val);
            ctx.branch_if(load_done, "L_load_done");

            let idx_64 = ctx.cvt_u64_u32(idx);
            let src_addr = ctx.add_u64(input_page_ptr, idx_64);
            let val = ctx.ld_global_u32(src_addr);
            let dst_off = ctx.add_u32_reg(warp_off, idx);
            ctx.st_shared_u32(dst_off, val);

            let idx_next = ctx.add_u32(idx, 4);
            ctx.st_shared_u32(warp_off, idx_next);
            ctx.branch("L_load_loop");

            ctx.label("L_load_done");

            // DEBUG: After data load complete
            ctx.emit_debug_marker(debug_ptr, 0xAA000002);

            // Initialize hash table (set all entries to 0xFFFFFFFF)
            let hash_base_off = ctx.add_u32(warp_off, HASH_TABLE_OFFSET);
            let init_idx = ctx.mov_u32_imm(0);
            ctx.st_shared_u32(warp_off, init_idx);
            let invalid_marker = ctx.mov_u32_imm(0xFFFFFFFF);
            let hash_table_size = ctx.mov_u32_imm(8192);

            ctx.label("L_hash_init_loop");
            let h_idx = ctx.ld_shared_u32(warp_off);
            let init_done = ctx.setp_ge_u32(h_idx, hash_table_size);
            ctx.branch_if(init_done, "L_hash_init_done");

            let hash_off = ctx.add_u32_reg(hash_base_off, h_idx);
            ctx.st_shared_u32(hash_off, invalid_marker);

            let h_idx_next = ctx.add_u32(h_idx, 4);
            ctx.st_shared_u32(warp_off, h_idx_next);
            ctx.branch("L_hash_init_loop");

            ctx.label("L_hash_init_done");

            // DEBUG: After hash init complete
            ctx.emit_debug_marker(debug_ptr, 0xAA000003);

            // Initialize compression state
            let state_off = ctx.add_u32(warp_off, STATE_OFFSET);
            let out_pos_state_off = ctx.add_u32(state_off, 4);
            let anchor_state_off = ctx.add_u32(state_off, 8);

            let zero_val = ctx.mov_u32_imm(0);
            ctx.st_shared_u32(state_off, zero_val); // in_pos = 0
            ctx.st_shared_u32(out_pos_state_off, zero_val); // out_pos = 0
            ctx.st_shared_u32(anchor_state_off, zero_val); // anchor = 0

            // Output page pointer
            let output_size = ctx.mov_u32_imm(4352);
            let output_offset = ctx.mul_lo_u32(page_idx, output_size);
            let output_offset_64 = ctx.cvt_u64_u32(output_offset);
            let output_page_ptr = ctx.add_u64(output_ptr, output_offset_64);

            let limit = ctx.mov_u32_imm(PAGE_SIZE - 12);
            let lz4_prime = ctx.mov_u32_imm(LZ4_PRIME);
            let hash_shift = ctx.mov_u32_imm(21);
            let hash_mask = ctx.mov_u32_imm(0x7FF);

            // Limit iterations for debugging
            let max_iters = ctx.mov_u32_imm(100);
            let iter_counter_off = ctx.add_u32(state_off, 12);
            ctx.st_shared_u32(iter_counter_off, zero_val);

            // ============================================================
            // Main compression loop with debug markers
            // ============================================================
            ctx.label("L_compress_loop");

            // DEBUG: Mark loop entry (include page_idx for identification)
            let marker = ctx.mov_u32_imm(MARKER_LOOP_ENTRY);
            let marker_with_page = ctx.or_u32(marker, page_idx);
            ctx.emit_debug_marker(debug_ptr, MARKER_LOOP_ENTRY);

            // Check iteration limit
            let iters = ctx.ld_shared_u32(iter_counter_off);
            let too_many_iters = ctx.setp_ge_u32(iters, max_iters);
            ctx.branch_if(too_many_iters, "L_emit_remaining");
            let iters_next = ctx.add_u32(iters, 1);
            ctx.st_shared_u32(iter_counter_off, iters_next);

            // Recompute warp_off fresh each iteration
            let fresh_tid = ctx.special_reg(PtxReg::TidX);
            let fresh_warp_id = ctx.shr_u32_imm(fresh_tid, 5);
            let fresh_warp_off = ctx.mul_u32(fresh_warp_id, WARP_SMEM_SIZE);

            let fresh_state_off = ctx.add_u32(fresh_warp_off, STATE_OFFSET);
            let fresh_out_pos_off = ctx.add_u32(fresh_state_off, 4);
            let fresh_anchor_off = ctx.add_u32(fresh_state_off, 8);
            let fresh_hash_base_off = ctx.add_u32(fresh_warp_off, HASH_TABLE_OFFSET);

            // Load current state
            let in_pos = ctx.ld_shared_u32(fresh_state_off);
            let out_pos = ctx.ld_shared_u32(fresh_out_pos_off);
            let anchor = ctx.ld_shared_u32(fresh_anchor_off);

            // DEBUG: Mark state loaded
            ctx.emit_debug_marker(debug_ptr, MARKER_STATE_LOADED);

            // Check bounds
            let at_limit = ctx.setp_ge_u32(in_pos, limit);
            ctx.branch_if(at_limit, "L_emit_remaining");

            // Load 4 bytes at current position
            let page_data_off = ctx.add_u32_reg(fresh_warp_off, in_pos);
            let curr_val = ctx.ld_shared_u32(page_data_off);

            // Compute hash
            let hash_tmp = ctx.mul_lo_u32(curr_val, lz4_prime);
            let hash_shifted = ctx.shr_u32(hash_tmp, hash_shift);
            let hash_idx = ctx.and_u32(hash_shifted, hash_mask);

            // DEBUG: Mark hash computed
            ctx.emit_debug_marker(debug_ptr, MARKER_HASH_COMPUTED);

            // Look up hash table
            let hash_entry_off = ctx.mul_u32(hash_idx, 4);
            let hash_addr_off = ctx.add_u32_reg(fresh_hash_base_off, hash_entry_off);
            let match_pos = ctx.ld_shared_u32(hash_addr_off);
            ctx.st_shared_u32(hash_addr_off, in_pos);

            // DEBUG: Mark hash lookup done
            ctx.emit_debug_marker(debug_ptr, MARKER_HASH_LOOKUP);

            // Check validity
            let no_match_candidate = ctx.setp_eq_u32(match_pos, invalid_marker);
            ctx.branch_if(no_match_candidate, "L_no_match");

            let offset = ctx.sub_u32_reg(in_pos, match_pos);
            let max_offset = ctx.mov_u32_imm(LZ4_MAX_OFFSET + 1);
            let offset_too_large = ctx.setp_ge_u32(offset, max_offset);
            ctx.branch_if(offset_too_large, "L_no_match");

            let offset_is_zero = ctx.setp_eq_u32(offset, zero_val);
            ctx.branch_if(offset_is_zero, "L_no_match");

            // Bounds check match_pos
            let max_match_pos = ctx.mov_u32_imm(PAGE_SIZE - 3);
            let match_oob = ctx.setp_ge_u32(match_pos, max_match_pos);
            ctx.branch_if(match_oob, "L_no_match");

            // Load and compare
            let match_data_off = ctx.add_u32_reg(fresh_warp_off, match_pos);
            let match_val = ctx.ld_shared_u32(match_data_off);

            let vals_equal = ctx.setp_eq_u32(curr_val, match_val);
            ctx.branch_if_not(vals_equal, "L_no_match");

            // ============================================================
            // Found a match!
            // ============================================================
            ctx.label("L_found_match");
            ctx.emit_debug_marker(debug_ptr, MARKER_FOUND_MATCH);

            // Calculate literal length
            let literal_len = ctx.sub_u32_reg(in_pos, anchor);

            // Build token
            let fifteen = ctx.mov_u32_imm(15);
            let lit_ge_15 = ctx.setp_ge_u32(literal_len, fifteen);
            let token_lit = ctx.selp_u32(lit_ge_15, fifteen, literal_len);
            let four_bits = ctx.mov_u32_imm(4);
            let token = ctx.shl_u32(token_lit, four_bits);

            // Write token
            let out_pos_64 = ctx.cvt_u64_u32(out_pos);
            let out_addr = ctx.add_u64(output_page_ptr, out_pos_64);
            ctx.st_global_u8(out_addr, token);
            let out_pos_1 = ctx.add_u32(out_pos, 1);

            // Handle extended literal length
            ctx.branch_if_not(lit_ge_15, "L_skip_ext_lit");

            let lit_minus_15 = ctx.sub_u32_reg(literal_len, fifteen);
            let out_pos_1_64 = ctx.cvt_u64_u32(out_pos_1);
            let ext_addr = ctx.add_u64(output_page_ptr, out_pos_1_64);
            ctx.st_global_u8(ext_addr, lit_minus_15);
            let out_pos_2 = ctx.add_u32(out_pos_1, 1);
            ctx.st_shared_u32(fresh_out_pos_off, out_pos_2);
            ctx.branch("L_copy_literals");

            ctx.label("L_skip_ext_lit");
            ctx.st_shared_u32(fresh_out_pos_off, out_pos_1);

            // Copy literals
            ctx.label("L_copy_literals");
            ctx.emit_debug_marker(debug_ptr, MARKER_COPY_LIT);

            let out_pos_cur = ctx.ld_shared_u32(fresh_out_pos_off);
            let copy_idx = ctx.mov_u32_imm(0);
            ctx.st_shared_u32(fresh_state_off, copy_idx);

            ctx.label("L_copy_lit_loop");
            let idx2 = ctx.ld_shared_u32(fresh_state_off);
            let copy_done = ctx.setp_ge_u32(idx2, literal_len);
            ctx.branch_if(copy_done, "L_copy_lit_done");

            // Load byte (via u32 + extract)
            let src_off = ctx.add_u32_reg(anchor, idx2);
            let src_smem_off = ctx.add_u32_reg(fresh_warp_off, src_off);
            let align_mask = ctx.mov_u32_imm(!3u32);
            let aligned_off = ctx.and_u32(src_smem_off, align_mask);
            let byte_mask = ctx.mov_u32_imm(3);
            let byte_idx = ctx.and_u32(src_smem_off, byte_mask);
            let word_val = ctx.ld_shared_u32(aligned_off);
            let shift_amt = ctx.mul_u32(byte_idx, 8);
            let shifted = ctx.shr_u32(word_val, shift_amt);
            let byte_mask_ff = ctx.mov_u32_imm(0xFF);
            let byte_val = ctx.and_u32(shifted, byte_mask_ff);

            // Store byte
            let dst_off = ctx.add_u32_reg(out_pos_cur, idx2);
            let dst_off_64 = ctx.cvt_u64_u32(dst_off);
            let dst_addr = ctx.add_u64(output_page_ptr, dst_off_64);
            ctx.st_global_u8(dst_addr, byte_val);

            let idx2_next = ctx.add_u32(idx2, 1);
            ctx.st_shared_u32(fresh_state_off, idx2_next);
            ctx.branch("L_copy_lit_loop");

            ctx.label("L_copy_lit_done");

            // Write match offset
            let out_pos_after_lit = ctx.add_u32_reg(out_pos_cur, literal_len);
            let out_pos_after_lit_64 = ctx.cvt_u64_u32(out_pos_after_lit);
            let offset_addr = ctx.add_u64(output_page_ptr, out_pos_after_lit_64);
            let mask_ff = ctx.mov_u32_imm(0xFF);
            let offset_lo = ctx.and_u32(offset, mask_ff);
            ctx.st_global_u8(offset_addr, offset_lo);

            let one_64 = ctx.mov_u64_imm(1);
            let offset_addr_1 = ctx.add_u64(offset_addr, one_64);
            let eight = ctx.mov_u32_imm(8);
            let offset_hi = ctx.shr_u32(offset, eight);
            ctx.st_global_u8(offset_addr_1, offset_hi);

            let out_pos_after_offset = ctx.add_u32(out_pos_after_lit, 2);

            // Update state
            let new_anchor = ctx.add_u32(in_pos, LZ4_MIN_MATCH);
            ctx.st_shared_u32(fresh_anchor_off, new_anchor);
            ctx.st_shared_u32(fresh_state_off, new_anchor);
            ctx.st_shared_u32(fresh_out_pos_off, out_pos_after_offset);

            ctx.branch("L_compress_loop");

            // ============================================================
            // No match - advance position
            // ============================================================
            ctx.label("L_no_match");
            ctx.emit_debug_marker(debug_ptr, MARKER_NO_MATCH);

            let in_pos_next = ctx.add_u32(in_pos, 1);
            ctx.st_shared_u32(fresh_state_off, in_pos_next);
            ctx.branch("L_compress_loop");

            // ============================================================
            // Emit remaining literals
            // ============================================================
            ctx.label("L_emit_remaining");
            ctx.emit_debug_marker(debug_ptr, MARKER_EMIT_REMAIN);

            // For this debug test, just store output size and exit
            let final_out_pos = ctx.ld_shared_u32(fresh_out_pos_off);

            // Store output size
            let page_idx_64 = ctx.cvt_u64_u32(page_idx);
            let four_64 = ctx.mov_u64_imm(4);
            let size_offset = ctx.mul_u64_reg(page_idx_64, four_64);
            let size_addr = ctx.add_u64(sizes_ptr, size_offset);
            ctx.st_global_u32(size_addr, final_out_pos);

            ctx.label("L_not_leader");
            ctx.ret();
        });

    PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel)
        .emit()
}

/// FKR-101-DEBUG: Run instrumented compression with debug buffer
/// NOTE: This test uses the buggy Lz4WarpCompressKernel which has F082 bug
#[test]
#[ignore = "Uses buggy Lz4WarpCompressKernel - F082 confirmed"]
fn fkr_101_debug_find_crash() {
    if !cuda_available() {
        eprintln!("SKIPPED: No CUDA");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    const NUM_PAGES: u32 = 1; // Single page for easier debugging

    // Create non-zero sequential data
    let mut input: Vec<u8> = Vec::with_capacity((NUM_PAGES * PAGE_SIZE) as usize);
    for page_idx in 0..NUM_PAGES {
        for byte_idx in 0..PAGE_SIZE {
            input.push(((page_idx * 17 + byte_idx) % 256) as u8);
        }
    }

    // Allocate GPU buffers
    let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, input.len()).unwrap();
    let mut output_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, (NUM_PAGES * 4352) as usize).unwrap();
    let mut sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, NUM_PAGES as usize).unwrap();
    let mut debug_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1024).unwrap();

    input_buf.copy_from_host(&input).unwrap();
    debug_buf.copy_from_host(&vec![0u32; 1024]).unwrap();

    // Build and load instrumented kernel
    let ptx = build_instrumented_compress_kernel();
    println!("=== Instrumented PTX (first 200 lines) ===");
    for (i, line) in ptx.lines().take(200).enumerate() {
        println!("{:4}: {}", i + 1, line);
    }

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (96, 1, 1), // 3 warps
        shared_mem: 0,
    };

    let num_pages = NUM_PAGES;
    let mut args: [*mut c_void; 5] = [
        input_buf.as_kernel_arg(),
        output_buf.as_kernel_arg(),
        sizes_buf.as_kernel_arg(),
        debug_buf.as_kernel_arg(),
        &num_pages as *const u32 as *mut c_void,
    ];

    println!("\nLaunching instrumented kernel...");

    unsafe {
        stream
            .launch_kernel(&mut module, "instrumented_compress", &config, &mut args)
            .expect("Kernel launch");
    }

    // Try to sync - this is where crash happens
    let sync_result = stream.synchronize();

    // Read debug buffer regardless of crash
    let mut debug_output = vec![0u32; 1024];
    let _ = debug_buf.copy_to_host(&mut debug_output);

    println!("\n=== Debug Buffer Contents ===");
    println!("Counter (markers written): {}", debug_output[0]);

    let marker_names = |m: u32| match m {
        0xAA000000 => "BOUNDS_CHECK_PASSED",
        0xAA000001 => "WARP_OFF_COMPUTED",
        0xAA000002 => "DATA_LOAD_DONE",
        0xAA000003 => "HASH_INIT_DONE",
        0xBB000000 => "LOAD_ITER",
        _ => match m & 0xFF000000 {
            0x01000000 => "LOOP_ENTRY",
            0x02000000 => "STATE_LOADED",
            0x03000000 => "HASH_COMPUTED",
            0x04000000 => "HASH_LOOKUP",
            0x05000000 => "FOUND_MATCH",
            0x06000000 => "COPY_LIT",
            0x07000000 => "NO_MATCH",
            0x08000000 => "EMIT_REMAIN",
            _ => "UNKNOWN",
        },
    };

    let count = (debug_output[0] as usize).min(100);
    for i in 0..count {
        let marker = debug_output[i + 1];
        println!("  [{:3}] 0x{:08X} ({})", i, marker, marker_names(marker));
    }

    // Check sync result
    if let Err(e) = sync_result {
        println!("\n!!! SYNC FAILED: {:?}", e);
        println!(
            "Last marker before crash: 0x{:08X} ({})",
            debug_output[count],
            marker_names(debug_output[count])
        );

        // This test is expected to crash - we want to see where
        panic!(
            "Crash detected after {} debug markers. Last: {}",
            count,
            marker_names(debug_output[count])
        );
    }

    println!("\nKernel completed successfully!");

    let mut sizes = vec![0u32; NUM_PAGES as usize];
    sizes_buf.copy_to_host(&mut sizes).unwrap();
    println!("Output sizes: {:?}", sizes);
}
