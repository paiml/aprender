//! CTA-level WMMA GEMM — 4 warps cooperatively compute 32×32 output tiles
//!
//! Architecture: 2×2 warp grid, each warp owns a 16×16 WMMA subtile.
//! All 4 warps share 32×K A tile and K×32 B tile in shared memory.
//! K-dimension tiled in chunks of 16 (WMMA tile depth).
//!
//! Performance optimizations:
//! - PERF-CTA-001: Correct row.row MMA layout for RowMajor A and B
//! - PERF-CTA-002: Fully unrolled cooperative load (no inner loop branches)
//! - PERF-CTA-003: Warp-uniform branching (warps 0-1→A, warps 2-3→B)
//! - PERF-CTA-004: Interior tile fast path — skip boundary checks for tiles
//!   fully within matrix bounds (eliminates 16 branches/thread/K-tile)
//! - PERF-CTA-005: No .maxnreg — let JIT optimize register allocation
//! - PERF-CTA-007: Double-buffered shared memory — overlap load of next K-tile
//!   with WMMA compute of current tile (hides ~400 cycle global memory latency)
//!
//! Launch: grid_2d((N+31)/32, (M+31)/32), block(128,1,1), shared_mem=2KB (4KB dbuf)
//! Input: FP16 (u16), Output: FP32

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

/// Build a CTA-tiled WMMA GEMM kernel for FP16 input → FP32 output.
///
/// When `double_buffer` is true, allocates 2× shared memory and overlaps
/// load of next K-tile with WMMA compute of current tile [25][48].
/// Expected 1.3-1.5× speedup by hiding ~400 cycle global memory latency.
/// Contract: cgp-gpu-gemm-dbuf-v1.yaml
pub fn build_cta_wmma_fp16(_m: u32, n: u32, k: u32) -> PtxKernel {
    build_cta_wmma_fp16_impl(_m, n, k, false)
}

/// Double-buffered variant — overlaps global load with WMMA compute.
pub fn build_cta_wmma_fp16_dbuf(_m: u32, n: u32, k: u32) -> PtxKernel {
    build_cta_wmma_fp16_impl(_m, n, k, true)
}

fn build_cta_wmma_fp16_impl(_m: u32, n: u32, k: u32, double_buffer: bool) -> PtxKernel {
    let smem_single = 32 * 16 * 2 + 16 * 32 * 2; // A[32×16] + B[16×32] in FP16
    let smem_bytes = if double_buffer { smem_single * 2 } else { smem_single };
    let n_k_tiles = (k + 15) / 16;

    PtxKernel::new("gemm_cta_wmma_fp16")
        // PERF-CTA-005: No .maxnreg — let ptxas choose optimal register allocation
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .param(PtxType::U32, "m_param")
        .param(PtxType::U32, "n_param")
        .param(PtxType::U32, "k_param")
        .shared_memory(smem_bytes)
        .build(move |ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
            let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

            // Constants
            let c_1 = ctx.mov_u32_imm(1);
            let c_2 = ctx.mov_u32_imm(2);
            let c_4 = ctx.mov_u32_imm(4);
            let c_5 = ctx.mov_u32_imm(5);
            let c_15 = ctx.mov_u32_imm(15);
            let c_16 = ctx.mov_u32_imm(16);
            let c_31 = ctx.mov_u32_imm(31);
            let c_32 = ctx.mov_u32_imm(32);
            let c_512 = ctx.mov_u32_imm(512);
            let c_1024 = ctx.mov_u32_imm(1024);
            let k_const = ctx.mov_u32_imm(k);
            let n_const = ctx.mov_u32_imm(n);
            let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);
            let zero_f32 = ctx.mov_f32_imm(0.0);
            let zero_f16 = ctx.cvt_f16_f32(zero_f32);

            // CTA tile position
            let cta_row = ctx.mul_u32_reg(ctaid_y, c_32);
            let cta_col = ctx.mul_u32_reg(ctaid_x, c_32);

            let m_param = ctx.load_param_u32("m_param");
            let n_param = ctx.load_param_u32("n_param");
            let k_param = ctx.load_param_u32("k_param");

            let cta_oob = ctx.setp_ge_u32(cta_row, m_param);
            ctx.branch_if(cta_oob, "exit");
            let cta_oob2 = ctx.setp_ge_u32(cta_col, n_param);
            ctx.branch_if(cta_oob2, "exit");

            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let c_ptr = ctx.load_param_u64("c_ptr");

            // Warp layout: 2×2 grid (warp_id = tid/32, 0-3)
            let warp_id = ctx.shr_u32(tid, c_5);
            let warp_row = ctx.shr_u32(warp_id, c_1);
            let warp_col = ctx.and_u32(warp_id, c_1);
            let warp_m_off = ctx.mul_u32_reg(warp_row, c_16);
            let warp_n_off = ctx.mul_u32_reg(warp_col, c_16);

            // Compute base index for cooperative load (tid * 8)
            let c_8 = ctx.mov_u32_imm(8);
            let my_base = ctx.mul_u32_reg(tid, c_8);

            // Determine if this thread loads A or B (warp-uniform, PERF-CTA-003)
            let is_a_warp = ctx.setp_lt_u32(my_base, c_512);

            // PERF-CTA-004: Pre-compute if CTA tile is interior (fully in-bounds)
            let cta_row_end = ctx.add_u32_reg(cta_row, c_31);
            let cta_col_end = ctx.add_u32_reg(cta_col, c_31);
            let row_interior = ctx.setp_lt_u32(cta_row_end, m_param);
            let col_interior = ctx.setp_lt_u32(cta_col_end, n_param);
            let mn_interior = ctx.and_pred(row_interior, col_interior);

            // PERF-CTA-006: Pre-compute per-element byte addresses OUTSIDE the K-loop.
            let mut a_base_addrs = Vec::with_capacity(8);
            let mut a_smem_addrs = Vec::with_capacity(8);
            let mut a_global_rows = Vec::with_capacity(8);

            let mut b_row_in_tiles = Vec::with_capacity(8);
            let mut b_col_globals = Vec::with_capacity(8);
            let mut b_smem_addrs = Vec::with_capacity(8);

            for i in 0..8u32 {
                let ci = ctx.mov_u32_imm(i);
                let idx = ctx.add_u32_reg(my_base, ci);

                // A-side: pre-compute byte address (row*K + col) * 2 + a_ptr
                let a_r = ctx.shr_u32(idx, c_4);
                let a_c = ctx.and_u32(idx, c_15);
                let a_gr = ctx.add_u32_reg(cta_row, a_r);
                let a_row_k = ctx.mul_u32_reg(a_gr, k_const);
                let a_flat_base = ctx.add_u32_reg(a_row_k, a_c);
                let a_byte_off = ctx.mul_wide_u32(a_flat_base, 2);
                let a_base = ctx.add_u64(a_ptr, a_byte_off);
                let smem_off = ctx.mul_u32_reg(idx, c_2);

                a_base_addrs.push(a_base);
                a_smem_addrs.push(smem_off);
                a_global_rows.push(a_gr);

                // B-side pre-computation
                let bi = ctx.sub_u32_reg(idx, c_512);
                let b_r = ctx.shr_u32(bi, c_5);
                let b_c = ctx.and_u32(bi, c_31);
                let b_gc = ctx.add_u32_reg(cta_col, b_c);
                let bsmem_off = ctx.mul_u32_reg(bi, c_2);
                let bsmem_addr = ctx.add_u32_reg(c_1024, bsmem_off);

                b_row_in_tiles.push(b_r);
                b_col_globals.push(b_gc);
                b_smem_addrs.push(bsmem_addr);
            }

            // Init WMMA accumulator
            let frag_c = ctx.wmma_init_c_zero();

            if double_buffer {
                // ═══════════════════════════════════════════════════════════════
                // PERF-CTA-007: Double-buffered K-loop
                //
                // Structure: Prologue → Main loop → Epilogue
                //   Prologue: load tile 0 into buf[0], bar_sync
                //   Loop:     swap buffers, load next tile into store_buf,
                //             WMMA on compute_buf (overlap!), bar_sync
                //   Epilogue: WMMA on last loaded tile
                //
                // Key: loads and WMMA operate on DIFFERENT buffers, so they
                // overlap on the GPU's memory and tensor core pipelines.
                // ═══════════════════════════════════════════════════════════════

                let smem_single_reg = ctx.mov_u32_imm(smem_single as u32);
                let store_buf_off = ctx.mov_u32_imm(0);

                // ─── PROLOGUE: load tile 0 into buf[0] ───
                {
                    let k_off = ctx.mov_u32_imm(0);
                    let k_byte_stride = ctx.mov_u64_imm(0);

                    // store_buf_off = 0, so buffered = base addresses
                    let pro_a_smem: Vec<_> =
                        a_smem_addrs.iter().map(|&a| ctx.add_u32_reg(a, store_buf_off)).collect();
                    let pro_b_smem: Vec<_> =
                        b_smem_addrs.iter().map(|&b| ctx.add_u32_reg(b, store_buf_off)).collect();

                    // PERF-CTA-004: interior check for tile 0
                    let k_tile_end = ctx.add_u32_reg(k_off, c_15);
                    let k_ok = ctx.setp_lt_u32(k_tile_end, k_param);
                    let fully_interior = ctx.and_pred(mn_interior, k_ok);
                    ctx.branch_if(fully_interior, "pro_fast_load");

                    // Slow path A
                    ctx.branch_if_not(is_a_warp, "pro_slow_b");
                    for i in 0..8usize {
                        ctx.st_shared_f16(pro_a_smem[i], zero_f16);
                        let skip = format!("pro_ssa{i}");
                        let ar_ok = ctx.setp_lt_u32(a_global_rows[i], m_param);
                        ctx.branch_if_not(ar_ok, &skip);
                        let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                        let ac_ok = ctx.setp_lt_u32(k_off, k_param);
                        ctx.branch_if_not(ac_ok, &skip);
                        let a_val = ctx.ld_global_f16(a_addr);
                        ctx.st_shared_f16(pro_a_smem[i], a_val);
                        ctx.label(&skip);
                    }
                    ctx.branch("pro_load_done");
                    ctx.label("pro_slow_b");
                    for i in 0..8usize {
                        let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                        ctx.st_shared_f16(pro_b_smem[i], zero_f16);
                        let skip = format!("pro_ssb{i}");
                        let br_ok = ctx.setp_lt_u32(b_gr, k_param);
                        ctx.branch_if_not(br_ok, &skip);
                        let bc_ok = ctx.setp_lt_u32(b_col_globals[i], n_param);
                        ctx.branch_if_not(bc_ok, &skip);
                        let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                        let b_boff = ctx.mul_wide_u32(b_flat, 2);
                        let b_addr = ctx.add_u64(b_ptr, b_boff);
                        let b_val = ctx.ld_global_f16(b_addr);
                        ctx.st_shared_f16(pro_b_smem[i], b_val);
                        ctx.label(&skip);
                    }
                    ctx.branch("pro_load_done");

                    // Fast path
                    ctx.label("pro_fast_load");
                    ctx.branch_if_not(is_a_warp, "pro_fast_b");
                    for i in 0..8usize {
                        let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                        let a_val = ctx.ld_global_f16(a_addr);
                        ctx.st_shared_f16(pro_a_smem[i], a_val);
                    }
                    ctx.branch("pro_load_done");
                    ctx.label("pro_fast_b");
                    for i in 0..8usize {
                        let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                        let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                        let b_boff = ctx.mul_wide_u32(b_flat, 2);
                        let b_addr = ctx.add_u64(b_ptr, b_boff);
                        let b_val = ctx.ld_global_f16(b_addr);
                        ctx.st_shared_f16(pro_b_smem[i], b_val);
                    }
                    ctx.label("pro_load_done");
                }
                ctx.bar_sync(0); // tile 0 ready in buf[0]

                // ─── MAIN LOOP: tiles 1..n_k_tiles-1 ───
                let k_tile = ctx.mov_u32_imm(1);
                ctx.label("k_loop");
                let k_done = ctx.setp_ge_u32(k_tile, n_k_tiles_reg);
                ctx.branch_if(k_done, "dbuf_epilogue");

                // Swap: store_buf_off ← smem_single - store_buf_off
                let new_store = ctx.sub_u32_reg(smem_single_reg, store_buf_off);
                ctx.mov_u32_reg(store_buf_off, new_store);
                // Compute buffer = opposite of store buffer
                let compute_buf_off = ctx.sub_u32_reg(smem_single_reg, store_buf_off);

                let k_off = ctx.mul_u32_reg(k_tile, c_16);
                let k_byte_stride = ctx.mul_wide_u32(k_off, 2);

                // Buffered smem addresses for store buffer
                let lp_a_smem: Vec<_> =
                    a_smem_addrs.iter().map(|&a| ctx.add_u32_reg(a, store_buf_off)).collect();
                let lp_b_smem: Vec<_> =
                    b_smem_addrs.iter().map(|&b| ctx.add_u32_reg(b, store_buf_off)).collect();

                // PERF-CTA-004: interior check
                let k_tile_end = ctx.add_u32_reg(k_off, c_15);
                let k_ok = ctx.setp_lt_u32(k_tile_end, k_param);
                let fully_interior = ctx.and_pred(mn_interior, k_ok);
                ctx.branch_if(fully_interior, "fast_load");

                // === LOAD: slow path (edge tiles) ===
                ctx.branch_if_not(is_a_warp, "slow_b");
                for i in 0..8usize {
                    ctx.st_shared_f16(lp_a_smem[i], zero_f16);
                    let skip = format!("ssa{i}");
                    let ar_ok = ctx.setp_lt_u32(a_global_rows[i], m_param);
                    ctx.branch_if_not(ar_ok, &skip);
                    let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                    let ac_ok = ctx.setp_lt_u32(k_off, k_param);
                    ctx.branch_if_not(ac_ok, &skip);
                    let a_val = ctx.ld_global_f16(a_addr);
                    ctx.st_shared_f16(lp_a_smem[i], a_val);
                    ctx.label(&skip);
                }
                ctx.branch("load_done");
                ctx.label("slow_b");
                for i in 0..8usize {
                    let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                    ctx.st_shared_f16(lp_b_smem[i], zero_f16);
                    let skip = format!("ssb{i}");
                    let br_ok = ctx.setp_lt_u32(b_gr, k_param);
                    ctx.branch_if_not(br_ok, &skip);
                    let bc_ok = ctx.setp_lt_u32(b_col_globals[i], n_param);
                    ctx.branch_if_not(bc_ok, &skip);
                    let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                    let b_boff = ctx.mul_wide_u32(b_flat, 2);
                    let b_addr = ctx.add_u64(b_ptr, b_boff);
                    let b_val = ctx.ld_global_f16(b_addr);
                    ctx.st_shared_f16(lp_b_smem[i], b_val);
                    ctx.label(&skip);
                }
                ctx.branch("load_done");

                // === FAST PATH: separated loads for true WMMA overlap ===
                // PERF-CTA-008: Issue all ld.global first (non-blocking),
                // then WMMA from compute buffer (tensor core pipeline overlaps
                // with in-flight global loads), then st.shared.
                ctx.label("fast_load");
                ctx.branch_if_not(is_a_warp, "fast_b");

                // Phase 1a: issue all A global loads (non-blocking)
                let mut a_vals = Vec::with_capacity(8);
                for i in 0..8usize {
                    let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                    a_vals.push(ctx.ld_global_f16(a_addr));
                }
                // Phase 2a: WMMA from compute buffer (overlaps with A loads)
                {
                    let smem_base = ctx.shared_base_addr();
                    let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_32);
                    let a_sub_buf = ctx.add_u32_reg(a_sub_bytes, compute_buf_off);
                    let a_sub_off = ctx.cvt_u64_u32(a_sub_buf);
                    let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                    let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);
                    let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                    let b_base_off = ctx.add_u32_reg(c_1024, b_sub_bytes);
                    let b_off_buf = ctx.add_u32_reg(b_base_off, compute_buf_off);
                    let b_off64 = ctx.cvt_u64_u32(b_off_buf);
                    let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                    let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 32, WmmaLayout::RowMajor);
                    let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                    for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                        ctx.mov_f32_reg(*c_reg, *d_reg);
                    }
                }
                // Phase 3a: store A loads to shared (stalls here, but WMMA done)
                for i in 0..8usize {
                    ctx.st_shared_f16(lp_a_smem[i], a_vals[i]);
                }
                ctx.branch("load_wmma_done");

                ctx.label("fast_b");
                // Phase 1b: issue all B global loads
                let mut b_vals = Vec::with_capacity(8);
                for i in 0..8usize {
                    let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                    let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                    let b_boff = ctx.mul_wide_u32(b_flat, 2);
                    let b_addr = ctx.add_u64(b_ptr, b_boff);
                    b_vals.push(ctx.ld_global_f16(b_addr));
                }
                // Phase 2b: WMMA from compute buffer
                {
                    let smem_base = ctx.shared_base_addr();
                    let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_32);
                    let a_sub_buf = ctx.add_u32_reg(a_sub_bytes, compute_buf_off);
                    let a_sub_off = ctx.cvt_u64_u32(a_sub_buf);
                    let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                    let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);
                    let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                    let b_base_off = ctx.add_u32_reg(c_1024, b_sub_bytes);
                    let b_off_buf = ctx.add_u32_reg(b_base_off, compute_buf_off);
                    let b_off64 = ctx.cvt_u64_u32(b_off_buf);
                    let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                    let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 32, WmmaLayout::RowMajor);
                    let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                    for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                        ctx.mov_f32_reg(*c_reg, *d_reg);
                    }
                }
                // Phase 3b: store B loads to shared
                for i in 0..8usize {
                    ctx.st_shared_f16(lp_b_smem[i], b_vals[i]);
                }

                ctx.label("load_wmma_done");
                // Slow path also needs WMMA — jump target after slow load+store
                ctx.branch("after_wmma");

                // === SLOW PATH WMMA ===
                // Slow path (edge tiles) already stored to shared above.
                // Now do WMMA for the slow path.
                ctx.label("load_done");
                {
                    let smem_base = ctx.shared_base_addr();
                    let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_32);
                    let a_sub_buf = ctx.add_u32_reg(a_sub_bytes, compute_buf_off);
                    let a_sub_off = ctx.cvt_u64_u32(a_sub_buf);
                    let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                    let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);
                    let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                    let b_base_off = ctx.add_u32_reg(c_1024, b_sub_bytes);
                    let b_off_buf = ctx.add_u32_reg(b_base_off, compute_buf_off);
                    let b_off64 = ctx.cvt_u64_u32(b_off_buf);
                    let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                    let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 32, WmmaLayout::RowMajor);
                    let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                    for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                        ctx.mov_f32_reg(*c_reg, *d_reg);
                    }
                }
                ctx.label("after_wmma");

                ctx.bar_sync(0); // sync both load and compute
                ctx.add_u32_inplace(k_tile, 1);
                ctx.branch("k_loop");

                // ─── EPILOGUE: WMMA on last loaded tile ───
                ctx.label("dbuf_epilogue");
                {
                    let smem_base = ctx.shared_base_addr();

                    let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_32);
                    let a_sub_buf = ctx.add_u32_reg(a_sub_bytes, store_buf_off);
                    let a_sub_off = ctx.cvt_u64_u32(a_sub_buf);
                    let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                    let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);

                    let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                    let b_base_off = ctx.add_u32_reg(c_1024, b_sub_bytes);
                    let b_off_buf = ctx.add_u32_reg(b_base_off, store_buf_off);
                    let b_off64 = ctx.cvt_u64_u32(b_off_buf);
                    let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                    let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 32, WmmaLayout::RowMajor);

                    let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                    for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                        ctx.mov_f32_reg(*c_reg, *d_reg);
                    }
                }
            } else {
                // ═══════════════════════════════════════════════════════════════
                // Single-buffer K-loop (original path)
                // ═══════════════════════════════════════════════════════════════
                let k_tile = ctx.mov_u32_imm(0);
                ctx.label("k_loop");
                let k_done = ctx.setp_ge_u32(k_tile, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_end");

                let k_off = ctx.mul_u32_reg(k_tile, c_16);
                // k_byte_stride BEFORE branch so it's defined on both paths
                let k_byte_stride = ctx.mul_wide_u32(k_off, 2);

                // PERF-CTA-004: Check if K-tile is fully in-bounds
                let k_tile_end = ctx.add_u32_reg(k_off, c_15);
                let k_ok = ctx.setp_lt_u32(k_tile_end, k_param);
                let fully_interior = ctx.and_pred(mn_interior, k_ok);
                ctx.branch_if(fully_interior, "fast_load");

                // === SLOW PATH: boundary-checked loads ===
                ctx.branch_if_not(is_a_warp, "slow_b");
                for i in 0..8usize {
                    ctx.st_shared_f16(a_smem_addrs[i], zero_f16);
                    let skip = format!("ssa{i}");
                    let ar_ok = ctx.setp_lt_u32(a_global_rows[i], m_param);
                    ctx.branch_if_not(ar_ok, &skip);
                    let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                    let ac_ok = ctx.setp_lt_u32(k_off, k_param);
                    ctx.branch_if_not(ac_ok, &skip);
                    let a_val = ctx.ld_global_f16(a_addr);
                    ctx.st_shared_f16(a_smem_addrs[i], a_val);
                    ctx.label(&skip);
                }
                ctx.branch("load_done");
                ctx.label("slow_b");
                for i in 0..8usize {
                    let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                    ctx.st_shared_f16(b_smem_addrs[i], zero_f16);
                    let skip = format!("ssb{i}");
                    let br_ok = ctx.setp_lt_u32(b_gr, k_param);
                    ctx.branch_if_not(br_ok, &skip);
                    let bc_ok = ctx.setp_lt_u32(b_col_globals[i], n_param);
                    ctx.branch_if_not(bc_ok, &skip);
                    let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                    let b_boff = ctx.mul_wide_u32(b_flat, 2);
                    let b_addr = ctx.add_u64(b_ptr, b_boff);
                    let b_val = ctx.ld_global_f16(b_addr);
                    ctx.st_shared_f16(b_smem_addrs[i], b_val);
                    ctx.label(&skip);
                }
                ctx.branch("load_done");

                // === FAST PATH: pre-computed base + k_byte_stride ===
                ctx.label("fast_load");
                ctx.branch_if_not(is_a_warp, "fast_b");
                for i in 0..8usize {
                    let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                    let a_val = ctx.ld_global_f16(a_addr);
                    ctx.st_shared_f16(a_smem_addrs[i], a_val);
                }
                ctx.branch("load_done");
                ctx.label("fast_b");
                for i in 0..8usize {
                    let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                    let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                    let b_boff = ctx.mul_wide_u32(b_flat, 2);
                    let b_addr = ctx.add_u64(b_ptr, b_boff);
                    let b_val = ctx.ld_global_f16(b_addr);
                    ctx.st_shared_f16(b_smem_addrs[i], b_val);
                }

                ctx.label("load_done");
                ctx.bar_sync(0);

                // === WMMA per warp ===
                let smem_base = ctx.shared_base_addr();

                let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_32);
                let a_sub_off = ctx.cvt_u64_u32(a_sub_bytes);
                let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);

                let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                let b_offset = ctx.add_u32_reg(c_1024, b_sub_bytes);
                let b_off64 = ctx.cvt_u64_u32(b_offset);
                let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 32, WmmaLayout::RowMajor);

                // PERF-CTA-001: row.row MMA matches RowMajor loads
                let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }

                ctx.bar_sync(1);
                ctx.add_u32_inplace(k_tile, 1);
                ctx.branch("k_loop");
                ctx.label("k_end");
            }

            // === Store C ===
            let c_row = ctx.add_u32_reg(cta_row, warp_m_off);
            let c_col = ctx.add_u32_reg(cta_col, warp_n_off);
            let cr_ok = ctx.setp_lt_u32(c_row, m_param);
            ctx.branch_if_not(cr_ok, "exit");
            let cc_ok = ctx.setp_lt_u32(c_col, n_param);
            ctx.branch_if_not(cc_ok, "exit");

            let c_row_off = ctx.mul_wide_u32_reg(c_row, n_param);
            let c_col_off = ctx.cvt_u64_u32(c_col);
            let c_base = ctx.add_u64(c_row_off, c_col_off);
            let c_base = ctx.mul_u64(c_base, 4);
            let c_addr = ctx.add_u64(c_ptr, c_base);

            ctx.wmma_store_d_f32(c_addr, &frag_c, n, WmmaLayout::RowMajor);

            ctx.label("exit");
            ctx.ret();
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::PtxModule;

    #[test]
    fn test_cta_wmma_generates_valid_ptx() {
        let kernel = build_cta_wmma_fp16(128, 128, 128);
        let module = PtxModule::new().add_kernel(kernel);
        let ptx = module.emit();
        assert!(ptx.contains(".entry gemm_cta_wmma_fp16"));
        assert!(ptx.contains("bar.sync"));
    }

    #[test]
    fn test_cta_wmma_has_shared_memory() {
        let kernel = build_cta_wmma_fp16(64, 64, 64);
        let module = PtxModule::new().add_kernel(kernel);
        let ptx = module.emit();
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn test_cta_wmma_uses_row_row_mma() {
        let kernel = build_cta_wmma_fp16(512, 512, 512);
        let module = PtxModule::new().add_kernel(kernel);
        let ptx = module.emit();
        assert!(
            ptx.contains("wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32"),
            "MMA must use row.row to match RowMajor loads for both A and B"
        );
    }

    #[test]
    fn test_cta_wmma_has_fast_and_slow_paths() {
        let kernel = build_cta_wmma_fp16(256, 256, 256);
        let module = PtxModule::new().add_kernel(kernel);
        let ptx = module.emit();
        assert!(ptx.contains("fast_load"), "must have interior tile fast path");
        assert!(ptx.contains("slow_b"), "must have edge tile slow path");
    }

    // ── Double-buffer FALSIFY tests ──

    #[test]
    fn test_dbuf_generates_valid_ptx() {
        let kernel = build_cta_wmma_fp16_dbuf(128, 128, 128);
        let module = PtxModule::new().add_kernel(kernel);
        let ptx = module.emit();
        assert!(ptx.contains(".entry gemm_cta_wmma_fp16"), "must emit valid kernel entry");
        assert!(ptx.contains("bar.sync"), "must have barrier sync");
    }

    #[test]
    fn test_dbuf_double_shared_memory() {
        // Single-buffer: 2048 bytes. Double-buffer: 4096 bytes.
        let single = build_cta_wmma_fp16(256, 256, 256);
        let double = build_cta_wmma_fp16_dbuf(256, 256, 256);
        let ptx_s = PtxModule::new().add_kernel(single).emit();
        let ptx_d = PtxModule::new().add_kernel(double).emit();

        // Extract shared memory size from ".shared .align 16 .b8 smem[N]"
        let extract_smem = |ptx: &str| -> usize {
            for line in ptx.lines() {
                if line.contains(".shared") && line.contains("smem[") {
                    let start = line.find("smem[").unwrap() + 5;
                    let end = line[start..].find(']').unwrap() + start;
                    return line[start..end].parse().unwrap();
                }
            }
            panic!("no .shared smem found");
        };

        let smem_s = extract_smem(&ptx_s);
        let smem_d = extract_smem(&ptx_d);
        assert_eq!(smem_s, 2048, "single-buffer smem must be 2048");
        assert_eq!(smem_d, 4096, "double-buffer smem must be 4096 (2x)");
    }

    #[test]
    fn test_dbuf_has_prologue_and_epilogue() {
        let kernel = build_cta_wmma_fp16_dbuf(512, 512, 512);
        let ptx = PtxModule::new().add_kernel(kernel).emit();

        // Prologue labels
        assert!(
            ptx.contains("pro_fast_load") || ptx.contains("pro_slow_b"),
            "double-buffer must have prologue load labels"
        );
        assert!(ptx.contains("pro_load_done"), "double-buffer must have prologue load_done");

        // Epilogue label
        assert!(ptx.contains("dbuf_epilogue"), "double-buffer must have epilogue label");
    }

    #[test]
    fn test_dbuf_has_buffer_swap() {
        let kernel = build_cta_wmma_fp16_dbuf(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();

        // Double-buffer swap uses sub.u32 to toggle store_buf_off
        let sub_count = ptx.matches("sub.u32").count();
        assert!(
            sub_count >= 2,
            "double-buffer needs >=2 sub.u32 for buffer swap (store + compute), got {sub_count}"
        );
    }

    #[test]
    fn test_dbuf_has_multiple_wmma_mma() {
        // Double-buffer emits WMMA 4 times: fast_a, fast_b, slow path, epilogue
        // (PERF-CTA-008: separated loads require per-branch WMMA for overlap)
        let kernel = build_cta_wmma_fp16_dbuf(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();

        let mma_count = ptx.matches("wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32").count();
        assert_eq!(
            mma_count, 4,
            "double-buffer must have 4 WMMA MMA ops (fast_a + fast_b + slow + epilogue), got {mma_count}"
        );
    }

    #[test]
    fn test_dbuf_single_tile_edge_case() {
        // K=16 → 1 tile. Prologue loads it, loop never executes, epilogue computes.
        let kernel = build_cta_wmma_fp16_dbuf(64, 64, 16);
        let module = PtxModule::new().add_kernel(kernel);
        let ptx = module.emit();
        assert!(ptx.contains(".entry gemm_cta_wmma_fp16"));
        // Must have WMMA ops in code (fast_a, fast_b, slow, epilogue)
        let mma_count = ptx.matches("wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32").count();
        assert_eq!(mma_count, 4);
    }

    #[test]
    fn test_single_buffer_k_byte_stride_before_branch() {
        // Verify that k_byte_stride (mul.wide.u32) appears BEFORE the branch
        // to fast_load, so it's always defined on both paths.
        let kernel = build_cta_wmma_fp16(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();

        // Find positions: mul.wide.u32 for k_byte_stride must come before fast_load label
        let mul_wide_pos = ptx.find("mul.wide.u32").expect("must have mul.wide.u32");
        let fast_load_pos = ptx.find("fast_load:").expect("must have fast_load label");
        assert!(
            mul_wide_pos < fast_load_pos,
            "k_byte_stride (mul.wide.u32) must be emitted before fast_load branch target"
        );
    }
}
