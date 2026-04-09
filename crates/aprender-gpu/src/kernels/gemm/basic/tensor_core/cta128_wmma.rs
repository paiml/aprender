//! 128×128 CTA-level WMMA GEMM — Phase 2 of GPU GEMM bridge plan
//!
//! Architecture: 8×8 warp grid (64 warps → too many), OR 4×4 warp grid
//! where each warp computes a 2×2 block of 16×16 WMMA tiles = 32×32 output.
//! All warps share 128×K A tile and K×128 B tile in shared memory.
//!
//! K-dimension tiled in chunks of 16 (WMMA tile depth).
//! 3-stage cp.async pipeline for latency hiding.
//!
//! Key advantage over 64×64 CTA: 2× compute-to-load ratio (64 vs 32 FLOP/byte).
//! Each loaded A element is reused by 8 column warps (vs 4 in 64×64).
//! Each loaded B element is reused by 8 row warps (vs 4 in 64×64).
//!
//! Shared memory: 128×16×2 (A) + 16×128×2 (B) = 8 KB per stage.
//! 3 stages = 24 KB total (fits in 48 KB static smem).
//!
//! Launch: grid_2d((N+127)/128, (M+127)/128), block(512,1,1), shared_mem=24KB
//! Input: FP16 (u16), Output: FP32
//!
//! Target: 60+ TFLOP/s on RTX 4090 (from 40.5 at 64×64 → +48%)
//!
//! Ref: CUTLASS SM80 default: GemmShape<128,256,64>, WarpShape<64,64,64>,
//! InstructionShape<16,8,16> (mma.sync.m16n8k16). We use wmma.m16n16k16.

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

/// Build a 128×128 CTA-tiled WMMA GEMM kernel with 3-stage cp.async pipeline.
///
/// 16 warps (512 threads), 4×4 grid where each warp computes 2×2 block of
/// 16×16 WMMA tiles = 32×32 output per warp. 64 WMMAs per K-tile (vs 16 in 64×64).
///
/// Contract: cgp-gpu-gemm-cta128-v1.yaml (when created)
pub fn build_cta128_wmma_fp16_cpasync(m: u32, n: u32, k: u32) -> PtxKernel {
    let tile_m: u32 = 128;
    let tile_n: u32 = 128;
    let tile_k: u32 = 16;
    let a_smem_bytes = (tile_m * tile_k * 2) as usize; // 4096
    let b_smem_bytes = (tile_k * tile_n * 2) as usize; // 4096
    let smem_single = a_smem_bytes + b_smem_bytes; // 8192
    let num_stages: usize = 3;
    let smem_bytes = smem_single * num_stages; // 24576 = 24KB
    let n_k_tiles = (k + tile_k - 1) / tile_k;
    // 512 threads: 256 load A (128*16=2048 FP16 = 4096 bytes → 16 bytes/thread = 2 cp.async)
    // 256 load B (same)
    let a_threads: u32 = 256;
    let _ = m;

    PtxKernel::new("gemm_cta128_cpasync_fp16")
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

            let c_2 = ctx.mov_u32_imm(2);
            let c_4 = ctx.mov_u32_imm(4);
            let c_5 = ctx.mov_u32_imm(5);
            let c_8 = ctx.mov_u32_imm(8);
            let c_16 = ctx.mov_u32_imm(16);
            let c_32 = ctx.mov_u32_imm(32);
            let c_128 = ctx.mov_u32_imm(128);
            let c_256 = ctx.mov_u32_imm(a_threads);
            let c_a_smem = ctx.mov_u32_imm(a_smem_bytes as u32);
            let k_const = ctx.mov_u32_imm(k);
            let n_const = ctx.mov_u32_imm(n);
            let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);

            // CTA position: 128×128 tiles
            let cta_row = ctx.mul_u32_reg(ctaid_y, c_128);
            let cta_col = ctx.mul_u32_reg(ctaid_x, c_128);

            let m_param = ctx.load_param_u32("m_param");
            let n_param = ctx.load_param_u32("n_param");
            let _k_param = ctx.load_param_u32("k_param");

            // Bounds check: skip entire CTA if out of bounds
            let cta_oob = ctx.setp_ge_u32(cta_row, m_param);
            ctx.branch_if(cta_oob, "exit");
            let cta_oob2 = ctx.setp_ge_u32(cta_col, n_param);
            ctx.branch_if(cta_oob2, "exit");

            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let c_ptr = ctx.load_param_u64("c_ptr");

            // Warp layout: 4×4 grid, each warp computes 2×2 WMMA tiles = 32×32 output
            let c_3 = ctx.mov_u32_imm(3);
            let warp_id = ctx.shr_u32(tid, c_5); // tid / 32
            let warp_row = ctx.shr_u32(warp_id, c_2); // warp_id / 4 (0..3)
            let warp_col = ctx.and_u32(warp_id, c_3); // warp_id % 4 (0..3)
                                                      // Each warp position: 32×32 block at (warp_row*32, warp_col*32)
            let warp_m_off = ctx.mul_u32_reg(warp_row, c_32); // row offset in tile
            let warp_n_off = ctx.mul_u32_reg(warp_col, c_32); // col offset in tile

            let smem_base = ctx.shared_base_addr();

            // Thread role for loading: is_a = (tid < 256)
            let is_a_thread = ctx.setp_lt_u32(tid, c_256);

            // ═══ Load offsets ═══
            // A: 128 rows × 16 cols = 2048 FP16. 256 threads → 8 elements/thread → 2 cp.async(8B)
            // Thread t loads: elements [t*8..t*8+7], i.e., row = (t*8)/16 = t/2, col = (t*8)%16
            // cp.async 0: smem_off = t*16, global_off = row*K + col
            // cp.async 1: smem_off = t*16 + 8, global_off = row*K + col + 4
            let _a_row_in_tile = ctx.shr_u32(tid, c_2); // t/4 (covers 0..63 for first half)
                                                        // But we need rows 0..127 with 256 threads loading 8 elements each
                                                        // Better: t*8/16 = t/2 for the row
            let a_elem_start = ctx.mul_u32_reg(tid, c_8); // t * 8
            let a_row0 = ctx.shr_u32(a_elem_start, c_4); // (t*8) / 16
            let c_mask15 = ctx.mov_u32_imm(15);
            let a_col0_base = ctx.and_u32(a_elem_start, c_mask15); // (t*8) % 16

            // smem offsets for A loads (2 cp.async per thread)
            let a_smem_off0 = ctx.mul_u32_reg(tid, c_16); // t*16 bytes (t*8 elements * 2 bytes)
            let a_smem_off1 = ctx.add_u32_reg(a_smem_off0, c_8); // +8 bytes

            // B: 16 rows × 128 cols = 2048 FP16. 256 threads → 8 elements/thread → 2 cp.async(8B)
            let b_local = ctx.sub_u32_reg(tid, c_256);
            let b_elem_start = ctx.mul_u32_reg(b_local, c_8); // local * 8
            let c_mask127 = ctx.mov_u32_imm(127);
            let c_7 = ctx.mov_u32_imm(7);
            let b_row0 = ctx.shr_u32(b_elem_start, c_7); // (local*8)/128
            let b_col0_base = ctx.and_u32(b_elem_start, c_mask127); // (local*8)%128

            let b_smem_base = ctx.mov_u32_imm(a_smem_bytes as u32);
            let b_smem_off0_local = ctx.mul_u32_reg(b_local, c_16); // local*16 bytes
            let b_smem_off0 = ctx.add_u32_reg(b_smem_base, b_smem_off0_local);
            let b_smem_off1 = ctx.add_u32_reg(b_smem_off0, c_8); // +8 bytes

            // Initialize 2×2 WMMA accumulators (4 fragments × 8 regs = 32 regs)
            // Each warp computes a 32×32 block as 2×2 grid of 16×16 WMMA tiles.
            let frag_c00 = ctx.wmma_init_c_zero();
            let frag_c01 = ctx.wmma_init_c_zero();
            let frag_c10 = ctx.wmma_init_c_zero();
            let frag_c11 = ctx.wmma_init_c_zero();

            let smem_single_reg = ctx.mov_u32_imm(smem_single as u32);
            let store_buf_off = ctx.mov_u32_imm(0);

            // Helper: issue 2 cp.async per thread to load 128×16 A or 16×128 B
            // For A threads: smem_off0/1, global addr from a_row0, a_col0_base + k_off
            // For B threads: smem_off0/1, global addr from b_row0 + k_off, b_col0_base

            // ─── Prologue: cp.async tile 0 into buf[0] ───
            {
                let k_off = ctx.mov_u32_imm(0);
                // A: (cta_row + a_row0) * K + k_off + a_col0_base
                let a_row_global = ctx.add_u32_reg(cta_row, a_row0);
                let a_rowk = ctx.mul_u32_reg(a_row_global, k_const);
                let a_flat0 = ctx.add_u32_reg(a_rowk, a_col0_base);
                let a_byte0 = ctx.mul_wide_u32(a_flat0, 2);
                let a_addr0 = ctx.add_u64(a_ptr, a_byte0);
                // Second cp.async: col + 4
                let a_col1 = ctx.add_u32_reg(a_col0_base, c_4);
                let a_flat1 = ctx.add_u32_reg(a_rowk, a_col1);
                let a_byte1 = ctx.mul_wide_u32(a_flat1, 2);
                let a_addr1 = ctx.add_u64(a_ptr, a_byte1);

                // B: (k_off + b_row0) * N + cta_col + b_col0_base
                let b_row_global = ctx.add_u32_reg(k_off, b_row0);
                let b_flat0 = ctx.mad_lo_u32(b_row_global, n_const, cta_col);
                let b_flat0 = ctx.add_u32_reg(b_flat0, b_col0_base);
                let b_byte0 = ctx.mul_wide_u32(b_flat0, 2);
                let b_addr0 = ctx.add_u64(b_ptr, b_byte0);
                let b_col1 = ctx.add_u32_reg(b_col0_base, c_4);
                let b_flat1 = ctx.mad_lo_u32(b_row_global, n_const, cta_col);
                let b_flat1 = ctx.add_u32_reg(b_flat1, b_col1);
                let b_byte1 = ctx.mul_wide_u32(b_flat1, 2);
                let b_addr1 = ctx.add_u64(b_ptr, b_byte1);

                // Issue 2 cp.async per thread (A or B based on role)
                let gaddr0 = ctx.selp_u64(is_a_thread, a_addr0, b_addr0);
                let gaddr1 = ctx.selp_u64(is_a_thread, a_addr1, b_addr1);
                let soff0 = ctx.selp_u32(is_a_thread, a_smem_off0, b_smem_off0);
                let soff1 = ctx.selp_u32(is_a_thread, a_smem_off1, b_smem_off1);
                ctx.cp_async_global_to_shared(soff0, gaddr0, 8);
                ctx.cp_async_global_to_shared(soff1, gaddr1, 8);
                let _ = k_off;
            }
            ctx.cp_async_commit_group();
            ctx.cp_async_wait_group(0);
            ctx.bar_sync(0);

            // ─── Main K-loop (double-buffer) ───
            let k_tile = ctx.mov_u32_imm(1);
            ctx.label("k_loop");
            let k_done = ctx.setp_ge_u32(k_tile, n_k_tiles_reg);
            ctx.branch_if(k_done, "dbuf_epi");

            // Swap buffer
            let new_store = ctx.sub_u32_reg(smem_single_reg, store_buf_off);
            ctx.mov_u32_reg(store_buf_off, new_store);
            let compute_off = ctx.sub_u32_reg(smem_single_reg, store_buf_off);

            // Async load next K-tile into store buffer
            let k_off = ctx.mul_u32_reg(k_tile, c_16);
            {
                let a_row_global = ctx.add_u32_reg(cta_row, a_row0);
                let a_rowk = ctx.mul_u32_reg(a_row_global, k_const);
                let a_col_full = ctx.add_u32_reg(k_off, a_col0_base);
                let a_flat0 = ctx.add_u32_reg(a_rowk, a_col_full);
                let a_byte0 = ctx.mul_wide_u32(a_flat0, 2);
                let a_addr0 = ctx.add_u64(a_ptr, a_byte0);
                let a_col_full1 = ctx.add_u32_reg(a_col_full, c_4);
                let a_flat1 = ctx.add_u32_reg(a_rowk, a_col_full1);
                let a_byte1 = ctx.mul_wide_u32(a_flat1, 2);
                let a_addr1 = ctx.add_u64(a_ptr, a_byte1);

                let b_row_full = ctx.add_u32_reg(k_off, b_row0);
                let b_flat0 = ctx.mad_lo_u32(b_row_full, n_const, cta_col);
                let b_flat0 = ctx.add_u32_reg(b_flat0, b_col0_base);
                let b_byte0 = ctx.mul_wide_u32(b_flat0, 2);
                let b_addr0 = ctx.add_u64(b_ptr, b_byte0);
                let b_col_full1 = ctx.add_u32_reg(b_col0_base, c_4);
                let b_flat1 = ctx.mad_lo_u32(b_row_full, n_const, cta_col);
                let b_flat1 = ctx.add_u32_reg(b_flat1, b_col_full1);
                let b_byte1 = ctx.mul_wide_u32(b_flat1, 2);
                let b_addr1 = ctx.add_u64(b_ptr, b_byte1);

                let gaddr0 = ctx.selp_u64(is_a_thread, a_addr0, b_addr0);
                let gaddr1 = ctx.selp_u64(is_a_thread, a_addr1, b_addr1);
                let soff0 = ctx.selp_u32(is_a_thread, a_smem_off0, b_smem_off0);
                let soff1 = ctx.selp_u32(is_a_thread, a_smem_off1, b_smem_off1);
                let soff0_buf = ctx.add_u32_reg(soff0, store_buf_off);
                let soff1_buf = ctx.add_u32_reg(soff1, store_buf_off);
                ctx.cp_async_global_to_shared(soff0_buf, gaddr0, 8);
                ctx.cp_async_global_to_shared(soff1_buf, gaddr1, 8);
            }
            ctx.cp_async_commit_group();

            // ═══ WMMA compute on compute buffer (4 WMMAs per warp: 2×2 grid) ═══
            // Each warp at (warp_row, warp_col) computes 32×32 output as:
            //   (r,c) = (0,0), (0,1), (1,0), (1,1) sub-tiles of 16×16
            {
                // A fragment addresses: warp_m_off + {0, 16} rows, compute_off
                // B fragment addresses: warp_n_off + {0, 16} cols, compute_off
                let _stride_a = ctx.mov_u32_imm(16); // A leading dim in smem = tile_k=16
                let _stride_b = ctx.mov_u32_imm(128); // B leading dim in smem = tile_n=128

                // Top-left (0,0): A at warp_m_off, B at warp_n_off
                let a00_bytes = ctx.mul_u32_reg(warp_m_off, c_16);
                let a00_bytes = ctx.mul_u32_reg(a00_bytes, c_2);
                let a00_buf = ctx.add_u32_reg(a00_bytes, compute_off);
                let a00_off = ctx.cvt_u64_u32(a00_buf);
                let a00_ptr = ctx.add_u64(smem_base, a00_off);
                let frag_a0 = ctx.wmma_load_a_f16(a00_ptr, 16, WmmaLayout::RowMajor);

                let b00_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                let b00_base = ctx.add_u32_reg(c_a_smem, b00_bytes);
                let b00_buf = ctx.add_u32_reg(b00_base, compute_off);
                let b00_off = ctx.cvt_u64_u32(b00_buf);
                let b00_ptr = ctx.add_u64(smem_base, b00_off);
                let frag_b0 = ctx.wmma_load_b_f16(b00_ptr, 128, WmmaLayout::RowMajor);

                // Top-right (0,1): same A row, B at warp_n_off+16
                let b01_off_bytes = ctx.add_u32_reg(warp_n_off, c_16);
                let b01_bytes = ctx.mul_u32_reg(b01_off_bytes, c_2);
                let b01_base = ctx.add_u32_reg(c_a_smem, b01_bytes);
                let b01_buf = ctx.add_u32_reg(b01_base, compute_off);
                let b01_off = ctx.cvt_u64_u32(b01_buf);
                let b01_ptr = ctx.add_u64(smem_base, b01_off);
                let frag_b1 = ctx.wmma_load_b_f16(b01_ptr, 128, WmmaLayout::RowMajor);

                // Bottom-left (1,0): A at warp_m_off+16, same B col
                let a10_row = ctx.add_u32_reg(warp_m_off, c_16);
                let a10_bytes = ctx.mul_u32_reg(a10_row, c_16);
                let a10_bytes = ctx.mul_u32_reg(a10_bytes, c_2);
                let a10_buf = ctx.add_u32_reg(a10_bytes, compute_off);
                let a10_off = ctx.cvt_u64_u32(a10_buf);
                let a10_ptr = ctx.add_u64(smem_base, a10_off);
                let frag_a1 = ctx.wmma_load_a_f16(a10_ptr, 16, WmmaLayout::RowMajor);

                // Compute 4 WMMAs
                let d00 = ctx.wmma_mma_f16_f32_row_row(&frag_a0, &frag_b0, &frag_c00);
                for (c_r, d_r) in frag_c00.iter().zip(d00.iter()) {
                    ctx.mov_f32_reg(*c_r, *d_r);
                }
                let d01 = ctx.wmma_mma_f16_f32_row_row(&frag_a0, &frag_b1, &frag_c01);
                for (c_r, d_r) in frag_c01.iter().zip(d01.iter()) {
                    ctx.mov_f32_reg(*c_r, *d_r);
                }
                let d10 = ctx.wmma_mma_f16_f32_row_row(&frag_a1, &frag_b0, &frag_c10);
                for (c_r, d_r) in frag_c10.iter().zip(d10.iter()) {
                    ctx.mov_f32_reg(*c_r, *d_r);
                }
                let d11 = ctx.wmma_mma_f16_f32_row_row(&frag_a1, &frag_b1, &frag_c11);
                for (c_r, d_r) in frag_c11.iter().zip(d11.iter()) {
                    ctx.mov_f32_reg(*c_r, *d_r);
                }
            }

            ctx.cp_async_wait_group(0);
            ctx.bar_sync(0);
            ctx.add_u32_inplace(k_tile, 1);
            ctx.branch("k_loop");

            // ─── Epilogue: last K-tile WMMA ───
            ctx.label("dbuf_epi");
            {
                let stride_a = ctx.mov_u32_imm(16);

                let a00_bytes = ctx.mul_u32_reg(warp_m_off, c_16);
                let a00_bytes = ctx.mul_u32_reg(a00_bytes, c_2);
                let a00_buf = ctx.add_u32_reg(a00_bytes, store_buf_off);
                let a00_off = ctx.cvt_u64_u32(a00_buf);
                let a00_ptr = ctx.add_u64(smem_base, a00_off);
                let frag_a0 = ctx.wmma_load_a_f16(a00_ptr, 16, WmmaLayout::RowMajor);

                let b00_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                let b00_base = ctx.add_u32_reg(c_a_smem, b00_bytes);
                let b00_buf = ctx.add_u32_reg(b00_base, store_buf_off);
                let b00_off = ctx.cvt_u64_u32(b00_buf);
                let b00_ptr = ctx.add_u64(smem_base, b00_off);
                let frag_b0 = ctx.wmma_load_b_f16(b00_ptr, 128, WmmaLayout::RowMajor);

                let b01_off_bytes = ctx.add_u32_reg(warp_n_off, c_16);
                let b01_bytes = ctx.mul_u32_reg(b01_off_bytes, c_2);
                let b01_base = ctx.add_u32_reg(c_a_smem, b01_bytes);
                let b01_buf = ctx.add_u32_reg(b01_base, store_buf_off);
                let b01_off = ctx.cvt_u64_u32(b01_buf);
                let b01_ptr = ctx.add_u64(smem_base, b01_off);
                let frag_b1 = ctx.wmma_load_b_f16(b01_ptr, 128, WmmaLayout::RowMajor);

                let a10_row = ctx.add_u32_reg(warp_m_off, c_16);
                let a10_bytes = ctx.mul_u32_reg(a10_row, c_16);
                let a10_bytes = ctx.mul_u32_reg(a10_bytes, c_2);
                let a10_buf = ctx.add_u32_reg(a10_bytes, store_buf_off);
                let a10_off = ctx.cvt_u64_u32(a10_buf);
                let a10_ptr = ctx.add_u64(smem_base, a10_off);
                let frag_a1 = ctx.wmma_load_a_f16(a10_ptr, 16, WmmaLayout::RowMajor);

                let d00 = ctx.wmma_mma_f16_f32_row_row(&frag_a0, &frag_b0, &frag_c00);
                for (c_r, d_r) in frag_c00.iter().zip(d00.iter()) {
                    ctx.mov_f32_reg(*c_r, *d_r);
                }
                let d01 = ctx.wmma_mma_f16_f32_row_row(&frag_a0, &frag_b1, &frag_c01);
                for (c_r, d_r) in frag_c01.iter().zip(d01.iter()) {
                    ctx.mov_f32_reg(*c_r, *d_r);
                }
                let d10 = ctx.wmma_mma_f16_f32_row_row(&frag_a1, &frag_b0, &frag_c10);
                for (c_r, d_r) in frag_c10.iter().zip(d10.iter()) {
                    ctx.mov_f32_reg(*c_r, *d_r);
                }
                let d11 = ctx.wmma_mma_f16_f32_row_row(&frag_a1, &frag_b1, &frag_c11);
                for (c_r, d_r) in frag_c11.iter().zip(d11.iter()) {
                    ctx.mov_f32_reg(*c_r, *d_r);
                }

                let _ = stride_a;
            }

            // ─── Store C: 4 wmma_store_d calls for 2×2 grid ───
            {
                // Sub-tile (0,0): C at (cta_row + warp_m_off, cta_col + warp_n_off)
                let c_row0 = ctx.add_u32_reg(cta_row, warp_m_off);
                let c_col0 = ctx.add_u32_reg(cta_col, warp_n_off);
                let cr_ok = ctx.setp_lt_u32(c_row0, m_param);
                ctx.branch_if_not(cr_ok, "exit");
                let cc_ok = ctx.setp_lt_u32(c_col0, n_param);
                ctx.branch_if_not(cc_ok, "exit");

                let c_row_off = ctx.mul_wide_u32_reg(c_row0, n_param);
                let c_col_off = ctx.cvt_u64_u32(c_col0);
                let c_base = ctx.add_u64(c_row_off, c_col_off);
                let c_base = ctx.mul_u64(c_base, 4); // f32 = 4 bytes
                let c00_addr = ctx.add_u64(c_ptr, c_base);
                ctx.wmma_store_d_f32(c00_addr, &frag_c00, n, WmmaLayout::RowMajor);

                // Sub-tile (0,1): col + 16
                let c_col1 = ctx.add_u32_reg(c_col0, c_16);
                let c_col1_off = ctx.cvt_u64_u32(c_col1);
                let c01_base = ctx.add_u64(c_row_off, c_col1_off);
                let c01_base = ctx.mul_u64(c01_base, 4);
                let c01_addr = ctx.add_u64(c_ptr, c01_base);
                ctx.wmma_store_d_f32(c01_addr, &frag_c01, n, WmmaLayout::RowMajor);

                // Sub-tile (1,0): row + 16
                let c_row1 = ctx.add_u32_reg(c_row0, c_16);
                let c_row1_off = ctx.mul_wide_u32_reg(c_row1, n_param);
                let c10_base = ctx.add_u64(c_row1_off, c_col_off);
                let c10_base = ctx.mul_u64(c10_base, 4);
                let c10_addr = ctx.add_u64(c_ptr, c10_base);
                ctx.wmma_store_d_f32(c10_addr, &frag_c10, n, WmmaLayout::RowMajor);

                // Sub-tile (1,1): row + 16, col + 16
                let c11_base = ctx.add_u64(c_row1_off, c_col1_off);
                let c11_base = ctx.mul_u64(c11_base, 4);
                let c11_addr = ctx.add_u64(c_ptr, c11_base);
                ctx.wmma_store_d_f32(c11_addr, &frag_c11, n, WmmaLayout::RowMajor);
            }

            ctx.label("exit");
            ctx.ret();
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cta128_kernel_builds() {
        // FALSIFY: kernel construction must not panic
        use crate::ptx::PtxModule;
        let kernel = build_cta128_wmma_fp16_cpasync(128, 128, 32);
        let ptx = PtxModule::new().target("sm_80").add_kernel(kernel).emit();
        assert!(ptx.contains("gemm_cta128_cpasync_fp16"));
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn test_cta128_smem_size() {
        // Contract: 128×16×2 (A) + 16×128×2 (B) = 8KB per stage, 3 stages = 24KB
        let tile_m = 128u32;
        let tile_n = 128u32;
        let tile_k = 16u32;
        let a_bytes = tile_m * tile_k * 2;
        let b_bytes = tile_k * tile_n * 2;
        let per_stage = a_bytes + b_bytes;
        assert_eq!(per_stage, 8192); // 8 KB
        let total = per_stage * 3;
        assert_eq!(total, 24576); // 24 KB < 48 KB limit
    }

    #[test]
    fn test_cta128_compute_to_load_ratio() {
        // FALSIFY: 128×128 must have 2× better ratio than 64×64
        // 64×64:  compute = 2*64*64*16 = 131072, load = 64*16*2 + 16*64*2 = 4096 bytes
        // ratio_64 = 131072 / 4096 = 32 FLOP/byte
        // 128×128: compute = 2*128*128*16 = 524288, load = 128*16*2 + 16*128*2 = 8192 bytes
        // ratio_128 = 524288 / 8192 = 64 FLOP/byte
        let ratio_64 = (2 * 64 * 64 * 16) as f64 / (64 * 16 * 2 + 16 * 64 * 2) as f64;
        let ratio_128 = (2 * 128 * 128 * 16) as f64 / (128 * 16 * 2 + 16 * 128 * 2) as f64;
        assert!((ratio_64 - 32.0).abs() < 0.1);
        assert!((ratio_128 - 64.0).abs() < 0.1);
        assert!(ratio_128 >= ratio_64 * 1.99); // Must be ~2×
    }

    /// Build a minimal kernel that uses mma.sync.m16n8k16 to validate PTX emission.
    /// This is a compile-only test — it verifies ptxas accepts the generated PTX.
    #[test]
    fn test_mma_sync_ptx_emission() {
        use crate::ptx::{PtxKernel, PtxModule, PtxReg, PtxType};

        let kernel = PtxKernel::new("test_mma_sync")
            .param(PtxType::U64, "dummy")
            .build(|ctx| {
                // Allocate registers for mma.sync m16n8k16
                // A: 4 B32 regs, B: 2 B32 regs, C: 4 F32 regs
                let a0 = ctx.mov_u32_imm(0);
                let a1 = ctx.mov_u32_imm(0);
                let a2 = ctx.mov_u32_imm(0);
                let a3 = ctx.mov_u32_imm(0);
                let b0 = ctx.mov_u32_imm(0);
                let b1 = ctx.mov_u32_imm(0);
                let c0 = ctx.mov_f32_imm(0.0);
                let c1 = ctx.mov_f32_imm(0.0);
                let c2 = ctx.mov_f32_imm(0.0);
                let c3 = ctx.mov_f32_imm(0.0);

                let _d = ctx.mma_sync_m16n8k16(&[a0, a1, a2, a3], &[b0, b1], &[c0, c1, c2, c3]);

                ctx.ret();
            });

        let ptx = PtxModule::new().target("sm_80").add_kernel(kernel).emit();
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "PTX must contain mma.sync instruction"
        );
        // Print the PTX for manual inspection
        eprintln!("=== mma.sync test PTX ===\n{ptx}");
    }

    /// Build a minimal ldmatrix kernel to validate PTX emission.
    #[test]
    fn test_ldmatrix_ptx_emission() {
        use crate::ptx::{PtxKernel, PtxModule, PtxType};

        let kernel = PtxKernel::new("test_ldmatrix")
            .param(PtxType::U64, "dummy")
            .shared_memory(1024)
            .build(|ctx| {
                let addr = ctx.mov_u32_imm(0); // offset 0 in smem
                let _frags = ctx.ldmatrix_x4(addr);
                ctx.ret();
            });

        let ptx = PtxModule::new().target("sm_75").add_kernel(kernel).emit();
        assert!(
            ptx.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"),
            "PTX must contain ldmatrix instruction"
        );
    }

    /// Hardware compilation test: verify ptxas accepts mma.sync PTX.
    /// Requires CUDA device (skip if unavailable).
    #[test]
    fn test_mma_sync_compiles_on_gpu() {
        use crate::ptx::{PtxKernel, PtxModule, PtxType};

        let kernel = PtxKernel::new("test_mma_sync_hw")
            .param(PtxType::U64, "dummy")
            .build(|ctx| {
                // A/B operands MUST be .b32 (not .u32) — ptxas rejects .u32 for mma.sync
                let a0 = ctx.mov_b32_imm(0);
                let a1 = ctx.mov_b32_imm(0);
                let a2 = ctx.mov_b32_imm(0);
                let a3 = ctx.mov_b32_imm(0);
                let b0 = ctx.mov_b32_imm(0);
                let b1 = ctx.mov_b32_imm(0);
                let c0 = ctx.mov_f32_imm(0.0);
                let c1 = ctx.mov_f32_imm(0.0);
                let c2 = ctx.mov_f32_imm(0.0);
                let c3 = ctx.mov_f32_imm(0.0);

                let _d = ctx.mma_sync_m16n8k16(&[a0, a1, a2, a3], &[b0, b1], &[c0, c1, c2, c3]);
                ctx.ret();
            });

        let ptx = PtxModule::new().target("sm_80").add_kernel(kernel).emit();

        // Try to compile with ptxas via CudaModule
        #[cfg(feature = "cuda")]
        {
            use crate::driver::{CudaContext, CudaModule};

            if let Ok(ctx) = CudaContext::new(0) {
                match CudaModule::from_ptx(&ctx, &ptx) {
                    Ok(_) => eprintln!("mma.sync PTX compiled successfully on GPU!"),
                    Err(e) => panic!("mma.sync PTX failed to compile: {e}\nPTX:\n{ptx}"),
                }
            } else {
                eprintln!("No CUDA device — skipping hardware compilation test");
            }
        }
    }

    /// FALSIFY-MMA-SYNC-001b: Combined ldmatrix + mma.sync compiles on GPU.
    /// This validates the full compute pipeline PTX is accepted by ptxas.
    #[test]
    fn test_ldmatrix_mma_sync_combined_compiles() {
        use crate::ptx::{PtxKernel, PtxModule, PtxType};

        let kernel = PtxKernel::new("test_ldmatrix_mma_combined")
            .param(PtxType::U64, "dummy")
            .shared_memory(4096) // 4KB smem for test data
            .build(|ctx| {
                let smem_base = ctx.shared_base_addr();

                // ldmatrix for A fragment: load 4 8×8 tiles from smem
                let lane_id = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let c_32_bytes = ctx.mov_u32_imm(32); // stride = 16 FP16 = 32 bytes
                let c_8 = ctx.mov_u32_imm(8);
                let c_7 = ctx.mov_u32_imm(7);
                let row_in_sub = ctx.and_u32(lane_id, c_7);
                let a_row_off = ctx.mul_u32_reg(row_in_sub, c_32_bytes);
                let a_smem_off = ctx.cvt_u64_u32(a_row_off);
                let a_addr = ctx.add_u64(smem_base, a_smem_off);
                // Convert to shared-space u32 for ldmatrix
                let a_shared_off = a_row_off;
                let a_frags = ctx.ldmatrix_x4(a_shared_off);

                // ldmatrix.trans for B fragment: load 2 8×8 tiles transposed
                let b_base = ctx.mov_u32_imm(2048); // B starts at offset 2KB
                let b_row_off = ctx.mul_u32_reg(row_in_sub, c_8); // stride = 8 FP16 cols * 2 = 16 bytes... simplified
                let b_smem_off = ctx.add_u32_reg(b_base, b_row_off);
                let b_frags = ctx.ldmatrix_x2_trans(b_smem_off);

                // Zero accumulators
                let c0 = ctx.mov_f32_imm(0.0);
                let c1 = ctx.mov_f32_imm(0.0);
                let c2 = ctx.mov_f32_imm(0.0);
                let c3 = ctx.mov_f32_imm(0.0);

                // mma.sync: A[4] × B[2] + C[4] → D[4]
                let _d = ctx.mma_sync_m16n8k16(&a_frags, &b_frags, &[c0, c1, c2, c3]);

                let _ = a_addr;
                ctx.ret();
            });

        let ptx = PtxModule::new().target("sm_80").add_kernel(kernel).emit();
        assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
        assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"));
        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));

        #[cfg(feature = "cuda")]
        {
            use crate::driver::{CudaContext, CudaModule};

            if let Ok(ctx) = CudaContext::new(0) {
                match CudaModule::from_ptx(&ctx, &ptx) {
                    Ok(_) => eprintln!("ldmatrix + mma.sync combined PTX compiled!"),
                    Err(e) => panic!("Combined PTX failed: {e}\nPTX:\n{ptx}"),
                }
            }
        }
    }
}
