//! Tensor Core FlashAttention kernel (FP16 WMMA for Q×K^T)

#![allow(clippy::similar_names)]

use super::AttentionKernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

impl AttentionKernel {
    /// Build Tensor Core FlashAttention using WMMA for Q×K^T
    ///
    /// Key optimization: Replace serial FP32 dot product with FP16 WMMA 16×16×16 tiles.
    /// For head_dim=128, we need 8 WMMA operations per S[i,j] tile (128/16=8).
    ///
    /// Algorithm:
    /// 1. Load Q tile [16×head_dim] to shared memory (FP32→FP16)
    /// 2. Load K tile [16×head_dim] to shared memory (FP32→FP16)
    /// 3. For each 16-element chunk along head_dim:
    ///    - WMMA: S_acc[16×16] += Q_frag[16×16] × K_frag^T[16×16]
    /// 4. Apply scale, online softmax, V multiplication
    ///
    /// Launch config: grid_2d(seq_len/16, num_heads, 32, 1) - one warp per 16×16 Q×K tile
    pub(super) fn build_tensor_core_attention(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let tile_q = 16_u32; // Fixed for WMMA
        let tile_kv = 16_u32; // Fixed for WMMA
        let scale = self.scale;
        let causal = self.causal;

        // Number of WMMA steps to accumulate the full dot product
        // For head_dim=128, n_k_steps = 8
        let n_k_steps = (head_dim + 15) / 16;

        // Shared memory layout:
        // Q tile: 16 × head_dim × 2 bytes (FP16)
        // K tile: 16 × head_dim × 2 bytes (FP16)
        // V tile: 16 × head_dim × 4 bytes (FP32)
        // S tile: 16 × 16 × 4 bytes (FP32 attention scores)
        let q_smem_size = tile_q * head_dim * 2;
        let k_smem_size = tile_kv * head_dim * 2;
        let v_smem_size = tile_kv * head_dim * 4;
        let s_smem_size = tile_q * tile_kv * 4;
        let smem_size = q_smem_size + k_smem_size + v_smem_size + s_smem_size;

        let kernel_name = if causal {
            "flash_attention_tensor_core_causal"
        } else {
            "flash_attention_tensor_core"
        };

        PtxKernel::new(kernel_name)
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "o_ptr")
            .param(PtxType::U32, "seq_len")
            .param(PtxType::U32, "head_dim")
            .param(PtxType::U32, "num_heads")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // WMMA operates at warp level (32 threads cooperatively)
                // Grid: (seq_len/16) x num_heads
                // Block: 32 threads (1 warp)

                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                // Load parameters
                let seq_len_param = ctx.load_param_u32("seq_len");
                let head_dim_param = ctx.load_param_u32("head_dim");
                let num_heads = ctx.load_param_u32("num_heads");
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let o_ptr = ctx.load_param_u64("o_ptr");

                // Block index determines which Q tile we're computing
                let q_block = ctaid_x;
                let head_idx = ctaid_y;

                // PARITY-114 FIX: Compute predicate but DON'T exit early
                // All threads must participate in barriers (WMMA requires full warp)
                let head_valid = ctx.setp_lt_u32(head_idx, num_heads);

                // Calculate head offset
                let head_stride = ctx.mul_u32_reg(seq_len_param, head_dim_param);
                let head_offset = ctx.mul_wide_u32_reg(head_idx, head_stride);
                let head_offset_bytes = ctx.mul_u64(head_offset, 4);

                // Shared memory base addresses (need actual smem pointer, not just offset)
                // For regular loads/stores, u32 offset from smem[0] works
                // For WMMA, we need the actual shared memory address
                let smem_ptr = ctx.shared_base_addr(); // u64 pointer to smem
                let q_smem_base = ctx.mov_u32_imm(0);
                let k_smem_base = ctx.mov_u32_imm(q_smem_size);
                let v_smem_base = ctx.mov_u32_imm(q_smem_size + k_smem_size);
                let s_smem_base = ctx.mov_u32_imm(q_smem_size + k_smem_size + v_smem_size);
                // Pre-compute u64 pointers for WMMA operations
                let q_smem_base_64 = ctx.cvt_u64_u32(q_smem_base);
                let q_smem_ptr = ctx.add_u64(smem_ptr, q_smem_base_64);
                let k_smem_base_64 = ctx.cvt_u64_u32(k_smem_base);
                let k_smem_ptr = ctx.add_u64(smem_ptr, k_smem_base_64);
                let s_smem_base_64 = ctx.cvt_u64_u32(s_smem_base);
                let s_smem_ptr = ctx.add_u64(smem_ptr, s_smem_base_64);

                // Q tile base address
                let tile_16 = ctx.mov_u32_imm(16);
                let q_row_start = ctx.mul_u32_reg(q_block, tile_16);
                let q_tile_offset = ctx.mul_wide_u32_reg(q_row_start, head_dim_param);
                let q_tile_offset_bytes = ctx.mul_u64(q_tile_offset, 4);
                let q_base = ctx.add_u64(q_ptr, head_offset_bytes);
                let q_tile_base = ctx.add_u64(q_base, q_tile_offset_bytes);

                // ===== Load Q tile to shared memory (FP32 → FP16) =====
                // Each of 32 threads loads multiple elements
                // Total: 16 × head_dim elements
                let q_total_elems = ctx.mov_u32_imm(16 * head_dim);
                let elems_per_thread = ctx.div_u32(q_total_elems, 32);
                let my_start = ctx.mul_u32_reg(tid, elems_per_thread);

                let load_idx = ctx.mov_u32_imm(0);
                ctx.label("load_q_loop");
                let load_done = ctx.setp_ge_u32(load_idx, elems_per_thread);
                ctx.branch_if(load_done, "load_q_end");

                let elem_idx = ctx.add_u32_reg(my_start, load_idx);
                let elem_check = ctx.setp_ge_u32(elem_idx, q_total_elems);
                ctx.branch_if(elem_check, "load_q_end");

                // Load from global (FP32)
                let q_global_offset = ctx.mul_wide_u32(elem_idx, 4);
                let q_addr = ctx.add_u64(q_tile_base, q_global_offset);
                let q_val_f32 = ctx.ld_global_f32(q_addr);

                // Convert to FP16 and store to shared
                let q_val_f16 = ctx.cvt_f16_f32(q_val_f32);
                let q_smem_offset = ctx.mul_u32(elem_idx, 2);
                let q_smem_addr = ctx.add_u32_reg(q_smem_base, q_smem_offset);
                ctx.st_shared_f16(q_smem_addr, q_val_f16);

                ctx.add_u32_inplace(load_idx, 1);
                ctx.branch("load_q_loop");
                ctx.label("load_q_end");

                // Initialize output accumulators (16 values per thread for the output row)
                // Each thread in the warp contributes to 16×16 output tile
                // For simplicity, we'll use per-row accumulators
                let o_acc = ctx.mov_f32_imm(0.0);
                let m_prev = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let l_prev = ctx.mov_f32_imm(0.0);

                // Number of KV blocks
                let num_kv_blocks = ctx.div_u32(seq_len_param, 16);
                let kv_block = ctx.mov_u32_imm(0);

                ctx.label("kv_loop_start");
                let kv_done = ctx.setp_ge_u32(kv_block, num_kv_blocks);
                ctx.branch_if(kv_done, "kv_loop_end");

                // Causal masking
                if causal {
                    let causal_skip = ctx.setp_lt_u32(q_block, kv_block);
                    ctx.branch_if(causal_skip, "kv_loop_end");
                }

                // Calculate K, V tile base addresses
                let kv_row_start = ctx.mul_u32_reg(kv_block, tile_16);
                let kv_tile_offset = ctx.mul_wide_u32_reg(kv_row_start, head_dim_param);
                let kv_tile_offset_bytes = ctx.mul_u64(kv_tile_offset, 4);
                let k_base = ctx.add_u64(k_ptr, head_offset_bytes);
                let k_tile_base = ctx.add_u64(k_base, kv_tile_offset_bytes);
                let v_base = ctx.add_u64(v_ptr, head_offset_bytes);
                let v_tile_base = ctx.add_u64(v_base, kv_tile_offset_bytes);

                // ===== Load K tile to shared memory (FP32 → FP16) =====
                let load_k_idx = ctx.mov_u32_imm(0);
                ctx.label("load_k_loop");
                let k_load_done = ctx.setp_ge_u32(load_k_idx, elems_per_thread);
                ctx.branch_if(k_load_done, "load_k_end");

                let k_elem_idx = ctx.add_u32_reg(my_start, load_k_idx);
                let k_elem_check = ctx.setp_ge_u32(k_elem_idx, q_total_elems);
                ctx.branch_if(k_elem_check, "load_k_end");

                let k_global_offset = ctx.mul_wide_u32(k_elem_idx, 4);
                let k_addr = ctx.add_u64(k_tile_base, k_global_offset);
                let k_val_f32 = ctx.ld_global_f32(k_addr);
                let k_val_f16 = ctx.cvt_f16_f32(k_val_f32);
                let k_smem_offset = ctx.mul_u32(k_elem_idx, 2);
                let k_smem_addr = ctx.add_u32_reg(k_smem_base, k_smem_offset);
                ctx.st_shared_f16(k_smem_addr, k_val_f16);

                ctx.add_u32_inplace(load_k_idx, 1);
                ctx.branch("load_k_loop");
                ctx.label("load_k_end");

                // ===== Load V tile to shared memory (FP32) =====
                let load_v_idx = ctx.mov_u32_imm(0);
                ctx.label("load_v_loop");
                let v_load_done = ctx.setp_ge_u32(load_v_idx, elems_per_thread);
                ctx.branch_if(v_load_done, "load_v_end");

                let v_elem_idx = ctx.add_u32_reg(my_start, load_v_idx);
                let v_elem_check = ctx.setp_ge_u32(v_elem_idx, q_total_elems);
                ctx.branch_if(v_elem_check, "load_v_end");

                let v_global_offset = ctx.mul_wide_u32(v_elem_idx, 4);
                let v_addr = ctx.add_u64(v_tile_base, v_global_offset);
                let v_val = ctx.ld_global_f32(v_addr);
                let v_smem_offset = ctx.mul_u32(v_elem_idx, 4);
                let v_smem_addr = ctx.add_u32_reg(v_smem_base, v_smem_offset);
                ctx.st_shared_f32(v_smem_addr, v_val);

                ctx.add_u32_inplace(load_v_idx, 1);
                ctx.branch("load_v_loop");
                ctx.label("load_v_end");

                ctx.bar_sync(0);

                // ===== Compute S = Q × K^T using WMMA =====
                // Initialize S tile accumulator to zero (8 f32 registers)
                let mut frag_c = Vec::with_capacity(8);
                for _ in 0..8 {
                    frag_c.push(ctx.mov_f32_imm(0.0));
                }

                // Loop over head_dim in steps of 16
                let k_step = ctx.mov_u32_imm(0);
                let n_k_steps_reg = ctx.mov_u32_imm(n_k_steps);

                ctx.label("wmma_loop_start");
                let wmma_done = ctx.setp_ge_u32(k_step, n_k_steps_reg);
                ctx.branch_if(wmma_done, "wmma_loop_end");

                // Q fragment address: q_smem_ptr + k_step * 16 * 2 bytes
                let q_frag_offset = ctx.mul_u32(k_step, 32); // 16 elements × 2 bytes
                let q_frag_offset_64 = ctx.cvt_u64_u32(q_frag_offset);
                let q_frag_addr = ctx.add_u64(q_smem_ptr, q_frag_offset_64);
                let frag_a = ctx.wmma_load_a_f16(q_frag_addr, head_dim, WmmaLayout::RowMajor);

                // K fragment address: k_smem_ptr + k_step * 16 * 2 - needs col-major for K^T
                let k_frag_offset = ctx.mul_u32(k_step, 32);
                let k_frag_offset_64 = ctx.cvt_u64_u32(k_frag_offset);
                let k_frag_addr = ctx.add_u64(k_smem_ptr, k_frag_offset_64);
                let frag_b = ctx.wmma_load_b_f16(k_frag_addr, head_dim, WmmaLayout::ColMajor);

                // WMMA MMA: D = A × B + C (accumulates into D fragment)
                let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                // Copy D -> C for next iteration's accumulation (8 f32 registers)
                for i in 0..8 {
                    ctx.mov_f32_reg(frag_c[i], frag_d[i]);
                }

                ctx.add_u32_inplace(k_step, 1);
                ctx.branch("wmma_loop_start");
                ctx.label("wmma_loop_end");

                // Store S tile (accumulated result in D) to shared memory for softmax
                ctx.wmma_store_d_f32(s_smem_ptr, &frag_d, 16, WmmaLayout::RowMajor);

                ctx.bar_sync(1);

                // ===== Apply scale and online softmax =====
                // Each thread handles one element of the 16×16 S tile
                // Thread tid handles element (tid/16, tid%16) for tid < 256
                // We have 32 threads, so each thread handles 8 elements

                let s_idx = ctx.mov_u32_imm(0);
                let loop_limit_8 = ctx.mov_u32_imm(8);
                let elems_256 = ctx.mov_u32_imm(256);
                let step_8 = ctx.mov_u32_imm(8);

                ctx.label("softmax_loop_start");
                let s_idx_check = ctx.setp_ge_u32(s_idx, loop_limit_8);
                ctx.branch_if(s_idx_check, "softmax_loop_end");

                // Calculate which S element this iteration handles
                let s_elem = ctx.mad_lo_u32(tid, step_8, s_idx);
                let s_elem_check = ctx.setp_ge_u32(s_elem, elems_256);
                ctx.branch_if(s_elem_check, "softmax_next");

                // Load S[i,j] from shared memory
                let s_offset = ctx.mul_u32(s_elem, 4);
                let s_addr = ctx.add_u32_reg(s_smem_base, s_offset);
                let s_val = ctx.ld_shared_f32(s_addr);

                // Apply scale
                let scale_reg = ctx.mov_f32_imm(scale);
                let s_scaled = ctx.mul_f32(s_val, scale_reg);

                // Online softmax update (simplified - each thread maintains local max/sum)
                let m_new = ctx.max_f32(m_prev, s_scaled);
                let m_diff = ctx.sub_f32(m_prev, m_new);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let m_diff_scaled = ctx.mul_f32(m_diff, log2_e);
                let scale_factor = ctx.ex2_f32(m_diff_scaled);

                let s_shifted = ctx.sub_f32(s_scaled, m_new);
                let s_shifted_scaled = ctx.mul_f32(s_shifted, log2_e);
                let p_val = ctx.ex2_f32(s_shifted_scaled);

                let l_scaled = ctx.mul_f32(scale_factor, l_prev);
                let l_new = ctx.add_f32(l_scaled, p_val);

                // Store scaled attention weight back
                ctx.st_shared_f32(s_addr, p_val);

                ctx.mov_f32_reg(m_prev, m_new);
                ctx.mov_f32_reg(l_prev, l_new);

                ctx.label("softmax_next");
                ctx.add_u32_inplace(s_idx, 1);
                ctx.branch("softmax_loop_start");
                ctx.label("softmax_loop_end");

                ctx.bar_sync(2);

                // ===== Compute O += softmax(S) × V =====
                // Load attention weights and multiply with V
                // This is another GEMM: P[16×16] × V[16×head_dim] = O_update[16×head_dim]
                // For simplicity, we'll use the scalar path for V multiplication
                // (Tensor Core V multiplication would require additional WMMA calls)

                let v_col = ctx.rem_u32(tid, head_dim);
                let v_row_idx = ctx.mov_u32_imm(0);

                ctx.label("v_loop_start");
                let v_loop_done = ctx.setp_ge_u32(v_row_idx, tile_16);
                ctx.branch_if(v_loop_done, "v_loop_end");

                // Load attention weight P[row, v_row_idx]
                let p_idx_base = ctx.mul_u32_reg(v_row_idx, tile_16);
                let p_offset = ctx.mul_u32(p_idx_base, 4);
                let p_addr = ctx.add_u32_reg(s_smem_base, p_offset);
                let p_weight = ctx.ld_shared_f32(p_addr);

                // Load V[v_row_idx, v_col]
                let v_idx = ctx.mad_lo_u32(v_row_idx, head_dim_param, v_col);
                let v_offset = ctx.mul_u32(v_idx, 4);
                let v_elem_addr = ctx.add_u32_reg(v_smem_base, v_offset);
                let v_elem = ctx.ld_shared_f32(v_elem_addr);

                // Accumulate: o_acc += p_weight * v_elem
                ctx.fma_f32_inplace(o_acc, p_weight, v_elem);

                ctx.add_u32_inplace(v_row_idx, 1);
                ctx.branch("v_loop_start");
                ctx.label("v_loop_end");

                ctx.bar_sync(3);

                ctx.add_u32_inplace(kv_block, 1);
                ctx.branch("kv_loop_start");
                ctx.label("kv_loop_end");

                // PARITY-114 FIX: Bounds check HERE (after all threads finished barriers)
                // Only threads with valid heads store to O
                ctx.branch_if_not(head_valid, "exit");

                // ===== Normalize and store output =====
                let o_normalized = ctx.div_f32(o_acc, l_prev);

                // Calculate output address
                let o_base = ctx.add_u64(o_ptr, head_offset_bytes);
                let tid_div_hd = ctx.div_u32(tid, head_dim);
                let o_row = ctx.mad_lo_u32(q_block, tile_16, tid_div_hd);
                let o_col = ctx.rem_u32(tid, head_dim);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let o_idx = ctx.mad_lo_u32(o_row, head_dim_reg, o_col);
                let o_offset = ctx.mul_wide_u32(o_idx, 4);
                let o_addr = ctx.add_u64(o_base, o_offset);

                ctx.st_global_f32(o_addr, o_normalized);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
