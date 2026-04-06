//! Standard FP32 FlashAttention kernel (serial dot product baseline)

use super::AttentionKernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

impl AttentionKernel {
    pub(super) fn build_flash_attention(&self) -> PtxKernel {
        // FlashAttention-style tiled attention
        // Per Dao et al. - never materialize full N×N matrix
        let head_dim = self.head_dim;
        let tile_q = self.tile_q;
        let tile_kv = self.tile_kv;
        let scale = self.scale;
        let causal = self.causal;

        // Shared memory for Q, K, V tiles
        let smem_size = (tile_q * head_dim + tile_kv * head_dim * 2) * 4;

        let kernel_name = if causal {
            "flash_attention_causal"
        } else {
            "flash_attention"
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
                // Thread and block indices
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let _ntid = ctx.special_reg(PtxReg::NtidX);

                // Load parameters
                let seq_len_param = ctx.load_param_u32("seq_len");
                let head_dim_param = ctx.load_param_u32("head_dim");
                let num_heads = ctx.load_param_u32("num_heads");
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let o_ptr = ctx.load_param_u64("o_ptr");

                // ctaid_x = Q block index, ctaid_y = head index
                let q_block = ctaid_x;
                let head_idx = ctaid_y;

                // PARITY-114 FIX: Compute predicate but DON'T exit early
                // All threads must participate in barriers
                let head_valid = ctx.setp_lt_u32(head_idx, num_heads);

                // Calculate head offset (head_idx * seq_len * head_dim)
                let head_stride = ctx.mul_u32_reg(seq_len_param, head_dim_param);
                let head_offset = ctx.mul_wide_u32_reg(head_idx, head_stride);
                let head_offset_bytes = ctx.mul_u64(head_offset, 4);

                // Q tile base address for this block
                let tile_q_imm = ctx.mov_u32_imm(tile_q);
                let q_row_start = ctx.mul_u32_reg(q_block, tile_q_imm);
                let q_tile_offset = ctx.mul_wide_u32_reg(q_row_start, head_dim_param);
                let q_tile_offset_bytes = ctx.mul_u64(q_tile_offset, 4);
                let q_base = ctx.add_u64(q_ptr, head_offset_bytes);
                let q_tile_base = ctx.add_u64(q_base, q_tile_offset_bytes);

                // ===== Initialize output accumulator and softmax stats =====
                // Each thread handles one position in the output
                let local_row = ctx.div_u32(tid, head_dim);
                let local_col = ctx.rem_u32(tid, head_dim);

                // PARITY-114 FIX: Compute predicate but DON'T exit early
                // This handles launch configs with more threads than tile_q * head_dim
                let tile_q_check = ctx.mov_u32_imm(tile_q);
                let thread_valid = ctx.setp_lt_u32(local_row, tile_q_check);

                // Initialize output accumulator to 0
                let o_acc = ctx.mov_f32_imm(0.0);
                // Running max for online softmax
                let m_prev = ctx.mov_f32_imm(f32::NEG_INFINITY);
                // Running sum of exp
                let l_prev = ctx.mov_f32_imm(0.0);

                // Calculate number of KV blocks
                let tile_kv_imm = ctx.mov_u32_imm(tile_kv);
                let num_kv_blocks = ctx.div_u32(seq_len_param, tile_kv);

                // ===== Pre-compute element offset (needed for output store after loop) =====
                // This must be computed BEFORE the loop, not inside it
                let local_row_64 = ctx.cvt_u64_u32(local_row);
                let local_col_64 = ctx.cvt_u64_u32(local_col);
                let head_dim_64 = ctx.cvt_u64_u32(head_dim_param);
                let q_elem_offset = ctx.mul_u64_reg(local_row_64, head_dim_64);
                let q_elem_offset_full = ctx.add_u64(q_elem_offset, local_col_64);
                let q_elem_offset_bytes = ctx.mul_u64(q_elem_offset_full, 4);

                // Shared memory base addresses (32-bit constants for shared memory addressing)
                let k_smem_base = tile_q * head_dim * 4;
                let v_smem_base = (tile_q * head_dim + tile_kv * head_dim) * 4;

                // Loop counter
                let kv_block = ctx.mov_u32_imm(0);

                ctx.label("kv_loop_start");

                // Check if we've processed all KV blocks
                let kv_done = ctx.setp_ge_u32(kv_block, num_kv_blocks);
                ctx.branch_if(kv_done, "kv_loop_end");

                // Causal masking: skip KV blocks that are entirely after current Q block
                // Use setp_lt and flip: if q_block < kv_block, skip
                if causal {
                    let causal_skip = ctx.setp_lt_u32(q_block, kv_block);
                    ctx.branch_if(causal_skip, "kv_loop_end");
                }

                // Calculate K, V tile base addresses
                let kv_row_start = ctx.mul_u32_reg(kv_block, tile_kv_imm);
                let kv_tile_offset = ctx.mul_wide_u32_reg(kv_row_start, head_dim_param);
                let kv_tile_offset_bytes = ctx.mul_u64(kv_tile_offset, 4);
                let k_base = ctx.add_u64(k_ptr, head_offset_bytes);
                let k_tile_base = ctx.add_u64(k_base, kv_tile_offset_bytes);
                let v_base = ctx.add_u64(v_ptr, head_offset_bytes);
                let v_tile_base = ctx.add_u64(v_base, kv_tile_offset_bytes);

                // ===== Load Q tile to shared memory =====
                // Q tile: tile_q * head_dim elements, one per thread (1:1 mapping)
                let q_addr = ctx.add_u64(q_tile_base, q_elem_offset_bytes);
                let q_val = ctx.ld_global_f32(q_addr);
                let q_smem_offset = ctx.mul_u32(tid, 4);
                ctx.st_shared_f32(q_smem_offset, q_val);

                // ===== Load K tile to shared memory (strided cooperative loading) =====
                // GH-32 FIX: K tile has tile_kv * head_dim elements which may exceed
                // thread count (tile_q * head_dim). Use strided loop to load all elements.
                let kv_total_reg = ctx.mov_u32_imm(tile_kv * head_dim);
                let k_load_base = ctx.mov_u32_imm(0);

                ctx.label("k_coop_load");
                let k_elem_idx = ctx.add_u32_reg(k_load_base, tid);
                let k_in_bounds = ctx.setp_lt_u32(k_elem_idx, kv_total_reg);
                ctx.branch_if_not(k_in_bounds, "k_coop_load_end");

                let k_offset = ctx.mul_wide_u32(k_elem_idx, 4);
                let k_addr = ctx.add_u64(k_tile_base, k_offset);
                let k_val = ctx.ld_global_f32(k_addr);
                let k_smem_base_u32 = ctx.mov_u32_imm(k_smem_base);
                let k_elem_bytes = ctx.mul_u32(k_elem_idx, 4);
                let k_smem_off = ctx.add_u32_reg(k_smem_base_u32, k_elem_bytes);
                ctx.st_shared_f32(k_smem_off, k_val);

                ctx.add_u32_inplace(k_load_base, tile_q * head_dim);
                ctx.branch("k_coop_load");
                ctx.label("k_coop_load_end");

                // ===== Load V tile to shared memory (strided cooperative loading) =====
                let v_load_base = ctx.mov_u32_imm(0);

                ctx.label("v_coop_load");
                let v_elem_idx = ctx.add_u32_reg(v_load_base, tid);
                let v_in_bounds = ctx.setp_lt_u32(v_elem_idx, kv_total_reg);
                ctx.branch_if_not(v_in_bounds, "v_coop_load_end");

                let v_offset = ctx.mul_wide_u32(v_elem_idx, 4);
                let v_addr = ctx.add_u64(v_tile_base, v_offset);
                let v_val = ctx.ld_global_f32(v_addr);
                let v_smem_base_u32 = ctx.mov_u32_imm(v_smem_base);
                let v_elem_bytes = ctx.mul_u32(v_elem_idx, 4);
                let v_smem_off = ctx.add_u32_reg(v_smem_base_u32, v_elem_bytes);
                ctx.st_shared_f32(v_smem_off, v_val);

                ctx.add_u32_inplace(v_load_base, tile_q * head_dim);
                ctx.branch("v_coop_load");
                ctx.label("v_coop_load_end");

                ctx.bar_sync(0);

                // ===== K-row loop: iterate over ALL K rows in the tile =====
                // GH-32 FIX: Each thread computes O[local_row, local_col] by iterating
                // over all tile_kv K rows, computing attention scores and accumulating
                // weighted V values using online softmax (Dao et al.).
                //
                // Previously, local_col was conflated as both the K-row index AND the
                // output dimension — producing one attention score per thread with no
                // cross-thread reduction. Now local_col is purely the output dimension
                // and k_row iterates over all K rows.
                let k_row = ctx.mov_u32_imm(0);
                let tile_kv_reg = ctx.mov_u32_imm(tile_kv);

                ctx.label("k_row_loop_start");
                let k_row_done = ctx.setp_ge_u32(k_row, tile_kv_reg);
                ctx.branch_if(k_row_done, "k_row_loop_end");

                // Causal masking: skip K rows where global K position > global Q position
                if causal {
                    let q_global_row = ctx.add_u32_reg(q_row_start, local_row);
                    let k_global_row = ctx.add_u32_reg(kv_row_start, k_row);
                    let causal_mask = ctx.setp_lt_u32(q_global_row, k_global_row);
                    ctx.branch_if(causal_mask, "k_row_next");
                }

                // Compute S = Q[local_row] · K[k_row] (dot product over head_dim)
                let s_acc = ctx.mov_f32_imm(0.0);
                let d_idx = ctx.mov_u32_imm(0);
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);

                ctx.label("dot_loop_start");
                let d_done = ctx.setp_ge_u32(d_idx, head_dim_param);
                ctx.branch_if(d_done, "dot_loop_end");

                // Load Q[local_row, d_idx] from shared memory
                let q_row_offset = ctx.mul_u32_reg(local_row, head_dim_u32);
                let q_elem_smem = ctx.add_u32_reg(q_row_offset, d_idx);
                let q_elem_smem_bytes = ctx.mul_u32(q_elem_smem, 4);
                let q_dot_val = ctx.ld_shared_f32(q_elem_smem_bytes);

                // Load K[k_row, d_idx] from shared memory
                // GH-32 FIX: Use k_row (loop variable) as K row index, NOT local_col
                let k_row_offset = ctx.mul_u32_reg(k_row, head_dim_u32);
                let k_elem_smem = ctx.add_u32_reg(k_row_offset, d_idx);
                let k_elem_smem_bytes = ctx.mul_u32(k_elem_smem, 4);
                let k_smem_base_loop = ctx.mov_u32_imm(k_smem_base);
                let k_elem_smem_full = ctx.add_u32_reg(k_smem_base_loop, k_elem_smem_bytes);
                let k_dot_val = ctx.ld_shared_f32(k_elem_smem_full);

                // Accumulate Q[i,d] * K[j,d]
                ctx.fma_f32_inplace(s_acc, q_dot_val, k_dot_val);

                ctx.add_u32_inplace(d_idx, 1);
                ctx.branch("dot_loop_start");
                ctx.label("dot_loop_end");

                // Apply scale factor
                let scale_reg = ctx.mov_f32_imm(scale);
                let s_scaled = ctx.mul_f32(s_acc, scale_reg);

                // Online softmax update
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

                // Update output accumulator
                // o_new = (scale_factor * l_prev * o_prev + p * V[k_row, local_col]) / l_new
                let o_scaled = ctx.mul_f32(o_acc, scale_factor);
                let o_weighted = ctx.mul_f32(o_scaled, l_prev);

                // Load V[k_row, local_col] from shared memory
                // GH-32 FIX: Correct V indexing — V[k_row, local_col] for output dim local_col
                // Previously used V[local_col, last_d_idx] which was wrong.
                let v_row_offset = ctx.mul_u32_reg(k_row, head_dim_u32);
                let v_elem_idx = ctx.add_u32_reg(v_row_offset, local_col);
                let v_elem_smem_bytes = ctx.mul_u32(v_elem_idx, 4);
                let v_smem_base_loop = ctx.mov_u32_imm(v_smem_base);
                let v_elem_smem_full = ctx.add_u32_reg(v_smem_base_loop, v_elem_smem_bytes);
                let v_out_val = ctx.ld_shared_f32(v_elem_smem_full);

                let pv = ctx.mul_f32(p_val, v_out_val);
                let o_sum = ctx.add_f32(o_weighted, pv);
                let o_new = ctx.div_f32(o_sum, l_new);

                // Update running stats for next iteration
                ctx.mov_f32_reg(m_prev, m_new);
                ctx.mov_f32_reg(l_prev, l_new);
                ctx.mov_f32_reg(o_acc, o_new);

                ctx.label("k_row_next");
                ctx.add_u32_inplace(k_row, 1);
                ctx.branch("k_row_loop_start");
                ctx.label("k_row_loop_end");

                ctx.bar_sync(1);

                // Increment KV block counter and loop back - IN-PLACE UPDATE
                ctx.add_u32_inplace(kv_block, 1);
                ctx.branch("kv_loop_start");

                ctx.label("kv_loop_end");

                // PARITY-114 FIX: Bounds check HERE (after all threads finished barriers)
                // Only threads with valid output coordinates store to O
                ctx.branch_if_not(head_valid, "exit");
                ctx.branch_if_not(thread_valid, "exit");

                // ===== Store output =====
                // Calculate output address
                let o_base = ctx.add_u64(o_ptr, head_offset_bytes);
                let o_tile_offset = ctx.mul_wide_u32_reg(q_row_start, head_dim_param);
                let o_tile_offset_bytes = ctx.mul_u64(o_tile_offset, 4);
                let o_tile_base = ctx.add_u64(o_base, o_tile_offset_bytes);
                let o_addr = ctx.add_u64(o_tile_base, q_elem_offset_bytes);

                // Store accumulated output (o_acc is always valid, even if loop never ran)
                ctx.st_global_f32(o_addr, o_acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
