//! Prefill attention, KV scatter, cache management, and warmup.
//!
//! Contains cuBLAS strided batched GEMM attention, fused attention,
//! FP16/FP8 weight cache management, and warmup routines.

use super::super::super::*;

impl CudaExecutor {
    // ========================================================================
    // PMAT-032: Parallel Prefill Attention via cuBLAS Strided Batched GEMM
    // ========================================================================

    /// PMAT-032: Inline PTX for causal mask + row-wise softmax.
    ///
    /// Grid: (num_heads, M, 1), Block: (32, 1, 1)
    /// Each block processes one (head, token) pair.
    ///
    /// Params: scores_ptr, M, total_len, base_seq_len, num_heads
    /// scores layout: [num_heads, M, total_len] row-major
    ///
    /// For token i (ctaid.y), valid positions are j < base_seq_len + i + 1.
    /// Sets invalid positions to -inf, then computes in-place softmax.
    const CAUSAL_MASK_SOFTMAX_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry causal_mask_softmax(
    .param .u64 param_scores,
    .param .u32 param_m,
    .param .u32 param_total_len,
    .param .u32 param_base_seq_len,
    .param .u32 param_num_heads
) {
    .reg .u64 %rd<8>;
    .reg .u32 %r<16>;
    .reg .f32 %f<8>;
    .reg .pred %p<4>;

    // head_idx = ctaid.x, token_idx = ctaid.y, tid = tid.x
    ld.param.u64 %rd0, [param_scores];
    ld.param.u32 %r0, [param_m];
    ld.param.u32 %r1, [param_total_len];
    ld.param.u32 %r2, [param_base_seq_len];
    ld.param.u32 %r3, [param_num_heads];

    mov.u32 %r4, %ctaid.x;   // head_idx
    mov.u32 %r5, %ctaid.y;   // token_idx
    mov.u32 %r6, %tid.x;     // lane_id
    mov.u32 %r7, %ntid.x;    // block_size (32)

    // valid_len = base_seq_len + token_idx + 1
    add.u32 %r8, %r2, %r5;
    add.u32 %r8, %r8, 1;

    // row_offset = (head_idx * M + token_idx) * total_len
    mad.lo.u32 %r9, %r4, %r0, %r5;
    mul.lo.u32 %r9, %r9, %r1;

    // row_ptr = scores + row_offset * 4
    cvt.u64.u32 %rd1, %r9;
    shl.b64 %rd1, %rd1, 2;
    add.u64 %rd1, %rd0, %rd1;

    // === Pass 1: Apply causal mask + find row max ===
    mov.f32 %f0, 0fFF800000;  // max = -inf

    mov.u32 %r10, %r6;  // j = tid
LOOP_MAX:
    setp.ge.u32 %p0, %r10, %r1;  // j >= total_len?
    @%p0 bra DONE_MAX;

    // addr = row_ptr + j * 4
    cvt.u64.u32 %rd2, %r10;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    // If j >= valid_len, mask to -inf
    setp.ge.u32 %p1, %r10, %r8;
    @%p1 bra STORE_NEG_INF;

    // Valid position: load and track max
    ld.global.f32 %f1, [%rd2];
    max.f32 %f0, %f0, %f1;
    bra NEXT_MAX;

STORE_NEG_INF:
    mov.f32 %f1, 0fFF800000;
    st.global.f32 [%rd2], %f1;

NEXT_MAX:
    add.u32 %r10, %r10, %r7;  // j += blockDim
    bra LOOP_MAX;

DONE_MAX:
    // Warp reduce max (butterfly)
    shfl.sync.bfly.b32 %f1, %f0, 16, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 8, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 4, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 2, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    shfl.sync.bfly.b32 %f1, %f0, 1, 31, 0xffffffff;
    max.f32 %f0, %f0, %f1;
    // %f0 = row_max (broadcast to all lanes via shfl)

    // === Pass 2: exp(x - max) and sum ===
    mov.f32 %f2, 0f00000000;  // sum = 0

    mov.u32 %r10, %r6;
LOOP_EXP:
    setp.ge.u32 %p0, %r10, %r1;
    @%p0 bra DONE_EXP;

    cvt.u64.u32 %rd2, %r10;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    ld.global.f32 %f1, [%rd2];
    sub.f32 %f1, %f1, %f0;       // x - max
    // Clamp to prevent overflow: if x-max < -88, set to 0
    mov.f32 %f3, 0fC2B00000;     // -88.0
    setp.lt.f32 %p2, %f1, %f3;
    @%p2 mov.f32 %f1, 0fC2B00000;

    // exp approximation using ex2 (exp2(x * log2(e)))
    mul.f32 %f1, %f1, 0f3FB8AA3B;  // x * log2(e) = x * 1.4426950408889634
    ex2.approx.f32 %f1, %f1;
    st.global.f32 [%rd2], %f1;
    add.f32 %f2, %f2, %f1;

    add.u32 %r10, %r10, %r7;
    bra LOOP_EXP;

DONE_EXP:
    // Warp reduce sum
    shfl.sync.bfly.b32 %f1, %f2, 16, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 8, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 4, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 2, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    shfl.sync.bfly.b32 %f1, %f2, 1, 31, 0xffffffff;
    add.f32 %f2, %f2, %f1;
    // %f2 = sum_exp

    // inv_sum = 1.0 / sum_exp
    rcp.approx.f32 %f2, %f2;

    // === Pass 3: normalize ===
    mov.u32 %r10, %r6;
LOOP_NORM:
    setp.ge.u32 %p0, %r10, %r1;
    @%p0 bra DONE_NORM;

    cvt.u64.u32 %rd2, %r10;
    shl.b64 %rd2, %rd2, 2;
    add.u64 %rd2, %rd1, %rd2;

    ld.global.f32 %f1, [%rd2];
    mul.f32 %f1, %f1, %f2;
    st.global.f32 [%rd2], %f1;

    add.u32 %r10, %r10, %r7;
    bra LOOP_NORM;

DONE_NORM:
    ret;
}
"#;

    /// PMAT-032: Ensure attention score scratch buffer is large enough
    fn ensure_attn_score_scratch(&mut self, size: usize) -> Result<(), GpuError> {
        if self.prefill_attn_scores_size >= size {
            return Ok(());
        }
        self.prefill_attn_scores = Some(GpuBuffer::new(&self.context, size)?);
        self.prefill_attn_scores_size = size;
        Ok(())
    }

    /// PMAT-032: Bulk scatter M tokens' K/V to cache positions [cache_len..cache_len+M]
    ///
    /// Uses stream-ordered async D2D copies on self.stream, so cuBLAS HGEMM writes
    /// (which also run on self.stream) are guaranteed to complete before the copies
    /// read the K/V buffers. No explicit stream.synchronize() needed.
    fn bulk_scatter_kv(
        &mut self,
        layer_idx: usize,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        m: u32,
        kv_dim: u32,
    ) -> Result<usize, GpuError> {
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);
        let stream_handle = self.stream.raw();

        let new_len = cache_len + m as usize;
        if new_len > max_len {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PMAT-032: KV cache overflow - max_len={max_len}, trying to add {m} at position {cache_len}"
            )));
        }

        let k_key = format!("kv_{layer_idx}_k");
        let v_key = format!("kv_{layer_idx}_v");

        // Scatter K and V for all M tokens using stream-ordered async copies
        for seq_idx in 0..m as usize {
            let position = cache_len + seq_idx;

            // Scatter K
            {
                let k_cache = self.kv_cache_gpu.get_mut(&k_key).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-032: K cache not initialized for layer {layer_idx}"
                    ))
                })?;
                for kv_head in 0..num_kv_heads {
                    let src_offset = kv_head * head_dim + seq_idx * kv_dim as usize;
                    let dst_offset = kv_head * (max_len * head_dim) + position * head_dim;
                    // SAFETY: k_buf_ptr is valid GPU buffer [M, kv_dim], offsets bounded,
                    // src_view lifetime managed manually (forget below)
                    let src_view = unsafe {
                        GpuBuffer::<f32>::from_raw_parts(
                            k_buf_ptr + (src_offset * std::mem::size_of::<f32>()) as u64,
                            head_dim,
                        )
                    };
                    // SAFETY: Both buffers valid until stream sync, stream_handle is valid
                    unsafe {
                        k_cache.copy_from_buffer_at_async_raw(
                            &src_view,
                            dst_offset,
                            0,
                            head_dim,
                            stream_handle,
                        )?;
                    }
                    std::mem::forget(src_view);
                }
            }

            // Scatter V
            {
                let v_cache = self.kv_cache_gpu.get_mut(&v_key).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-032: V cache not initialized for layer {layer_idx}"
                    ))
                })?;
                for kv_head in 0..num_kv_heads {
                    let src_offset = kv_head * head_dim + seq_idx * kv_dim as usize;
                    let dst_offset = kv_head * (max_len * head_dim) + position * head_dim;
                    // SAFETY: v_buf_ptr is valid GPU buffer [M, kv_dim], offsets bounded
                    let src_view = unsafe {
                        GpuBuffer::<f32>::from_raw_parts(
                            v_buf_ptr + (src_offset * std::mem::size_of::<f32>()) as u64,
                            head_dim,
                        )
                    };
                    // SAFETY: Both buffers valid until stream sync, stream_handle is valid
                    unsafe {
                        v_cache.copy_from_buffer_at_async_raw(
                            &src_view,
                            dst_offset,
                            0,
                            head_dim,
                            stream_handle,
                        )?;
                    }
                    std::mem::forget(src_view);
                }
            }
        }

        // Update cache length once for all M tokens
        self.kv_cache_lengths.insert(layer_idx, new_len);

        Ok(cache_len)
    }

    /// PMAT-032: Parallel prefill attention using cuBLAS strided batched GEMM.
    ///
    /// Replaces M sequential `incremental_attention_into_for_capture` calls with:
    /// 1. Bulk KV scatter (M tokens at once)
    /// 2. QK^T via cuBLAS strided batched GEMM (handles GQA)
    /// 3. Causal mask + softmax (single PTX kernel)
    /// 4. Attn × V via cuBLAS strided batched GEMM
    ///
    /// For M=8, 28 layers: 672 kernel launches → 196 (3.4× reduction).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_attention_cublas(
        &mut self,
        layer_idx: usize,
        _q_buf: &GpuBuffer<f32>,
        _k_buf: &GpuBuffer<f32>,
        _v_buf: &GpuBuffer<f32>,
        _attn_out_buf: &GpuBuffer<f32>,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        m: u32,
        q_dim: u32,
        kv_dim: u32,
    ) -> Result<(), GpuError> {
        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let max_len = self.kv_cache_max_len;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let cache_len = self.kv_cache_lengths.get(&layer_idx).copied().unwrap_or(0);

        // PMAT-052: Zero-copy path for fresh prefill (cache_len == 0).
        //
        // Five-Whys: c=1 TTFT has ~14,000 D2D copies from bulk_scatter_kv.
        // Why? M tokens × num_kv_heads × 2 (K/V) per layer × 28 layers.
        // Why? Packed K/V [M, kv_dim] needs rearranging to [num_kv_heads, max_len, head_dim].
        // Why? cuBLAS reads cache with lda=head_dim.
        // Fix: Read K/V directly from packed buffer (lda=kv_dim). Scatter to cache afterward
        // via single PTX kernel per layer (28 launches replaces 14,000 D2D copies).
        if cache_len == 0 {
            let total_len = m as usize;

            // PMAT-069: Fused prefill attention — 1 kernel/layer replaces 5 cuBLAS + PTX launches.
            // Opt-in: FUSED_PREFILL_ATTN=1 (currently 2.8x slower than cuBLAS due to
            // single-warp-per-head design — needs multi-block tiling to be competitive).
            let use_fused_attn = std::env::var("FUSED_PREFILL_ATTN")
                .map(|v| v == "1")
                .unwrap_or(false);

            if use_fused_attn {
                return self.prefill_attention_fused(
                    layer_idx,
                    q_buf_ptr,
                    k_buf_ptr,
                    v_buf_ptr,
                    attn_out_ptr,
                    m,
                    q_dim,
                    kv_dim,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    heads_per_kv,
                    max_len,
                );
            }

            // Score scratch buffer
            self.ensure_cublas()?;
            let score_size = num_heads * m as usize * total_len;
            self.ensure_attn_score_scratch(score_size)?;
            let score_ptr = self
                .prefill_attn_scores
                .as_ref()
                .expect("score scratch just allocated")
                .as_ptr();

            // 1. QK^T — reading K directly from packed buffer (lda=kv_dim)
            let handle = self.cublas_handle.as_ref().expect("cublas initialized");

            for kv_group in 0..num_kv_heads {
                let first_q_head = kv_group * heads_per_kv;
                let k_head_ptr =
                    k_buf_ptr + (kv_group * head_dim * std::mem::size_of::<f32>()) as u64;
                let q_head_ptr =
                    q_buf_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;
                let s_head_ptr = score_ptr
                    + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;

                handle.gemm_f32_strided_batched(
                    trueno_gpu::driver::GemmOp::Trans,   // K^T
                    trueno_gpu::driver::GemmOp::NoTrans, // Q
                    total_len as i32,                    // m (rows of C)
                    m as i32,                            // n (cols of C)
                    head_dim as i32,                     // k
                    scale,                               // alpha = 1/sqrt(d)
                    k_head_ptr,                          // A = packed K
                    kv_dim as i32,                       // lda = kv_dim (NOT head_dim)
                    0,                                   // stride_a = 0 (shared K)
                    q_head_ptr,                          // B = Q
                    q_dim as i32,                        // ldb = q_dim
                    head_dim as i64,                     // stride_b = head_dim
                    0.0,                                 // beta
                    s_head_ptr,                          // C = score
                    total_len as i32,                    // ldc
                    (m as usize * total_len) as i64,     // stride_c
                    heads_per_kv as i32,                 // batch
                )?;
            }

            // 2. Causal mask + softmax
            if !self.modules.contains_key("causal_mask_softmax") {
                let module = self.compile_ptx(Self::CAUSAL_MASK_SOFTMAX_PTX)?;
                self.modules
                    .insert("causal_mask_softmax".to_string(), module);
            }

            {
                let module = self
                    .modules
                    .get_mut("causal_mask_softmax")
                    .expect("just inserted");
                let config = LaunchConfig {
                    grid: (num_heads as u32, m, 1),
                    block: (32, 1, 1),
                    shared_mem: 0,
                };

                let mut s_ptr = score_ptr;
                let mut m_val = m;
                let mut tl_val = total_len as u32;
                let mut base_val = 0u32; // No prior cache
                let mut nh_val = num_heads as u32;

                // SAFETY: score buffer valid, dimensions verified
                unsafe {
                    self.stream.launch_kernel(
                        module,
                        "causal_mask_softmax",
                        &config,
                        &mut [
                            std::ptr::from_mut(&mut s_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut tl_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut base_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut nh_val) as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            // 3. Attn × V — reading V directly from packed buffer (lda=kv_dim)
            let handle = self.cublas_handle.as_ref().expect("cublas initialized");

            for kv_group in 0..num_kv_heads {
                let first_q_head = kv_group * heads_per_kv;
                let v_head_ptr =
                    v_buf_ptr + (kv_group * head_dim * std::mem::size_of::<f32>()) as u64;
                let p_head_ptr = score_ptr
                    + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;
                let o_head_ptr =
                    attn_out_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;

                handle.gemm_f32_strided_batched(
                    trueno_gpu::driver::GemmOp::NoTrans, // V
                    trueno_gpu::driver::GemmOp::NoTrans, // P
                    head_dim as i32,                     // m (rows of C)
                    m as i32,                            // n (cols of C)
                    total_len as i32,                    // k (inner dim)
                    1.0,                                 // alpha
                    v_head_ptr,                          // A = packed V
                    kv_dim as i32,                       // lda = kv_dim (NOT head_dim)
                    0,                                   // stride_a = 0 (shared V)
                    p_head_ptr,                          // B = P (softmax output)
                    total_len as i32,                    // ldb
                    (m as usize * total_len) as i64,     // stride_b
                    0.0,                                 // beta
                    o_head_ptr,                          // C = output
                    q_dim as i32,                        // ldc = q_dim
                    head_dim as i64,                     // stride_c
                    heads_per_kv as i32,                 // batch
                )?;
            }

            // 4. Scatter K/V from packed buffer to single KV cache via PTX kernel
            let k_key = format!("kv_{layer_idx}_k");
            let v_key = format!("kv_{layer_idx}_v");
            let k_cache_ptr = self
                .kv_cache_gpu
                .get(&k_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?
                .as_ptr();
            let v_cache_ptr = self
                .kv_cache_gpu
                .get(&v_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?
                .as_ptr();

            if !self.modules.contains_key("scatter_packed_kv") {
                let module = self.compile_ptx(Self::SCATTER_PACKED_KV_PTX)?;
                self.modules.insert("scatter_packed_kv".to_string(), module);
            }

            let module = self
                .modules
                .get_mut("scatter_packed_kv")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (m, num_kv_heads as u32, 1),
                block: (head_dim as u32, 1, 1),
                shared_mem: 0,
            };

            let mut k_src = k_buf_ptr;
            let mut v_src = v_buf_ptr;
            let mut k_dst = k_cache_ptr;
            let mut v_dst = v_cache_ptr;
            let mut kv_dim_val = kv_dim;
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;

            // SAFETY: All GPU buffers valid, grid/block dimensions verified
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "scatter_packed_kv",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut k_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut kv_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }

            // Update single KV cache length
            self.kv_cache_lengths.insert(layer_idx, m as usize);

            Ok(())
        } else {
            // Incremental path: cache_len > 0, need data already in cache + new tokens.
            // Fall back to bulk_scatter_kv (scatter first, then read from cache).
            self.bulk_scatter_kv(layer_idx, k_buf_ptr, v_buf_ptr, m, kv_dim as u32)?;
            let total_len = cache_len + m as usize;

            self.ensure_cublas()?;
            let score_size = num_heads * m as usize * total_len;
            self.ensure_attn_score_scratch(score_size)?;
            let score_ptr = self
                .prefill_attn_scores
                .as_ref()
                .expect("score scratch just allocated")
                .as_ptr();

            let k_key = format!("kv_{layer_idx}_k");
            let v_key = format!("kv_{layer_idx}_v");
            let k_cache_ptr = self
                .kv_cache_gpu
                .get(&k_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?
                .as_ptr();
            let v_cache_ptr = self
                .kv_cache_gpu
                .get(&v_key)
                .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?
                .as_ptr();

            let handle = self.cublas_handle.as_ref().expect("cublas initialized");

            for kv_group in 0..num_kv_heads {
                let first_q_head = kv_group * heads_per_kv;
                let k_head_ptr = k_cache_ptr
                    + (kv_group * max_len * head_dim * std::mem::size_of::<f32>()) as u64;
                let q_head_ptr =
                    q_buf_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;
                let s_head_ptr = score_ptr
                    + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;

                handle.gemm_f32_strided_batched(
                    trueno_gpu::driver::GemmOp::Trans,   // K^T
                    trueno_gpu::driver::GemmOp::NoTrans, // Q
                    total_len as i32,                    // m (rows of C)
                    m as i32,                            // n (cols of C)
                    head_dim as i32,                     // k
                    scale,                               // alpha = 1/sqrt(d)
                    k_head_ptr,                          // A = K cache
                    head_dim as i32,                     // lda
                    0,
                    q_head_ptr,
                    q_dim as i32,
                    head_dim as i64,
                    0.0,
                    s_head_ptr,
                    total_len as i32,
                    (m as usize * total_len) as i64,
                    heads_per_kv as i32,
                )?;
            }

            // Causal mask + softmax
            if !self.modules.contains_key("causal_mask_softmax") {
                let module = self.compile_ptx(Self::CAUSAL_MASK_SOFTMAX_PTX)?;
                self.modules
                    .insert("causal_mask_softmax".to_string(), module);
            }

            {
                let module = self
                    .modules
                    .get_mut("causal_mask_softmax")
                    .expect("just inserted");
                let config = LaunchConfig {
                    grid: (num_heads as u32, m, 1),
                    block: (32, 1, 1),
                    shared_mem: 0,
                };

                let mut s_ptr = score_ptr;
                let mut m_val = m;
                let mut tl_val = total_len as u32;
                let mut base_val = cache_len as u32;
                let mut nh_val = num_heads as u32;

                // SAFETY: score buffer valid, dimensions verified
                unsafe {
                    self.stream.launch_kernel(
                        module,
                        "causal_mask_softmax",
                        &config,
                        &mut [
                            std::ptr::from_mut(&mut s_ptr) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut tl_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut base_val) as *mut std::ffi::c_void,
                            std::ptr::from_mut(&mut nh_val) as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }

            let handle = self.cublas_handle.as_ref().expect("cublas initialized");

            for kv_group in 0..num_kv_heads {
                let first_q_head = kv_group * heads_per_kv;
                let v_head_ptr = v_cache_ptr
                    + (kv_group * max_len * head_dim * std::mem::size_of::<f32>()) as u64;
                let p_head_ptr = score_ptr
                    + (first_q_head * m as usize * total_len * std::mem::size_of::<f32>()) as u64;
                let o_head_ptr =
                    attn_out_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;

                handle.gemm_f32_strided_batched(
                    trueno_gpu::driver::GemmOp::NoTrans,
                    trueno_gpu::driver::GemmOp::NoTrans,
                    head_dim as i32,
                    m as i32,
                    total_len as i32,
                    1.0,
                    v_head_ptr,
                    head_dim as i32,
                    0,
                    p_head_ptr,
                    total_len as i32,
                    (m as usize * total_len) as i64,
                    0.0,
                    o_head_ptr,
                    q_dim as i32,
                    (head_dim) as i64,
                    heads_per_kv as i32,
                )?;
            }

            Ok(())
        }
    }

    /// PMAT-051: Scatter K/V from packed QKV buffer directly to batched KV cache.
    ///
    /// Five-Whys: bulk_scatter_kv makes 56,000 D2D copies → 128.7ms Attn+Scatter.
    /// Why? Each token×head needs cuMemcpyDtoDAsync (interleaved→contiguous).
    /// Fix: Single PTX kernel per prompt per layer scatters all K/V in one launch.
    /// Grid=(seq_len, num_kv_heads), Block=(head_dim). Each thread copies K+V f32.
    ///
    /// Replaces scatter_single_kv_to_batched_layer (which required intermediate single cache).
    const SCATTER_PACKED_KV_PTX: &str = r#"
.version 7.5
.target sm_75
.address_size 64

.visible .entry scatter_packed_kv(
    .param .u64 p_k_src,
    .param .u64 p_v_src,
    .param .u64 p_k_dst,
    .param .u64 p_v_dst,
    .param .u32 p_kv_dim,
    .param .u32 p_head_dim,
    .param .u32 p_max_len
) {
    .reg .u32 %r<8>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd0, [p_k_src];
    ld.param.u64 %rd1, [p_v_src];
    ld.param.u64 %rd2, [p_k_dst];
    ld.param.u64 %rd3, [p_v_dst];
    ld.param.u32 %r0, [p_kv_dim];
    ld.param.u32 %r1, [p_head_dim];
    ld.param.u32 %r2, [p_max_len];

    mov.u32 %r3, %ctaid.x;   // token_idx
    mov.u32 %r4, %ctaid.y;   // head_idx
    mov.u32 %r5, %tid.x;     // dim_idx

    // src_offset = token * kv_dim + head * head_dim + dim
    mad.lo.u32 %r6, %r3, %r0, %r5;
    mad.lo.u32 %r6, %r4, %r1, %r6;

    // dst_offset = (head * max_len + token) * head_dim + dim
    mad.lo.u32 %r7, %r4, %r2, %r3;
    mad.lo.u32 %r7, %r7, %r1, %r5;

    // Convert to byte offsets
    cvt.u64.u32 %rd4, %r6;
    shl.b64 %rd4, %rd4, 2;
    cvt.u64.u32 %rd5, %r7;
    shl.b64 %rd5, %rd5, 2;

    // K: packed to cache
    add.u64 %rd6, %rd0, %rd4;
    ld.global.f32 %f0, [%rd6];
    add.u64 %rd7, %rd2, %rd5;
    st.global.f32 [%rd7], %f0;

    // V: packed to cache
    add.u64 %rd8, %rd1, %rd4;
    ld.global.f32 %f1, [%rd8];
    add.u64 %rd9, %rd3, %rd5;
    st.global.f32 [%rd9], %f1;

    ret;
}
"#;

    /// PMAT-051: Prefill attention reading K/V directly from packed QKV buffer.
    ///
    /// Five-Whys: Attn+Scatter=128.7ms (58% of multi-prompt prefill).
    /// Why? bulk_scatter_kv makes 56,000 individual cuMemcpyDtoDAsync calls.
    /// Why? Packed K/V [M, kv_dim] needs rearranging to [num_kv_heads, max_len, head_dim].
    /// Why? cuBLAS expected lda=head_dim from cache layout.
    /// Fix: Use lda=kv_dim to read directly from packed buffer. Zero D2D copies.
    ///
    /// Then scatter K/V from packed buffer directly to batched KV cache via PTX kernel,
    /// bypassing the intermediate single KV cache entirely.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_attention_from_packed(
        &mut self,
        layer_idx: usize,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        seq_len: u32,
        q_dim: u32,
        kv_dim: u32,
        slot_idx: usize,
    ) -> Result<(), GpuError> {
        if seq_len == 0 {
            return Ok(());
        }

        let num_heads = self.kv_num_heads;
        let num_kv_heads = self.kv_num_kv_heads;
        let head_dim = self.kv_head_dim;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_len = seq_len as usize; // No prior cache — fresh prompt

        // Score scratch buffer
        self.ensure_cublas()?;
        let score_size = num_heads * seq_len as usize * total_len;
        self.ensure_attn_score_scratch(score_size)?;
        let score_ptr = self
            .prefill_attn_scores
            .as_ref()
            .expect("score scratch just allocated")
            .as_ptr();

        // 1. QK^T via cuBLAS — reading K directly from packed buffer (lda=kv_dim)
        //
        // K packed: [S, kv_dim] row-major → col-major per head: [head_dim, S], lda = kv_dim
        // (heads are interleaved with kv_dim stride between tokens)
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");

        for kv_group in 0..num_kv_heads {
            let first_q_head = kv_group * heads_per_kv;
            let k_head_ptr = k_buf_ptr + (kv_group * head_dim * std::mem::size_of::<f32>()) as u64;
            let q_head_ptr =
                q_buf_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;
            let s_head_ptr = score_ptr
                + (first_q_head * seq_len as usize * total_len * std::mem::size_of::<f32>()) as u64;

            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::Trans,     // K^T
                trueno_gpu::driver::GemmOp::NoTrans,   // Q
                total_len as i32,                      // m (rows of C)
                seq_len as i32,                        // n (cols of C)
                head_dim as i32,                       // k
                scale,                                 // alpha = 1/sqrt(d)
                k_head_ptr,                            // A = packed K
                kv_dim as i32,                         // lda = kv_dim (NOT head_dim)
                0,                                     // stride_a = 0 (shared K)
                q_head_ptr,                            // B = Q
                q_dim as i32,                          // ldb = q_dim
                head_dim as i64,                       // stride_b = head_dim
                0.0,                                   // beta
                s_head_ptr,                            // C = score
                total_len as i32,                      // ldc
                (seq_len as usize * total_len) as i64, // stride_c
                heads_per_kv as i32,                   // batch
            )?;
        }

        // 2. Causal mask + softmax
        if !self.modules.contains_key("causal_mask_softmax") {
            let module = self.compile_ptx(Self::CAUSAL_MASK_SOFTMAX_PTX)?;
            self.modules
                .insert("causal_mask_softmax".to_string(), module);
        }

        {
            let module = self
                .modules
                .get_mut("causal_mask_softmax")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (num_heads as u32, seq_len, 1),
                block: (32, 1, 1),
                shared_mem: 0,
            };

            let mut s_ptr = score_ptr;
            let mut m_val = seq_len;
            let mut tl_val = total_len as u32;
            let mut base_val = 0u32; // No prior cache
            let mut nh_val = num_heads as u32;

            // SAFETY: score buffer valid, dimensions verified
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "causal_mask_softmax",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut s_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut tl_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut base_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut nh_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // 3. Attn × V — reading V directly from packed buffer (lda=kv_dim)
        let handle = self.cublas_handle.as_ref().expect("cublas initialized");

        for kv_group in 0..num_kv_heads {
            let first_q_head = kv_group * heads_per_kv;
            let v_head_ptr = v_buf_ptr + (kv_group * head_dim * std::mem::size_of::<f32>()) as u64;
            let p_head_ptr = score_ptr
                + (first_q_head * seq_len as usize * total_len * std::mem::size_of::<f32>()) as u64;
            let o_head_ptr =
                attn_out_ptr + (first_q_head * head_dim * std::mem::size_of::<f32>()) as u64;

            handle.gemm_f32_strided_batched(
                trueno_gpu::driver::GemmOp::NoTrans,   // V
                trueno_gpu::driver::GemmOp::NoTrans,   // P
                head_dim as i32,                       // m (rows of C)
                seq_len as i32,                        // n (cols of C)
                total_len as i32,                      // k (inner dim)
                1.0,                                   // alpha
                v_head_ptr,                            // A = packed V
                kv_dim as i32,                         // lda = kv_dim (NOT head_dim)
                0,                                     // stride_a = 0 (shared V)
                p_head_ptr,                            // B = P (softmax output)
                total_len as i32,                      // ldb
                (seq_len as usize * total_len) as i64, // stride_b
                0.0,                                   // beta
                o_head_ptr,                            // C = output
                q_dim as i32,                          // ldc = q_dim
                head_dim as i64,                       // stride_c
                heads_per_kv as i32,                   // batch
            )?;
        }

        // 4. Scatter K/V from packed buffer directly to batched KV cache
        let stride = self.batched_kv_stride;
        if stride > 0 {
            let max_len = self.kv_cache_max_len;
            let slot_offset_bytes = (slot_idx * stride * std::mem::size_of::<f32>()) as u64;

            let batched_k_ptr = self
                .batched_kv_k_caches
                .get(&layer_idx)
                .ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-051: batched K cache layer {} not found",
                        layer_idx
                    ))
                })?
                .as_ptr();
            let batched_v_ptr = self
                .batched_kv_v_caches
                .get(&layer_idx)
                .ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PMAT-051: batched V cache layer {} not found",
                        layer_idx
                    ))
                })?
                .as_ptr();

            // Compile scatter kernel on first use
            if !self.modules.contains_key("scatter_packed_kv") {
                let module = self.compile_ptx(Self::SCATTER_PACKED_KV_PTX)?;
                self.modules.insert("scatter_packed_kv".to_string(), module);
            }

            let module = self
                .modules
                .get_mut("scatter_packed_kv")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (seq_len, num_kv_heads as u32, 1),
                block: (head_dim as u32, 1, 1),
                shared_mem: 0,
            };

            let mut k_src = k_buf_ptr;
            let mut v_src = v_buf_ptr;
            let mut k_dst = batched_k_ptr + slot_offset_bytes;
            let mut v_dst = batched_v_ptr + slot_offset_bytes;
            let mut kv_dim_val = kv_dim;
            let mut head_dim_val = head_dim as u32;
            let mut max_len_val = max_len as u32;

            // SAFETY: All GPU buffers valid, grid/block dimensions verified
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "scatter_packed_kv",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut k_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_src) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v_dst) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut kv_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }

            // Update batched KV length for this slot
            if slot_idx < self.batched_kv_lengths.len() {
                self.batched_kv_lengths[slot_idx] = seq_len as usize;
            }
        }

        Ok(())
    }

    /// PMAT-024/026: Batched GEMV with cuBLAS GEMM fallback for prefill
    ///
    /// When M >= threshold and weights are Q4K or Q6K, uses cuBLAS GEMM
    /// (HGEMM or SGEMM) instead of batched GEMV.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_gemv_or_gemm(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        packed_input: &GpuBuffer<f32>,
        packed_output: &GpuBuffer<f32>,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n_per_seq: u32,
        k_per_seq: u32,
    ) -> Result<(), GpuError> {
        // PMAT-054B: W4A16 WMMA GEMM with pre-computed scales for batched decode (M=2-8).
        // Pre-computed FP16 scales eliminate GGML decode on GPU (~20→5 insn/elem).
        // Only for small M (batched decode). Large M (prefill) uses FP8 cuBLASLt.
        // M<=8 guard: WMMA 32×32 tiles waste 75-94% compute at M<8, and FP8
        // cuBLASLt is faster at M>=5 (PMAT-093).
        if self.gpu_profile.w4a16_interleaved
            && m >= 2
            && m <= 8
            && !self.is_capturing
            && qtype == WeightQuantType::Q4K
            && self.interleaved_weight_cache.contains_key(&weight_ptr)
        {
            return self.w4a16_wmma_q4k_gemm(
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            );
        }

        // PMAT-090: FP8 cuBLASLt GEMM for batched decode on sm_89+.
        // PMAT-093: FP8 threshold raised from M>=2 to M>=5.
        // At M=4: FP8 conversion overhead (absmax+convert per projection) cancels
        // tensor core gains. DP4A with fused QKV+gate+up has lower launch count.
        // Benchmarks (yoga 4060L, 1900MHz, 60s, isolated):
        //   c=4 FP8: 235.0 aggregate, 16.5ms ITL
        //   c=4 DP4A: 242.1 aggregate, 15.9ms ITL (+3%)
        //   c=8 FP8: 405.6 aggregate, 18.9ms ITL
        //   c=8 DP4A: 329.1 aggregate, 23.4ms ITL (-19%)
        // FP8 crossover point: ~M=5 where tensor core BW dominates conversion cost.
        // FP8 weight cache persists from prefill — no additional warmup needed.
        if self.gpu_profile.fp8_decode
            && m >= 5
            && !self.is_capturing
            && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
        {
            return self.cublas_prefill_gemm(
                qtype,
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            );
        }

        // PMAT-061: cuBLAS HGEMM for M>1 decode when FP16 weight cache is available.
        // Five-Whys: c=4 decode 0.56x (23.4ms/step vs llama.cpp 13.1ms).
        // Why? Batched Q4K GEMV is compute-bound at M=4 (3.25x instruction scaling).
        // Why? Each weight SB requires M activation loads + M DP4A chains.
        // Why? Dequant uses ~20 bitmask ops/SB — kernel is ALU-heavy at any M.
        // Fix: cuBLAS HGEMM uses tensor cores (memory-bound, ~1x scaling with M).
        // FP16 reads 3.5x more data but tensor cores hide M×compute.
        // Estimated: HGEMM M=4 ~15.6ms (BW-limited) vs GEMV M=4 ~23.4ms (compute-limited).
        if self.hgemm_batched_decode_active
            && m >= 2
            && !self.is_capturing
            && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
            && self.fp16_weight_cache.contains_key(&weight_ptr)
            && std::env::var("HGEMM_BATCHED_DECODE").as_deref() != Ok("0")
        {
            return self.cublas_prefill_gemm(
                qtype,
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            );
        }

        // GH-141: Batched HW DP4A for Q4K on DP4A-capable GPUs (sm_75+).
        // Reads Q4K weights (0.5625 B/elem) + Q8_1 activations (1.125 B/elem)
        // = 1.69 B/elem total, vs cuBLAS SGEMM 8 B/elem. 4.7x less bandwidth.
        // PMAT-056: Removed !self.is_capturing guard — DP4A kernels are pure GPU
        // kernels (no H2D copies), graph-capturable. Old guard forced FP32 fallback.
        let use_batched_dp4a = qtype == WeightQuantType::Q4K
            && m >= 2
            && m <= 8
            && self.gpu_profile.q4k == crate::cuda::gpu_profile::Q4kVariant::HwDp4a
            && !self.is_prefilling
            && std::env::var("BATCHED_DP4A").as_deref() != Ok("0");

        if use_batched_dp4a {
            return self.batched_hw_dp4a_q4k_gemv_into(
                weight_ptr,
                packed_input,
                packed_output,
                m,
                n_per_seq,
                k_per_seq,
            );
        }

        // GH-141: Never use cuBLAS during CUDA graph capture. cuBLAS calls
        // are not reliably capturable — they may use internal workspace or
        // stream management that the graph infrastructure cannot replay.
        // Use batched GEMV (custom PTX) instead, which is always capturable.
        let use_cublas = m >= super::cublas_gemm_threshold()
            && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
            && std::env::var("CUBLAS_PREFILL").as_deref() != Ok("0")
            && !self.is_capturing;

        if use_cublas {
            self.cublas_prefill_gemm(
                qtype,
                weight_ptr,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            )
        } else {
            self.batched_gemv_with_fallback(
                qtype,
                weight_ptr,
                packed_input,
                packed_output,
                packed_input_ptr,
                packed_output_ptr,
                m,
                n_per_seq,
                k_per_seq,
            )
        }
    }

    /// PMAT-061: Check if FP16 weight cache is populated (for HGEMM batched decode routing).
    pub(crate) fn has_fp16_weight_cache(&self) -> bool {
        !self.fp16_weight_cache.is_empty()
    }

    /// GH-141: Clear FP16 weight cache to free VRAM for batched KV caches.
    ///
    /// FP16 weights are only needed during HGEMM prefill (not GEMV decode).
    /// Clearing them before batched decode frees ~2944 MB on Qwen2.5 1.5B,
    /// allowing batched KV cache allocation on 8GB GPUs.
    /// The cache is re-warmed lazily on next prefill via get_or_cache_fp16_weight.
    pub(crate) fn clear_fp16_weight_cache(&mut self) {
        let cache_mb = self
            .fp16_weight_cache
            .values()
            .map(|b| b.len() * 2)
            .sum::<usize>() as f64
            / 1_048_576.0;
        let count = self.fp16_weight_cache.len();
        self.fp16_weight_cache.clear();
        // Also clear FP16 activation scratch (tied to HGEMM path)
        self.fp16_activation_scratch = None;
        self.fp16_activation_scratch_size = 0;
        if count > 0 {
            eprintln!(
                "[GH-141] Cleared FP16 weight cache: {} matrices ({:.1} MB freed)",
                count, cache_mb,
            );
        }
    }

    /// GH-141: Clear prefill CUDA graphs to free VRAM.
    /// Prefill graphs are not used during batched decode and can be recaptured later.
    pub(crate) fn clear_prefill_graphs(&mut self) {
        let count = self.prefill_graphs.len();
        self.prefill_graphs.clear();
        self.prefill_graph_input_buf = None;
        // PMAT-059: Reset prefill capture flag so graphs can be recaptured
        self.prefill_graph_capture_failed = false;
        if count > 0 {
            eprintln!("[GH-141] Cleared {} prefill graphs", count);
        }
    }

    /// PMAT-037: Pre-populate FP16 weight cache for HGEMM decode.
    ///
    /// Must be called BEFORE CUDA graph capture, because graph capture doesn't
    /// allow dynamic GPU memory allocation. Also warms up cuBLAS workspace.
    pub(crate) fn warmup_hgemm_cache(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        let start = std::time::Instant::now();
        let mut cached = 0usize;

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                break;
            }
            let lw = self.get_indexed_layer(layer_idx).clone();

            // Cache all 7 weight matrices per layer
            let weights = [
                (lw.attn_q_qtype, lw.attn_q_ptr, q_dim, hidden_dim), // Q: [q_dim, hidden]
                (lw.attn_k_qtype, lw.attn_k_ptr, kv_dim, hidden_dim), // K: [kv_dim, hidden]
                (lw.attn_v_qtype, lw.attn_v_ptr, kv_dim, hidden_dim), // V: [kv_dim, hidden]
                (lw.attn_output_qtype, lw.attn_output_ptr, hidden_dim, q_dim), // O: [hidden, q_dim]
                (
                    lw.ffn_gate_qtype,
                    lw.ffn_gate_ptr,
                    intermediate_dim,
                    hidden_dim,
                ), // gate
                (lw.ffn_up_qtype, lw.ffn_up_ptr, intermediate_dim, hidden_dim), // up
                (
                    lw.ffn_down_qtype,
                    lw.ffn_down_ptr,
                    hidden_dim,
                    intermediate_dim,
                ), // down
            ];

            for (qtype, ptr, n, k) in weights {
                if ptr != 0
                    && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
                    && self.fp16_weight_cache.get(&ptr).is_none()
                {
                    self.get_or_cache_fp16_weight(qtype, ptr, n, k)?;
                    cached += 1;
                }
            }
        }

        // Also cache LM head
        if self.lm_head_ptr != 0
            && (self.lm_head_qtype == WeightQuantType::Q4K
                || self.lm_head_qtype == WeightQuantType::Q6K)
        {
            let lm_ptr = self.lm_head_ptr;
            let lm_qtype = self.lm_head_qtype;
            if self.fp16_weight_cache.get(&lm_ptr).is_none() {
                self.get_or_cache_fp16_weight(lm_qtype, lm_ptr, vocab_size, hidden_dim)?;
                cached += 1;
            }
        }

        // PMAT-063: Pre-allocate cuBLAS workspace for CUDA graph capture.
        // Without this, graph capture forces workspace-free algorithms (7x slower).
        // 32 MB workspace covers all standard GEMM shapes for Qwen 1.5B.
        if cached > 0 {
            self.ensure_cublas_workspace()?;

            // PMAT-063: Warm up cuBLAS JIT for common prefill sequence lengths.
            // First cuBLAS GEMM per (M,N,K) takes ~42ms (algorithm selection);
            // subsequent calls take <0.1ms. Pre-warming all common M values
            // moves this one-time cost from first-request to model load.
            //
            // Measured shapes for Qwen2.5-Coder-1.5B (M=seq_len):
            //   (N=1536, K=1536): Q/O projections
            //   (N=256,  K=1536): K/V projections (GQA, 2 KV heads)
            //   (N=8960, K=1536): gate/up projections
            //   (N=1536, K=8960): down projection
            // Each unique M triggers ~42ms JIT for first shape + ~5ms for remaining.
            let max_m = 512i32; // Cover sequences up to 512 tokens
            let max_dim = std::cmp::max(hidden_dim, intermediate_dim) as usize;
            self.ensure_fp16_activation_scratch(max_m as usize * max_dim)?;
            let dummy_input = GpuBuffer::<u16>::new(&self.context, max_m as usize * max_dim)?;
            let dummy_out = GpuBuffer::<f32>::new(&self.context, max_m as usize * max_dim)?;

            if let Some((&_ptr, fp16_buf)) = self.fp16_weight_cache.iter().next() {
                let handle = self.cublas_handle.as_ref().expect("cublas initialized");

                let shapes = [
                    (hidden_dim as i32, hidden_dim as i32), // Q/O
                    (
                        (self.kv_num_kv_heads * self.kv_head_dim) as i32,
                        hidden_dim as i32,
                    ), // K/V
                    (intermediate_dim as i32, hidden_dim as i32), // gate/up
                    (hidden_dim as i32, intermediate_dim as i32), // down
                ];

                // Warm common M values: powers of 2 + typical chat lengths
                let m_values: &[i32] = &[8, 16, 32, 64, 125, 128, 256, 512];
                let mut warmed = 0u32;
                for &m in m_values {
                    for &(n, k) in &shapes {
                        if n > 0 && k > 0 {
                            let _ = handle.gemm_f16_to_f32(
                                trueno_gpu::driver::GemmOp::Trans,
                                trueno_gpu::driver::GemmOp::NoTrans,
                                n,
                                m,
                                k,
                                0.0,
                                fp16_buf.as_ptr(),
                                k,
                                dummy_input.as_ptr(),
                                k,
                                0.0,
                                dummy_out.as_ptr(),
                                n,
                            );
                            warmed += 1;
                        }
                    }
                }
                self.stream.synchronize()?;
                eprintln!(
                    "[PMAT-063] cuBLAS JIT warmed: {} shapes ({}×{} M×NK combos)",
                    warmed,
                    m_values.len(),
                    shapes.len()
                );
            }
        }

        let elapsed = start.elapsed();
        let cache_mb = self
            .fp16_weight_cache
            .values()
            .map(|b| b.len() * 2)
            .sum::<usize>() as f64
            / 1_048_576.0;
        eprintln!(
            "[PMAT-037] FP16 weight cache: {} matrices cached ({:.1} MB) in {:.1}ms",
            cached,
            cache_mb,
            elapsed.as_secs_f64() * 1000.0,
        );

        Ok(())
    }

    /// PMAT-053: Pre-populate FP8 E4M3 weight cache.
    ///
    /// Similar to `warmup_hgemm_cache` but caches as FP8 (1 byte/elem = 50% of FP16).
    /// Must be called BEFORE CUDA graph capture. Requires sm_89+.
    pub(crate) fn warmup_fp8_cache(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        // GH-542: FP8 E4M3 cuBLASLt GEMM requires Ada/Hopper (sm_89-sm_90).
        // Blackwell (sm_100+, cc >= 100) has incompatible FP8 tensor cores that
        // cause CUDA_ERROR_ILLEGAL_ADDRESS during warmup, poisoning the context.
        if self.gpu_profile.cc < 89 || self.gpu_profile.cc >= 100 {
            eprintln!(
                "[PMAT-053] FP8 cache skipped: requires sm_89-sm_9x (have {}, cc={})",
                self.gpu_profile.sm_target, self.gpu_profile.cc
            );
            return Ok(());
        }

        let start = std::time::Instant::now();
        self.ensure_cublas()?;
        let mut cached = 0usize;

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                break;
            }
            let lw = self.get_indexed_layer(layer_idx).clone();

            let weights = [
                (lw.attn_q_qtype, lw.attn_q_ptr, q_dim, hidden_dim),
                (lw.attn_k_qtype, lw.attn_k_ptr, kv_dim, hidden_dim),
                (lw.attn_v_qtype, lw.attn_v_ptr, kv_dim, hidden_dim),
                (lw.attn_output_qtype, lw.attn_output_ptr, hidden_dim, q_dim),
                (
                    lw.ffn_gate_qtype,
                    lw.ffn_gate_ptr,
                    intermediate_dim,
                    hidden_dim,
                ),
                (lw.ffn_up_qtype, lw.ffn_up_ptr, intermediate_dim, hidden_dim),
                (
                    lw.ffn_down_qtype,
                    lw.ffn_down_ptr,
                    hidden_dim,
                    intermediate_dim,
                ),
            ];

            for (qtype, ptr, n, k) in weights {
                if ptr != 0
                    && (qtype == WeightQuantType::Q4K || qtype == WeightQuantType::Q6K)
                    && !self.fp8_weight_cache.contains_key(&ptr)
                {
                    self.get_or_cache_fp8_weight(qtype, ptr, n, k)?;
                    cached += 1;
                }
            }
        }

        // Also cache LM head
        if self.lm_head_ptr != 0
            && (self.lm_head_qtype == WeightQuantType::Q4K
                || self.lm_head_qtype == WeightQuantType::Q6K)
        {
            let lm_ptr = self.lm_head_ptr;
            let lm_qtype = self.lm_head_qtype;
            if !self.fp8_weight_cache.contains_key(&lm_ptr) {
                self.get_or_cache_fp8_weight(lm_qtype, lm_ptr, vocab_size, hidden_dim)?;
                cached += 1;
            }
        }

        self.stream.synchronize()?;

        // PMAT-082: cuBLASLt FP8 JIT warmup — eliminate 3ms first-request latency.
        // First cuBLASLt FP8 GEMM triggers JIT compilation; warm it here at model load.
        if cached > 0 {
            if self.cublaslt_handle.is_none() {
                self.cublaslt_handle = Some(trueno_gpu::driver::CublasLtHandle::new()?);
            }

            // Use smallest realistic M (padded to 16) with the first cached FP8 weight.
            // One GEMM is enough — cuBLASLt JIT is per-handle, not per-shape.
            let m_warmup: i32 = 16;
            let n_warmup = hidden_dim as i32;
            let k_warmup = hidden_dim as i32;
            let warmup_input_count = m_warmup as usize * k_warmup as usize;
            let warmup_output_count = n_warmup as usize * m_warmup as usize;

            // Allocate scratch BEFORE borrowing lt_handle (avoid borrow conflict)
            self.ensure_fp8_activation_scratch(warmup_input_count)?;
            self.ensure_fp16_activation_scratch(warmup_output_count)?;
            let input_ptr = self
                .fp8_activation_scratch
                .as_ref()
                .expect("scratch just allocated")
                .as_ptr();
            let output_ptr = self
                .fp16_activation_scratch
                .as_ref()
                .expect("scratch just allocated")
                .as_ptr();

            // Extract weight ptr before borrowing lt_handle
            let fp8_weight_ptr = self.fp8_weight_cache.values().next().map(|b| b.as_ptr());

            if let Some(w_ptr) = fp8_weight_ptr {
                let lt_handle = self.cublaslt_handle.as_ref().expect("just created");
                let _ = lt_handle.gemm_fp8_e4m3_to_f16(
                    trueno_gpu::driver::GemmOp::Trans,
                    trueno_gpu::driver::GemmOp::NoTrans,
                    n_warmup,
                    m_warmup,
                    k_warmup,
                    1.0, // dummy alpha
                    w_ptr,
                    k_warmup,
                    input_ptr,
                    k_warmup,
                    0.0,
                    output_ptr,
                    n_warmup,
                    &self.stream,
                );
                self.stream.synchronize()?;
                eprintln!("[PMAT-082] cuBLASLt FP8 JIT warmed ({n_warmup}×{m_warmup}×{k_warmup})");
            }
        }

        let elapsed = start.elapsed();
        let cache_mb = self
            .fp8_weight_cache
            .values()
            .map(|b| b.len())
            .sum::<usize>() as f64
            / 1_048_576.0;
        eprintln!(
            "[PMAT-053] FP8 weight cache: {} matrices cached ({:.1} MB) in {:.1}ms",
            cached,
            cache_mb,
            elapsed.as_secs_f64() * 1000.0,
        );

        Ok(())
    }

    /// PMAT-053: Clear FP8 weight cache to free VRAM.
    pub(crate) fn clear_fp8_weight_cache(&mut self) {
        let cache_mb = self
            .fp8_weight_cache
            .values()
            .map(|b| b.len())
            .sum::<usize>() as f64
            / 1_048_576.0;
        let count = self.fp8_weight_cache.len();
        self.fp8_weight_cache.clear();
        self.fp8_activation_scratch = None;
        self.fp8_activation_scratch_size = 0;
        if count > 0 {
            eprintln!(
                "[PMAT-053] Cleared FP8 weight cache: {} matrices ({:.1} MB freed)",
                count, cache_mb,
            );
        }
    }

    /// PMAT-069: Fused prefill attention — single PTX kernel per layer.
    ///
    /// Replaces 5 cuBLAS + PTX launches (2 QK^T + 1 softmax + 2 Attn×V) with
    /// 1 fused kernel launch using online softmax (no N×N materialization).
    /// For 28 layers: 140 → 28 launches. Saves ~7.5ms TTFT.
    ///
    /// After attention, scatters K/V to KV cache for subsequent decode tokens.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_attention_fused(
        &mut self,
        layer_idx: usize,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        m: u32,
        q_dim: u32,
        kv_dim: u32,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        heads_per_kv: usize,
        max_len: usize,
    ) -> Result<(), GpuError> {
        use trueno_gpu::kernels::{Kernel as _, PrefillAttentionKernel};

        // Compile and cache fused attention kernel
        if !self.modules.contains_key("fused_prefill_attn") {
            let kernel = PrefillAttentionKernel::new(head_dim as u32, heads_per_kv as u32);
            let ptx = kernel.emit_ptx_for_target(&self.kernels.sm_target);
            let module = self.compile_ptx(&ptx)?;
            self.modules
                .insert("fused_prefill_attn".to_string(), module);
            eprintln!(
                "[PMAT-069] Compiled fused prefill attention: head_dim={}, heads_per_kv={}",
                head_dim, heads_per_kv,
            );
        }

        // Launch fused attention kernel
        {
            let module = self
                .modules
                .get_mut("fused_prefill_attn")
                .expect("just inserted");
            let config = LaunchConfig {
                grid: (num_heads as u32, 1, 1),
                block: (32, 1, 1),
                shared_mem: 0,
            };

            let mut q = q_buf_ptr;
            let mut k = k_buf_ptr;
            let mut v = v_buf_ptr;
            let mut o = attn_out_ptr;
            let mut m_val = m;
            let mut qs = q_dim;
            let mut kvs = kv_dim;
            let mut nqh = num_heads as u32;

            // SAFETY: GPU buffers valid, grid covers all Q heads, block is 1 warp
            unsafe {
                self.stream.launch_kernel(
                    module,
                    "fused_prefill_attention_causal",
                    &config,
                    &mut [
                        std::ptr::from_mut(&mut q) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut k) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut v) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut o) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut qs) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut kvs) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut nqh) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // Scatter K/V from packed buffer to KV cache (needed for subsequent decode)
        let k_key = format!("kv_{layer_idx}_k");
        let v_key = format!("kv_{layer_idx}_v");
        let k_cache_ptr = self
            .kv_cache_gpu
            .get(&k_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("K cache not found".to_string()))?
            .as_ptr();
        let v_cache_ptr = self
            .kv_cache_gpu
            .get(&v_key)
            .ok_or_else(|| GpuError::InvalidLaunchConfig("V cache not found".to_string()))?
            .as_ptr();

        if !self.modules.contains_key("scatter_packed_kv") {
            let module = self.compile_ptx(Self::SCATTER_PACKED_KV_PTX)?;
            self.modules.insert("scatter_packed_kv".to_string(), module);
        }

        let module = self
            .modules
            .get_mut("scatter_packed_kv")
            .expect("just inserted");
        let config = LaunchConfig {
            grid: (m, num_kv_heads as u32, 1),
            block: (head_dim as u32, 1, 1),
            shared_mem: 0,
        };

        let mut k_src = k_buf_ptr;
        let mut v_src = v_buf_ptr;
        let mut k_dst = k_cache_ptr;
        let mut v_dst = v_cache_ptr;
        let mut kv_dim_val = kv_dim;
        let mut head_dim_val = head_dim as u32;
        let mut max_len_val = max_len as u32;

        // SAFETY: All GPU buffers valid, grid/block dimensions verified
        unsafe {
            self.stream.launch_kernel(
                module,
                "scatter_packed_kv",
                &config,
                &mut [
                    std::ptr::from_mut(&mut k_src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut v_src) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut v_dst) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut kv_dim_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut head_dim_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut max_len_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Update single KV cache length
        self.kv_cache_lengths.insert(layer_idx, m as usize);

        Ok(())
    }

    // ========================================================================
    // PMAT-091: Interleaved Q4K weight cache for coalesced WMMA GEMM
    // ========================================================================

    /// Warmup interleaved Q4K weight cache at model init.
    ///
    /// Downloads Q4K weights from GPU, repacks to column-interleaved tile layout
    /// on CPU, re-uploads to GPU. Only runs once per model load.
    /// Total overhead: ~200ms for 197 matrices (Q4K data ~850MB, PCIe limited).
    pub(crate) fn warmup_interleaved_cache(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        _vocab_size: u32,
    ) -> Result<(), GpuError> {
        let start = std::time::Instant::now();
        let mut cached = 0usize;

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                break;
            }
            let lw = self.get_indexed_layer(layer_idx).clone();

            let weights = [
                (lw.attn_q_qtype, lw.attn_q_ptr, q_dim, hidden_dim),
                (lw.attn_k_qtype, lw.attn_k_ptr, kv_dim, hidden_dim),
                (lw.attn_v_qtype, lw.attn_v_ptr, kv_dim, hidden_dim),
                (lw.attn_output_qtype, lw.attn_output_ptr, hidden_dim, q_dim),
                (
                    lw.ffn_gate_qtype,
                    lw.ffn_gate_ptr,
                    intermediate_dim,
                    hidden_dim,
                ),
                (lw.ffn_up_qtype, lw.ffn_up_ptr, intermediate_dim, hidden_dim),
                (
                    lw.ffn_down_qtype,
                    lw.ffn_down_ptr,
                    hidden_dim,
                    intermediate_dim,
                ),
            ];

            for (qtype, ptr, n, k) in weights {
                if ptr != 0
                    && qtype == WeightQuantType::Q4K
                    && !self.interleaved_weight_cache.contains_key(&ptr)
                {
                    self.get_or_cache_interleaved_weight(ptr, n, k)?;
                    cached += 1;
                }
            }
        }

        // NOTE: LM head skipped — WMMA for decode only, LM head uses existing path

        self.stream.synchronize()?;
        eprintln!(
            "[PMAT-091] Interleaved Q4K cache warmed: {} matrices in {:?}",
            cached,
            start.elapsed()
        );
        Ok(())
    }

    /// Get or create interleaved Q4K weight buffer for a given weight pointer.
    ///
    /// Downloads Q4K bytes from GPU, repacks to W4A16 tile layout (PMAT-054B)
    /// via `repack_q4k_w4a16()`, uploads pre-computed scale data to new GPU buffer.
    ///
    /// GH-143: Only available on x86_64 — WMMA kernels require desktop GPUs.
    #[cfg(target_arch = "x86_64")]
    fn get_or_cache_interleaved_weight(
        &mut self,
        weight_ptr: u64,
        n: u32,
        k: u32,
    ) -> Result<u64, GpuError> {
        if let Some(buf) = self.interleaved_weight_cache.get(&weight_ptr) {
            return Ok(buf.as_ptr());
        }

        let n_usize = n as usize;
        let k_usize = k as usize;
        let num_sb = k_usize / 256;
        let q4k_byte_count = n_usize * num_sb * 144;

        // Step 1: Download Q4K bytes from GPU to CPU
        let weight_view = unsafe { GpuBuffer::<u8>::from_raw_parts(weight_ptr, q4k_byte_count) };
        let mut q4k_host = vec![0u8; q4k_byte_count];
        weight_view.copy_to_host(&mut q4k_host)?;
        std::mem::forget(weight_view); // Don't free — we don't own this memory

        // Step 2: Repack to W4A16 format (pre-computed FP16 scales, PMAT-054B)
        let w4a16 = trueno_gpu::kernels::repack_q4k_w4a16(&q4k_host, n_usize, k_usize);

        // Step 3: Upload W4A16 data to GPU
        let il_buf = GpuBuffer::<u8>::from_host(&self.context, &w4a16)?;
        let il_ptr = il_buf.as_ptr();

        self.interleaved_weight_cache.insert(weight_ptr, il_buf);
        Ok(il_ptr)
    }

    /// GH-143: Stub for non-x86_64 — WMMA interleaved weight repacking not available.
    #[cfg(not(target_arch = "x86_64"))]
    fn get_or_cache_interleaved_weight(
        &mut self,
        _weight_ptr: u64,
        _n: u32,
        _k: u32,
    ) -> Result<u64, GpuError> {
        Err(GpuError::InvalidParameter(
            "WMMA interleaved weight repacking is only available on x86_64".to_string(),
        ))
    }

    /// PMAT-091: Launch interleaved WMMA Q4K GEMM for batched decode.
    ///
    /// Same grid/block as multi-warp WMMA but reads from column-interleaved
    /// weight layout. Adjacent threads access adjacent qs bytes — perfect coalescing.
    #[allow(clippy::too_many_arguments)]
    fn interleaved_wmma_q4k_gemm(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        // Get cached interleaved weight pointer
        let il_ptr = self
            .interleaved_weight_cache
            .get(&weight_ptr)
            .expect("interleaved weight should be cached by warmup")
            .as_ptr();

        // Pad M and N to multiples of 32 for 2×2 WMMA tile safety
        let m_padded = (m + 31) & !31;
        let n_padded = (n + 31) & !31;
        let needs_padding = m_padded != m || n_padded != n;

        let kernel_type = KernelType::InterleavedWmmaQ4KGemm {
            m: m_padded,
            n: n_padded,
            k,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("interleaved_wmma_q4k_gemm_{m_padded}_{n_padded}_{k}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        // If padding needed, use WMMA scratch buffer
        let actual_output_ptr = if needs_padding {
            let padded_count = m_padded as usize * n_padded as usize;
            self.ensure_wmma_scratch(padded_count)?;
            self.wmma_scratch
                .as_ref()
                .expect("wmma scratch allocated")
                .as_ptr()
        } else {
            packed_output_ptr
        };

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        // Grid: (ceil(N/32), ceil(M/32)), Block: 128 (4 warps for 2×2 WMMA tiles)
        let grid_x = n_padded / 32;
        let grid_y = m_padded / 32;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, 128, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = il_ptr;
        let mut ptr_c = actual_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // Copy valid [M, N] from padded buffer to actual output
        if needs_padding {
            self.stream.synchronize()?;
            for row in 0..m {
                let src_offset = row as u64 * n_padded as u64 * 4;
                let dst_offset = row as u64 * n as u64 * 4;
                self.stream.memcpy_dtod_sync(
                    packed_output_ptr + dst_offset,
                    actual_output_ptr + src_offset,
                    n as usize * 4,
                )?;
            }
        }

        Ok(())
    }

    /// PMAT-054B: Launch W4A16 WMMA Q4K GEMM with pre-computed FP16 scales.
    ///
    /// Same grid/block as PMAT-091 interleaved kernel, but Phase 2 dequant
    /// reduced from ~20 insn + 5-8 global loads to ~5 insn + 3 loads per element.
    /// Pre-computed eff_scale and eff_min eliminate GGML scale decoding on GPU.
    #[allow(clippy::too_many_arguments)]
    fn w4a16_wmma_q4k_gemm(
        &mut self,
        weight_ptr: u64,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<(), GpuError> {
        let w4_ptr = self
            .interleaved_weight_cache
            .get(&weight_ptr)
            .expect("W4A16 weight should be cached by warmup")
            .as_ptr();

        let m_padded = (m + 31) & !31;
        let n_padded = (n + 31) & !31;
        let needs_padding = m_padded != m || n_padded != n;

        let kernel_type = KernelType::W4a16WmmaQ4KGemm {
            m: m_padded,
            n: n_padded,
            k,
        };
        let kernel_name = self.kernels.kernel_name(&kernel_type);
        let cache_key = format!("w4a16_wmma_q4k_gemm_{m_padded}_{n_padded}_{k}");

        if !self.modules.contains_key(&cache_key) {
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(cache_key.clone(), module);
        }

        let actual_output_ptr = if needs_padding {
            let padded_count = m_padded as usize * n_padded as usize;
            self.ensure_wmma_scratch(padded_count)?;
            self.wmma_scratch
                .as_ref()
                .expect("wmma scratch allocated")
                .as_ptr()
        } else {
            packed_output_ptr
        };

        let module = self
            .modules
            .get_mut(&cache_key)
            .expect("module just inserted");

        let grid_x = n_padded / 32;
        let grid_y = m_padded / 32;
        let config = LaunchConfig::grid_2d(grid_x, grid_y, 128, 1);

        let mut ptr_a = packed_input_ptr;
        let mut ptr_b = w4_ptr;
        let mut ptr_c = actual_output_ptr;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut [
                    std::ptr::from_mut(&mut ptr_a) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_b) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut ptr_c) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut m_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut n_val) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut k_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        if needs_padding {
            self.stream.synchronize()?;
            for row in 0..m {
                let src_offset = row as u64 * n_padded as u64 * 4;
                let dst_offset = row as u64 * n as u64 * 4;
                self.stream.memcpy_dtod_sync(
                    packed_output_ptr + dst_offset,
                    actual_output_ptr + src_offset,
                    n as usize * 4,
                )?;
            }
        }

        Ok(())
    }
}
