#![allow(unsafe_code)]
#![allow(trivial_casts)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CublasHandle, CudaContext, CudaModule, CudaStream};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{
    Batched4DGemmKernel, BatchedRopeNeoxBackwardKernel, BatchedSoftmaxKernel,
    BatchedToInterleavedKernel, BatchedTransposeKernel, BatchedVectorizedRmsNormKernel,
    ElementwiseMulKernel, FusedSwigluKernel, GemmKernel, InterleavedToBatchedKernel, Kernel,
    Nf4GemmKernel, Nf4GemmTransposeKernel, ResidualAddKernel, ScaleKernel, SiluKernel,
};

use crate::autograd::cuda_tensor::{CudaTensorError, Result};

/// Cached compiled CUDA modules for forward kernels
#[cfg(feature = "cuda")]
pub(super) static FORWARD_KERNEL_CACHE: OnceLock<Mutex<ForwardKernelCache>> = OnceLock::new();

/// Cache for compiled forward kernel modules
///
/// Stores the device's SM target (e.g. "sm_89") detected at init time.
/// All PTX must be emitted for this target before compilation.
///
/// # Contract: F-PTX-001 (Target Parity)
///
/// PTX `.target` directive MUST match the device compute capability.
/// The cache validates this at compile time and rejects mismatched PTX.
#[cfg(feature = "cuda")]
pub(super) struct ForwardKernelCache {
    ctx: std::sync::Arc<CudaContext>,
    modules: HashMap<String, CudaModule>,
    /// Device SM target string (e.g. "sm_89" for RTX 4090)
    sm_target: String,
    /// cuBLAS handle (ALB-075): forward=tensor cores, backward=SIMD (ALB-076/trueno#170)
    cublas: Option<CublasHandle>,
}

#[cfg(feature = "cuda")]
impl ForwardKernelCache {
    pub(super) fn new(ctx: std::sync::Arc<CudaContext>) -> Self {
        // Detect device compute capability at construction time.
        // Falls back to sm_70 if detection fails (should never happen
        // since we already have a valid CudaContext).
        let sm_target = ctx.sm_target().unwrap_or_else(|_| "sm_70".to_string());

        // entrenar#318: Forward uses TF32 tensor cores (~41x faster than SIMD on sm_89).
        // ALB-076: TF32 is safe for forward (NoTrans/NoTrans). Backward uses SIMD handle.
        let cublas = match CublasHandle::new_with_tensor_cores(&ctx) {
            Ok(handle) => {
                eprintln!("[CUDA] cuBLAS initialized — forward TF32 tensor cores (41x vs SIMD)");
                Some(handle)
            }
            Err(e) => {
                eprintln!("[CUDA] cuBLAS not available ({e:?}), using PTX GEMMs");
                None
            }
        };

        eprintln!("[CUDA] Kernel cache initialized for target: {sm_target}");
        Self { ctx, modules: HashMap::new(), sm_target, cublas }
    }

    /// Get a reference to the cuBLAS handle, if available.
    pub(super) fn cublas(&self) -> Option<&CublasHandle> {
        self.cublas.as_ref()
    }

    /// Bind cuBLAS to a stream for the current training step.
    pub(super) fn set_cublas_stream(&self, stream: &CudaStream) -> Result<()> {
        if let Some(ref handle) = self.cublas {
            handle.set_stream(stream).map_err(|e| {
                CudaTensorError::KernelError(format!("cuBLAS set_stream failed: {e:?}"))
            })?;
        }
        Ok(())
    }

    /// Get the device SM target for PTX emission.
    ///
    /// Consumers MUST use this to emit PTX via `kernel.emit_ptx_for_target(cache.sm_target())`.
    pub(super) fn sm_target(&self) -> &str {
        &self.sm_target
    }

    /// Look up a previously compiled module by key (KAIZEN-058).
    ///
    /// Returns `Some` if the module is already cached (post-pre-warm: always).
    /// Callers should use this before generating PTX to avoid unnecessary
    /// multi-KB String allocations (~1000 per training step).
    pub(super) fn get_cached(&mut self, name: &str) -> Option<&mut CudaModule> {
        self.modules.get_mut(name)
    }

    /// Compile PTX and cache the resulting module.
    ///
    /// # Contract: F-PTX-001 (Target Parity)
    ///
    /// Validates that the PTX `.target` directive matches the device's compute
    /// capability. Rejects PTX compiled for the wrong architecture.
    pub(super) fn get_or_compile(&mut self, name: &str, ptx: &str) -> Result<&mut CudaModule> {
        use std::collections::hash_map::Entry;

        // F-PTX-001: Validate PTX target matches device
        if let Some(target_line) = ptx.lines().find(|l| l.starts_with(".target ")) {
            let ptx_target = target_line.trim().trim_start_matches(".target ");
            if ptx_target != self.sm_target {
                return Err(CudaTensorError::KernelError(format!(
                    "F-PTX-001 violated: PTX target '{ptx_target}' != device target '{}'. \
                     Use kernel.emit_ptx_for_target(\"{}\") instead of emit_ptx().",
                    self.sm_target, self.sm_target
                )));
            }
        }

        match self.modules.entry(name.to_string()) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                // PMAT-698i: diagnostic logging. Surfaces every forward-cache
                // JIT event with its kernel name so missing pre-warm entries
                // are identifiable in O(1) instead of O(N) iterations.
                eprintln!("[FWD-CACHE] Compiling '{name}' (ptx_len={})", ptx.len());
                // trueno#200: Use from_ptx_direct on Blackwell
                let (major, _) = self.ctx.compute_capability().map_err(|e| {
                    CudaTensorError::KernelError(format!("compute_capability: {e:?}"))
                })?;
                let module = if major >= 12 {
                    CudaModule::from_ptx_direct(&self.ctx, ptx)
                } else {
                    CudaModule::from_ptx(&self.ctx, ptx)
                }
                .map_err(|err| {
                    CudaTensorError::KernelError(format!("Failed to compile {name}: {err:?}"))
                })?;
                eprintln!("[FWD-CACHE] OK '{name}'");
                Ok(e.insert(module))
            }
        }
    }

    /// Pre-warm all kernels needed for transformer forward pass.
    ///
    /// # Contract: C-PREWARM-001 (JIT Before Payload)
    ///
    /// - **Precondition**: Kernel cache initialized, GPU VRAM mostly free (no blocks uploaded yet)
    /// - **Postcondition**: All forward-pass PTX modules JIT-compiled and cached
    /// - **Invariant**: Subsequent `get_or_compile()` calls for these keys hit cache (zero JIT)
    ///
    /// CUDA's `cuModuleLoadDataEx` JIT compiler needs device memory for compilation.
    /// If called after uploading 36 transformer blocks (~22 GB), the near-OOM state causes
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` during JIT (trueno#107). Pre-warming compiles all PTX
    /// while VRAM is free, avoiding this failure mode entirely.
    pub(super) fn pre_warm_for_model(
        &mut self,
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<()> {
        let s = max_seq_len as u32;
        let h = hidden_size as u32;
        let q_dim = (num_heads * head_dim) as u32; // Q/O projection dim (may differ from h)
        let kv_h = (num_kv_heads * head_dim) as u32;
        let i = intermediate_size as u32;
        let nh = num_heads as u32;
        let _nkv = num_kv_heads as u32;
        let hd = head_dim as u32;
        let sh = s * h; // seq_len * hidden_size
        let si = s * i; // seq_len * intermediate_size

        let mut count = 0u32;
        let target = self.sm_target.clone();

        // Helper: generate PTX and compile.
        //
        // PMAT-698j: previously hardcoded "silu_forward" as the cache key,
        // which meant every warm!() call collided on the same HashMap entry.
        // Only the FIRST kernel compiled actually got stored; all subsequent
        // warm!() invocations short-circuited because "silu_forward" was
        // already occupied. At runtime every other kernel (rmsnorm, rope,
        // softmax, swiglu, residual, etc.) cache-missed under its real key
        // and JIT-compiled mid-training — on Blackwell sm_121 that
        // corrupted the CUDA stream and surfaced as the cascading "Block 0
        // upload failed" / "forward_backward_with_grad returned None"
        // errors hunted across PMAT-698e..i.
        //
        // Discovered by PMAT-698i diagnostic logging: [FWD-CACHE] showed
        // every "pre-warmed" kernel actually JIT'd at first use because
        // the cache only contained one entry. One-character fix.
        macro_rules! warm {
            ($key:expr, $kernel:expr) => {{
                let key = $key;
                let ptx = $kernel.emit_ptx_for_target(&target);
                self.get_or_compile(&key, &ptx)?;
                count += 1;
            }};
        }

        // 1. RMSNorm (batched: single launch for all rows via grid.y)
        // ALB-076: Use BatchedVectorizedRmsNormKernel instead of per-row RmsNormKernel
        //
        // PMAT-698k: the runtime key format includes the eps as bit-pattern
        // suffix (normalization.rs:139:
        //   let key = format!("batched_rmsnorm_fwd_{hidden_size}_eps{eps_bits:08x}"))
        // Pre-warm key used to omit the eps suffix → cache miss at runtime →
        // JIT mid-forward → Blackwell sm_121 stream poisoning.
        //
        // PMAT-698n: PMAT-698k pre-warmed at eps=1e-5 (0x3727c5ac) but the
        // dominant model (Qwen2 / Qwen2.5) uses rms_norm_eps=1e-6
        // (0x358637bd). Live diagnostic confirmed the runtime key on the
        // Phase 3 dispatch was `batched_rmsnorm_fwd_896_eps358637bd`. Switch
        // the pre-warm default to 1e-6 (Qwen2 standard) AND additionally
        // pre-warm 1e-5 (Llama/Mistral standard) for cross-family coverage.
        // The cost of pre-warming both is ~30 KB of cache headroom.
        let qwen2_eps_bits = 1.0e-6_f32.to_bits(); // 0x358637bd
        let llama_eps_bits = 1.0e-5_f32.to_bits(); // 0x3727c5ac
        warm!(
            format!("batched_rmsnorm_fwd_{h}_eps{qwen2_eps_bits:08x}"),
            BatchedVectorizedRmsNormKernel::new(h, 1)
        );
        if qwen2_eps_bits != llama_eps_bits {
            warm!(
                format!("batched_rmsnorm_fwd_{h}_eps{llama_eps_bits:08x}"),
                BatchedVectorizedRmsNormKernel::new(h, 1)
            );
        }

        // 1b. Fused residual add + RMSNorm (post-attention norm in the NF4
        // QLoRA block). FALSIFY-CUDA-FUSED-RMSNORM-DEADLOCK-001 fix #4: this
        // kernel previously had NO pre-warm entry and JIT-compiled
        // mid-training (Blackwell stream-poisoning class, PMAT-698). Warm at
        // both Qwen2 (1e-6) and Llama (1e-5) eps like batched_rmsnorm_fwd.
        {
            use trueno_gpu::kernels::BatchedFusedResidualRmsNormKernel;
            for eps in [1.0e-6_f32, 1.0e-5_f32] {
                let eps_bits = eps.to_bits();
                warm!(
                    format!("batched_fused_residual_rmsnorm_{h}_eps{eps_bits:08x}"),
                    BatchedFusedResidualRmsNormKernel::new(h, 1).with_epsilon(eps)
                );
            }
        }

        // PMAT-700 (SPEC-BLACKWELL-FIX-001 Fix #2): when cuBLAS is available
        // and the runtime takes its fast path for the standard 2D GEMMs
        // (Q/K/V/O/gate/up/down projections — see ALB-075 dispatch in
        // gemm.rs:47-49 and cuda_block.rs:2895), pre-warming the PTX
        // equivalents is wasted VRAM. On sm_121 (Blackwell GB10) the
        // resulting JIT-cache footprint pushes block upload over the budget
        // and CUDA_ERROR_OUT_OF_MEMORY fires at "Block 0 upload". Skipping
        // these four pre-warms when cuBLAS is bound saves ~5-7 PTX modules
        // per cache (more on multi-block-size models) and unblocks gx10
        // dispatch without any runtime path change.
        //
        // Falsifier: F-BLACKWELL-CUBLAS-PREWARM-001 — assert the cache
        // module count after pre_warm_for_model decreases when cuBLAS is
        // present, and that runtime forward still produces identical
        // results on a known input (cuBLAS path was already taken).
        let has_cublas = self.cublas.is_some();
        if !has_cublas {
            // 2. GEMM: Q/O projections (S, H, H)
            warm!(format!("gemm_forward_{s}_{h}_{h}"), GemmKernel::naive(s, h, h));

            // 3. GEMM: K/V projections (S, H, kv_hidden)
            if kv_h != h {
                warm!(format!("gemm_forward_{s}_{h}_{kv_h}"), GemmKernel::naive(s, kv_h, h));
            }

            // 4. GEMM: gate/up projections (S, H, I)
            warm!(format!("gemm_forward_{s}_{h}_{i}"), GemmKernel::naive(s, i, h));

            // 5. GEMM: down projection (S, I, H)
            warm!(format!("gemm_forward_{s}_{i}_{h}"), GemmKernel::naive(s, h, i));
        } else {
            eprintln!("[CUDA] Skipping PTX pre-warm for 4 GEMM kernels (cuBLAS active — PMAT-700)");
        }

        // PMAT-698k + PMAT-698p: pre-warm batched_rope_fwd at BOTH seq_len=1
        // (Phase 3 single-token smoke) AND APR_DISTILL_SMOKE_SEQ_LEN
        // (default 256 — Phase 4 real-corpus seq). Runtime keys
        // (normalization.rs:339):
        //   batched_rope_fwd_{num_heads}_{head_dim}_{seq_len}_th{theta_bits:08x}
        // Stage C/D dispatch on gx10 confirmed runtime emits 2 [FWD-CACHE]
        // Compiling events post-pre-warm for rope_fwd at seq=256 — avoidable
        // JIT-cache pressure that PMAT-700-B closed for GEMMs.
        use trueno_gpu::kernels::BatchedRopeNeoxKernel;
        let qwen_theta = 1_000_000.0_f32;
        let qwen_theta_bits = qwen_theta.to_bits();
        let phase4_rope_seq: u32 = std::env::var("APR_DISTILL_SMOKE_SEQ_LEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let nkv = _nkv;
        for rope_seq in [1_u32, phase4_rope_seq] {
            warm!(
                format!("batched_rope_neox_fwd_{nh}_{hd}_{rope_seq}_th{qwen_theta_bits:08x}"),
                BatchedRopeNeoxKernel::new(nh, hd, rope_seq, qwen_theta)
            );
            if nkv != nh {
                warm!(
                    format!("batched_rope_neox_fwd_{nkv}_{hd}_{rope_seq}_th{qwen_theta_bits:08x}"),
                    BatchedRopeNeoxKernel::new(nkv, hd, rope_seq, qwen_theta)
                );
            }
        }

        // 6. Fused SwiGLU
        warm!("fused_swiglu_forward".to_string(), FusedSwigluKernel::new(si));

        // 7. Residual add (seq * hidden)
        warm!("residual_add_forward".to_string(), ResidualAddKernel::new(sh));

        // 8. Interleaved-to-batched (dimension-independent: one module handles all dims)
        warm!("interleaved_to_batched".to_string(), InterleavedToBatchedKernel::new(s, nh, hd));

        // 9. Batched transpose (dimension-independent: one module handles all dims)
        warm!("batched_transpose".to_string(), BatchedTransposeKernel::new(nh, s, hd));

        // 10. Batched 4D GEMM: Q@K^T (1, NH, S, S, HD)
        warm!(
            format!("batched_4d_gemm_1_{nh}_{s}_{s}_{hd}"),
            Batched4DGemmKernel::new(1, nh, s, s, hd)
        );

        // 11. Scale: attention scores (NH * S * S)
        let score_n = nh * s * s;
        warm!("scale_forward".to_string(), ScaleKernel::new(score_n));

        // 12. Batched softmax (dimension-independent: one module handles all dims)
        let softmax_rows = nh * s;
        warm!("batched_softmax_forward".to_string(), BatchedSoftmaxKernel::new(softmax_rows, s));

        // 13. Batched 4D GEMM: attn@V (1, NH, S, HD, S)
        warm!(
            format!("batched_4d_gemm_1_{nh}_{s}_{hd}_{s}"),
            Batched4DGemmKernel::new(1, nh, s, hd, s)
        );

        // 13b. Batched 4D GEMM: attention backward grad_V^T (1, NH, HD, S, S)
        warm!(
            format!("batched_4d_gemm_1_{nh}_{hd}_{s}_{s}"),
            Batched4DGemmKernel::new(1, nh, hd, s, s)
        );

        // 14. Batched-to-interleaved (dimension-independent: one module handles all dims)
        warm!("batched_to_interleaved".to_string(), BatchedToInterleavedKernel::new(s, nh, hd));

        // 15. Element-wise multiply (used in FFN backward for SwiGLU gate * up)
        warm!("elementwise_mul_forward".to_string(), ElementwiseMulKernel::new(si));

        // 16. SiLU forward activation (standalone, used in LoRA FFN path)
        warm!("silu_forward".to_string(), SiluKernel::new(si));

        // 17-20. NF4 quantized GEMM variants (trueno#108: QLoRA support)
        // Same 4 GEMM shapes but with Nf4GemmKernel instead of GemmKernel.
        // Only compiled if K is divisible by 64 (NF4 block size).
        if h.is_multiple_of(64) {
            // NF4 cache keys exclude M (seq_len) — PTX is shape-independent
            // (m/n/k are runtime params). Including M causes cache misses when
            // actual seq_len != max_seq_len, triggering on-demand JIT that fails
            // after GPU memory is loaded (trueno#184).
            //
            // Attention projections use q_dim (= num_heads * head_dim) which may
            // differ from hidden_size (e.g. Qwen3-4B: h=2560, q_dim=4096).
            // Q proj: input[S,h] @ W_q[h, q_dim] — key {h}_{q_dim}
            warm!(format!("nf4_gemm_forward_{h}_{q_dim}"), Nf4GemmKernel::new(s, q_dim, h));
            // O proj: input[S,q_dim] @ W_o[q_dim, h] — key {q_dim}_{h}
            if q_dim != h {
                warm!(format!("nf4_gemm_forward_{q_dim}_{h}"), Nf4GemmKernel::new(s, h, q_dim));
            }
            if kv_h != h && kv_h != q_dim && kv_h.is_multiple_of(64) {
                warm!(format!("nf4_gemm_forward_{h}_{kv_h}"), Nf4GemmKernel::new(s, kv_h, h));
            }
            if i.is_multiple_of(64) {
                warm!(format!("nf4_gemm_forward_{h}_{i}"), Nf4GemmKernel::new(s, i, h));
                warm!(format!("nf4_gemm_forward_{i}_{h}"), Nf4GemmKernel::new(s, h, i));
            }
        }

        // PMAT-475: Fused NF4 Gate+Up GEMM for FFN (shared input load).
        if h.is_multiple_of(64) && i.is_multiple_of(64) {
            use trueno_gpu::kernels::FusedNf4GateUpGemmKernel;
            warm!(format!("fused_nf4_gate_up_{h}_{i}"), FusedNf4GateUpGemmKernel::new(s, i, h));
        }
        // PMAT-478: Fused K+V GEMM for GQA attention (reuses Gate+Up kernel).
        if h.is_multiple_of(64) && kv_h.is_multiple_of(64) && kv_h != i {
            use trueno_gpu::kernels::FusedNf4GateUpGemmKernel;
            warm!(
                format!("fused_nf4_gate_up_{h}_{kv_h}"),
                FusedNf4GateUpGemmKernel::new(s, kv_h, h)
            );
        }

        // 19-22. NF4 transposed GEMM for QLoRA backward (ENT-153).
        // C[M×K] = A[M×N] @ B[K×N]^T — gradient propagation through frozen NF4 layers.
        if h.is_multiple_of(64) {
            // Q proj backward: grad[S,q_dim] @ W_q[h, q_dim]^T → [S,h]
            warm!(
                format!("nf4_gemm_transpose_{q_dim}_{h}"),
                Nf4GemmTransposeKernel::new(s, q_dim, h)
            );
            // O proj backward: grad[S,h] @ W_o[q_dim, h]^T → [S,q_dim]
            if q_dim != h {
                warm!(
                    format!("nf4_gemm_transpose_{h}_{q_dim}"),
                    Nf4GemmTransposeKernel::new(s, h, q_dim)
                );
            }
            if kv_h != h && kv_h != q_dim && kv_h.is_multiple_of(64) {
                // K/V proj backward: grad[S,kv_h] @ W_k[h, kv_h]^T → [S,h]
                warm!(
                    format!("nf4_gemm_transpose_{kv_h}_{h}"),
                    Nf4GemmTransposeKernel::new(s, kv_h, h)
                );
            }
            if i.is_multiple_of(64) {
                // Gate/Up backward: grad[S,I] @ W_gate[h,I]^T → [S,h]
                warm!(format!("nf4_gemm_transpose_{i}_{h}"), Nf4GemmTransposeKernel::new(s, i, h));
                // Down backward: grad[S,h] @ W_down[I,h]^T → [S,I]
                warm!(format!("nf4_gemm_transpose_{h}_{i}"), Nf4GemmTransposeKernel::new(s, h, i));
            }
        }

        eprintln!("[CUDA] Pre-warmed {count} forward kernels (JIT compiled before block upload)");
        Ok(())
    }

    /// Pre-warm LoRA backward GEMM kernels for QLoRA training (ENT-153).
    ///
    /// The LoRA backward uses regular fp32 GEMMs for:
    /// - Forward LoRA: x @ A → [S, R], inter @ B → [S, proj_dim]
    /// - Backward A: x^T @ grad_inter → grad_A [H, R]
    /// - Backward B: inter^T @ grad_proj → grad_B [R, proj_dim]
    /// - Backward input: grad_proj @ B^T → [S, R], then [S, R] @ A^T → [S, H]
    ///
    /// These shapes are small (rank << hidden_size) but must still be JIT-compiled.
    pub(super) fn pre_warm_lora_backward(
        &mut self,
        hidden_size: usize,
        q_dim: usize,
        kv_hidden_size: usize,
        max_seq_len: usize,
        lora_rank: usize,
    ) -> Result<()> {
        if lora_rank == 0 {
            return Ok(());
        }

        let s = max_seq_len as u32;
        let h = hidden_size as u32;
        let r = lora_rank as u32;
        let qd = q_dim as u32;
        let kv = kv_hidden_size as u32;

        let mut count = 0u32;
        let target = self.sm_target.clone();

        macro_rules! warm {
            ($key:expr, $kernel:expr) => {{
                let ptx = $kernel.emit_ptx_for_target(&target);
                self.get_or_compile(&$key, &ptx)?;
                count += 1;
            }};
        }

        // LoRA forward GEMMs (also needed in backward for activation checkpointing)
        // x[S,H] @ A[H,R] → [S,R]
        warm!(format!("gemm_forward_{s}_{h}_{r}"), GemmKernel::naive(s, r, h));
        // inter[S,R] @ B[R,qd] → [S,qd]
        warm!(format!("gemm_forward_{s}_{r}_{qd}"), GemmKernel::naive(s, qd, r));
        // inter[S,R] @ B[R,kv] → [S,kv]
        if kv != qd {
            warm!(format!("gemm_forward_{s}_{r}_{kv}"), GemmKernel::naive(s, kv, r));
        }

        // LoRA backward GEMMs (gemm_backward_a and gemm_backward_b use regular GEMM shapes)
        // grad_B = inter^T[R,S] @ grad_proj[S,qd] → [R,qd]
        // This is a GEMM with M=R, N=qd, K=S
        warm!(format!("gemm_forward_{r}_{s}_{qd}"), GemmKernel::naive(r, qd, s));
        if kv != qd {
            warm!(format!("gemm_forward_{r}_{s}_{kv}"), GemmKernel::naive(r, kv, s));
        }

        // grad_li = grad_proj[S,qd] @ B^T[qd,R] → [S,R]
        // This is effectively GEMM with M=S, N=R, K=qd
        warm!(format!("gemm_forward_{s}_{qd}_{r}"), GemmKernel::naive(s, r, qd));
        if kv != qd {
            warm!(format!("gemm_forward_{s}_{kv}_{r}"), GemmKernel::naive(s, r, kv));
        }

        // grad_A = x^T[H,S] @ grad_li[S,R] → [H,R]
        warm!(format!("gemm_forward_{h}_{s}_{r}"), GemmKernel::naive(h, r, s));

        // grad_input += grad_li[S,R] @ A^T[R,H] → [S,H]
        warm!(format!("gemm_forward_{s}_{r}_{h}"), GemmKernel::naive(s, h, r));

        eprintln!("[CUDA] Pre-warmed {count} LoRA backward kernels");
        Ok(())
    }
}

/// Initialize forward kernel cache with CUDA context
#[cfg(feature = "cuda")]
pub fn init_forward_kernel_cache(ctx: std::sync::Arc<CudaContext>) -> Result<()> {
    FORWARD_KERNEL_CACHE.get_or_init(|| Mutex::new(ForwardKernelCache::new(ctx)));
    Ok(())
}
/// Pre-allocate cuBLAS workspace for CUDA graph capture (PMAT-063).
#[cfg(feature = "cuda")]
pub fn set_cublas_workspace(ptr: u64, size: usize) -> Result<()> {
    let c = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let c = c.lock().map_err(|_| CudaTensorError::KernelError("lock".into()))?;
    if let Some(h) = c.cublas() {
        h.set_workspace(ptr, size).map_err(|e| CudaTensorError::KernelError(format!("{e}")))?;
    }
    Ok(())
}
/// Bind cuBLAS handle to a stream (ALB-075).
#[cfg(feature = "cuda")]
pub fn set_forward_cublas_stream(stream: &CudaStream) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;
    cache.set_cublas_stream(stream)
}

/// Pre-warm forward kernels (C-PREWARM-001: JIT before block upload).
#[cfg(feature = "cuda")]
pub fn pre_warm_forward_kernels(
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) -> Result<()> {
    // trueno#200: Pre-warm backward kernels too (Blackwell JIT crash workaround)
    pre_warm_backward_kernels_in_forward_cache(num_heads, num_kv_heads, head_dim, max_seq_len)?;
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;
    cache.pre_warm_for_model(
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
    )
}

/// Pre-warm backward kernels in forward cache (trueno#200 Blackwell).
///
/// CONTRACT: All backward kernels must be compiled before GPU work starts.
/// On Blackwell (sm_121), cuModuleLoadData fails during active GPU computation.
#[cfg(feature = "cuda")]
fn pre_warm_backward_kernels_in_forward_cache(
    num_heads: usize,
    _num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;

    let target = cache.sm_target.clone();
    let _nh = num_heads as u32;
    let _hd = head_dim as u32;
    let _s = max_seq_len as u32;

    macro_rules! warm {
        ($key:expr, $kernel:expr) => {{
            let ptx = $kernel.emit_ptx_for_target(&target);
            cache.get_or_compile(&$key, &ptx)?;
        }};
    }

    // Batched RoPE backward — missing from pre_warm_for_model, causes
    // CUDA context poisoning on Blackwell when compiled during backward pass.
    // Need BOTH num_heads AND num_kv_heads variants (GQA uses different head count for K/V).
    //
    // FALSIFY-CUDA-ROPE-THETA-CACHE-KEY-001: cache key now includes theta_bits
    // (matching runtime in `batched_rope_neox_backward`). The hardcoded
    // 1_000_000.0 here matches Qwen2 / Qwen2.5 default; for Llama
    // pretrain (theta=10000) the runtime call will compile its own
    // module on first use, no longer silently shadowing the Qwen warm.
    let nh = num_heads as u32;
    let nkv = _num_kv_heads as u32;
    let hd = head_dim as u32;
    let s = max_seq_len as u32;
    let qwen_theta_bits = 1_000_000.0_f32.to_bits();
    warm!(
        format!("batched_rope_neox_bwd_{nh}_{hd}_{s}_th{qwen_theta_bits:08x}"),
        BatchedRopeNeoxBackwardKernel::new(nh, hd, s, 1_000_000.0)
    );
    if nkv != nh {
        warm!(
            format!("batched_rope_neox_bwd_{nkv}_{hd}_{s}_th{qwen_theta_bits:08x}"),
            BatchedRopeNeoxBackwardKernel::new(nkv, hd, s, 1_000_000.0)
        );
    }

    eprintln!("  ✓ Backward rope kernel pre-warmed in forward cache");
    Ok(())
}

/// Pre-warm LoRA backward GEMM kernels for QLoRA training (ENT-153).
///
/// Must be called BEFORE uploading transformer blocks. Compiles the
/// small-matrix GEMMs needed for LoRA gradient computation.
#[cfg(feature = "cuda")]
pub fn pre_warm_lora_backward_kernels(
    hidden_size: usize,
    q_dim: usize,
    kv_hidden_size: usize,
    max_seq_len: usize,
    lora_rank: usize,
) -> Result<()> {
    let cache = FORWARD_KERNEL_CACHE.get().ok_or(CudaTensorError::DeviceNotInitialized)?;
    let mut cache = cache.lock().map_err(|_err| {
        CudaTensorError::KernelError("Failed to acquire kernel cache lock".to_string())
    })?;
    cache.pre_warm_lora_backward(hidden_size, q_dim, kv_hidden_size, max_seq_len, lora_rank)
}
