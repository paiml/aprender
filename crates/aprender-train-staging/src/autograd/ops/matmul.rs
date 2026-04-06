//! Matrix multiplication autograd operations
//!
//! Uses realizar's CUDA executor for GPU acceleration, falls back to trueno SIMD GEMM on CPU.
//! Both forward AND backward passes use CUDA GEMM for full GPU acceleration.
//! Instrumented with TRACER for empirical overhead analysis.

use crate::autograd::{BackwardOp, Tensor};
use crate::trace::{TraceStep, TRACER};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

#[cfg(all(feature = "realizar", feature = "cuda"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(all(feature = "realizar", feature = "cuda"))]
use std::sync::{Mutex, OnceLock};

#[cfg(all(feature = "realizar", feature = "cuda"))]
use realizar::cuda::CudaExecutor;

/// Once a realizador CUDA matmul fails (typically JIT OOM after GPU VRAM is filled
/// by NF4 block upload), disable all further attempts. Without this flag, every
/// matmul call re-attempts CUDA, fails, and falls back to CPU — producing thousands
/// of log lines per training step and adding ~100ms overhead per call.
#[cfg(all(feature = "realizar", feature = "cuda"))]
static CUDA_MATMUL_DISABLED: AtomicBool = AtomicBool::new(false);

/// Global CUDA executor (singleton, initialized once)
#[cfg(all(feature = "realizar", feature = "cuda"))]
static CUDA_EXECUTOR: OnceLock<Option<Mutex<CudaExecutor>>> = OnceLock::new();

/// Get or initialize CUDA executor
#[cfg(all(feature = "realizar", feature = "cuda"))]
fn get_cuda_executor() -> Option<&'static Mutex<CudaExecutor>> {
    CUDA_EXECUTOR
        .get_or_init(|| match CudaExecutor::new(0) {
            Ok(executor) => {
                TRACER.end(TraceStep::Transfer, "realizar CUDA executor initialized on GPU 0");
                Some(Mutex::new(executor))
            }
            Err(_e) => {
                CUDA_MATMUL_DISABLED.store(true, Ordering::Relaxed);
                None
            }
        })
        .as_ref()
}

/// Transpose a row-major matrix (rows x cols) to (cols x rows)
/// Uses cache-efficient blocked transpose for large matrices
#[inline]
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    contract_pre_transpose!(data);
    TRACER.start(TraceStep::Transpose);
    let mut transposed = vec![0.0f32; rows * cols];

    const BLOCK_SIZE: usize = 32;
    if rows >= BLOCK_SIZE && cols >= BLOCK_SIZE {
        transpose_blocked(data, &mut transposed, rows, cols, BLOCK_SIZE);
    } else {
        transpose_simple(data, &mut transposed, rows, cols);
    }

    TRACER.end(TraceStep::Transpose, format!("{rows}x{cols}"));
    transposed
}

/// Autograd-aware transpose that preserves the backward chain (KAIZEN-018).
///
/// Creates a new tensor with transposed data AND a backward op that
/// accumulates the inverse-transposed gradient on the original tensor.
/// This ensures gradient flow through LoRA weight transposes.
///
/// # Contract (C-LORA-GRAD-001)
///
/// - **Precondition**: `tensor` has shape (rows, cols) in row-major layout
/// - **Postcondition**: Returns tensor with shape (cols, rows), backward chain connected
/// - **Invariant**: `original.grad()` receives the correctly transposed gradient
pub fn transpose_tracked(tensor: &Tensor, rows: usize, cols: usize) -> Tensor {
    contract_pre_transpose_tracked!();
    let data = tensor.data();
    let slice = data.as_slice().expect("transpose_tracked: tensor must be contiguous");
    let transposed_data = transpose(slice, rows, cols);
    let mut result = Tensor::from_vec(transposed_data, tensor.requires_grad());

    if tensor.requires_grad() {
        let backward_op = Rc::new(TransposeBackward {
            original: tensor.clone(),
            rows,
            cols,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

/// Backward op for autograd-aware transpose (KAIZEN-018).
///
/// Given forward: result = transpose(original, rows, cols)
/// Backward: grad_original = transpose(grad_result, cols, rows)
/// (The inverse of an (r,c) transpose is a (c,r) transpose.)
struct TransposeBackward {
    original: Tensor,
    rows: usize,
    cols: usize,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for TransposeBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            let grad_slice = grad.as_slice().expect("gradient must be contiguous");
            // Inverse transpose: (cols, rows) → (rows, cols)
            let grad_original = transpose(grad_slice, self.cols, self.rows);
            self.original.accumulate_grad(Array1::from(grad_original));
            if let Some(op) = self.original.backward_op() {
                op.backward();
            }
        }
    }
}

/// Blocked transpose for cache efficiency on large matrices.
#[inline]
fn transpose_blocked(src: &[f32], dst: &mut [f32], rows: usize, cols: usize, block: usize) {
    for r_block in (0..rows).step_by(block) {
        for c_block in (0..cols).step_by(block) {
            let r_end = (r_block + block).min(rows);
            let c_end = (c_block + block).min(cols);
            for r in r_block..r_end {
                for c in c_block..c_end {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}

/// Simple transpose for small matrices.
#[inline]
fn transpose_simple(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/// Compute matrix multiplication using realizar CUDA if available, else SIMD CPU.
///
/// After the first CUDA failure (typically JIT OOM when VRAM is occupied by NF4
/// block uploads), all subsequent calls skip CUDA entirely and use trueno SIMD.
#[cfg(all(feature = "realizar", feature = "cuda"))]
pub fn matmul_compute(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    contract_pre_matmul!(a);
    // Fast path: skip CUDA entirely once disabled (common during QLoRA training
    // where NF4 blocks fill VRAM before realizador can JIT-compile gemm_tiled)
    if !CUDA_MATMUL_DISABLED.load(Ordering::Relaxed) {
        if let Some(executor_mutex) = get_cuda_executor() {
            if let Ok(mut executor) = executor_mutex.lock() {
                match cuda_matmul(&mut executor, a, b, m, k, n) {
                    Ok(result) => return result,
                    Err(_e) => {
                        // First failure: disable all future CUDA matmul attempts
                        CUDA_MATMUL_DISABLED.store(true, Ordering::Relaxed);
                        TRACER.end(
                            TraceStep::Matmul,
                            "realizar CUDA matmul disabled (JIT failure), using trueno SIMD",
                        );
                    }
                }
            }
        }
    }

    // wgpu GPU fallback (AMD/Intel/Apple GPUs via Vulkan/Metal/DX12)
    // KAIZEN-004: Skip per-op wgpu when batched forward pass is active
    #[cfg(feature = "gpu")]
    if !WGPU_BATCH_MODE.load(std::sync::atomic::Ordering::Relaxed) && m * k * n > 32_768 {
        if let Some(result) = wgpu_matmul(a, b, m, k, n) {
            return result;
        }
    }

    // trueno SIMD fallback (rayon-parallel if trueno/parallel enabled)
    cpu_matmul(a, b, m, k, n)
}

/// Pre-warm realizador's CUDA GEMM kernels for all training shapes.
///
/// Realizador JIT-compiles `gemm_tiled` per unique (M,K,N) shape. If compilation
/// happens after transformer block upload fills VRAM, JIT fails with
/// CUDA_ERROR_ILLEGAL_ADDRESS and `CUDA_MATMUL_DISABLED` gets set, forcing ALL
/// matmul to CPU SIMD (~100x slower).
///
/// This function pre-warms with every (M,K,N) triplet used during training:
/// - Forward: linear projections (Q,K,V,O), FFN (gate,up,down)
/// - Backward: transposed shapes for grad_A and grad_B
/// - LoRA: A and B projection shapes
/// - Classifier head
///
/// Call this BEFORE uploading transformer blocks (C-PREWARM-001).
#[cfg(all(feature = "realizar", feature = "cuda"))]
pub fn pre_warm_realizador_gemm(
    seq_len: usize,
    hidden_size: usize,
    kv_hidden_size: usize,
    intermediate_size: usize,
    lora_rank: usize,
    num_classes: usize,
) -> usize {
    let executor_mutex = match get_cuda_executor() {
        Some(e) => e,
        None => return 0,
    };
    let mut executor = match executor_mutex.lock() {
        Ok(e) => e,
        Err(_) => return 0,
    };

    // Collect all unique (M, K, N) shapes used during training
    let s = seq_len;
    let h = hidden_size;
    let kv = kv_hidden_size;
    let i = intermediate_size;
    let r = lora_rank;

    let mut shapes: Vec<(usize, usize, usize)> = vec![
        // Forward linear projections
        (s, h, h),  // Q, O projections
        (s, h, kv), // K, V projections
        (s, h, i),  // FFN gate, up
        (s, i, h),  // FFN down
        // LoRA forward
        (s, h, r),  // LoRA A (Q/O/gate/up)
        (s, r, h),  // LoRA B (Q/O)
        (s, kv, r), // LoRA A (K/V) — if kv != h
        (s, r, kv), // LoRA B (K/V)
        // Backward: grad_A = grad_C @ B^T → (M, N_fwd, K_fwd)
        // For (s,h,h): grad_A is (s,h,h) — same
        (s, kv, h), // K/V backward grad_A: (s, kv) @ (kv, h)
        (s, i, h),  // Gate/Up backward grad_A — same as FFN down forward
        (s, h, i),  // Down backward grad_A — same as FFN gate forward
        // Backward: grad_B = A^T @ grad_C → (K_fwd, M, N_fwd)
        (h, s, h),  // Q/O backward grad_B: (h, s) @ (s, h)
        (h, s, kv), // K/V backward grad_B: (h, s) @ (s, kv)
        (h, s, i),  // Gate/Up backward grad_B: (h, s) @ (s, i)
        (i, s, h),  // Down backward grad_B: (i, s) @ (s, h)
        // LoRA backward
        (s, r, h),  // LoRA A backward grad_A — same as LoRA B forward
        (h, s, r),  // LoRA A backward grad_B
        (s, h, r),  // LoRA B backward grad_A — same as LoRA A forward
        (r, s, h),  // LoRA B backward grad_B
        (r, s, kv), // LoRA B (K/V) backward grad_B
        // Classifier head
        (1, h, num_classes),
    ];

    // Deduplicate
    shapes.sort_unstable();
    shapes.dedup();
    // Remove zero-dimension shapes
    shapes.retain(|&(m, k, n)| m > 0 && k > 0 && n > 0);

    let mut warmed = 0usize;
    for &(m, k, n) in &shapes {
        let a = vec![0.0f32; m * k];
        let b = vec![0.0f32; k * n];
        match cuda_matmul(&mut executor, &a, &b, m, k, n) {
            Ok(_) => warmed += 1,
            Err(e) => {
                eprintln!("[CUDA] realizador GEMM pre-warm failed for ({m},{k},{n}): {e}");
            }
        }
    }

    if warmed == 0 {
        CUDA_MATMUL_DISABLED.store(true, Ordering::Relaxed);
    }

    warmed
}

/// CUDA matrix multiplication via realizar's CudaExecutor
#[cfg(all(feature = "realizar", feature = "cuda"))]
fn cuda_matmul(
    executor: &mut CudaExecutor,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    TRACER.start(TraceStep::Alloc);
    let mut c = vec![0.0f32; m * n];
    TRACER.end(TraceStep::Alloc, format!("{m}x{n}"));

    TRACER.start(TraceStep::Matmul);
    executor.gemm(a, b, &mut c, m as u32, n as u32, k as u32).map_err(|e| format!("{e:?}"))?;
    TRACER.end(TraceStep::Matmul, format!("{m}x{k}x{n}"));
    Ok(c)
}

/// CPU fallback using trueno SIMD GEMM
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    if let Err(_e) = trueno::blis::gemm(m, n, k, a, b, &mut c) {
        // Naive triple-loop fallback (trueno BLIS should never fail in practice)
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    c
}

/// KAIZEN-004: When WgpuForwardPass is handling the forward pass in batch mode,
/// suppress per-op wgpu matmul. Attention matmuls go to CPU SIMD instead,
/// avoiding buffer upload/download overhead and GPU contention with the batched FFN path.
#[cfg(feature = "gpu")]
static WGPU_BATCH_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Suppress per-op wgpu matmul (use CPU SIMD instead).
///
/// Call this before running attention on CPU while WgpuForwardPass handles FFN.
/// Per-op wgpu adds ~3-5ms overhead per matmul (buffer upload/compute/download).
/// For 144 attention matmuls per sample, that's 430-720ms of pure overhead.
/// CPU SIMD is equally fast and doesn't compete for GPU bandwidth.
#[cfg(feature = "gpu")]
pub fn suppress_per_op_wgpu() {
    WGPU_BATCH_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Re-enable per-op wgpu matmul.
#[cfg(feature = "gpu")]
pub fn unsuppress_per_op_wgpu() {
    WGPU_BATCH_MODE.store(false, std::sync::atomic::Ordering::Relaxed);
}

/// CPU/wgpu path (no CUDA feature)
///
/// Tries wgpu GPU matmul first (Vulkan/Metal/DX12), falls back to rayon-parallel
/// trueno BLIS GEMM on CPU. The wgpu path uses trueno's GpuDevice for cross-platform
/// GPU compute on AMD, Intel, and Apple GPUs.
#[cfg(not(all(feature = "realizar", feature = "cuda")))]
pub fn matmul_compute(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    {
        // KAIZEN-004: Skip per-op wgpu when batched forward pass is active.
        // Attention matmuls use CPU SIMD instead — equally fast, no buffer overhead.
        if !WGPU_BATCH_MODE.load(std::sync::atomic::Ordering::Relaxed) && m * k * n > 32_768 {
            if let Some(result) = wgpu_matmul(a, b, m, k, n) {
                return result;
            }
        }
    }
    cpu_matmul(a, b, m, k, n)
}

/// wgpu GPU matmul via trueno GpuDevice (Vulkan/Metal/DX12)
///
/// Uses a singleton GpuDevice to avoid per-call device creation overhead.
/// Returns None if GPU is unavailable or matmul fails (auto-fallback to CPU).
#[cfg(feature = "gpu")]
fn wgpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::OnceLock;
    static WGPU_DISABLED: AtomicBool = AtomicBool::new(false);
    static WGPU_LOGGED: AtomicBool = AtomicBool::new(false);
    static WGPU_CALLS: AtomicU64 = AtomicU64::new(0);
    static WGPU_DEVICE: OnceLock<Option<trueno::backends::gpu::GpuDevice>> = OnceLock::new();

    if WGPU_DISABLED.load(Ordering::Relaxed) {
        return None;
    }

    let device_opt = WGPU_DEVICE.get_or_init(|| {
        if !trueno::backends::gpu::GpuBackend::is_available() {
            eprintln!("[wgpu] No GPU available, using CPU");
            return None;
        }
        match trueno::backends::gpu::GpuDevice::new() {
            Ok(d) => {
                eprintln!("[wgpu] GPU device initialized for matmul");
                Some(d)
            }
            Err(e) => {
                eprintln!("[wgpu] GPU init failed: {e}, using CPU");
                None
            }
        }
    });

    let device = match device_opt.as_ref() {
        Some(d) => d,
        None => {
            WGPU_DISABLED.store(true, Ordering::Relaxed);
            return None;
        }
    };

    let mut result = vec![0.0f32; m * n];
    match device.matmul(a, b, &mut result, m, k, n) {
        Ok(()) => {
            let calls = WGPU_CALLS.fetch_add(1, Ordering::Relaxed);
            if !WGPU_LOGGED.swap(true, Ordering::Relaxed) {
                eprintln!("[wgpu] GPU matmul active ({m}x{k}x{n})");
            }
            // KAIZEN-003: Demote to 10k intervals; previous 1k floods logs
            if calls > 0 && calls.is_multiple_of(10_000) {
                eprintln!("[wgpu] {calls} GPU matmuls completed");
            }
            Some(result)
        }
        Err(_e) => {
            WGPU_DISABLED.store(true, Ordering::Relaxed);
            None
        }
    }
}

/// Matrix multiplication
///
/// Computes C = A @ B where:
/// - A is m×k (flattened to length m*k)
/// - B is k×n (flattened to length k*n)
/// - C is m×n (flattened to length m*n)
///
/// Uses GPU acceleration when available (requires `gpu` feature).
///
/// # Arguments
/// * `a` - Left matrix (m×k flattened)
/// * `b` - Right matrix (k×n flattened)
/// * `m` - Number of rows in A
/// * `k` - Number of columns in A (= rows in B)
/// * `n` - Number of columns in B
#[provable_contracts_macros::contract("matmul-v1", equation = "matmul")]
pub fn matmul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
    assert_eq!(a.len(), m * k, "Matrix A size mismatch");
    assert_eq!(b.len(), k * n, "Matrix B size mismatch");

    // Compute C = A @ B using GPU if available
    let result_data = matmul_compute(
        a.data().as_slice().expect("matrix A must be contiguous"),
        b.data().as_slice().expect("matrix B must be contiguous"),
        m,
        k,
        n,
    );

    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut result = Tensor::new(Array1::from(result_data), requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op = Rc::new(MatmulBackward {
            a: a_clone,
            b: b_clone,
            m,
            k,
            n,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    contract_post_matmul!(result.data().as_slice().unwrap_or(&[]));
    result
}

struct MatmulBackward {
    a: Tensor,
    b: Tensor,
    m: usize,
    k: usize,
    n: usize,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for MatmulBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            // ∂L/∂A = ∂L/∂C @ B^T  (m×n) @ (n×k) = (m×k)
            // ∂L/∂B = A^T @ ∂L/∂C  (k×m) @ (m×n) = (k×n)

            let grad_c = grad_output.as_slice().expect("gradient output must be contiguous");
            let a_data = self.a.data();
            let b_data = self.b.data();
            let a_slice = a_data.as_slice().expect("matrix A must be contiguous");
            let b_slice = b_data.as_slice().expect("matrix B must be contiguous");

            if self.a.requires_grad() {
                // grad_A = grad_C @ B^T
                // grad_C is (m, n), B is (k, n), B^T is (n, k)
                // Result: (m, n) @ (n, k) = (m, k)
                let b_t = transpose(b_slice, self.k, self.n);
                let grad_a = matmul_compute(grad_c, &b_t, self.m, self.n, self.k);
                self.a.accumulate_grad(Array1::from(grad_a));
            }

            if self.b.requires_grad() {
                // grad_B = A^T @ grad_C
                // A is (m, k), A^T is (k, m), grad_C is (m, n)
                // Result: (k, m) @ (m, n) = (k, n)
                let a_t = transpose(a_slice, self.m, self.k);
                let grad_b = matmul_compute(&a_t, grad_c, self.k, self.m, self.n);
                self.b.accumulate_grad(Array1::from(grad_b));
            }

            // Recursively call backward on inputs
            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
            if let Some(op) = self.b.backward_op() {
                op.backward();
            }
        }
    }
}

/// Matrix multiply with B transposed: C = A @ B^T (KAIZEN-011)
///
/// # Contract (C-MATMUL-NT-001)
///
/// - **Precondition**: A is (M, K), B is (N, K) — B's second dim matches A's second dim
/// - **Postcondition**: Output is (M, N) where C(i,j) = Σ_k A(i,k) * B(j,k)
/// - **Invariant**: Gradients flow to BOTH A and B (not transposed copies)
///
/// This is essential for LoRA where A_lora is stored as (rank, d_in) and we need
/// `x @ A_lora^T` without creating a transposed copy that breaks gradient flow.
///
/// # Backward
///
/// - `∂L/∂A = ∂L/∂C @ B`  — (M,N) @ (N,K) = (M,K)
/// - `∂L/∂B = ∂L/∂C^T @ A` — (N,M) @ (M,K) = (N,K)
#[provable_contracts_macros::contract("matmul-v1", equation = "matmul_nt")]
pub fn matmul_nt(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
    assert_eq!(
        a.len(),
        m * k,
        "Matrix A size mismatch: expected {}×{} = {}, got {}",
        m,
        k,
        m * k,
        a.len()
    );
    assert_eq!(
        b.len(),
        n * k,
        "Matrix B size mismatch: expected {}×{} = {}, got {}",
        n,
        k,
        n * k,
        b.len()
    );

    let a_slice = a.data();
    let b_slice = b.data();
    let a_data = a_slice.as_slice().expect("matrix A must be contiguous");
    let b_data = b_slice.as_slice().expect("matrix B must be contiguous");

    // C = A @ B^T: C(i,j) = Σ_k A(i,k) * B(j,k)
    let result_data = matmul_nt_compute(a_data, b_data, m, k, n);

    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut result = Tensor::new(Array1::from(result_data), requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op = Rc::new(MatmulNtBackward {
            a: a_clone,
            b: b_clone,
            m,
            k,
            n,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    contract_post_matmul!(result.data().as_slice().unwrap_or(&[]));
    result
}

/// Raw compute for C = A @ B^T using trueno SIMD GEMM
///
/// A is (M, K), B is (N, K), output is (M, N)
pub fn matmul_nt_compute(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // Transpose B to (K, N) then use standard matmul
    let b_t = transpose(b, n, k); // (N,K) → (K,N)
    cpu_matmul(a, &b_t, m, k, n)
}

struct MatmulNtBackward {
    a: Tensor,
    b: Tensor,
    m: usize,
    k: usize,
    n: usize,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for MatmulNtBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            // C = A @ B^T where A is (M,K), B is (N,K), C is (M,N)
            //
            // ∂L/∂A = ∂L/∂C @ B     (M,N) @ (N,K) = (M,K)
            // ∂L/∂B = ∂L/∂C^T @ A   (N,M) @ (M,K) = (N,K)

            let grad_c = grad_output.as_slice().expect("gradient output must be contiguous");

            if self.a.requires_grad() {
                // grad_A = grad_C @ B  (standard matmul)
                let b_data = self.b.data();
                let b_slice = b_data.as_slice().expect("matrix B must be contiguous");
                let grad_a = matmul_compute(grad_c, b_slice, self.m, self.n, self.k);
                self.a.accumulate_grad(Array1::from(grad_a));
            }

            if self.b.requires_grad() {
                // grad_B = grad_C^T @ A
                let a_data = self.a.data();
                let a_slice = a_data.as_slice().expect("matrix A must be contiguous");
                let grad_c_t = transpose(grad_c, self.m, self.n);
                let grad_b = matmul_compute(&grad_c_t, a_slice, self.n, self.m, self.k);
                self.b.accumulate_grad(Array1::from(grad_b));
            }

            // Recursively propagate
            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
            if let Some(op) = self.b.backward_op() {
                op.backward();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_identity() {
        // 1x1 matrix
        let data = vec![5.0];
        let result = transpose(&data, 1, 1);
        assert_eq!(result, vec![5.0]);
    }

    #[test]
    fn test_transpose_2x3() {
        // 2x3 matrix
        // [1, 2, 3]
        // [4, 5, 6]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose(&data, 2, 3);
        // Expected 3x2:
        // [1, 4]
        // [2, 5]
        // [3, 6]
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_3x2() {
        // 3x2 matrix
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = transpose(&data, 3, 2);
        // Expected 2x3:
        assert_eq!(result, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_matmul_compute_2x2() {
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = A @ B = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = matmul_compute(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_compute_2x3_3x2() {
        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // B = [[7, 8], [9, 10], [11, 12]] (3x2)
        // C = A @ B (2x2)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul_compute(&a, &b, 2, 3, 2);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_no_grad() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), false);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), false);
        let c = matmul(&a, &b, 2, 2, 2);
        assert!(!c.requires_grad());
        assert_eq!(
            c.data().as_slice().expect("operation should succeed"),
            &[19.0, 22.0, 43.0, 50.0]
        );
    }

    #[test]
    fn test_matmul_with_grad() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), true);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), true);
        let c = matmul(&a, &b, 2, 2, 2);
        assert!(c.requires_grad());
        assert!(c.backward_op().is_some());
    }

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), true);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), true);
        let c = matmul(&a, &b, 2, 2, 2);

        // Set gradient of output
        c.set_grad(Array1::from(vec![1.0, 1.0, 1.0, 1.0]));

        // Trigger backward
        if let Some(op) = c.backward_op() {
            op.backward();
        }

        // Check gradients are accumulated
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }

    #[test]
    fn test_matmul_a_requires_grad_only() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), true);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), false);
        let c = matmul(&a, &b, 2, 2, 2);
        assert!(c.requires_grad());

        c.set_grad(Array1::from(vec![1.0, 1.0, 1.0, 1.0]));
        if let Some(op) = c.backward_op() {
            op.backward();
        }

        assert!(a.grad().is_some());
        assert!(b.grad().is_none());
    }

    #[test]
    fn test_matmul_b_requires_grad_only() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), false);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), true);
        let c = matmul(&a, &b, 2, 2, 2);
        assert!(c.requires_grad());

        c.set_grad(Array1::from(vec![1.0, 1.0, 1.0, 1.0]));
        if let Some(op) = c.backward_op() {
            op.backward();
        }

        assert!(a.grad().is_none());
        assert!(b.grad().is_some());
    }

    #[test]
    #[should_panic(expected = "Contract [matmul] Pre-condition violated: a.len() == m * k")]
    fn test_matmul_size_mismatch_a() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0]), false);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), false);
        let _ = matmul(&a, &b, 2, 2, 2);
    }

    #[test]
    #[should_panic(expected = "Contract [matmul] Pre-condition violated: b.len() == k * n")]
    fn test_matmul_size_mismatch_b() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), false);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0]), false);
        let _ = matmul(&a, &b, 2, 2, 2);
    }

    #[test]
    fn test_transpose_double_transpose() {
        // Transpose twice should give original
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t1 = transpose(&data, 2, 3);
        let t2 = transpose(&t1, 3, 2);
        assert_eq!(data, t2);
    }

    // =========================================================================
    // FALSIFY-MM: matmul-kernel-v1.yaml contract (entrenar autograd matmul)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had 10 matmul tests but zero FALSIFY-MM-* tests
    //   Why 2: unit tests verify 2x2 cases and backward, not invariants
    //   Why 3: no mapping from matmul-kernel-v1.yaml to entrenar test names
    //   Why 4: entrenar predates the provable-contracts YAML convention
    //   Why 5: matmul was "obviously correct" (textbook GEMM + autograd)
    //
    // References:
    //   - provable-contracts/contracts/matmul-kernel-v1.yaml
    // =========================================================================

    /// FALSIFY-MM-001e: Shape correctness — output is [m, n]
    #[test]
    fn falsify_mm_001e_shape_correctness() {
        for (m, k, n) in [(2, 3, 4), (1, 5, 1), (4, 4, 4), (3, 1, 2)] {
            let result = matmul_compute(&vec![1.0; m * k], &vec![1.0; k * n], m, k, n);
            assert_eq!(
                result.len(),
                m * n,
                "FALSIFIED MM-001e: output len = {}, expected {} for ({m}x{k}) @ ({k}x{n})",
                result.len(),
                m * n
            );
        }
    }

    /// FALSIFY-MM-005e: Identity matrix — A @ I = A
    #[test]
    fn falsify_mm_005e_identity_matrix() {
        let m = 3;
        let k = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut identity = vec![0.0; k * k];
        for i in 0..k {
            identity[i * k + i] = 1.0;
        }
        let result = matmul_compute(&a, &identity, m, k, k);
        for (i, (&got, &exp)) in result.iter().zip(a.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "FALSIFIED MM-005e: (A@I)[{i}] = {got}, expected {exp}"
            );
        }
    }

    /// FALSIFY-MM-002e: Numerical accuracy against reference
    #[test]
    fn falsify_mm_002e_numerical_accuracy() {
        // 2x3 @ 3x2 known result
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let result = matmul_compute(&a, &b, 2, 3, 2);
        let expected = [58.0, 64.0, 139.0, 154.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "FALSIFIED MM-002e: result[{i}] = {got}, expected {exp}"
            );
        }
    }

    // =========================================================================
    // matmul_nt tests (KAIZEN-011)
    // =========================================================================

    #[test]
    fn test_matmul_nt_compute_2x2() {
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = A @ B^T
        // B^T = [[5, 7], [6, 8]]
        // C = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
        //   = [[17, 23], [39, 53]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = matmul_nt_compute(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![17.0, 23.0, 39.0, 53.0]);
    }

    #[test]
    fn test_matmul_nt_compute_2x3_4x3() {
        // A = [[1,2,3],[4,5,6]] (2x3), B = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]] (4x3)
        // C = A @ B^T (2x4)
        // B^T cols = rows of B
        // C(0,0) = 1*1+2*0+3*0 = 1
        // C(0,1) = 1*0+2*1+3*0 = 2
        // C(0,2) = 1*0+2*0+3*1 = 3
        // C(0,3) = 1*1+2*1+3*1 = 6
        // C(1,0) = 4*1+5*0+6*0 = 4
        // C(1,1) = 4*0+5*1+6*0 = 5
        // C(1,2) = 4*0+5*0+6*1 = 6
        // C(1,3) = 4*1+5*1+6*1 = 15
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let c = matmul_nt_compute(&a, &b, 2, 3, 4);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
    }

    #[test]
    fn test_matmul_nt_equivalence_to_transpose_matmul() {
        // Verify: matmul_nt(A, B, m, k, n) == matmul(A, B^T, m, k, n)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 2x3
        let b_t = transpose(&b, 2, 3); // 3x2

        let c_nt = matmul_nt_compute(&a, &b, 2, 3, 2);
        let c_ref = matmul_compute(&a, &b_t, 2, 3, 2);

        for (i, (&got, &exp)) in c_nt.iter().zip(c_ref.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "matmul_nt[{i}] = {got}, matmul(A, B^T)[{i}] = {exp}"
            );
        }
    }

    #[test]
    fn test_matmul_nt_backward_grad_flows_to_b() {
        // KAIZEN-011: Verify gradients flow to the ORIGINAL B tensor (not a copy)
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), false); // 2x2
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), true); // 2x2, requires_grad

        let c = matmul_nt(&a, &b, 2, 2, 2);
        assert!(c.requires_grad());

        c.set_grad(Array1::from(vec![1.0, 1.0, 1.0, 1.0]));
        if let Some(op) = c.backward_op() {
            op.backward();
        }

        // B must have received gradient
        let b_grad = b.grad().expect("KAIZEN-011: B must receive gradient from matmul_nt");

        // grad_B = grad_C^T @ A = [[1,1],[1,1]]^T @ [[1,2],[3,4]]
        // = [[1,1],[1,1]] @ [[1,2],[3,4]] = [[4,6],[4,6]]
        let expected_grad_b = vec![4.0, 6.0, 4.0, 6.0];
        for (i, (&got, &exp)) in b_grad.iter().zip(expected_grad_b.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-4, "KAIZEN-011: grad_B[{i}] = {got}, expected {exp}");
        }
    }

    #[test]
    fn test_matmul_nt_backward_grad_flows_to_a() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), true); // 2x2
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0, 8.0]), false); // 2x2

        let c = matmul_nt(&a, &b, 2, 2, 2);
        c.set_grad(Array1::from(vec![1.0, 1.0, 1.0, 1.0]));
        if let Some(op) = c.backward_op() {
            op.backward();
        }

        let a_grad = a.grad().expect("A must receive gradient");

        // grad_A = grad_C @ B = [[1,1],[1,1]] @ [[5,6],[7,8]] = [[12,14],[12,14]]
        let expected_grad_a = vec![12.0, 14.0, 12.0, 14.0];
        for (i, (&got, &exp)) in a_grad.iter().zip(expected_grad_a.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-4, "grad_A[{i}] = {got}, expected {exp}");
        }
    }

    mod mm_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // FALSIFY-MM-001e-prop: Shape correctness for random dimensions
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn falsify_mm_001e_prop_shape(
                m in 1..=8usize,
                k in 1..=8usize,
                n in 1..=8usize,
            ) {
                let result = matmul_compute(&vec![1.0; m * k], &vec![1.0; k * n], m, k, n);
                prop_assert_eq!(result.len(), m * n);
            }
        }

        // FALSIFY-MM-005e-prop: Identity matrix for random dimensions
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn falsify_mm_005e_prop_identity(
                m in 1..=6usize,
                k in 1..=6usize,
                seed in 0..500u32,
            ) {
                let a: Vec<f32> = (0..m * k)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                    .collect();
                let mut identity = vec![0.0; k * k];
                for i in 0..k {
                    identity[i * k + i] = 1.0;
                }
                let result = matmul_compute(&a, &identity, m, k, k);
                for (i, (&got, &exp)) in result.iter().zip(a.iter()).enumerate() {
                    prop_assert!(
                        (got - exp).abs() < 1e-4,
                        "FALSIFIED MM-005e-prop: (A@I)[{}] = {}, expected {}",
                        i, got, exp
                    );
                }
            }
        }

        // FALSIFY-MM-NT-001: matmul_nt equivalence to manual transpose
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn falsify_mm_nt_equivalence(
                m in 1..=6usize,
                k in 1..=6usize,
                n in 1..=6usize,
                seed in 0..500u32,
            ) {
                let a: Vec<f32> = (0..m * k)
                    .map(|i| ((i as f32 + seed as f32) * 0.31).sin())
                    .collect();
                let b: Vec<f32> = (0..n * k)
                    .map(|i| ((i as f32 + seed as f32 + 100.0) * 0.47).cos())
                    .collect();

                let c_nt = matmul_nt_compute(&a, &b, m, k, n);
                let b_t = transpose(&b, n, k);
                let c_ref = matmul_compute(&a, &b_t, m, k, n);

                for (i, (&got, &exp)) in c_nt.iter().zip(c_ref.iter()).enumerate() {
                    prop_assert!(
                        (got - exp).abs() < 1e-3,
                        "matmul_nt[{}] = {}, expected {}",
                        i, got, exp
                    );
                }
            }
        }
    }

    /// KAIZEN-018: Verify transpose_tracked backward propagates gradient
    /// to the original tensor through the inverse transpose.
    #[test]
    fn test_transpose_tracked_backward_gradient_flow() {
        // Original tensor A: 2×3 matrix [1,2,3,4,5,6], requires_grad=true
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);

        // Tracked transpose: A^T is 3×2
        let a_t = transpose_tracked(&a, 2, 3);
        assert_eq!(a_t.len(), 6);

        // Verify transposed data is correct
        let at_data = a_t.data();
        let at_slice = at_data.as_slice().expect("contiguous");
        assert_eq!(at_slice, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Set gradient on transposed tensor (as if backward computed it)
        // Gradient shape matches A^T: 3×2
        a_t.set_grad(Array1::from(vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0]));

        // Trigger backward: should transpose grad back (3×2 → 2×3) and accumulate on a
        if let Some(op) = a_t.backward_op() {
            op.backward();
        }

        // Check that the original tensor has the correctly transposed gradient
        let grad = a.grad().expect("original tensor should have gradient");
        let grad_slice = grad.as_slice().expect("contiguous");
        // Transpose of 3×2 [10,40,20,50,30,60] = 2×3 [10,20,30,40,50,60]
        assert_eq!(grad_slice, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    /// KAIZEN-018: Verify that transpose_tracked + matmul backward flows
    /// gradient to the original (non-transposed) LoRA parameter.
    #[test]
    fn test_transpose_tracked_lora_gradient_chain() {
        // Simulate LoRA forward: y = x @ A^T where A is (rank=2, d_in=3)
        let lora_a = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], true);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true); // 1×3 input

        // Tracked transpose: A^T is (d_in=3, rank=2)
        let lora_a_t = transpose_tracked(&lora_a, 2, 3);

        // Matmul: (1, 3) @ (3, 2) = (1, 2)
        let result = matmul(&x, &lora_a_t, 1, 3, 2);
        assert_eq!(result.len(), 2);

        // Set gradient on result (as if loss backward computed it)
        result.set_grad(Array1::from(vec![1.0, 1.0]));

        // Trigger backward chain: result → matmul backward → lora_a_t → transpose backward → lora_a
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        // The original lora_a should now have a gradient
        let grad = lora_a.grad().expect("LoRA A should receive gradient via transpose_tracked");
        assert_eq!(grad.len(), 6);

        // Verify gradient is finite and non-zero
        for (i, &val) in grad.as_slice().expect("contiguous").iter().enumerate() {
            assert!(val.is_finite(), "Gradient element {i} is not finite: {val}");
        }
        let grad_sum: f32 = grad.iter().sum();
        assert!(grad_sum.abs() > 1e-6, "Gradient should be non-zero, got sum={grad_sum}");
    }
}
