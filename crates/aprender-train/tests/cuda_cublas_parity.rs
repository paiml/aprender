//! cuBLAS vs NF4 fused kernel parity tests (FALSIFY-PARITY-V2-001).
//!
//! These tests verify that the cuBLAS GEMM path and the fused NF4 kernel path
//! produce numerically equivalent results for single GEMM operations.
//!
//! After 8 failed cuBLAS integration attempts, this test suite isolates the
//! single-GEMM correctness question before any full training pipeline test.
//!
//! # Running
//!
//! These tests require a CUDA-capable GPU and must be run with:
//! ```bash
//! cargo test --features cuda -p entrenar -- --ignored test_single_gemm_parity
//! cargo test --features cuda -p entrenar -- --ignored test_backward_gemm_parity
//! cargo test --features cuda -p entrenar -- --ignored test_cublas_forward_parity
//! ```

#![allow(clippy::unwrap_used)]

#[cfg(feature = "cuda")]
mod parity_tests {
    use std::sync::Arc;

    use trueno_gpu::driver::{CudaContext, CudaStream, GpuBuffer};
    use trueno_gpu::kernels::{dequantize_nf4, quantize_nf4, Nf4Quantized, NF4_BLOCK_SIZE};

    use entrenar::autograd::cuda_forward::{
        gemm_forward, gemm_nf4_backward_a, gemm_nf4_forward, init_forward_kernel_cache,
        pre_warm_forward_kernels,
    };

    /// Deterministic pseudo-random f32 in [-1, 1] from a seed and index.
    ///
    /// Uses a simple LCG to avoid pulling in rand for GPU tests.
    fn pseudo_random(seed: u64, idx: usize) -> f32 {
        let x = seed.wrapping_mul(6364136223846793005).wrapping_add(idx as u64);
        let x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let bits = ((x >> 16) & 0x7FFF) as f32;
        (bits / 16383.5) - 1.0 // Map to [-1, 1]
    }

    /// Generate a random f32 matrix with values in [-1, 1].
    fn random_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        (0..rows * cols).map(|i| pseudo_random(seed, i)).collect()
    }

    /// Compute max absolute difference between two slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "slice length mismatch");
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// CPU reference GEMM: C[M,N] = A[M,K] @ B[K,N]
    fn cpu_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += a[row * k + i] * b[i * n + col];
                }
                c[row * n + col] = acc;
            }
        }
        c
    }

    /// Quantize a weight matrix W[N,K] (HuggingFace convention) to NF4,
    /// uploading separate data and scales buffers as expected by the fused kernel.
    ///
    /// The fused NF4 kernel expects column-major block order for B[K,N]:
    /// scales[col * num_k_blocks + block_idx], which is the same as
    /// row-major block order of W[N,K] when quantized flat.
    fn quantize_and_upload_nf4(
        ctx: &CudaContext,
        weights: &[f32],
        total: usize,
    ) -> (GpuBuffer<u8>, GpuBuffer<f32>, Nf4Quantized) {
        assert_eq!(weights.len(), total);
        assert!(
            total.is_multiple_of(NF4_BLOCK_SIZE),
            "total {total} not divisible by NF4 block size"
        );

        let q = quantize_nf4(weights, total / NF4_BLOCK_SIZE, NF4_BLOCK_SIZE);
        let nf4_buf = GpuBuffer::from_host(ctx, &q.data).unwrap();
        let scales_buf = GpuBuffer::from_host(ctx, &q.scales).unwrap();
        (nf4_buf, scales_buf, q)
    }

    /// Dequantize NF4 weights and transpose from [N,K] to [K,N] for cuBLAS path.
    ///
    /// This replicates the `dequant_transpose_upload` closure from cuda_block.rs.
    fn dequant_transpose_upload(
        ctx: &CudaContext,
        q: &Nf4Quantized,
        n: usize,
        k: usize,
    ) -> GpuBuffer<f32> {
        let deq = dequantize_nf4(q);
        assert_eq!(deq.len(), n * k);

        // Transpose [N,K] -> [K,N]
        let mut transposed = vec![0.0f32; n * k];
        for row in 0..n {
            for col in 0..k {
                transposed[col * n + row] = deq[row * k + col];
            }
        }
        GpuBuffer::from_host(ctx, &transposed).unwrap()
    }

    /// Initialize CUDA context, stream, and kernel cache for tests.
    ///
    /// Returns (ctx, stream) ready for GEMM operations.
    fn init_cuda() -> (Arc<CudaContext>, CudaStream) {
        let ctx = Arc::new(CudaContext::new(0).expect("CUDA device 0 required"));
        let stream = CudaStream::new(&ctx).expect("Failed to create CUDA stream");

        // Initialize kernel cache (idempotent — safe to call multiple times)
        init_forward_kernel_cache(ctx.clone()).expect("Failed to init kernel cache");

        // Pre-warm kernels for our test dimensions
        // hidden=64, intermediate=128, heads=4, kv_heads=1, head_dim=16, seq=8
        pre_warm_forward_kernels(64, 128, 4, 1, 16, 8).expect("Failed to pre-warm kernels");

        (ctx, stream)
    }

    // ========================================================================
    // Test 1: Single GEMM parity (forward)
    // ========================================================================

    /// FALSIFY-PARITY-V2-001: Single GEMM parity between NF4 fused and cuBLAS paths.
    ///
    /// Creates random A[8, 64] and W[64, 64] (representing q_proj weights).
    /// - NF4 path: quantize W, run gemm_nf4_forward
    /// - cuBLAS path: dequantize W, transpose, run gemm_forward
    /// - Compare: max |fused - cublas| < 0.1
    ///
    /// NF4 quantization introduces ~0.05 error per element, so 0.1 tolerance
    /// accounts for accumulated quantization noise across the K dimension.
    #[test]
    #[ignore] // Requires GPU — run with: cargo test --features cuda -- --ignored
    fn test_single_gemm_parity() {
        let (ctx, stream) = init_cuda();

        let m: usize = 8; // seq_len
        let k: usize = 64; // hidden_size (must be divisible by 64 for NF4)
        let n: usize = 64; // output dim

        // Generate random activation A[M, K] and weight W[N, K] (HuggingFace layout)
        let a_host = random_matrix(m, k, 42);
        let w_host = random_matrix(n, k, 137); // W is [N, K] = [64, 64]

        // --- NF4 fused path ---
        let (nf4_data, nf4_scales, nf4_q) = quantize_and_upload_nf4(&ctx, &w_host, n * k);

        let a_gpu = GpuBuffer::from_host(&ctx, &a_host).unwrap();
        let mut c_fused = GpuBuffer::<f32>::new(&ctx, m * n).unwrap();

        gemm_nf4_forward(
            &a_gpu,
            &nf4_data,
            &nf4_scales,
            &mut c_fused,
            m as u32,
            k as u32,
            n as u32,
            &stream,
        )
        .expect("gemm_nf4_forward failed");

        stream.synchronize().unwrap();

        let mut c_fused_host = vec![0.0f32; m * n];
        c_fused.copy_to_host(&mut c_fused_host).unwrap();

        // --- cuBLAS path (dequantize + transpose + gemm_forward) ---
        // W_t[K, N] = transpose(dequant(W_nf4))
        let w_t_gpu = dequant_transpose_upload(&ctx, &nf4_q, n, k);

        let mut c_cublas = GpuBuffer::<f32>::new(&ctx, m * n).unwrap();

        // gemm_forward: C[M,N] = A[M,K] @ B[K,N]
        gemm_forward(&a_gpu, &w_t_gpu, &mut c_cublas, m as u32, k as u32, n as u32, &stream)
            .expect("gemm_forward failed");

        stream.synchronize().unwrap();

        let mut c_cublas_host = vec![0.0f32; m * n];
        c_cublas.copy_to_host(&mut c_cublas_host).unwrap();

        // --- CPU reference (using dequantized weights for ground truth) ---
        let w_deq = dequantize_nf4(&nf4_q);
        // Transpose [N,K] -> [K,N] for CPU GEMM: C = A[M,K] @ W_t[K,N]
        let mut w_t_cpu = vec![0.0f32; k * n];
        for row in 0..n {
            for col in 0..k {
                w_t_cpu[col * n + row] = w_deq[row * k + col];
            }
        }
        let c_ref = cpu_gemm(&a_host, &w_t_cpu, m, k, n);

        // --- Compare ---
        let diff_fused_cublas = max_abs_diff(&c_fused_host, &c_cublas_host);
        let diff_fused_ref = max_abs_diff(&c_fused_host, &c_ref);
        let diff_cublas_ref = max_abs_diff(&c_cublas_host, &c_ref);

        eprintln!("=== Single GEMM Parity (forward) ===");
        eprintln!("  Dimensions: A[{m},{k}] @ W_nf4[{k},{n}]");
        eprintln!("  max |fused - cublas|  = {diff_fused_cublas:.6}");
        eprintln!("  max |fused - cpu_ref| = {diff_fused_ref:.6}");
        eprintln!("  max |cublas - cpu_ref| = {diff_cublas_ref:.6}");
        eprintln!("  c_fused[:5]  = {:?}", &c_fused_host[..5.min(c_fused_host.len())]);
        eprintln!("  c_cublas[:5] = {:?}", &c_cublas_host[..5.min(c_cublas_host.len())]);
        eprintln!("  c_ref[:5]    = {:?}", &c_ref[..5.min(c_ref.len())]);

        // Fused vs cuBLAS should match closely (both use same dequantized values,
        // but different compute paths). Tolerance 0.1 accounts for NF4 quantization
        // noise accumulated across K=64.
        assert!(
            diff_fused_cublas < 0.1,
            "PARITY FAILURE: fused vs cuBLAS max diff = {diff_fused_cublas:.6} >= 0.1. \
             The NF4 fused kernel and cuBLAS path disagree."
        );

        // Both GPU paths should match CPU reference closely
        assert!(
            diff_fused_ref < 0.1,
            "PARITY FAILURE: fused vs CPU ref max diff = {diff_fused_ref:.6} >= 0.1"
        );
        assert!(
            diff_cublas_ref < 0.1,
            "PARITY FAILURE: cuBLAS vs CPU ref max diff = {diff_cublas_ref:.6} >= 0.1"
        );

        eprintln!("  PASS: All parity checks within tolerance.");
    }

    // ========================================================================
    // Test 2: Single backward GEMM parity
    // ========================================================================

    /// FALSIFY-PARITY-V2-002: Backward GEMM parity between NF4 transpose and cuBLAS.
    ///
    /// The backward pass computes: grad_input[M,K] = grad_output[M,N] @ W[K,N]^T
    ///
    /// - NF4 path: gemm_nf4_backward_a(grad, nf4_data, nf4_scales, ...)
    /// - cuBLAS path: gemm_forward(grad, W_dequant, ...) with appropriate transpose
    /// - Compare: max |fused - cublas| < 0.1
    #[test]
    #[ignore] // Requires GPU
    fn test_backward_gemm_parity() {
        let (ctx, stream) = init_cuda();

        let m: usize = 8; // seq_len
        let k: usize = 64; // input dim (grad_input columns)
        let n: usize = 64; // output dim (grad_output columns, W rows)

        // W[N, K] = [64, 64] in HuggingFace layout
        let w_host = random_matrix(n, k, 137);
        // grad_output[M, N] = [8, 64]
        let grad_out_host = random_matrix(m, n, 999);

        // --- NF4 backward path ---
        let (nf4_data, nf4_scales, nf4_q) = quantize_and_upload_nf4(&ctx, &w_host, n * k);

        let grad_out_gpu = GpuBuffer::from_host(&ctx, &grad_out_host).unwrap();
        let mut grad_in_fused = GpuBuffer::<f32>::new(&ctx, m * k).unwrap();

        // gemm_nf4_backward_a: grad_input[M,K] = grad_output[M,N] @ dequant(W_nf4[K,N])^T
        gemm_nf4_backward_a(
            &grad_out_gpu,
            &nf4_data,
            &nf4_scales,
            &mut grad_in_fused,
            m as u32,
            n as u32,
            k as u32,
            &stream,
        )
        .expect("gemm_nf4_backward_a failed");

        stream.synchronize().unwrap();

        let mut grad_in_fused_host = vec![0.0f32; m * k];
        grad_in_fused.copy_to_host(&mut grad_in_fused_host).unwrap();

        // --- cuBLAS path ---
        // For backward: grad_input[M,K] = grad_output[M,N] @ W[N,K]
        // where W is the original [N,K] matrix (NOT transposed).
        // This is standard GEMM: C[M,K] = A[M,N] @ B[N,K].
        //
        // We dequantize W[N,K] and upload as-is (no transpose needed).
        let w_deq = dequantize_nf4(&nf4_q);
        let w_gpu = GpuBuffer::from_host(&ctx, &w_deq).unwrap();

        let mut grad_in_cublas = GpuBuffer::<f32>::new(&ctx, m * k).unwrap();

        // gemm_forward: C[M,K] = A[M,N] @ B[N,K]
        gemm_forward(
            &grad_out_gpu,
            &w_gpu,
            &mut grad_in_cublas,
            m as u32,
            n as u32, // k param in gemm_forward = shared dim = N
            k as u32, // n param in gemm_forward = output cols = K
            &stream,
        )
        .expect("gemm_forward for backward failed");

        stream.synchronize().unwrap();

        let mut grad_in_cublas_host = vec![0.0f32; m * k];
        grad_in_cublas.copy_to_host(&mut grad_in_cublas_host).unwrap();

        // --- CPU reference ---
        // grad_input[M,K] = grad_output[M,N] @ W[N,K]
        // W[N,K] dequantized is in row-major [N,K] — use directly as B[N,K]
        let grad_ref = cpu_gemm(&grad_out_host, &w_deq, m, n, k);

        // --- Compare ---
        let diff_fused_cublas = max_abs_diff(&grad_in_fused_host, &grad_in_cublas_host);
        let diff_fused_ref = max_abs_diff(&grad_in_fused_host, &grad_ref);
        let diff_cublas_ref = max_abs_diff(&grad_in_cublas_host, &grad_ref);

        eprintln!("=== Single GEMM Parity (backward) ===");
        eprintln!("  Dimensions: grad_out[{m},{n}] @ W_nf4[{n},{k}]^T -> grad_in[{m},{k}]");
        eprintln!("  max |fused - cublas|  = {diff_fused_cublas:.6}");
        eprintln!("  max |fused - cpu_ref| = {diff_fused_ref:.6}");
        eprintln!("  max |cublas - cpu_ref| = {diff_cublas_ref:.6}");
        eprintln!(
            "  grad_fused[:5]  = {:?}",
            &grad_in_fused_host[..5.min(grad_in_fused_host.len())]
        );
        eprintln!(
            "  grad_cublas[:5] = {:?}",
            &grad_in_cublas_host[..5.min(grad_in_cublas_host.len())]
        );
        eprintln!("  grad_ref[:5]    = {:?}", &grad_ref[..5.min(grad_ref.len())]);

        assert!(
            diff_fused_cublas < 0.1,
            "PARITY FAILURE: backward fused vs cuBLAS max diff = {diff_fused_cublas:.6} >= 0.1. \
             The NF4 transpose kernel and cuBLAS backward disagree."
        );
        assert!(
            diff_fused_ref < 0.1,
            "PARITY FAILURE: backward fused vs CPU ref max diff = {diff_fused_ref:.6} >= 0.1"
        );
        assert!(
            diff_cublas_ref < 0.1,
            "PARITY FAILURE: backward cuBLAS vs CPU ref max diff = {diff_cublas_ref:.6} >= 0.1"
        );

        eprintln!("  PASS: All backward parity checks within tolerance.");
    }

    // ========================================================================
    // Test 3: Full forward parity (all 7 projections through one layer)
    // ========================================================================

    /// FALSIFY-PARITY-V2-003: Full single-layer forward parity.
    ///
    /// Creates two sets of GEMM operations from the SAME NF4-quantized weights:
    /// - Path A (fused): gemm_nf4_forward for each projection
    /// - Path B (cuBLAS): dequantize+transpose+gemm_forward for each projection
    ///
    /// Tests all 7 projections that a transformer layer uses:
    /// q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    #[test]
    #[ignore] // Requires GPU
    fn test_cublas_forward_parity() {
        let (ctx, stream) = init_cuda();

        // Small test dimensions (not full Qwen3-4B)
        let hidden_size: usize = 64;
        let q_dim: usize = 64; // num_heads * head_dim = 4 * 16
        let kv_hidden: usize = 16; // num_kv_heads * head_dim = 1 * 16
        let intermediate: usize = 128;
        let seq_len: usize = 8;

        // Define all 7 projection shapes: (name, M, K, N)
        // Forward: output[M,N] = input[M,K] @ W[K,N]
        // W is [N,K] in HF layout, quantized flat, then dequant+transpose to [K,N]
        let projections = [
            ("q_proj", seq_len, hidden_size, q_dim),
            ("k_proj", seq_len, hidden_size, kv_hidden),
            ("v_proj", seq_len, hidden_size, kv_hidden),
            ("o_proj", seq_len, q_dim, hidden_size),
            ("gate_proj", seq_len, hidden_size, intermediate),
            ("up_proj", seq_len, hidden_size, intermediate),
            ("down_proj", seq_len, intermediate, hidden_size),
        ];

        let mut all_pass = true;

        for (idx, &(name, m, k, n)) in projections.iter().enumerate() {
            // Skip projections where dimensions aren't NF4-compatible
            if (n * k) % NF4_BLOCK_SIZE != 0 {
                eprintln!("  SKIP {name}: n*k={} not divisible by {NF4_BLOCK_SIZE}", n * k);
                continue;
            }

            let seed_a = 42 + idx as u64;
            let seed_w = 137 + idx as u64;

            // Generate test data
            let a_host = random_matrix(m, k, seed_a);
            let w_host = random_matrix(n, k, seed_w); // W[N,K] HuggingFace layout

            // --- NF4 fused path ---
            let (nf4_data, nf4_scales, nf4_q) = quantize_and_upload_nf4(&ctx, &w_host, n * k);
            let a_gpu = GpuBuffer::from_host(&ctx, &a_host).unwrap();
            let mut c_fused = GpuBuffer::<f32>::new(&ctx, m * n).unwrap();

            gemm_nf4_forward(
                &a_gpu,
                &nf4_data,
                &nf4_scales,
                &mut c_fused,
                m as u32,
                k as u32,
                n as u32,
                &stream,
            )
            .unwrap_or_else(|e| panic!("{name}: gemm_nf4_forward failed: {e:?}"));

            stream.synchronize().unwrap();

            let mut c_fused_host = vec![0.0f32; m * n];
            c_fused.copy_to_host(&mut c_fused_host).unwrap();

            // --- cuBLAS path ---
            let w_t_gpu = dequant_transpose_upload(&ctx, &nf4_q, n, k);
            let mut c_cublas = GpuBuffer::<f32>::new(&ctx, m * n).unwrap();

            gemm_forward(&a_gpu, &w_t_gpu, &mut c_cublas, m as u32, k as u32, n as u32, &stream)
                .unwrap_or_else(|e| panic!("{name}: gemm_forward failed: {e:?}"));

            stream.synchronize().unwrap();

            let mut c_cublas_host = vec![0.0f32; m * n];
            c_cublas.copy_to_host(&mut c_cublas_host).unwrap();

            // --- Compare ---
            let diff = max_abs_diff(&c_fused_host, &c_cublas_host);
            let pass = diff < 0.1;

            eprintln!(
                "  {name}: [{m},{k}] @ [{k},{n}] -> max|fused-cublas| = {diff:.6}  {}",
                if pass { "PASS" } else { "FAIL" }
            );

            if !pass {
                eprintln!("    c_fused[:5]  = {:?}", &c_fused_host[..5.min(c_fused_host.len())]);
                eprintln!("    c_cublas[:5] = {:?}", &c_cublas_host[..5.min(c_cublas_host.len())]);
                all_pass = false;
            }
        }

        assert!(
            all_pass,
            "PARITY FAILURE: One or more projections failed the fused vs cuBLAS parity check."
        );
        eprintln!("=== Full forward parity: ALL PASS ===");
    }

    // ========================================================================
    // Test 4: NF4 quantization round-trip sanity (no GPU required)
    // ========================================================================

    /// Verify that NF4 quantize -> dequantize produces reasonable values.
    ///
    /// This is a prerequisite for the GPU parity tests: if quantization itself
    /// is broken, GPU tests will fail for the wrong reason.
    #[test]
    #[ignore] // Grouped with GPU tests for convenience
    fn test_nf4_quantize_roundtrip_sanity() {
        let k: usize = 64;
        let n: usize = 64;
        let total = n * k;

        let weights = random_matrix(n, k, 137);
        let q = quantize_nf4(&weights, total / NF4_BLOCK_SIZE, NF4_BLOCK_SIZE);
        let deq = dequantize_nf4(&q);

        assert_eq!(deq.len(), total);

        // NF4 round-trip error should be small for normally-distributed values
        let max_err = weights
            .iter()
            .zip(deq.iter())
            .map(|(&orig, &deq_val)| (orig - deq_val).abs())
            .fold(0.0f32, f32::max);

        let mean_err: f32 = weights
            .iter()
            .zip(deq.iter())
            .map(|(&orig, &deq_val)| (orig - deq_val).abs())
            .sum::<f32>()
            / total as f32;

        eprintln!("=== NF4 Quantization Round-trip ===");
        eprintln!("  Shape: [{n}, {k}] ({total} values)");
        eprintln!("  Max absolute error: {max_err:.6}");
        eprintln!("  Mean absolute error: {mean_err:.6}");
        eprintln!("  Num blocks: {}", q.num_blocks());

        // Max error should be bounded by the codebook gap * absmax
        // For values in [-1, 1], max error ~ 0.16 * absmax
        assert!(
            max_err < 1.0,
            "NF4 round-trip max error {max_err:.6} >= 1.0 — quantization is broken"
        );
        assert!(
            mean_err < 0.3,
            "NF4 round-trip mean error {mean_err:.6} >= 0.3 — quantization quality too low"
        );
    }

    // ========================================================================
    // Test 5: Dimension sweep to catch shape-dependent bugs
    // ========================================================================

    /// Test forward GEMM parity across multiple dimension configurations.
    ///
    /// Some bugs only manifest at specific K/N ratios or when K >> N.
    /// This sweep tests the projections that have non-square shapes.
    #[test]
    #[ignore] // Requires GPU
    fn test_gemm_parity_dimension_sweep() {
        let (ctx, stream) = init_cuda();

        // All dimensions must be multiples of 64 for NF4
        let test_cases: Vec<(&str, usize, usize, usize)> = vec![
            ("square_64", 8, 64, 64),
            ("wide_128", 8, 64, 128),
            ("tall_128", 8, 128, 64),
            ("large_square", 4, 128, 128),
            ("skinny_seq1", 1, 64, 64),
            ("long_seq16", 16, 64, 64),
        ];

        let mut all_pass = true;

        for (name, m, k, n) in &test_cases {
            let a_host = random_matrix(*m, *k, 42);
            let w_host = random_matrix(*n, *k, 137);

            let (nf4_data, nf4_scales, nf4_q) = quantize_and_upload_nf4(&ctx, &w_host, *n * *k);
            let a_gpu = GpuBuffer::from_host(&ctx, &a_host).unwrap();
            let mut c_fused = GpuBuffer::<f32>::new(&ctx, *m * *n).unwrap();

            if let Err(e) = gemm_nf4_forward(
                &a_gpu,
                &nf4_data,
                &nf4_scales,
                &mut c_fused,
                *m as u32,
                *k as u32,
                *n as u32,
                &stream,
            ) {
                eprintln!("  {name}: [{m},{k}]@[{k},{n}] NF4 forward FAILED: {e:?}");
                all_pass = false;
                continue;
            }

            stream.synchronize().unwrap();

            let w_t_gpu = dequant_transpose_upload(&ctx, &nf4_q, *n, *k);
            let mut c_cublas = GpuBuffer::<f32>::new(&ctx, *m * *n).unwrap();

            if let Err(e) = gemm_forward(
                &a_gpu,
                &w_t_gpu,
                &mut c_cublas,
                *m as u32,
                *k as u32,
                *n as u32,
                &stream,
            ) {
                eprintln!("  {name}: [{m},{k}]@[{k},{n}] cuBLAS forward FAILED: {e:?}");
                all_pass = false;
                continue;
            }

            stream.synchronize().unwrap();

            let mut c_fused_host = vec![0.0f32; *m * *n];
            c_fused.copy_to_host(&mut c_fused_host).unwrap();
            let mut c_cublas_host = vec![0.0f32; *m * *n];
            c_cublas.copy_to_host(&mut c_cublas_host).unwrap();

            let diff = max_abs_diff(&c_fused_host, &c_cublas_host);
            let pass = diff < 0.1;

            eprintln!(
                "  {name}: [{m},{k}]@[{k},{n}] -> max|diff| = {diff:.6}  {}",
                if pass { "PASS" } else { "FAIL" }
            );

            if !pass {
                all_pass = false;
            }
        }

        assert!(all_pass, "PARITY FAILURE: Dimension sweep had failures.");
        eprintln!("=== Dimension sweep: ALL PASS ===");
    }
}
