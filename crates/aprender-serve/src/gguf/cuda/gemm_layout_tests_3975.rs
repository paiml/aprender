// aprender#3975: `CudaExecutor::gemm` documents B as `[k, n]`, but for `m == 1` it
// swaps in the `Gemv` kernel, which reads B as `[n, k]`. The GPU/CPU trace test
// matched "the [k,n] buffer read as [n,k]" to 7 significant digits.
//
// Every `gemm` caller is pinned here at m = 1 AND m = 2 against a CPU oracle, on
// a NON-SQUARE weight, so a layout misread cannot hide and a fix that repairs one
// side of the m = 1 / m > 1 split cannot silently break the other:
//
//   caller                                      | passes B as    | RED today at
//   --------------------------------------------|----------------|-------------
//   CudaExecutor::gemm (its own contract)       | [k, n]         | m = 1
//   CudaScheduler::matmul (GpuModel)            | [k, n]         | m = 1
//   CachedSync::batch_matmul_gpu_prefer_cuda    | [out,in]=[n,k] | m = 2
//   OwnedQuantizedModel::fused_matmul (cuda)    | [out,in]=[n,k] | m = 2
//
// Fix (quorum: explicit layout at the API): `gemm` is `[k, n]` at every m;
// `gemm_bt` / `CudaScheduler::matmul_bt` take `[n, k]`. The fused_matmul CUDA
// branch was dead (nothing set `cuda_executor`) and was deleted; its row now
// pins that attaching an executor cannot route the product through `gemm` again.
//
// GPU tests: they SKIP (with a printed line) on a host without a CUDA device, so they
// live under `gguf::cuda::`, the module path ci.yml's `cuda-unit` lane (yoga, a real
// GPU) selects. That lane is where they are a gate.
#[cfg(all(test, feature = "cuda"))]
mod gemm_layout_tests_3975 {
    const IN: usize = 4;
    const OUT: usize = 3;

    /// Row-major `W[out, in]` — the APR/GGUF weight contract.
    fn w_out_in() -> Vec<f32> {
        vec![
            1.0, 0.0, -1.0, 2.0, //
            0.5, 1.0, 0.0, -1.0, //
            2.0, -2.0, 1.0, 0.0,
        ]
    }

    /// The same weight as `[in, out]` (= `[k, n]`), what `gemm` documents for B.
    fn w_in_out() -> Vec<f32> {
        let w = w_out_in();
        (0..IN)
            .flat_map(|i| (0..OUT).map(move |o| (i, o)))
            .map(|(i, o)| w[o * IN + i])
            .collect()
    }

    /// `m` input rows; row 0 alone is the m = 1 case.
    fn x(m: usize) -> Vec<f32> {
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -3.0]][..m].concat()
    }

    /// CPU oracle: `y[r, o] = sum_i x[r, i] * W[o, i]`.
    fn oracle(x: &[f32]) -> Vec<f32> {
        let w = w_out_in();
        x.chunks(IN)
            .flat_map(|row| {
                (0..OUT)
                    .map(|o| (0..IN).map(|i| row[i] * w[o * IN + i]).sum::<f32>())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn assert_matches(got: &[f32], m: usize, caller: &str) {
        let want = oracle(&x(m));
        let off = got.iter().zip(&want).any(|(g, w)| (g - w).abs() > 1e-4);
        assert!(
            got.len() == want.len() && !off,
            "#3975 {caller} at m={m}: got {got:?}, CPU oracle y = x*W^T is {want:?} -- \
             a mismatch here is the weight read in the wrong layout"
        );
    }

    /// GPU-FREE, so it runs on every cuda-feature build: the new `GemmBtTiled`
    /// module is not empty, declares the entry the launcher looks up, and ptxas
    /// assembles it. (#3970 found a kernel whose generate_ptx returned "".)
    #[test]
    fn gemm_bt_tiled_ptx_names_its_entry_and_assembles() {
        use crate::cuda::{CudaKernels, KernelType};
        let kernels = CudaKernels::new();
        let kt = KernelType::GemmBtTiled { m: 37, n: 45, k: 70, tile_size: 16 };
        let ptx = kernels.generate_ptx(&kt);
        let name = kernels.kernel_name(&kt);
        assert!(
            ptx.contains(&format!(".visible .entry {name}(")),
            "GemmBtTiled PTX ({} bytes) does not declare the entry `{name}` the launcher looks up",
            ptx.len()
        );
        assert!(ptx.is_ascii(), "ptxas rejects non-ASCII anywhere in a module");
        let declared = crate::test_ptxas::declared_target(&ptx);
        let assemble = |ptx: &str, first: &[&str]| crate::test_ptxas::assemble(ptx, "gemm_bt", first);

        // Row 1: the real module assembles (at its declared target, or the first newer
        // arch this ptxas still defines).
        if let Err(e) = assemble(&ptx, &[declared.as_str()]) {
            panic!("ptxas rejected GemmBtTiled: {e}");
        }
        // Row 2: the fallback engages. `sm_10` is defined by no ptxas, standing in for
        // CI's yoga, whose CUDA no longer defines `sm_70` (run 35847926651).
        let used = assemble(&ptx, &["sm_10"]).expect("the unknown-arch fallback must engage");
        assert_ne!(used, "sm_10");
        // Row 3: the fallback never masks a real error.
        let bogus = ptx.replacen("ret;", "bogus.plant.u32 %r0, %r0;\n    ret;", 1);
        assert_ne!(bogus, ptx, "the plant must land");
        assert!(assemble(&bogus, &[declared.as_str()]).is_err(), "a bogus instruction must fail ptxas");
    }

    #[test]
    fn gemm_honours_its_k_n_contract_at_m1_and_m2() {
        let mut exec = crate::cuda_executor_or_skip!(0);
        for m in [1, 2] {
            let mut c = vec![0.0f32; m * OUT];
            exec.gemm(&x(m), &w_in_out(), &mut c, m as u32, OUT as u32, IN as u32)
                .expect("gemm");
            assert_matches(&c, m, "CudaExecutor::gemm");
        }
    }

    #[test]
    fn cuda_scheduler_matmul_honours_k_n_at_m1_and_m2() {
        let mut sched = crate::cuda_scheduler_or_skip!();
        for m in [1, 2] {
            let got = sched.matmul(&x(m), &w_in_out(), m, IN, OUT).expect("matmul");
            assert_matches(&got, m, "CudaScheduler::matmul (GpuModel)");
        }
    }

    /// One 4x3 fixture is an anecdote. Sweep shapes that straddle the 16- and
    /// 32-wide tiles (and m = 1) through both entry points against the oracle.
    #[test]
    fn gemm_and_gemm_bt_agree_with_the_oracle_across_tile_boundaries() {
        let mut exec = crate::cuda_executor_or_skip!(0);
        for &(m, k, n) in &[(1, 70, 45), (2, 70, 45), (17, 33, 16), (37, 70, 45), (64, 128, 96)] {
            let x: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.25).collect();
            let w: Vec<f32> = (0..n * k).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.125).collect(); // [n, k]
            let w_kn: Vec<f32> = (0..k * n).map(|j| w[(j % n) * k + j / n]).collect();
            let want: Vec<f32> = (0..m * n)
                .map(|j| (0..k).map(|i| x[(j / n) * k + i] * w[(j % n) * k + i]).sum())
                .collect();
            let (m32, n32, k32) = (m as u32, n as u32, k as u32);
            let mut kn = vec![0.0f32; m * n];
            exec.gemm(&x, &w_kn, &mut kn, m32, n32, k32).expect("gemm");
            let mut nk = vec![0.0f32; m * n];
            exec.gemm_bt(&x, &w, &mut nk, m32, n32, k32).expect("gemm_bt");
            for (name, got) in [("gemm [k,n]", &kn), ("gemm_bt [n,k]", &nk)] {
                let worst = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
                assert!(worst < 1e-3, "#3975 {name} at (m,k,n)=({m},{k},{n}): max |err| {worst}");
            }
        }
    }

    fn test_model() -> crate::gguf::OwnedQuantizedModel {
        use crate::gguf::{ArchConstraints, GGUFConfig};
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: ArchConstraints::from_architecture("llama"),
            hidden_dim: 64,
            intermediate_dim: 128,
            num_layers: 1,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 256,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            query_pre_attn_scalar: None,
            bos_token_id: None,
            eos_token_id: None,
        };
        crate::api::test_helpers::create_test_quantized_model(&config)
    }

    #[test]
    fn cached_batch_matmul_reads_a_dequantized_out_in_weight_at_m1_and_m2() {
        let cached = crate::gguf::OwnedQuantizedModelCachedSync::new(test_model());
        let has_cuda = cached.get_cuda_scheduler().map(|g| g.is_some()).unwrap_or(false);
        if !has_cuda {
            eprintln!("SKIP: no CudaScheduler -- the wgpu fallback is not what #3975 pins");
            return;
        }
        for m in [1, 2] {
            // The production callers (batch_ffn_gpu, batch_qkv_projection_gpu, ...)
            // pass `dequantize_weight(..)` output: row-major [out, in].
            let got = cached
                .batch_matmul_gpu_prefer_cuda(&x(m), &w_out_in(), m, IN, OUT)
                .expect("batch matmul");
            assert_matches(&got, m, "CachedSync::batch_matmul_gpu_prefer_cuda");
        }
    }

    #[test]
    fn fused_matmul_cuda_dequant_fallback_reads_out_in_at_m1_and_m2() {
        let exec = crate::cuda_executor_or_skip!(0);
        let mut model = test_model();
        model.cuda_executor = Some(std::sync::Mutex::new(exec));
        // F32 has no native quantized GEMV: before #3975 every m took dequant +
        // `gemm` here. That branch is deleted, so this must agree with the oracle.
        let weight = crate::gguf::OwnedQuantizedTensor {
            data: w_out_in().iter().flat_map(|v| v.to_le_bytes()).collect(),
            in_dim: IN,
            out_dim: OUT,
            qtype: crate::gguf::GGUF_TYPE_F32,
        };
        for m in [1, 2] {
            let got = model.fused_matmul(&x(m), &weight).expect("fused_matmul");
            assert_matches(&got, m, "OwnedQuantizedModel::fused_matmul (cuda)");
        }
    }
}
