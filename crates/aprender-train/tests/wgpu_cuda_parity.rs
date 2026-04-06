//! Step 0e parity gate: WgpuTrainer must match CudaTrainer
//!
//! This test runs the same GEMM forward, backward, and AdamW operations
//! on both backends and verifies the results match within tolerance.
//!
//! Run: cargo test --features "gpu,cuda" --test wgpu_cuda_parity

#[cfg(all(feature = "gpu", feature = "cuda"))]
mod parity {
    use entrenar::autograd::cuda_training::CudaTrainer;
    use entrenar::autograd::wgpu_training::WgpuTrainer;

    /// Step 0e: Forward GEMM parity
    #[test]
    fn test_forward_gemm_parity() {
        let cuda = match CudaTrainer::new() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("CUDA not available, skipping parity test");
                return;
            }
        };
        let wgpu = match WgpuTrainer::new() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("wgpu not available, skipping parity test");
                return;
            }
        };

        let m = 32u32;
        let k = 64u32;
        let n = 48u32;

        // Same input data for both
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.01).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.007).cos()).collect();

        // CUDA forward
        let a_cuda = cuda.upload(&a_data).unwrap();
        let b_cuda = cuda.upload(&b_data).unwrap();
        let mut c_cuda = cuda.zeros((m * n) as usize).unwrap();
        cuda.matmul_forward(&a_cuda, &b_cuda, &mut c_cuda, m, k, n).unwrap();
        let cuda_result = cuda.download(&c_cuda).unwrap();

        // wgpu forward
        let a_wgpu = wgpu.upload(&a_data);
        let b_wgpu = wgpu.upload(&b_data);
        let c_wgpu = wgpu.zeros((m * n) as usize);
        wgpu.matmul_forward(&a_wgpu, &b_wgpu, &c_wgpu, m, k, n);
        let wgpu_result = wgpu.download(&c_wgpu);

        // Compare
        let mut max_err: f32 = 0.0;
        for i in 0..(m * n) as usize {
            let err = (cuda_result[i] - wgpu_result[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("[PARITY] Forward GEMM max error: {max_err}");
        assert!(max_err < 0.01, "Forward GEMM parity failed: max_err={max_err} (threshold 0.01)");
    }

    /// Step 0e: Backward GEMM parity
    #[test]
    fn test_backward_gemm_parity() {
        let cuda = match CudaTrainer::new() {
            Ok(t) => t,
            Err(_) => return,
        };
        let wgpu = match WgpuTrainer::new() {
            Ok(t) => t,
            Err(_) => return,
        };

        let m = 16u32;
        let k = 32u32;
        let n = 24u32;

        let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.009).cos()).collect();
        let gc_data: Vec<f32> = (0..m * n).map(|i| ((i as f32) * 0.011).sin()).collect();

        // CUDA backward
        let a_c = cuda.upload(&a_data).unwrap();
        let b_c = cuda.upload(&b_data).unwrap();
        let gc_c = cuda.upload(&gc_data).unwrap();
        let mut ga_c = cuda.zeros((m * k) as usize).unwrap();
        let mut gb_c = cuda.zeros((k * n) as usize).unwrap();
        cuda.matmul_backward(&a_c, &b_c, &gc_c, &mut ga_c, &mut gb_c, m, k, n).unwrap();
        let ga_cuda = cuda.download(&ga_c).unwrap();
        let gb_cuda = cuda.download(&gb_c).unwrap();

        // wgpu backward
        let a_w = wgpu.upload(&a_data);
        let b_w = wgpu.upload(&b_data);
        let gc_w = wgpu.upload(&gc_data);
        let ga_w = wgpu.zeros((m * k) as usize);
        let gb_w = wgpu.zeros((k * n) as usize);
        wgpu.matmul_backward(&a_w, &b_w, &gc_w, &ga_w, &gb_w, m, k, n);
        let ga_wgpu = wgpu.download(&ga_w);
        let gb_wgpu = wgpu.download(&gb_w);

        // Compare grad_a
        let mut max_err_a: f32 = 0.0;
        for i in 0..(m * k) as usize {
            let err = (ga_cuda[i] - ga_wgpu[i]).abs();
            if err > max_err_a {
                max_err_a = err;
            }
        }

        // Compare grad_b
        let mut max_err_b: f32 = 0.0;
        for i in 0..(k * n) as usize {
            let err = (gb_cuda[i] - gb_wgpu[i]).abs();
            if err > max_err_b {
                max_err_b = err;
            }
        }

        eprintln!("[PARITY] Backward grad_a max error: {max_err_a}");
        eprintln!("[PARITY] Backward grad_b max error: {max_err_b}");
        assert!(max_err_a < 0.01, "grad_a parity failed: {max_err_a}");
        assert!(max_err_b < 0.01, "grad_b parity failed: {max_err_b}");
    }

    /// Step 0e: AdamW parity
    #[test]
    fn test_adamw_parity() {
        let mut cuda = match CudaTrainer::new() {
            Ok(t) => t,
            Err(_) => return,
        };
        let mut wgpu = match WgpuTrainer::new() {
            Ok(t) => t,
            Err(_) => return,
        };

        let n = 64;
        let params_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let grads_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let wd = 0.01;

        // CUDA AdamW
        let mut p_c = cuda.upload(&params_data).unwrap();
        let g_c = cuda.upload(&grads_data).unwrap();
        let mut m_c = cuda.zeros(n).unwrap();
        let mut v_c = cuda.zeros(n).unwrap();
        cuda.adamw_step(&mut p_c, &g_c, &mut m_c, &mut v_c, lr, beta1, beta2, eps, wd).unwrap();
        let cuda_params = cuda.download(&p_c).unwrap();

        // wgpu AdamW
        let p_w = wgpu.upload(&params_data);
        let g_w = wgpu.upload(&grads_data);
        let m_w = wgpu.zeros(n);
        let v_w = wgpu.zeros(n);
        wgpu.adamw_step(&p_w, &g_w, &m_w, &v_w, lr, beta1, beta2, eps, wd);
        let wgpu_params = wgpu.download(&p_w);

        let mut max_err: f32 = 0.0;
        for i in 0..n {
            let err = (cuda_params[i] - wgpu_params[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        eprintln!("[PARITY] AdamW max error: {max_err}");
        assert!(max_err < 1e-4, "AdamW parity failed: {max_err}");
    }
}
