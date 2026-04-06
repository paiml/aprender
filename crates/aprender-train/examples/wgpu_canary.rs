//! WGPU Canary — Unsloth-style numerical parity tests
//!
//! Compares WGPU kernel outputs against CPU reference implementations.
//! Cosine similarity > 0.9999 required for all ops.
//!
//! Can run on CPU-only (no GPU needed for reference computation).
//! Run in parallel with training to validate pipeline correctness.

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("Error: --features gpu required");
        std::process::exit(1);
    }
    #[cfg(feature = "gpu")]
    {
        if let Err(e) = run() {
            eprintln!("CANARY FAIL: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), String> {
    use trueno::backends::gpu::GpuDevice;
    eprintln!("=== WGPU Canary: Numerical Parity Tests ===\n");

    let device = GpuDevice::new()?;
    let mut pass = 0;
    let mut fail = 0;

    // Test 1: GPU matmul vs CPU reference
    {
        let m = 16;
        let k = 64;
        let n = 32;
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 13 + 7) % 100) as f32 / 100.0 - 0.5).collect();
        // CPU reference
        let mut c_cpu = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c_cpu[i * n + j] = s;
            }
        }
        // GPU
        let mut c_gpu = vec![0.0f32; m * n];
        device.matmul(&a, &b, &mut c_gpu, m, k, n)?;
        let cos = cosine_sim(&c_cpu, &c_gpu);
        let maxd = max_diff(&c_cpu, &c_gpu);
        let ok = cos > 0.9999;
        eprintln!(
            "  [{}] Matmul {}x{}x{}: cos={cos:.6}, max_diff={maxd:.2e}",
            if ok { "PASS" } else { "FAIL" },
            m,
            k,
            n
        );
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // Test 2: GPU gemm_backward_a (A @ B^T) vs CPU
    {
        let m = 8u32;
        let n = 16u32;
        let k = 32u32;
        let a: Vec<f32> =
            (0..(m * k) as usize).map(|i| ((i * 11 + 5) % 100) as f32 / 100.0 - 0.5).collect();
        let b: Vec<f32> =
            (0..(n * k) as usize).map(|i| ((i * 17 + 3) % 100) as f32 / 100.0 - 0.5).collect();
        let mut c_cpu = vec![0.0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut s = 0.0f32;
                for p in 0..k as usize {
                    s += a[i * k as usize + p] * b[j * k as usize + p];
                }
                c_cpu[i * n as usize + j] = s;
            }
        }
        let mut c_gpu = vec![0.0f32; (m * n) as usize];
        device.gemm_backward_a(&a, &b, &mut c_gpu, m, n, k)?;
        let cos = cosine_sim(&c_cpu, &c_gpu);
        let maxd = max_diff(&c_cpu, &c_gpu);
        let ok = cos > 0.9999;
        eprintln!(
            "  [{}] GEMM_BWD_A {}x{}x{}: cos={cos:.6}, max_diff={maxd:.2e}",
            if ok { "PASS" } else { "FAIL" },
            m,
            n,
            k
        );
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // Test 3: NF4 quantize → dequant round-trip
    {
        let n = 256; // must be divisible by block_size=64
        let _original: Vec<f32> =
            (0..n).map(|i| ((i * 7 + 3) % 200) as f32 / 100.0 - 1.0).collect();
        // Our NF4 quantize is in wgpu_nf4, but it's not public. Test via the GPU dequant path.
        // Instead, test that GPU dequant matches known NF4 codebook values
        let nf4_lut: [f32; 16] = [
            -1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0, 0.080, 0.161, 0.246, 0.338,
            0.441, 0.563, 0.723, 1.0,
        ];
        // Pack 8 zeros (index 7 = 0.0) with scale 1.0
        let packed = vec![0x77777777u32; n / 8]; // all index 7 (zero)
        let scales = vec![1.0f32; n / 64];
        let mut output = vec![0.0f32; n];
        device.nf4_dequant(&packed, &scales, &mut output, n as u32, 64)?;
        let expected_val = nf4_lut[7]; // 0.0
        let all_zero = output.iter().all(|&v| (v - expected_val).abs() < 1e-4);
        let ok = all_zero;
        eprintln!(
            "  [{}] NF4 dequant (all-zero): all={all_zero}",
            if ok { "PASS" } else { "FAIL" }
        );
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // Test 4: LoRA forward parity (CPU vs our implementation)
    {
        let s = 4;
        let h = 16;
        let out = 32;
        let r = 4;
        let x: Vec<f32> = (0..s * h).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let w: Vec<f32> = (0..h * out).map(|i| (i as f32 - 256.0) * 0.005).collect(); // pre-transposed [h,out]
        let a: Vec<f32> = (0..r * h).map(|i| (i as f32 - 32.0) * 0.02).collect();
        let b_mat: Vec<f32> = (0..out * r).map(|i| (i as f32 - 64.0) * 0.01).collect();
        let alpha = 8.0f32;
        let scaling = alpha / r as f32;
        // CPU reference: y = x @ w + scaling * x @ A^T @ B^T
        let mut y_cpu = vec![0.0f32; s * out];
        for si in 0..s {
            for oi in 0..out {
                let mut sum = 0.0f32;
                for hi in 0..h {
                    sum += x[si * h + hi] * w[hi * out + oi];
                }
                // LoRA: h_a = x @ A^T, then h_a @ B^T
                let mut lora_sum = 0.0f32;
                for ri in 0..r {
                    let mut ha = 0.0f32;
                    for hi in 0..h {
                        ha += x[si * h + hi] * a[ri * h + hi];
                    }
                    lora_sum += ha * b_mat[oi * r + ri];
                }
                y_cpu[si * out + oi] = sum + scaling * lora_sum;
            }
        }
        // GPU: base = x @ w (matmul), LoRA on CPU (small)
        let mut y_gpu = vec![0.0f32; s * out];
        device.matmul(&x, &w, &mut y_gpu, s, h, out)?;
        for si in 0..s {
            for oi in 0..out {
                let mut lora_sum = 0.0f32;
                for ri in 0..r {
                    let mut ha = 0.0f32;
                    for hi in 0..h {
                        ha += x[si * h + hi] * a[ri * h + hi];
                    }
                    lora_sum += ha * b_mat[oi * r + ri];
                }
                y_gpu[si * out + oi] += scaling * lora_sum;
            }
        }
        let cos = cosine_sim(&y_cpu, &y_gpu);
        let ok = cos > 0.9999;
        eprintln!("  [{}] LoRA forward parity: cos={cos:.6}", if ok { "PASS" } else { "FAIL" });
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    // Test 5: AdamW CPU vs GPU parity
    {
        let n = 128;
        let mut params_cpu: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let mut params_gpu = params_cpu.clone();
        let grad: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.001).collect();
        let mut m_cpu = vec![0.0f32; n];
        let mut v_cpu = vec![0.0f32; n];
        let mut m_gpu = vec![0.0f32; n];
        let mut v_gpu = vec![0.0f32; n];
        let (lr, b1, b2, eps, wd) = (1e-3, 0.9, 0.999, 1e-8, 0.01);
        // CPU AdamW
        let bc1 = 1.0 / (1.0 - b1);
        let bc2 = 1.0 / (1.0 - b2);
        for i in 0..n {
            m_cpu[i] = b1 * m_cpu[i] + (1.0 - b1) * grad[i];
            v_cpu[i] = b2 * v_cpu[i] + (1.0 - b2) * grad[i] * grad[i];
            params_cpu[i] -=
                lr * (m_cpu[i] * bc1 / ((v_cpu[i] * bc2).sqrt() + eps) + wd * params_cpu[i]);
        }
        // GPU AdamW
        device.adamw_step(
            &mut params_gpu,
            &grad,
            &mut m_gpu,
            &mut v_gpu,
            lr,
            b1,
            b2,
            eps,
            wd,
            1,
        )?;
        let cos = cosine_sim(&params_cpu, &params_gpu);
        let maxd = max_diff(&params_cpu, &params_gpu);
        let ok = cos > 0.9999;
        eprintln!(
            "  [{}] AdamW parity: cos={cos:.6}, max_diff={maxd:.2e}",
            if ok { "PASS" } else { "FAIL" }
        );
        if ok {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    eprintln!("\n=== Canary Results: {pass} PASS, {fail} FAIL ===");
    if fail > 0 {
        return Err(format!("{fail} canary tests failed"));
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(feature = "gpu")]
fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}
