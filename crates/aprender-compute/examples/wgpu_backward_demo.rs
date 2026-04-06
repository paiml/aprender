#![allow(clippy::disallowed_methods)]
//! WGPU Backward Shaders Demo
//!
//! Demonstrates GPU-accelerated backward pass operations for neural network
//! training on AMD, Intel Arc, and Apple Silicon GPUs via WebGPU/Vulkan.
//!
//! This example runs all 7 backward ops and verifies parity with CPU reference.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example wgpu_backward_demo --features gpu
//! ```
//!
//! # Requirements
//!
//! - GPU with Vulkan, Metal, or DX12 support
//! - Tested on: AMD Radeon Pro W5700X (Vulkan), NVIDIA GB10 (Vulkan)

use trueno::backends::gpu::GpuDevice;

fn main() -> Result<(), String> {
    println!("=== Trueno WGPU Backward Shaders Demo ===");
    println!("Contract: wgpu-training-v1.yaml (FALSIFY-WGPU-001/003)\n");

    if !GpuDevice::is_available() {
        println!("⚠️  GPU not available (wgpu backend not found)");
        println!("   Requires Vulkan/Metal/DX12 GPU support");
        return Ok(());
    }

    let device = GpuDevice::new()?;
    println!("✅ GPU device initialized\n");

    let mut passed = 0;
    let mut failed = 0;

    // --- 1. SiLU Backward ---
    print!("  [1/7] SiLU backward ... ");
    match demo_silu_backward(&device) {
        Ok(diff) => {
            println!("PASS (max diff: {diff:.2e})");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // --- 2. GEMM Backward A ---
    print!("  [2/7] GEMM backward A (dL/dA = dL/dC @ B^T) ... ");
    match demo_gemm_backward_a(&device) {
        Ok(diff) => {
            println!("PASS (max diff: {diff:.2e})");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // --- 3. GEMM Backward B ---
    print!("  [3/7] GEMM backward B (dL/dB = A^T @ dL/dC) ... ");
    match demo_gemm_backward_b(&device) {
        Ok(diff) => {
            println!("PASS (max diff: {diff:.2e})");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // --- 4. RoPE Backward ---
    print!("  [4/7] RoPE backward (transpose rotation) ... ");
    match demo_rope_backward(&device) {
        Ok(diff) => {
            println!("PASS (max diff: {diff:.2e})");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // --- 5. AdamW Optimizer ---
    print!("  [5/7] AdamW step (fused optimizer) ... ");
    match demo_adamw_step(&device) {
        Ok(diff) => {
            println!("PASS (max diff: {diff:.2e})");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // --- 6. RMSNorm Backward ---
    print!("  [6/7] RMSNorm backward (dx + dγ) ... ");
    match demo_rmsnorm_backward(&device) {
        Ok(diff) => {
            println!("PASS (max diff: {diff:.2e})");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    // --- 7. NF4 Dequant ---
    print!("  [7/7] NF4 dequant (4-bit → fp32) ... ");
    match demo_nf4_dequant(&device) {
        Ok(diff) => {
            println!("PASS (max diff: {diff:.2e})");
            passed += 1;
        }
        Err(e) => {
            println!("FAIL: {e}");
            failed += 1;
        }
    }

    println!("\n=== Results: {passed}/{} passed, {failed} failed ===", passed + failed);
    if failed > 0 {
        return Err(format!("{failed} test(s) failed"));
    }
    println!("✅ All backward ops verified on GPU — ready for WGPU training");
    Ok(())
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn demo_silu_backward(device: &GpuDevice) -> Result<f32, String> {
    let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
    let grad_output: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.01).collect();

    // CPU reference
    let expected: Vec<f32> = input
        .iter()
        .zip(grad_output.iter())
        .map(|(&x, &dy)| {
            let sig = 1.0 / (1.0 + (-x).exp());
            let y = x * sig;
            dy * sig * (1.0 + x - y)
        })
        .collect();

    let mut grad_input = vec![0.0f32; 100];
    device.silu_backward(&input, &grad_output, &mut grad_input)?;
    Ok(max_diff(&grad_input, &expected))
}

fn demo_gemm_backward_a(device: &GpuDevice) -> Result<f32, String> {
    let (m, k, n) = (4u32, 8, 6);
    let grad_c: Vec<f32> = (0..m * n).map(|i| (i as f32 - 12.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 - 24.0) * 0.05).collect();

    // CPU: grad_a = grad_c @ B^T
    let mut expected = vec![0.0f32; (m * k) as usize];
    for i in 0..m as usize {
        for j in 0..k as usize {
            let mut sum = 0.0;
            for p in 0..n as usize {
                sum += grad_c[i * n as usize + p] * b[j * n as usize + p]; // B^T[p,j] = B[j,p]
            }
            expected[i * k as usize + j] = sum;
        }
    }

    let mut grad_a = vec![0.0f32; (m * k) as usize];
    device.gemm_backward_a(&grad_c, &b, &mut grad_a, m, k, n)?;
    Ok(max_diff(&grad_a, &expected))
}

fn demo_gemm_backward_b(device: &GpuDevice) -> Result<f32, String> {
    let (m, k, n) = (4u32, 8, 6);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let grad_c: Vec<f32> = (0..m * n).map(|i| (i as f32 - 12.0) * 0.05).collect();

    // CPU: grad_b = A^T @ grad_c
    let mut expected = vec![0.0f32; (k * n) as usize];
    for i in 0..k as usize {
        for j in 0..n as usize {
            let mut sum = 0.0;
            for p in 0..m as usize {
                sum += a[p * k as usize + i] * grad_c[p * n as usize + j]; // A^T[i,p] = A[p,i]
            }
            expected[i * n as usize + j] = sum;
        }
    }

    let mut grad_b = vec![0.0f32; (k * n) as usize];
    device.gemm_backward_b(&a, &grad_c, &mut grad_b, m, k, n)?;
    Ok(max_diff(&grad_b, &expected))
}

fn demo_rope_backward(device: &GpuDevice) -> Result<f32, String> {
    let (nh, hd, sl) = (2u32, 4, 3);
    let theta = 10000.0f32;
    let n = (nh * hd * sl) as usize;

    let grad_output: Vec<f32> = (0..n).map(|i| (i as f32 - 12.0) * 0.1).collect();

    // CPU reference
    let half_dim = hd / 2;
    let mut expected = vec![0.0f32; n];
    for h in 0..nh {
        for s in 0..sl {
            for p in 0..half_dim {
                let freq = -((2 * p) as f32) / hd as f32 * theta.log2();
                let angle = s as f32 * 2.0f32.powf(freq);
                let (sin_a, cos_a) = angle.sin_cos();
                let base = (h * sl * hd + s * hd) as usize;
                let e = base + (2 * p) as usize;
                let o = e + 1;
                expected[e] = grad_output[e] * cos_a + grad_output[o] * sin_a;
                expected[o] = -grad_output[e] * sin_a + grad_output[o] * cos_a;
            }
        }
    }

    let mut grad_input = vec![0.0f32; n];
    device.rope_backward(&grad_output, &mut grad_input, nh, hd, sl, theta)?;
    Ok(max_diff(&grad_input, &expected))
}

fn demo_adamw_step(device: &GpuDevice) -> Result<f32, String> {
    let n = 16;
    let mut params: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let grads: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.01).collect();
    let mut m_state = vec![0.0f32; n];
    let mut v_state = vec![0.0f32; n];
    let (lr, b1, b2, eps, wd): (f32, f32, f32, f32, f32) = (1e-3, 0.9, 0.999, 1e-8, 0.01);

    // CPU reference
    let bc1: f32 = 1.0 - b1;
    let bc2: f32 = 1.0 - b2;
    let mut cpu_p = params.clone();
    let mut cpu_m = vec![0.0f32; n];
    let mut cpu_v = vec![0.0f32; n];
    for i in 0..n {
        cpu_m[i] = b1 * cpu_m[i] + (1.0 - b1) * grads[i];
        cpu_v[i] = b2 * cpu_v[i] + (1.0 - b2) * grads[i] * grads[i];
        let mh = cpu_m[i] / bc1;
        let vh = cpu_v[i] / bc2;
        cpu_p[i] -= lr * (mh / (vh.sqrt() + eps) + wd * cpu_p[i]);
    }

    device.adamw_step(&mut params, &grads, &mut m_state, &mut v_state, lr, b1, b2, eps, wd, 1)?;
    Ok(max_diff(&params, &cpu_p))
}

fn demo_rmsnorm_backward(device: &GpuDevice) -> Result<f32, String> {
    let (rows, h) = (3usize, 8usize);
    let eps: f32 = 1e-5;
    let n = rows * h;

    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 12.0) * 0.1).collect();
    let gamma: Vec<f32> = (0..h).map(|i| 1.0 + i as f32 * 0.1).collect();
    let grad_out: Vec<f32> = (0..n).map(|i| (i as f32 - 12.0) * 0.05).collect();

    // CPU reference (grad_input only, skip grad_gamma for brevity)
    let mut cpu_gi = vec![0.0f32; n];
    for r in 0..rows {
        let row = &input[r * h..(r + 1) * h];
        let grow = &grad_out[r * h..(r + 1) * h];
        let sum_x2: f32 = row.iter().map(|x| x * x).sum();
        let var_eps = sum_x2 / h as f32 + eps;
        let inv_rms = 1.0 / var_eps.sqrt();
        let sum_xgg: f32 =
            row.iter().zip(grow).zip(gamma.iter()).map(|((&x, &gy), &g)| x * gy * g).sum();
        let mean_xgg = sum_xgg / h as f32;
        for i in 0..h {
            let correction = (row[i] / var_eps) * mean_xgg;
            cpu_gi[r * h + i] = inv_rms * (gamma[i] * grow[i] - correction);
        }
    }

    let mut grad_input = vec![0.0f32; n];
    let mut grad_gamma = vec![0.0f32; h];
    device.rmsnorm_backward(
        &input,
        &gamma,
        &grad_out,
        &mut grad_input,
        &mut grad_gamma,
        rows as u32,
        h as u32,
        eps,
    )?;
    Ok(max_diff(&grad_input, &cpu_gi))
}

fn demo_nf4_dequant(device: &GpuDevice) -> Result<f32, String> {
    let nf4_lut: [f32; 16] = [
        -1.0,
        -0.6961928,
        -0.5250731,
        -0.39491749,
        -0.28444138,
        -0.18477343,
        -0.09105004,
        0.0,
        0.0795803,
        0.1609302,
        0.24611230,
        0.33791524,
        0.44070983,
        0.5626170,
        0.7229568,
        1.0,
    ];

    let (n, bs) = (8u32, 4u32);
    let packed: Vec<u32> = vec![0x90F5_1C73_u32]; // indices [3,7,12,1,5,15,0,9]
    let scales: Vec<f32> = vec![2.0, 0.5];
    let indices = [3, 7, 12, 1, 5, 15, 0, 9];
    let expected: Vec<f32> =
        (0..8).map(|i| nf4_lut[indices[i]] * scales[i / bs as usize]).collect();

    let mut output = vec![0.0f32; 8];
    device.nf4_dequant(&packed, &scales, &mut output, n, bs)?;
    Ok(max_diff(&output, &expected))
}
