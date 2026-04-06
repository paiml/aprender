//! Minimal test comparing QKV projection between CPU and GPU
//!
//! Run with: cargo test --test test_qkv_parity --features cuda -- --nocapture

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
    use realizar::gpu::adapters::AprF32ToGpuAdapter;

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn correlation(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let a_mean: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let b_mean: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let mut cov = 0.0f64;
        let mut a_var = 0.0f64;
        let mut b_var = 0.0f64;
        for i in 0..n {
            let a_d = a[i] as f64 - a_mean;
            let b_d = b[i] as f64 - b_mean;
            cov += a_d * b_d;
            a_var += a_d * a_d;
            b_var += b_d * b_d;
        }
        if a_var > 0.0 && b_var > 0.0 {
            cov / (a_var.sqrt() * b_var.sqrt())
        } else {
            0.0
        }
    }

    #[test]
    fn test_qkv_projection_parity() {
        eprintln!("\n=== QKV PROJECTION PARITY TEST ===\n");

        let hidden_dim = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = hidden_dim + 2 * kv_dim; // 64 + 64 = 128

        // Create simple input
        let input: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.1).collect();
        eprintln!("Input L2: {:.6}", l2_norm(&input));
        eprintln!("Input first 5: {:?}", &input[..5]);

        // Create QKV weight [qkv_dim, hidden_dim] row-major
        let qkv_weight: Vec<f32> = (0..qkv_dim * hidden_dim)
            .map(|i| ((i as f32) * 0.01).sin() * 0.1)
            .collect();
        eprintln!("QKV weight L2: {:.6}", l2_norm(&qkv_weight));
        eprintln!("QKV weight first 5: {:?}", &qkv_weight[..5]);

        // CPU matmul (APR style): output[o] = sum_i(input[i] * weight[o * hidden_dim + i])
        let mut cpu_output = vec![0.0f32; qkv_dim];
        for o in 0..qkv_dim {
            for i in 0..hidden_dim {
                cpu_output[o] += input[i] * qkv_weight[o * hidden_dim + i];
            }
        }
        eprintln!("\nCPU QKV output L2: {:.6}", l2_norm(&cpu_output));
        eprintln!("CPU QKV first 10: {:?}", &cpu_output[..10]);

        // GPU matmul: transpose weight to [hidden_dim, qkv_dim], then do C = A @ B
        // Transpose: [qkv_dim, hidden_dim] -> [hidden_dim, qkv_dim]
        let mut qkv_weight_t = vec![0.0f32; qkv_dim * hidden_dim];
        for i in 0..qkv_dim {
            for j in 0..hidden_dim {
                qkv_weight_t[j * qkv_dim + i] = qkv_weight[i * hidden_dim + j];
            }
        }
        eprintln!("\nTransposed weight L2: {:.6}", l2_norm(&qkv_weight_t));

        // GPU style matmul: C = A @ B where A[1, hidden_dim], B[hidden_dim, qkv_dim]
        // C[0, j] = sum_i(A[0, i] * B[i, j])
        let mut gpu_output = vec![0.0f32; qkv_dim];
        for j in 0..qkv_dim {
            for i in 0..hidden_dim {
                gpu_output[j] += input[i] * qkv_weight_t[i * qkv_dim + j];
            }
        }
        eprintln!("GPU-style QKV output L2: {:.6}", l2_norm(&gpu_output));
        eprintln!("GPU-style QKV first 10: {:?}", &gpu_output[..10]);

        // Compare
        eprintln!(
            "\nCorrelation: {:.6}",
            correlation(&cpu_output, &gpu_output)
        );
        let max_diff = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(&c, &g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Max diff: {:.6}", max_diff);

        if max_diff > 1e-5 {
            eprintln!("\nMISMATCH! Element-wise comparison:");
            for i in 0..10 {
                eprintln!(
                    "  [{}] CPU: {:.6}, GPU: {:.6}, diff: {:.6}",
                    i,
                    cpu_output[i],
                    gpu_output[i],
                    (cpu_output[i] - gpu_output[i]).abs()
                );
            }
        } else {
            eprintln!("\nPARITY VERIFIED - CPU and GPU produce same output!");
        }
    }
}
