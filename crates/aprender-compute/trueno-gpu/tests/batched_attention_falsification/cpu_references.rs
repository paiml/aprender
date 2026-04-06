//! CPU reference implementations for batched attention pipeline verification.

/// CPU reference: interleaved to batched layout conversion
pub fn cpu_interleaved_to_batched(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d_model = n_heads * head_dim;
    let mut output = vec![0.0f32; seq_len * d_model];

    for s in 0..seq_len {
        for h in 0..n_heads {
            for d in 0..head_dim {
                let in_idx = s * d_model + h * head_dim + d;
                let out_idx = h * seq_len * head_dim + s * head_dim + d;
                output[out_idx] = input[in_idx];
            }
        }
    }
    output
}

/// CPU reference: batched transpose
pub fn cpu_batched_transpose(input: &[f32], batch: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * rows * cols];

    for b in 0..batch {
        for r in 0..rows {
            for c in 0..cols {
                let in_idx = b * rows * cols + r * cols + c;
                let out_idx = b * rows * cols + c * rows + r;
                output[out_idx] = input[in_idx];
            }
        }
    }
    output
}

/// CPU reference: batched GEMM
/// A: [batch, m, k], B: [batch, k, n] -> C: [batch, m, n]
pub fn cpu_batched_gemm(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; batch * m * n];

    for ba in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a_idx = ba * m * k + i * k + kk;
                    let b_idx = ba * k * n + kk * n + j;
                    sum += a[a_idx] * b[b_idx];
                }
                let c_idx = ba * m * n + i * n + j;
                c[c_idx] = sum;
            }
        }
    }
    c
}

/// CPU reference: batched to interleaved layout conversion
pub fn cpu_batched_to_interleaved(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d_model = n_heads * head_dim;
    let mut output = vec![0.0f32; seq_len * d_model];

    for h in 0..n_heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                let in_idx = h * seq_len * head_dim + s * head_dim + d;
                let out_idx = s * d_model + h * head_dim + d;
                output[out_idx] = input[in_idx];
            }
        }
    }
    output
}
