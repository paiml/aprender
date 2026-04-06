//! WGSL backward (gradient) shaders for training
//!
//! Contract: wgpu-training-v1.yaml (FALSIFY-WGPU-001)
//!
//! Each shader computes gradients for its corresponding forward operation.
//! All shaders match the CPU reference within ε < 1e-4 (fp32).
//!
//! ## Available Backward Shaders
//!
//! - [`SILU_BACKWARD_SHADER`]: SiLU activation gradient
//! - [`GEMM_BACKWARD_A_SHADER`]: dL/dA = dL/dC @ B^T
//! - [`GEMM_BACKWARD_B_SHADER`]: dL/dB = A^T @ dL/dC
//! - [`RMSNORM_BACKWARD_SHADER`]: RMSNorm gradient (dx, dγ)
//! - [`ROPE_BACKWARD_SHADER`]: RoPE gradient (negated sin rotation)
//! - [`SOFTMAX_BACKWARD_SHADER`]: Softmax Jacobian-vector product
//! - [`CROSS_ENTROPY_BACKWARD_SHADER`]: Fused log-softmax + NLL gradient
//! - [`ADAMW_STEP_SHADER`]: AdamW optimizer step
//! - [`NF4_DEQUANT_SHADER`]: NF4 4-bit weight dequantization

// ============================================================================
// SiLU Backward: grad_x = grad_out * (σ(x) + x·σ(x)·(1 - σ(x)))
//              = grad_out * σ(x) * (1 + x - x·σ(x))
//
// Reference: Elfwing et al., "Sigmoid-Weighted Linear Units" (arXiv:1702.03118)
// ============================================================================

/// SiLU (Swish) backward shader
///
/// Forward: y = x * σ(x)
/// Backward: dy/dx = σ(x) * (1 + x - y) where y = x * σ(x)
///
/// Binding 0: input x (read)
/// Binding 1: grad_output dL/dy (read)
/// Binding 2: grad_input dL/dx (write)
/// Binding 3: uniform { n: u32 }
pub const SILU_BACKWARD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

struct Params {
    n: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x + global_id.y * 65535u * 256u;
    if (idx >= params.n) {
        return;
    }

    let x = input[idx];
    let grad_out = grad_output[idx];

    // σ(x) = 1 / (1 + exp(-x))
    let sigmoid_x = 1.0 / (1.0 + exp(-x));

    // y = x * σ(x) (forward output)
    let y = x * sigmoid_x;

    // silu'(x) = σ(x) * (1 + x - y)
    let silu_prime = sigmoid_x * (1.0 + x - y);

    grad_input[idx] = grad_out * silu_prime;
}
"#;

// ============================================================================
// GEMM Backward A: grad_A[M,K] = grad_C[M,N] @ B^T[N,K]
//
// Reuses tiled matmul pattern. The "B transposed" is handled by swapping
// the indexing: B[j,i] instead of B[i,j].
// ============================================================================

/// GEMM backward for A: dL/dA = dL/dC @ B^T
///
/// Forward: C[M,N] = A[M,K] @ B[K,N]
/// Backward: dL/dA[M,K] = dL/dC[M,N] @ B^T[N,K]
///
/// Binding 0: grad_c (dL/dC) [M*N] (read)
/// Binding 1: b [K*N] (read) — accessed transposed
/// Binding 2: grad_a (dL/dA) [M*K] (write)
/// Binding 3: uniform { M, K, N }
pub const GEMM_BACKWARD_A_SHADER: &str = r#"
const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> grad_c: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_a: array<f32>;

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

var<workgroup> tile_gc: array<f32, 256>;
var<workgroup> tile_bt: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = global_id.x;  // M dimension
    let col = global_id.y;  // K dimension
    let lr = local_id.x;
    let lc = local_id.y;

    var acc: f32 = 0.0;

    // Tile over N (reduction dimension for dA = dC @ B^T)
    let num_tiles = (dims.N + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of grad_c[row, t*TILE + lc]
        let gc_col = t * TILE + lc;
        if (row < dims.M && gc_col < dims.N) {
            tile_gc[lr * TILE + lc] = grad_c[row * dims.N + gc_col];
        } else {
            tile_gc[lr * TILE + lc] = 0.0;
        }

        // Load tile of B^T[t*TILE + lr, col] = B[col, t*TILE + lr]
        // B is stored as B[K,N] row-major, so B[k,n] = b[k*N + n]
        // B^T[n,k] = B[k,n] = b[k*N + n]
        let bt_row = t * TILE + lr;
        if (col < dims.K && bt_row < dims.N) {
            tile_bt[lr * TILE + lc] = b[col * dims.N + bt_row];
        } else {
            tile_bt[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        // Accumulate: grad_a[row, col] += sum_n grad_c[row, n] * B^T[n, col]
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc += tile_gc[lr * TILE + k] * tile_bt[k * TILE + lc];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.K) {
        grad_a[row * dims.K + col] = acc;
    }
}
"#;

/// GEMM backward for B: dL/dB = A^T @ dL/dC
///
/// Forward: C[M,N] = A[M,K] @ B[K,N]
/// Backward: dL/dB[K,N] = A^T[K,M] @ dL/dC[M,N]
///
/// Binding 0: a [M*K] (read) — accessed transposed
/// Binding 1: grad_c (dL/dC) [M*N] (read)
/// Binding 2: grad_b (dL/dB) [K*N] (write)
/// Binding 3: uniform { M, K, N }
pub const GEMM_BACKWARD_B_SHADER: &str = r#"
const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> grad_c: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_b: array<f32>;

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

var<workgroup> tile_at: array<f32, 256>;
var<workgroup> tile_gc: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = global_id.x;  // K dimension
    let col = global_id.y;  // N dimension
    let lr = local_id.x;
    let lc = local_id.y;

    var acc: f32 = 0.0;

    // Tile over M (reduction dimension for dB = A^T @ dC)
    let num_tiles = (dims.M + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A^T[row, t*TILE + lc] = A[t*TILE + lc, row]
        let at_col = t * TILE + lc;
        if (row < dims.K && at_col < dims.M) {
            tile_at[lr * TILE + lc] = a[at_col * dims.K + row];
        } else {
            tile_at[lr * TILE + lc] = 0.0;
        }

        // Load tile of grad_c[t*TILE + lr, col]
        let gc_row = t * TILE + lr;
        if (gc_row < dims.M && col < dims.N) {
            tile_gc[lr * TILE + lc] = grad_c[gc_row * dims.N + col];
        } else {
            tile_gc[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc += tile_at[lr * TILE + k] * tile_gc[k * TILE + lc];
        }

        workgroupBarrier();
    }

    if (row < dims.K && col < dims.N) {
        grad_b[row * dims.N + col] = acc;
    }
}
"#;

// ============================================================================
// RMSNorm Backward
//
// Forward: y_i = x_i / rms(x) * γ_i, where rms(x) = sqrt(mean(x²) + ε)
// Backward:
//   dL/dx_i = (1/rms) * (γ_i * dL/dy_i - x_i/rms² * mean(x · dL/dy · γ))
//   dL/dγ_i = Σ_batch (dL/dy_i * x_i / rms)
//
// Uses workgroup reduction for mean computation (one workgroup per row).
// ============================================================================

/// RMSNorm backward shader
///
/// Binding 0: input x [num_rows * hidden_dim] (read)
/// Binding 1: gamma [hidden_dim] (read)
/// Binding 2: grad_output dL/dy [num_rows * hidden_dim] (read)
/// Binding 3: grad_input dL/dx [num_rows * hidden_dim] (write)
/// Binding 4: grad_gamma dL/dγ [hidden_dim] (read_write, atomicAdd)
/// Binding 5: uniform { num_rows, hidden_dim, eps }
pub const RMSNORM_BACKWARD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_gamma: array<atomic<u32>>;

struct Params {
    num_rows: u32,
    hidden_dim: u32,
    eps_bits: u32,  // f32 eps reinterpreted as u32 (WGSL uniform limitation)
    _pad: u32,
}

@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> shared_sum_x2: array<f32, 256>;
var<workgroup> shared_sum_xgg: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let row = wg_id.x;
    let tid = local_id.x;
    let h = params.hidden_dim;
    let eps = bitcast<f32>(params.eps_bits);

    if (row >= params.num_rows) {
        return;
    }

    let row_offset = row * h;

    // Pass 1: Compute sum(x²) and sum(x·dL/dy·γ) via stride loop
    var local_sum_x2: f32 = 0.0;
    var local_sum_xgg: f32 = 0.0;

    for (var i = tid; i < h; i = i + 256u) {
        let x_val = input[row_offset + i];
        let gy_val = grad_output[row_offset + i];
        let g_val = gamma[i];

        local_sum_x2 += x_val * x_val;
        local_sum_xgg += x_val * gy_val * g_val;
    }

    shared_sum_x2[tid] = local_sum_x2;
    shared_sum_xgg[tid] = local_sum_xgg;
    workgroupBarrier();

    // Workgroup reduction (256 → 1)
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sum_x2[tid] += shared_sum_x2[tid + stride];
            shared_sum_xgg[tid] += shared_sum_xgg[tid + stride];
        }
        workgroupBarrier();
    }

    let sum_x2 = shared_sum_x2[0];
    let sum_xgg = shared_sum_xgg[0];

    // Compute rms and mean_xgg
    let h_f32 = f32(h);
    let mean_x2 = sum_x2 / h_f32;
    let variance_eps = mean_x2 + eps;
    let rms = sqrt(variance_eps);
    let inv_rms = 1.0 / rms;
    let mean_xgg = sum_xgg / h_f32;

    // Pass 2: Compute and store grad_x, accumulate grad_gamma
    for (var i = tid; i < h; i = i + 256u) {
        let x_val = input[row_offset + i];
        let gy_val = grad_output[row_offset + i];
        let g_val = gamma[i];

        // grad_x = (1/rms) * (γ * dL/dy - x/var_eps * mean_xgg)
        let gamma_gy = g_val * gy_val;
        let correction = (x_val / variance_eps) * mean_xgg;
        let grad_x = inv_rms * (gamma_gy - correction);
        grad_input[row_offset + i] = grad_x;

        // grad_gamma[i] += dL/dy * x / rms (accumulated across rows via atomic)
        let gg_contrib = gy_val * x_val * inv_rms;
        let gg_bits = bitcast<u32>(gg_contrib);
        // Atomic float add via CAS loop (WGSL doesn't have native atomicAdd for f32)
        var old_bits = atomicLoad(&grad_gamma[i]);
        loop {
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val + gg_contrib;
            let new_bits = bitcast<u32>(new_val);
            let result = atomicCompareExchangeWeak(&grad_gamma[i], old_bits, new_bits);
            if (result.exchanged) {
                break;
            }
            old_bits = result.old_value;
        }
    }
}
"#;

// ============================================================================
// RoPE Backward: same rotation but with negated sin
//
// Forward: (x_even, x_odd) → (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
// Backward: (dx_even, dx_odd) → (dx_even*cos + dx_odd*sin, -dx_even*sin + dx_odd*cos)
//
// The backward is the TRANSPOSE of the forward rotation matrix.
// ============================================================================

/// RoPE backward shader
///
/// Binding 0: grad_output [batch * num_heads * seq_len * head_dim] (read)
/// Binding 1: grad_input [same shape] (write)
/// Binding 2: uniform { num_heads, head_dim, seq_len, theta_log2 }
pub const ROPE_BACKWARD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;

struct Params {
    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
    theta_log2: f32,  // log2(theta), e.g. log2(10000) ≈ 13.29
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x + global_id.y * 65535u * 256u;
    let half_dim = params.head_dim / 2u;
    let total_pairs = params.num_heads * params.seq_len * half_dim;

    if (idx >= total_pairs) {
        return;
    }

    // Decompose idx into (head, pos, pair)
    let pair = idx % half_dim;
    let remaining = idx / half_dim;
    let pos = remaining % params.seq_len;
    let head = remaining / params.seq_len;

    // Compute rotation angle: θ_i = pos / θ^(2i/d)
    let freq_exp = -f32(2u * pair) / f32(params.head_dim) * params.theta_log2;
    let inv_freq = exp2(freq_exp);
    let angle = f32(pos) * inv_freq;
    let cos_angle = cos(angle);
    let sin_angle = sin(angle);

    // Element indices
    let base = head * params.seq_len * params.head_dim + pos * params.head_dim;
    let even_idx = base + 2u * pair;
    let odd_idx = base + 2u * pair + 1u;

    let dy_even = grad_output[even_idx];
    let dy_odd = grad_output[odd_idx];

    // Backward rotation (transpose of forward):
    // dx_even = dy_even * cos + dy_odd * sin
    // dx_odd  = -dy_even * sin + dy_odd * cos
    grad_input[even_idx] = dy_even * cos_angle + dy_odd * sin_angle;
    grad_input[odd_idx] = -dy_even * sin_angle + dy_odd * cos_angle;
}
"#;

// ============================================================================
// AdamW Optimizer Step
//
// For each parameter:
//   m = β1 * m + (1 - β1) * grad
//   v = β2 * v + (1 - β2) * grad²
//   m_hat = m / (1 - β1^t)
//   v_hat = v / (1 - β2^t)
//   param = param - lr * (m_hat / (sqrt(v_hat) + ε) + weight_decay * param)
// ============================================================================

/// AdamW optimizer step shader
///
/// Binding 0: params (read_write)
/// Binding 1: grads (read)
/// Binding 2: m (first moment, read_write)
/// Binding 3: v (second moment, read_write)
/// Binding 4: uniform { n, lr, beta1, beta2, eps, weight_decay, bc1, bc2 }
pub const ADAMW_STEP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;

struct AdamWParams {
    n: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,  // 1 - β1^t
    bias_correction2: f32,  // 1 - β2^t
}

@group(0) @binding(4) var<uniform> hp: AdamWParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x + global_id.y * 65535u * 256u;
    if (idx >= hp.n) {
        return;
    }

    let g = grads[idx];

    // Update moments
    m[idx] = hp.beta1 * m[idx] + (1.0 - hp.beta1) * g;
    v[idx] = hp.beta2 * v[idx] + (1.0 - hp.beta2) * g * g;

    // Bias correction
    let m_hat = m[idx] / hp.bias_correction1;
    let v_hat = v[idx] / hp.bias_correction2;

    // Weight decay + parameter update
    let p = params[idx];
    params[idx] = p - hp.lr * (m_hat / (sqrt(v_hat) + hp.eps) + hp.weight_decay * p);
}
"#;

// ============================================================================
// NF4 Dequantization: 4-bit NormalFloat to fp32
//
// Each byte stores two 4-bit values. block_size=64 elements share one scale.
// Lookup table maps 4-bit index → fp32 value, then multiply by scale.
// ============================================================================

/// NF4 weight dequantization shader
///
/// Binding 0: packed_data [n/2] as u32 (read) — each u32 has 8 nibbles
/// Binding 1: scales [n/block_size] (read)
/// Binding 2: output [n] (write)
/// Binding 3: uniform { n, block_size }
pub const NF4_DEQUANT_SHADER: &str = r#"
// NF4 codebook (same as trueno::quantize::NF4_LUT)
const NF4_LUT: array<f32, 16> = array<f32, 16>(
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
);

@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    n: u32,
    block_size: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 2D dispatch for large tensors (>16M elements): idx = x + y * 65535 * 256
    let idx = global_id.x + global_id.y * 65535u * 256u;
    if (idx >= params.n) {
        return;
    }

    // Each byte has 2 nibbles: low nibble = even index, high nibble = odd index
    let byte_idx = idx / 2u;
    let packed_val = packed[byte_idx / 4u];
    let byte_in_u32 = byte_idx % 4u;
    let byte_val = (packed_val >> (byte_in_u32 * 8u)) & 0xFFu;

    var nibble: u32;
    if (idx % 2u == 0u) {
        nibble = byte_val & 0xFu;  // low nibble
    } else {
        nibble = (byte_val >> 4u) & 0xFu;  // high nibble
    }

    let scale = scales[idx / params.block_size];
    output[idx] = NF4_LUT[nibble] * scale;
}
"#;

// ============================================================================
// Fused Cross-Entropy Forward: loss = -log(softmax(logits)[label])
//
// One workgroup per token position. Each computes:
// 1. max(logits) for numerical stability
// 2. logsumexp = max + log(Σ exp(logit - max))
// 3. loss = -logits[label] + logsumexp
//
// Saves logsumexp per position for backward pass.
// Response-only masking: positions outside [loss_start, loss_end) contribute 0.
//
// Contract: fused-cross-entropy-v1 / fused_forward
// ============================================================================

/// Fused cross-entropy forward loss shader
///
/// Each workgroup computes loss for one token position.
/// Outputs: losses[pos] and logsumexp[pos] (saved for backward).
pub const CROSS_ENTROPY_FORWARD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;   // [seq_len, vocab_size]
@group(0) @binding(1) var<storage, read> labels: array<u32>;   // [seq_len] — target token IDs
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;     // [seq_len] — per-token loss
@group(0) @binding(3) var<storage, read_write> logsumexp: array<f32>;  // [seq_len] — saved for backward

struct CEParams {
    seq_len: u32,
    vocab_size: u32,
    loss_start: u32,   // first response token position
    loss_end: u32,     // last+1 response token position
}

@group(0) @binding(4) var<uniform> params: CEParams;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x;
    if (pos >= params.seq_len) { return; }

    // Skip non-response positions
    if (pos < params.loss_start || pos >= params.loss_end) {
        losses[pos] = 0.0;
        logsumexp[pos] = 0.0;
        return;
    }

    let offset = pos * params.vocab_size;
    let label = labels[pos];

    // Pass 1: find max for numerical stability
    var max_val: f32 = -1e30;
    for (var v = 0u; v < params.vocab_size; v++) {
        max_val = max(max_val, logits[offset + v]);
    }

    // Pass 2: compute sum(exp(logit - max))
    var sum_exp: f32 = 0.0;
    for (var v = 0u; v < params.vocab_size; v++) {
        sum_exp += exp(logits[offset + v] - max_val);
    }

    let lse = max_val + log(sum_exp);
    logsumexp[pos] = lse;

    // Cross-entropy loss: -logits[label] + logsumexp
    if (label < params.vocab_size) {
        losses[pos] = -logits[offset + label] + lse;
    } else {
        losses[pos] = 0.0;  // padding token
    }
}
"#;

// ============================================================================
// Fused Cross-Entropy Backward: grad_logits = softmax(logits) - one_hot(label)
//
// Writes gradient IN-PLACE into the logits buffer (no allocation).
// Uses saved logsumexp from forward pass.
//
// Contract: fused-cross-entropy-v1 / fused_backward
// ============================================================================

/// Fused cross-entropy backward shader — writes gradient in-place into logits
pub const CROSS_ENTROPY_BACKWARD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> logits: array<f32>; // [seq_len, vocab_size] — overwritten with gradient
@group(0) @binding(1) var<storage, read> labels: array<u32>;       // [seq_len]
@group(0) @binding(2) var<storage, read> logsumexp: array<f32>;    // [seq_len] — from forward

struct CEBackParams {
    seq_len: u32,
    vocab_size: u32,
    loss_start: u32,
    loss_end: u32,
    scale: f32,    // 1.0 / num_response_tokens
}

@group(0) @binding(3) var<uniform> params: CEBackParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 2D dispatch for large tensors (seq × vocab > 65535 × 256)
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = params.seq_len * params.vocab_size;
    if (idx >= total) { return; }

    let pos = idx / params.vocab_size;
    let v = idx % params.vocab_size;

    // Zero gradient for non-response positions
    if (pos < params.loss_start || pos >= params.loss_end) {
        logits[idx] = 0.0;
        return;
    }

    let lse = logsumexp[pos];
    let logit = logits[idx];

    // softmax(logit) = exp(logit - logsumexp)
    var grad = exp(logit - lse);

    // Subtract 1 at the label position
    let label = labels[pos];
    if (v == label) {
        grad -= 1.0;
    }

    // Scale by 1/num_response_tokens
    logits[idx] = grad * params.scale;
}
"#;
