//! GH-278: Gated Delta Net Linear Attention (Qwen3.5)
//!
//! Implements the Gated Delta Net recurrent mechanism used in Qwen3.5's
//! "linear attention" layers. This is NOT classical linear attention —
//! it's a state-space model with gated delta rule updates.
//!
//! # Architecture
//!
//! Each linear attention layer maintains a recurrent state S ∈ ℝ^{k×v}
//! per head, updated via the gated delta rule:
//!
//! ```text
//! Equation (GDN-1): S_t = exp(g_t) · S_{t-1} + k_t ⊗ δ_t
//! Equation (GDN-2): δ_t = β_t · (v_t − S_{t-1}^T k_t)
//! Equation (GDN-3): o_t = S_t^T q_t
//! ```
//!
//! where:
//! - `g_t = −exp(A_log) · softplus(a_t + dt_bias)` is the decay factor
//! - `β_t = σ(b_t)` is the update gate (sigmoid)
//! - Q, K are L2-normalized before use
//!
//! # Weights (per layer)
//!
//! | Weight | Shape | Purpose |
//! |--------|-------|---------|
//! | `qkv_weight` | `[2·key_dim + value_dim, hidden_dim]` | Combined Q/K/V projection |
//! | `z_weight` | `[value_dim, hidden_dim]` | Gate projection for output norm |
//! | `b_weight` | `[num_v_heads, hidden_dim]` | Beta gate projection |
//! | `a_weight` | `[num_v_heads, hidden_dim]` | Decay projection |
//! | `conv1d_weight` | `[conv_dim, kernel_size]` | Depthwise causal convolution |
//! | `A_log` | `[num_v_heads]` | Logged decay base |
//! | `dt_bias` | `[num_v_heads]` | Time-step bias |
//! | `norm_weight` | `[head_v_dim]` | Gated RMSNorm weight |
//! | `out_weight` | `[hidden_dim, value_dim]` | Output projection |
//!
//! # State (per layer)
//!
//! - Recurrent: `[num_v_heads, key_head_dim, value_head_dim]`
//! - Conv buffer: `[conv_dim, kernel_size]`

use super::model::GpuModel;
use super::types::GpuModelConfig;
use crate::error::{RealizarError, Result};

// =============================================================================
// State Management
// =============================================================================

/// Recurrent + convolution state for all linear attention layers.
///
/// Contract: `states.len() == num_layers` (indexed by block_idx).
/// Non-linear layers have empty state vectors (zero cost).
#[derive(Debug, Clone)]
pub struct LinearAttnState {
    /// Per-layer recurrent state: `[num_v_heads · key_head_dim · value_head_dim]`
    /// Flattened row-major: `state[h * kd * vd + i * vd + j]`
    pub recurrent: Vec<Vec<f32>>,
    /// Per-layer conv buffer: `[conv_dim · kernel_size]`
    /// Circular buffer, newest at position `(step % kernel_size)`.
    /// Layout: `buf[channel * kernel_size + time_offset]`
    pub conv_buf: Vec<Vec<f32>>,
    /// Per-layer step counter for conv buffer position
    pub conv_steps: Vec<usize>,
}

impl LinearAttnState {
    /// Create state for a model with hybrid attention.
    ///
    /// Allocates recurrent state only for linear layers (checked via `config.is_linear_layer(i)`).
    #[must_use]
    pub fn new(config: &GpuModelConfig) -> Self {
        let num_layers = config.num_layers;
        let num_v_heads = config.linear_num_value_heads.unwrap_or(0);
        let kd = config.linear_key_head_dim.unwrap_or(0);
        let vd = config.linear_value_head_dim.unwrap_or(0);
        let conv_dim = config.linear_conv_dim();
        // N-04 (Meyer DbC): 0 when missing — no Qwen3.5-specific default.
        let kernel_size = config.linear_conv_kernel_dim.unwrap_or(0);

        let mut recurrent = Vec::with_capacity(num_layers);
        let mut conv_buf = Vec::with_capacity(num_layers);
        let mut conv_steps = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            if config.is_linear_layer(i) {
                recurrent.push(vec![0.0f32; num_v_heads * kd * vd]);
                conv_buf.push(vec![0.0f32; conv_dim * kernel_size]);
                conv_steps.push(0);
            } else {
                recurrent.push(Vec::new());
                conv_buf.push(Vec::new());
                conv_steps.push(0);
            }
        }

        Self {
            recurrent,
            conv_buf,
            conv_steps,
        }
    }

    /// Reset all states to zero (for new sequence).
    pub fn reset(&mut self) {
        for s in &mut self.recurrent {
            s.fill(0.0);
        }
        for b in &mut self.conv_buf {
            b.fill(0.0);
        }
        for c in &mut self.conv_steps {
            *c = 0;
        }
    }
}

// =============================================================================
// Forward Pass — Incremental (Single Token Decode)
// =============================================================================

/// Forward pass through a linear attention block for a single token.
///
/// Implements the full Gated Delta Net layer:
/// 1. Pre-norm (RMSNorm)
/// 2. QKV + gate projections
/// 3. Causal Conv1D update
/// 4. SiLU activation
/// 5. Gated delta rule recurrence (Equations GDN-1..3)
/// 6. Gated RMSNorm output
/// 7. Output projection + residual
/// 8. Post-norm + MLP (SwiGLU) + residual
///
/// # Arguments
///
/// * `model` - GPU model with block weights
/// * `input` - Hidden state for single token `[hidden_dim]`
/// * `block_idx` - Transformer block index
/// * `state` - Mutable linear attention state
///
/// # Errors
///
/// Returns error if projections fail or dimensions mismatch.
#[allow(clippy::many_single_char_names)]
pub fn forward_linear_block_incremental(
    model: &mut GpuModel,
    input: &[f32],
    block_idx: usize,
    state: &mut LinearAttnState,
) -> Result<Vec<f32>> {
    contract_pre_gdn_sublayer!();
    let hidden_dim = model.config.hidden_dim;
    let intermediate_dim = model.config.intermediate_dim;

    // N-04 (Meyer DbC): 0 when missing — no Qwen3.5-specific defaults.
    let num_k_heads = model.config.linear_num_key_heads.unwrap_or(0);
    let num_v_heads = model.config.linear_num_value_heads.unwrap_or(0);
    let kd = model.config.linear_key_head_dim.unwrap_or(0);
    let vd = model.config.linear_value_head_dim.unwrap_or(0);
    let kernel_size = model.config.linear_conv_kernel_dim.unwrap_or(0);

    let key_dim = num_k_heads * kd;
    let value_dim = num_v_heads * vd;
    let conv_dim = 2 * key_dim + value_dim;

    let block = &model.block_weights[block_idx];
    let linear = block
        .linear_attn
        .as_ref()
        .ok_or_else(|| RealizarError::InvalidShape {
            reason: format!("GH-278: Block {block_idx} is linear but has no LinearAttnWeights"),
        })?;

    // ── Step 1: Pre-norm ──
    let normed = GpuModel::layer_norm_static(
        input,
        &block.attn_norm_weight,
        &block.attn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // ── Step 2: Projections ──
    // QKV: [hidden_dim] → [conv_dim] (conv_dim = 2*key_dim + value_dim)
    let qkv = model
        .scheduler
        .matmul(&normed, &block.qkv_weight, 1, hidden_dim, conv_dim)?;

    // Gate z: [hidden_dim] → [value_dim]
    let z = model
        .scheduler
        .matmul(&normed, &linear.z_weight, 1, hidden_dim, value_dim)?;

    // Beta gate b: [hidden_dim] → [num_v_heads]
    let b = model
        .scheduler
        .matmul(&normed, &linear.b_weight, 1, hidden_dim, num_v_heads)?;

    // Decay projection a: [hidden_dim] → [num_v_heads]
    let a = model
        .scheduler
        .matmul(&normed, &linear.a_weight, 1, hidden_dim, num_v_heads)?;

    // ── Step 3: Causal Conv1D update + SiLU ──
    let qkv_activated = causal_conv1d_update(
        &qkv,
        &mut state.conv_buf[block_idx],
        &mut state.conv_steps[block_idx],
        &linear.conv1d_weight,
        conv_dim,
        kernel_size,
    );

    // ── Step 4: Split Q, K, V ──
    let q_raw = &qkv_activated[..key_dim];
    let k_raw = &qkv_activated[key_dim..2 * key_dim];
    let v = &qkv_activated[2 * key_dim..];

    // ── Step 5: Reshape + repeat_interleave K heads → V heads ──
    let heads_ratio = num_v_heads / num_k_heads;
    let mut q = vec![0.0f32; num_v_heads * kd];
    let mut k = vec![0.0f32; num_v_heads * kd];

    for kh in 0..num_k_heads {
        let src = &q_raw[kh * kd..(kh + 1) * kd];
        for r in 0..heads_ratio {
            let vh = kh * heads_ratio + r;
            q[vh * kd..(vh + 1) * kd].copy_from_slice(src);
        }
    }
    for kh in 0..num_k_heads {
        let src = &k_raw[kh * kd..(kh + 1) * kd];
        for r in 0..heads_ratio {
            let vh = kh * heads_ratio + r;
            k[vh * kd..(vh + 1) * kd].copy_from_slice(src);
        }
    }

    // ── Step 6: Compute decay g and beta ──
    let mut g = vec![0.0f32; num_v_heads];
    let mut beta = vec![0.0f32; num_v_heads];

    for h in 0..num_v_heads {
        // Equation: g = −exp(A_log) · softplus(a + dt_bias)
        let a_val = linear.a_log[h].exp();
        let sp_input = a[h] + linear.dt_bias[h];
        let softplus_val = softplus(sp_input);
        g[h] = -a_val * softplus_val;

        // Equation: β = σ(b)
        beta[h] = sigmoid(b[h]);
    }

    // ── Step 7: Gated delta rule recurrence (GDN-1..3) ──
    let mut output = vec![0.0f32; num_v_heads * vd];

    gated_delta_rule_step(
        &q,
        &k,
        v,
        &g,
        &beta,
        &mut state.recurrent[block_idx],
        &mut output,
        num_v_heads,
        kd,
        vd,
    );

    // ── Step 8: Gated RMSNorm — norm(output) * silu(z) ──
    let normed_output = rms_norm_gated(
        &output,
        &z,
        &linear.norm_weight,
        num_v_heads,
        vd,
        model.config.eps,
    );

    // ── Step 9: Output projection ──
    let projected =
        model
            .scheduler
            .matmul(&normed_output, &block.out_weight, 1, value_dim, hidden_dim)?;

    // ── Step 10: Residual 1 ──
    let mut residual1: Vec<f32> = input
        .iter()
        .zip(projected.iter())
        .enumerate()
        .map(|(i, (&inp, &proj))| inp + proj + block.out_bias[i])
        .collect();

    // ── Step 11: FFN (identical to standard attention blocks) ──
    let ffn_normed = GpuModel::layer_norm_static(
        &residual1,
        &block.ffn_norm_weight,
        &block.ffn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    let activated: Vec<f32> = if let Some(ref gate_weight) = block.ffn_gate_weight {
        // SwiGLU: silu(gate(x)) * up(x)
        let up_out = model.scheduler.matmul(
            &ffn_normed,
            &block.ffn_fc1_weight,
            1,
            hidden_dim,
            intermediate_dim,
        )?;
        let gate_out =
            model
                .scheduler
                .matmul(&ffn_normed, gate_weight, 1, hidden_dim, intermediate_dim)?;

        up_out
            .iter()
            .zip(gate_out.iter())
            .map(|(&u, &g_val)| {
                let silu_g = g_val / (1.0 + (-g_val).exp());
                silu_g * u
            })
            .collect()
    } else {
        // GELU fallback
        let fc1_out = model.scheduler.matmul(
            &ffn_normed,
            &block.ffn_fc1_weight,
            1,
            hidden_dim,
            intermediate_dim,
        )?;

        fc1_out
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let x = x + block.ffn_fc1_bias[i];
                0.5 * x
                    * (1.0
                        + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3)))
                            .tanh())
            })
            .collect()
    };

    let fc2_out = model.scheduler.matmul(
        &activated,
        &block.ffn_fc2_weight,
        1,
        intermediate_dim,
        hidden_dim,
    )?;

    // Residual 2
    for (i, x) in residual1.iter_mut().enumerate() {
        *x += fc2_out[i] + block.ffn_fc2_bias[i];
    }

    contract_post_gdn_sublayer!(&residual1);
    Ok(residual1)
}

/// Expand Q and K from key_heads to value_heads via repeat_interleave.
fn repeat_interleave_qk(
    q_raw: &[f32],
    k_raw: &[f32],
    num_k_heads: usize,
    num_v_heads: usize,
    kd: usize,
    heads_ratio: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; num_v_heads * kd];
    let mut k = vec![0.0f32; num_v_heads * kd];
    for kh in 0..num_k_heads {
        let src_q = &q_raw[kh * kd..(kh + 1) * kd];
        let src_k = &k_raw[kh * kd..(kh + 1) * kd];
        for r in 0..heads_ratio {
            let vh = kh * heads_ratio + r;
            q[vh * kd..(vh + 1) * kd].copy_from_slice(src_q);
            k[vh * kd..(vh + 1) * kd].copy_from_slice(src_k);
        }
    }
    (q, k)
}

// =============================================================================
// Forward Pass — Prefill (Full Sequence)
// =============================================================================

/// Forward pass through a linear attention block for a full sequence (prefill).
///
/// Processes all tokens sequentially using the recurrent form.
/// This is O(n · d²) where d = key_head_dim, compared to O(n² · d)
/// for standard softmax attention.
pub fn forward_linear_block_with_cache(
    model: &mut GpuModel,
    input: &[f32],
    seq_len: usize,
    block_idx: usize,
    state: &mut LinearAttnState,
) -> Result<Vec<f32>> {
    let hidden_dim = model.config.hidden_dim;
    let intermediate_dim = model.config.intermediate_dim;

    // N-04 (Meyer DbC): 0 when missing — no Qwen3.5-specific defaults.
    let num_k_heads = model.config.linear_num_key_heads.unwrap_or(0);
    let num_v_heads = model.config.linear_num_value_heads.unwrap_or(0);
    let kd = model.config.linear_key_head_dim.unwrap_or(0);
    let vd = model.config.linear_value_head_dim.unwrap_or(0);
    let kernel_size = model.config.linear_conv_kernel_dim.unwrap_or(0);

    let key_dim = num_k_heads * kd;
    let value_dim = num_v_heads * vd;
    let conv_dim = 2 * key_dim + value_dim;

    let block = &model.block_weights[block_idx];
    let linear = block
        .linear_attn
        .as_ref()
        .ok_or_else(|| RealizarError::InvalidShape {
            reason: format!("GH-278: Block {block_idx} is linear but has no LinearAttnWeights"),
        })?;

    // ── Step 1: Pre-norm (full sequence) ──
    let normed = GpuModel::layer_norm_static(
        input,
        &block.attn_norm_weight,
        &block.attn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    // ── Step 2: Batch projections ──
    let qkv_all =
        model
            .scheduler
            .matmul(&normed, &block.qkv_weight, seq_len, hidden_dim, conv_dim)?;

    let z_all =
        model
            .scheduler
            .matmul(&normed, &linear.z_weight, seq_len, hidden_dim, value_dim)?;

    let b_all =
        model
            .scheduler
            .matmul(&normed, &linear.b_weight, seq_len, hidden_dim, num_v_heads)?;

    let a_all =
        model
            .scheduler
            .matmul(&normed, &linear.a_weight, seq_len, hidden_dim, num_v_heads)?;

    // ── Step 3: Causal Conv1D on full sequence + SiLU ──
    let qkv_activated = causal_conv1d_sequence(
        &qkv_all,
        &mut state.conv_buf[block_idx],
        &mut state.conv_steps[block_idx],
        &linear.conv1d_weight,
        seq_len,
        conv_dim,
        kernel_size,
    );

    // ── Step 4: Process each position through recurrence ──
    let heads_ratio = num_v_heads / num_k_heads;
    let mut attn_output = vec![0.0f32; seq_len * value_dim];

    for pos in 0..seq_len {
        let qkv_pos = &qkv_activated[pos * conv_dim..(pos + 1) * conv_dim];
        let q_raw = &qkv_pos[..key_dim];
        let k_raw = &qkv_pos[key_dim..2 * key_dim];
        let v = &qkv_pos[2 * key_dim..];
        let b_pos = &b_all[pos * num_v_heads..(pos + 1) * num_v_heads];
        let a_pos = &a_all[pos * num_v_heads..(pos + 1) * num_v_heads];

        // Repeat_interleave Q, K from key_heads to value_heads
        let (q, k) = repeat_interleave_qk(q_raw, k_raw, num_k_heads, num_v_heads, kd, heads_ratio);

        // Compute decay and beta for this position
        let mut g = vec![0.0f32; num_v_heads];
        let mut beta = vec![0.0f32; num_v_heads];
        for h in 0..num_v_heads {
            let a_val = linear.a_log[h].exp();
            let sp_input = a_pos[h] + linear.dt_bias[h];
            g[h] = -a_val * softplus(sp_input);
            beta[h] = sigmoid(b_pos[h]);
        }

        // Gated delta rule step
        let mut out_pos = vec![0.0f32; num_v_heads * vd];
        gated_delta_rule_step(
            &q,
            &k,
            v,
            &g,
            &beta,
            &mut state.recurrent[block_idx],
            &mut out_pos,
            num_v_heads,
            kd,
            vd,
        );

        attn_output[pos * value_dim..(pos + 1) * value_dim].copy_from_slice(&out_pos);
    }

    // ── Step 5: Gated RMSNorm (per position) ──
    let mut normed_output = vec![0.0f32; seq_len * value_dim];
    for pos in 0..seq_len {
        let out_slice = &attn_output[pos * value_dim..(pos + 1) * value_dim];
        let z_slice = &z_all[pos * value_dim..(pos + 1) * value_dim];
        let normed_pos = rms_norm_gated(
            out_slice,
            z_slice,
            &linear.norm_weight,
            num_v_heads,
            vd,
            model.config.eps,
        );
        normed_output[pos * value_dim..(pos + 1) * value_dim].copy_from_slice(&normed_pos);
    }

    // ── Step 6: Output projection ──
    let projected = model.scheduler.matmul(
        &normed_output,
        &block.out_weight,
        seq_len,
        value_dim,
        hidden_dim,
    )?;

    // ── Step 7: Residual 1 ──
    let mut residual1: Vec<f32> = input
        .iter()
        .zip(projected.iter())
        .enumerate()
        .map(|(i, (&inp, &proj))| inp + proj + block.out_bias[i % hidden_dim])
        .collect();

    // ── Step 8: FFN (same as standard attention) ──
    let ffn_normed = GpuModel::layer_norm_static(
        &residual1,
        &block.ffn_norm_weight,
        &block.ffn_norm_bias,
        hidden_dim,
        model.config.eps,
    );

    let activated: Vec<f32> = if let Some(ref gate_weight) = block.ffn_gate_weight {
        let up_out = model.scheduler.matmul(
            &ffn_normed,
            &block.ffn_fc1_weight,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )?;
        let gate_out = model.scheduler.matmul(
            &ffn_normed,
            gate_weight,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )?;

        up_out
            .iter()
            .zip(gate_out.iter())
            .map(|(&u, &g_val)| {
                let silu_g = g_val / (1.0 + (-g_val).exp());
                silu_g * u
            })
            .collect()
    } else {
        let fc1_out = model.scheduler.matmul(
            &ffn_normed,
            &block.ffn_fc1_weight,
            seq_len,
            hidden_dim,
            intermediate_dim,
        )?;

        fc1_out
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let x = x + block.ffn_fc1_bias[i % intermediate_dim];
                0.5 * x
                    * (1.0
                        + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3)))
                            .tanh())
            })
            .collect()
    };

    let fc2_out = model.scheduler.matmul(
        &activated,
        &block.ffn_fc2_weight,
        seq_len,
        intermediate_dim,
        hidden_dim,
    )?;

    for (i, x) in residual1.iter_mut().enumerate() {
        *x += fc2_out[i] + block.ffn_fc2_bias[i % hidden_dim];
    }

    Ok(residual1)
}

// =============================================================================
// Core Kernels
// =============================================================================

/// Gated delta rule — single step recurrence.
///
/// Implements Equations GDN-1, GDN-2, GDN-3 with L2-normalized Q and K.
///
/// ```text
/// For each head h ∈ [0, num_v_heads):
///   S_h ← exp(g_h) · S_h                             (GDN-1: decay)
///   mem_h = S_h^T k_h                                  (read from state)
///   δ_h = β_h · (v_h − mem_h)                         (GDN-2: delta)
///   S_h ← S_h + k_h ⊗ δ_h                            (GDN-1: write)
///   o_h = S_h^T q_h                                    (GDN-3: output)
/// ```
///
/// # Complexity
///
/// O(num_v_heads · key_head_dim · value_head_dim) per token.
/// For Qwen3.5-9B: 32 × 128 × 128 = 524,288 FLOPs/token/layer.
fn gated_delta_rule_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    num_v_heads: usize,
    kd: usize,
    vd: usize,
) {
    contract_pre_decay!();
    contract_pre_read!();
    contract_pre_write!();
    contract_pre_delta!();
    contract_pre_output!();
    // Contract: dimension checks
    debug_assert_eq!(q.len(), num_v_heads * kd, "Q dim mismatch");
    debug_assert_eq!(k.len(), num_v_heads * kd, "K dim mismatch");
    debug_assert_eq!(v.len(), num_v_heads * vd, "V dim mismatch");
    debug_assert_eq!(g.len(), num_v_heads, "g dim mismatch");
    debug_assert_eq!(beta.len(), num_v_heads, "beta dim mismatch");
    debug_assert_eq!(state.len(), num_v_heads * kd * vd, "state dim mismatch");
    debug_assert_eq!(output.len(), num_v_heads * vd, "output dim mismatch");

    for h in 0..num_v_heads {
        let state_h = &mut state[h * kd * vd..(h + 1) * kd * vd];
        let q_h = &q[h * kd..(h + 1) * kd];
        let k_h = &k[h * kd..(h + 1) * kd];
        let v_h = &v[h * vd..(h + 1) * vd];
        let out_h = &mut output[h * vd..(h + 1) * vd];

        gated_delta_rule_head(q_h, k_h, v_h, g[h], beta[h], state_h, out_h, kd, vd);
    }
    contract_post_decay!(&output);
}

/// Single-head gated delta rule recurrence (GDN-1/2/3).
fn gated_delta_rule_head(
    q_h: &[f32],
    k_h: &[f32],
    v_h: &[f32],
    g_h: f32,
    beta_h: f32,
    state_h: &mut [f32],
    out_h: &mut [f32],
    kd: usize,
    vd: usize,
) {
    // L2-normalize Q and K (use_qk_l2norm_in_kernel=True)
    let q_norm = l2_normalize(q_h);
    let k_norm = l2_normalize(k_h);

    // GDN-1a: Decay — S_h ← exp(g_h) · S_h
    let decay = g_h.exp();
    for s in state_h.iter_mut() {
        *s *= decay;
    }

    // Read from state — mem = S^T k (state is [kd, vd] row-major)
    // mem[j] = Σ_i state[i, j] · k[i]
    let mut mem = vec![0.0f32; vd];
    for i in 0..kd {
        let k_i = k_norm[i];
        if k_i.abs() > f32::EPSILON {
            let row_start = i * vd;
            for j in 0..vd {
                mem[j] += state_h[row_start + j] * k_i;
            }
        }
    }

    // GDN-1b: Write — S ← S + k ⊗ δ (GDN-2: δ = β · (v − mem))
    for i in 0..kd {
        let k_i = k_norm[i];
        if k_i.abs() > f32::EPSILON {
            let row_start = i * vd;
            for j in 0..vd {
                let delta_j = beta_h * (v_h[j] - mem[j]);
                state_h[row_start + j] += k_i * delta_j;
            }
        }
    }

    // GDN-3: Output — o = S^T q
    out_h.fill(0.0);
    for i in 0..kd {
        let q_i = q_norm[i];
        if q_i.abs() > f32::EPSILON {
            let row_start = i * vd;
            for j in 0..vd {
                out_h[j] += state_h[row_start + j] * q_i;
            }
        }
    }
}

/// Causal Conv1D update for a single token (incremental decode).
///
/// Updates the circular conv buffer and computes the convolution output
/// for the newest position. Applies SiLU activation after convolution.
///
/// ```text
/// For each channel c:
///   buf[c, step % K] = input[c]
///   output[c] = SiLU(Σ_{k=0}^{K-1} weight[c, k] · buf[c, (step - K + 1 + k) % K])
/// ```
fn causal_conv1d_update(
    input: &[f32],
    conv_buf: &mut [f32],
    step: &mut usize,
    weight: &[f32],
    conv_dim: usize,
    kernel_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), conv_dim, "Conv input dim mismatch");
    debug_assert_eq!(
        conv_buf.len(),
        conv_dim * kernel_size,
        "Conv buf dim mismatch"
    );
    debug_assert_eq!(
        weight.len(),
        conv_dim * kernel_size,
        "Conv weight dim mismatch"
    );

    let current_step = *step;

    // Store input into circular buffer
    let buf_pos = current_step % kernel_size;
    for c in 0..conv_dim {
        conv_buf[c * kernel_size + buf_pos] = input[c];
    }

    // Compute depthwise convolution
    // Weight layout: weight[k] corresponds to time offset (kernel_size - 1 - k) steps back
    // weight[0] = oldest, weight[kernel_size-1] = most recent (current step)
    let mut output = vec![0.0f32; conv_dim];
    let num_valid = (current_step + 1).min(kernel_size);
    for c in 0..conv_dim {
        let mut sum = 0.0f32;
        // Only iterate over valid (non-zero) positions
        for j in 0..num_valid {
            // j=0 is most recent (current_step), j=1 is one step back, etc.
            let buf_idx = (current_step.wrapping_sub(j)) % kernel_size;
            let w_idx = kernel_size - 1 - j; // weight[kernel_size-1] = most recent
            sum += weight[c * kernel_size + w_idx] * conv_buf[c * kernel_size + buf_idx];
        }
        // SiLU activation
        output[c] = silu(sum);
    }

    *step = current_step + 1;
    output
}

/// Causal Conv1D for a full sequence (prefill).
///
/// Applies depthwise causal convolution over the full sequence,
/// then SiLU activation. Updates conv buffer with final positions
/// for subsequent incremental decode.
fn causal_conv1d_sequence(
    input: &[f32],
    conv_buf: &mut [f32],
    step: &mut usize,
    weight: &[f32],
    seq_len: usize,
    conv_dim: usize,
    kernel_size: usize,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), seq_len * conv_dim);

    let mut output = vec![0.0f32; seq_len * conv_dim];

    for pos in 0..seq_len {
        let in_pos = &input[pos * conv_dim..(pos + 1) * conv_dim];

        // Update circular buffer
        let buf_pos = (*step + pos) % kernel_size;
        for c in 0..conv_dim {
            conv_buf[c * kernel_size + buf_pos] = in_pos[c];
        }

        // Depthwise causal convolution at this position
        let current = *step + pos;
        for c in 0..conv_dim {
            let mut sum = 0.0f32;
            for k in 0..kernel_size {
                let lookback = kernel_size - 1 - k;
                if current >= lookback {
                    let time_idx = (current - lookback) % kernel_size;
                    sum += weight[c * kernel_size + k] * conv_buf[c * kernel_size + time_idx];
                }
                // else: before sequence start, zero-padded (buffer init to 0)
            }
            output[pos * conv_dim + c] = silu(sum);
        }
    }

    *step += seq_len;
    output
}

/// Gated RMSNorm: `RMSNorm(x) * SiLU(z)`
///
/// Applied per-head: norm weight is `[head_v_dim]`, shared across all heads.
///
/// ```text
/// For each head h:
///   x_h = x[h * vd .. (h+1) * vd]
///   z_h = z[h * vd .. (h+1) * vd]
///   rms = sqrt(mean(x_h²) + eps)
///   output_h = (weight * x_h / rms) * silu(z_h)
/// ```
fn rms_norm_gated(
    x: &[f32],
    z: &[f32],
    weight: &[f32],
    num_v_heads: usize,
    vd: usize,
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(x.len(), num_v_heads * vd);
    debug_assert_eq!(z.len(), num_v_heads * vd);
    debug_assert_eq!(weight.len(), vd);

    let mut output = vec![0.0f32; num_v_heads * vd];

    for h in 0..num_v_heads {
        let x_h = &x[h * vd..(h + 1) * vd];
        let z_h = &z[h * vd..(h + 1) * vd];
        let out_h = &mut output[h * vd..(h + 1) * vd];

        // RMS = sqrt(mean(x²) + eps)
        let mean_sq: f32 = x_h.iter().map(|&v| v * v).sum::<f32>() / vd as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        for j in 0..vd {
            // RMSNorm * SiLU(gate)
            out_h[j] = weight[j] * x_h[j] * inv_rms * silu(z_h[j]);
        }
    }

    output
}

// =============================================================================
// Math Primitives
// =============================================================================

/// L2-normalize a vector: x / ||x||₂
///
/// Returns zero vector if ||x||₂ < ε to avoid division by zero.
#[inline]
fn l2_normalize(x: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = x.iter().map(|&v| v * v).sum();
    if norm_sq < f32::EPSILON {
        return vec![0.0f32; x.len()];
    }
    let inv_norm = 1.0 / norm_sq.sqrt();
    x.iter().map(|&v| v * inv_norm).collect()
}

/// SiLU (Sigmoid Linear Unit): x · σ(x)
///
/// ONE PATH: Delegates to `trueno::silu_scalar` (UCBD §4).
#[inline]
fn silu(x: f32) -> f32 {
    trueno::silu_scalar(x)
}

/// Sigmoid: σ(x) = 1 / (1 + exp(-x))
///
/// ONE PATH: Delegates to `trueno::sigmoid_scalar` (UCBD §4).
#[inline]
fn sigmoid(x: f32) -> f32 {
    trueno::sigmoid_scalar(x)
}

/// Softplus: log(1 + exp(x))
///
/// Uses the numerically stable version that avoids overflow for large x.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // softplus(x) ≈ x for large x
    } else if x < -20.0 {
        0.0 // softplus(x) ≈ 0 for large negative x
    } else {
        (1.0 + x.exp()).ln()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-4);
        // silu(x) ≈ x for large x
        assert!((silu(10.0) - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(10.0) - 1.0).abs() < 1e-4);
        assert!((sigmoid(-10.0) - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - 0.693_147_2).abs() < 1e-4); // ln(2)
        assert!((softplus(25.0) - 25.0).abs() < 1e-4); // large x → x
        assert!(softplus(-25.0).abs() < 1e-4); // large negative → 0
    }

    #[test]
    fn test_l2_normalize() {
        let v = vec![3.0, 4.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);

        // Norm should be 1.0
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_zero() {
        let v = vec![0.0, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_gated_delta_rule_single_head() {
        // 1 head, kd=2, vd=2
        let q = vec![1.0, 0.0]; // unit Q pointing along dim 0
        let k = vec![0.0, 1.0]; // unit K pointing along dim 1
        let v = vec![1.0, 2.0];
        let g = vec![0.0]; // no decay (exp(0) = 1)
        let beta = vec![1.0]; // full update
        let mut state = vec![0.0f32; 4]; // [2, 2] = zeros
        let mut output = vec![0.0f32; 2];

        gated_delta_rule_step(&q, &k, &v, &g, &beta, &mut state, &mut output, 1, 2, 2);

        // After step:
        // k_norm = [0, 1] (already unit)
        // q_norm = [1, 0] (already unit)
        // mem = S^T k = [0, 0] (state was zero)
        // delta = 1.0 * ([1, 2] - [0, 0]) = [1, 2]
        // state += outer([0, 1], [1, 2]) = [[0, 0], [1, 2]]
        // output = S^T q = [[0, 0], [1, 2]]^T @ [1, 0] = [0, 0]
        // Hmm, q is along dim 0 but state only has values in row 1
        // So output should be [state[0,0], state[0,1]] = [0, 0]
        assert!((output[0]).abs() < 1e-5);
        assert!((output[1]).abs() < 1e-5);

        // Second step with same inputs should accumulate
        gated_delta_rule_step(&q, &k, &v, &g, &beta, &mut state, &mut output, 1, 2, 2);

        // Now state row 1 has accumulated more, but output reads from row 0 (q direction)
        // Since q is along dim 0 and k writes to dim 1, output stays near 0
    }

    #[test]
    fn test_gated_delta_rule_aligned() {
        // Q and K aligned — should produce strong output
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0]; // aligned with Q
        let v = vec![3.0, 7.0];
        let g = vec![0.0];
        let beta = vec![1.0];
        let mut state = vec![0.0f32; 4];
        let mut output = vec![0.0f32; 2];

        gated_delta_rule_step(&q, &k, &v, &g, &beta, &mut state, &mut output, 1, 2, 2);

        // k writes to row 0, q reads from row 0
        // delta = v - 0 = [3, 7]
        // state row 0 = [3, 7]
        // output = state^T @ q_norm = [3, 7] (q_norm = [1, 0], so reads row 0)
        assert!((output[0] - 3.0).abs() < 1e-4);
        assert!((output[1] - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_decay_reduces_state() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![1.0, 1.0];
        let g = vec![-1.0]; // decay = exp(-1) ≈ 0.368
        let beta = vec![1.0];
        let mut state = vec![10.0, 10.0, 10.0, 10.0]; // pre-loaded state
        let mut output = vec![0.0f32; 2];

        gated_delta_rule_step(&q, &k, &v, &g, &beta, &mut state, &mut output, 1, 2, 2);

        // State should be decayed then updated
        // decay = exp(-1) ≈ 0.368
        // After decay: state row 0 ≈ [3.68, 3.68]
        // mem = state^T @ k_norm = state[0,:] = [3.68, 3.68]
        // delta = 1.0 * ([1, 1] - [3.68, 3.68]) = [-2.68, -2.68]
        // state row 0 += 1.0 * [-2.68, -2.68] = [1.0, 1.0]
        // output = state^T @ q = [1.0, 1.0]
        assert!((output[0] - 1.0).abs() < 0.1);
        assert!((output[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_causal_conv1d_update_single() {
        let conv_dim = 2;
        let kernel_size = 3;
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // [2, 3]
        let mut buf = vec![0.0f32; 6]; // [2, 3]
        let mut step = 0;

        let input = vec![1.0, 2.0];
        let out = causal_conv1d_update(&input, &mut buf, &mut step, &weight, conv_dim, kernel_size);

        // First step: only kernel[2] (most recent) has data
        // Channel 0: weight[2] * input[0] = 0.3 * 1.0 = 0.3 → silu(0.3)
        // Channel 1: weight[5] * input[1] = 0.6 * 2.0 = 1.2 → silu(1.2)
        assert_eq!(step, 1);
        assert!(out[0] > 0.0);
        assert!(out[1] > 0.0);
    }

    #[test]
    fn test_rms_norm_gated_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 heads, vd=2
        let z = vec![0.0, 0.0, 0.0, 0.0]; // gate = silu(0) = 0
        let weight = vec![1.0, 1.0]; // identity weight

        let out = rms_norm_gated(&x, &z, &weight, 2, 2, 1e-5);

        // silu(0) = 0, so output should be all zeros
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn test_rms_norm_gated_with_gate() {
        let x = vec![2.0, 0.0]; // 1 head, vd=2
        let z = vec![10.0, 10.0]; // large gate → silu ≈ 10
        let weight = vec![1.0, 1.0];

        let out = rms_norm_gated(&x, &z, &weight, 1, 2, 1e-5);

        // RMS of [2, 0] = sqrt((4+0)/2) = sqrt(2)
        // Normed: [2/sqrt(2), 0] = [sqrt(2), 0]
        // silu(10) ≈ 10
        // output ≈ [sqrt(2)*10, 0] ≈ [14.14, 0]
        assert!((out[0] - 14.142).abs() < 0.1);
        assert!(out[1].abs() < 0.01);
    }

    #[test]
    fn test_linear_attn_state_new() {
        let config = GpuModelConfig {
            vocab_size: 100,
            hidden_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 4,
            intermediate_dim: 128,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: Some(vec![
                "attention".to_string(),
                "linear".to_string(),
                "attention".to_string(),
                "linear".to_string(),
            ]),
            linear_key_head_dim: Some(8),
            linear_value_head_dim: Some(8),
            linear_num_key_heads: Some(2),
            linear_num_value_heads: Some(4),
            linear_conv_kernel_dim: Some(4),
            constraints: None,
            num_experts: None,
            num_experts_per_tok: None,
            expert_intermediate_size: None,
        };

        let state = LinearAttnState::new(&config);

        // Non-linear layers should have empty state
        assert!(state.recurrent[0].is_empty());
        assert!(state.recurrent[2].is_empty());

        // Linear layers should have allocated state
        // recurrent: num_v_heads * kd * vd = 4 * 8 * 8 = 256
        assert_eq!(state.recurrent[1].len(), 256);
        assert_eq!(state.recurrent[3].len(), 256);

        // conv: conv_dim * kernel_size = (2*16 + 32) * 4 = 64 * 4 = 256
        let conv_dim = 2 * (2 * 8) + 4 * 8; // 2*key_dim + value_dim
        assert_eq!(state.conv_buf[1].len(), conv_dim * 4);
    }

    #[test]
    fn test_linear_attn_state_reset() {
        let config = GpuModelConfig {
            vocab_size: 100,
            hidden_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 2,
            intermediate_dim: 128,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: Some(vec!["linear".to_string(), "linear".to_string()]),
            linear_key_head_dim: Some(4),
            linear_value_head_dim: Some(4),
            linear_num_key_heads: Some(2),
            linear_num_value_heads: Some(2),
            linear_conv_kernel_dim: Some(4),
            constraints: None,
            num_experts: None,
            num_experts_per_tok: None,
            expert_intermediate_size: None,
        };

        let mut state = LinearAttnState::new(&config);

        // Dirty the state
        state.recurrent[0].fill(1.0);
        state.conv_buf[0].fill(2.0);
        state.conv_steps[0] = 42;

        state.reset();

        assert!(state.recurrent[0].iter().all(|&v| v == 0.0));
        assert!(state.conv_buf[0].iter().all(|&v| v == 0.0));
        assert_eq!(state.conv_steps[0], 0);
    }
}
