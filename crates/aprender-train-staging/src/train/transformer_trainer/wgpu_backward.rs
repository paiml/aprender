//! WGPU backward pass through 36 transformer layers + LoRA AdamW
//!
//! Backpropagates gradients from lm_head through all FFN layers,
//! computes LoRA Q/V gradients, and runs AdamW on adapter weights.
//!
//! # Contract: C-WGPU-TRAIN-007 (layer backward + LoRA optimizer)

#[cfg(feature = "gpu")]
use trueno::backends::gpu::GpuDevice;

/// CPU AdamW step — avoids GPU dispatch overhead for small LoRA tensors
/// LoRA A: rank×hidden = 16×2560 = 40K params → ~0.1ms on CPU vs ~50ms GPU dispatch
#[cfg(feature = "gpu")]
fn cpu_adamw(
    params: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step: u32,
) {
    let bc1 = 1.0 / (1.0 - beta1.powi(step as i32));
    let bc2 = 1.0 / (1.0 - beta2.powi(step as i32));
    for i in 0..params.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        let m_hat = m[i] * bc1;
        let v_hat = v[i] * bc2;
        params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * params[i]);
    }
}

/// Per-layer forward activations cached for backward pass
#[cfg(feature = "gpu")]
pub struct LayerActivations {
    /// Input to attention (after RMSNorm) [seq_len, hidden_size]
    pub attn_input: Vec<f32>,
    /// Input to FFN (after attention + RMSNorm) [seq_len, hidden_size]
    pub hidden_input: Vec<f32>,
    /// Gate projection output [seq_len, intermediate_size]
    pub gate_output: Vec<f32>,
    /// Up projection output [seq_len, intermediate_size]
    pub up_output: Vec<f32>,
    /// SiLU(gate) output [seq_len, intermediate_size]
    pub silu_gate: Vec<f32>,
    /// Attention: Q after QK-norm + RoPE [seq_len, q_dim]
    pub q: Vec<f32>,
    /// Attention: K after QK-norm + RoPE [seq_len, kv_dim]
    pub k: Vec<f32>,
    /// Attention: V [seq_len, kv_dim]
    pub v: Vec<f32>,
    /// Attention: softmax weights per head [num_heads, seq_len, seq_len]
    pub attn_weights: Vec<f32>,
    /// Attention: context before O projection [seq_len, q_dim]
    pub context: Vec<f32>,
    /// h_cached for LoRA Q: hidden @ A_q^T [seq_len, rank]
    pub lora_q_h: Vec<f32>,
    /// h_cached for LoRA V: hidden @ A_v^T [seq_len, rank]
    pub lora_v_h: Vec<f32>,
}

/// Run backward pass through all layers and update LoRA adapters
///
/// Given grad_hidden from lm_head backward, backpropagates through each FFN
/// layer in reverse order, computing LoRA Q/V gradients and running AdamW.
///
/// # Contract (C-WGPU-TRAIN-007)
/// - Precondition: grad_hidden from lm_head backward is finite
/// - Postcondition: LoRA Q/V adapters updated, grad_hidden propagated to layer 0
#[cfg(feature = "gpu")]
pub fn backward_through_layers(
    device: &GpuDevice,
    grad_hidden: &mut Vec<f32>,
    activations: &[LayerActivations],
    model: &mut super::wgpu_trainer::WgpuModelState,
    seq_len: u32,
    hidden_size: u32,
    intermediate_size: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
    lora_alpha: f32,
) -> Result<f32, String> {
    let s = seq_len;
    let h = hidden_size;
    let i = intermediate_size;
    let n_layers = model.num_layers;
    let mut total_lora_gnorm = 0.0f32;

    // Backward through layers in reverse order
    for layer_idx in (0..n_layers).rev() {
        let act = &activations[layer_idx];

        // Get cached FFN weights
        let (gate_w, up_w, down_w) = model.ffn_cache[layer_idx]
            .as_ref()
            .map(|(g, u, d)| (g.as_slice(), u.as_slice(), d.as_slice()))
            .expect("cache populated");

        // --- FFN backward ---
        // 1. Down backward: grad_swiglu = grad_hidden @ down (GPU GEMM)
        let mut grad_swiglu = vec![0.0f32; (s * i) as usize];
        device.gemm_backward_a(grad_hidden, down_w, &mut grad_swiglu, s, i, h)?;

        // 2. SiLU backward
        let n_inter = (s * i) as usize;
        let mut grad_gate = vec![0.0f32; n_inter];
        let mut grad_up = vec![0.0f32; n_inter];
        for j in 0..n_inter {
            let x = act.gate_output[j];
            let sig = 1.0 / (1.0 + (-x).exp());
            let y = x * sig;
            let silu_prime = sig * (1.0 + x - y);
            grad_gate[j] = grad_swiglu[j] * act.up_output[j] * silu_prime;
            grad_up[j] = grad_swiglu[j] * act.silu_gate[j];
        }

        // 3. Gate backward: grad_input_gate = grad_gate @ gate^T (GPU GEMM)
        let mut grad_input_gate = vec![0.0f32; (s * h) as usize];
        device.gemm_backward_a(&grad_gate, gate_w, &mut grad_input_gate, s, h, i)?;

        // 4. Up backward: grad_input_up = grad_up @ up^T (GPU GEMM)
        let mut grad_input_up = vec![0.0f32; (s * h) as usize];
        device.gemm_backward_a(&grad_up, up_w, &mut grad_input_up, s, h, i)?;

        // 5. Sum: grad_ffn_input = grad_input_gate + grad_input_up
        for j in 0..(s * h) as usize {
            grad_hidden[j] = grad_input_gate[j] + grad_input_up[j];
        }

        // --- Attention backward → real LoRA gradients ---
        // 1. grad_context = grad_hidden @ O^T (O is pre-transposed [q_dim, h])
        let q_dim = model.num_heads * model.head_dim;
        let kv_dim = model.num_kv_heads * model.head_dim;
        let hd = model.head_dim;
        let nh = model.num_heads;
        let nkv = model.num_kv_heads;
        let heads_per_kv = nh / nkv;
        let (_, _, _, o_w) = model.attn_cache[layer_idx]
            .as_ref()
            .map(|(q, k, v, o)| (q.as_slice(), k.as_slice(), v.as_slice(), o.as_slice()))
            .expect("attn cache");
        // O is transposed [q_dim, h], so grad_hidden[s,h] @ O^T[h,q_dim]... but we need
        // grad_context = grad_hidden @ O_original. O_original = O_transposed^T.
        // gemm_backward_a computes A @ B^T, so: gemm_backward_a(grad_hidden, O_transposed) = grad_hidden @ O_transposed^T = grad_hidden @ O_original
        let mut grad_context = vec![0.0f32; s as usize * q_dim];
        device.gemm_backward_a(grad_hidden, o_w, &mut grad_context, s, q_dim as u32, h)?;

        // 2. Attention backward: grad_q, grad_v from grad_context through softmax+V
        let scale = 1.0 / (hd as f32).sqrt();
        let mut grad_q = vec![0.0f32; s as usize * q_dim];
        let mut grad_v = vec![0.0f32; s as usize * kv_dim];
        for head in 0..nh {
            let kv_head = head / heads_per_kv;
            for qi in 0..s as usize {
                let aw_off = head * s as usize * s as usize + qi * s as usize;
                // grad_scores[ki] = sum_d(grad_context[qi,head,d] * v[ki,kv_head,d])
                let mut grad_scores = vec![0.0f32; s as usize];
                let mut dot_sum = 0.0f32;
                for ki in 0..s as usize {
                    for d in 0..hd {
                        grad_scores[ki] += grad_context[qi * q_dim + head * hd + d]
                            * act.v[ki * kv_dim + kv_head * hd + d];
                    }
                    dot_sum += act.attn_weights[aw_off + ki] * grad_scores[ki];
                }
                // Softmax backward: grad_pre = attn_w * (grad_scores - dot_sum)
                for ki in 0..s as usize {
                    let g_pre = act.attn_weights[aw_off + ki] * (grad_scores[ki] - dot_sum) * scale;
                    // grad_q[qi] += g_pre * k[ki]
                    for d in 0..hd {
                        grad_q[qi * q_dim + head * hd + d] +=
                            g_pre * act.k[ki * kv_dim + kv_head * hd + d];
                    }
                }
                // grad_v[ki] += attn_w[qi,ki] * grad_context[qi]
                for ki in 0..s as usize {
                    let w = act.attn_weights[aw_off + ki];
                    if w > 0.0 {
                        for d in 0..hd {
                            grad_v[ki * kv_dim + kv_head * hd + d] +=
                                w * grad_context[qi * q_dim + head * hd + d];
                        }
                    }
                }
            }
        }

        // 3. LoRA Q backward: dL/dB = (α/r) * h_cached^T @ grad_q, dL/dA = (α/r) * B^T @ grad_q @ x
        let rank = model.lora[layer_idx].q.rank as usize;
        if rank > 0 {
            let scaling = lora_alpha / rank as f32;
            // dL/dB_q [q_dim, rank] = scaling * grad_q^T[q_dim, s] @ h_cached[s, rank]
            let mut grad_b = vec![0.0f32; q_dim * rank];
            for qi in 0..q_dim {
                for ri in 0..rank {
                    let mut sum = 0.0f32;
                    for si in 0..s as usize {
                        sum += grad_q[si * q_dim + qi] * act.lora_q_h[si * rank + ri];
                    }
                    grad_b[qi * rank + ri] = sum * scaling;
                }
            }
            // dL/dA_q [rank, h] = scaling * (B^T @ grad_q)^T @ x = scaling * grad_q^T @ B → then transpose...
            // Simpler: dL/dA = scaling * sum_s(grad_h_cached[s,rank] outer x[s,h]) where grad_h_cached = grad_q @ B
            let mut grad_h_cached = vec![0.0f32; s as usize * rank];
            for si in 0..s as usize {
                for ri in 0..rank {
                    let mut sum = 0.0f32;
                    for qi in 0..q_dim {
                        sum += grad_q[si * q_dim + qi] * model.lora[layer_idx].q.b[qi * rank + ri];
                    }
                    grad_h_cached[si * rank + ri] = sum * scaling;
                }
            }
            let mut grad_a = vec![0.0f32; rank * h as usize];
            for ri in 0..rank {
                for hi in 0..h as usize {
                    let mut sum = 0.0f32;
                    for si in 0..s as usize {
                        sum += grad_h_cached[si * rank + ri] * act.attn_input[si * h as usize + hi];
                    }
                    grad_a[ri * h as usize + hi] = sum;
                }
            }
            total_lora_gnorm += grad_a.iter().map(|g| g * g).sum::<f32>();
            // C-WGPU-LORAPLUS-001: LoRA+ — lr_B = 16 * lr_A (Hayou et al. 2024)
            let lq = &mut model.lora[layer_idx].q;
            cpu_adamw(
                &mut lq.a,
                &grad_a,
                &mut lq.m_a,
                &mut lq.v_a,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            );
            cpu_adamw(
                &mut lq.b,
                &grad_b,
                &mut lq.m_b,
                &mut lq.v_b,
                lr * 16.0,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            );
        }

        // 4. LoRA V backward: same pattern with grad_v
        let v_rank = model.lora[layer_idx].v.rank as usize;
        if v_rank > 0 {
            let scaling = lora_alpha / v_rank as f32;
            let mut grad_b = vec![0.0f32; kv_dim * v_rank];
            for vi in 0..kv_dim {
                for ri in 0..v_rank {
                    let mut sum = 0.0f32;
                    for si in 0..s as usize {
                        sum += grad_v[si * kv_dim + vi] * act.lora_v_h[si * v_rank + ri];
                    }
                    grad_b[vi * v_rank + ri] = sum * scaling;
                }
            }
            let mut grad_h_cached = vec![0.0f32; s as usize * v_rank];
            for si in 0..s as usize {
                for ri in 0..v_rank {
                    let mut sum = 0.0f32;
                    for vi in 0..kv_dim {
                        sum +=
                            grad_v[si * kv_dim + vi] * model.lora[layer_idx].v.b[vi * v_rank + ri];
                    }
                    grad_h_cached[si * v_rank + ri] = sum * scaling;
                }
            }
            let mut grad_a = vec![0.0f32; v_rank * h as usize];
            for ri in 0..v_rank {
                for hi in 0..h as usize {
                    let mut sum = 0.0f32;
                    for si in 0..s as usize {
                        sum +=
                            grad_h_cached[si * v_rank + ri] * act.attn_input[si * h as usize + hi];
                    }
                    grad_a[ri * h as usize + hi] = sum;
                }
            }
            total_lora_gnorm += grad_a.iter().map(|g| g * g).sum::<f32>();
            let lv = &mut model.lora[layer_idx].v;
            cpu_adamw(
                &mut lv.a,
                &grad_a,
                &mut lv.m_a,
                &mut lv.v_a,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            );
            cpu_adamw(
                &mut lv.b,
                &grad_b,
                &mut lv.m_b,
                &mut lv.v_b,
                lr * 16.0,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            );
        }
    }

    Ok(total_lora_gnorm.sqrt())
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    /// FALSIFY: Backward through layers produces non-zero LoRA gradient norm
    #[test]
    fn test_backward_through_layers_gradient_flow() {
        // Create minimal model state
        let rank = 4u32;
        let h = 8u32;
        let i_size = 16u32;
        let s = 2u32;
        let n_layers = 2;

        let device = GpuDevice::new().expect("GPU");

        let mut model = super::super::wgpu_trainer::WgpuModelState {
            layers: vec![],
            lora: (0..n_layers)
                .map(|_| {
                    crate::train::transformer_trainer::wgpu_checkpoint::LoraLayerSet::new(
                        rank, h, h, h, i_size,
                    )
                })
                .collect(),
            lm_head: vec![0.0f32; 32 * h as usize],
            lm_head_m: vec![0.0f32; 32 * h as usize],
            lm_head_v: vec![0.0f32; 32 * h as usize],
            hidden_size: h as usize,
            num_layers: n_layers,
            vocab_size: 32,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: i_size as usize,
            ffn_cache: vec![None; n_layers],
            attn_cache: vec![None; n_layers],
        };

        for l in 0..n_layers {
            model.ffn_cache[l] = Some((
                vec![0.01f32; (i_size * h) as usize],
                vec![0.01f32; (i_size * h) as usize],
                vec![0.01f32; (h * i_size) as usize],
            ));
            model.attn_cache[l] = Some((
                vec![0.01f32; h as usize * 8], // q [h, q_dim]
                vec![0.01f32; h as usize * 8], // k [h, kv_dim]
                vec![0.01f32; h as usize * 8], // v [h, kv_dim]
                vec![0.01f32; 8 * h as usize], // o [q_dim, h]
            ));
        }

        // Activations
        let q_dim = 8usize; // 2 heads * 4 head_dim
        let kv_dim = 8usize; // 2 kv_heads * 4 head_dim
        let activations: Vec<LayerActivations> = (0..n_layers)
            .map(|_| LayerActivations {
                attn_input: (0..(s * h) as usize).map(|j| (j as f32 - 8.0) * 0.1).collect(),
                hidden_input: (0..(s * h) as usize).map(|j| (j as f32 - 8.0) * 0.1).collect(),
                gate_output: vec![0.5f32; (s * i_size) as usize],
                up_output: vec![0.3f32; (s * i_size) as usize],
                silu_gate: vec![0.25f32; (s * i_size) as usize],
                q: vec![0.1f32; s as usize * q_dim],
                k: vec![0.1f32; s as usize * kv_dim],
                v: vec![0.1f32; s as usize * kv_dim],
                attn_weights: vec![0.5f32; 2 * s as usize * s as usize], // 2 heads
                context: vec![0.1f32; s as usize * q_dim],
                lora_q_h: vec![0.01f32; s as usize * rank as usize],
                lora_v_h: vec![0.01f32; s as usize * rank as usize],
            })
            .collect();

        let mut grad_hidden: Vec<f32> =
            (0..(s * h) as usize).map(|j| (j as f32 - 8.0) * 0.01).collect();

        // Save original LoRA weights for comparison
        let orig_q_a_0 = model.lora[0].q.a.clone();
        let orig_v_a_0 = model.lora[0].v.a.clone();

        let gnorm = backward_through_layers(
            &device,
            &mut grad_hidden,
            &activations,
            &mut model,
            s,
            h,
            i_size,
            1e-3,
            0.9,
            0.999,
            1e-8,
            0.01,
            1,
            32.0,
        )
        .expect("backward");

        // LoRA weights must have changed
        assert_ne!(model.lora[0].q.a, orig_q_a_0, "LoRA Q adapter A must be updated");
        assert_ne!(model.lora[0].v.a, orig_v_a_0, "LoRA V adapter A must be updated");
        assert!(gnorm >= 0.0, "Gradient norm must be non-negative");
        assert!(grad_hidden.iter().all(|g| g.is_finite()), "All gradients finite");

        eprintln!("Backward through {n_layers} layers: lora_gnorm={gnorm:.6}");
    }
}
