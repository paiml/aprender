use std::f32::consts::E;

/// SiLU activation function
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus activation function
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// L2 Normalization with epsilon over the given slice.
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let mut sq_sum = 0.0;
    for &v in x.iter() {
        sq_sum += v * v;
    }
    let scale = 1.0 / (sq_sum + eps).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Gated RMSNorm: (RMSNorm(input) * silu(gate))
/// Norm is computed independently over chunks of size `head_v_dim`.
pub fn gated_rmsnorm(
    input: &[f32],
    gate: &[f32],
    weight: &[f32],
    eps: f32,
    head_v_dim: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), gate.len());
    assert_eq!(input.len(), output.len());
    assert_eq!(weight.len(), head_v_dim);
    assert_eq!(input.len() % head_v_dim, 0);

    for (chunk_in, (chunk_gate, chunk_out)) in input.chunks_exact(head_v_dim).zip(
        gate.chunks_exact(head_v_dim)
            .zip(output.chunks_exact_mut(head_v_dim)),
    ) {
        let mut sq_sum = 0.0;
        for &v in chunk_in {
            sq_sum += v * v;
        }
        let rms_scale = 1.0 / ((sq_sum / head_v_dim as f32) + eps).sqrt();

        for i in 0..head_v_dim {
            let norm_v = chunk_in[i] * rms_scale * weight[i];
            chunk_out[i] = norm_v * silu(chunk_gate[i]);
        }
    }
}

/// Causal Conv1d over seq for a single time step.
/// state is `(kernel_size - 1) * channels` floats.
/// For each channel, we shift the past inputs left and insert the new input.
pub fn causal_conv1d(
    input: &[f32],
    state: &mut [f32],
    weight: &[f32],
    kernel_size: usize,
    channels: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), channels);
    assert_eq!(state.len(), (kernel_size - 1) * channels);
    assert_eq!(weight.len(), kernel_size * channels);
    assert_eq!(output.len(), channels);

    for c in 0..channels {
        let mut sum = 0.0;
        let s_offset = c * (kernel_size - 1);
        let w_offset = c * kernel_size;

        for k in 0..(kernel_size - 1) {
            sum += state[s_offset + k] * weight[w_offset + k];
        }
        sum += input[c] * weight[w_offset + kernel_size - 1];

        // shift left (oldest is at 0, newest is at kernel_size - 2)
        for k in 0..(kernel_size - 2) {
            state[s_offset + k] = state[s_offset + k + 1];
        }
        if kernel_size > 1 {
            state[s_offset + kernel_size - 2] = input[c];
        }

        output[c] = sum;
    }
}

/// The gated delta-rule recurrence for a single token exactly as delta-net-base.cpp computes it.
pub fn delta_rule_recurrence(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    gate: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    num_v_heads: usize,
    head_v_dim: usize,
) {
    assert_eq!(q.len(), num_v_heads * head_v_dim);
    assert_eq!(k.len(), num_v_heads * head_v_dim);
    assert_eq!(v.len(), num_v_heads * head_v_dim);
    assert_eq!(beta.len(), num_v_heads);
    assert_eq!(gate.len(), num_v_heads);
    assert_eq!(state.len(), num_v_heads * head_v_dim * head_v_dim);
    assert_eq!(output.len(), num_v_heads * head_v_dim);

    let scale = 1.0 / (head_v_dim as f32).sqrt();

    for h in 0..num_v_heads {
        let q_h = &q[h * head_v_dim..(h + 1) * head_v_dim];
        let k_h = &k[h * head_v_dim..(h + 1) * head_v_dim];
        let v_h = &v[h * head_v_dim..(h + 1) * head_v_dim];
        let beta_val = beta[h];
        let gate_val = gate[h];

        let state_offset = h * head_v_dim * head_v_dim;
        let s_h = &mut state[state_offset..state_offset + head_v_dim * head_v_dim];

        // 1. S_h *= exp(gate_val)
        let exp_gate = gate_val.exp();
        for s in s_h.iter_mut() {
            *s *= exp_gate;
        }

        // 2. delta = (v_h - S_h^T * k_h) * beta_val
        // Note: s_h[j * head_v_dim + i] is S[i][j].
        // So row j of s_h in memory is column j of S.
        // sum = dot(row j of s_h, k_h)
        let mut delta = vec![0.0; head_v_dim];
        for j in 0..head_v_dim {
            let row_j = &s_h[j * head_v_dim..(j + 1) * head_v_dim];
            let mut sum = 0.0;
            for i in 0..head_v_dim {
                sum += row_j[i] * k_h[i];
            }
            delta[j] = (v_h[j] - sum) * beta_val;
        }

        // 3. S_h += k_h * delta^T
        for j in 0..head_v_dim {
            let row_j = &mut s_h[j * head_v_dim..(j + 1) * head_v_dim];
            let d_j = delta[j];
            for i in 0..head_v_dim {
                row_j[i] += k_h[i] * d_j;
            }
        }

        // 4. out_h = S_h^T * q_h * scale
        for j in 0..head_v_dim {
            let row_j = &s_h[j * head_v_dim..(j + 1) * head_v_dim];
            let mut sum = 0.0;
            for i in 0..head_v_dim {
                sum += row_j[i] * q_h[i];
            }
            output[h * head_v_dim + j] = sum * scale;
        }
    }
}

/// Partial RoPE over rope_dimension_count dims honouring rope.dimension_sections.
/// Sections are lengths of [rope, zero, rope, zero] applied to the feature dimension.
pub fn apply_rope_sections(
    q: &mut [f32],
    k: &mut [f32],
    pos: u32,
    head_dim: usize,
    num_q_heads: usize,
    num_k_heads: usize,
    freq_base: f32,
    sections: &[usize; 4],
) {
    let mut compute_rope = |x: &mut [f32], num_heads: usize| {
        for h in 0..num_heads {
            let x_h = &mut x[h * head_dim..(h + 1) * head_dim];
            let mut offset = 0;

            // Section 0: RoPE
            let sec0 = sections[0];
            for i in (0..sec0).step_by(2) {
                let theta = (pos as f32) / freq_base.powf((i as f32) / (sec0 as f32));
                let cos = theta.cos();
                let sin = theta.sin();

                let idx0 = offset + i;
                let idx1 = offset + i + 1;
                let x0 = x_h[idx0];
                let x1 = x_h[idx1];
                x_h[idx0] = x0 * cos - x1 * sin;
                x_h[idx1] = x0 * sin + x1 * cos;
            }
            offset += sec0;

            // Section 1: Skip
            offset += sections[1];

            // Section 2: RoPE
            let sec2 = sections[2];
            for i in (0..sec2).step_by(2) {
                let theta = (pos as f32) / freq_base.powf((i as f32) / (sec2 as f32));
                let cos = theta.cos();
                let sin = theta.sin();

                let idx0 = offset + i;
                let idx1 = offset + i + 1;
                let x0 = x_h[idx0];
                let x1 = x_h[idx1];
                x_h[idx0] = x0 * cos - x1 * sin;
                x_h[idx1] = x0 * sin + x1 * cos;
            }
        }
    };

    compute_rope(q, num_q_heads);
    compute_rope(k, num_k_heads);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - (1.0 / (1.0 + (-1.0_f32).exp()))).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let mut x = [1.0, 2.0, 2.0];
        l2_norm(&mut x, 0.0);
        assert!((x[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((x[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((x[2] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_causal_conv1d() {
        let input = [1.0, 2.0];
        let mut state = [0.1, 0.2, 0.3, 0.4]; // 2 past states per channel. Channels = 2.
        let weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // kernel_size = 3. format: [w_0, w_1, w_2] per channel. Wait, weight memory is [kernel_size * channels].
                                                     // Is it channel outer or time outer?
                                                     // causal_conv1d uses `w_offset = c * kernel_size`, so channel outer.
                                                     // weight for c=0: [1.0, 2.0, 3.0]. weight for c=1: [4.0, 5.0, 6.0].
                                                     // state for c=0: [0.1, 0.2]. state for c=1: [0.3, 0.4].

        let mut output = [0.0, 0.0];
        causal_conv1d(&input, &mut state, &weight, 3, 2, &mut output);

        // c = 0: sum = 0.1 * 1.0 + 0.2 * 2.0 + 1.0 * 3.0 = 0.1 + 0.4 + 3.0 = 3.5
        assert!((output[0] - 3.5).abs() < 1e-5);
        // c = 1: sum = 0.3 * 4.0 + 0.4 * 5.0 + 2.0 * 6.0 = 1.2 + 2.0 + 12.0 = 15.2
        assert!((output[1] - 15.2).abs() < 1e-5);

        // State should shift: [0.2, 1.0, 0.4, 2.0]
        assert!((state[0] - 0.2).abs() < 1e-5);
        assert!((state[1] - 1.0).abs() < 1e-5);
        assert!((state[2] - 0.4).abs() < 1e-5);
        assert!((state[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_delta_rule_recurrence() {
        let num_v_heads = 1;
        let head_v_dim = 2;
        let q = [1.0, 2.0];
        let k = [0.5, 0.5];
        let v = [1.0, -1.0];
        let beta = [0.5];
        let gate = [0.0]; // exp(0) = 1.0

        // State S is 2x2.
        // s_h[j * 2 + i] is S[i][j].
        // Memory: [S[0][0], S[1][0], S[0][1], S[1][1]]
        // Let S be identity.
        let mut state = [
            1.0, 0.0, // column 0
            0.0, 1.0, // column 1
        ];
        let mut output = [0.0, 0.0];

        delta_rule_recurrence(
            &q,
            &k,
            &v,
            &beta,
            &gate,
            &mut state,
            &mut output,
            num_v_heads,
            head_v_dim,
        );

        // 1. S_h *= 1.0 -> same
        // 2. delta[j] = (v[j] - S^T * k) * 0.5
        // S^T * k = [1 0; 0 1] * [0.5, 0.5] = [0.5, 0.5]
        // v - S^T k = [0.5, -1.5]
        // delta = [0.25, -0.75]
        // 3. S += k * delta^T
        // k * delta^T = [0.5; 0.5] * [0.25, -0.75] = [0.125, -0.375; 0.125, -0.375]
        // New S = [1.125, -0.375; 0.125, 0.625]
        // 4. out = S^T * q * scale (scale = 1/sqrt(2) = 0.7071)
        // S^T = [1.125, 0.125; -0.375, 0.625]
        // S^T * [1.0, 2.0] = [1.125 + 0.25, -0.375 + 1.25] = [1.375, 0.875]
        // out = [1.375 / sqrt(2), 0.875 / sqrt(2)]

        let s2 = 2.0_f32.sqrt();
        assert!((output[0] - 1.375 / s2).abs() < 1e-5);
        assert!((output[1] - 0.875 / s2).abs() < 1e-5);

        // Check state
        assert!((state[0] - 1.125).abs() < 1e-5); // S[0][0]
        assert!((state[1] - 0.125).abs() < 1e-5); // S[1][0]
        assert!((state[2] - -0.375).abs() < 1e-5); // S[0][1]
        assert!((state[3] - 0.625).abs() < 1e-5); // S[1][1]
    }
}
