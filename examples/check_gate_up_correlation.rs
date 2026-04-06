//! Check correlation between gate and up at each layer
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("Gate-Up correlation analysis:\n");

    for layer_idx in 0..5 {
        let layer = &model.layers()[layer_idx];

        // Attention (simplified)
        let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
        let OwnedQKVWeights::Separate {
            q: q_weight,
            k: _,
            v: v_weight,
        } = &layer.qkv_weight
        else {
            panic!("Expected separate")
        };
        let _ =
            fused_q4k_parallel_matvec(&q_weight.data, &normed, q_weight.in_dim, q_weight.out_dim)
                .expect("test");
        let v =
            fused_q4k_parallel_matvec(&v_weight.data, &normed, v_weight.in_dim, v_weight.out_dim)
                .expect("test");

        let head_dim = hidden_dim / model.config().num_heads;
        let group_size = model.config().num_heads / model.config().num_kv_heads;
        let mut attn_out = Vec::with_capacity(hidden_dim);
        for h in 0..model.config().num_heads {
            let kv_head = h / group_size;
            let st = kv_head * head_dim;
            attn_out.extend_from_slice(&v[st..st + head_dim]);
        }

        let attn_proj = fused_q4k_parallel_matvec(
            &layer.attn_output_weight.data,
            &attn_out,
            layer.attn_output_weight.in_dim,
            layer.attn_output_weight.out_dim,
        )
        .expect("test");
        for i in 0..hidden_dim {
            hidden[i] += attn_proj[i];
        }

        // FFN
        let ffn_input = if let Some(ref norm) = layer.ffn_norm_weight {
            rms_norm(&hidden, norm, eps)
        } else {
            hidden.clone()
        };

        if let Some(ref gate_weight) = layer.ffn_gate_weight {
            let ffn_up = fused_q4k_parallel_matvec(
                &layer.ffn_up_weight.data,
                &ffn_input,
                layer.ffn_up_weight.in_dim,
                layer.ffn_up_weight.out_dim,
            )
            .expect("test");
            let mut ffn_gate = fused_q4k_parallel_matvec(
                &gate_weight.data,
                &ffn_input,
                gate_weight.in_dim,
                gate_weight.out_dim,
            )
            .expect("test");

            // Calculate correlation BEFORE silu
            let dot_pre: f32 = ffn_gate.iter().zip(ffn_up.iter()).map(|(a, b)| a * b).sum();
            let corr_pre = dot_pre / (l2_norm(&ffn_gate) * l2_norm(&ffn_up));

            silu(&mut ffn_gate);

            // Calculate correlation AFTER silu
            let dot_post: f32 = ffn_gate.iter().zip(ffn_up.iter()).map(|(a, b)| a * b).sum();
            let corr_post = dot_post / (l2_norm(&ffn_gate) * l2_norm(&ffn_up));

            // Element-wise product
            let ffn_hidden: Vec<f32> = ffn_gate
                .iter()
                .zip(ffn_up.iter())
                .map(|(a, b)| a * b)
                .collect();

            // Check for sign consistency
            let same_sign = ffn_gate
                .iter()
                .zip(ffn_up.iter())
                .filter(|(a, b)| a.signum() == b.signum())
                .count();

            println!(
                "Layer {}: corr_pre={:.4}, corr_post={:.4}, same_sign={}/{}",
                layer_idx,
                corr_pre,
                corr_post,
                same_sign,
                ffn_gate.len()
            );
            println!(
                "         gate*up L2={:.4}, expected≈{:.4}",
                l2_norm(&ffn_hidden),
                l2_norm(&ffn_gate) * l2_norm(&ffn_up) / (ffn_gate.len() as f32).sqrt()
            );

            let ffn_out = realizar::quantize::fused_q6k_parallel_matvec(
                &layer.ffn_down_weight.data,
                &ffn_hidden,
                layer.ffn_down_weight.in_dim,
                layer.ffn_down_weight.out_dim,
            )
            .expect("test");
            for i in 0..hidden_dim {
                hidden[i] += ffn_out[i];
            }
        }
        println!();
    }
}
