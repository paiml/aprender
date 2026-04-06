//! Debug layer 21 FFN step-by-step

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k, fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (l2_norm(a) * l2_norm(b))
}

fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn fused_matmul(input: &[f32], data: &[u8], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        _ => panic!("Unsupported qtype: {}", qtype),
    }
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let intermediate_dim = model.config().intermediate_dim;
    let eps = model.config().eps;

    println!("=== Layer 21 FFN Debug ===\n");
    println!(
        "hidden_dim: {}, intermediate_dim: {}",
        hidden_dim, intermediate_dim
    );

    // Token 450 = "▁The"
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let mut hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    // Process layers 0-20 to get to layer 21 input
    for layer_idx in 0..21 {
        let layer = &model.layers()[layer_idx];

        // Attention
        let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
        let OwnedQKVWeights::Separate {
            q: q_weight,
            k: k_weight,
            v: v_weight,
        } = &layer.qkv_weight
        else {
            panic!("Expected separate")
        };
        let _q = fused_matmul(
            &normed,
            &q_weight.data,
            q_weight.qtype,
            q_weight.in_dim,
            q_weight.out_dim,
        );
        let _k = fused_matmul(
            &normed,
            &k_weight.data,
            k_weight.qtype,
            k_weight.in_dim,
            k_weight.out_dim,
        );
        let v = fused_matmul(
            &normed,
            &v_weight.data,
            v_weight.qtype,
            v_weight.in_dim,
            v_weight.out_dim,
        );

        let head_dim = hidden_dim / model.config().num_heads;
        let group_size = model.config().num_heads / model.config().num_kv_heads;
        let mut attn_out = Vec::with_capacity(hidden_dim);
        for h in 0..model.config().num_heads {
            let kv_head = h / group_size;
            let start = kv_head * head_dim;
            attn_out.extend_from_slice(&v[start..start + head_dim]);
        }

        let attn_proj = fused_matmul(
            &attn_out,
            &layer.attn_output_weight.data,
            layer.attn_output_weight.qtype,
            layer.attn_output_weight.in_dim,
            layer.attn_output_weight.out_dim,
        );
        for i in 0..hidden_dim {
            hidden[i] += attn_proj[i];
        }

        // FFN
        let ffn_input = rms_norm(&hidden, layer.ffn_norm_weight.as_ref().expect("test"), eps);
        if let Some(ref gate_weight) = layer.ffn_gate_weight {
            let ffn_up = fused_matmul(
                &ffn_input,
                &layer.ffn_up_weight.data,
                layer.ffn_up_weight.qtype,
                layer.ffn_up_weight.in_dim,
                layer.ffn_up_weight.out_dim,
            );
            let mut ffn_gate = fused_matmul(
                &ffn_input,
                &gate_weight.data,
                gate_weight.qtype,
                gate_weight.in_dim,
                gate_weight.out_dim,
            );
            silu(&mut ffn_gate);
            let mut ffn_hidden = vec![0.0f32; intermediate_dim];
            for i in 0..intermediate_dim {
                ffn_hidden[i] = ffn_gate[i] * ffn_up[i];
            }
            let ffn_out = fused_matmul(
                &ffn_hidden,
                &layer.ffn_down_weight.data,
                layer.ffn_down_weight.qtype,
                layer.ffn_down_weight.in_dim,
                layer.ffn_down_weight.out_dim,
            );
            for i in 0..hidden_dim {
                hidden[i] += ffn_out[i];
            }
        }

        if layer_idx == 20 {
            println!("After layer 20 (input to layer 21):");
            println!("  Hidden L2: {:.4}", l2_norm(&hidden));
            println!(
                "  Hidden first 10: {:?}",
                &hidden[0..10]
                    .iter()
                    .map(|x| format!("{:.6}", x))
                    .collect::<Vec<_>>()
            );
        }
    }

    // Now process layer 21 with detailed tracing
    let layer = &model.layers()[21];
    let hidden_before = hidden.clone();

    println!("\n=== Layer 21 Detailed Trace ===\n");

    // Attention
    let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
    println!("1. Attention norm output L2: {:.4}", l2_norm(&normed));

    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: k_weight,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate")
    };
    let _q = fused_matmul(
        &normed,
        &q_weight.data,
        q_weight.qtype,
        q_weight.in_dim,
        q_weight.out_dim,
    );
    let _k = fused_matmul(
        &normed,
        &k_weight.data,
        k_weight.qtype,
        k_weight.in_dim,
        k_weight.out_dim,
    );
    let v = fused_matmul(
        &normed,
        &v_weight.data,
        v_weight.qtype,
        v_weight.in_dim,
        v_weight.out_dim,
    );

    let head_dim = hidden_dim / model.config().num_heads;
    let group_size = model.config().num_heads / model.config().num_kv_heads;
    let mut attn_out = Vec::with_capacity(hidden_dim);
    for h in 0..model.config().num_heads {
        let kv_head = h / group_size;
        let start = kv_head * head_dim;
        attn_out.extend_from_slice(&v[start..start + head_dim]);
    }

    let attn_proj = fused_matmul(
        &attn_out,
        &layer.attn_output_weight.data,
        layer.attn_output_weight.qtype,
        layer.attn_output_weight.in_dim,
        layer.attn_output_weight.out_dim,
    );

    println!("2. Attention projection L2: {:.4}", l2_norm(&attn_proj));
    println!(
        "   Cosine(hidden, attn_proj): {:.4}",
        cosine_similarity(&hidden, &attn_proj)
    );

    for i in 0..hidden_dim {
        hidden[i] += attn_proj[i];
    }
    println!("3. After attention residual L2: {:.4}", l2_norm(&hidden));

    // FFN with detailed tracing
    let ffn_norm_weight = layer.ffn_norm_weight.as_ref().expect("test");
    let ffn_input = rms_norm(&hidden, ffn_norm_weight, eps);

    println!(
        "\n4. FFN norm (input to FFN) L2: {:.4}",
        l2_norm(&ffn_input)
    );
    println!(
        "   FFN norm first 10: {:?}",
        &ffn_input[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    let gate_weight = layer.ffn_gate_weight.as_ref().expect("test");
    let up_weight = &layer.ffn_up_weight;
    let down_weight = &layer.ffn_down_weight;

    println!("\n   Weight info:");
    println!(
        "   Gate: in={}, out={}, qtype={}",
        gate_weight.in_dim, gate_weight.out_dim, gate_weight.qtype
    );
    println!(
        "   Up:   in={}, out={}, qtype={}",
        up_weight.in_dim, up_weight.out_dim, up_weight.qtype
    );
    println!(
        "   Down: in={}, out={}, qtype={}",
        down_weight.in_dim, down_weight.out_dim, down_weight.qtype
    );

    // FFN up projection
    let ffn_up = fused_matmul(
        &ffn_input,
        &up_weight.data,
        up_weight.qtype,
        up_weight.in_dim,
        up_weight.out_dim,
    );
    println!("\n5. FFN up L2: {:.4}", l2_norm(&ffn_up));
    println!(
        "   FFN up first 10: {:?}",
        &ffn_up[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "   FFN up min/max: {:.4} / {:.4}",
        ffn_up.iter().copied().fold(f32::INFINITY, f32::min),
        ffn_up.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // FFN gate projection
    let mut ffn_gate = fused_matmul(
        &ffn_input,
        &gate_weight.data,
        gate_weight.qtype,
        gate_weight.in_dim,
        gate_weight.out_dim,
    );
    println!("\n6. FFN gate (before SiLU) L2: {:.4}", l2_norm(&ffn_gate));
    println!(
        "   FFN gate first 10: {:?}",
        &ffn_gate[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    silu(&mut ffn_gate);
    println!("\n7. FFN gate (after SiLU) L2: {:.4}", l2_norm(&ffn_gate));
    println!(
        "   FFN gate first 10: {:?}",
        &ffn_gate[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // Gate * Up
    let mut ffn_hidden = vec![0.0f32; intermediate_dim];
    for i in 0..intermediate_dim {
        ffn_hidden[i] = ffn_gate[i] * ffn_up[i];
    }
    println!(
        "\n8. FFN hidden (gate * up) L2: {:.4}",
        l2_norm(&ffn_hidden)
    );
    println!(
        "   FFN hidden first 10: {:?}",
        &ffn_hidden[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "   FFN hidden min/max: {:.4} / {:.4}",
        ffn_hidden.iter().copied().fold(f32::INFINITY, f32::min),
        ffn_hidden.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // FFN down projection
    let ffn_out = fused_matmul(
        &ffn_hidden,
        &down_weight.data,
        down_weight.qtype,
        down_weight.in_dim,
        down_weight.out_dim,
    );
    println!("\n9. FFN down (output) L2: {:.4}", l2_norm(&ffn_out));
    println!(
        "   FFN down first 10: {:?}",
        &ffn_out[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // Analysis
    println!("\n=== Analysis ===\n");
    println!("Hidden before layer 21 L2: {:.4}", l2_norm(&hidden_before));
    println!("FFN contribution L2: {:.4}", l2_norm(&ffn_out));
    println!(
        "Cosine(hidden_before, FFN_out): {:.4}",
        cosine_similarity(&hidden_before, &ffn_out)
    );

    // Apply FFN residual
    for i in 0..hidden_dim {
        hidden[i] += ffn_out[i];
    }
    println!("\nHidden after layer 21 L2: {:.4}", l2_norm(&hidden));
    println!("  Expected (HF): ~72.40");
    println!("  Ratio (ours/HF): {:.4}", l2_norm(&hidden) / 72.40);

    // Dequantize down weight for analysis
    println!("\n=== Down Weight Analysis ===\n");
    let dequant_down = dequantize_q4_k(&down_weight.data).expect("Failed to dequantize");
    println!("Dequantized down weight L2: {:.4}", l2_norm(&dequant_down));
    println!("Expected (HF layer 21): ~61.25");

    // Check first row of down weight
    let row_0: Vec<f32> = dequant_down[0..intermediate_dim].to_vec();
    println!("Down weight row 0 L2: {:.4}", l2_norm(&row_0));
    println!(
        "Down weight row 0 first 10: {:?}",
        &row_0[0..10]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );
}
