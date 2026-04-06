//! Check index 5475 across layers
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q4_k, fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;
const Q4_K_BLOCK_SIZE: usize = 144;

#[allow(dead_code)]
fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
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
        _ => panic!(""),
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

    println!("=== Checking index 5475 across layers ===\n");

    // First, check the weights at row 5475 for each layer
    println!("Weight row 5475 (first superblock) across layers:");
    for layer_idx in 0..5 {
        let layer = &model.layers()[layer_idx];
        if let Some(ref gw) = layer.ffn_gate_weight {
            // Row 5475 starts at byte offset 5475 * bytes_per_row
            // For Q4_K with 2048 input: each row has 2048/256 = 8 superblocks = 8*144 = 1152 bytes
            let bytes_per_row = (gw.in_dim / 256) * Q4_K_BLOCK_SIZE;
            let row_start = 5475 * bytes_per_row;
            let first_block = &gw.data[row_start..row_start + Q4_K_BLOCK_SIZE];
            let dequant = dequantize_q4_k(first_block).expect("test");
            let l2: f32 = dequant.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!(
                "  Layer {} gate row 5475 block 0: L2={:.4}, first 5: {:?}",
                layer_idx,
                l2,
                &dequant[..5]
            );
        }
    }

    println!("\nFFN output values at index 5475 per layer:");

    for layer_idx in 0..5 {
        let layer = &model.layers()[layer_idx];
        let normed = rms_norm(&hidden, &layer.attn_norm_weight, eps);
        let OwnedQKVWeights::Separate {
            q: q_weight,
            k: _,
            v: v_weight,
        } = &layer.qkv_weight
        else {
            panic!("")
        };
        let _ = fused_matmul(
            &normed,
            &q_weight.data,
            q_weight.qtype,
            q_weight.in_dim,
            q_weight.out_dim,
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
            attn_out.extend_from_slice(&v[kv_head * head_dim..(kv_head + 1) * head_dim]);
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
        let ffn_input = layer
            .ffn_norm_weight
            .as_ref()
            .map_or(hidden.clone(), |n| rms_norm(&hidden, n, eps));

        if let Some(ref gw) = layer.ffn_gate_weight {
            let up = fused_matmul(
                &ffn_input,
                &layer.ffn_up_weight.data,
                layer.ffn_up_weight.qtype,
                layer.ffn_up_weight.in_dim,
                layer.ffn_up_weight.out_dim,
            );
            let mut gate = fused_matmul(&ffn_input, &gw.data, gw.qtype, gw.in_dim, gw.out_dim);

            println!(
                "  Layer {}: gate[5475]={:.4}, up[5475]={:.4}",
                layer_idx, gate[5475], up[5475]
            );

            silu(&mut gate);
            let ffn_h: Vec<f32> = gate.iter().zip(up.iter()).map(|(a, b)| a * b).collect();
            let out = fused_matmul(
                &ffn_h,
                &layer.ffn_down_weight.data,
                layer.ffn_down_weight.qtype,
                layer.ffn_down_weight.in_dim,
                layer.ffn_down_weight.out_dim,
            );
            for i in 0..hidden_dim {
                hidden[i] += out[i];
            }
        }
    }
}
