//! Quick diagnostic: does the 7B teacher actually populate q4k_layers fully?

use realizar::apr_transformer::AprTransformer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr";
    println!("Loading {}...", path);
    let t = AprTransformer::from_apr_file(path)?;

    println!("num_layers: {}", t.config().num_layers);

    match &t.q4k_layers {
        None => {
            println!("❌ q4k_layers is None — ALL inference uses F32 matmul!");
        },
        Some(layers) => {
            println!("✓ q4k_layers populated with {} entries", layers.len());
            for (i, l) in layers.iter().enumerate() {
                let q = l.attn_q_weight.as_ref().map(|v| v.len()).unwrap_or(0);
                let k = l.attn_k_weight.as_ref().map(|v| v.len()).unwrap_or(0);
                let v = l.attn_v_weight.as_ref().map(|v| v.len()).unwrap_or(0);
                let o = l.attn_output_weight.as_ref().map(|v| v.len()).unwrap_or(0);
                let g = l.ffn_gate_weight.as_ref().map(|v| v.len()).unwrap_or(0);
                let u = l.ffn_up_weight.as_ref().map(|v| v.len()).unwrap_or(0);
                let d = l.ffn_down_weight.as_ref().map(|v| v.len()).unwrap_or(0);
                if i < 5 || i == 27 {
                    println!(
                        "  layer {}: Q={}b K={}b V={}b O={}b G={}b U={}b D={}b",
                        i, q, k, v, o, g, u, d
                    );
                }
            }
        },
    }
    Ok(())
}
