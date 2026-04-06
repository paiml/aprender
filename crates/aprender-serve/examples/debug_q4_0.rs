use realizar::gguf::{GGUFModel, MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 0.5B Q4_0 model (produces garbage)
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";

    eprintln!("Loading: {}", path);
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    eprintln!("\n=== Model Config ===");
    eprintln!("architecture: {}", model.config().architecture);
    eprintln!("hidden_dim: {}", model.config().hidden_dim);
    eprintln!("num_heads: {}", model.config().num_heads);
    eprintln!("num_kv_heads: {}", model.config().num_kv_heads);
    eprintln!("vocab_size: {}", model.config().vocab_size);
    eprintln!("intermediate_dim: {}", model.config().intermediate_dim);

    let hidden_dim = model.config().hidden_dim;
    let head_dim = hidden_dim / model.config().num_heads;
    let kv_dim = model.config().num_kv_heads * head_dim;

    eprintln!("\n=== Derived Dimensions ===");
    eprintln!("head_dim: {}", head_dim);
    eprintln!("kv_dim: {}", kv_dim);

    eprintln!("\n=== Layer 0 QKV ===");
    let layer = &model.layers()[0];
    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Fused(t) => {
            eprintln!(
                "Fused: in_dim={}, out_dim={}, qtype={}, data_len={}",
                t.in_dim,
                t.out_dim,
                t.qtype,
                t.data.len()
            );
        },
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            eprintln!(
                "Q: in_dim={}, out_dim={}, qtype={}, data_len={}",
                q.in_dim,
                q.out_dim,
                q.qtype,
                q.data.len()
            );
            eprintln!(
                "K: in_dim={}, out_dim={}, qtype={}, data_len={}",
                k.in_dim,
                k.out_dim,
                k.qtype,
                k.data.len()
            );
            eprintln!(
                "V: in_dim={}, out_dim={}, qtype={}, data_len={}",
                v.in_dim,
                v.out_dim,
                v.qtype,
                v.data.len()
            );

            // Verify expected sizes for Q4_0
            let q_expected = (q.in_dim * q.out_dim).div_ceil(32) * 18;
            let k_expected = (k.in_dim * k.out_dim).div_ceil(32) * 18;
            let v_expected = (v.in_dim * v.out_dim).div_ceil(32) * 18;
            eprintln!("\nExpected Q4_0 byte sizes:");
            eprintln!("Q expected: {} (actual: {})", q_expected, q.data.len());
            eprintln!("K expected: {} (actual: {})", k_expected, k.data.len());
            eprintln!("V expected: {} (actual: {})", v_expected, v.data.len());
        },
    }

    eprintln!("\n=== Layer 0 FFN ===");
    eprintln!(
        "ffn_up: in_dim={}, out_dim={}, qtype={}, data_len={}",
        layer.ffn_up_weight.in_dim,
        layer.ffn_up_weight.out_dim,
        layer.ffn_up_weight.qtype,
        layer.ffn_up_weight.data.len()
    );
    eprintln!(
        "ffn_down: in_dim={}, out_dim={}, qtype={}, data_len={}",
        layer.ffn_down_weight.in_dim,
        layer.ffn_down_weight.out_dim,
        layer.ffn_down_weight.qtype,
        layer.ffn_down_weight.data.len()
    );
    if let Some(gate) = &layer.ffn_gate_weight {
        eprintln!(
            "ffn_gate: in_dim={}, out_dim={}, qtype={}, data_len={}",
            gate.in_dim,
            gate.out_dim,
            gate.qtype,
            gate.data.len()
        );
    }

    // Also check the raw GGUF tensor info
    eprintln!("\n=== Raw GGUF Tensor Info (Layer 0) ===");
    let data = std::fs::read(path)?;
    let gguf_model = GGUFModel::from_bytes(&data)?;
    for tensor in &gguf_model.tensors {
        if tensor.name.contains("blk.0.") {
            eprintln!(
                "{}: dims={:?}, qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );
        }
    }

    Ok(())
}
