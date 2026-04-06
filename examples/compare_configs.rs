use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Model Config Comparison ===\n");

    let path_05b = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let path_15b = "/home/noah/.cache/huggingface/models/qwen2.5-coder-1.5b-gguf/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    for (name, path) in [("Qwen2-0.5B", path_05b), ("Qwen2.5-Coder-1.5B", path_15b)] {
        println!("=== {} ===", name);
        let mapped = MappedGGUFModel::from_path(path)?;
        let model = OwnedQuantizedModel::from_mapped(&mapped)?;

        let c = &model.config();
        println!("  architecture: {}", c.architecture);
        println!("  hidden_dim: {}", c.hidden_dim);
        println!("  num_layers: {}", c.num_layers);
        println!("  num_heads: {}", c.num_heads);
        println!("  num_kv_heads: {}", c.num_kv_heads);
        println!("  vocab_size: {}", c.vocab_size);
        println!("  intermediate_dim: {}", c.intermediate_dim);
        println!("  eps: {:e}", c.eps);
        println!("  rope_theta: {}", c.rope_theta);
        println!("  rope_type: {}", c.rope_type);

        // Check QKV weight shapes
        let layer0 = &model.layers()[0];
        match &layer0.qkv_weight {
            OwnedQKVWeights::Separate { q, k, v } => {
                println!("  QKV: separate");
                println!("    Q: {}x{} qtype={}", q.in_dim, q.out_dim, q.qtype);
                println!("    K: {}x{} qtype={}", k.in_dim, k.out_dim, k.qtype);
                println!("    V: {}x{} qtype={}", v.in_dim, v.out_dim, v.qtype);
            },
            OwnedQKVWeights::Fused(w) => {
                println!("  QKV: fused {}x{} qtype={}", w.in_dim, w.out_dim, w.qtype);
            },
        }

        // Check head dimensions
        let head_dim = c.hidden_dim / c.num_heads;
        let kv_dim = c.num_kv_heads * head_dim;
        println!("  head_dim: {}", head_dim);
        println!("  kv_dim: {}", kv_dim);
        println!("  GQA ratio: {}:1", c.num_heads / c.num_kv_heads);
        println!();
    }

    Ok(())
}
