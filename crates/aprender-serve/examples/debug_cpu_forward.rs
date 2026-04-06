//! Debug CPU forward to compare with my trace
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");

    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_layers = model.config().num_layers;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / model.config().num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let mut cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);

    // Direct call to forward_single_with_cache
    let logits = model.forward_single_with_cache(791, &mut cache, 0)?;

    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "CPU forward: argmax={}, logit={:.4}",
        argmax, logits[argmax]
    );
    println!("Logits sum: {:.4}", logits.iter().sum::<f32>());
    println!(
        "First 10 logits: {:?}",
        &logits[..10]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    Ok(())
}
