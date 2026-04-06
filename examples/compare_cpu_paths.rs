//! Compare different CPU forward paths (KV cache vs fresh)
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_layers = model.config().num_layers;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / model.config().num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let token: u32 = 791;

    // Path 1: forward_single_with_cache (position 0)
    let mut cache1 = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let logits1 = model.forward_single_with_cache(token, &mut cache1, 0)?;
    let argmax1 = logits1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "Path 1 (cache pos=0): argmax={}, logit={:.4}, sum={:.4}",
        argmax1,
        logits1[argmax1],
        logits1.iter().sum::<f32>()
    );

    // Path 2: forward_single_with_cache (fresh cache)
    let mut cache2 = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let logits2 = model.forward_single_with_cache(token, &mut cache2, 0)?;

    // Check for NaN
    let nan_count = logits2.iter().filter(|x| x.is_nan()).count();
    let inf_count = logits2.iter().filter(|x| x.is_infinite()).count();
    println!(
        "Path 2 (fresh cache): len={}, nan={}, inf={}",
        logits2.len(),
        nan_count,
        inf_count
    );
    println!("  first 5 logits: {:?}", &logits2[..5]);

    if nan_count == 0 && inf_count == 0 {
        let argmax2 = logits2
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        println!("  argmax={}, logit={:.4}", argmax2, logits2[argmax2]);

        // Should match
        if argmax1 == argmax2 {
            println!("\n✓ Both paths produced same argmax");
        } else {
            println!("\n✗ Paths diverged: {} vs {}", argmax1, argmax2);
        }
    }

    Ok(())
}
