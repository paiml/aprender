//! PAR-001c: Detailed layer 0 trace to find divergence point
//!
//! This traces each step of layer 0:
//! 1. Embedding lookup
//! 2. Attention norm (RMS)
//! 3. QKV projection
//! 4. RoPE application
//! 5. Attention computation
//! 6. Output projection + residual
//! 7. FFN norm
//! 8. FFN (gate * up -> SiLU -> down)
//! 9. FFN residual

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

fn minmax(v: &[f32]) -> (f32, f32) {
    (
        v.iter().copied().fold(f32::INFINITY, f32::min),
        v.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    )
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001c: Detailed Layer 0 Trace ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    println!("Config:");
    println!(
        "  hidden_dim={}, num_heads={}, num_kv_heads={}",
        hidden_dim, num_heads, num_kv_heads
    );
    println!("  head_dim={}, kv_dim={}", head_dim, kv_dim);
    println!();

    // Test token
    let token_id: u32 = 26222; // "Once"
    println!(
        "Token: {} (id={})",
        vocab.get(token_id as usize).unwrap_or(&"<UNK>".to_string()),
        token_id
    );
    println!();

    // Step 1: Embedding
    let hidden = model.embed(&[token_id]);
    println!("1. Embedding:");
    println!(
        "   L2={:.4}, mean={:.6}, range=[{:.4}, {:.4}]",
        l2_norm(&hidden),
        mean(&hidden),
        minmax(&hidden).0,
        minmax(&hidden).1
    );
    println!("   First 5: {:?}", &hidden[..5]);

    // Step 2: Attention norm (RMS)
    let layer = &model.layers()[0];
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + model.config().eps).sqrt();
    let inv_rms = 1.0 / rms;
    let normed: Vec<f32> = hidden
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| x * inv_rms * w)
        .collect();

    println!("\n2. Attention Norm (RMS):");
    println!("   rms={:.6}, inv_rms={:.4}", rms, inv_rms);
    println!(
        "   L2={:.4}, mean={:.6}, range=[{:.4}, {:.4}]",
        l2_norm(&normed),
        mean(&normed),
        minmax(&normed).0,
        minmax(&normed).1
    );
    println!("   First 5: {:?}", &normed[..5]);

    // Step 3: QKV projection
    let qkv = model
        .qkv_matmul(&normed, &layer.qkv_weight)
        .expect("QKV matmul failed");
    let q = &qkv[..hidden_dim];
    let k = &qkv[hidden_dim..hidden_dim + kv_dim];
    let v = &qkv[hidden_dim + kv_dim..];

    println!("\n3. QKV Projection:");
    println!("   QKV total len={}", qkv.len());
    println!(
        "   Q: L2={:.4}, range=[{:.4}, {:.4}], first 5: {:?}",
        l2_norm(q),
        minmax(q).0,
        minmax(q).1,
        &q[..5]
    );
    println!(
        "   K: L2={:.4}, range=[{:.4}, {:.4}], first 5: {:?}",
        l2_norm(k),
        minmax(k).0,
        minmax(k).1,
        &k[..5]
    );
    println!(
        "   V: L2={:.4}, range=[{:.4}, {:.4}], first 5: {:?}",
        l2_norm(v),
        minmax(v).0,
        minmax(v).1,
        &v[..5]
    );

    // Step 4: Run a single forward pass to see final output
    println!("\n4. Full Forward Pass with KV Cache:");
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 256);
    let logits = model
        .forward_single_with_cache(token_id, &mut cache, 0)
        .expect("Forward failed");

    println!(
        "   Logits: L2={:.4}, range=[{:.4}, {:.4}]",
        l2_norm(&logits),
        minmax(&logits).0,
        minmax(&logits).1
    );

    // Get top predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n   Top 5 predictions:");
    for (rank, (idx, score)) in indexed.iter().take(5).enumerate() {
        let token_str = if *idx < vocab.len() {
            vocab[*idx].replace('▁', " ")
        } else {
            format!("<{}>", idx)
        };
        println!(
            "     #{}: '{}' (id={}, score={:.4})",
            rank + 1,
            token_str,
            idx,
            score
        );
    }

    // Check cache contents after first token
    println!("\n5. KV Cache after first token:");
    let k_cache = cache.get_k(0);
    let v_cache = cache.get_v(0);
    println!(
        "   Layer 0 K cache: len={}, first 5: {:?}",
        k_cache.len(),
        &k_cache[..5.min(k_cache.len())]
    );
    println!(
        "   Layer 0 V cache: len={}, first 5: {:?}",
        v_cache.len(),
        &v_cache[..5.min(v_cache.len())]
    );

    // Step 5: Try a second token and see if cache is used correctly
    println!("\n6. Second Token Forward (testing cache):");
    let token2: u32 = 14990; // " upon" or similar
    let logits2 = model
        .forward_single_with_cache(token2, &mut cache, 1)
        .expect("Forward 2 failed");

    let mut indexed2: Vec<(usize, f32)> = logits2.iter().copied().enumerate().collect();
    indexed2.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "   Token: {} (id={})",
        vocab.get(token2 as usize).unwrap_or(&"<UNK>".to_string()),
        token2
    );
    println!("   Top 3 predictions:");
    for (rank, (idx, score)) in indexed2.iter().take(3).enumerate() {
        let token_str = if *idx < vocab.len() {
            vocab[*idx].replace('▁', " ")
        } else {
            format!("<{}>", idx)
        };
        println!(
            "     #{}: '{}' (id={}, score={:.4})",
            rank + 1,
            token_str,
            idx,
            score
        );
    }

    println!("\n=== Trace complete ===");
}
