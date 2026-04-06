use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    // Load raw model
    let data =
        fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").expect("test");
    let model = GGUFModel::from_bytes(&data).expect("test");

    // Get token_embd tensor info
    let tensor = model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("test");
    println!("token_embd.weight:");
    println!("  dims: {:?}", tensor.dims);
    println!(
        "  qtype: {} ({})",
        tensor.qtype,
        if tensor.qtype == 0 { "F32" } else { "other" }
    );
    println!("  offset: {}", tensor.offset);

    // Load token embedding as f32
    let embed = model
        .get_tensor_f32("token_embd.weight", &data)
        .expect("test");
    println!("  loaded len: {}", embed.len());

    let hidden_dim = 2048;
    let _vocab_size = 32000;

    // Check token 0 embedding (UNK)
    println!("\nToken 0 (UNK) embedding (first 10 values):");
    for (i, &v) in embed[0..10].iter().enumerate() {
        println!("  [{}] = {:.6}", i, v);
    }

    // Check token 1 embedding (BOS)
    let bos_start = hidden_dim;
    println!("\nToken 1 (BOS) embedding (first 10 values):");
    for (i, &v) in embed[bos_start..bos_start + 10].iter().enumerate() {
        println!("  [{}] = {:.6}", i, v);
    }

    // Check token 3681 embedding (▁Paris)
    let paris_start = 3681 * hidden_dim;
    println!("\nToken 3681 (▁Paris) embedding (first 10 values):");
    for (i, &v) in embed[paris_start..paris_start + 10].iter().enumerate() {
        println!("  [{}] = {:.6}", i, v);
    }

    // Check if embeddings are different for different tokens
    let bos_emb = &embed[bos_start..bos_start + hidden_dim];
    let paris_emb = &embed[paris_start..paris_start + hidden_dim];

    let diff: f32 = bos_emb
        .iter()
        .zip(paris_emb.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!(
        "\nL1 distance between BOS and Paris embeddings: {:.4}",
        diff
    );

    // Check embedding stats
    let min = embed.iter().copied().fold(f32::INFINITY, f32::min);
    let max = embed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = embed.iter().sum::<f32>() / embed.len() as f32;
    println!("\nEmbedding stats:");
    println!("  min: {:.6}", min);
    println!("  max: {:.6}", max);
    println!("  mean: {:.6}", mean);

    let non_zero = embed.iter().filter(|&&x| x != 0.0).count();
    println!(
        "  non-zero: {}/{} ({:.1}%)",
        non_zero,
        embed.len(),
        100.0 * non_zero as f64 / embed.len() as f64
    );
}
