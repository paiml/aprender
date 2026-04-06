//! Trace SafeTensors inference to diagnose garbage output

use realizar::apr_transformer::GenerateConfig;
use realizar::safetensors_infer::SafetensorsToAprConverter;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = Path::new("/home/noah/models/qwen2.5-coder-0.5b-instruct/model.safetensors");

    println!("Loading SafeTensors model...");
    let transformer = SafetensorsToAprConverter::convert(model_path)?;

    println!("\n=== Model Config ===");
    println!("  hidden_dim: {}", transformer.config().hidden_dim);
    println!("  num_layers: {}", transformer.config().num_layers);
    println!("  num_heads: {}", transformer.config().num_heads);
    println!("  num_kv_heads: {}", transformer.config().num_kv_heads);
    println!("  vocab_size: {}", transformer.config().vocab_size);
    println!(
        "  intermediate_dim: {}",
        transformer.config().intermediate_dim
    );

    let hidden_dim = transformer.config().hidden_dim;

    println!("\n=== Embedding Stats ===");
    let emb = &transformer.token_embedding;
    let emb_len = emb.len();
    let expected = transformer.config().vocab_size * hidden_dim;
    println!("  embedding len: {} (expected: {})", emb_len, expected);

    // Check token 0 (padding)
    let tok0_start = 0;
    let tok0 = &emb[tok0_start..tok0_start + 10.min(hidden_dim)];
    println!("  token 0 first 10: {:?}", tok0);

    // Check token 220 (commonly used)
    let tok220_start = 220 * hidden_dim;
    let tok220 = &emb[tok220_start..tok220_start + 10.min(hidden_dim)];
    println!("  token 220 first 10: {:?}", tok220);
    let tok220_full = &emb[tok220_start..tok220_start + hidden_dim];
    let tok220_l2: f32 = tok220_full.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("  token 220 L2 norm: {:.6}", tok220_l2);

    // Check token 17
    let tok17_start = 17 * hidden_dim;
    let tok17 = &emb[tok17_start..tok17_start + 10.min(hidden_dim)];
    println!("  token 17 first 10: {:?}", tok17);

    // Check token 1000 (should be a real word token, not special)
    let tok1000_start = 1000 * hidden_dim;
    let tok1000 = &emb[tok1000_start..tok1000_start + 10.min(hidden_dim)];
    println!("  token 1000 first 10: {:?}", tok1000);
    let tok1000_full = &emb[tok1000_start..tok1000_start + hidden_dim];
    let tok1000_l2: f32 = tok1000_full.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("  token 1000 L2 norm: {:.6}", tok1000_l2);

    // Check token 10000 (should be a real word token)
    let tok10000_start = 10000 * hidden_dim;
    let tok10000 = &emb[tok10000_start..tok10000_start + 10.min(hidden_dim)];
    println!("  token 10000 first 10: {:?}", tok10000);
    let tok10000_full = &emb[tok10000_start..tok10000_start + hidden_dim];
    let tok10000_l2: f32 = tok10000_full.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("  token 10000 L2 norm: {:.6}", tok10000_l2);

    // Find first non-zero token
    let mut first_nonzero = None;
    for i in 0..1000 {
        let start = i * hidden_dim;
        let tok_l2: f32 = emb[start..start + hidden_dim]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        if tok_l2 > 0.1 {
            first_nonzero = Some((i, tok_l2));
            break;
        }
    }
    println!("  first non-zero token in 0..1000: {:?}", first_nonzero);

    let emb_mean: f32 = emb.iter().sum::<f32>() / emb_len as f32;
    let emb_min = emb.iter().copied().fold(f32::INFINITY, f32::min);
    let emb_max = emb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  mean: {:.6}, min: {:.6}, max: {:.6}",
        emb_mean, emb_min, emb_max
    );

    println!("\n=== LM Head Stats ===");
    let lm = &transformer.lm_head_weight;
    let lm_len = lm.len();
    let lm_expected = transformer.config().vocab_size * transformer.config().hidden_dim;
    println!("  lm_head len: {} (expected: {})", lm_len, lm_expected);
    println!("  first 10: {:?}", &lm[..10.min(lm_len)]);
    let lm_mean: f32 = lm.iter().sum::<f32>() / lm_len as f32;
    let lm_min = lm.iter().copied().fold(f32::INFINITY, f32::min);
    let lm_max = lm.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  mean: {:.6}, min: {:.6}, max: {:.6}",
        lm_mean, lm_min, lm_max
    );

    // Simple inference test
    println!("\n=== Simple Forward Pass ===");
    // Token ID 220 is often a space or common token
    let test_tokens: Vec<u32> = vec![220, 17, 10];
    println!("  input tokens: {:?}", test_tokens);

    let config = GenerateConfig::default();

    let result = transformer.generate_with_cache(&test_tokens, &config)?;
    println!("  output tokens: {:?}", result);

    // Check if output tokens are reasonable
    let generated = &result[test_tokens.len()..];
    println!("  generated tokens: {:?}", generated);

    for &tok in generated {
        if tok > transformer.config().vocab_size as u32 {
            println!(
                "  WARNING: Token {} > vocab_size {}!",
                tok,
                transformer.config().vocab_size
            );
        }
    }

    Ok(())
}
