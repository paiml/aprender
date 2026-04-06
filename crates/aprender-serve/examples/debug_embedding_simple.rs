//! Simple debug - check if APR embedding matches first few values
use realizar::apr_transformer::AprTransformer;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/tmp/qwen2.5-coder-1.5b-q4k.apr";

    println!("Loading APR...");
    let start = Instant::now();
    let apr = AprTransformer::from_apr_file(apr_path)?;
    println!("Loaded in {:.2}s", start.elapsed().as_secs_f32());

    println!("\nConfig:");
    println!("  hidden_dim: {}", apr.config.hidden_dim);
    println!("  vocab_size: {}", apr.config.vocab_size);
    println!("  num_layers: {}", apr.config.num_layers);
    println!("  intermediate_dim: {}", apr.config.intermediate_dim);

    let hidden = apr.config.hidden_dim;
    println!(
        "\nToken embedding size: {} (expect {})",
        apr.token_embedding.len(),
        apr.config.vocab_size * hidden
    );

    // Check token 0 embedding
    println!("\nToken 0 embedding (first 10):");
    println!("  {:?}", &apr.token_embedding[..10]);

    // Check token 151643 (BOS) embedding
    let bos = 151643usize;
    let bos_start = bos * hidden;
    if bos_start + 10 < apr.token_embedding.len() {
        println!("\nToken {} (BOS) embedding (first 10):", bos);
        println!("  {:?}", &apr.token_embedding[bos_start..bos_start + 10]);
    }

    // Check LM head
    println!(
        "\nLM head weight size: {} (expect {})",
        apr.lm_head_weight.len(),
        hidden * apr.config.vocab_size
    );

    // Check layer 0 weights
    if !apr.layers.is_empty() {
        let layer = &apr.layers[0];
        println!("\nLayer 0 weights:");
        println!("  attn_norm: {} elements", layer.attn_norm_weight.len());
        println!("  qkv_weight: {} elements", layer.qkv_weight.len());
        println!(
            "  ffn_gate: {:?} elements",
            layer.ffn_gate_weight.as_ref().map(|w| w.len())
        );
        println!("  ffn_up: {} elements", layer.ffn_up_weight.len());
        println!("  ffn_down: {} elements", layer.ffn_down_weight.len());
    }

    Ok(())
}
