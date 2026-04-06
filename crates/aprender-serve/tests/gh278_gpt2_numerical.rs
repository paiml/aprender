//! GH-278: Numerical verification of GPT-2 GGUF inference
//!
//! This test loads the GPT-2 124M GGUF model and traces numerical values
//! at each forward pass step to identify exactly where inference diverges
//! from the HuggingFace ground truth oracle.
//!
//! Ground truth: "The capital of France is" → " the capital of the French Republic, and..."
//! First generated token: 262 (" the")

use std::path::Path;

/// GPT-2 prompt "The capital of France is" tokenized (BPE)
const PROMPT_TOKENS: &[u32] = &[464, 3139, 286, 4881, 318];

/// Expected first generated token from HuggingFace ground truth
const EXPECTED_FIRST_TOKEN: u32 = 262; // " the"

#[test]
fn gh278_gpt2_gguf_forward_numerical() {
    let model_path = "/home/noah/src/tiny-model-ground-truth/models/gpt2-124m-int4.gguf";
    if !Path::new(model_path).exists() {
        eprintln!("SKIP: GPT-2 GGUF not found at {model_path}");
        return;
    }

    // Load model
    let mapped =
        realizar::gguf::MappedGGUFModel::from_path(model_path).expect("Failed to load GPT-2 GGUF");
    let model = realizar::gguf::OwnedQuantizedModel::from_mapped(&mapped)
        .expect("Failed to create owned model");

    // Print model config
    eprintln!("=== GPT-2 124M GGUF Model Config ===");
    eprintln!("  architecture: {}", model.config().architecture);
    eprintln!("  hidden_dim: {}", model.config().hidden_dim);
    eprintln!("  num_layers: {}", model.config().num_layers);
    eprintln!("  num_heads: {}", model.config().num_heads);
    eprintln!("  num_kv_heads: {}", model.config().num_kv_heads);
    eprintln!("  intermediate_dim: {}", model.config().intermediate_dim);
    eprintln!("  vocab_size: {}", model.config().vocab_size);
    eprintln!("  eps: {}", model.config().eps);
    eprintln!(
        "  has position_embedding: {}",
        model.position_embedding().is_some()
    );

    // Print first few embedding values for token 464 ("The")
    let embed_start = 464 * model.config().hidden_dim;
    eprintln!("\n=== Token Embedding for 'The' (id=464) ===");
    eprintln!(
        "  first 5: {:?}",
        &model.token_embedding()[embed_start..embed_start + 5]
    );

    // Print position embedding values for position 0
    if let Some(pos_emb) = model.position_embedding() {
        eprintln!("\n=== Position Embedding (pos=0) ===");
        eprintln!("  first 5: {:?}", &pos_emb[..5]);
        eprintln!("  total len: {} (expected {})", pos_emb.len(), 1024 * 768);
    } else {
        eprintln!("\n!!! MISSING POSITION EMBEDDING !!!");
    }

    // Print layer 0 weight diagnostics
    let layer0 = &model.layers()[0];
    eprintln!("\n=== Layer 0 Weight Diagnostics ===");
    eprintln!(
        "  attn_norm_weight[0..3]: {:?}",
        &layer0.attn_norm_weight[..3]
    );
    eprintln!("  has attn_norm_bias: {}", layer0.attn_norm_bias.is_some());
    eprintln!(
        "  has ffn_norm_weight: {}",
        layer0.ffn_norm_weight.is_some()
    );
    eprintln!("  has ffn_norm_bias: {}", layer0.ffn_norm_bias.is_some());
    eprintln!(
        "  has ffn_gate_weight: {}",
        layer0.ffn_gate_weight.is_some()
    );
    eprintln!(
        "  has attn_output_bias: {}",
        layer0.attn_output_bias.is_some()
    );
    eprintln!("  has ffn_up_bias: {}", layer0.ffn_up_bias.is_some());
    eprintln!("  has ffn_down_bias: {}", layer0.ffn_down_bias.is_some());
    eprintln!("  has qkv_bias: {}", layer0.qkv_bias.is_some());

    // Print ffn_up weight info
    eprintln!("\n=== Layer 0 FFN Up Weight ===");
    eprintln!("  in_dim: {}", layer0.ffn_up_weight.in_dim);
    eprintln!("  out_dim: {}", layer0.ffn_up_weight.out_dim);
    eprintln!("  qtype: {}", layer0.ffn_up_weight.qtype);
    eprintln!("  data len: {} bytes", layer0.ffn_up_weight.data.len());

    // Print first few f32 values of ffn_up weight
    let ffn_up_f32: Vec<f32> = layer0.ffn_up_weight.data[..20]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!("  first 5 f32 values: {:?}", ffn_up_f32);

    // Print Q weight info
    match &layer0.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            eprintln!("\n=== Layer 0 Q Weight (Separate) ===");
            eprintln!(
                "  q: in_dim={}, out_dim={}, qtype={}",
                q.in_dim, q.out_dim, q.qtype
            );
            eprintln!(
                "  k: in_dim={}, out_dim={}, qtype={}",
                k.in_dim, k.out_dim, k.qtype
            );
            eprintln!(
                "  v: in_dim={}, out_dim={}, qtype={}",
                v.in_dim, v.out_dim, v.qtype
            );
            let q_f32: Vec<f32> = q.data[..20]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            eprintln!("  q first 5 f32: {:?}", q_f32);
        },
        realizar::gguf::OwnedQKVWeights::Fused(t) => {
            eprintln!("\n=== Layer 0 QKV Weight (Fused) ===");
            eprintln!(
                "  in_dim={}, out_dim={}, qtype={}",
                t.in_dim, t.out_dim, t.qtype
            );
        },
    }

    // Run forward pass
    eprintln!("\n=== Running Forward Pass ===");
    eprintln!("  prompt tokens: {:?}", PROMPT_TOKENS);
    let logits = model.forward(PROMPT_TOKENS).expect("Forward pass failed");

    eprintln!("\n=== Logits Analysis ===");
    eprintln!(
        "  logits len: {} (expected vocab_size={})",
        logits.len(),
        model.config().vocab_size
    );

    // Find top-5 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("  Top 10 tokens:");
    for (i, (token_id, logit)) in indexed.iter().take(10).enumerate() {
        let marker = if *token_id as u32 == EXPECTED_FIRST_TOKEN {
            " ← EXPECTED"
        } else {
            ""
        };
        eprintln!(
            "    #{}: token={}, logit={:.4}{}",
            i + 1,
            token_id,
            logit,
            marker
        );
    }

    // Check if NaN/Inf in logits
    let nan_count = logits.iter().filter(|x| x.is_nan()).count();
    let inf_count = logits.iter().filter(|x| x.is_infinite()).count();
    let zero_count = logits.iter().filter(|x| **x == 0.0).count();
    eprintln!("\n  NaN count: {}", nan_count);
    eprintln!("  Inf count: {}", inf_count);
    eprintln!("  Zero count: {}", zero_count);
    eprintln!("  Logit[262] (expected top): {:.4}", logits[262]);

    // Print some logit statistics
    let sum: f32 = logits.iter().sum();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    eprintln!("  Logit sum: {:.4}", sum);
    eprintln!("  Logit max: {:.4}", max);
    eprintln!("  Logit min: {:.4}", min);

    // The actual assertion
    let predicted_token = indexed[0].0 as u32;
    eprintln!(
        "\n=== RESULT: predicted={}, expected={} ===",
        predicted_token, EXPECTED_FIRST_TOKEN
    );

    assert_eq!(
        predicted_token, EXPECTED_FIRST_TOKEN,
        "GPT-2 GGUF: expected first token {} but got {}",
        EXPECTED_FIRST_TOKEN, predicted_token
    );
}
