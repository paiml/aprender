//! Detailed trace of forward pass
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let bos = 151643u32;

    println!("=== Forward Pass Trace ===\n");
    println!("Config:");
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!("  eps: {:.1e}", model.config().eps);
    println!("  rope_type: {}", model.config().rope_type);

    // 1. Check initial embedding
    let emb_start = bos as usize * hidden_dim;
    let emb = &model.token_embedding()[emb_start..emb_start + hidden_dim];

    let emb_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let emb_sum: f32 = emb.iter().sum();
    let emb_max: f32 = emb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let emb_min: f32 = emb.iter().copied().fold(f32::INFINITY, f32::min);

    println!("\n1. Initial Embedding (BOS token):");
    println!("   norm: {:.4}", emb_norm);
    println!("   sum: {:.4}", emb_sum);
    println!("   max: {:.4}, min: {:.4}", emb_max, emb_min);
    println!("   first 8: {:?}", &emb[..8]);

    // Check token 0 embedding
    let tok0_emb = &model.token_embedding()[0..hidden_dim];
    let tok0_norm: f32 = tok0_emb.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("\n2. Token 0 (\"!\") Embedding:");
    println!("   norm: {:.4}", tok0_norm);
    println!("   first 8: {:?}", &tok0_emb[..8]);

    // Cosine similarity between BOS and token 0
    let dot: f32 = emb.iter().zip(tok0_emb.iter()).map(|(a, b)| a * b).sum();
    let cos_sim = dot / (emb_norm * tok0_norm);
    println!("\n   Cosine sim (BOS, token_0): {:.4}", cos_sim);

    // 3. Run forward
    let logits = model.forward(&[bos])?;

    println!("\n3. Logits Analysis:");
    println!("   Token 0 logit: {:.4}", logits[0]);
    println!("   Token 19 (\"4\") logit: {:.4}", logits[19]);

    // Find top 5
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("\n   Top 5 predictions:");
    for (tok, logit) in indexed.iter().take(5) {
        println!("     Token {}: {:.4}", tok, logit);
    }

    // 4. Compute expected logit for token 0 if hidden ≈ BOS embedding (after forward)
    // If the model did no computation (identity), then:
    // logit[0] = hidden · token_emb[0] = BOS_emb · token_0_emb = dot product

    let expected_identity_logit = dot; // This is BOS · token_0
    println!("\n4. If forward were identity (hidden = BOS_emb):");
    println!("   Expected logit[0] = {:.4}", expected_identity_logit);
    println!("   Actual logit[0] = {:.4}", logits[0]);
    println!(
        "   Ratio actual/expected = {:.2}x",
        logits[0] / expected_identity_logit
    );

    // The actual logit is much higher than expected for identity
    // This means the hidden state has been amplified or rotated towards token 0

    // 5. Estimate hidden state norm from logits
    // If logit[0] = hidden · tok0_emb, and hidden is aligned with tok0_emb:
    // ||hidden|| = logit[0] / ||tok0_emb||
    let estimated_hidden_norm = logits[0] / tok0_norm;
    println!("\n5. Estimated hidden state (if aligned with token 0):");
    println!("   ||hidden|| ≈ {:.4}", estimated_hidden_norm);
    println!("   Original BOS emb norm: {:.4}", emb_norm);
    println!("   Amplification: {:.2}x", estimated_hidden_norm / emb_norm);

    // 6. Check LM head info
    println!("\n6. LM Head Weight:");
    println!("   qtype: {}", model.lm_head_weight().qtype);
    println!(
        "   in_dim: {} (should be hidden_dim={})",
        model.lm_head_weight().in_dim,
        hidden_dim
    );
    println!(
        "   out_dim: {} (should be vocab_size)",
        model.lm_head_weight().out_dim
    );

    Ok(())
}
