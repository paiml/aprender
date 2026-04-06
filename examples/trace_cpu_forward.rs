//! Trace CPU forward step by step to find divergence

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;
    let test_token = 791u32;

    println!("=== CPU Forward Trace ===");
    println!("hidden_dim = {}", hidden_dim);
    println!("eps = {}", eps);
    println!("token = {}", test_token);

    // Step 1: Embedding
    let embedding = model.embed(&[test_token]);
    println!(
        "\n1. Embedding (first 5): [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
        embedding[0], embedding[1], embedding[2], embedding[3], embedding[4]
    );
    let emb_sum: f32 = embedding.iter().sum();
    println!("   sum = {:.6}", emb_sum);

    // Step 2: Manual RMSNorm
    let gamma = &model.layers()[0].attn_norm_weight;
    let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let normed_manual: Vec<f32> = embedding
        .iter()
        .zip(gamma.iter())
        .map(|(x, g)| x * inv_rms * g)
        .collect();
    println!(
        "\n2. Manual RMSNorm (first 5): [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
        normed_manual[0], normed_manual[1], normed_manual[2], normed_manual[3], normed_manual[4]
    );
    let norm_sum: f32 = normed_manual.iter().sum();
    println!("   sum = {:.6}, inv_rms = {:.8}", norm_sum, inv_rms);

    // Step 3: Direct Q4K GEMV (standalone function)
    let OwnedQKVWeights::Separate { q: q_weight, .. } = &model.layers()[0].qkv_weight else {
        panic!("Expected separate QKV")
    };
    println!("\n3. Q weight info:");
    println!(
        "   in_dim = {}, out_dim = {}, qtype = {}",
        q_weight.in_dim, q_weight.out_dim, q_weight.qtype
    );
    println!("   data len = {} bytes", q_weight.data.len());
    println!("   first 16 bytes: {:?}", &q_weight.data[..16]);

    let q_standalone = fused_q4k_parallel_matvec(
        &q_weight.data,
        &normed_manual,
        q_weight.in_dim,
        q_weight.out_dim,
    )?;
    println!(
        "\n4. Standalone Q4K GEMV (first 5): [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
        q_standalone[0], q_standalone[1], q_standalone[2], q_standalone[3], q_standalone[4]
    );
    let q_sum: f32 = q_standalone.iter().sum();
    println!("   sum = {:.6}", q_sum);

    // Step 5: Run model.forward() with CPU_DEBUG_LAYERS to see what it produces
    println!("\n5. Running model.forward() with debug...");
    std::env::set_var("CPU_DEBUG_LAYERS", "1");
    let _logits = model.forward(&[test_token])?;

    // The debug output should appear above, showing what forward() produces
    println!("\nCompare the [CPU-L0] Q (before RoPE) output above with step 4.");
    println!("If they differ, the bug is in how model.forward() calls fused_matmul.");

    Ok(())
}
