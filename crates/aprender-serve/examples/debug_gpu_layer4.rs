//! Debug GPU layer 4 divergence
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string());
    let mapped = MappedGGUFModel::from_path(&path).expect("Failed to load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to parse");

    let hidden_dim = model.config().hidden_dim;
    let token_id = 791u32;

    // Get embedding
    let start = token_id as usize * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!(
        "Token {} embedding first 5: {:?}",
        token_id,
        &embedding[..5]
    );

    // Run CPU forward for layers 0-4
    let cpu_hidden = embedding;
    for layer_idx in 0..=4 {
        let _before = cpu_hidden[0..3].to_vec();

        // Simple CPU forward for this layer
        let layer = &model.layers()[layer_idx];

        // RMSNorm
        let eps = model.config().eps;
        let rms = (cpu_hidden.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32 + eps).sqrt();
        let normed: Vec<f32> = cpu_hidden
            .iter()
            .zip(layer.attn_norm_weight.iter())
            .map(|(x, w)| (x / rms) * w)
            .collect();

        // Q projection only to check divergence
        let realizar::gguf::OwnedQKVWeights::Separate { q: q_weight, .. } = &layer.qkv_weight
        else {
            panic!("Expected separate QKV")
        };

        let q = realizar::quantize::fused_q4k_parallel_matvec(
            &q_weight.data,
            &normed,
            q_weight.in_dim,
            q_weight.out_dim,
        )
        .expect("q4k");

        println!("Layer {} Q first 5: {:?}", layer_idx, &q[..5]);

        // Do full layer forward (simplified - just for debugging)
        // Note: This is a simplified forward, real code is more complex
    }
}
