//! Compare CPU vs GPU layer outputs
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;
use realizar::rms_norm;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string());
    let mapped = MappedGGUFModel::from_path(&path).expect("Failed to load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to parse");

    let hidden_dim = model.config().hidden_dim;
    let eps = model.config().eps;

    // Token 791 embedding
    let token_id = 791u32;
    let start = token_id as usize * hidden_dim;
    let mut cpu_hidden: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    // GPU outputs from the debug log
    let gpu_layer_outputs = [
        (0, [-1.0179534f32, 0.19496298, 0.04026422]),
        (1, [-4.5706506, -0.66239583, 2.1656928]),
        (2, [-9.529866, 8.064635, 1.6177181]),
        (3, [-9.905468, 2.823577, -3.177155]),
        (4, [-9.626493, 1.5636191, -4.7433143]),
        (5, [-9.668855, 1.7540426, -4.677788]),
        (6, [-9.617183, 1.7619439, -4.3580623]),
        (7, [-9.680186, 1.8482988, -4.239305]),
        (8, [-9.554973, 1.7573862, -4.6568527]),
        (9, [-9.616547, 1.9435388, -4.4903913]),
    ];

    println!("CPU layer-by-layer processing (simplified):");
    for layer_idx in 0..10 {
        let layer = &model.layers()[layer_idx];

        // RMSNorm -> Q projection (simplified check)
        let normed = rms_norm(&cpu_hidden, &layer.attn_norm_weight, eps);

        // Q weight
        let OwnedQKVWeights::Separate { q: q_weight, .. } = &layer.qkv_weight else {
            panic!("Expected separate QKV")
        };
        let q =
            fused_q4k_parallel_matvec(&q_weight.data, &normed, q_weight.in_dim, q_weight.out_dim)
                .expect("q4k");

        // For layer output comparison, just run a simpler forward
        // (Full layer forward is complex - let's just compare first few values)

        // Get GPU values for this layer
        if let Some(&(_, _gpu_vals)) = gpu_layer_outputs.iter().find(|(i, _)| *i == layer_idx) {
            // We need to run full layer to get output
            // For now just print Q projection comparison
            println!(
                "Layer {} Q first 3: CPU={:?}",
                layer_idx,
                &q[..3.min(q.len())]
            );
        }

        // Simplified: just accumulate hidden (won't match exactly without full layer)
        // This is just to verify dimensions
        cpu_hidden = vec![0.0f32; hidden_dim]; // Reset for next iter
    }

    // The real test: run full CPU forward and compare with GPU output
    println!("\nRunning full CPU forward...");
    let cpu_logits = model.forward(&[token_id]).expect("CPU forward failed");
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!("CPU logits argmax: {}", cpu_argmax);

    // Check GPU output
    println!("GPU generated token: 74403");

    // Compare logit distributions
    println!("\nLogit comparison at token 16 vs 74403:");
    println!("CPU logit[16] = {:.4}", cpu_logits[16]);
    println!("CPU logit[74403] = {:.4}", cpu_logits[74403]);
}
