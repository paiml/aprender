//! CORRECTNESS-002: Step-by-step Layer 0 CPU vs GPU comparison
//!
//! Traces layer 0 to find exact divergence point
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_layer0_stepwise

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CORRECTNESS-002: Layer 0 step-by-step trace\n");

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;

    let test_token: u32 = 791;
    eprintln!("Token: {}, hidden_dim: {}", test_token, hidden_dim);

    // CPU: Get embedding (token_embedding is already f32)
    let embedding_offset = (test_token as usize) * hidden_dim;
    let cpu_embedding: Vec<f32> =
        model.token_embedding()[embedding_offset..embedding_offset + hidden_dim].to_vec();

    let embed_sum: f32 = cpu_embedding.iter().sum();
    eprintln!(
        "[CPU] Embedding sum={:.6}, first 5: {:?}",
        embed_sum,
        &cpu_embedding[..5]
    );

    // CPU: RMSNorm on embedding
    let layer = &model.layers()[0];
    let norm_weight = &layer.attn_norm_weight; // Already f32

    let eps = model.config().eps;
    let sum_sq: f32 = cpu_embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

    let cpu_normed: Vec<f32> = cpu_embedding
        .iter()
        .zip(norm_weight.iter())
        .map(|(h, w)| h / rms * w)
        .collect();

    let normed_sum: f32 = cpu_normed.iter().sum();
    eprintln!(
        "[CPU] RMSNorm: eps={}, rms={:.6}, sum={:.6}, first 5: {:?}",
        eps,
        rms,
        normed_sum,
        &cpu_normed[..5]
    );

    // Now run GPU and compare
    #[cfg(feature = "cuda")]
    {
        use realizar::gguf::OwnedQuantizedModelCuda;

        eprintln!("\n=== GPU Comparison ===");

        // Create GPU model
        let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
        let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
        let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
        cuda_model.preload_weights_gpu()?;

        // Get GPU embedding via the model() accessor
        let gpu_embedding: Vec<f32> = cuda_model.model().token_embedding
            [embedding_offset..embedding_offset + hidden_dim]
            .to_vec();

        let gpu_embed_sum: f32 = gpu_embedding.iter().sum();
        eprintln!(
            "[GPU] Embedding sum={:.6}, first 5: {:?}",
            gpu_embed_sum,
            &gpu_embedding[..5]
        );

        // Check embedding match
        let embed_match = cpu_embedding
            .iter()
            .zip(gpu_embedding.iter())
            .all(|(c, g)| (c - g).abs() < 1e-6);
        eprintln!("Embedding match: {}", embed_match);

        // Get GPU norm weight
        let gpu_layer = &cuda_model.model().layers()[0];
        let gpu_norm_weight = &gpu_layer.attn_norm_weight;

        let norm_weight_match = norm_weight
            .iter()
            .zip(gpu_norm_weight.iter())
            .all(|(c, g)| (c - g).abs() < 1e-6);
        eprintln!("Norm weight match: {}", norm_weight_match);

        // Manual CPU RMSNorm using GPU's embedding (to isolate norm kernel)
        let gpu_sum_sq: f32 = gpu_embedding.iter().map(|x| x * x).sum();
        let gpu_rms = (gpu_sum_sq / hidden_dim as f32 + eps).sqrt();
        let cpu_computed_normed: Vec<f32> = gpu_embedding
            .iter()
            .zip(gpu_norm_weight.iter())
            .map(|(h, w)| h / gpu_rms * w)
            .collect();

        let cpu_computed_sum: f32 = cpu_computed_normed.iter().sum();
        eprintln!(
            "[CPU computed from GPU embedding] RMSNorm: rms={:.6}, sum={:.6}, first 5: {:?}",
            gpu_rms,
            cpu_computed_sum,
            &cpu_computed_normed[..5]
        );

        // Now the GPU forward should start from the same embedding
        // The debug output from forward_gpu_resident shows:
        // [PAR-058-L0] RMSNorm OK, first 3: [-0.9617413, 0.3485948, 0.40064743]
        // Let's check if this matches CPU:
        eprintln!("\nExpected GPU RMSNorm output (from forward debug): [-0.9617413, 0.3485948, 0.40064743]");
        eprintln!(
            "CPU computed RMSNorm first 3: [{:.7}, {:.7}, {:.7}]",
            cpu_computed_normed[0], cpu_computed_normed[1], cpu_computed_normed[2]
        );

        // Check if CPU computed matches GPU reported
        let gpu_reported = [-0.9617413f32, 0.3485948, 0.40064743];
        let cpu_first3 = [
            cpu_computed_normed[0],
            cpu_computed_normed[1],
            cpu_computed_normed[2],
        ];
        let rmsnorm_match = gpu_reported
            .iter()
            .zip(cpu_first3.iter())
            .all(|(g, c)| (g - c).abs() < 0.001);
        eprintln!("RMSNorm match: {}", rmsnorm_match);

        if !rmsnorm_match {
            eprintln!("\n!!! RMSNorm diverges !!!");
            eprintln!("  GPU reported: {:?}", gpu_reported);
            eprintln!("  CPU computed: {:?}", cpu_first3);
            eprintln!(
                "  Diff: [{:.6}, {:.6}, {:.6}]",
                gpu_reported[0] - cpu_first3[0],
                gpu_reported[1] - cpu_first3[1],
                gpu_reported[2] - cpu_first3[2]
            );

            // Debug: check individual components
            eprintln!("\nDebugging RMSNorm:");
            eprintln!("  hidden[0] = {:.6}", gpu_embedding[0]);
            eprintln!("  norm_weight[0] = {:.6}", gpu_norm_weight[0]);
            eprintln!("  rms = {:.6}", gpu_rms);
            eprintln!(
                "  expected output[0] = {:.6} / {:.6} * {:.6} = {:.6}",
                gpu_embedding[0],
                gpu_rms,
                gpu_norm_weight[0],
                gpu_embedding[0] / gpu_rms * gpu_norm_weight[0]
            );
        }

        // Now let's also compare L0 layer output
        // CPU debug shows: After layer 0: sum=22.795757, hidden[0..4]=[-1.0401434, 0.1749062, -0.01589083, 0.31208688]
        // GPU debug shows: [PAR-058-L0] Layer output OK, first 3: [-0.75882584, -0.019156456, -0.3514105]
        eprintln!("\n=== Layer 0 Output Comparison ===");
        eprintln!("CPU layer 0 output: [-1.0401434, 0.1749062, -0.01589083, 0.31208688]");
        eprintln!("GPU layer 0 output: [-0.75882584, -0.019156456, -0.3514105]");
        eprintln!("DIVERGENT!");
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CUDA not enabled");
    }

    Ok(())
}
