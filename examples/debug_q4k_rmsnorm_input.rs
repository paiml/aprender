//! Debug Q4K GEMV with RMSNorm output as input
//! Tests if the kernel works with the actual RMSNorm values from forward path
use realizar::cuda::CudaExecutor;
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;

fn main() {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let model_path = "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("load");
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped).expect("cpu model");

    // Get layer 0 Q weight
    let layer0 = &cpu_model.layers()[0];
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: _,
        v: _,
        ..
    } = &layer0.qkv_weight
    else {
        panic!("Expected separate Q/K/V weights")
    };

    println!(
        "\nQ weight: in_dim={}, out_dim={}",
        q_weight.in_dim, q_weight.out_dim
    );
    println!(
        "Q weight data len: {}, first 20 bytes: {:?}",
        q_weight.data.len(),
        &q_weight.data[..20]
    );

    // Get the actual RMSNorm output from a forward pass
    // Token "Hello" = 9707
    let token_id = 9707u32;
    let embedding = cpu_model.embed(&[token_id]);

    // Apply RMSNorm to get the actual input
    let attn_norm = &layer0.attn_norm_weight;
    let eps = cpu_model.config().eps;

    // Compute RMSNorm on CPU
    let hidden_dim = cpu_model.config().hidden_dim;
    let mut rms_sum = 0.0f32;
    for &x in &embedding {
        rms_sum += x * x;
    }
    let rms = (rms_sum / hidden_dim as f32 + eps).sqrt();
    let rms_inv = 1.0 / rms;

    let normed: Vec<f32> = embedding
        .iter()
        .zip(attn_norm.iter())
        .map(|(&x, &g)| x * rms_inv * g)
        .collect();

    println!("\nRMSNorm input (normed)[0..8]: {:?}", &normed[..8]);
    println!("RMSNorm sum: {:.6}", normed.iter().sum::<f32>());

    // CPU Q4K GEMV with RMSNorm output
    let cpu_output =
        fused_q4k_parallel_matvec(&q_weight.data, &normed, q_weight.in_dim, q_weight.out_dim)
            .expect("cpu q4k gemv");

    println!("\nCPU Q4K output[0..8]: {:?}", &cpu_output[..8]);

    // GPU Q4K GEMV with RMSNorm output
    println!("\nCreating CUDA executor...");
    let mut executor = CudaExecutor::new(0).expect("cuda executor");

    let weight_name = "test_q_weight";
    executor
        .load_quantized_weights(weight_name, &q_weight.data)
        .expect("upload weight");

    // Test 1: q4k_gemv_cached (verified working)
    let mut gpu_output_cached = vec![0.0f32; q_weight.out_dim];
    executor
        .q4k_gemv_cached(
            weight_name,
            &normed,
            &mut gpu_output_cached,
            q_weight.out_dim as u32,
            q_weight.in_dim as u32,
        )
        .expect("gpu q4k gemv cached");

    println!(
        "\nGPU Q4K (cached) output[0..8]: {:?}",
        &gpu_output_cached[..8]
    );

    // Test 2: q4k_gemv_cached_tiled (same as workspace path)
    let mut gpu_output_tiled = vec![0.0f32; q_weight.out_dim];
    executor
        .q4k_gemv_cached_tiled(
            weight_name,
            &normed,
            &mut gpu_output_tiled,
            q_weight.out_dim as u32,
            q_weight.in_dim as u32,
        )
        .expect("gpu q4k gemv tiled");

    println!("GPU Q4K (tiled) output[0..8]: {:?}", &gpu_output_tiled[..8]);

    // Compare
    println!("\n=== CPU vs GPU Cached ===");
    for i in 0..5 {
        let diff = cpu_output[i] - gpu_output_cached[i];
        println!(
            "  [{}]: CPU={:8.4}, GPU={:8.4}, diff={:8.4}",
            i, cpu_output[i], gpu_output_cached[i], diff
        );
    }

    println!("\n=== CPU vs GPU Tiled ===");
    for i in 0..5 {
        let diff = cpu_output[i] - gpu_output_tiled[i];
        println!(
            "  [{}]: CPU={:8.4}, GPU={:8.4}, diff={:8.4}",
            i, cpu_output[i], gpu_output_tiled[i], diff
        );
    }

    let max_diff_cached = cpu_output
        .iter()
        .zip(gpu_output_cached.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let max_diff_tiled = cpu_output
        .iter()
        .zip(gpu_output_tiled.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    println!("\nMax diff CPU vs Cached: {:.6}", max_diff_cached);
    println!("Max diff CPU vs Tiled: {:.6}", max_diff_tiled);

    if max_diff_cached > 1.0 || max_diff_tiled > 1.0 {
        println!("\n[ERROR] Significant mismatch detected!");
    } else {
        println!("\n[OK] All kernels match!");
    }
}
