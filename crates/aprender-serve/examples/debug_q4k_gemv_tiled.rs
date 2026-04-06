//! Debug Q4K GEMV Tiled - test q4k_gemv_into_tiled via wrapper
//! This uses the exact same kernel code as transformer_layer_workspace_inner
use realizar::cuda::CudaExecutor;
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;

fn main() {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let model_path = "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("load");
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped).expect("cpu model");

    // Get layer 0 Q weight info
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

    println!("\nQ weight info:");
    println!("  in_dim: {}", q_weight.in_dim);
    println!("  out_dim: {}", q_weight.out_dim);
    println!("  data len: {} bytes", q_weight.data.len());
    println!(
        "  First 20 bytes: {:?}",
        &q_weight.data[..20.min(q_weight.data.len())]
    );

    // Create a simple test input (same as debug_q4k_gemv.rs)
    let hidden_dim = cpu_model.config().hidden_dim;
    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i % 10) as f32 - 5.0) * 0.1)
        .collect();
    println!("\nTest input[0..10]: {:?}", &input[..10]);

    // CPU Q4K GEMV
    let cpu_output =
        fused_q4k_parallel_matvec(&q_weight.data, &input, q_weight.in_dim, q_weight.out_dim)
            .expect("cpu q4k gemv");

    println!("\nCPU Q4K output[0..10]: {:?}", &cpu_output[..10]);

    // GPU Q4K GEMV using q4k_gemv_cached (TiledQ4KGemv - verified working)
    println!("\nCreating CUDA executor...");
    let mut executor = CudaExecutor::new(0).expect("cuda executor");

    // Upload Q weight to GPU
    let weight_name = "test_q_weight";
    executor
        .load_quantized_weights(weight_name, &q_weight.data)
        .expect("upload weight");

    // Test 1: Original q4k_gemv_cached (uses TiledQ4KGemv, verified working)
    let mut gpu_output_cached = vec![0.0f32; q_weight.out_dim];
    executor
        .q4k_gemv_cached(
            weight_name,
            &input,
            &mut gpu_output_cached,
            q_weight.out_dim as u32,
            q_weight.in_dim as u32,
        )
        .expect("gpu q4k gemv cached");

    println!(
        "\nGPU Q4K (cached) output[0..10]: {:?}",
        &gpu_output_cached[..10]
    );

    // Test 2: q4k_gemv_cached_tiled (uses q4k_gemv_into_tiled, same as workspace path)
    let mut gpu_output_tiled = vec![0.0f32; q_weight.out_dim];
    executor
        .q4k_gemv_cached_tiled(
            weight_name,
            &input,
            &mut gpu_output_tiled,
            q_weight.out_dim as u32,
            q_weight.in_dim as u32,
        )
        .expect("gpu q4k gemv tiled");

    println!(
        "GPU Q4K (tiled) output[0..10]: {:?}",
        &gpu_output_tiled[..10]
    );

    // Compare all three
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

    println!("\n=== GPU Cached vs GPU Tiled ===");
    for i in 0..5 {
        let diff = gpu_output_cached[i] - gpu_output_tiled[i];
        println!(
            "  [{}]: Cached={:8.4}, Tiled={:8.4}, diff={:8.4}",
            i, gpu_output_cached[i], gpu_output_tiled[i], diff
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
    let max_diff_gpu = gpu_output_cached
        .iter()
        .zip(gpu_output_tiled.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    println!("\nMax diff CPU vs Cached: {:.6}", max_diff_cached);
    println!("Max diff CPU vs Tiled: {:.6}", max_diff_tiled);
    println!("Max diff Cached vs Tiled: {:.6}", max_diff_gpu);

    if max_diff_cached > 1.0 || max_diff_tiled > 1.0 {
        println!("\n[ERROR] Significant mismatch detected!");
    } else if max_diff_gpu > 0.0001 {
        println!("\n[WARNING] GPU kernels produce different results!");
    } else {
        println!("\n[OK] All kernels match!");
    }
}
