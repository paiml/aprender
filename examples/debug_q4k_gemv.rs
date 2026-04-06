//! Debug Q4K GEMV - compare CPU vs GPU for single Q weight
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
    println!("  qtype: {} (Q4_K = 12)", q_weight.qtype);
    println!(
        "  First 20 bytes: {:?}",
        &q_weight.data[..20.min(q_weight.data.len())]
    );

    // Create a simple test input
    let hidden_dim = cpu_model.config().hidden_dim;
    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i % 10) as f32 - 5.0) * 0.1)
        .collect();
    println!("\nTest input[0..10]: {:?}", &input[..10]);
    println!("Input sum: {:.6}", input.iter().sum::<f32>());

    // CPU Q4K GEMV
    let cpu_output =
        fused_q4k_parallel_matvec(&q_weight.data, &input, q_weight.in_dim, q_weight.out_dim)
            .expect("cpu q4k gemv");

    println!("\nCPU Q4K output[0..10]: {:?}", &cpu_output[..10]);
    println!("CPU sum: {:.6}", cpu_output.iter().sum::<f32>());

    // GPU Q4K GEMV
    println!("\nCreating CUDA executor...");
    let mut executor = CudaExecutor::new(0).expect("cuda executor");

    // Upload Q weight to GPU
    let weight_name = "test_q_weight";
    executor
        .load_quantized_weights(weight_name, &q_weight.data)
        .expect("upload weight");

    // Run GPU Q4K GEMV
    let mut gpu_output = vec![0.0f32; q_weight.out_dim];
    executor
        .q4k_gemv_cached(
            weight_name,
            &input,
            &mut gpu_output,
            q_weight.out_dim as u32,
            q_weight.in_dim as u32,
        )
        .expect("gpu q4k gemv");

    println!("\nGPU Q4K output[0..10]: {:?}", &gpu_output[..10]);
    println!("GPU sum: {:.6}", gpu_output.iter().sum::<f32>());

    // Compare
    println!("\n=== Comparison ===");
    for i in 0..10 {
        let diff = cpu_output[i] - gpu_output[i];
        println!(
            "  [{}]: CPU={:8.4}, GPU={:8.4}, diff={:8.4}",
            i, cpu_output[i], gpu_output[i], diff
        );
    }

    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f32 = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(c, g)| (c - g).abs())
        .sum::<f32>()
        / cpu_output.len() as f32;

    println!("\nMax absolute diff: {:.6}", max_diff);
    println!("Mean absolute diff: {:.6}", mean_diff);

    if max_diff > 1.0 {
        println!("\n❌ Q4K GEMV mismatch detected - GPU kernel may be wrong!");
    } else if max_diff > 0.01 {
        println!("\n⚠️ Q4K GEMV has small numerical differences (expected with quantization)");
    } else {
        println!("\n✅ Q4K GEMV matches!");
    }
}
