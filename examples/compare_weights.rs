//! Verify Q weight data matches between CPU and GPU

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    let mapped = MappedGGUFModel::from_path(&path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== CPU Q weight data (layer 0) ===");
    let (q_data, q_in_dim, q_out_dim, q_qtype) = match &model.layers()[0].qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, .. } => {
            (&q.data, q.in_dim, q.out_dim, q.qtype)
        },
        _ => panic!("Expected separate QKV weights"),
    };

    let cpu_checksum: u64 = q_data.iter().map(|&b| b as u64).sum();
    let cpu_first_32: Vec<u8> = q_data.iter().take(32).copied().collect();

    println!(
        "  in_dim = {}, out_dim = {}, qtype = {}",
        q_in_dim, q_out_dim, q_qtype
    );
    println!("  data len = {} bytes", q_data.len());
    println!("  first 32 bytes = {:?}", cpu_first_32);
    println!("  full checksum = {}", cpu_checksum);

    // Now load to GPU and verify
    println!("\n=== GPU Q weight data (layer 0) ===");
    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;

    // Get GPU weight pointer info from executor
    // (we can't directly read GPU memory without sync, but the sizes should match)
    println!("  GPU weights preloaded successfully");
    println!("  If GPU kernel produces different results, the bug is in the kernel or params");

    // Run a simple forward to trigger any dimension issues
    println!("\n=== Testing forward ===");
    let token_id = 791u32;

    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let mut dummy_cache = realizar::gguf::OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        100,
    );

    let gpu_logits = cuda_model.forward_gpu_resident(token_id, &mut dummy_cache, 0)?;
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v));
    println!("GPU argmax: {:?}", gpu_argmax);

    let cpu_logits = model.forward(&[token_id])?;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v));
    println!("CPU argmax: {:?}", cpu_argmax);

    if cpu_argmax.map(|(i, _)| i) != gpu_argmax.map(|(i, _)| i) {
        println!("\nMISMATCH - Bug is in Q4K GEMV kernel or its parameters");
    } else {
        println!("\nMATCH - No bug detected");
    }

    Ok(())
}
