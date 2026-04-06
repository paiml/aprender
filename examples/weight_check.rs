//! Check if CPU and GPU use same Q weight data

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    let mapped = MappedGGUFModel::from_path(&path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Get layer 0 Q weight
    let layer0 = &model.layers()[0];
    let (q_data, q_in_dim, q_out_dim, q_qtype) = match &layer0.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, .. } => {
            (&q.data, q.in_dim, q.out_dim, q.qtype)
        },
        _ => panic!("Expected separate QKV weights"),
    };

    // Compute checksum of first 1024 bytes
    let checksum: u64 = q_data.iter().take(1024).map(|&b| b as u64).sum();
    let first_8: Vec<u8> = q_data.iter().take(8).copied().collect();

    println!("Layer 0 Q weight:");
    println!("  in_dim = {}", q_in_dim);
    println!("  out_dim = {}", q_out_dim);
    println!("  qtype = {} (Q4_K=12, Q5_K=13, Q6_K=14)", q_qtype);
    println!("  data len = {} bytes", q_data.len());
    println!("  first 8 bytes = {:?}", first_8);
    println!("  checksum (first 1024 bytes) = {}", checksum);

    Ok(())
}
