//! Check what's special about token 74403 in GPU output

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let head_dim = hidden_dim / cpu_model.config().num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let vocab_size = cpu_model.config().vocab_size;

    let test_token: u32 = 791;

    // CPU
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;

    // GPU
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    // Check token 74403
    let token = 74403;
    eprintln!("=== Token {} Analysis ===", token);
    eprintln!("CPU logit: {:.4}", cpu_logits[token]);
    eprintln!("GPU logit: {:.4}", gpu_logits[token]);

    // Find CPU and GPU argmax
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();

    eprintln!(
        "\nCPU argmax: token {} with logit {:.4}",
        cpu_argmax.0, cpu_argmax.1
    );
    eprintln!(
        "GPU argmax: token {} with logit {:.4}",
        gpu_argmax.0, gpu_argmax.1
    );

    // Check logits around token 74403
    eprintln!("\n=== Logits around token 74403 ===");
    for t in [74400, 74401, 74402, 74403, 74404, 74405] {
        if t < vocab_size {
            eprintln!(
                "  Token {}: CPU={:.4}, GPU={:.4}, diff={:.4}",
                t,
                cpu_logits[t],
                gpu_logits[t],
                gpu_logits[t] - cpu_logits[t]
            );
        }
    }

    // Check Q6K super-block boundaries near token 74403
    // Q6K: 256 values per super-block, 1260 bytes per row
    // Token 74403 is row 74403 in the LM head matrix
    let sb_per_row = hidden_dim.div_ceil(256); // = 6
    let bytes_per_row = sb_per_row * 210; // Q6K bytes
    let row_offset = 74403 * bytes_per_row;
    eprintln!("\n=== Q6K Layout Analysis ===");
    eprintln!("Super-blocks per row: {}", sb_per_row);
    eprintln!("Bytes per row: {}", bytes_per_row);
    eprintln!("Row 74403 offset: {} bytes", row_offset);

    Ok(())
}
