//! Check QKV tensor structure in GGUF files
use realizar::gguf::MappedGGUFModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Qwen2-0.5B QKV Tensor Structure ===\n");

    let qwen_path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let qwen_mapped = MappedGGUFModel::from_path(qwen_path)?;

    println!("Qwen2-0.5B Layer 0 attention tensors:");
    for tensor in &qwen_mapped.model.tensors {
        if tensor.name.contains("blk.0")
            && (tensor.name.contains("attn") || tensor.name.contains("ffn"))
        {
            println!(
                "  {}: dims={:?}, qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );
        }
    }

    // Check bias tensors specifically
    println!("\nQwen2-0.5B ALL bias tensors (layer 0):");
    for tensor in &qwen_mapped.model.tensors {
        if tensor.name.contains("blk.0") && tensor.name.contains("bias") {
            let data = qwen_mapped
                .model
                .get_tensor_f32(&tensor.name, qwen_mapped.data())?;
            let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            println!(
                "  {}: dims={:?}, len={}, norm={:.4}, range=[{:.4}, {:.4}]",
                tensor.name,
                tensor.dims,
                data.len(),
                norm,
                min,
                max
            );
        }
    }

    // Now check TinyLlama for comparison
    println!("\n=== TinyLlama QKV Tensor Structure ===\n");

    let tiny_path = "/tmp/tinyllama.gguf";
    let tiny_mapped = MappedGGUFModel::from_path(tiny_path)?;

    println!("TinyLlama Layer 0 attention tensors:");
    for tensor in &tiny_mapped.model.tensors {
        if tensor.name.contains("blk.0")
            && (tensor.name.contains("attn") || tensor.name.contains("ffn"))
        {
            println!(
                "  {}: dims={:?}, qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );
        }
    }

    // Key dimensions check
    println!("\n=== Key Observations ===\n");

    // Check if Qwen2 uses fused QKV or separate
    let has_fused_qkv = qwen_mapped
        .model
        .tensors
        .iter()
        .any(|t| t.name.contains("attn_qkv"));
    let has_separate_qkv = qwen_mapped
        .model
        .tensors
        .iter()
        .any(|t| t.name.contains("attn_q.weight"));
    println!("Qwen2 uses fused QKV: {}", has_fused_qkv);
    println!("Qwen2 uses separate Q/K/V: {}", has_separate_qkv);

    let tiny_has_fused = tiny_mapped
        .model
        .tensors
        .iter()
        .any(|t| t.name.contains("attn_qkv"));
    let tiny_has_separate = tiny_mapped
        .model
        .tensors
        .iter()
        .any(|t| t.name.contains("attn_q.weight"));
    println!("TinyLlama uses fused QKV: {}", tiny_has_fused);
    println!("TinyLlama uses separate Q/K/V: {}", tiny_has_separate);

    // Print actual tensor names for Qwen2 QKV
    println!("\nQwen2 attention weight tensors (all layers):");
    for tensor in &qwen_mapped.model.tensors {
        if tensor.name.ends_with(".weight")
            && tensor.name.contains("attn")
            && !tensor.name.contains("norm")
        {
            println!("  {}: dims={:?}", tensor.name, tensor.dims);
        }
    }

    Ok(())
}
