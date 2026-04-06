//! Verify Q8_0 dequantization for token embedding
use realizar::gguf::MappedGGUFModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;

    // Find token_embd.weight tensor info
    let tensor = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd.weight not found");

    println!("Token embedding tensor:");
    println!("  dims: {:?}", tensor.dims);
    println!("  qtype: {}", tensor.qtype);
    println!("  offset: {}", tensor.offset);

    let vocab_size = tensor.dims[0] as usize;
    let hidden_dim = tensor.dims[1] as usize;
    let num_elements = vocab_size * hidden_dim;

    println!("  vocab_size: {}", vocab_size);
    println!("  hidden_dim: {}", hidden_dim);
    println!("  num_elements: {}", num_elements);

    // Q8_0 block size: 34 bytes (2 f16 scale + 32 i8 quants)
    let num_blocks = num_elements.div_ceil(32);
    let expected_bytes = num_blocks * 34;

    println!("  num_blocks: {}", num_blocks);
    println!("  expected_bytes (34 per block): {}", expected_bytes);

    // Get actual tensor data size
    let _actual_start = mapped.model.tensor_data_start + tensor.offset as usize;

    // Try to dequantize the first few blocks
    let token_emb = mapped
        .model
        .get_tensor_f32("token_embd.weight", mapped.data())?;

    println!("\nDequantized token embedding:");
    println!("  length: {}", token_emb.len());
    println!("  expected: {}", num_elements);

    // Check first token (token 0 = "!") embedding
    let tok0_emb = &token_emb[0..hidden_dim];
    let tok0_norm: f32 = tok0_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let tok0_sum: f32 = tok0_emb.iter().sum();

    println!("\nToken 0 (\"!\") embedding:");
    println!("  first 8: {:?}", &tok0_emb[..8]);
    println!("  norm: {:.4}", tok0_norm);
    println!("  sum: {:.4}", tok0_sum);

    // Check BOS token (151643) embedding
    let bos = 151643usize;
    let bos_start = bos * hidden_dim;
    let bos_emb = &token_emb[bos_start..bos_start + hidden_dim];
    let bos_norm: f32 = bos_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    let bos_sum: f32 = bos_emb.iter().sum();

    println!("\nBOS token ({}) embedding:", bos);
    println!("  first 8: {:?}", &bos_emb[..8]);
    println!("  norm: {:.4}", bos_norm);
    println!("  sum: {:.4}", bos_sum);

    // Verify embedding is not all zeros
    let non_zero = tok0_emb.iter().filter(|&&x| x.abs() > 1e-10).count();
    println!(
        "\nToken 0: {} non-zero values out of {}",
        non_zero, hidden_dim
    );

    Ok(())
}
