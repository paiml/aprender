//! Check embedding dequantization  
use realizar::gguf::GGUFModel;
use realizar::quantize::dequantize_q4_0;
use std::fs;

fn stats(x: &[f32]) -> (f32, f32, f32, f32) {
    let min = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
    let std: f32 = (x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32).sqrt();
    (min, max, mean, std)
}

fn main() {
    let data =
        fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").expect("test");
    let model = GGUFModel::from_bytes(&data).expect("test");

    let t = model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("test");
    println!("token_embd.weight:");
    println!("  dims (after reverse): {:?}", t.dims);
    println!("  qtype: {} (2=Q4_0)", t.qtype);
    println!("  offset: {}", t.offset);

    let hidden_dim = 2048;
    let vocab_size = 32000;
    let num_elements = hidden_dim * vocab_size;
    println!("  expected elements: {}", num_elements);

    // Read the raw quantized data
    let abs_offset = model.tensor_data_start + t.offset as usize;
    let num_blocks = num_elements / 32; // Q4_0: 32 elements per block
    let byte_size = num_blocks * 18; // Q4_0: 18 bytes per block
    println!("  num_blocks: {}, byte_size: {}", num_blocks, byte_size);

    // Dequantize the full tensor
    let quant_data = &data[abs_offset..abs_offset + byte_size];
    let values = dequantize_q4_0(quant_data).expect("test");
    println!("  dequantized elements: {}", values.len());

    // Check multiple tokens
    for tok_id in [0, 1, 2, 10, 100, 500, 1000, 5000, 10000, 31999] {
        let start = tok_id * hidden_dim;
        let end = start + hidden_dim;
        if end <= values.len() {
            let tok_embed = &values[start..end];
            let (min, max, mean, std) = stats(tok_embed);
            println!(
                "  token {}: min={:.6}, max={:.6}, mean={:.6}, std={:.6}",
                tok_id, min, max, mean, std
            );
            if std < 0.0001 {
                println!("           SUSPICIOUSLY LOW STD!");
            }
        }
    }

    // Check raw blocks for token 1 (BOS)
    // Token 1 starts at element 2048, which is block 2048/32 = 64
    println!("\nChecking raw Q4_0 blocks for token 1:");
    let block_idx = hidden_dim / 32;
    let num_blocks_per_tok = hidden_dim / 32;
    println!(
        "  starts at block {}, {} blocks per token",
        block_idx, num_blocks_per_tok
    );

    for i in 0..3 {
        let b = block_idx + i;
        let block_start = b * 18;
        let scale_bytes = &quant_data[block_start..block_start + 2];
        let scale = half::f16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();
        let quants = &quant_data[block_start + 2..block_start + 18];
        println!(
            "    block {}: scale={:.6}, first 4 quant bytes: {:02x} {:02x} {:02x} {:02x}",
            b, scale, quants[0], quants[1], quants[2], quants[3]
        );
    }

    // Check raw blocks for token 100
    println!("\nChecking raw Q4_0 blocks for token 100:");
    let block_idx = 100 * hidden_dim / 32;
    for i in 0..3 {
        let b = block_idx + i;
        let block_start = b * 18;
        let scale_bytes = &quant_data[block_start..block_start + 2];
        let scale = half::f16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();
        let quants = &quant_data[block_start + 2..block_start + 18];
        println!(
            "    block {}: scale={:.6}, first 4 quant bytes: {:02x} {:02x} {:02x} {:02x}",
            b, scale, quants[0], quants[1], quants[2], quants[3]
        );
    }
}
