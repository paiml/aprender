//! Check raw norm weights from GGUF
use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let data =
        fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").expect("test");
    let model = GGUFModel::from_bytes(&data).expect("test");

    // Find norm weight tensors
    let norm_tensors: Vec<_> = model
        .tensors
        .iter()
        .filter(|t| t.name.contains("norm"))
        .collect();

    println!("Norm tensors in GGUF:");
    for t in &norm_tensors {
        println!("  {} - dims: {:?}, qtype: {}", t.name, t.dims, t.qtype);
    }

    // Check blk.0.attn_norm.weight specifically
    if let Some(t) = model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_norm.weight")
    {
        println!("\nblk.0.attn_norm.weight details:");
        println!("  dims: {:?}", t.dims);
        println!("  qtype: {} (0=F32, 1=F16, ...)", t.qtype);
        println!("  offset: {}", t.offset);

        // Read raw bytes
        let abs_offset = model.tensor_data_start + t.offset as usize;
        let num_elements: usize = t.dims.iter().map(|&d| d as usize).product();
        let byte_size = if t.qtype == 0 {
            num_elements * 4 // F32
        } else if t.qtype == 1 {
            num_elements * 2 // F16
        } else {
            println!("  Unknown qtype!");
            return;
        };

        println!("  num_elements: {}", num_elements);
        println!("  byte_size: {}", byte_size);

        if t.qtype == 0 {
            // F32 - read directly
            let raw_bytes = &data[abs_offset..abs_offset + byte_size];
            let values: Vec<f32> = raw_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            println!("\n  First 10 values: {:?}", &values[..10.min(values.len())]);

            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
            println!("  Stats: min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
        } else if t.qtype == 1 {
            // F16
            let raw_bytes = &data[abs_offset..abs_offset + byte_size];
            let values: Vec<f32> = raw_bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();

            println!("\n  First 10 values: {:?}", &values[..10.min(values.len())]);

            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
            println!("  Stats: min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
        }
    }

    // Also check output_norm.weight
    if let Some(t) = model
        .tensors
        .iter()
        .find(|t| t.name == "output_norm.weight")
    {
        println!("\noutput_norm.weight details:");
        println!("  dims: {:?}", t.dims);
        println!("  qtype: {}", t.qtype);

        // Get values
        let values = model
            .get_tensor_f32("output_norm.weight", &data)
            .expect("test");
        println!("  First 10 values: {:?}", &values[..10.min(values.len())]);

        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
        println!("  Stats: min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
    }
}
