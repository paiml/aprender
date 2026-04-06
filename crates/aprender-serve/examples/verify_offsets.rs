use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let data =
        fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").expect("test");
    let model = GGUFModel::from_bytes(&data).expect("test");

    println!(
        "File size: {} bytes ({:.2} MB)",
        data.len(),
        data.len() as f64 / 1024.0 / 1024.0
    );
    println!("Tensor data start: {}", model.tensor_data_start);

    // Check a few key tensors
    let tensors_to_check = [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "output.weight",
    ];

    println!("\nKey tensors:");
    for name in tensors_to_check {
        if let Some(t) = model.tensors.iter().find(|t| t.name == name) {
            let num_elements: u64 = t.dims.iter().product();

            // Calculate expected byte size based on qtype
            let byte_size = match t.qtype {
                0 => num_elements * 4,                    // F32
                2 => (num_elements.div_ceil(32)) * 18,    // Q4_0
                12 => (num_elements.div_ceil(256)) * 144, // Q4_K
                _ => 0,
            };

            let abs_offset = model.tensor_data_start + t.offset as usize;
            let abs_end = abs_offset + byte_size as usize;

            println!("  {}:", name);
            println!("    dims: {:?}", t.dims);
            println!("    qtype: {}", t.qtype);
            println!("    offset (relative): {}", t.offset);
            println!("    offset (absolute): {}", abs_offset);
            println!("    byte_size: {}", byte_size);
            println!(
                "    end offset: {} (valid: {})",
                abs_end,
                abs_end <= data.len()
            );

            // Read first few bytes to verify we're reading correct location
            if abs_offset + 20 <= data.len() {
                let first_bytes = &data[abs_offset..abs_offset + 20];
                println!("    first 20 bytes: {:?}", first_bytes);
            }
        }
    }
}
