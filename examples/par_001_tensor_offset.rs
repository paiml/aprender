//! PAR-001e: Verify tensor offset and byte extraction for V weight
//!
//! The V scale values are suspiciously small (0.000009). This suggests
//! we might be reading from the wrong offset in the GGUF file.

use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001e: Tensor Offset Verification ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");

    // Find Q, K, V tensor info
    let tensor_names = [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_up.weight",
    ];

    println!("Tensor data starts at: {}", mapped.model.tensor_data_start);
    println!();

    for name in tensor_names {
        if let Some(tensor) = mapped.model.tensors.iter().find(|t| t.name == name) {
            let absolute_offset = mapped.model.tensor_data_start + tensor.offset as usize;
            let qtype_name = match tensor.qtype {
                12 => "Q4_K",
                14 => "Q6_K",
                _ => "other",
            };

            println!("{}", name);
            println!("  dims: {:?}", tensor.dims);
            println!("  qtype: {} ({})", tensor.qtype, qtype_name);
            println!("  relative offset: {}", tensor.offset);
            println!("  absolute offset: {}", absolute_offset);

            // Calculate expected byte size
            let num_elements: usize = tensor.dims.iter().map(|&d| d as usize).product();
            let bytes_per_superblock = if tensor.qtype == 12 { 144 } else { 210 };
            let expected_bytes = (num_elements / 256) * bytes_per_superblock;
            println!("  num_elements: {}", num_elements);
            println!(
                "  expected bytes: {} (for {} superblocks)",
                expected_bytes,
                num_elements / 256
            );

            // Read first few bytes at the tensor's absolute offset
            let data = mapped.data();
            if absolute_offset + 32 <= data.len() {
                println!("  first 32 bytes at offset:");
                println!("    {:02x?}", &data[absolute_offset..absolute_offset + 32]);

                // If Q6_K, try reading d from offset 208
                if tensor.qtype == 14 && absolute_offset + 210 <= data.len() {
                    let d_offset = absolute_offset + 208;
                    let d_bytes = [data[d_offset], data[d_offset + 1]];
                    let d = half::f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();
                    println!("  Q6_K d at offset 208: {:.6} (bytes: {:02x?})", d, d_bytes);
                }

                // If Q4_K, try reading d from offset 0
                if tensor.qtype == 12 && absolute_offset + 4 <= data.len() {
                    let d_bytes = [data[absolute_offset], data[absolute_offset + 1]];
                    let d = half::f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();
                    let dmin_bytes = [data[absolute_offset + 2], data[absolute_offset + 3]];
                    let dmin = half::f16::from_bits(u16::from_le_bytes(dmin_bytes)).to_f32();
                    println!(
                        "  Q4_K d={:.6}, dmin={:.6} (bytes: {:02x?} {:02x?})",
                        d, dmin, d_bytes, dmin_bytes
                    );
                }
            }
            println!();
        }
    }

    // Also check if tensors are adjacent (no gaps or overlaps)
    println!("=== Tensor Layout Analysis ===\n");
    let mut sorted_tensors: Vec<_> = mapped.model.tensors.iter().collect();
    sorted_tensors.sort_by_key(|t| t.offset);

    let mut prev_end = 0u64;
    for tensor in sorted_tensors.iter().take(10) {
        let num_elements: usize = tensor.dims.iter().map(|&d| d as usize).product();
        let bytes_per_superblock = match tensor.qtype {
            12 => 144,
            14 => 210,
            _ => 0,
        };
        let tensor_bytes = if bytes_per_superblock > 0 {
            (num_elements / 256) * bytes_per_superblock
        } else {
            num_elements * 4 // F32
        };

        let gap = tensor.offset as i64 - prev_end as i64;
        println!(
            "{:40} offset={:10} size={:10} gap={:+6}",
            tensor.name, tensor.offset, tensor_bytes, gap
        );
        prev_end = tensor.offset + tensor_bytes as u64;
    }

    println!("\n=== Analysis complete ===");
}
