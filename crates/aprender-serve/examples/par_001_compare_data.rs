//! PAR-001f: Compare V weight data access patterns
//!
//! Verify that OwnedQuantizedTensor.data matches direct mapped data access

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001f: V Weight Data Comparison ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    // Get V weight from OwnedQuantizedModel
    let layer = &model.layers()[0];
    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
            println!("From OwnedQuantizedModel:");
            println!(
                "  V in_dim={}, out_dim={}, qtype={}",
                v.in_dim, v.out_dim, v.qtype
            );
            println!("  V data.len()={}", v.data.len());
            println!(
                "  V first 32 bytes: {:02x?}",
                &v.data[..32.min(v.data.len())]
            );

            // Read bytes 208-210 (d value position)
            if v.data.len() >= 210 {
                let d_bytes = [v.data[208], v.data[209]];
                let d = half::f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();
                println!("  V d at offset 208: {:.6} (bytes: {:02x?})", d, d_bytes);
            }

            // Also check Q for comparison
            println!(
                "\n  Q in_dim={}, out_dim={}, qtype={}",
                q.in_dim, q.out_dim, q.qtype
            );
            println!("  Q data.len()={}", q.data.len());
            println!(
                "  Q first 32 bytes: {:02x?}",
                &q.data[..32.min(q.data.len())]
            );

            // Q4_K d is at offset 0
            if q.data.len() >= 4 {
                let d_bytes = [q.data[0], q.data[1]];
                let d = half::f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();
                let dmin_bytes = [q.data[2], q.data[3]];
                let dmin = half::f16::from_bits(u16::from_le_bytes(dmin_bytes)).to_f32();
                println!(
                    "  Q d={:.6}, dmin={:.6} (bytes: {:02x?} {:02x?})",
                    d, dmin, d_bytes, dmin_bytes
                );
            }

            println!(
                "\n  K in_dim={}, out_dim={}, qtype={}",
                k.in_dim, k.out_dim, k.qtype
            );
            println!(
                "  K first 32 bytes: {:02x?}",
                &k.data[..32.min(k.data.len())]
            );
        },
        _ => println!("QKV is fused"),
    }

    // Now get V weight from direct mapped access
    println!("\n\nFrom MappedGGUFModel (direct):");
    if let Some(v_tensor) = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_v.weight")
    {
        let data = mapped.data();
        let absolute_offset = mapped.model.tensor_data_start + v_tensor.offset as usize;

        println!("  tensor offset: {}", v_tensor.offset);
        println!("  absolute offset: {}", absolute_offset);
        println!("  dims: {:?}", v_tensor.dims);
        println!("  qtype: {}", v_tensor.qtype);

        if absolute_offset + 210 <= data.len() {
            println!(
                "  Direct first 32 bytes: {:02x?}",
                &data[absolute_offset..absolute_offset + 32]
            );

            // Read d at offset 208
            let d_offset = absolute_offset + 208;
            let d_bytes = [data[d_offset], data[d_offset + 1]];
            let d = half::f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();
            println!(
                "  Direct d at offset 208: {:.6} (bytes: {:02x?})",
                d, d_bytes
            );
        }
    }

    // Check if maybe Q and V data got swapped somehow
    println!("\n\n=== Cross-check: Did Q and V get swapped? ===");
    if let Some(q_tensor) = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
    {
        let data = mapped.data();
        let q_offset = mapped.model.tensor_data_start + q_tensor.offset as usize;
        println!(
            "Q tensor at offset {}: first 32 = {:02x?}",
            q_tensor.offset,
            &data[q_offset..q_offset + 32]
        );
    }
    if let Some(v_tensor) = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_v.weight")
    {
        let data = mapped.data();
        let v_offset = mapped.model.tensor_data_start + v_tensor.offset as usize;
        println!(
            "V tensor at offset {}: first 32 = {:02x?}",
            v_tensor.offset,
            &data[v_offset..v_offset + 32]
        );
    }

    // Try interpreting V data as Q4_K to see if it makes more sense
    println!("\n\n=== Try interpreting V as Q4_K ===");
    if let realizar::gguf::OwnedQKVWeights::Separate { v, .. } = &model.layers()[0].qkv_weight {
        // If V was actually Q4_K, d would be at offset 0
        if v.data.len() >= 4 {
            let d_bytes = [v.data[0], v.data[1]];
            let d = half::f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();
            let dmin_bytes = [v.data[2], v.data[3]];
            let dmin = half::f16::from_bits(u16::from_le_bytes(dmin_bytes)).to_f32();
            println!("  If Q4_K: d={:.6}, dmin={:.6}", d, dmin);
        }

        // The V first 2 bytes are [28, eb], which as f16 would be:
        let d = half::f16::from_bits(u16::from_le_bytes([0x28, 0xeb])).to_f32();
        println!("  V first 2 bytes as f16: {:.6}", d);
    }

    println!("\n=== Analysis complete ===");
}
