//! PAR-001g: Check Q6_K superblock layout in V weight
//!
//! The d value at offset 208 is tiny (0.000009). Let's check multiple
//! superblocks to verify the layout is correct.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::dequantize_q6_k;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001g: Q6_K Superblock Analysis ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let layer = &model.layers()[0];
    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { v, .. } => {
            println!(
                "V weight: in_dim={}, out_dim={}, qtype={}",
                v.in_dim, v.out_dim, v.qtype
            );
            println!("V data.len()={}\n", v.data.len());

            let superblocks_per_row = v.in_dim.div_ceil(256);
            let bytes_per_row = superblocks_per_row * 210;

            println!("Superblocks per row: {}", superblocks_per_row);
            println!("Bytes per row: {}\n", bytes_per_row);

            // Check first row (8 superblocks)
            println!("Row 0 superblock analysis:");
            for sb in 0..superblocks_per_row.min(8) {
                let sb_offset = sb * 210;
                let d_offset = sb_offset + 208;

                if d_offset + 2 <= v.data.len() {
                    let d_bytes = [v.data[d_offset], v.data[d_offset + 1]];
                    let d = half::f16::from_bits(u16::from_le_bytes(d_bytes)).to_f32();

                    // Also read the first scale value (at offset 192)
                    let sc0 = v.data[sb_offset + 192] as i8;

                    // Sample ql values
                    let ql0 = v.data[sb_offset];
                    let ql_mid = v.data[sb_offset + 64];

                    println!("  SB{}: d={:12.6e} at byte {}, sc[0]={:4}, ql[0]=0x{:02x}, ql[64]=0x{:02x}",
                             sb, d, d_offset, sc0, ql0, ql_mid);
                }
            }

            // Dequantize first superblock only (256 elements)
            println!("\nDequantize first superblock (bytes 0..210):");
            let sb0_data = &v.data[0..210];
            match dequantize_q6_k(sb0_data) {
                Ok(sb0_dequant) => {
                    let l2 = (sb0_dequant.iter().map(|x| x * x).sum::<f32>()).sqrt();
                    let max = sb0_dequant
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let min = sb0_dequant.iter().copied().fold(f32::INFINITY, f32::min);
                    let mean = sb0_dequant.iter().sum::<f32>() / sb0_dequant.len() as f32;
                    let nonzero = sb0_dequant.iter().filter(|&&x| x.abs() > 1e-6).count();

                    println!(
                        "  len={}, L2={:.6}, range=[{:.6}, {:.6}]",
                        sb0_dequant.len(),
                        l2,
                        min,
                        max
                    );
                    println!(
                        "  mean={:.6}, nonzero={}/{}",
                        mean,
                        nonzero,
                        sb0_dequant.len()
                    );
                    println!("  first 10: {:?}", &sb0_dequant[..10]);
                },
                Err(e) => println!("  Error: {:?}", e),
            }

            // What if we interpret the superblock with d at offset 0 instead?
            println!("\n=== Alternative: What if d is at offset 0? ===");
            // Q6_K layout in llama.cpp is: ql[128] + qh[64] + scales[16] + d[2]
            // But what if the GGUF uses a different layout?
            // Let's try: d[2] + ql[128] + qh[64] + scales[16]
            let d_alt_bytes = [v.data[0], v.data[1]];
            let d_alt = half::f16::from_bits(u16::from_le_bytes(d_alt_bytes)).to_f32();
            println!("d at offset 0: {:.6e} (bytes: {:02x?})", d_alt, d_alt_bytes);

            // Check if any 2-byte sequence in the first superblock looks like a reasonable scale
            println!(
                "\nScanning for reasonable f16 scale values (0.0001 to 1.0) in first 32 bytes:"
            );
            for i in 0..16 {
                let bytes = [v.data[i * 2], v.data[i * 2 + 1]];
                let val = half::f16::from_bits(u16::from_le_bytes(bytes)).to_f32();
                if val.is_finite() && val.abs() >= 0.0001 && val.abs() <= 1.0 {
                    println!("  offset {}: {:.6} (bytes: {:02x?})", i * 2, val, bytes);
                }
            }

            // Check around offset 208
            println!("\nBytes around offset 208:");
            for off in [200, 202, 204, 206, 208] {
                let bytes = [v.data[off], v.data[off + 1]];
                let val = half::f16::from_bits(u16::from_le_bytes(bytes)).to_f32();
                println!("  offset {}: {:12.6e} (bytes: {:02x?})", off, val, bytes);
            }

            // Let's look at all 210 bytes of superblock 0 to understand the pattern
            println!("\nSuperblock 0 layout (210 bytes):");
            println!("  ql[0..32]:     {:02x?}", &v.data[0..32]);
            println!("  ql[64..96]:    {:02x?}", &v.data[64..96]);
            println!("  qh[128..144]:  {:02x?}", &v.data[128..144]);
            println!("  scales[192..208]: {:02x?}", &v.data[192..208]);
            println!("  d[208..210]:   {:02x?}", &v.data[208..210]);
        },
        _ => println!("QKV is fused"),
    }

    println!("\n=== Analysis complete ===");
}
