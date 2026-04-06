//! Compare dequantized Q weight row with HuggingFace reference

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::dequantize_q4_k;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    // Get Q weight
    let OwnedQKVWeights::Separate { q: q_weight, .. } = &model.layers()[0].qkv_weight else {
        panic!("Expected separate")
    };

    println!("=== Q Weight Row Comparison ===\n");
    println!(
        "Q weight: in_dim={}, out_dim={}, qtype={}",
        q_weight.in_dim, q_weight.out_dim, q_weight.qtype
    );

    // Dequantize full weight
    let dequant = dequantize_q4_k(&q_weight.data).expect("Failed");
    println!(
        "Dequantized length: {} (expected {})",
        dequant.len(),
        q_weight.in_dim * q_weight.out_dim
    );

    // For row-major [out_dim, in_dim]: row r is at indices [r*in_dim .. (r+1)*in_dim]
    let in_dim = q_weight.in_dim;

    // Row 0: first output element's weights
    println!("\nRow 0 (out[0]'s weights):");
    println!("  First 20 elements: {:?}", &dequant[0..20]);

    // HF reference row 0 first 20 elements (will get from Python)
    // For now, print L2 of row 0
    let row0_l2: f32 = dequant[0..in_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("  Row 0 L2: {:.6}", row0_l2);

    // Check specific pattern: are values at [0] and [32] related?
    println!("\nValues at key indices in row 0:");
    for i in [0, 1, 31, 32, 33, 63, 64, 65] {
        println!("  dequant[{}] = {:.6}", i, dequant[i]);
    }

    // Check first superblock (256 values) of row 0
    println!("\nRow 0 first 256 values (first superblock):");
    println!(
        "  [0..32]   L2: {:.6}",
        dequant[0..32].iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    println!(
        "  [32..64]  L2: {:.6}",
        dequant[32..64].iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    println!(
        "  [64..96]  L2: {:.6}",
        dequant[64..96].iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    println!(
        "  [96..128] L2: {:.6}",
        dequant[96..128].iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Full weight L2
    let full_l2: f32 = dequant.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nFull weight L2: {:.4} (HF: 33.5851)", full_l2);
}
