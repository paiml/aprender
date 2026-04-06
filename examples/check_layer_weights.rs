//! Check weight statistics per layer
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::dequantize_q4_k;

const Q4_K_BLOCK_SIZE: usize = 144;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    println!("FFN Gate/Up weight statistics per layer:\n");

    for layer_idx in 0..5 {
        let layer = &model.layers()[layer_idx];

        if let Some(ref gate) = layer.ffn_gate_weight {
            // Get first superblock (256 values)
            let first_block = &gate.data[..Q4_K_BLOCK_SIZE];
            let dequant = dequantize_q4_k(first_block).expect("test");
            let l2: f32 = dequant.iter().map(|x| x * x).sum::<f32>().sqrt();
            let min = dequant.iter().copied().fold(f32::INFINITY, f32::min);
            let max = dequant.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "Layer {} gate: L2={:.4}, min={:.4}, max={:.4}",
                layer_idx, l2, min, max
            );
        }

        let up = &layer.ffn_up_weight;
        let first_block = &up.data[..Q4_K_BLOCK_SIZE];
        let dequant = dequantize_q4_k(first_block).expect("test");
        let l2: f32 = dequant.iter().map(|x| x * x).sum::<f32>().sqrt();
        let min = dequant.iter().copied().fold(f32::INFINITY, f32::min);
        let max = dequant.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "Layer {} up:   L2={:.4}, min={:.4}, max={:.4}",
            layer_idx, l2, min, max
        );
        println!();
    }
}
