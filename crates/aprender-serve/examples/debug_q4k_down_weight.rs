//! Debug Q4_K FFN down weight - compare dequantized values with HuggingFace

#![allow(clippy::needless_range_loop)]

use realizar::gguf::MappedGGUFModel;
use realizar::quantize::dequantize_q4_k;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");

    // Get layer 2 FFN down weight
    let tensor = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "blk.2.ffn_down.weight")
        .expect("No tensor");

    // Get data via mapped.data() and tensor offset
    let all_data = mapped.data();
    let absolute_offset = mapped.model.tensor_data_start + tensor.offset as usize;

    println!("=== Q4_K FFN Down Weight (Layer 2) ===\n");
    println!("Dims: {:?}", tensor.dims);
    println!("Tensor offset: {}", tensor.offset);
    println!("Absolute offset: {}", absolute_offset);
    println!("qtype: {}", tensor.qtype);

    // GGUF stores [out_dim, in_dim] after reversal = [2048, 5632]
    let out_dim = tensor.dims[0] as usize; // 2048
    let in_dim = tensor.dims[1] as usize; // 5632

    println!("out_dim (hidden): {}", out_dim);
    println!("in_dim (intermediate): {}", in_dim);

    // Q4_K: 144 bytes per superblock, 256 elements
    let num_elements = out_dim * in_dim;
    let num_superblocks = num_elements.div_ceil(256);
    let expected_bytes = num_superblocks * 144;

    println!("Expected superblocks: {}", num_superblocks);
    println!("Expected data bytes: {}", expected_bytes);

    let data = &all_data[absolute_offset..absolute_offset + expected_bytes];
    println!("Data length: {} bytes", data.len());

    // Dequantize the weight
    let dequant = dequantize_q4_k(data).expect("Failed to dequantize");
    println!(
        "Dequantized length: {} (expected {})",
        dequant.len(),
        num_elements
    );

    // Compute L2 norm
    let l2: f32 = dequant.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Full weight L2: {:.4} (HF: 59.2032)", l2);

    // HF reference for row 0 first 10:
    // [0.00011110305786132812, 0.0067138671875, -0.00433349609375,
    //  0.0042724609375, 0.003997802734375, -0.003570556640625,
    //  0.010986328125, -0.0021820068359375, -0.007354736328125, 0.01397705078125]

    println!("\n=== HF row 0 first 10 (reference) ===");
    let hf_row0 = [
        0.000_111_103_06_f32,
        0.006_713_867,
        -0.004_333_496,
        0.004_272_461,
        0.003_997_802_7,
        -0.003_570_556_6,
        0.010_986_328,
        -0.002_182_006_8,
        -0.007_354_736_3,
        0.013_977_051,
    ];
    for (i, v) in hf_row0.iter().enumerate() {
        println!("  HF[0,{}] = {:.6}", i, v);
    }

    // Q4_K layout: dequant contains out_dim * in_dim values
    // For row-major [out_dim, in_dim]: element at [i, j] is at index i * in_dim + j
    // So row 0 is dequant[0..in_dim]

    println!("\n=== Our row 0 first 10 (row-major layout) ===");
    for j in 0..10 {
        println!("  Our[0,{}] = {:.6}", j, dequant[j]);
    }

    // Check row 0 L2
    let row0_l2: f32 = (0..in_dim)
        .map(|j| {
            let v = dequant[j];
            v * v
        })
        .sum::<f32>()
        .sqrt();
    println!("\nRow 0 L2: {:.4} (HF: 1.2970)", row0_l2);

    // Compare element by element
    println!("\n=== Element-wise comparison ===");
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for (i, &hf_val) in hf_row0.iter().enumerate() {
        let our_val = dequant[i];
        let diff = (our_val - hf_val).abs();
        println!(
            "  [0,{}]: Ours={:10.6} HF={:10.6} diff={:.6}",
            i, our_val, hf_val, diff
        );
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    println!(
        "\nMax difference: {:.6} at index {}",
        max_diff, max_diff_idx
    );

    // Q4_K superblock info
    println!("\n=== Q4_K superblock structure ===");
    let superblocks = data.len() / 144;
    println!("Total superblocks: {}", superblocks);
    println!("Expected for [2048, 5632]: {}", num_superblocks);

    // Check first superblock (256 values)
    println!("\nFirst 256 dequant values summary:");
    let first256_l2: f32 = dequant[0..256].iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("  L2: {:.4}", first256_l2);
    println!(
        "  Min: {:.6}",
        dequant[0..256]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Max: {:.6}",
        dequant[0..256]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    );
}
