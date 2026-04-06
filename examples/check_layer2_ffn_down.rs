use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::dequantize_q4_k;

const Q4_K_BLOCK_SIZE: usize = 144;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let layer = &model.layers()[2];
    let down = &layer.ffn_down_weight;

    println!(
        "Layer 2 FFN down: in_dim={}, out_dim={}, qtype={}",
        down.in_dim, down.out_dim, down.qtype
    );

    // For Q4_K row-major: each row has 5632/256 = 22 superblocks = 22*144 = 3168 bytes
    let bytes_per_row = (down.in_dim / 256) * Q4_K_BLOCK_SIZE;
    let superblock_idx = 5475 / 256; // = 21
    let pos_in_block = 5475 % 256; // = 99

    println!(
        "Column 5475 is in superblock {}, position {}",
        superblock_idx, pos_in_block
    );

    // Get column 5475 values for first 10 rows
    println!("\nColumn 5475 values (first 10 rows):");
    for row in 0..10 {
        let row_start = row * bytes_per_row;
        let block_start = row_start + superblock_idx * Q4_K_BLOCK_SIZE;
        let block_data = &down.data[block_start..block_start + Q4_K_BLOCK_SIZE];
        let dequant = dequantize_q4_k(block_data).expect("test");
        println!("  row {}: col_5475={:.6}", row, dequant[pos_in_block]);
    }

    // Compare with column 100 for reference
    println!("\nColumn 100 values (first 10 rows):");
    for row in 0..10 {
        let row_start = row * bytes_per_row;
        let block_start = row_start; // superblock 0
        let block_data = &down.data[block_start..block_start + Q4_K_BLOCK_SIZE];
        let dequant = dequantize_q4_k(block_data).expect("test");
        println!("  row {}: col_100={:.6}", row, dequant[100]);
    }

    // Calculate full column L2 norms
    let mut col_5475_l2 = 0.0f32;
    let mut col_100_l2 = 0.0f32;
    for row in 0..down.out_dim {
        let row_start = row * bytes_per_row;

        let block_start = row_start + superblock_idx * Q4_K_BLOCK_SIZE;
        let block_data = &down.data[block_start..block_start + Q4_K_BLOCK_SIZE];
        let dequant = dequantize_q4_k(block_data).expect("test");
        col_5475_l2 += dequant[pos_in_block] * dequant[pos_in_block];

        let block_start = row_start; // superblock 0
        let block_data = &down.data[block_start..block_start + Q4_K_BLOCK_SIZE];
        let dequant = dequantize_q4_k(block_data).expect("test");
        col_100_l2 += dequant[100] * dequant[100];
    }

    println!("\nColumn L2 norms:");
    println!("  col_5475 L2: {:.6}", col_5475_l2.sqrt());
    println!("  col_100 L2:  {:.6}", col_100_l2.sqrt());
}
