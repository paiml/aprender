//! Debug Q4_K embedding raw bytes

use realizar::gguf::MappedGGUFModel;
use realizar::quantize::dequantize_q4_k;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let data = mapped.data();
    let model = &mapped.model;

    println!("=== Q4_K Embedding Raw Bytes ===\n");

    // Find token_embd.weight tensor
    let tensor = model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("test");
    println!("Tensor: {}", tensor.name);
    println!("  dims: {:?}", tensor.dims);
    println!("  qtype: {} (12=Q4_K)", tensor.qtype);
    println!("  offset: {}", tensor.offset);

    let hidden_dim = tensor.dims[1] as usize; // 2048
    let vocab_size = tensor.dims[0] as usize; // 32000
    println!("  vocab_size: {}, hidden_dim: {}", vocab_size, hidden_dim);

    // Calculate tensor data start
    let tensor_offset = model.tensor_data_start + tensor.offset as usize;
    println!("  data_start: {}", tensor_offset);

    // Q4_K: 144 bytes per 256 values
    let super_blocks_per_row = hidden_dim.div_ceil(256); // 8
    let bytes_per_row = super_blocks_per_row * 144; // 1152
    println!(
        "  super_blocks_per_row: {}, bytes_per_row: {}",
        super_blocks_per_row, bytes_per_row
    );

    // Token 450's row data
    let token_id = 450usize;
    let row_start = tensor_offset + token_id * bytes_per_row;
    let row_end = row_start + bytes_per_row;
    println!("\nToken {} row data:", token_id);
    println!("  byte range: [{}..{}]", row_start, row_end);

    // Dequantize just this row
    let row_data = &data[row_start..row_end];
    println!("  row_data.len: {}", row_data.len());

    // First super-block header (d, dmin, scales, etc.)
    let d = f16_to_f32(&row_data[0..2]);
    let dmin = f16_to_f32(&row_data[2..4]);
    println!("\n  First super-block:");
    println!("    d (f16): {:.8}", d);
    println!("    dmin (f16): {:.8}", dmin);
    println!("    scales[0..4]: {:?}", &row_data[4..8]);

    // Dequantize the row
    let row_dequant = dequantize_q4_k(row_data).expect("Failed to dequantize");
    println!("\n  Dequantized row:");
    println!("    len: {}", row_dequant.len());
    println!("    L2: {:.4}", l2_norm(&row_dequant));
    println!(
        "    first 20: {:?}",
        &row_dequant[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Compare with full tensor dequantization
    // (This takes the full tensor and extracts row 450)
    let full_tensor_bytes = (vocab_size * hidden_dim).div_ceil(256) * 144;
    let full_data = &data[tensor_offset..tensor_offset + full_tensor_bytes];
    let full_dequant = dequantize_q4_k(full_data).expect("Failed");
    let full_row_start = token_id * hidden_dim;
    let full_row: Vec<f32> = full_dequant[full_row_start..full_row_start + hidden_dim].to_vec();

    println!("\nFull tensor row 450:");
    println!("  L2: {:.4}", l2_norm(&full_row));
    println!(
        "  first 20: {:?}",
        &full_row[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Are they the same?
    let diff_l2: f32 = row_dequant
        .iter()
        .zip(full_row.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("\nL2 of difference: {:.6}", diff_l2);

    // Compare with OwnedQuantizedModel's token_embedding
    let owned_model = realizar::gguf::OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let owned_start = token_id * hidden_dim;
    let owned_row: Vec<f32> =
        owned_model.token_embedding()[owned_start..owned_start + hidden_dim].to_vec();
    println!("\nOwnedQuantizedModel token_embedding row 450:");
    println!("  L2: {:.4}", l2_norm(&owned_row));
    println!(
        "  first 20: {:?}",
        &owned_row[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    let diff_owned: f32 = row_dequant
        .iter()
        .take(hidden_dim)
        .zip(owned_row.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("L2 of difference (row vs owned): {:.6}", diff_owned);
}

fn f16_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}
