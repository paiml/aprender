//! Check weight data layout

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::RealizarError;

fn main() -> Result<(), RealizarError> {
    let model_path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let layer = &model.layers()[0];
    let weight = &layer.ffn_down_weight;

    println!("FFN down weight:");
    println!("  in_dim:  {}", weight.in_dim);
    println!("  out_dim: {}", weight.out_dim);
    println!("  qtype:   {}", weight.qtype);
    println!("  data.len(): {} bytes", weight.data.len());

    let expected_sb = (weight.out_dim * weight.in_dim).div_ceil(256);
    let expected_bytes = expected_sb * 144;
    println!("  Expected super-blocks: {}", expected_sb);
    println!("  Expected bytes: {}", expected_bytes);

    // Check first few super-blocks
    println!("\nFirst 3 super-blocks:");
    for sb in 0..3 {
        let offset = sb * 144;
        let d = read_f16(&weight.data[offset..offset + 2]);
        let dmin = read_f16(&weight.data[offset + 2..offset + 4]);
        println!(
            "  SB {}: offset={}, d={:.6}, dmin={:.6}",
            sb, offset, d, dmin
        );
    }

    // The weight layout might be different - check if it's row-major or super-block-major
    // In GGUF, quantized weights are typically stored as a contiguous array of super-blocks
    // covering (out_dim, in_dim) in row-major order

    // For Q4_K with out_dim=1536, in_dim=8960:
    // - Each row has ceil(8960/256) = 35 super-blocks
    // - Total super-blocks = 1536 * 35 = 53,760
    // - Total bytes = 53,760 * 144 = 7,741,440

    let rows = weight.out_dim;
    let cols = weight.in_dim;
    let sb_per_row = cols.div_ceil(256);
    println!("\nExpected layout (row-major):");
    println!("  Rows: {}", rows);
    println!("  Cols: {}", cols);
    println!("  Super-blocks per row: {}", sb_per_row);
    println!("  Total super-blocks: {}", rows * sb_per_row);
    println!("  Total bytes: {}", rows * sb_per_row * 144);

    // What if it's stored as (in_dim, out_dim) transposed?
    let sb_per_row_t = rows.div_ceil(256);
    println!("\nAlternative layout (transposed):");
    println!("  Rows: {}", cols);
    println!("  Cols: {}", rows);
    println!("  Super-blocks per row: {}", sb_per_row_t);
    println!("  Total super-blocks: {}", cols * sb_per_row_t);
    println!("  Total bytes: {}", cols * sb_per_row_t * 144);

    // Try calling the matmul with the original function
    println!("\nTesting original matmul...");
    let activations: Vec<f32> = vec![0.1; weight.in_dim];
    let mut output = vec![0.0f32; weight.out_dim];

    let result = realizar::quantize::fused_q4k_parallel_matvec_into(
        &weight.data,
        &activations,
        weight.in_dim,
        weight.out_dim,
        &mut output,
    );
    println!("Result: {:?}", result.is_ok());
    println!("First 5 outputs: {:?}", &output[..5]);
    println!("Any NaN: {}", output.iter().any(|x| x.is_nan()));
    println!("Any Inf: {}", output.iter().any(|x| x.is_infinite()));

    Ok(())
}

fn read_f16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}
