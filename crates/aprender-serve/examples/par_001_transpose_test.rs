//! PAR-001k: Test transposed weight access
//!
//! Maybe the V weights are stored column-major (transposed) compared to
//! our row-major reading. Let's test if transposed access produces better alignment.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::dequantize_q6_k;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001k: Transposed Weight Access Test ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let layer = &model.layers()[0];

    // Get real input
    let token_id: u32 = 26222;
    let hidden = model.embed(&[token_id]);
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + model.config().eps).sqrt();
    let normed: Vec<f32> = hidden
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| x / rms * w)
        .collect();
    let normed_l2 = l2_norm(&normed);

    println!("Input normed L2: {:.4}\n", normed_l2);

    match &layer.qkv_weight {
        realizar::gguf::OwnedQKVWeights::Separate { v, .. } => {
            println!("V: in_dim={}, out_dim={}", v.in_dim, v.out_dim);

            // Current interpretation:
            // - 256 output rows
            // - Each row has 2048 elements (8 superblocks × 256)
            // - bytes_per_row = 8 × 210 = 1680

            // Alternative interpretation (transposed):
            // - 2048 output columns
            // - Each column has 256 elements (1 superblock)
            // - bytes_per_column = 210

            let superblock_size = 210;
            let num_superblocks = v.data.len() / superblock_size;
            println!(
                "Total data: {} bytes = {} superblocks",
                v.data.len(),
                num_superblocks
            );

            println!("\n=== Current interpretation (row-major) ===");
            println!("256 rows × 8 superblocks per row");

            // Compute row 0 dot product
            let row0_data = &v.data[0..8 * superblock_size];
            let row0_weights = dequantize_q6_k(row0_data).expect("row0 dequant failed");
            let row0_dot: f32 = row0_weights
                .iter()
                .zip(normed.iter())
                .map(|(w, x)| w * x)
                .sum();
            println!(
                "Row 0: {} elements, L2={:.4}, dot={:.6}",
                row0_weights.len(),
                l2_norm(&row0_weights),
                row0_dot
            );

            println!("\n=== Alternative interpretation (column-major) ===");
            println!("2048 columns × 1 superblock per column, then transpose");

            // In this interpretation:
            // - Superblock i contains 256 consecutive elements of "column i"
            // - But wait, we have 2048 superblocks, not 2048 columns
            // - Actually, we have 256 rows × 8 superblocks = 2048 superblocks
            // - Column-major would be: for each of 2048 columns, store 256 elements
            // - That's 2048 superblocks of 256 elements each = 524288 elements

            // Let's try: interpret each superblock as a column of 256 elements
            // The first 256 superblocks would give us a 256×256 submatrix
            // Then we can compute V @ input[0:256] for the first 256 inputs

            // Actually, let's try a simpler test: what if rows and columns are swapped?
            // Instead of output[o] = sum(V[o, i] * input[i])
            // We compute output[o] = sum(V[i, o] * input[i])

            // With column-major storage, V[i, o] would be at superblock (o * num_in_superblocks + i // 256)
            // This is getting complex. Let's try a different approach.

            // Hypothesis: The V weights are stored such that superblock k corresponds to:
            // - Row interpretation: row (k // 8), elements (k % 8) * 256 to (k % 8 + 1) * 256
            // - Column interpretation: column k, elements 0 to 255

            // Let's dequantize the first 256 superblocks (256 × 256 = 65536 values)
            // and try both interpretations
            println!("\nDequantizing first 256 superblocks (65536 values)...");
            let first_256_sb = &v.data[0..256 * superblock_size];
            let first_256_values = dequantize_q6_k(first_256_sb).expect("dequant failed");
            println!("Got {} values", first_256_values.len());

            // Row-major: rows 0-31, 8 superblocks each
            // first_256_values[i*256 + j] = V[i // 8][i % 8 * 256 + j]? No wait...
            // Each superblock has 256 values, so first_256_values is superblocks 0-255 concatenated
            // Superblock k has values at first_256_values[k*256..(k+1)*256]

            // Current (row-major): superblock k is part of row (k // 8), specifically elements (k % 8)*256 .. (k%8+1)*256
            // So V[row, col] = first_256_values[(row * 8 + col // 256) * 256 + col % 256]
            //                = first_256_values[row * 2048 + col]  (which is just linear indexing of 256 rows × 2048 cols)

            // Wait, that's for the first 32 rows only (32 × 8 = 256 superblocks)

            // Let me try computing the output using column-major interpretation for first 32 rows
            println!("\nColumn-major interpretation test:");
            println!("Treating each superblock as a column of 256 values");
            println!("So first_256_values[sb*256 + elem] = V[elem, sb]");

            // With this interpretation, the output for row r would be:
            // output[r] = sum over col: V[r, col] * input[col]
            // But V[r, col] is in superblock col at position r
            // = first_256_values[col * 256 + r]

            // This only works for first 256 columns (256 superblocks)
            // Let's compute dot product for row 0 with first 256 columns only:
            let mut transposed_dot = 0.0f32;
            for col in 0..256.min(v.in_dim) {
                // V[0, col] = superblock col, element 0
                // = first_256_values[col * 256 + 0] = first_256_values[col * 256]
                transposed_dot += first_256_values[col * 256] * normed[col];
            }
            println!(
                "Row 0 dot (transposed, first 256 cols): {:.6}",
                transposed_dot
            );

            // Compare to normal row 0 dot with first 256 elements:
            let normal_first256_dot: f32 = row0_weights[..256]
                .iter()
                .zip(normed[..256].iter())
                .map(|(w, x)| w * x)
                .sum();
            println!(
                "Row 0 dot (normal, first 256 cols): {:.6}",
                normal_first256_dot
            );

            // Also compute full transposed output for row 0 using all superblocks
            println!("\nFull transposed computation for row 0:");
            // We need 2048 columns, so we need superblocks 0-2047 (but we only have 256 per row × 256 rows = 2048 total)
            // Superblock k is: if row-major, row = k//8, sublock_in_row = k%8
            //                  if col-major, col = k, row is in the 256 elements of the superblock

            // For full transposed: V[0, col] = superblock col, element 0
            // But we only have 2048 superblocks, so col goes 0-2047
            // However, that would give us 2048 columns × 256 rows = 524288 elements
            // Which matches the V weight dimensions!

            // Let's try full transposed access
            let all_values = dequantize_q6_k(&v.data).expect("full dequant failed");
            println!("Dequantized {} values total", all_values.len());

            // Transposed: V[row, col] = all_values[col * 256 + row]
            // Output[row] = sum over col: V[row, col] * input[col]
            //             = sum over col: all_values[col * 256 + row] * input[col]

            let mut transposed_output = vec![0.0f32; 256];
            for row in 0..256 {
                for col in 0..2048 {
                    transposed_output[row] += all_values[col * 256 + row] * normed[col];
                }
            }
            let transposed_l2 = l2_norm(&transposed_output);
            let transposed_nonzero = transposed_output
                .iter()
                .filter(|&&x| x.abs() > 0.01)
                .count();
            println!(
                "\nTransposed V output: L2={:.4}, non-zero(>0.01)={}/{}",
                transposed_l2, transposed_nonzero, 256
            );
            println!("First 5: {:?}", &transposed_output[..5]);

            // Compare to normal (row-major) output
            use realizar::quantize::fused_q6k_parallel_matvec;
            let normal_output = fused_q6k_parallel_matvec(&v.data, &normed, v.in_dim, v.out_dim)
                .expect("fused matvec failed");
            let normal_l2 = l2_norm(&normal_output);
            let normal_nonzero = normal_output.iter().filter(|&&x| x.abs() > 0.01).count();
            println!(
                "\nNormal V output: L2={:.4}, non-zero(>0.01)={}/{}",
                normal_l2, normal_nonzero, 256
            );
            println!("First 5: {:?}", &normal_output[..5]);
        },
        _ => println!("QKV is fused"),
    }

    println!("\n=== Analysis complete ===");
}
