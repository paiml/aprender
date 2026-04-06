//! Debug Q6_K layout for V weight

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::dequantize_q6_k;

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let layer = &model.layers()[0];
    let OwnedQKVWeights::Separate {
        q: _,
        k: _,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate")
    };

    // V weight: in_dim=2048, out_dim=256
    // Total elements: 2048 * 256 = 524,288
    // Q6_K superblocks: 524,288 / 256 = 2048 superblocks
    // Each superblock: 210 bytes, 256 values

    println!(
        "Weight shape: [in_dim={}, out_dim={}]",
        v_weight.in_dim, v_weight.out_dim
    );
    println!("Total elements: {}", v_weight.in_dim * v_weight.out_dim);
    println!(
        "Num superblocks: {}",
        (v_weight.in_dim * v_weight.out_dim) / 256
    );

    // Dequantize first superblock (first 256 values)
    let sb0 = dequantize_q6_k(&v_weight.data[..210]).expect("test");
    println!("\nSuperblock 0 (first 256 values):");
    println!("  Values 0..5: {:?}", &sb0[..5]);
    println!("  Values 251..256: {:?}", &sb0[251..256]);

    // Dequantize second superblock (values 256..512)
    let sb1 = dequantize_q6_k(&v_weight.data[210..420]).expect("test");
    println!("\nSuperblock 1 (values 256..512):");
    println!("  Values 0..5: {:?}", &sb1[..5]);

    // Now dequantize all and check what the layout is
    let num_blocks = 2048;
    let mut all_vals = Vec::new();
    for i in 0..num_blocks {
        let block = dequantize_q6_k(&v_weight.data[i * 210..(i + 1) * 210]).expect("test");
        all_vals.extend(block);
    }

    // HuggingFace V weight is [256, 2048] in row-major
    // So W[0, :] is the first row (2048 values)
    // And W[:, 0] is the first column (256 values)

    // Let's check: is the dequantized data stored as:
    // Option A: [256 rows, 2048 cols] row-major => first 2048 values = W[0, :]
    // Option B: [2048 cols, 256 rows] col-major => first 256 values = W[:, 0]

    // HuggingFace W[0, :5] = [0.0281, 0.0059, -0.0003, -0.0056, 0.0075]
    // HuggingFace W[:5, 0] = [0.0281, 0.0176, 0.0359, 0.0165, -0.0222]

    println!("\nChecking layout interpretation:");
    println!("If row-major [256, 2048], W[0, :5] = {:?}", &all_vals[..5]);
    println!(
        "If col-major [2048, 256], W[:5, 0] = {:?}",
        [
            all_vals[0],
            all_vals[2048],
            all_vals[4096],
            all_vals[6144],
            all_vals[8192]
        ]
    );

    println!("\nHuggingFace reference:");
    println!("  W[0, :5] = [0.0281, 0.0059, -0.0003, -0.0056, 0.0075]");
    println!("  W[:5, 0] = [0.0281, 0.0176, 0.0359, 0.0165, -0.0222]");

    // The fused_q6k_parallel_matvec assumes:
    // - 2048 columns (in_dim)
    // - Each column has one superblock of 256 values
    // - So superblock 0 = column 0 of W^T = row 0 of W
    // Wait, that doesn't add up. Let me think again...

    // Actually, the fused function has in_dim=2048 columns, each with 256 rows.
    // That's a [2048, 256] matrix in column-major = W^T stored column-major.
    // To get W, we need W[i, j] = data[j][i] where data is W^T column-major.

    // For the matvec y = W @ x:
    // y[i] = sum_j W[i, j] * x[j] = sum_j data[j][i] * x[j]
    // Which is what the fused function computes.

    // So the fused function should be correct IF the data is [2048, 256] column-major.
    // Let me verify: is superblock 0 the first column (256 values for column 0)?

    println!("\nSuperblock 0 should be W^T[:, 0] = W[0, :256] (first 256 of row 0)");
    println!("  But HF W is [256, 2048], so W[0, :256] doesn't make sense...");

    // Oh! I think I see the confusion. The dimensions are:
    // - HF: W is [256, 2048], y = W @ x gives y[256], x[2048]
    // - GGUF stores: [2048, 256] = W^T

    // For y = W @ x where W is [256, 2048]:
    // y[i] = sum_j W[i, j] * x[j] for j in 0..2048

    // If GGUF stores W^T as [2048, 256] column-major:
    // W^T[:, i] = W[i, :] is stored contiguously
    // So to compute y[i] = sum_j W[i, j] * x[j], we need to access W[i, j] = W^T[j, i]

    // The fused function accesses data in the pattern:
    // for col in 0..2048:
    //   column = dequantize(superblock[col])  # 256 values = W^T[:, col] = W[*, col]
    //   for row in 0..256:
    //     output[row] += column[row] * x[col]
    //
    // This computes: output[row] += W[row, col] * x[col]
    // Which is exactly y[row] = sum_col W[row, col] * x[col]

    // So the algorithm is correct, but maybe the data isn't laid out as expected?

    // Let me verify by checking what W[0, 0] should be vs what we get
    println!("\nW[0, 0] should be HF W[0, 0] = 0.0281");
    println!("Superblock 0, element 0 = {:?}", sb0[0]);

    // If superblock 0 is W[:, 0] (first column), then sb0[0] = W[0, 0]
    // If superblock 0 is W[0, :256] (first 256 of first row), then sb0[0] = W[0, 0] too

    // Let's check W[0, 1] and W[1, 0]:
    println!("W[0, 1] should be 0.0059 (HF)");
    println!("W[1, 0] should be 0.0176 (HF)");
    println!("sb0[1] = {:?}", sb0[1]); // If col-major, this is W[1, 0]
                                       // If row-major, this is W[0, 1]
}
