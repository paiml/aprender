//! Test V weight matvec manually

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{dequantize_q6_k, fused_q6k_parallel_matvec};
use realizar::rms_norm;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim; // 2048
    let eps = model.config().eps;

    // Token 450 embedding
    let start = 450 * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    let layer = &model.layers()[0];
    let normed = rms_norm(&embedding, &layer.attn_norm_weight, eps);

    println!("Input L2: {:.6}", l2_norm(&normed));
    println!("Input first 5: {:?}", &normed[..5]);

    let OwnedQKVWeights::Separate {
        q: _,
        k: _,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate")
    };

    // V weight is [in_dim=2048, out_dim=256] in GGUF column-major
    // HuggingFace stores as [out_dim=256, in_dim=2048] row-major

    println!("\nV weight dimensions:");
    println!("  in_dim (stored): {}", v_weight.in_dim);
    println!("  out_dim (stored): {}", v_weight.out_dim);

    // Method 1: Use fused_q6k_parallel_matvec
    let v1 = fused_q6k_parallel_matvec(&v_weight.data, &normed, v_weight.in_dim, v_weight.out_dim)
        .expect("test");
    println!("\nMethod 1 (fused_q6k_parallel_matvec):");
    println!("  Output L2: {:.6}", l2_norm(&v1));
    println!("  Output first 5: {:?}", &v1[..5]);

    // Method 2: Dequantize and do manual matvec
    // For column-major [2048, 256], the layout is:
    // - Elements 0..2048 are column 0 (output dimension 0)
    // - Elements 2048..4096 are column 1 (output dimension 1)
    // So y[i] = sum_j W[j, i] * x[j] where W is [2048, 256] column-major

    // Dequantize all blocks
    let total_elements = v_weight.in_dim * v_weight.out_dim; // 2048 * 256
    let num_blocks = total_elements / 256;
    let mut full_weight = Vec::new();
    for i in 0..num_blocks {
        let block_data = &v_weight.data[i * 210..(i + 1) * 210];
        let dequant = dequantize_q6_k(block_data).expect("test");
        full_weight.extend(dequant);
    }

    // But wait - is the GGUF layout [in_dim, out_dim] or [out_dim, in_dim]?
    // Let's check by computing both ways

    // Assume GGUF is [2048, 256] column-major
    // So W[j, i] is at index i*2048 + j
    let mut v2_col_major = vec![0.0f32; 256];
    for i in 0..256 {
        let mut sum = 0.0f32;
        for j in 0..2048 {
            sum += full_weight[i * 2048 + j] * normed[j];
        }
        v2_col_major[i] = sum;
    }
    println!("\nMethod 2a (manual, assume [2048, 256] col-major -> W[j,i] = data[i*2048+j]):");
    println!("  Output L2: {:.6}", l2_norm(&v2_col_major));
    println!("  Output first 5: {:?}", &v2_col_major[..5]);

    // Assume GGUF is [256, 2048] row-major
    // So W[i, j] is at index i*2048 + j
    let mut v2_row_major = vec![0.0f32; 256];
    for i in 0..256 {
        let mut sum = 0.0f32;
        for j in 0..2048 {
            sum += full_weight[i * 2048 + j] * normed[j];
        }
        v2_row_major[i] = sum;
    }
    println!("\nMethod 2b (manual, assume [256, 2048] row-major -> W[i,j] = data[i*2048+j]):");
    println!("  Output L2: {:.6}", l2_norm(&v2_row_major));
    println!("  Output first 5: {:?}", &v2_row_major[..5]);

    // HuggingFace reference
    println!("\nHuggingFace V projection:");
    println!("  L2: 0.197834");
    println!("  First 5: [-0.0018, 0.0031, -0.0022, -0.0012, 0.0032]");
}
