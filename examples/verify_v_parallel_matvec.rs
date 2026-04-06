//! Verify fused_q6k_parallel_matvec gives correct V projection

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q6k_parallel_matvec;
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

    let OwnedQKVWeights::Separate {
        q: _,
        k: _,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate")
    };

    println!("\nV weight dimensions:");
    println!("  in_dim: {}", v_weight.in_dim);
    println!("  out_dim: {}", v_weight.out_dim);
    println!("  data len: {} bytes", v_weight.data.len());

    // Use fused_q6k_parallel_matvec (row-major)
    // This expects: out_dim rows of in_dim elements each
    // V weight is [256, 2048] row-major, so out_dim=256, in_dim=2048
    let v_rowmajor =
        fused_q6k_parallel_matvec(&v_weight.data, &normed, v_weight.in_dim, v_weight.out_dim)
            .expect("test");

    println!("\nfused_q6k_parallel_matvec output:");
    println!("  L2: {:.6}", l2_norm(&v_rowmajor));
    println!("  First 5: {:?}", &v_rowmajor[..5]);

    println!("\nHuggingFace reference:");
    println!("  L2: 0.197834");
    println!("  First 5: [-0.0018, 0.0031, -0.0022, -0.0012, 0.0032]");

    println!("\nDifference from HF:");
    let hf_ref = [
        -0.0018329927,
        0.0030865134,
        -0.0021960088,
        -0.0011552889,
        0.0032128505,
    ];
    for i in 0..5 {
        println!(
            "  [{}]: our={:.6}, hf={:.6}, diff={:.6}",
            i,
            v_rowmajor[i],
            hf_ref[i],
            v_rowmajor[i] - hf_ref[i]
        );
    }
}
