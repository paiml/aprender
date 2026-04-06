//! Trace layer 0 QKV projections in detail

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;
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
    let token_id = 450usize;
    let start = token_id * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("=== Layer 0 QKV Trace ===\n");
    println!("Embedding L2: {:.6}", l2_norm(&embedding));
    println!("Embedding first 5: {:?}", &embedding[..5]);

    let layer = &model.layers()[0];

    // RMSNorm
    let normed = rms_norm(&embedding, &layer.attn_norm_weight, eps);
    println!("\nAfter RMSNorm:");
    println!("  L2: {:.6}", l2_norm(&normed));
    println!("  First 5: {:?}", &normed[..5]);
    println!(
        "  Mean: {:.6}",
        normed.iter().sum::<f32>() / hidden_dim as f32
    );
    println!("  Std: {:.6}", {
        let mean = normed.iter().sum::<f32>() / hidden_dim as f32;
        (normed.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32).sqrt()
    });

    // QKV weights
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: k_weight,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate QKV")
    };

    println!("\nQ weight info:");
    println!("  Shape: {}x{}", q_weight.in_dim, q_weight.out_dim);
    println!("  qtype: {}", q_weight.qtype);
    println!("  Data len: {} bytes", q_weight.data.len());

    // Q projection
    let q = fused_q4k_parallel_matvec(&q_weight.data, &normed, q_weight.in_dim, q_weight.out_dim)
        .expect("test");
    println!("\nQ projection output:");
    println!("  L2: {:.6}", l2_norm(&q));
    println!("  First 10: {:?}", &q[..10]);
    println!("  Last 10: {:?}", &q[q.len() - 10..]);
    println!("  Mean: {:.6}", q.iter().sum::<f32>() / q.len() as f32);
    println!(
        "  Min: {:.6}",
        q.iter().copied().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Max: {:.6}",
        q.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // K projection
    let k = fused_q4k_parallel_matvec(&k_weight.data, &normed, k_weight.in_dim, k_weight.out_dim)
        .expect("test");
    println!("\nK projection output:");
    println!("  Shape: {}", k.len());
    println!("  L2: {:.6}", l2_norm(&k));
    println!("  First 10: {:?}", &k[..10]);

    // V projection
    let v = fused_q4k_parallel_matvec(&v_weight.data, &normed, v_weight.in_dim, v_weight.out_dim)
        .expect("test");
    println!("\nV projection output:");
    println!("  Shape: {}", v.len());
    println!("  L2: {:.6}", l2_norm(&v));
    println!("  First 10: {:?}", &v[..10]);

    // Check if Q has any extreme values
    let q_sorted: Vec<f32> = {
        let mut s: Vec<f32> = q.iter().map(|x| x.abs()).collect();
        s.sort_by(|a, b| b.partial_cmp(a).expect("test"));
        s
    };
    println!("\nQ top 10 absolute values: {:?}", &q_sorted[..10]);
}
