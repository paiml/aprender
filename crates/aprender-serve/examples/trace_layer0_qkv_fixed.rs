//! Trace layer 0 QKV projections with correct qtype handling

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
use realizar::rms_norm;

const GGUF_TYPE_Q4_K: u32 = 12;
const GGUF_TYPE_Q6_K: u32 = 14;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn fused_matmul(input: &[f32], data: &[u8], qtype: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(data, input, in_dim, out_dim).expect("test"),
        GGUF_TYPE_Q6_K => {
            // V weights are column-major with out_dim=256
            fused_q6k_parallel_matvec(data, input, in_dim, out_dim).expect("test")
        },
        _ => panic!("Unsupported qtype: {}", qtype),
    }
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

    println!("=== Layer 0 QKV Trace (Fixed) ===\n");
    println!("Embedding L2: {:.6}", l2_norm(&embedding));

    let layer = &model.layers()[0];

    // RMSNorm
    let normed = rms_norm(&embedding, &layer.attn_norm_weight, eps);
    println!("After RMSNorm L2: {:.6}", l2_norm(&normed));

    // QKV weights
    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: k_weight,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate QKV")
    };

    println!("\nWeight types:");
    println!("  Q: qtype={} (Q4_K=12, Q6_K=14)", q_weight.qtype);
    println!("  K: qtype={}", k_weight.qtype);
    println!("  V: qtype={}", v_weight.qtype);

    // Q projection (Q4_K)
    let q = fused_matmul(
        &normed,
        &q_weight.data,
        q_weight.qtype,
        q_weight.in_dim,
        q_weight.out_dim,
    );
    println!("\nQ projection: L2={:.6}, shape={}", l2_norm(&q), q.len());
    println!("  First 5: {:?}", &q[..5]);

    // K projection (Q4_K)
    let k = fused_matmul(
        &normed,
        &k_weight.data,
        k_weight.qtype,
        k_weight.in_dim,
        k_weight.out_dim,
    );
    println!("\nK projection: L2={:.6}, shape={}", l2_norm(&k), k.len());
    println!("  First 5: {:?}", &k[..5]);

    // V projection (Q6_K column-major)
    let v = fused_matmul(
        &normed,
        &v_weight.data,
        v_weight.qtype,
        v_weight.in_dim,
        v_weight.out_dim,
    );
    println!("\nV projection: L2={:.6}, shape={}", l2_norm(&v), v.len());
    println!("  First 5: {:?}", &v[..5]);
    println!("  Contains NaN: {}", v.iter().any(|x| x.is_nan()));
    println!(
        "  Min: {:.6}",
        v.iter().copied().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Max: {:.6}",
        v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
}
