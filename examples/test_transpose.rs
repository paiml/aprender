//! Test if transposing weights helps - check a simple matmul
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_parallel_matvec;

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = 2048;
    let token_id = 450u32;

    // Get embedding
    let start = token_id as usize * hidden_dim;
    let embedding = &model.token_embedding()[start..start + hidden_dim];

    // Try layer 0 attention Q projection
    let layer = &model.layers()[0];
    let realizar::gguf::OwnedQKVWeights::Separate { q: q_weight, .. } = &layer.qkv_weight else {
        panic!("")
    };

    println!(
        "Q weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
        q_weight.in_dim,
        q_weight.out_dim,
        q_weight.qtype,
        q_weight.data.len()
    );

    // Our current computation (row-major assumption)
    let q_out =
        fused_q4k_parallel_matvec(&q_weight.data, embedding, q_weight.in_dim, q_weight.out_dim)
            .expect("test");
    println!(
        "\nRow-major result: L2={:.4}, first 5: {:?}",
        l2_norm(&q_out),
        &q_out[..5]
    );

    // For GGML, the data might be column-major
    // If stored column-major with shape [out_dim, in_dim], we need to transpose
    // Actually, let's just check what dimensions GGUF reports vs what we use

    // Check tensor info from model
    for tensor in &mapped.model.tensors {
        if tensor.name.contains("blk.0") && tensor.name.contains(".attn_q.") {
            println!("\nTensor '{}': dims={:?}", tensor.name, tensor.dims);
        }
    }
}
