//! Compare CPU vs GPU V values for layer 0

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let num_heads = cpu_model.config().num_heads;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;
    let eps = cpu_model.config().eps;

    let test_token: u32 = 791;
    let embedding = cpu_model.embed(&[test_token]);

    // CPU: RMSNorm + QKV projection
    let layer = &cpu_model.layers()[0];
    let ss: f32 = embedding.iter().map(|x| x * x).sum();
    let rms = (ss / hidden_dim as f32 + eps).sqrt();
    let cpu_normed: Vec<f32> = embedding
        .iter()
        .zip(layer.attn_norm_weight.iter())
        .map(|(&x, &w)| (x / rms) * w)
        .collect();

    let qkv_cpu = cpu_model.qkv_matmul(&cpu_normed, &layer.qkv_weight)?;
    let v_cpu = &qkv_cpu[q_dim + kv_dim..];

    eprintln!("\n=== CPU V values (layer 0) ===");
    eprintln!("V first 5: {:?}", &v_cpu[..5]);
    eprintln!("V sum: {:.6}", v_cpu.iter().sum::<f32>());

    // At position 0, attention output = V (softmax of single element = 1.0)
    // So CPU attention output first 5 = V first 5
    eprintln!("\n=== CPU Attention (should equal V at pos 0) ===");
    eprintln!("Attn first 5: {:?}", &v_cpu[..5]);

    eprintln!("\n=== GPU V values (from debug) ===");
    eprintln!("V first 5: [-0.11200829, -0.066736706, 0.2103174, 0.13783944, -0.58773506]");
    eprintln!("Attn first 3: [-0.11200829, -0.066736706, 0.2103174]");

    // Compare
    let gpu_v = [
        -0.11200829f32,
        -0.066736706,
        0.2103174,
        0.13783944,
        -0.58773506,
    ];
    eprintln!("\n=== Comparison ===");
    for i in 0..5 {
        let diff = (v_cpu[i] - gpu_v[i]).abs();
        eprintln!(
            "  [{}]: CPU={:.6}, GPU={:.6}, diff={:.6}",
            i, v_cpu[i], gpu_v[i], diff
        );
    }

    Ok(())
}
