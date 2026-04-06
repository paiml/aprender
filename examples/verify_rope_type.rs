//! Verify RoPE type detection and application for Qwen2
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load Qwen2 model
    let qwen_path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let qwen_mapped = MappedGGUFModel::from_path(qwen_path)?;
    let qwen_model = OwnedQuantizedModel::from_mapped(&qwen_mapped)?;

    println!("=== Qwen2 Config ===");
    println!("Architecture: {}", qwen_model.config().architecture);
    println!("Hidden dim: {}", qwen_model.config().hidden_dim);
    println!("Num heads: {}", qwen_model.config().num_heads);
    println!("Num KV heads: {}", qwen_model.config().num_kv_heads);
    println!("RoPE theta: {}", qwen_model.config().rope_theta);
    println!(
        "RoPE type: {} (0=NORM adjacent pairs, 2=NEOX split halves)",
        qwen_model.config().rope_type
    );
    println!("Epsilon: {}", qwen_model.config().eps);

    // Check raw metadata
    println!("\n=== Raw GGUF Metadata (RoPE related) ===");
    for (key, value) in &qwen_mapped.model.metadata {
        if key.contains("rope") || key.contains("eps") {
            println!("  {}: {:?}", key, value);
        }
    }

    // Also load TinyLlama for comparison
    let tiny_path = "/tmp/tinyllama.gguf";
    if std::path::Path::new(tiny_path).exists() {
        let tiny_mapped = MappedGGUFModel::from_path(tiny_path)?;
        let tiny_model = OwnedQuantizedModel::from_mapped(&tiny_mapped)?;

        println!("\n=== TinyLlama Config (comparison) ===");
        println!("Architecture: {}", tiny_model.config().architecture);
        println!("Hidden dim: {}", tiny_model.config().hidden_dim);
        println!("Num heads: {}", tiny_model.config().num_heads);
        println!("Num KV heads: {}", tiny_model.config().num_kv_heads);
        println!("RoPE theta: {}", tiny_model.config().rope_theta);
        println!(
            "RoPE type: {} (0=NORM adjacent pairs, 2=NEOX split halves)",
            tiny_model.config().rope_type
        );
        println!("Epsilon: {}", tiny_model.config().eps);

        println!("\n=== TinyLlama Raw Metadata (RoPE related) ===");
        for (key, value) in &tiny_mapped.model.metadata {
            if key.contains("rope") || key.contains("eps") {
                println!("  {}: {:?}", key, value);
            }
        }
    }

    // Manually test RoPE rotation
    println!("\n=== Manual RoPE Test ===");
    let head_dim = qwen_model.config().hidden_dim / qwen_model.config().num_heads;
    let half_dim = head_dim / 2;
    println!("Head dim: {}", head_dim);
    println!("Half dim: {}", half_dim);

    // Create test input: [1, 2, 3, 4, ...] for one head
    let mut x_neox: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let mut x_norm: Vec<f32> = x_neox.clone();

    // Pre-compute cos/sin for position 0
    let theta = qwen_model.config().rope_theta;
    let mut cos_vals = vec![0.0f32; half_dim];
    let mut sin_vals = vec![0.0f32; half_dim];
    let position = 0usize;
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let (sin_v, cos_v) = angle.sin_cos();
        cos_vals[i] = cos_v;
        sin_vals[i] = sin_v;
    }

    println!("\nPosition 0 cos/sin (first 4):");
    println!("  cos: {:?}", &cos_vals[..4.min(cos_vals.len())]);
    println!("  sin: {:?}", &sin_vals[..4.min(sin_vals.len())]);

    // Apply NEOX style (split halves)
    {
        let (first_half, second_half) = x_neox.split_at_mut(half_dim);
        for i in 0..half_dim {
            let v1 = first_half[i];
            let v2 = second_half[i];
            first_half[i] = v1 * cos_vals[i] - v2 * sin_vals[i];
            second_half[i] = v1 * sin_vals[i] + v2 * cos_vals[i];
        }
    }

    // Apply NORM style (adjacent pairs)
    for i in 0..half_dim {
        let x0 = x_norm[2 * i];
        let x1 = x_norm[2 * i + 1];
        x_norm[2 * i] = x0 * cos_vals[i] - x1 * sin_vals[i];
        x_norm[2 * i + 1] = x0 * sin_vals[i] + x1 * cos_vals[i];
    }

    println!(
        "\nInput (first 8): {:?}",
        &(0..head_dim).map(|i| (i + 1) as f32).collect::<Vec<_>>()[..8.min(head_dim)]
    );
    println!("NEOX result (first 8): {:?}", &x_neox[..8.min(head_dim)]);
    println!("NORM result (first 8): {:?}", &x_norm[..8.min(head_dim)]);

    // At position 0, both should give the same result because cos(0)=1, sin(0)=0
    let neox_norm: f32 = x_neox.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_norm: f32 = x_norm.iter().map(|x| x * x).sum::<f32>().sqrt();
    let orig_norm: f32 = (1..=head_dim as i32)
        .map(|i| (i * i) as f32)
        .sum::<f32>()
        .sqrt();
    println!("\nNorm preservation check:");
    println!("  Original norm: {:.4}", orig_norm);
    println!("  NEOX norm: {:.4}", neox_norm);
    println!("  NORM norm: {:.4}", norm_norm);

    // Check position 5 where rotation actually happens
    println!("\n=== Position 5 Test ===");
    let mut x_neox_5: Vec<f32> = (0..head_dim).map(|i| (i + 1) as f32).collect();
    let mut x_norm_5: Vec<f32> = x_neox_5.clone();
    let position = 5usize;

    for i in 0..half_dim {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let (sin_v, cos_v) = angle.sin_cos();
        cos_vals[i] = cos_v;
        sin_vals[i] = sin_v;
    }

    println!("Position 5 cos/sin (first 4):");
    println!("  cos: {:?}", &cos_vals[..4.min(cos_vals.len())]);
    println!("  sin: {:?}", &sin_vals[..4.min(sin_vals.len())]);

    // Apply NEOX style
    {
        let (first_half, second_half) = x_neox_5.split_at_mut(half_dim);
        for i in 0..half_dim {
            let v1 = first_half[i];
            let v2 = second_half[i];
            first_half[i] = v1 * cos_vals[i] - v2 * sin_vals[i];
            second_half[i] = v1 * sin_vals[i] + v2 * cos_vals[i];
        }
    }

    // Apply NORM style
    for i in 0..half_dim {
        let x0 = x_norm_5[2 * i];
        let x1 = x_norm_5[2 * i + 1];
        x_norm_5[2 * i] = x0 * cos_vals[i] - x1 * sin_vals[i];
        x_norm_5[2 * i + 1] = x0 * sin_vals[i] + x1 * cos_vals[i];
    }

    println!(
        "\nNEOX result (first 8): {:?}",
        &x_neox_5[..8.min(head_dim)]
    );
    println!("NORM result (first 8): {:?}", &x_norm_5[..8.min(head_dim)]);

    // Check if results differ significantly
    let diff: f32 = x_neox_5
        .iter()
        .zip(x_norm_5.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / head_dim as f32;
    println!("\nMean absolute diff between NEOX and NORM: {:.6}", diff);

    Ok(())
}
