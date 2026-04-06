//! Compare CPU vs GPU RoPE
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let head_dim = hidden_dim / num_heads;
    let half_dim = head_dim / 2;
    let rope_theta = model.config().rope_theta;
    let rope_type = model.config().rope_type;

    println!(
        "rope_theta={}, rope_type={}, head_dim={}, half_dim={}",
        rope_theta, rope_type, head_dim, half_dim
    );

    // Test data: single head, simple values
    let position = 5usize;
    let mut test_data = vec![0.0f32; head_dim];
    for (i, val) in test_data.iter_mut().enumerate() {
        *val = (i + 1) as f32;
    }

    // CPU RoPE (NEOX style for rope_type == 2)
    let mut cpu_result = test_data.clone();
    if rope_type == 2 {
        // NEOX: split halves
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;
        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
            let angle = pos_f32 * freq;
            let (sin_v, cos_v) = angle.sin_cos();
            let x0 = test_data[i]; // First half
            let x1 = test_data[i + half_dim]; // Second half
            cpu_result[i] = x0 * cos_v - x1 * sin_v;
            cpu_result[i + half_dim] = x0 * sin_v + x1 * cos_v;
        }
    } else {
        // NORM: adjacent pairs
        let pos_f32 = position as f32;
        let head_dim_f32 = head_dim as f32;
        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
            let angle = pos_f32 * freq;
            let (sin_v, cos_v) = angle.sin_cos();
            let x0 = test_data[2 * i];
            let x1 = test_data[2 * i + 1];
            cpu_result[2 * i] = x0 * cos_v - x1 * sin_v;
            cpu_result[2 * i + 1] = x0 * sin_v + x1 * cos_v;
        }
    }

    // GPU RoPE (always uses NORM style - this is the bug!)
    let mut gpu_result = test_data.clone();
    let pos_f32 = position as f32;
    let head_dim_f32 = head_dim as f32;
    for i in 0..half_dim {
        let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim_f32);
        let angle = pos_f32 * freq;
        let (sin_v, cos_v) = angle.sin_cos();
        // GPU always uses adjacent pairs (NORM style)
        let x0 = test_data[2 * i];
        let x1 = test_data[2 * i + 1];
        gpu_result[2 * i] = x0 * cos_v - x1 * sin_v;
        gpu_result[2 * i + 1] = x0 * sin_v + x1 * cos_v;
    }

    println!("\nCPU (rope_type={}):", rope_type);
    println!("  first 8: {:?}", &cpu_result[..8]);
    println!(
        "  [half_dim..half_dim+8]: {:?}",
        &cpu_result[half_dim..half_dim + 8]
    );

    println!("\nGPU (always NORM):");
    println!("  first 8: {:?}", &gpu_result[..8]);
    println!(
        "  [half_dim..half_dim+8]: {:?}",
        &gpu_result[half_dim..half_dim + 8]
    );

    // Check if they match
    let mut max_diff = 0.0f32;
    for i in 0..head_dim {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("\nMax diff between CPU and GPU: {}", max_diff);

    if max_diff > 0.001 {
        println!("\n*** MISMATCH DETECTED - GPU RoPE is using wrong style! ***");
    }

    Ok(())
}
