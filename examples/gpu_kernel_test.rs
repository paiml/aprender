#![allow(dead_code)]
//! GH-559: Minimal GPU kernel correctness test
//!
//! Tests individual CUDA kernels (RMSNorm, GEMV) in isolation to find which
//! kernel produces wrong results on Blackwell sm_121.
//!
//! Run with: cargo run --example gpu_kernel_test --release --features cuda

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::apr::MappedAprModel;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    println!("GH-559: GPU Kernel Correctness Test");
    println!("====================================");
    println!("Model: {}", path);

    // Load model
    let data = std::fs::read(&path)?;
    let format = realizar::format::detect_format(&data[..8.min(data.len())])?;
    drop(data);

    let model = match format {
        realizar::format::ModelFormat::Apr { .. } => {
            let mapped = MappedAprModel::from_path(std::path::Path::new(&path))?;
            OwnedQuantizedModel::from_apr(&mapped)?
        },
        _ => {
            let mapped = MappedGGUFModel::from_path(&path)?;
            OwnedQuantizedModel::from_mapped(&mapped)?
        },
    };

    let hidden_dim = model.config().hidden_dim;
    let epsilon = model.config().rms_norm_epsilon;
    let token_id = model.config().bos_token_id.unwrap_or(1) as u32;

    println!("Hidden dim: {}", hidden_dim);
    println!("Epsilon: {}", epsilon);
    println!("BOS token: {}", token_id);

    // Get CPU embedding
    let cpu_embed = model.embed(&[token_id]);
    println!("\n=== Test 1: Embedding ===");
    println!(
        "CPU embed first 5: {:?}",
        &cpu_embed[..5.min(cpu_embed.len())]
    );
    let cpu_embed_sum: f32 = cpu_embed.iter().sum();
    println!("CPU embed sum: {:.6}", cpu_embed_sum);

    // CPU RMSNorm
    println!("\n=== Test 2: RMSNorm ===");
    let sum_sq: f32 = cpu_embed.iter().map(|x| x * x).sum();
    let rms = (sum_sq / cpu_embed.len() as f32 + epsilon).sqrt();
    println!("CPU RMS value: {:.6}", rms);

    // Get layer 0 attn_norm gamma from model
    let gamma = &model.layers[0].attn_norm_gamma;
    println!(
        "Gamma len: {}, first 5: {:?}",
        gamma.len(),
        &gamma[..5.min(gamma.len())]
    );

    let cpu_normed: Vec<f32> = cpu_embed
        .iter()
        .zip(gamma.iter())
        .map(|(x, g)| (x / rms) * g)
        .collect();
    println!(
        "CPU normed first 5: {:?}",
        &cpu_normed[..5.min(cpu_normed.len())]
    );
    let cpu_normed_sum: f32 = cpu_normed.iter().sum();
    println!("CPU normed sum: {:.6}", cpu_normed_sum);

    // GPU: Create CUDA model (skip parity gate)
    println!("\n=== GPU Setup ===");
    std::env::set_var("SKIP_PARITY_GATE", "1");
    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;
    std::env::remove_var("SKIP_PARITY_GATE");
    println!("GPU model ready");

    // GPU: Upload embedding
    let gpu_embed = cuda_model.executor_mut().upload_input(&cpu_embed)?;
    let mut gpu_embed_host = vec![0.0f32; cpu_embed.len()];
    gpu_embed.copy_to_host(&mut gpu_embed_host)?;
    let embed_cos = cosine_sim(&cpu_embed, &gpu_embed_host);
    println!("Embed upload cosine: {:.6}", embed_cos);

    // GPU: RMSNorm
    let gpu_normed =
        cuda_model
            .executor_mut()
            .rmsnorm_layer0_attn(&gpu_embed, hidden_dim as u32, epsilon)?;
    cuda_model.executor_mut().sync_stream()?;
    let mut gpu_normed_host = vec![0.0f32; hidden_dim];
    gpu_normed.copy_to_host(&mut gpu_normed_host)?;

    let norm_cos = cosine_sim(&cpu_normed, &gpu_normed_host);
    println!(
        "\nGPU normed first 5: {:?}",
        &gpu_normed_host[..5.min(gpu_normed_host.len())]
    );
    let gpu_normed_sum: f32 = gpu_normed_host.iter().sum();
    println!("GPU normed sum: {:.6}", gpu_normed_sum);
    println!("RMSNorm cosine: {:.6}", norm_cos);

    if norm_cos >= 0.99 {
        println!("PASS: RMSNorm kernel is correct on this GPU");
    } else {
        println!(
            "FAIL: RMSNorm kernel produces wrong output (cosine={:.6})",
            norm_cos
        );
        println!("Root cause: RMSNorm PTX kernel, not GEMV");
        return Ok(());
    }

    // GPU: Q GEMV (Layer 0 Q projection)
    println!("\n=== Test 3: Q4K GEMV ===");
    let q_dim = (model.config().num_heads * (hidden_dim / model.config().num_heads)) as u32;
    let gpu_q =
        cuda_model
            .executor_mut()
            .test_gemv_layer0_q(&gpu_normed, q_dim, hidden_dim as u32)?;
    cuda_model.executor_mut().sync_stream()?;
    let mut gpu_q_host = vec![0.0f32; q_dim as usize];
    gpu_q.copy_to_host(&mut gpu_q_host)?;

    let q_sum: f32 = gpu_q_host.iter().sum();
    let q_rms: f32 =
        (gpu_q_host.iter().map(|x| x * x).sum::<f32>() / gpu_q_host.len() as f32).sqrt();
    println!(
        "GPU Q first 5: {:?}",
        &gpu_q_host[..5.min(gpu_q_host.len())]
    );
    println!("GPU Q sum={:.4}, rms={:.4}", q_sum, q_rms);

    // CPU Q GEMV for comparison
    let cpu_q = model.layers[0].attn_q_weight.matvec(&cpu_normed);
    let cpu_q_sum: f32 = cpu_q.iter().sum();
    let cpu_q_rms: f32 = (cpu_q.iter().map(|x| x * x).sum::<f32>() / cpu_q.len() as f32).sqrt();
    println!("CPU Q first 5: {:?}", &cpu_q[..5.min(cpu_q.len())]);
    println!("CPU Q sum={:.4}, rms={:.4}", cpu_q_sum, cpu_q_rms);

    let q_cos = cosine_sim(&cpu_q, &gpu_q_host);
    println!("Q GEMV cosine: {:.6}", q_cos);

    if q_cos >= 0.99 {
        println!("PASS: Q4K GEMV kernel is correct on this GPU");
    } else {
        println!(
            "FAIL: Q4K GEMV kernel produces wrong output (cosine={:.6})",
            q_cos
        );
        println!("Root cause: Q4K GEMV PTX kernel");
    }

    println!("\n=== Summary ===");
    println!(
        "Embedding upload: cosine={:.6} {}",
        embed_cos,
        if embed_cos >= 0.99 { "PASS" } else { "FAIL" }
    );
    println!(
        "RMSNorm:          cosine={:.6} {}",
        norm_cos,
        if norm_cos >= 0.99 { "PASS" } else { "FAIL" }
    );
    println!(
        "Q4K GEMV:         cosine={:.6} {}",
        q_cos,
        if q_cos >= 0.99 { "PASS" } else { "FAIL" }
    );

    Ok(())
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
    }
}
