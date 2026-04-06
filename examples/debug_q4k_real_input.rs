//! CORRECTNESS-002: Q4K test with actual normalized hidden state input
//!
//! Tests if GPU Q4K produces correct output when using real input
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_q4k_real_input

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::quantize::fused_q4k_dot;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CORRECTNESS-002: Q4K with real normalized hidden input\n");

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let test_token: u32 = 791;

    // Get embedding
    let embedding_offset = (test_token as usize) * hidden_dim;
    let cpu_embedding: Vec<f32> =
        model.token_embedding()[embedding_offset..embedding_offset + hidden_dim].to_vec();

    // RMSNorm on embedding
    let layer = &model.layers()[0];
    let norm_weight = &layer.attn_norm_weight;
    let eps = model.config().eps;
    let sum_sq: f32 = cpu_embedding.iter().map(|x| x * x).sum();
    let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();

    let normed_input: Vec<f32> = cpu_embedding
        .iter()
        .zip(norm_weight.iter())
        .map(|(h, w)| h / rms * w)
        .collect();

    eprintln!("[Input] Normed hidden first 5: {:?}", &normed_input[..5]);

    // Get Q weight data
    let (q_data, q_in_dim, q_out_dim) = match &layer.qkv_weight {
        OwnedQKVWeights::Separate { q, .. } => (&q.data, q.in_dim, q.out_dim),
        OwnedQKVWeights::Fused(_) => {
            eprintln!("Fused QKV - cannot test separately");
            return Ok(());
        },
    };

    let sb_per_row = q_in_dim.div_ceil(256);
    let bytes_per_row = sb_per_row * 144;

    eprintln!(
        "Q weight: in_dim={}, out_dim={}, bytes_per_row={}",
        q_in_dim, q_out_dim, bytes_per_row
    );

    // CPU Q projection using the normalized input
    let cpu_q: Vec<f32> = (0..q_out_dim)
        .map(|row| {
            let row_start = row * bytes_per_row;
            let row_data = &q_data[row_start..row_start + bytes_per_row];
            fused_q4k_dot(row_data, &normed_input).unwrap_or(f32::NAN)
        })
        .collect();

    eprintln!(
        "[CPU] Q first 5: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        cpu_q[0], cpu_q[1], cpu_q[2], cpu_q[3], cpu_q[4]
    );

    // GPU Q projection with same normalized input
    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;
        use trueno_gpu::driver::GpuBuffer;

        let mut executor = CudaExecutor::new(0)?;
        let context = executor.context();

        let weight_buf = GpuBuffer::<u8>::from_host(context, q_data)?;
        let weight_ptr = weight_buf.as_ptr();
        let input_buf = GpuBuffer::<f32>::from_host(context, &normed_input)?;
        let output_buf = GpuBuffer::<f32>::new(context, q_out_dim)?;

        executor.q4k_gemv_into(
            weight_ptr,
            &input_buf,
            &output_buf,
            q_out_dim as u32,
            q_in_dim as u32,
        )?;
        executor.synchronize()?;

        let mut gpu_q = vec![0.0f32; q_out_dim];
        output_buf.copy_to_host(&mut gpu_q)?;

        eprintln!(
            "[GPU] Q first 5: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            gpu_q[0], gpu_q[1], gpu_q[2], gpu_q[3], gpu_q[4]
        );

        // GPU reported value from forward pass was:
        // [PAR-058-L0] Q OK, first 3: [0.09119174, 0.45370343, -0.17122838]
        eprintln!("\nGPU forward reported Q: [0.0912, 0.4537, -0.1712]");

        // Check if our direct GPU call matches
        let direct_gpu_match = (gpu_q[0] - cpu_q[0]).abs() < 0.01
            && (gpu_q[1] - cpu_q[1]).abs() < 0.01
            && (gpu_q[2] - cpu_q[2]).abs() < 0.01;

        if direct_gpu_match {
            eprintln!("[direct-Q4K] Direct GPU call matches CPU!");
        } else {
            eprintln!("[direct-Q4K] Direct GPU call DIFFERS from CPU!");
            eprintln!(
                "  Diff: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                cpu_q[0] - gpu_q[0],
                cpu_q[1] - gpu_q[1],
                cpu_q[2] - gpu_q[2],
                cpu_q[3] - gpu_q[3],
                cpu_q[4] - gpu_q[4]
            );
        }

        // Calculate correlation
        let mut dot = 0.0f64;
        let mut cpu_sq = 0.0f64;
        let mut gpu_sq = 0.0f64;
        for i in 0..q_out_dim {
            let c = cpu_q[i] as f64;
            let g = gpu_q[i] as f64;
            dot += c * g;
            cpu_sq += c * c;
            gpu_sq += g * g;
        }
        let corr = dot / (cpu_sq.sqrt() * gpu_sq.sqrt());
        eprintln!("\nCorrelation (direct test): {:.6}", corr);

        // Check if the GPU forward reported value matches what we get
        let forward_reported = [0.09119174f32, 0.45370343, -0.17122838];
        let forward_match = (gpu_q[0] - forward_reported[0]).abs() < 0.01;
        eprintln!("Forward-reported Q matches direct GPU: {}", forward_match);

        if !forward_match {
            eprintln!("  Direct GPU Q[0] = {:.6}", gpu_q[0]);
            eprintln!("  Forward reported Q[0] = {:.6}", forward_reported[0]);
            eprintln!("  This means the forward pass uses different input or weights!");

            // Check if forward reported matches CPU
            let forward_cpu_match = (cpu_q[0] - forward_reported[0]).abs() < 0.01;
            eprintln!("  Forward-reported Q matches CPU: {}", forward_cpu_match);
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CUDA not enabled");
    }

    Ok(())
}
