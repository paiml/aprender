//! CORRECTNESS-011: BrickProfiler checksum divergence detection
//!
//! This is the CORRECT way to find GPU/CPU divergence per the spec:
//! "use BrickProfiler checksum divergence" (not ad-hoc debug prints)
//!
//! Run with: cargo run --example brick_divergence_trace --release --features cuda

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    println!("CORRECTNESS-011: BrickProfiler Divergence Detection");
    println!("====================================================");
    println!("Model: {}", path);

    let mapped = MappedGGUFModel::from_path(&path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let token_id = 791u32;
    let position: usize = 0;

    println!("\nToken ID: {}", token_id);
    println!("Position: {}", position);

    // ========================================================================
    // Phase 1: CPU Forward with Checksum Recording
    // ========================================================================
    println!("\n=== Phase 1: CPU Forward (Reference) ===");

    // Create CPU profiler and enable checksum recording
    let mut cpu_profiler = trueno::BrickProfiler::enabled();

    // Run CPU forward - need to instrument model to record checksums
    // For now, just run forward and record final output
    let cpu_logits = model.forward(&[token_id])?;

    // Record CPU output checksum
    cpu_profiler.record_checksum("final_logits", 0, position as u32, &cpu_logits);

    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));

    println!("CPU argmax: {:?}", cpu_argmax);
    println!(
        "CPU logits checksum: {:016x}",
        cpu_profiler
            .get_checksums()
            .first()
            .map(|c| c.checksum)
            .unwrap_or(0)
    );

    // ========================================================================
    // Phase 2: GPU Forward with Checksum Recording
    // ========================================================================
    println!("\n=== Phase 2: GPU Forward (Test) ===");

    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;
    cuda_model.clear_decode_graph();

    // Enable profiling on GPU executor
    cuda_model.enable_profiling();

    // Disable CUDA graphs for debugging
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    // Create GPU profiler for checksum recording
    let mut gpu_profiler = trueno::BrickProfiler::enabled();

    // Run GPU forward
    let mut dummy_cache = realizar::gguf::OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        100,
    );
    let gpu_logits = cuda_model.forward_gpu_resident(token_id, &mut dummy_cache, position)?;

    // Record GPU output checksum
    gpu_profiler.record_checksum("final_logits", 0, position as u32, &gpu_logits);

    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));

    println!("GPU argmax: {:?}", gpu_argmax);
    println!(
        "GPU logits checksum: {:016x}",
        gpu_profiler
            .get_checksums()
            .first()
            .map(|c| c.checksum)
            .unwrap_or(0)
    );

    // ========================================================================
    // Phase 3: Automated Divergence Detection
    // ========================================================================
    println!("\n=== Phase 3: Divergence Detection ===");

    if let Some(divergence) = gpu_profiler.find_divergence(&cpu_profiler) {
        println!("DIVERGENCE DETECTED!");
        println!("  Kernel: {}", divergence.kernel_name);
        println!("  Layer: {}", divergence.layer_idx);
        println!("  Position: {}", divergence.position);
        println!("  Expected checksum: {:016x}", divergence.expected_checksum);
        println!("  Actual checksum:   {:016x}", divergence.actual_checksum);
    } else {
        println!("No divergence detected in recorded checksums.");
    }

    // ========================================================================
    // Phase 4: Detailed Comparison
    // ========================================================================
    println!("\n=== Phase 4: Detailed Analysis ===");

    // Check if argmax matches
    let cpu_idx = cpu_argmax.map(|(i, _)| i);
    let gpu_idx = gpu_argmax.map(|(i, _)| i);

    if cpu_idx == gpu_idx {
        println!("PASS: CPU and GPU argmax match: {:?}", cpu_idx);
    } else {
        println!("FAIL: Argmax mismatch!");
        println!("  CPU argmax: {:?}", cpu_idx);
        println!("  GPU argmax: {:?}", gpu_idx);

        // Detailed comparison
        if let (Some((ci, cv)), Some((gi, gv))) = (cpu_argmax, gpu_argmax) {
            println!("\nLogit values at argmax positions:");
            println!("  CPU[{}] = {:.6}", ci, cv);
            println!("  GPU[{}] = {:.6}", gi, gv);
            println!("  CPU[{}] = {:.6}", gi, cpu_logits.get(gi).unwrap_or(&0.0));
            println!("  GPU[{}] = {:.6}", ci, gpu_logits.get(ci).unwrap_or(&0.0));
        }

        // Correlation analysis
        if cpu_logits.len() == gpu_logits.len() {
            let mean_cpu: f32 = cpu_logits.iter().sum::<f32>() / cpu_logits.len() as f32;
            let mean_gpu: f32 = gpu_logits.iter().sum::<f32>() / gpu_logits.len() as f32;

            let mut cov = 0.0f32;
            let mut var_cpu = 0.0f32;
            let mut var_gpu = 0.0f32;

            for (c, g) in cpu_logits.iter().zip(gpu_logits.iter()) {
                let dc = c - mean_cpu;
                let dg = g - mean_gpu;
                cov += dc * dg;
                var_cpu += dc * dc;
                var_gpu += dg * dg;
            }

            let corr = cov / (var_cpu.sqrt() * var_gpu.sqrt() + 1e-10);
            println!("\nCorrelation: {:.6}", corr);
            println!("Mean CPU: {:.6}, Mean GPU: {:.6}", mean_cpu, mean_gpu);

            // Linear regression: GPU = a*CPU + b
            let slope = cov / (var_cpu + 1e-10);
            let intercept = mean_gpu - slope * mean_cpu;
            println!("Linear fit: GPU ≈ {:.4}*CPU + {:.4}", slope, intercept);

            if corr > 0.9 && (slope - 1.0).abs() > 0.01 {
                println!("\nDIAGNOSIS: High correlation but slope != 1.0");
                println!("This suggests a systematic scaling error in one of:");
                println!("  - RMSNorm (epsilon or weight application)");
                println!("  - Attention scaling (1/sqrt(d))");
                println!("  - LM head projection");
            }
        }
    }

    // Print profiler summary from GPU executor
    println!("\n=== GPU Brick Timing Summary ===");
    println!("{}", cuda_model.profiler_summary());

    Ok(())
}
