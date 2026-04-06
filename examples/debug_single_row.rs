//! CORRECTNESS-002: Debug single row computation
//!
//! Manually compute one output row and compare CPU vs GPU.
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_single_row

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with: cargo run --features cuda --example debug_single_row");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuda_main()
}

#[cfg(feature = "cuda")]
fn cuda_main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    };
    use realizar::quantize::fused_q6k_dot;
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let vocab_size = cpu_model.config().vocab_size;
    let num_layers = cpu_model.config().num_layers;
    let kv_dim = cpu_model.config().num_kv_heads * (hidden_dim / cpu_model.config().num_heads);

    eprintln!("hidden_dim={}, vocab_size={}", hidden_dim, vocab_size);

    // Compute bytes per row for Q6K
    let super_blocks_per_row = hidden_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 210;
    eprintln!(
        "Super-blocks per row: {}, bytes per row: {}",
        super_blocks_per_row, bytes_per_row
    );

    // Get LM head data
    let lm_data = &cpu_model.lm_head_weight().data;
    eprintln!("LM head data len: {}", lm_data.len());
    eprintln!(
        "Expected len: {} rows * {} bytes = {}",
        vocab_size,
        bytes_per_row,
        vocab_size * bytes_per_row
    );

    // Run CPU forward to get normed_hidden
    let test_token: u32 = 791;
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;
    eprintln!(
        "\nCPU forward complete. First 5 logits: {:?}",
        &cpu_logits[..5]
    );

    // Now manually verify a few rows using CPU Q6K
    eprintln!("\n=== Manual single-row verification ===");

    // We need the normed_hidden from the forward pass. Let's extract it.
    // Since we can't easily get it, let's use a simple test input instead.
    let test_input: Vec<f32> = (0..hidden_dim).map(|i| ((i as f32) * 0.01).sin()).collect();
    let input_sum: f32 = test_input.iter().sum();
    eprintln!("Test input sum: {:.4}", input_sum);

    // Manually compute row 0 and row 100 using CPU Q6K dot
    for row in [0, 100, 1000, 10000, 50000] {
        if row >= vocab_size {
            break;
        }
        let row_start = row * bytes_per_row;
        let row_end = row_start + bytes_per_row;
        let row_data = &lm_data[row_start..row_end];

        let cpu_row_result = fused_q6k_dot(row_data, &test_input)?;
        eprintln!("Row {}: CPU Q6K dot = {:.6}", row, cpu_row_result);

        // Print first super-block info for this row
        // Q6_K layout: ql (128) + qh (64) + scales (16) + d (2) = 210 bytes
        let d_bytes = &row_data[208..210];
        let d_f16 = half::f16::from_bits(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));
        eprintln!(
            "  Row {} sb0: d={:.6}, scales[0..4]={:?}",
            row,
            d_f16.to_f32(),
            &row_data[192..196]
        );
    }

    // Now let's run GPU and capture the normed_hidden to compare
    eprintln!("\n=== GPU path with debug ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    std::env::set_var("GPU_DEBUG", "1");
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    eprintln!(
        "\nGPU forward complete. First 5 logits: {:?}",
        &gpu_logits[..5]
    );

    // Compare specific rows
    eprintln!("\n=== Row comparison (CPU forward vs GPU forward) ===");
    for row in [0, 100, 1000, 10000, 50000] {
        if row >= vocab_size {
            break;
        }
        eprintln!(
            "Row {}: CPU={:.6}, GPU={:.6}, diff={:.6}",
            row,
            cpu_logits[row],
            gpu_logits[row],
            cpu_logits[row] - gpu_logits[row]
        );
    }

    // Check pattern: are GPU results from different rows?
    eprintln!("\n=== Pattern analysis ===");
    eprintln!("Looking for GPU[0] in CPU logits...");
    let gpu0 = gpu_logits[0];
    for (i, &v) in cpu_logits.iter().enumerate() {
        if (v - gpu0).abs() < 0.5 {
            eprintln!("  CPU[{}] = {:.4} ~ GPU[0] = {:.4}", i, v, gpu0);
            if i < 20 {
                continue;
            }
            break;
        }
    }

    // Check if GPU rows are shuffled or transposed
    eprintln!("\n=== Checking if GPU outputs are from different input positions ===");
    // If the kernel is using wrong offsets, GPU[i] might equal CPU[j] for some relationship
    // Let's check a few candidates

    // Calculate correlation with different row offsets
    for offset in [0, 1, -1, 256, -256, hidden_dim as i64, -(hidden_dim as i64)] {
        let mut dot = 0.0f64;
        let mut cpu_sq = 0.0f64;
        let mut gpu_sq = 0.0f64;
        let mut count = 0;

        for (i, &g_val) in gpu_logits.iter().enumerate().take(vocab_size.min(10000)) {
            let j = i as i64 + offset;
            if j < 0 || j >= vocab_size as i64 {
                continue;
            }
            let c = cpu_logits[j as usize] as f64;
            let g = g_val as f64;
            dot += c * g;
            cpu_sq += c * c;
            gpu_sq += g * g;
            count += 1;
        }

        if count > 0 && cpu_sq > 0.0 && gpu_sq > 0.0 {
            let corr = dot / (cpu_sq.sqrt() * gpu_sq.sqrt());
            if corr.abs() > 0.1 {
                eprintln!("  Offset {}: corr = {:.4}", offset, corr);
            }
        }
    }

    Ok(())
}
