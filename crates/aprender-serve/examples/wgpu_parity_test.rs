#![allow(clippy::many_single_char_names, clippy::needless_range_loop)]
//! FALSIFY-MBP-001: wgpu parity test on Blackwell sm_121
//!
//! Tests whether trueno's WgslForwardPass produces correct results on the
//! same GPU where CUDA PTX JIT fails (cosine=-0.005).
//!
//! Run with: cargo run --example wgpu_parity_test --release --features gpu

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::apr::MappedAprModel;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};
    use realizar::gpu::adapters::wgpu_adapter;
    use trueno::backends::gpu::GpuDevice;

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/src/apr-leaderboard/checkpoints/qwen2.5-coder-7b-instruct-q4k.apr".to_string()
    });

    println!("FALSIFY-MBP-001: wgpu Parity Test");
    println!("==================================");
    println!("Model: {}", path);

    // Load model
    let data = std::fs::read(&path)?;
    let format = realizar::format::detect_format(&data[..8.min(data.len())])?;
    drop(data);

    let model = if format == realizar::format::ModelFormat::Apr {
        let mapped = MappedAprModel::from_path(std::path::Path::new(&path))?;
        OwnedQuantizedModel::from_apr(&mapped)?
    } else {
        let mapped = MappedGGUFModel::from_path(&path)?;
        OwnedQuantizedModel::from_mapped(&mapped)?
    };

    let config = model.config();
    let hidden_dim = config.hidden_dim;
    let num_layers = config.num_layers;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let intermediate_dim = config.intermediate_dim;
    let vocab_size = config.vocab_size;
    let eps = config.eps;
    let token_id = config.bos_token_id.unwrap_or(1) as u32;

    println!(
        "Hidden: {}, Heads: {}/{}, Layers: {}, Vocab: {}",
        hidden_dim, num_heads, num_kv_heads, num_layers, vocab_size
    );
    println!("Token: {} (BOS), Epsilon: {}", token_id, eps);

    // CPU forward (reference)
    println!("\n=== CPU Forward ===");
    let kv_dim = num_kv_heads * head_dim;
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 2);
    let cpu_logits = model.forward_single_with_cache(token_id, &mut cpu_cache, 0)?;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v));
    println!("CPU argmax: {:?}", cpu_argmax);
    println!(
        "CPU logits sum: {:.4}, first 5: {:?}",
        cpu_logits.iter().sum::<f32>(),
        &cpu_logits[..5.min(cpu_logits.len())]
    );

    // wgpu forward
    println!("\n=== wgpu Setup ===");

    if !GpuDevice::is_available() {
        println!("SKIP: wgpu not available on this system");
        return Ok(());
    }

    let gpu = GpuDevice::new().map_err(|e| format!("GPU init: {e}"))?;
    println!("wgpu device ready");

    // Create WgslForwardPass with device/queue from GpuDevice
    let mut fwd = trueno::backends::gpu::WgslForwardPass::new(
        gpu.device,
        gpu.queue,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_dim,
    );

    // Dequantize weights
    println!("Dequantizing weights...");
    let weights = wgpu_adapter::dequant_model_weights(&model)?;

    // Upload weights
    for (name, data, _rows, _cols) in &weights {
        fwd.upload_weight(name, data);
    }

    // Init GPU KV caches
    fwd.init_kv_cache(num_layers);

    // KV caches (CPU-side)
    let mut kv_caches: Vec<(Vec<f32>, Vec<f32>)> = (0..num_layers)
        .map(|_| (vec![0.0f32; 64 * kv_dim], vec![0.0f32; 64 * kv_dim]))
        .collect();

    // Run wgpu forward layer by layer
    println!("\n=== wgpu Forward (layer-by-layer) ===");
    let mut hidden = model.embed(&[token_id]);
    println!(
        "Embedding: sum={:.4}, first 5: {:?}",
        hidden.iter().sum::<f32>(),
        &hidden[..5.min(hidden.len())]
    );

    for layer_idx in 0..num_layers {
        let prefix = format!("layer.{layer_idx}");
        let (ref mut kv_k, ref mut kv_v) = kv_caches[layer_idx];
        fwd.forward_layer(&mut hidden, &prefix, 0, kv_k, kv_v)
            .map_err(|e| format!("layer {layer_idx}: {e}"))?;
        if layer_idx % 7 == 0 || layer_idx == num_layers - 1 {
            println!(
                "  Layer {}/{}: sum={:.4}, first 3: {:?}",
                layer_idx,
                num_layers,
                hidden.iter().sum::<f32>(),
                &hidden[..3.min(hidden.len())]
            );
        }
    }

    // Output norm + LM head on CPU (same as CPU forward)
    let output_norm = model.output_norm_weight();
    let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
    let rms = (sq_sum / hidden.len() as f32 + eps).sqrt();
    let normed: Vec<f32> = hidden
        .iter()
        .zip(output_norm.iter())
        .map(|(x, g)| (x / rms) * g)
        .collect();

    // LM head matmul
    let lm_head_f32: Vec<f32> = weights
        .iter()
        .find(|(n, _, _, _)| n == "lm_head")
        .map(|(_, d, _, _)| d.clone())
        .unwrap_or_default();
    let mut wgpu_logits = vec![0.0f32; vocab_size];
    for row in 0..vocab_size {
        let mut sum = 0.0f32;
        for col in 0..hidden_dim {
            sum += lm_head_f32[row * hidden_dim + col] * normed[col];
        }
        wgpu_logits[row] = sum;
    }

    let wgpu_argmax = wgpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v));
    println!("wgpu argmax: {:?}", wgpu_argmax);
    println!(
        "wgpu logits sum: {:.4}, first 5: {:?}",
        wgpu_logits.iter().sum::<f32>(),
        &wgpu_logits[..5.min(wgpu_logits.len())]
    );

    // Parity comparison
    let cosine = cosine_sim(&cpu_logits, &wgpu_logits);
    println!("\n=== FALSIFY-MBP-001 Result ===");
    println!("Cosine similarity: {:.6}", cosine);

    if cosine >= 0.98 {
        println!(
            "PASS: wgpu produces correct results (cosine={:.6} >= 0.98)",
            cosine
        );
        println!("wgpu is a viable fix for the CUDA PTX JIT bug on sm_121");
    } else {
        println!("FAIL: wgpu cosine={:.6} < 0.98", cosine);
    }

    Ok(())
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot: f64 = 0.0;
    let mut na: f64 = 0.0;
    let mut nb: f64 = 0.0;
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
    }
}
