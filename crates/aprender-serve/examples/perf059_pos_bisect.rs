//! PERF-059 (#2785): per-POSITION CPU-vs-GPU logit differential on the M=1
//! `forward_gpu_resident` path.
//!
//! #2785 proposed that a 7B GGUF reaches a code path that still assumes a
//! quantisation type, and that this is why the 7B CUDA path produces garbage.
//! This instrument answers the prior half of that question directly: it runs
//! the SAME probe through the CPU reference (`forward_single_with_cache`) and
//! the M=1 GPU-resident path, keeping the full logit vector at EVERY position,
//! and prints per-position cosine and argmax. If the M=1 oracle were mis-reading
//! a weight type the divergence would appear here, at whatever position the
//! affected weight first matters.
//!
//! Measured on unmodified main (34248e8fe), RTX 4090, the 7B Q4_K_M over a
//! 57-token probe: cosine >= 0.999812 at every position. The M=1 oracle is
//! sound on that model, which is what moved PERF-059 off the qtype hypothesis
//! and onto the batched path.
//!
//! This is a bisection instrument, not a test: it prints, it does not assert.
//! The band-level assertion lives in scripts/perf059_band_ladder.sh.
//!
//!   MODEL_PATH=... PROMPT=... NTOK=... MAXSEQ=...
//!   cargo run --release --features cuda --example perf059_pos_bisect

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    };

    let path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf".to_string());
    let prompt = std::env::var("PROMPT")
        .unwrap_or_else(|_| "Write a Python function that returns the sum of a list.".to_string());

    let mapped = MappedGGUFModel::from_path(&path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let cfg = cpu_model.config().clone();
    println!(
        "model={path}\n hidden={} layers={} heads={} kv_heads={} inter={} vocab={}",
        cfg.hidden_dim,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.intermediate_dim,
        cfg.vocab_size
    );

    let mut tokens: Vec<u32> = mapped
        .model
        .encode(&prompt)
        .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());
    if let Some(n) = std::env::var("NTOK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        tokens.truncate(n);
    }
    println!("probe tokens = {}", tokens.len());

    let max_seq = std::env::var("MAXSEQ")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(tokens.len() + 4);
    println!("max_seq_len = {max_seq}");
    let kv_dim = cfg.num_kv_heads * (cfg.hidden_dim / cfg.num_heads);

    // CPU reference
    let mut cpu_cache = OwnedQuantizedKVCache::new(cfg.num_layers, kv_dim, max_seq);
    let mut cpu: Vec<Vec<f32>> = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        cpu.push(cpu_model.forward_single_with_cache(t, &mut cpu_cache, pos)?);
    }

    // GPU M=1 resident path
    let mut cuda = OwnedQuantizedModelCuda::with_max_seq_len(
        OwnedQuantizedModel::from_mapped(&mapped)?,
        0,
        max_seq,
    )
    .map_err(|e| format!("cuda init failed: {:?}", e.error))?;
    // Freshly constructed model: its GPU KV cache is already clean.
    let mut gpu_cache = OwnedQuantizedKVCache::new(cfg.num_layers, kv_dim, max_seq);
    let mut gpu: Vec<Vec<f32>> = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        gpu.push(cuda.forward_gpu_resident(t, &mut gpu_cache, pos)?);
    }

    println!("\npos  cosine     cpu_argmax gpu_argmax  cpu_max     gpu_max");
    for pos in 0..tokens.len() {
        let (a, b) = (&cpu[pos], &gpu[pos]);
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            let (x, y) = (f64::from(*x), f64::from(*y));
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let cos = if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na.sqrt() * nb.sqrt())
        };
        let am = |v: &Vec<f32>| {
            v.iter()
                .enumerate()
                .max_by(|p, q| p.1.partial_cmp(q.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let mx = |v: &Vec<f32>| v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "{:3}  {:.6}  {:9}  {:9}  {:10.4}  {:10.4}",
            pos,
            cos,
            am(a),
            am(b),
            mx(a),
            mx(b)
        );
    }
    Ok(())
}
