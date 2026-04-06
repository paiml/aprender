//! Quick tok/s benchmark with KV cache (GQA-aware)
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};
use std::{env, time::Instant};

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).expect("Usage: bench_toks <model.gguf>");

    // Load model
    let load_start = Instant::now();
    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");
    let load_time = load_start.elapsed();

    let model_name = path.split('/').next_back().unwrap_or(path);
    println!("Model: {}", model_name);
    println!("Load time: {:.2?}", load_time);
    println!(
        "Config: {} layers, {} hidden, {} heads, {} kv_heads",
        model.config().num_layers,
        model.config().hidden_dim,
        model.config().num_heads,
        model.config().num_kv_heads
    );
    println!();

    // Encode prompt
    let prompt = "Once upon a time";
    let prompt_tokens = mapped.model.encode(prompt).expect("test");
    println!("Prompt: '{}' ({} tokens)", prompt, prompt_tokens.len());

    // Create KV cache with GQA-aware dimensions
    let max_seq_len = 256;
    let mut cache = OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        max_seq_len,
    );

    // Prefill: process prompt tokens
    let prefill_start = Instant::now();
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        let _ = model.forward_single_with_cache(tok, &mut cache, pos);
    }
    let prefill_time = prefill_start.elapsed();
    println!(
        "Prefill: {} tokens in {:.2?} ({:.1} tok/s)",
        prompt_tokens.len(),
        prefill_time,
        prompt_tokens.len() as f64 / prefill_time.as_secs_f64()
    );

    // Get initial logits for first generated token
    let mut logits_owned = model
        .forward_single_with_cache(
            prompt_tokens[prompt_tokens.len() - 1],
            &mut cache,
            prompt_tokens.len() - 1,
        )
        .expect("test");

    // Reset cache and re-prefill for clean benchmark
    cache = OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        max_seq_len,
    );

    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        logits_owned = model
            .forward_single_with_cache(tok, &mut cache, pos)
            .expect("test");
    }

    // Benchmark decode
    let num_tokens = 50;
    let mut generated_tokens = Vec::with_capacity(num_tokens);

    let decode_start = Instant::now();
    let mut pos = prompt_tokens.len();

    for _ in 0..num_tokens {
        // Greedy sampling
        let next_token = logits_owned
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("test"))
            .map(|(i, _)| i as u32)
            .expect("test");

        generated_tokens.push(next_token);

        // Forward single token with cache
        logits_owned = model
            .forward_single_with_cache(next_token, &mut cache, pos)
            .expect("test");
        pos += 1;
    }
    let decode_time = decode_start.elapsed();

    let tok_per_sec = num_tokens as f64 / decode_time.as_secs_f64();
    let ms_per_tok = decode_time.as_millis() as f64 / num_tokens as f64;

    // Decode output
    let mut output = String::new();
    for &tok_id in &generated_tokens {
        if (tok_id as usize) < vocab.len() {
            let tok_str = &vocab[tok_id as usize];
            output.push_str(&tok_str.replace("▁", " ").replace('\u{0120}', " "));
        }
    }

    println!();
    println!("Generated {} tokens in {:.2?}", num_tokens, decode_time);
    println!();
    println!("┌─────────────────────────────────────┐");
    println!(
        "│ {:>6.1} tok/s │ {:>6.1} ms/tok │",
        tok_per_sec, ms_per_tok
    );
    println!("└─────────────────────────────────────┘");
    println!();
    println!("Output:{}", output);
}
