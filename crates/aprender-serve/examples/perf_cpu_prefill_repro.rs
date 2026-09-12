//! PERF: reproduce the prefill-to-decode ratio on the CPU quantized path (#2787).
//!
//! Not a gate, not a receipt producer. It times the two phases of the SAME
//! code path the W1 workload runs through — `OwnedQuantizedModel` prefill
//! followed by greedy decode — and prints every replicate.
//!
//! Usage:
//!   perf_cpu_prefill_repro <model.gguf> [prompt_tokens] [gen_tokens] [reps]

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};
use realizar::RealizarError;
use std::time::Instant;

fn main() -> Result<(), RealizarError> {
    let mut args = std::env::args().skip(1);
    // NO DEFAULT MODEL PATH. A fallback here is one developer's filesystem
    // shipped inside an example -- `scripts/check_hardcoded_paths.sh` counts it
    // as a shipped machine-specific path, and it is one: on any other box the
    // example would fail at load with a path the user never typed. The model
    // the #2787 measurement actually used is named in
    // `evidence/perf-2787/provenance.txt`, which is where a path belongs.
    let Some(model_path) = args.next() else {
        eprintln!("usage: perf_cpu_prefill_repro <model.gguf> [prompt_tokens] [gen_tokens] [reps]");
        eprintln!();
        eprintln!("The model path is required; there is no default.");
        eprintln!("See evidence/perf-2787/provenance.txt for the model this was measured with.");
        std::process::exit(2);
    };
    let n_prompt: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(513);
    let n_gen: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(128);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);

    eprintln!("model: {model_path}");
    let load_start = Instant::now();
    let mapped = MappedGGUFModel::from_path(&model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    eprintln!("loaded in {:?}", load_start.elapsed());
    let cfg = model.config();
    eprintln!(
        "config: layers={} hidden={} inter={} heads={} kv_heads={} vocab={}",
        cfg.num_layers,
        cfg.hidden_dim,
        cfg.intermediate_dim,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.vocab_size
    );

    // Deterministic prompt: a fixed English body repeated to the target length,
    // tokenized by the model's own vocabulary, then truncated/padded to exactly
    // n_prompt tokens so the shape is the declared one.
    let body = "The quick brown fox jumps over the lazy dog while the compiler \
                optimizes a tight inner loop over quantized weights. ";
    let mut text = String::new();
    while text.len() < n_prompt * 6 {
        text.push_str(body);
    }
    let mut prompt = mapped
        .model
        .encode(&text)
        .ok_or_else(|| RealizarError::InvalidShape {
            reason: "tokenizer produced no ids".to_string(),
        })?;
    prompt.truncate(n_prompt);
    if prompt.len() < n_prompt {
        return Err(RealizarError::InvalidShape {
            reason: format!("only {} tokens, wanted {}", prompt.len(), n_prompt),
        });
    }
    eprintln!("prompt: {} tokens", prompt.len());
    let arm = if std::env::var("APR_BATCHED_PREFILL").as_deref() == Ok("0") {
        "pertoken"
    } else {
        "batched"
    };
    eprintln!(
        "arm: {arm}  supports_batched_prefill={}  chunk={:?}",
        model.supports_batched_prefill(),
        std::env::var("APR_PREFILL_CHUNK").ok()
    );

    println!("arm,rep,prefill_tok,prefill_s,prefill_tok_s,decode_tok,decode_s,decode_tok_s,ratio,t_req_s");
    for rep in 0..reps {
        let mut cache = OwnedQuantizedKVCache::from_config(cfg, n_prompt + n_gen);

        let t0 = Instant::now();
        // PREFILL-CPU (#2787): the entry point both arms share. Batched when
        // APR_BATCHED_PREFILL != "0" and the model is covered; per-token loop
        // otherwise. ONE binary, so the two arms differ only in this dispatch.
        let mut logits = model.prefill_prompt(&prompt, &mut cache)?;
        let prefill_s = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let mut out = Vec::with_capacity(n_gen);
        for i in 0..n_gen {
            let next = realizar::gguf::ops::argmax(&logits);
            out.push(next);
            logits = model.forward_single_with_cache(next, &mut cache, n_prompt + i)?;
        }
        let decode_s = t1.elapsed().as_secs_f64();

        let p_tps = n_prompt as f64 / prefill_s;
        let d_tps = n_gen as f64 / decode_s;
        println!(
            "{arm},{rep},{n_prompt},{prefill_s:.4},{p_tps:.3},{n_gen},{decode_s:.4},{d_tps:.3},{:.4},{:.3}",
            p_tps / d_tps,
            prefill_s + decode_s
        );
        if rep == 0 {
            eprintln!("[{arm}] all {} generated ids: {:?}", out.len(), out);
            eprintln!("[{arm}] text: {:?}", mapped.model.decode(&out));
        }
    }
    Ok(())
}
