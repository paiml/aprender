//! M32d numerical equivalence test for the qwen3_moe KV cache path.
//!
//! Validates that `run_qwen3_moe_generate` (post-M32d default: cache-on,
//! per-token incremental decode) produces IDENTICAL greedy outputs to the
//! legacy full-prefill-per-token loop (cache-off, pre-M32d behavior).
//!
//! ## Why this test exists
//!
//! KV cache vs full-prefill compute attention in different orders.
//! Sums-of-products on f32 are non-associative, so logits may differ
//! at the ULP level. The greedy-argmax classifier is robust to those
//! differences IFF the per-token attention is mathematically equivalent.
//! This test pins that equivalence into CI as an opt-in regression check.
//!
//! ## Gating
//!
//! `#[ignore]` by default. Activated by:
//!
//! ```text
//! QWEN3_MOE_GGUF_PATH=/path/to/qwen3-moe.gguf \
//!   cargo test --test moe_kv_cache_equivalence \
//!   -p aprender-serve --features cuda --release -- --ignored --nocapture
//! ```
//!
//! When `QWEN3_MOE_GGUF_PATH` is unset, the test prints SKIP and passes.

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::infer::qwen3_moe_generate::run_qwen3_moe_generate;

fn gguf_path() -> Option<String> {
    std::env::var("QWEN3_MOE_GGUF_PATH").ok()
}

/// Legacy full-prefill-per-token decode (pre-M32d behavior). Used as the
/// ground-truth comparison oracle for the cache-on path.
fn legacy_full_prefill_generate(
    mapped: &MappedGGUFModel,
    model: &OwnedQuantizedModel,
    input_tokens: &[u32],
    max_tokens: usize,
) -> Vec<u32> {
    let num_experts = mapped.model.expert_count().expect("expert_count");
    let num_experts_per_tok = mapped.model.expert_used_count().expect("expert_used_count");
    let moe_intermediate = mapped
        .model
        .expert_feed_forward_length()
        .expect("expert_feed_forward_length");

    let data = mapped.data();
    let num_layers = model.config().num_layers;
    let mut moe_layers = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        moe_layers.push(load_qwen3_moe_layer(&mapped.model, data, layer_idx).expect("layer load"));
    }

    let mut tokens = input_tokens.to_vec();
    for _ in 0..max_tokens {
        let logits = model
            .forward_qwen3_moe(
                &tokens,
                &moe_layers,
                num_experts,
                num_experts_per_tok,
                moe_intermediate,
                data,
            )
            .expect("forward_qwen3_moe");

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .expect("argmax");
        tokens.push(next_token);
    }
    tokens
}

#[test]
#[ignore = "requires real Qwen3-MoE GGUF via QWEN3_MOE_GGUF_PATH env var"]
fn moe_kv_cache_matches_full_prefill_on_first_4_tokens() {
    let Some(path) = gguf_path() else {
        eprintln!(
            "SKIP: QWEN3_MOE_GGUF_PATH not set. M32d numerical equivalence \
             requires a real qwen3_moe GGUF on disk."
        );
        return;
    };

    let mapped = MappedGGUFModel::from_path(&path)
        .unwrap_or_else(|e| panic!("Failed to mmap GGUF at {path}: {e}"));
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped");

    // Same input + max_tokens for both paths.
    let input_tokens: Vec<u32> = vec![9707, 198]; // "Hello\n" — small fixed prompt
    let max_tokens = 4;

    // (a) cache-on path (M32d default).
    let cache_on_start = std::time::Instant::now();
    let cache_on_tokens = run_qwen3_moe_generate(
        &mapped,
        &model,
        &input_tokens,
        &QuantizedGenerateConfig {
            max_tokens,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
            ..QuantizedGenerateConfig::default()
        },
    )
    .expect("run_qwen3_moe_generate (cache-on)");
    let cache_on_wall = cache_on_start.elapsed();

    // (b) cache-off path (pre-M32d legacy full-prefill).
    let cache_off_start = std::time::Instant::now();
    let cache_off_tokens =
        legacy_full_prefill_generate(&mapped, &model, &input_tokens, max_tokens);
    let cache_off_wall = cache_off_start.elapsed();

    // Both runs include the prompt; compare full sequences.
    eprintln!("cache-on wall:  {cache_on_wall:?}");
    eprintln!("cache-off wall: {cache_off_wall:?}");
    eprintln!("cache-on tokens:  {cache_on_tokens:?}");
    eprintln!("cache-off tokens: {cache_off_tokens:?}");

    // M32d invariant: greedy outputs must be byte-identical.
    // Float-equivalence on intermediate logits would not survive (different
    // sum-of-products order), but the argmax classifier IS robust to those
    // ULP-scale differences IFF the underlying attention math is correct.
    assert_eq!(
        cache_on_tokens, cache_off_tokens,
        "M32d KV cache must produce identical greedy tokens to full-prefill.\n\
         cache-on:  {:?}\n\
         cache-off: {:?}",
        cache_on_tokens, cache_off_tokens
    );

    // Perf sanity: cache-on should not be SLOWER than cache-off on 4 tokens.
    // (At small token counts, prefill amortization is comparable; at 100+
    // tokens cache-on should be MUCH faster — that's the post-M32d goal,
    // but this test only validates correctness, not speedup.)
    eprintln!(
        "M32d numerical equivalence DISCHARGED: {} tokens match across cache-on / cache-off paths.",
        cache_on_tokens.len()
    );
}
