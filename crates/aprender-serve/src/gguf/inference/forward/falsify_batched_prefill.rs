//! THE FALSIFIER for batched CPU prefill (PREFILL-CPU, #2787).
//!
//! Batched prefill exists to make the prompt phase faster **without changing
//! what the model computes**. A prefill that is 6x faster and shifts the tokens
//! is worthless, so the property under test is equality, not plausibility:
//!
//! 1. the KV cache the batched pass writes is BITWISE the cache the per-token
//!    loop writes,
//! 2. the logits it returns for the last prompt token are BITWISE the logits the
//!    per-token loop returns, and
//! 3. greedy continuation from either state emits the same token IDs.
//!
//! Delete `prefill_batched.rs`'s batched branch (or `batched_matmul.rs`) and
//! these go RED. `discrimination_*` below stay GREEN in that state — they prove
//! the inputs actually differ across positions, so the equalities above are
//! constraints rather than tautologies.
//!
//! The model here is synthetic and small on purpose: the equality is a property
//! of the arithmetic, not of any particular checkpoint, and a unit test may not
//! depend on a 4.4 GB GGUF. The end-to-end measurement on the real W1 model is
//! recorded on the pull request, not here.

use crate::gguf::test_helpers::create_q4k_test_data;
use crate::gguf::{
    GGUFConfig, OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedLayer, OwnedQuantizedModel,
};

/// A LLaMA-shaped GQA model in the covered class: RMSNorm, RoPE, SwiGLU, Q4_K.
///
/// `hidden_dim = 256` so every matmul lands on the Q8_K super-block path (the
/// one the 7B W1 model uses), `num_kv_heads < num_heads` so GQA head mapping is
/// exercised, and two layers so a bug in layer-to-layer state carry shows up.
fn covered_model() -> OwnedQuantizedModel {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_heads: 8,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 512,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    let hidden_dim = config.hidden_dim;
    let inter = config.intermediate_dim;
    let head_dim = hidden_dim / config.num_heads;
    let kv_dim = config.num_kv_heads * head_dim;

    let layer = || OwnedQuantizedLayer {
        attn_norm_weight: (0..hidden_dim)
            .map(|i| 0.8 + (i % 7) as f32 * 0.05)
            .collect(),
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Separate {
            q: create_q4k_test_data(hidden_dim, hidden_dim),
            k: create_q4k_test_data(hidden_dim, kv_dim),
            v: create_q4k_test_data(hidden_dim, kv_dim),
        },
        qkv_bias: None,
        attn_output_weight: create_q4k_test_data(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_q4k_test_data(hidden_dim, inter),
        ffn_up_bias: None,
        ffn_down_weight: create_q4k_test_data(inter, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_q4k_test_data(hidden_dim, inter)),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(
            (0..hidden_dim)
                .map(|i| 0.9 + (i % 5) as f32 * 0.04)
                .collect(),
        ),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
        post_attn_norm_weight: None,
        post_ffw_norm_weight: None,
    };

    OwnedQuantizedModel {
        // Embeddings MUST differ per token: a constant table would make every
        // row of the batch identical and the equality assertions vacuous.
        token_embedding: (0..config.vocab_size * hidden_dim)
            .map(|i| {
                let (t, d) = (i / hidden_dim, i % hidden_dim);
                ((t * 31 + d * 17) % 97) as f32 / 97.0 - 0.5
            })
            .collect(),
        position_embedding: None,
        layers: vec![layer(), layer()],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: create_q4k_test_data(hidden_dim, config.vocab_size),
        lm_head_bias: None,
        config,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

/// Deterministic prompt long enough to span several `PREFILL_CHUNK` chunks, so
/// the chunk boundary itself is under test.
fn prompt(len: usize, vocab: usize) -> Vec<u32> {
    (0..len).map(|i| ((i * 37 + 11) % vocab) as u32).collect()
}

/// Run the prompt one token at a time — the code path that shipped before this
/// change. Returns (last logits, the cache it built).
fn sequential(
    model: &OwnedQuantizedModel,
    p: &[u32],
    cap: usize,
) -> (Vec<f32>, OwnedQuantizedKVCache) {
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, cap);
    let mut logits = Vec::new();
    for (pos, &t) in p.iter().enumerate() {
        logits = model
            .forward_single_with_cache(t, &mut cache, pos)
            .expect("per-token prefill must succeed");
    }
    (logits, cache)
}

fn batched(
    model: &OwnedQuantizedModel,
    p: &[u32],
    cap: usize,
) -> (Vec<f32>, OwnedQuantizedKVCache) {
    let mut cache = OwnedQuantizedKVCache::from_config(&model.config, cap);
    let logits = model
        .forward_prefill_batched(p, &mut cache, 0)
        .expect("batched prefill must succeed on a covered model");
    (logits, cache)
}

#[test]
fn covered_model_is_actually_covered() {
    // If this ever goes false the equality tests below would pass by never
    // taking the batched path at all.
    assert!(
        covered_model().supports_batched_prefill(),
        "the falsifier's model must be in the covered class or it tests nothing"
    );
}

#[test]
fn falsify_batched_prefill_logits_are_bitwise_identical() {
    let model = covered_model();
    let p = prompt(77, model.config.vocab_size);
    let cap = p.len() + 16;

    let (seq_logits, _) = sequential(&model, &p, cap);
    let (bat_logits, _) = batched(&model, &p, cap);

    assert_eq!(seq_logits.len(), bat_logits.len());
    for (i, (a, b)) in seq_logits.iter().zip(bat_logits.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit {i}: per-token {a} != batched {b}"
        );
    }
}

#[test]
fn falsify_batched_prefill_kv_cache_is_bitwise_identical() {
    let model = covered_model();
    let p = prompt(77, model.config.vocab_size);
    let cap = p.len() + 16;

    let (_, seq_cache) = sequential(&model, &p, cap);
    let (_, bat_cache) = batched(&model, &p, cap);

    assert_eq!(
        seq_cache.len(),
        bat_cache.len(),
        "batched prefill left the cache at a different position"
    );
    assert_eq!(seq_cache.len(), p.len());
    for layer in 0..model.config.num_layers {
        for (name, a, b) in [
            ("K", seq_cache.get_k(layer), bat_cache.get_k(layer)),
            ("V", seq_cache.get_v(layer), bat_cache.get_v(layer)),
        ] {
            assert_eq!(
                a.len(),
                b.len(),
                "layer {layer} {name} cache length differs"
            );
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "layer {layer} {name}[{i}]: per-token {x} != batched {y}"
                );
            }
        }
    }
}

#[test]
fn falsify_greedy_continuation_emits_the_same_tokens() {
    let model = covered_model();
    let p = prompt(77, model.config.vocab_size);
    let cap = p.len() + 16;

    let continue_from = |mut logits: Vec<f32>, mut cache: OwnedQuantizedKVCache| {
        let mut out = Vec::new();
        for i in 0..12 {
            let next = crate::gguf::ops::argmax(&logits);
            out.push(next);
            logits = model
                .forward_single_with_cache(next, &mut cache, p.len() + i)
                .expect("decode step must succeed");
        }
        out
    };

    let (sl, sc) = sequential(&model, &p, cap);
    let (bl, bc) = batched(&model, &p, cap);
    assert_eq!(
        continue_from(sl, sc),
        continue_from(bl, bc),
        "greedy continuation diverged — a faster prefill that changes the tokens is worthless"
    );
}

#[test]
fn falsify_chunk_size_does_not_change_the_answer() {
    // `APR_PREFILL_CHUNK` only sets how many rows share a weight sweep. If the
    // answer moves with it, the batching is carrying state across rows it
    // should not.
    let model = covered_model();
    let p = prompt(77, model.config.vocab_size);
    let cap = p.len() + 16;
    let (reference, _) = sequential(&model, &p, cap);

    for chunk in [1usize, 3, 32, 64, 200] {
        let mut cache = OwnedQuantizedKVCache::from_config(&model.config, cap);
        // Chunking is a pure loop bound in `forward_prefill_batched`; drive it
        // by slicing the prompt so the test needs no process-wide env mutation
        // (which would race the other tests in this binary).
        let mut logits = Vec::new();
        let mut off = 0usize;
        while off < p.len() {
            let n = chunk.min(p.len() - off);
            logits = model
                .forward_prefill_batched(&p[off..off + n], &mut cache, off)
                .expect("batched prefill must succeed on a covered model");
            off += n;
        }
        for (i, (a, b)) in reference.iter().zip(logits.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "chunk {chunk} logit {i}: per-token {a} != batched {b}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// DISCRIMINATION CASES — these stay GREEN when the falsifiers above are RED.
// They prove the equalities are constraints, not tautologies.
// ---------------------------------------------------------------------------

#[test]
fn discrimination_positions_produce_different_states() {
    // If every position produced the same K/V, "the caches match" would hold for
    // any implementation, including one that ignored position entirely.
    let model = covered_model();
    let p = prompt(77, model.config.vocab_size);
    let (_, cache) = sequential(&model, &p, p.len() + 16);
    let kv_dim = model.config.kv_dim();
    let k = cache.get_k(0);
    assert!(k.len() >= 3 * kv_dim);
    assert_ne!(
        &k[..kv_dim],
        &k[kv_dim..2 * kv_dim],
        "positions 0 and 1 wrote identical keys — the equality falsifier would be vacuous"
    );
    assert!(
        k.iter().any(|v| *v != 0.0),
        "the whole K cache is zero — every equality here would hold trivially"
    );
}

#[test]
fn discrimination_wrong_start_position_changes_the_answer() {
    // The falsifier compares two paths that agree on position. This proves the
    // comparison is sensitive to position at all: the SAME token prefilled at a
    // different `start_pos` gets different RoPE angles and must write a
    // different key. Without this, a batched pass that fed every row position 0
    // would still satisfy "the caches match" if the reference had the same bug.
    //
    // Checked on the K cache rather than the logits deliberately: this synthetic
    // checkpoint's lm_head saturates, so its logits are position-insensitive and
    // would make a logits-based discrimination vacuous. The cache is where the
    // position actually lands.
    let model = covered_model();
    let p = prompt(32, model.config.vocab_size);
    let cap = p.len() + 64;
    let kv_dim = model.config.kv_dim();

    let mut c0 = OwnedQuantizedKVCache::from_config(&model.config, cap);
    model
        .forward_prefill_batched(&p, &mut c0, 0)
        .expect("batched prefill must succeed");

    // Same tokens, positions shifted by 8 — prime the cache with 8 real slots
    // first so the shift is a genuine position change, not a length change.
    let mut c8 = OwnedQuantizedKVCache::from_config(&model.config, cap);
    let pad = prompt(8, model.config.vocab_size);
    model
        .forward_prefill_batched(&pad, &mut c8, 0)
        .expect("batched prefill must succeed");
    model
        .forward_prefill_batched(&p, &mut c8, 8)
        .expect("batched prefill must succeed");

    // p[0] at position 0 vs p[0] at position 8, layer 0.
    let at_zero = &c0.get_k(0)[..kv_dim];
    let at_eight = &c8.get_k(0)[8 * kv_dim..9 * kv_dim];
    assert_ne!(
        at_zero, at_eight,
        "the same token at positions 0 and 8 wrote the same key — position has no effect, so the equality falsifier cannot see a position bug"
    );
}

#[test]
fn uncovered_models_fall_back_rather_than_run_differently() {
    // A model outside the covered class must be REFUSED by the guard, so
    // `prefill_prompt` takes the per-token loop. Silently batching it would be
    // the failure mode this guard exists to prevent.
    let mut m = covered_model();
    m.layers[1].post_ffw_norm_weight = Some(vec![1.0f32; m.config.hidden_dim]);
    assert!(
        !m.supports_batched_prefill(),
        "a Gemma-style post-FFN norm is not implemented in the batched path and must not be claimed"
    );

    let mut m2 = covered_model();
    m2.layers[0].ffn_gate_weight = None;
    assert!(
        !m2.supports_batched_prefill(),
        "a non-gated FFN is not implemented in the batched path and must not be claimed"
    );

    // And the fallback still produces the right answer.
    let p = prompt(9, m.config.vocab_size);
    let cap = p.len() + 4;
    let (reference, _) = sequential(&m, &p, cap);
    let mut cache = OwnedQuantizedKVCache::from_config(&m.config, cap);
    let got = m
        .prefill_prompt(&p, &mut cache)
        .expect("fallback prefill must succeed");
    for (a, b) in reference.iter().zip(got.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
