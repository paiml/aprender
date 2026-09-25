//! #4268: `apr run` and `apr run --batch` serve a dense GGUF model through the
//! one engine. The witness records every `Session::generate`; a route that
//! goes back to its own loop leaves no entry and turns these red.

use super::*;
use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{GGUFConfig, OwnedQuantizedModel, QuantizedGenerateConfig};
use crate::session::entries_for;

fn model() -> OwnedQuantizedModel {
    create_test_model_with_config(&GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        rope_theta: 10000.0,
        context_length: 64,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    })
}

fn greedy(max_tokens: usize) -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    }
}

fn cpu_only() -> InferenceConfig {
    let mut config = InferenceConfig::new("/dense-4268.gguf");
    config.no_gpu = true;
    config
}

#[test]
fn run_serves_a_dense_model_through_the_engine() {
    // Prompts no other test uses, so the witness entries are this test's own.
    let prompt = [42_u32, 4268 % 100, 17, 99, 3];
    let want = model()
        .generate_with_cache(&prompt, &greedy(6))
        .expect("reference loop");
    let (tokens, used_gpu, gpu_attempted) =
        run_gguf_generate(model(), &prompt, &greedy(6), &cpu_only()).expect("run");
    assert_eq!(tokens, want);
    assert!(!used_gpu && !gpu_attempted);
    let entries = entries_for(&prompt);
    assert_eq!(entries.len(), 1, "run did not go through the engine");
    assert_eq!(entries[0].arch, "llama");
    assert!(!entries[0].on_gpu);
}

#[test]
fn run_trace_keeps_the_instrumented_loop() {
    let prompt = [42_u32, 68, 17, 99, 4];
    let traced = QuantizedGenerateConfig {
        trace: true,
        ..greedy(2)
    };
    run_gguf_generate(model(), &prompt, &traced, &cpu_only()).expect("traced run");
    assert!(
        entries_for(&prompt).is_empty(),
        "--trace lost its brick profiler"
    );
}

#[test]
fn batch_serves_a_dense_model_through_the_engine() {
    let mut batch = BatchModel {
        #[cfg(feature = "cuda")]
        gpu: None,
        #[cfg(feature = "gpu")]
        wgpu: None,
        cpu: Some(dense_cpu(model())),
    };
    let first = [42_u32, 68, 17, 99, 5];
    let second = [42_u32, 68, 17, 99, 6];
    for prompt in [first, second] {
        let want = model()
            .generate_with_cache(&prompt, &greedy(4))
            .expect("reference loop");
        let (tokens, used_gpu) = batch.generate(&prompt, &greedy(4)).expect("batch prompt");
        assert_eq!(tokens, want, "a later prompt must answer as a fresh one");
        assert!(!used_gpu);
        assert_eq!(
            entries_for(&prompt).len(),
            1,
            "batch did not go through the engine"
        );
    }
}
