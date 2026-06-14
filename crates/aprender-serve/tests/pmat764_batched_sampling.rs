//! PMAT-764: CUDA continuous-batching must honor per-request temperature/top_k.
//!
//! Pre-fix, the batched decode (`batched_decode_step` -> `forward_batched_to_token_ids`)
//! performed on-GPU greedy ARGMAX for EVERY slot, ignoring each request's temperature/top_k.
//! So two requests with the SAME prompt but different temperature produced IDENTICAL output.
//!
//! This loads the real 0.5B model on the GPU, runs a 2-slot batch (slot 0 greedy, slot 1
//! high-temperature), and asserts the high-temp slot DIVERGES from greedy. Pre-fix the two
//! sequences were identical -> this assertion is the PMAT-764 falsifier. Skips gracefully if
//! the model or CUDA is unavailable.
//!
//! VALIDATED 2026-06-14 on RTX 4090 with qwen2.5-coder-0.5b-instruct-q4k.apr (run with
//! SKIP_PARITY_GATE=1 to bypass a SEPARATE pre-existing CPU/GPU forward-parity-gate failure
//! for this apr — unrelated to the sampling dispatch under test):
//!   slot0 greedy:  [785, 11, 1879, 374, 198, 4279, 144328, ...]
//!   slot1 temp1.5: [785, 11, 1879, 374, 86879, 84899, 73787, ...]
//! Same prompt prefix, immediate divergence at the first generated token -> per-request
//! temperature is honored. (Without the fix both slots were byte-identical greedy.)
#![cfg(feature = "cuda")]

use realizar::apr::MappedAprModel;
use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig};

const MODEL: &str = "/mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-q4k.apr";

fn cfg(temperature: f32, top_k: usize, seed: u64) -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens: 32,
        temperature,
        top_k,
        seed,
        stop_tokens: vec![],
        ..Default::default()
    }
}

#[test]
fn pmat764_batched_honors_per_request_temperature() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("SKIP pmat764: model missing: {MODEL}");
        return;
    }
    let mapped = match MappedAprModel::from_path(MODEL) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP pmat764: MappedAprModel::from_path: {e}");
            return;
        },
    };
    let model = match OwnedQuantizedModel::from_apr(&mapped) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP pmat764: from_apr: {e}");
            return;
        },
    };
    let mut cuda = match OwnedQuantizedModelCuda::new(model, 0) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP pmat764: CUDA unavailable: {e}");
            return;
        },
    };

    // Same prompt for both slots (content is irrelevant to the divergence test).
    let prompt: Vec<u32> = vec![785, 11, 1879, 374];
    let prompts = vec![prompt.clone(), prompt.clone()];
    // slot 0: greedy (temp=0, top_k=1). slot 1: high-temperature sampling.
    let configs = vec![cfg(0.0, 1, 1), cfg(1.5, 50, 12345)];
    let on_tokens: Vec<Box<dyn FnMut(u32) -> bool + Send>> =
        vec![Box::new(|_| true), Box::new(|_| true)];

    let seqs = cuda
        .generate_batched_streaming(&prompts, &configs, on_tokens)
        .expect("batched generation failed");

    assert_eq!(seqs.len(), 2, "expected 2 slot sequences");
    eprintln!(
        "PMAT-764 slot0 (greedy)  len={}: {:?}",
        seqs[0].len(),
        seqs[0]
    );
    eprintln!(
        "PMAT-764 slot1 (temp1.5) len={}: {:?}",
        seqs[1].len(),
        seqs[1]
    );
    assert!(
        seqs[0].len() > prompt.len(),
        "greedy slot generated nothing"
    );
    assert!(
        seqs[1].len() > prompt.len(),
        "sampled slot generated nothing"
    );
    // The fix: high-temperature sampling must DIVERGE from greedy. Pre-fix (forced greedy
    // argmax for ALL slots) the two identical-prompt sequences were equal -> this fails.
    assert_ne!(
        seqs[0], seqs[1],
        "PMAT-764: batched decode ignored per-request temperature (temp=1.5 slot == greedy slot)"
    );
}
