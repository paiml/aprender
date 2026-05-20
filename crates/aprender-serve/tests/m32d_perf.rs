//! M32d sustained-throughput perf gate.
//!
//! Pins the post-M32d throughput target into CI as a regression check.
//! The scope doc (`docs/specifications/m32d-moe-kv-cache-scope.md`)
//! commits to ≥ 5 tok/s sustained on Qwen3-Coder-30B-A3B-Instruct-Q4_K_M
//! (vs ~0.5 tok/s pre-M32d full-prefill-per-token). This test asserts the
//! number stays above the floor.
//!
//! ## Gating
//!
//! `#[ignore]` by default. Activated by:
//!
//! ```text
//! QWEN3_MOE_GGUF_PATH=/path/to/qwen3-moe.gguf \
//!   cargo test --test m32d_perf \
//!   -p aprender-serve --features cuda --release -- --ignored --nocapture
//! ```
//!
//! Measurement recorded on initial M32d ship (2026-05-20):
//! `9.62 tok/s sustained on 32 tokens, 9-token prompt`.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::infer::qwen3_moe_generate::run_qwen3_moe_generate;

const M32D_TPS_FLOOR: f64 = 5.0;
const M32D_PERF_GENERATE_TOKENS: usize = 32;

#[test]
#[ignore = "requires real Qwen3-MoE GGUF via QWEN3_MOE_GGUF_PATH env var"]
fn m32d_sustained_throughput_at_least_5_tps() {
    let Some(path) = std::env::var("QWEN3_MOE_GGUF_PATH").ok() else {
        eprintln!("SKIP: QWEN3_MOE_GGUF_PATH not set. M32d perf gate requires real GGUF.");
        return;
    };

    let mapped = MappedGGUFModel::from_path(&path)
        .unwrap_or_else(|e| panic!("Failed to mmap GGUF at {path}: {e}"));
    let model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped");

    // 9-token prompt — matches the order-of-magnitude of real chat prompts
    // after `apr code`'s system+context wrapping for a small user message.
    let input_tokens: Vec<u32> = vec![9707, 11, 5168, 0, 358, 614, 264, 3405, 13];

    let start = std::time::Instant::now();
    let tokens = run_qwen3_moe_generate(
        &mapped,
        &model,
        &input_tokens,
        &QuantizedGenerateConfig {
            max_tokens: M32D_PERF_GENERATE_TOKENS,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
            ..QuantizedGenerateConfig::default()
        },
    )
    .expect("run_qwen3_moe_generate");
    let wall = start.elapsed();

    let generated = tokens.len() - input_tokens.len();
    let tps = generated as f64 / wall.as_secs_f64();

    eprintln!(
        "M32d perf: {generated} tokens in {wall:?} = {tps:.2} tok/s sustained (floor: {M32D_TPS_FLOOR:.1})"
    );
    eprintln!("Generated tokens: {:?}", &tokens[input_tokens.len()..]);

    assert!(
        tps >= M32D_TPS_FLOOR,
        "M32d sustained throughput {tps:.2} tok/s < {M32D_TPS_FLOOR} tok/s floor. \
         Either KV cache regressed or the test host is unusually slow."
    );
}
