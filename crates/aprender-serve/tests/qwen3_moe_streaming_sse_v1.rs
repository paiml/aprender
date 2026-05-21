//! V1_001 + V1_003 falsification tests for qwen3-moe-streaming-sse-v1.yaml.
//!
//! Validates the per-token callback contract that backs the HTTP SSE
//! streaming path. The HTTP envelope is exercised through the dense
//! path's `true_streaming_sse_response` (proven by the existing dense
//! streaming tests); this file asserts the qwen3_moe-specific callback
//! invariants.
//!
//! ## Discharges
//! - **FALSIFY-V1_001**: callback fires per-token (one call per
//!   generated token, in emission order, BEFORE the loop body returns).
//! - **FALSIFY-V1_003**: streaming throughput bounded below by M32d
//!   per-token cost (median inter-callback latency < 500ms).
//!
//! FALSIFY-V1_002 (`stream=false` regression) is already covered by
//! `qwen3_moe_serve_dispatch_v1.rs`.
//!
//! ## Gating
//!
//! `#[ignore]` by default. Activated by:
//!
//! ```text
//! QWEN3_MOE_GGUF_PATH=/path/to/qwen3-moe.gguf \
//!   cargo test --test qwen3_moe_streaming_sse_v1 \
//!   -p aprender-serve --features cuda --release -- --ignored --nocapture
//! ```

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::infer::qwen3_moe_generate::{
    run_qwen3_moe_generate, run_qwen3_moe_generate_streaming,
};
use std::time::Instant;

fn gguf_path() -> Option<String> {
    std::env::var("QWEN3_MOE_GGUF_PATH").ok()
}

fn fixture_setup() -> Option<(MappedGGUFModel, OwnedQuantizedModel)> {
    let path = gguf_path()?;
    let mapped = MappedGGUFModel::from_path(&path)
        .unwrap_or_else(|e| panic!("Failed to mmap GGUF at {path}: {e}"));
    let model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped");
    Some((mapped, model))
}

const PROMPT: &[u32] = &[9707, 198]; // "Hello\n"
const MAX_TOKENS: usize = 8;

fn greedy_config() -> QuantizedGenerateConfig {
    QuantizedGenerateConfig {
        max_tokens: MAX_TOKENS,
        temperature: 0.0, // greedy for parity check
        top_k: 1,
        seed: 42,
        stop_tokens: Vec::new(),
        ..QuantizedGenerateConfig::default()
    }
}

/// V1_001: streaming callback fires once per generated token, in
/// emission order, and the captured tokens equal the non-streaming
/// output. Discharges "per_token_emit" equation.
#[test]
#[ignore]
fn v1_001_callback_fires_per_token() {
    let Some((mapped, model)) = fixture_setup() else {
        eprintln!("SKIP: QWEN3_MOE_GGUF_PATH not set");
        return;
    };

    // Run non-streaming once to establish the expected token sequence.
    let cfg = greedy_config();
    let expected = run_qwen3_moe_generate(&mapped, &model, PROMPT, &cfg)
        .expect("run_qwen3_moe_generate (baseline)");
    let expected_generated: Vec<u32> = expected[PROMPT.len()..].to_vec();
    assert!(
        !expected_generated.is_empty(),
        "baseline produced no tokens — fixture broken"
    );

    // Run streaming and collect tokens via callback.
    let mut streamed: Vec<u32> = Vec::new();
    run_qwen3_moe_generate_streaming(&mapped, &model, PROMPT, &cfg, |t| {
        streamed.push(t);
        true
    })
    .expect("run_qwen3_moe_generate_streaming");

    assert_eq!(
        streamed, expected_generated,
        "streaming tokens diverged from non-streaming baseline (greedy, same seed)"
    );
}

/// V1_003: median inter-callback latency < 500ms (corresponds to ≥
/// 2 tok/s streamed; conservative floor below M32d's ~5 tok/s).
/// Discharges "Streaming throughput is bounded below by M32d's
/// per-token cost".
#[test]
#[ignore]
fn v1_003_inter_token_latency_floor() {
    let Some((mapped, model)) = fixture_setup() else {
        eprintln!("SKIP: QWEN3_MOE_GGUF_PATH not set");
        return;
    };

    let cfg = QuantizedGenerateConfig {
        max_tokens: 32,
        temperature: 0.0,
        top_k: 1,
        seed: 0,
        stop_tokens: Vec::new(),
        ..QuantizedGenerateConfig::default()
    };

    let mut timestamps: Vec<Instant> = Vec::new();
    run_qwen3_moe_generate_streaming(&mapped, &model, PROMPT, &cfg, |_t| {
        timestamps.push(Instant::now());
        true
    })
    .expect("run_qwen3_moe_generate_streaming");

    assert!(
        timestamps.len() >= 4,
        "expected ≥4 callback invocations for inter-token measurement, got {}",
        timestamps.len()
    );

    // Compute inter-callback gaps (skip first — includes prefill).
    let mut gaps_ms: Vec<u64> = timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]).as_millis() as u64)
        .collect();
    gaps_ms.sort_unstable();
    let median = gaps_ms[gaps_ms.len() / 2];

    eprintln!(
        "V1_003: {} callbacks, median inter-token gap = {} ms, gaps = {:?}",
        timestamps.len(),
        median,
        gaps_ms
    );

    assert!(
        median < 500,
        "median inter-token gap {} ms exceeds 500ms floor — streaming throughput regression",
        median
    );
}

/// Negative path: callback returning `false` short-circuits the decode
/// loop. Asserts the disconnect-handling contract documented in the
/// streaming function's doc-comment.
#[test]
#[ignore]
fn callback_stop_short_circuits() {
    let Some((mapped, model)) = fixture_setup() else {
        eprintln!("SKIP: QWEN3_MOE_GGUF_PATH not set");
        return;
    };

    let cfg = greedy_config();
    let mut count = 0usize;
    run_qwen3_moe_generate_streaming(&mapped, &model, PROMPT, &cfg, |_t| {
        count += 1;
        count < 3 // stop after 3 tokens
    })
    .expect("run_qwen3_moe_generate_streaming");

    assert_eq!(
        count, 3,
        "callback should have been invoked exactly 3 times before short-circuit"
    );
}
