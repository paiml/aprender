//! V1_001..V1_004 falsification tests for qwen3-moe-sampling-v1.yaml.
//!
//! Discharges the four falsification gates by running
//! `run_qwen3_moe_generate` against a real Qwen3-MoE GGUF with various
//! sampling configurations and asserting the expected determinism
//! invariants.
//!
//! ## Gating
//!
//! `#[ignore]` by default. Activated by:
//!
//! ```text
//! QWEN3_MOE_GGUF_PATH=/path/to/qwen3-moe.gguf \
//!   cargo test --test qwen3_moe_sampling_v1 \
//!   -p aprender-serve --features cuda --release -- --ignored --nocapture
//! ```

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
use realizar::infer::qwen3_moe_generate::run_qwen3_moe_generate;

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
const MAX_TOKENS: usize = 4;

fn generate(
    mapped: &MappedGGUFModel,
    model: &OwnedQuantizedModel,
    cfg: QuantizedGenerateConfig,
) -> Vec<u32> {
    run_qwen3_moe_generate(mapped, model, PROMPT, &cfg).expect("run_qwen3_moe_generate")
}

#[test]
#[ignore = "requires real Qwen3-MoE GGUF via QWEN3_MOE_GGUF_PATH env var"]
fn v1_001_greedy_fallback_temperature_zero_is_deterministic() {
    let Some((mapped, model)) = fixture_setup() else {
        eprintln!("SKIP: QWEN3_MOE_GGUF_PATH not set");
        return;
    };

    let mk_cfg = || QuantizedGenerateConfig {
        max_tokens: MAX_TOKENS,
        temperature: 0.0, // greedy
        top_k: 1,
        top_p: 1.0,
        seed: 42,
        stop_tokens: Vec::new(),
        ..QuantizedGenerateConfig::default()
    };

    let a = generate(&mapped, &model, mk_cfg());
    let b = generate(&mapped, &model, mk_cfg());
    let c = generate(&mapped, &model, mk_cfg());

    eprintln!("V1_001 run a: {a:?}");
    eprintln!("V1_001 run b: {b:?}");
    eprintln!("V1_001 run c: {c:?}");

    assert_eq!(a, b, "V1_001: greedy must be deterministic (a vs b)");
    assert_eq!(b, c, "V1_001: greedy must be deterministic (b vs c)");

    eprintln!("V1_001 DISCHARGED: greedy fallback (temperature=0) is deterministic.");
}

#[test]
#[ignore = "requires real Qwen3-MoE GGUF via QWEN3_MOE_GGUF_PATH env var"]
fn v1_002_temperature_positive_with_fixed_seed_is_deterministic() {
    let Some((mapped, model)) = fixture_setup() else {
        eprintln!("SKIP: QWEN3_MOE_GGUF_PATH not set");
        return;
    };

    let mk_cfg = || QuantizedGenerateConfig {
        max_tokens: MAX_TOKENS,
        temperature: 0.7,
        top_k: 50,
        top_p: 0.95,
        seed: 42,
        stop_tokens: Vec::new(),
        ..QuantizedGenerateConfig::default()
    };

    let a = generate(&mapped, &model, mk_cfg());
    let b = generate(&mapped, &model, mk_cfg());

    eprintln!("V1_002 run a (seed=42): {a:?}");
    eprintln!("V1_002 run b (seed=42): {b:?}");

    assert_eq!(
        a, b,
        "V1_002: temperature>0 + fixed seed must be deterministic"
    );

    eprintln!("V1_002 DISCHARGED: seeded RNG produces reproducible sampling.");
}

#[test]
#[ignore = "requires real Qwen3-MoE GGUF via QWEN3_MOE_GGUF_PATH env var"]
fn v1_003_temperature_positive_different_seeds_diverge() {
    let Some((mapped, model)) = fixture_setup() else {
        eprintln!("SKIP: QWEN3_MOE_GGUF_PATH not set");
        return;
    };

    let mk_cfg = |seed| QuantizedGenerateConfig {
        max_tokens: 8, // a bit more to give RNG room to diverge
        temperature: 0.9,
        top_k: 50,
        top_p: 0.95,
        seed,
        stop_tokens: Vec::new(),
        ..QuantizedGenerateConfig::default()
    };

    let a = generate(&mapped, &model, mk_cfg(42));
    let b = generate(&mapped, &model, mk_cfg(43));

    eprintln!("V1_003 seed=42: {a:?}");
    eprintln!("V1_003 seed=43: {b:?}");

    // Tokens [0..PROMPT.len()] are the prompt — same in both runs. Compare
    // only the generated portion.
    let gen_a = &a[PROMPT.len()..];
    let gen_b = &b[PROMPT.len()..];
    assert_ne!(
        gen_a, gen_b,
        "V1_003: different seeds must produce different generated tokens"
    );

    eprintln!("V1_003 DISCHARGED: different seeds produce different sampling paths.");
}

#[test]
#[ignore = "requires real Qwen3-MoE GGUF via QWEN3_MOE_GGUF_PATH env var"]
fn v1_004_top_k_one_is_greedy_regardless_of_temperature() {
    let Some((mapped, model)) = fixture_setup() else {
        eprintln!("SKIP: QWEN3_MOE_GGUF_PATH not set");
        return;
    };

    // top_k=1 + very high temperature should STILL pick argmax (greedy fallback).
    let high_temp_top_k_one = QuantizedGenerateConfig {
        max_tokens: MAX_TOKENS,
        temperature: 5.0, // would be very flat distribution if applied
        top_k: 1,         // but top_k=1 → greedy
        top_p: 1.0,
        seed: 12345,
        stop_tokens: Vec::new(),
        ..QuantizedGenerateConfig::default()
    };

    let pure_greedy = QuantizedGenerateConfig {
        max_tokens: MAX_TOKENS,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        seed: 12345,
        stop_tokens: Vec::new(),
        ..QuantizedGenerateConfig::default()
    };

    let a = generate(&mapped, &model, high_temp_top_k_one);
    let b = generate(&mapped, &model, pure_greedy);

    eprintln!("V1_004 high-temp top_k=1: {a:?}");
    eprintln!("V1_004 pure greedy:       {b:?}");

    assert_eq!(
        a, b,
        "V1_004: top_k=1 must produce greedy output regardless of temperature"
    );

    eprintln!("V1_004 DISCHARGED: top_k=1 forces greedy regardless of temperature.");
}
