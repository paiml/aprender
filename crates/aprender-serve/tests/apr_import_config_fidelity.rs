//! OBLIG-APR-IMPORT-CONFIG-FIDELITY — GGUF→APR import preserves forward-affecting config.
//!
//! GROUNDED FINDING (PMAT class, reproduced on real GB10): a converted `.apr`
//! qwen2.5-coder-1.5b model FAILS the GPU F2 per-position parity gate at pos-11
//! (argmax mismatch, cosine 0.9788 < 0.98) → silent CPU fallback ~9 tok/s, while
//! the SAME logical model as `.gguf` PASSES (min cosine 0.9972) → GPU 113 tok/s.
//!
//! ORACLE = the `.gguf` path (`GGUFConfig::from_gguf`, used by
//! `OwnedQuantizedModel::from_mapped`). The `.apr` path (`GGUFConfig::from_apr`,
//! used by `OwnedQuantizedModel::from_apr`) MUST produce the byte-identical
//! forward-affecting config. Any field that differs is the bug.
//!
//! This is a DIAGNOSTIC + FALSIFIER. The `dump` test prints every field for both
//! paths (run with `--nocapture`). The `fidelity` test asserts equality on the
//! forward/attention/RoPE fields and is the load-bearing gate.

use std::path::Path;

use realizar::apr::MappedAprModel;
use realizar::gguf::{GGUFConfig, MappedGGUFModel};

/// Candidate model paths (host-gated; auto-skip if absent).
const GGUF_CANDIDATES: &[&str] = &[
    "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    "/root/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
];

/// A freshly-converted `.apr` produced by `apr import <gguf> --preserve-q4k`.
/// The test harness writes this beside the GGUF or to a scratch dir.
const APR_CANDIDATES: &[&str] = &[
    // Fresh import produced by the CPU harness (preferred — guarantees same logical model).
    "/tmp/claude-1000/-home-noah-src-aprender/fc7c8724-5434-4eaa-a264-dca9afc15d6f/scratchpad/qwen-fresh.apr",
    "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4k.apr",
    "/root/models/qwen2.5-coder-1.5b-instruct-q4_k_m.apr",
];

fn first_existing(candidates: &[&'static str]) -> Option<&'static str> {
    candidates.iter().copied().find(|p| Path::new(p).exists())
}

fn load_gguf_config(path: &str) -> GGUFConfig {
    let mapped =
        MappedGGUFModel::from_path(path).unwrap_or_else(|e| panic!("mmap GGUF {path}: {e:?}"));
    GGUFConfig::from_gguf(&mapped.model).expect("GGUFConfig::from_gguf")
}

fn load_apr_config(path: &str) -> GGUFConfig {
    let mapped =
        MappedAprModel::from_path(path).unwrap_or_else(|e| panic!("mmap APR {path}: {e:?}"));
    let vocab_size = mapped.metadata.vocab_size.unwrap_or(0);
    GGUFConfig::from_apr(&mapped, vocab_size).expect("GGUFConfig::from_apr")
}

fn dump_config(label: &str, c: &GGUFConfig) {
    eprintln!("─── {label} ───");
    eprintln!("  architecture        = {}", c.architecture);
    eprintln!("  hidden_dim          = {}", c.hidden_dim);
    eprintln!("  num_layers          = {}", c.num_layers);
    eprintln!("  num_heads           = {}", c.num_heads);
    eprintln!("  num_kv_heads        = {}", c.num_kv_heads);
    eprintln!("  vocab_size          = {}", c.vocab_size);
    eprintln!("  intermediate_dim    = {}", c.intermediate_dim);
    eprintln!("  context_length      = {}", c.context_length);
    eprintln!("  rope_theta          = {}", c.rope_theta);
    eprintln!("  rope_type           = {}", c.rope_type);
    eprintln!("  eps                 = {:e}", c.eps);
    eprintln!("  explicit_head_dim   = {:?}", c.explicit_head_dim);
    eprintln!("  head_dim()          = {}", c.head_dim());
    eprintln!("  q_dim()             = {}", c.q_dim());
    eprintln!("  kv_dim()            = {}", c.kv_dim());
    eprintln!("  attn_scale()        = {}", c.attn_scale());
    eprintln!("  query_pre_attn_sclr = {:?}", c.query_pre_attn_scalar);
    eprintln!("  bos_token_id        = {:?}", c.bos_token_id);
    eprintln!("  eos_token_id        = {:?}", c.eos_token_id);
}

/// Diagnostic dump — run with `-- --nocapture` to pin diverging fields.
#[test]
fn dump_apr_vs_gguf_config() {
    let (Some(gguf), Some(apr)) = (
        first_existing(GGUF_CANDIDATES),
        first_existing(APR_CANDIDATES),
    ) else {
        eprintln!("[apr_import_config_fidelity] SKIP: host lacks qwen2.5-coder fixtures");
        return;
    };
    let gc = load_gguf_config(gguf);
    let ac = load_apr_config(apr);
    eprintln!("\n=== APR-vs-GGUF CONFIG DIFF ({gguf} | {apr}) ===");
    dump_config("GGUF (ORACLE)", &gc);
    dump_config("APR", &ac);
}

/// OBLIG-APR-IMPORT-CONFIG-FIDELITY — the round-tripped `.apr` config MUST equal
/// the `.gguf` (oracle) config on every forward-affecting field.
///
/// RED before the fix: `eps` (and/or `context_length`) diverges because the
/// GGUF→APR converter does not stamp `rms_norm_eps` (and other) keys into the
/// APR metadata, so `from_apr` silently falls back to an architecture default
/// that may not match the GGUF's stored value. GREEN after the converter stamps
/// the forward-affecting keys. MUTATION-VERIFY: reverting the stamp → RED.
#[test]
fn apr_import_preserves_forward_affecting_config() {
    let (Some(gguf), Some(apr)) = (
        first_existing(GGUF_CANDIDATES),
        first_existing(APR_CANDIDATES),
    ) else {
        eprintln!("[apr_import_config_fidelity] SKIP: host lacks qwen2.5-coder fixtures");
        return;
    };
    let gc = load_gguf_config(gguf);
    let ac = load_apr_config(apr);

    // Dump on every run so a failure shows both sides.
    dump_config("GGUF (ORACLE)", &gc);
    dump_config("APR", &ac);

    // Forward/attention/RoPE-affecting fields — must match the oracle exactly.
    assert_eq!(ac.architecture, gc.architecture, "architecture diverged");
    assert_eq!(ac.hidden_dim, gc.hidden_dim, "hidden_dim diverged");
    assert_eq!(ac.num_layers, gc.num_layers, "num_layers diverged");
    assert_eq!(ac.num_heads, gc.num_heads, "num_heads diverged");
    assert_eq!(ac.num_kv_heads, gc.num_kv_heads, "num_kv_heads diverged");
    assert_eq!(ac.vocab_size, gc.vocab_size, "vocab_size diverged");
    assert_eq!(
        ac.intermediate_dim, gc.intermediate_dim,
        "intermediate_dim diverged"
    );
    assert_eq!(ac.head_dim(), gc.head_dim(), "head_dim diverged");
    assert_eq!(ac.q_dim(), gc.q_dim(), "q_dim diverged");
    assert_eq!(ac.kv_dim(), gc.kv_dim(), "kv_dim diverged");
    assert_eq!(ac.rope_type, gc.rope_type, "rope_type diverged");

    // rope_theta — exact f32 equality (no quantization tolerance for a config scalar).
    assert_eq!(
        ac.rope_theta, gc.rope_theta,
        "rope_theta diverged: apr={} gguf={}",
        ac.rope_theta, gc.rope_theta
    );

    // eps — the RMSNorm epsilon feeds every layer norm. A divergence here shifts
    // every hidden state and compounds position-by-position.
    assert_eq!(
        ac.eps, gc.eps,
        "eps (rms_norm_eps) diverged: apr={:e} gguf={:e}",
        ac.eps, gc.eps
    );

    // attn_scale — 1/sqrt(d). Feeds the softmax temperature at every position.
    assert_eq!(
        ac.attn_scale(),
        gc.attn_scale(),
        "attn_scale diverged: apr={} gguf={}",
        ac.attn_scale(),
        gc.attn_scale()
    );

    // context_length — RoPE position span / max-seq. Diverging here can change
    // position-dependent scaling on long-context models.
    assert_eq!(
        ac.context_length, gc.context_length,
        "context_length diverged: apr={} gguf={}",
        ac.context_length, gc.context_length
    );
}
