//! BEAT-UNSLOTH-COLDSTART-SPEED — Pillar-3 (Unsloth) speed beat. **NIGHTLY ONLY.**
//! (DRAFT / uncommitted scout artifact — PMAT-XXX, measured 2026-06-15.)
//!
//! ## The honest win
//! For a ONE-SHOT LoRA-adapter operation invoked from the shell (init a rank-r
//! PEFT adapter over a model's attention projections and write the standard
//! adapter_config.json + adapter_model.safetensors), apr is a pure-Rust STATIC
//! BINARY whose whole process is ~few ms, while the Unsloth stack pays MANY
//! seconds just to `import unsloth` (which transitively imports torch +
//! transformers + peft + triton and runs the unsloth monkey-patches) before any
//! adapter work begins. Measured END-TO-END PROCESS wall-clock on
//! noah-Lambda-Vector (48-core x86, CPU, warm uv cache, median of 5 + warmup):
//!
//!   apr   full process (cold-start + LoRA-init + PEFT export): ~3-8 ms
//!   `import unsloth` only (no work):                           ~7077 ms
//!   torch+transformers+peft import only (the stack Unsloth wraps): ~2947 ms
//!
//! => apr is ~900-2000x faster than `import unsloth`'s startup alone, and
//!    ~400-900x faster than the torch+peft stack import alone — BEFORE either
//!    incumbent has done one byte of adapter work. apr does the FULL adapter op
//!    in that window.
//!
//! ## Why this is HONEST and host-independent (startup, not algorithm)
//! This is a STARTUP-COST beat, identical in spirit to the shipped
//! beat_pytorch_coldstart_speed.rs (~1500x). apr CONCEDES in-loop QLoRA
//! fine-tune THROUGHPUT on GPU (Unsloth's Triton-fused kernels + bitsandbytes
//! win the per-step decode/backward race — see docs/BEATS.md Pillar-3 CONCEDED).
//! apr's wedge here is the cold-start of a one-shot CLI invocation: the
//! static-binary advantage is architecture-independent, so it is robust across
//! CI hosts (unlike bandwidth-bound elementwise beats). It pairs with the two
//! shipped Pillar-3 CORRECTNESS beats — NF4≡bitsandbytes (PMAT-745) and LoRA
//! merge forward-equivalence (PMAT-747) — to make the full claim: apr replaces
//! Unsloth's QLoRA *pipeline* correctly AND starts up 100s-1000s× faster for
//! one-shot CLI use, conceding only raw GPU in-loop throughput.
//!
//! ## Why a ratio, measured same-host/same-run
//! Time apr AND the incumbent on the SAME host, SAME run; gate the ratio
//! apr_ms / incumbent_ms. The gate ceiling is set conservatively at 0.10 (apr
//! must stay >= 10x faster) so CI host variance / a faster future torch import
//! cannot trip it, but a regression that loses apr's static-binary cold-start
//! advantage (e.g. a heavy startup dep creeping into the binary) would fail.
//!
//! Run:
//!   cargo test -p aprender-train --test beat_unsloth_coldstart_speed -- --ignored --nocapture

#![cfg(test)]

use std::process::Command;
use std::time::Instant;

const RUNS: usize = 5;
/// apr must be at least 10x faster than the incumbent stack startup (ratio = apr/incumbent).
const RATIO_CEILING: f64 = 0.10;

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The actual one-shot LoRA-adapter workload, run in the re-exec'd child process.
/// Builds a small rank-8 PEFT adapter over the four attention projections of a
/// toy model and writes adapter_config.json + adapter_model.safetensors — the
/// standard PEFT bundle that `peft.PeftModel.from_pretrained()` can load.
fn apr_lora_workload() {
    use entrenar::lora::{LoRAConfig, LoRALayer, PeftAdapterBundle};
    use entrenar::Tensor;

    let d = 256usize; // toy attention proj dims
    let rank = 8usize;
    let alpha = 16.0f32;
    let config = LoRAConfig::new(rank, alpha).target_attention_projections();

    let mut bundle = PeftAdapterBundle::new(config).with_base_model("toy/model");
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        let base = Tensor::zeros(d * d, false);
        let layer = LoRALayer::new(base, d, d, rank, alpha);
        bundle.add_adapter(format!("model.layers.0.self_attn.{proj}"), &layer);
    }

    let out = std::env::temp_dir().join("apr_unsloth_beat_adapter");
    bundle.save_peft(&out).expect("apr PEFT export");
    assert!(out.join("adapter_config.json").exists());
    assert!(out.join("adapter_model.safetensors").exists());
}

/// Time apr's OWN process end-to-end by re-exec'ing the test binary in a
/// "work only" mode. The child does the full one-shot LoRA-init + PEFT export
/// and exits; we time the whole child process (the cold-start a user pays).
fn time_apr_process(self_exe: &std::path::Path) -> f64 {
    let run = || {
        Command::new(self_exe)
            .env("APR_UNSLOTH_COLDSTART_CHILD", "1")
            .output()
            .expect("re-exec apr child")
    };
    let _ = run(); // warmup
    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let out = run();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        assert!(out.status.success(), "apr child failed");
    }
    median(&times)
}

/// Time the incumbent stack's cold-start as a full process.
/// Primary: `import unsloth` (the headline Unsloth startup). Robust fallback:
/// torch+transformers+peft import (the stack Unsloth wraps), used when unsloth
/// is not installable on a CPU CI host. Both are measured the SAME way; the test
/// records which incumbent was used.
fn time_incumbent_process() -> (f64, &'static str) {
    let try_stack = |args: &[&str]| -> Option<f64> {
        let run = || Command::new("uv").args(args).output();
        // probe
        match run() {
            Ok(o) if o.status.success() => {}
            _ => return None,
        }
        let _ = run(); // warmup
        let mut times = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let t = Instant::now();
            let out = run().expect("incumbent run");
            times.push(t.elapsed().as_secs_f64() * 1000.0);
            if !out.status.success() {
                return None;
            }
        }
        Some(median(&times))
    };

    // Primary: import unsloth
    if let Some(ms) = try_stack(&["run", "--with", "unsloth", "python3", "-c", "import unsloth"]) {
        return (ms, "import unsloth");
    }
    // Fallback: the torch+peft+transformers stack Unsloth wraps
    let ms = try_stack(&[
        "run",
        "--with",
        "peft",
        "--with",
        "transformers",
        "--with",
        "torch",
        "python3",
        "-c",
        "import torch, transformers, peft",
    ])
    .expect("neither unsloth nor torch+peft installable via uv (nightly-only beat needs uv)");
    (ms, "import torch+transformers+peft (unsloth-wrapped stack)")
}

#[test]
#[ignore = "nightly-only: needs uv + (unsloth | torch+peft) (beat-speed-nightly.yml)"]
fn beat_unsloth_coldstart_speed() {
    // Child mode: do the LoRA work and exit so the parent can time our whole process.
    if std::env::var("APR_UNSLOTH_COLDSTART_CHILD").is_ok() {
        apr_lora_workload();
        return;
    }
    let self_exe = std::env::current_exe().expect("current_exe");
    let apr_ms = time_apr_process(&self_exe);
    let (incumbent_ms, incumbent) = time_incumbent_process();

    let ratio = apr_ms / incumbent_ms;
    let speedup = incumbent_ms / apr_ms;
    eprintln!(
        "BEAT-UNSLOTH-COLDSTART-SPEED: apr={apr_ms:.3}ms incumbent[{incumbent}]={incumbent_ms:.1}ms \
         ratio={ratio:.5} (apr {speedup:.0}x faster), one-shot rank-8 4-proj PEFT adapter, median of {RUNS}"
    );

    assert!(
        ratio <= RATIO_CEILING,
        "FALSIFY-BEAT-UNSLOTH-COLDSTART-SPEED: apr/incumbent ratio {ratio:.5} > {RATIO_CEILING:.2} \
         — apr lost its static-binary cold-start advantage for one-shot LoRA-adapter ops \
         (apr={apr_ms:.3}ms, incumbent[{incumbent}]={incumbent_ms:.1}ms)"
    );
}
