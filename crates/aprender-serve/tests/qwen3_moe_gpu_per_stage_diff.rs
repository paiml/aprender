//! M-MOE-SUB-3 — heavy CPU-vs-GPU per-stage diff for `apr trace --save-tensor`
//! on Qwen3-Coder-30B-A3B MoE GGUF.
//!
//! Contract: [`contracts/trace-moe-gpu-sub-stages-v1.yaml`] v1.2.0 step
//! M-MOE-SUB-3.
//!
//! ## Why
//!
//! `qwen3_moe_gpu_parity::falsify_qw3_moe_gpu_parity_001_cosine_vs_cpu`
//! observes 100% NaN at lm_head from `forward_qwen3_moe_cuda` while CPU
//! `forward_qwen3_moe` produces finite output on the same input.
//! Steps 1-9 of the GPU forward are CPU-identical (precondition checks,
//! embedding, attention norm, QKV, RoPE, attention, output projection,
//! FFN norm). The only GPU-specific stage is the MoE FFN
//! (step 10 — `moe_ffn_forward_layer_cuda`).
//!
//! This heavy diagnostic test runs CPU-traced + GPU-traced forward bodies
//! (M-MOE-SUB-2 step (a) + (b)) with a `SaveTensorPlan` capturing
//! `MoeRouter` and `MoeFfnOut` for every layer, then computes per-layer
//! per-stage cosine similarity to **identify the first layer where the
//! GPU diverges from the CPU**. That layer is the M-GPU-MOE-1.4 bug-
//! origin candidate.
//!
//! ## Heavy-test layout
//!
//! 1. The 17.3 GB `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` weights,
//!    mmap'd. Cached on lambda-vector at the paths in
//!    `CANONICAL_QWEN3_CODER_GGUF_PATHS` (skip-if-not-present pattern).
//! 2. Requires CUDA (RTX 4090 on lambda-vector). Gated behind
//!    `#[cfg(feature = "cuda")]` and `#[ignore]`.
//!
//! Invocation:
//!     cargo test -p aprender-serve --test qwen3_moe_gpu_per_stage_diff \
//!         --features cuda -- --include-ignored --nocapture
//!
//! ## What the test does
//!
//! 1. Loads the GGUF once (single mmap), builds `moe_layers` once.
//! 2. Builds 2 CPU `OwnedQuantizedModel` instances (one for each path).
//! 3. Builds CPU `SaveTensorPlan` writing to `/tmp/moe-sub-cpu-<pid>/` for
//!    stages `MoeRouter,MoeFfnOut` over all 48 layers.
//! 4. Runs `forward_qwen3_moe_traced_with_plan` with that plan.
//! 5. Builds GPU `OwnedQuantizedModelCuda`, GPU `SaveTensorPlan` writing
//!    to `/tmp/moe-sub-gpu-<pid>/`.
//! 6. Runs `forward_qwen3_moe_cuda_traced_with_plan` with that plan.
//! 7. For each (layer, stage) pair, reads both files, computes cosine
//!    similarity, classifies as `MATCH ≥0.99 / DIVERGE / NAN_GPU /
//!    NAN_CPU / MISSING`.
//! 8. Prints a layer-by-layer table; identifies first divergent layer.
//!
//! ## What the test does NOT do
//!
//! - Does NOT assert pass criteria. This is a diagnostic harness for
//!   M-GPU-MOE-1.4 bisection — the assertion lives at the qwen3-moe-
//!   forward-gpu-v1 level (FALSIFY-QW3-MOE-GPU-PARITY-001 cosine
//!   threshold; existing test in `qwen3_moe_gpu_parity.rs`). Once the
//!   bug is fixed, the assertion can be added back here as
//!   FALSIFY-MOE-SUB-002 byte-identity check.
//! - Does NOT clean up `/tmp/moe-sub-*` dirs (operator inspects them
//!   for raw bytes if cosine is ambiguous).

#![cfg(feature = "cuda")]

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};
use realizar::inference_trace::save_tensor::read_tensor_file;
use realizar::inference_trace::save_tensor_plan::SaveTensorPlan;

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/cache/apr-home/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

const EXPECTED_NUM_LAYERS: usize = 48;
const EXPECTED_INTERMEDIATE: usize = 768;
const EXPECTED_N_EXPERTS: usize = 128;
const EXPECTED_K: usize = 8;

const CANONICAL_PROMPT_TOKENS: &[u32] = &[785, 9217, 308];

const MATCH_COSINE: f32 = 0.99;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::NAN;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum LayerVerdict {
    Match,
    Diverge,
    NanGpu,
    NanCpu,
    NanBoth,
    Missing,
}

fn read_or_none(path: &Path) -> Option<Vec<f32>> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let (_header, values) = read_tensor_file(&mut reader).ok()?;
    Some(values)
}

fn classify_pair(cpu: &Option<Vec<f32>>, gpu: &Option<Vec<f32>>) -> (LayerVerdict, Option<f32>) {
    let cpu = match cpu {
        Some(v) => v,
        None => return (LayerVerdict::Missing, None),
    };
    let gpu = match gpu {
        Some(v) => v,
        None => return (LayerVerdict::Missing, None),
    };
    let cpu_has_nan = cpu.iter().any(|v| !v.is_finite());
    let gpu_has_nan = gpu.iter().any(|v| !v.is_finite());
    if cpu_has_nan && gpu_has_nan {
        return (LayerVerdict::NanBoth, None);
    }
    if cpu_has_nan {
        return (LayerVerdict::NanCpu, None);
    }
    if gpu_has_nan {
        return (LayerVerdict::NanGpu, None);
    }
    let cos = cosine_similarity(cpu, gpu);
    if cos >= MATCH_COSINE {
        (LayerVerdict::Match, Some(cos))
    } else {
        (LayerVerdict::Diverge, Some(cos))
    }
}

#[test]
#[ignore]
fn falsify_moe_sub_002_cpu_gpu_traced_per_stage_diff() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "M-MOE-SUB-3 heavy diff: skipped — no cached Qwen3-Coder GGUF in {CANONICAL_QWEN3_CODER_GGUF_PATHS:?}"
        );
        return;
    };

    eprintln!("M-MOE-SUB-3 heavy diff: CPU-vs-GPU per-stage at MoeRouter + MoeFfnOut");
    eprintln!("  gguf:       {gguf_path}");
    eprintln!("  prompt:     {CANONICAL_PROMPT_TOKENS:?}");
    eprintln!("  layers:     0..{EXPECTED_NUM_LAYERS}");
    eprintln!("  stages:     moe_router,moe_ffn_out");

    let pid = std::process::id();
    let cpu_dir = PathBuf::from(format!("/tmp/moe-sub-cpu-{pid}"));
    let gpu_dir = PathBuf::from(format!("/tmp/moe-sub-gpu-{pid}"));
    eprintln!("  cpu_dir:    {}", cpu_dir.display());
    eprintln!("  gpu_dir:    {}", gpu_dir.display());

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("mmap GGUF");
    let data = mapped.data();

    let mut moe_layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        moe_layers.push(
            load_qwen3_moe_layer(&mapped.model, data, layer_idx)
                .unwrap_or_else(|e| panic!("layer {layer_idx} MoE load failed: {e:?}")),
        );
    }

    let layer_range_str = format!("0..{EXPECTED_NUM_LAYERS}");
    let cpu_plan =
        SaveTensorPlan::from_cli("moe_router,moe_ffn_out", &layer_range_str, cpu_dir.clone())
            .expect("cpu SaveTensorPlan from_cli");
    let gpu_plan =
        SaveTensorPlan::from_cli("moe_router,moe_ffn_out", &layer_range_str, gpu_dir.clone())
            .expect("gpu SaveTensorPlan from_cli");

    let cpu_model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #1");
    eprintln!("  running CPU traced forward (a few minutes)...");
    let cpu_start = std::time::Instant::now();
    let _cpu_trace = cpu_model
        .forward_qwen3_moe_traced_with_plan(
            CANONICAL_PROMPT_TOKENS,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
            Some(&cpu_plan),
        )
        .expect("CPU traced forward should succeed (CPU MoE is finite per §40)");
    eprintln!("  cpu_elapsed = {:?}", cpu_start.elapsed());

    let gpu_inner_model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #2");
    let mut gpu_model = OwnedQuantizedModelCuda::new(gpu_inner_model, 0)
        .expect("OwnedQuantizedModelCuda::new(model, 0) on RTX 4090");
    eprintln!("  running GPU traced forward...");
    let gpu_start = std::time::Instant::now();
    let _gpu_trace = gpu_model
        .forward_qwen3_moe_cuda_traced_with_plan(
            CANONICAL_PROMPT_TOKENS,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
            Some(&gpu_plan),
        )
        .expect("GPU traced forward should reach lm_head (logits may be NaN; per-stage capture is what we're after)");
    eprintln!("  gpu_elapsed = {:?}", gpu_start.elapsed());

    eprintln!();
    eprintln!("layer | moe_router (cos / verdict)        | moe_ffn_out (cos / verdict)");
    eprintln!("------|-----------------------------------|-----------------------------------");

    let mut first_diverge_router: Option<usize> = None;
    let mut first_diverge_ffn_out: Option<usize> = None;
    let mut first_nan_router: Option<usize> = None;
    let mut first_nan_ffn_out: Option<usize> = None;

    for layer in 0..EXPECTED_NUM_LAYERS {
        let cpu_router_path = cpu_dir
            .join(format!("layer-{layer}"))
            .join("moe_router.bin");
        let gpu_router_path = gpu_dir
            .join(format!("layer-{layer}"))
            .join("moe_router.bin");
        let cpu_ffn_out_path = cpu_dir
            .join(format!("layer-{layer}"))
            .join("moe_ffn_out.bin");
        let gpu_ffn_out_path = gpu_dir
            .join(format!("layer-{layer}"))
            .join("moe_ffn_out.bin");

        let (router_verdict, router_cos) = classify_pair(
            &read_or_none(&cpu_router_path),
            &read_or_none(&gpu_router_path),
        );
        let (ffn_out_verdict, ffn_out_cos) = classify_pair(
            &read_or_none(&cpu_ffn_out_path),
            &read_or_none(&gpu_ffn_out_path),
        );

        let router_str = match (router_verdict, router_cos) {
            (LayerVerdict::Match, Some(c)) => format!("{c:.6} MATCH"),
            (LayerVerdict::Diverge, Some(c)) => format!("{c:.6} DIVERGE"),
            (v, _) => format!("            {v:?}"),
        };
        let ffn_out_str = match (ffn_out_verdict, ffn_out_cos) {
            (LayerVerdict::Match, Some(c)) => format!("{c:.6} MATCH"),
            (LayerVerdict::Diverge, Some(c)) => format!("{c:.6} DIVERGE"),
            (v, _) => format!("            {v:?}"),
        };

        eprintln!("L{layer:02}   | {router_str:33} | {ffn_out_str}");

        if first_diverge_router.is_none() && router_verdict == LayerVerdict::Diverge {
            first_diverge_router = Some(layer);
        }
        if first_diverge_ffn_out.is_none() && ffn_out_verdict == LayerVerdict::Diverge {
            first_diverge_ffn_out = Some(layer);
        }
        if first_nan_router.is_none() && router_verdict == LayerVerdict::NanGpu {
            first_nan_router = Some(layer);
        }
        if first_nan_ffn_out.is_none() && ffn_out_verdict == LayerVerdict::NanGpu {
            first_nan_ffn_out = Some(layer);
        }
    }

    eprintln!();
    eprintln!("M-MOE-SUB-3 bisection summary:");
    eprintln!("  first DIVERGE on moe_router  : {first_diverge_router:?}");
    eprintln!("  first DIVERGE on moe_ffn_out : {first_diverge_ffn_out:?}");
    eprintln!("  first NaN_GPU on moe_router  : {first_nan_router:?}");
    eprintln!("  first NaN_GPU on moe_ffn_out : {first_nan_ffn_out:?}");
    eprintln!();
    eprintln!(
        "If first_NaN_GPU(moe_router) == 0: bug is in the F32 router @ hidden CPU dot product (unlikely)."
    );
    eprintln!(
        "If first_NaN_GPU(moe_ffn_out) == 0 and moe_router @ L0 is finite: bug is in expert_swiglu_cuda."
    );
    eprintln!(
        "If first_NaN_GPU(moe_ffn_out) > 0 and earlier layers MATCH: bug is layer-N specific (rare)."
    );
}

#[test]
fn classify_pair_match_when_cosine_above_threshold() {
    let cpu = Some(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let gpu = Some(vec![0.99_f32, 0.01, 0.0, 0.0]);
    let (verdict, cos) = classify_pair(&cpu, &gpu);
    assert_eq!(verdict, LayerVerdict::Match);
    assert!(cos.unwrap() >= MATCH_COSINE);
}

#[test]
fn classify_pair_diverge_when_cosine_below_threshold() {
    let cpu = Some(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let gpu = Some(vec![0.0_f32, 1.0, 0.0, 0.0]);
    let (verdict, _cos) = classify_pair(&cpu, &gpu);
    assert_eq!(verdict, LayerVerdict::Diverge);
}

#[test]
fn classify_pair_nan_gpu() {
    let cpu = Some(vec![1.0_f32, 2.0, 3.0]);
    let gpu = Some(vec![f32::NAN, 0.0, 0.0]);
    let (verdict, _cos) = classify_pair(&cpu, &gpu);
    assert_eq!(verdict, LayerVerdict::NanGpu);
}

#[test]
fn classify_pair_nan_cpu() {
    let cpu = Some(vec![f32::NAN, 0.0, 0.0]);
    let gpu = Some(vec![1.0_f32, 2.0, 3.0]);
    let (verdict, _cos) = classify_pair(&cpu, &gpu);
    assert_eq!(verdict, LayerVerdict::NanCpu);
}

#[test]
fn classify_pair_nan_both() {
    let cpu = Some(vec![f32::NAN, 0.0]);
    let gpu = Some(vec![f32::INFINITY, 0.0]);
    let (verdict, _cos) = classify_pair(&cpu, &gpu);
    assert_eq!(verdict, LayerVerdict::NanBoth);
}

#[test]
fn classify_pair_missing_when_either_none() {
    let some = Some(vec![1.0_f32, 2.0]);
    let none = None;
    assert_eq!(classify_pair(&none, &some).0, LayerVerdict::Missing);
    assert_eq!(classify_pair(&some, &none).0, LayerVerdict::Missing);
}
