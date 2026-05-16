//! FALSIFY-QW3-MOE-PER-LAYER-001 — per-layer MoE FFN output cosine between
//! CPU LAZY-FUSED-MATVEC and GPU `q6k_gemv` warp-shuffle for **M-GPU-MOE-3**
//! (issue #1583).
//!
//! ## What this test asserts
//!
//! For every decoder layer L ∈ \[0, 48) of `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`,
//! the aggregated MoE FFN output (`SaveTensorStage::MoeFfnOut`) must satisfy
//! `cos(cpu_moe_ffn_out[L], gpu_moe_ffn_out[L]) ≥ 0.99` for the same prompt.
//!
//! ## Why this falsifier exists
//!
//! Issue #1583 cites "7-8 layers (L7, L9, L12, L20, L23, L29, L46)" at
//! cos 0.94-0.987 between CPU and GPU forward — but no in-tree falsifier
//! captured that claim. The end-to-end `qwen3_moe_gpu_parity.rs` test only
//! checks the FINAL logits cos, so it can't isolate which layer first drops
//! below 0.99. This test fills that gap by using the existing
//! `forward_qwen3_moe_traced` (CPU) and `forward_qwen3_moe_cuda_traced` (GPU)
//! plumbed through `SaveTensorPlan::MoeFfnOut` to dump per-layer vectors to
//! disk, then computing per-layer cosine in-process.
//!
//! When `M-GPU-MOE-3` is closed (PTX fp64 accumulator or contiguous chunking),
//! this test asserts all 48 layers above 0.99 and the contract
//! `qwen3-moe-forward-gpu-v1` can flip v1.7.0 → v1.8.0 ACTIVE_RUNTIME.
//!
//! ## How to run
//!
//! ```
//! cargo test --release --features cuda \
//!   -p aprender-serve --test qwen3_moe_per_layer_gpu_parity \
//!   -- --ignored --nocapture
//! ```
//!
//! Skipped on non-CUDA hosts (`#![cfg(feature = "cuda")]` gate). When the
//! Qwen3-Coder-30B GGUF is absent the test logs and returns early — fixture
//! absence is not a defect (M32c.2.2.2.1.4 convention).

#![cfg(feature = "cuda")]

use realizar::gguf::qwen3_moe_load::load_qwen3_moe_layer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};
use realizar::inference_trace::save_tensor_plan::SaveTensorPlan;
use realizar::inference_trace::save_tensor_stage::SaveTensorStage;

use std::path::{Path, PathBuf};

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/cache/apr-home/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

const EXPECTED_NUM_LAYERS: usize = 48;
const EXPECTED_INTERMEDIATE: usize = 768;
const EXPECTED_N_EXPERTS: usize = 128;
const EXPECTED_K: usize = 8;

const COSINE_THRESHOLD: f32 = 0.99;

/// 1-token prompt keeps the forward pass O(minutes) instead of O(tens-of-
/// minutes) for the 30B Q4_K_M weight set. Per-layer divergence is shape-
/// dependent on weights × activations, both of which are present at seq=1.
const PROMPT_TOKENS: &[u32] = &[785];

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine: vectors must be same length");
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)) as f32
}

/// Read a save-tensor file emitted by `maybe_save_stage`. File format:
/// 4 bytes "APRT" magic + 4 bytes layer (LE u32) + 4 bytes dim (LE u32)
/// + dim × 4 bytes f32 (LE).
fn read_stage_file(path: &Path) -> std::io::Result<(u32, Vec<f32>)> {
    let bytes = std::fs::read(path)?;
    assert!(bytes.len() >= 12, "stage file < 12-byte header: {}", path.display());
    assert_eq!(&bytes[0..4], b"APRT", "magic must be APRT: {}", path.display());
    let layer = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let dim = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    assert_eq!(
        bytes.len(),
        12 + dim * 4,
        "stage file body length mismatch: {}",
        path.display()
    );
    let mut values = Vec::with_capacity(dim);
    for chunk in bytes[12..].chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((layer, values))
}

fn make_moe_ffn_out_plan(output_dir: PathBuf) -> SaveTensorPlan {
    SaveTensorPlan::from_cli(
        "moe_ffn_out",
        &format!("0..{EXPECTED_NUM_LAYERS}"),
        output_dir,
    )
    .expect("MoeFfnOut plan from_cli must succeed for layer range 0..48")
}

/// FALSIFY-QW3-MOE-PER-LAYER-001 — per-layer MoE FFN output cos ≥ 0.99
/// between CPU LAZY-FUSED-MATVEC and GPU `q6k_gemv` warp-shuffle.
#[test]
#[ignore = "requires cached Qwen3-Coder-30B-A3B-Instruct-Q4_K_M GGUF + CUDA RTX 4090; takes ~5 min"]
fn falsify_qw3_moe_per_layer_001_cosine_per_layer() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "FALSIFY-QW3-MOE-PER-LAYER-001: skipped — no cached Qwen3-Coder GGUF in {CANONICAL_QWEN3_CODER_GGUF_PATHS:?}"
        );
        return;
    };

    eprintln!("FALSIFY-QW3-MOE-PER-LAYER-001: per-layer MoeFfnOut cos ≥ {COSINE_THRESHOLD}");
    eprintln!("  gguf:   {gguf_path}");
    eprintln!("  prompt: {PROMPT_TOKENS:?}");

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("mmap GGUF");
    let data = mapped.data();

    let mut moe_layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        moe_layers.push(
            load_qwen3_moe_layer(&mapped.model, data, layer_idx)
                .unwrap_or_else(|e| panic!("layer {layer_idx} MoE load failed: {e:?}")),
        );
    }

    // Separate output dirs for CPU and GPU dumps so per-layer files don't
    // collide on disk. Both plans capture only MoeFfnOut across all layers.
    let tmpdir = tempfile::tempdir().expect("create tempdir");
    let cpu_dir = tmpdir.path().join("cpu");
    let gpu_dir = tmpdir.path().join("gpu");
    let cpu_plan = make_moe_ffn_out_plan(cpu_dir.clone());
    let gpu_plan = make_moe_ffn_out_plan(gpu_dir.clone());

    // ----- CPU traced forward -----
    let cpu_model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #1");
    eprintln!("FALSIFY-QW3-MOE-PER-LAYER-001: running CPU traced forward (a few minutes)...");
    let cpu_start = std::time::Instant::now();
    let _cpu_trace = cpu_model
        .forward_qwen3_moe_traced_with_plan(
            PROMPT_TOKENS,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
            Some(&cpu_plan),
        )
        .expect("CPU traced forward must succeed");
    let cpu_elapsed = cpu_start.elapsed();
    eprintln!("FALSIFY-QW3-MOE-PER-LAYER-001: CPU forward done in {cpu_elapsed:?}");

    // ----- GPU traced forward -----
    let gpu_inner =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #2");
    let mut gpu_model = OwnedQuantizedModelCuda::new(gpu_inner, 0)
        .expect("OwnedQuantizedModelCuda::new(model, 0) must succeed on RTX 4090");
    eprintln!("FALSIFY-QW3-MOE-PER-LAYER-001: running GPU traced forward...");
    let gpu_start = std::time::Instant::now();
    let _gpu_trace = gpu_model
        .forward_qwen3_moe_cuda_traced_with_plan(
            PROMPT_TOKENS,
            &moe_layers,
            EXPECTED_N_EXPERTS,
            EXPECTED_K,
            EXPECTED_INTERMEDIATE,
            data,
            Some(&gpu_plan),
        )
        .expect("GPU traced forward must succeed");
    let gpu_elapsed = gpu_start.elapsed();
    eprintln!("FALSIFY-QW3-MOE-PER-LAYER-001: GPU forward done in {gpu_elapsed:?}");

    // ----- per-layer cos -----
    let mut per_layer_cos: Vec<(usize, f32)> = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    let mut violators: Vec<(usize, f32)> = Vec::new();

    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        let cpu_path = cpu_plan.stage_path(SaveTensorStage::MoeFfnOut, layer_idx as u32);
        let gpu_path = gpu_plan.stage_path(SaveTensorStage::MoeFfnOut, layer_idx as u32);

        let (cpu_layer_hdr, cpu_vec) = read_stage_file(&cpu_path)
            .unwrap_or_else(|e| panic!("read CPU layer {layer_idx} ({}): {e:?}", cpu_path.display()));
        let (gpu_layer_hdr, gpu_vec) = read_stage_file(&gpu_path)
            .unwrap_or_else(|e| panic!("read GPU layer {layer_idx} ({}): {e:?}", gpu_path.display()));

        assert_eq!(
            cpu_layer_hdr as usize, layer_idx,
            "CPU file header layer mismatch for L{layer_idx}"
        );
        assert_eq!(
            gpu_layer_hdr as usize, layer_idx,
            "GPU file header layer mismatch for L{layer_idx}"
        );
        assert_eq!(
            cpu_vec.len(),
            gpu_vec.len(),
            "L{layer_idx} dim mismatch: cpu={} gpu={}",
            cpu_vec.len(),
            gpu_vec.len()
        );

        let cos = cosine_similarity(&cpu_vec, &gpu_vec);
        per_layer_cos.push((layer_idx, cos));
        if cos < COSINE_THRESHOLD {
            violators.push((layer_idx, cos));
        }
    }

    eprintln!("FALSIFY-QW3-MOE-PER-LAYER-001: per-layer cos vector:");
    for (idx, cos) in &per_layer_cos {
        eprintln!(
            "  L{idx:02} cos={cos:.6}{}",
            if *cos < COSINE_THRESHOLD { "  <-- BELOW 0.99" } else { "" }
        );
    }

    assert!(
        violators.is_empty(),
        "FALSIFY-QW3-MOE-PER-LAYER-001: {} layer(s) below cos≥{COSINE_THRESHOLD}: {violators:?}",
        violators.len()
    );
}
