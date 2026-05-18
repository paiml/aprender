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
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
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
    assert!(
        bytes.len() >= 12,
        "stage file < 12-byte header: {}",
        path.display()
    );
    assert_eq!(
        &bytes[0..4],
        b"APRT",
        "magic must be APRT: {}",
        path.display()
    );
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

/// PR-3e probe plan: capture BOTH MoeRouter (post-softmax + renormalize top-k
/// weights, `[k=8]` per layer) AND MoeFfnOut (aggregated FFN output,
/// `[hidden_dim]` per layer). Used by `falsify_qw3_moe_l47_router_probe`
/// to disambiguate H(ii) routing-divergence from post-routing divergence at L47.
fn make_router_and_ffn_out_plan(output_dir: PathBuf) -> SaveTensorPlan {
    SaveTensorPlan::from_cli(
        "moe_router,moe_ffn_out",
        &format!("0..{EXPECTED_NUM_LAYERS}"),
        output_dir,
    )
    .expect("MoeRouter+MoeFfnOut plan from_cli must succeed for layer range 0..48")
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

        let (cpu_layer_hdr, cpu_vec) = read_stage_file(&cpu_path).unwrap_or_else(|e| {
            panic!("read CPU layer {layer_idx} ({}): {e:?}", cpu_path.display())
        });
        let (gpu_layer_hdr, gpu_vec) = read_stage_file(&gpu_path).unwrap_or_else(|e| {
            panic!("read GPU layer {layer_idx} ({}): {e:?}", gpu_path.display())
        });

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
            if *cos < COSINE_THRESHOLD {
                "  <-- BELOW 0.99"
            } else {
                ""
            }
        );
    }

    assert!(
        violators.is_empty(),
        "FALSIFY-QW3-MOE-PER-LAYER-001: {} layer(s) below cos≥{COSINE_THRESHOLD}: {violators:?}",
        violators.len()
    );
}

/// FALSIFY-QW3-MOE-L47-ROUTER-PROBE — disambiguates H(ii) routing divergence
/// from post-routing divergence at L47.
///
/// Context: PR-3 hardware verification (2026-05-17 lambda-vector RTX 4090)
/// showed 47/48 layers cos ≥ 0.99 for `MoeFfnOut` but L47 alone at cos=0.961.
/// PR-3d falsified H(i) qtype-mismatch (L0/L46/L47 identical shapes + qtypes).
/// PR-3e tests the remaining dominant hypothesis H(ii): the L47 cliff is
/// driven by **MoE expert routing divergence** — by L47 the CPU-vs-GPU
/// hidden state has drifted by ~0.002, and at L47 that drift straddles a
/// top-k expert boundary, causing CPU and GPU to pick different expert sets.
///
/// ## What this probe asserts
///
/// Captures both `MoeRouter` (post-softmax + renormalize top-k weights,
/// `[k=8]` per layer) and `MoeFfnOut` (aggregated FFN output, `[hidden_dim]`
/// per layer) for both CPU and GPU forwards. Then prints both per-layer
/// cosine vectors side-by-side and isolates L47's behavior.
///
/// ## How to interpret
///
/// - **If `MoeRouter` cos at L47 ≈ 1.0** (e.g. > 0.995) AND `MoeFfnOut` at
///   L47 ≈ 0.961: H(ii) is **FALSIFIED**. CPU and GPU pick the same experts
///   with near-identical weights; the L47 divergence happens AFTER routing.
///   Next investigation target: per-expert FfnSwigl pathology at L47's
///   specific input (e.g. expert weight cancellation).
///
/// - **If `MoeRouter` cos at L47 is much lower** (e.g. < 0.99) than other
///   layers' `MoeRouter`: H(ii) is alive. The router weight vectors
///   themselves diverge between CPU and GPU at L47. Need PR-3e2 to capture
///   top-k INDICES separately to confirm whether expert SETS differ (vs
///   just weights differing for the same set).
///
/// Note: `MoeRouter` saves WEIGHTS only, not INDICES. If CPU picks experts
/// `{3, 17, 45}` with weights `[0.5, 0.3, 0.2]` and GPU picks `{3, 17, 46}`
/// with weights `[0.5, 0.3, 0.2]`, the saved tensors are byte-identical.
/// So a HIGH router-cos at L47 does NOT prove identical expert sets —
/// it only proves the weight DISTRIBUTION SHAPE matches. The probe is
/// useful for ruling out H(ii) only via the negative direction (if router
/// weights diverge wildly, indices likely differ too).
///
/// ## Cascade context
///
/// - PR-3 ran the per-layer falsifier — 47/48 PASS, L47 surfaces
///   (#1583 comment-4470195446).
/// - PR-3b shipped contract v1.7.0 → v1.7.1 (#1739).
/// - PR-3c shipped scope-doc update + L47 sub-cascade (#1740).
/// - PR-3d falsified H(i) qtype-mismatch (#1583 comment-4470216021).
/// - **PR-3e (this PR)** probes H(ii) via MoeRouter weight cos.
/// - PR-3e2 (follow-up if needed): persist top-k INDICES via new
///   `SaveTensorStage::MoeRouterIndices` variant.
/// - PR-3f+ : fix L47 based on PR-3e/PR-3e2 outcome.
#[test]
#[ignore = "requires cached Qwen3-Coder-30B-A3B-Instruct-Q4_K_M GGUF + CUDA RTX 4090; takes ~5 min"]
fn falsify_qw3_moe_l47_router_probe() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "FALSIFY-QW3-MOE-L47-ROUTER-PROBE: skipped — no cached Qwen3-Coder GGUF in {CANONICAL_QWEN3_CODER_GGUF_PATHS:?}"
        );
        return;
    };

    eprintln!("FALSIFY-QW3-MOE-L47-ROUTER-PROBE: per-layer MoeRouter+MoeFfnOut cos for H(ii) disambiguation");
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

    let tmpdir = tempfile::tempdir().expect("create tempdir");
    let cpu_dir = tmpdir.path().join("cpu");
    let gpu_dir = tmpdir.path().join("gpu");
    let cpu_plan = make_router_and_ffn_out_plan(cpu_dir.clone());
    let gpu_plan = make_router_and_ffn_out_plan(gpu_dir.clone());

    // ----- CPU traced forward -----
    let cpu_model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #1");
    eprintln!("FALSIFY-QW3-MOE-L47-ROUTER-PROBE: running CPU traced forward...");
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

    // ----- GPU traced forward -----
    let gpu_inner =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #2");
    let mut gpu_model = OwnedQuantizedModelCuda::new(gpu_inner, 0)
        .expect("OwnedQuantizedModelCuda::new(model, 0) must succeed on RTX 4090");
    eprintln!("FALSIFY-QW3-MOE-L47-ROUTER-PROBE: running GPU traced forward...");
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

    // ----- per-layer cos for BOTH stages -----
    let mut router_cos: Vec<(usize, f32)> = Vec::with_capacity(EXPECTED_NUM_LAYERS);
    let mut ffn_out_cos: Vec<(usize, f32)> = Vec::with_capacity(EXPECTED_NUM_LAYERS);

    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        let cpu_r_path = cpu_plan.stage_path(SaveTensorStage::MoeRouter, layer_idx as u32);
        let gpu_r_path = gpu_plan.stage_path(SaveTensorStage::MoeRouter, layer_idx as u32);
        let cpu_o_path = cpu_plan.stage_path(SaveTensorStage::MoeFfnOut, layer_idx as u32);
        let gpu_o_path = gpu_plan.stage_path(SaveTensorStage::MoeFfnOut, layer_idx as u32);

        let (_, cpu_r) = read_stage_file(&cpu_r_path).expect("read CPU MoeRouter");
        let (_, gpu_r) = read_stage_file(&gpu_r_path).expect("read GPU MoeRouter");
        let (_, cpu_o) = read_stage_file(&cpu_o_path).expect("read CPU MoeFfnOut");
        let (_, gpu_o) = read_stage_file(&gpu_o_path).expect("read GPU MoeFfnOut");

        router_cos.push((layer_idx, cosine_similarity(&cpu_r, &gpu_r)));
        ffn_out_cos.push((layer_idx, cosine_similarity(&cpu_o, &gpu_o)));
    }

    eprintln!("FALSIFY-QW3-MOE-L47-ROUTER-PROBE: per-layer cos (router | ffn_out):");
    eprintln!("  L## | MoeRouter | MoeFfnOut");
    for ((idx, rcos), (_, ocos)) in router_cos.iter().zip(ffn_out_cos.iter()) {
        let marker = if *ocos < COSINE_THRESHOLD {
            "  <-- FfnOut BELOW 0.99"
        } else if *rcos < COSINE_THRESHOLD {
            "  <-- Router BELOW 0.99"
        } else {
            ""
        };
        eprintln!("  L{idx:02} | {rcos:.6}  | {ocos:.6}{marker}");
    }

    // Specifically focus on L47:
    let (_, l47_rcos) = router_cos[47];
    let (_, l47_ocos) = ffn_out_cos[47];
    eprintln!(
        "FALSIFY-QW3-MOE-L47-ROUTER-PROBE: L47 router_cos={l47_rcos:.6} ffn_out_cos={l47_ocos:.6}"
    );

    let h2_falsified = l47_rcos > 0.995 && l47_ocos < COSINE_THRESHOLD;
    let h2_alive = l47_rcos < COSINE_THRESHOLD;

    eprintln!(
        "FALSIFY-QW3-MOE-L47-ROUTER-PROBE: verdict — H(ii) routing-divergence: {}",
        if h2_falsified {
            "FALSIFIED (router weights agree, divergence is post-routing — investigate per-expert FfnSwigl at L47)"
        } else if h2_alive {
            "STILL ALIVE (router weights themselves diverge — PR-3e2 should capture indices to confirm SET divergence)"
        } else {
            "INCONCLUSIVE (intermediate state — router cos > 0.99 but ≤ 0.995; index boundary cases possible)"
        }
    );

    // This is a PROBE, not a hard-fail falsifier. Print the verdict; do not
    // assert. The verdict drives the next PR's investigation target.
}

/// (indices, u32 cast to f32 on emit, reinterpreted as u32 on read) +
/// MoeFfnOut. The triple is the minimal set needed to confirm or falsify
/// H(ii) expert-set divergence at L47.
fn make_router_indices_plan(output_dir: PathBuf) -> SaveTensorPlan {
    SaveTensorPlan::from_cli(
        "moe_router,moe_router_indices,moe_ffn_out",
        &format!("0..{EXPECTED_NUM_LAYERS}"),
        output_dir,
    )
    .expect("MoeRouter+MoeRouterIndices+MoeFfnOut plan from_cli must succeed for layer range 0..48")
}

/// Read a `MoeRouterIndices` stage file. Indices were stored as f32 cast
/// from u32 in the emit path (`forward_qwen3_moe_traced.rs`,
/// `forward_qwen3_moe_cuda_traced.rs`). Reinterpret them back to u32 here.
/// The cast is lossless for any expert id < 2^24; Qwen3 has 128 experts so
/// we never hit the limit.
fn read_indices_stage_file(path: &Path) -> std::io::Result<(u32, Vec<u32>)> {
    let (layer, f32s) = read_stage_file(path)?;
    let indices: Vec<u32> = f32s.iter().map(|f| *f as u32).collect();
    Ok((layer, indices))
}

///
/// ```
/// cargo test --release --features cuda \
///   -p aprender-serve --test qwen3_moe_per_layer_gpu_parity \
///   falsify_qw3_moe_l47_router_indices \
///   -- --ignored --nocapture
/// ```
///
/// Hardware: RTX 4090 (sm_89), GGUF cached at `/home/noah/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`.
#[test]
#[ignore = "requires cached Qwen3-Coder-30B-A3B-Instruct-Q4_K_M GGUF + CUDA RTX 4090; takes ~5 min"]
fn falsify_qw3_moe_l47_router_indices() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!(
            "FALSIFY-QW3-MOE-L47-ROUTER-INDICES: skipped — no cached Qwen3-Coder GGUF in {CANONICAL_QWEN3_CODER_GGUF_PATHS:?}"
        );
        return;
    };

    eprintln!(
        "FALSIFY-QW3-MOE-L47-ROUTER-INDICES: definitive H(ii) verdict via top-k INDICES capture"
    );
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

    let tmpdir = tempfile::tempdir().expect("create tempdir");
    let cpu_dir = tmpdir.path().join("cpu");
    let gpu_dir = tmpdir.path().join("gpu");
    let cpu_plan = make_router_indices_plan(cpu_dir.clone());
    let gpu_plan = make_router_indices_plan(gpu_dir.clone());

    // ----- CPU traced forward -----
    let cpu_model =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #1");
    eprintln!("FALSIFY-QW3-MOE-L47-ROUTER-INDICES: running CPU traced forward...");
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

    // ----- GPU traced forward -----
    let gpu_inner =
        OwnedQuantizedModel::from_mapped(&mapped).expect("OwnedQuantizedModel::from_mapped #2");
    let mut gpu_model = OwnedQuantizedModelCuda::new(gpu_inner, 0)
        .expect("OwnedQuantizedModelCuda::new(model, 0) must succeed on RTX 4090");
    eprintln!("FALSIFY-QW3-MOE-L47-ROUTER-INDICES: running GPU traced forward...");
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

    // ----- compare top-k INDICES per layer -----
    let mut layer_diverges: Vec<(usize, Vec<u32>, Vec<u32>)> = Vec::new();

    for layer_idx in 0..EXPECTED_NUM_LAYERS {
        let cpu_path = cpu_plan.stage_path(SaveTensorStage::MoeRouterIndices, layer_idx as u32);
        let gpu_path = gpu_plan.stage_path(SaveTensorStage::MoeRouterIndices, layer_idx as u32);

        let (_, mut cpu_indices) =
            read_indices_stage_file(&cpu_path).expect("read CPU MoeRouterIndices");
        let (_, mut gpu_indices) =
            read_indices_stage_file(&gpu_path).expect("read GPU MoeRouterIndices");

        // Sort to compare as SETS (top-k order is descending by weight, which
        // can differ even when sets are identical due to floating-point ties).
        cpu_indices.sort_unstable();
        gpu_indices.sort_unstable();

        if cpu_indices != gpu_indices {
            layer_diverges.push((layer_idx, cpu_indices, gpu_indices));
        }
    }

    eprintln!("FALSIFY-QW3-MOE-L47-ROUTER-INDICES: per-layer expert-set comparison:");
    if layer_diverges.is_empty() {
        eprintln!("  All 48 layers: CPU expert SET == GPU expert SET");
    } else {
        for (idx, cpu_ids, gpu_ids) in &layer_diverges {
            let cpu_only: Vec<u32> = cpu_ids
                .iter()
                .filter(|x| !gpu_ids.contains(x))
                .copied()
                .collect();
            let gpu_only: Vec<u32> = gpu_ids
                .iter()
                .filter(|x| !cpu_ids.contains(x))
                .copied()
                .collect();
            eprintln!(
                "  L{idx:02} DIVERGE — cpu_only={cpu_only:?} gpu_only={gpu_only:?} cpu={cpu_ids:?} gpu={gpu_ids:?}"
            );
        }
    }

    // Specifically L47 — the cliff layer:
    let l47_cpu = cpu_plan.stage_path(SaveTensorStage::MoeRouterIndices, 47);
    let l47_gpu = gpu_plan.stage_path(SaveTensorStage::MoeRouterIndices, 47);
    let (_, mut cpu_l47) = read_indices_stage_file(&l47_cpu).expect("read CPU L47");
    let (_, mut gpu_l47) = read_indices_stage_file(&l47_gpu).expect("read GPU L47");
    cpu_l47.sort_unstable();
    gpu_l47.sort_unstable();

    eprintln!("FALSIFY-QW3-MOE-L47-ROUTER-INDICES: L47 verdict:");
    eprintln!("  cpu sorted top-{EXPECTED_K}: {cpu_l47:?}");
    eprintln!("  gpu sorted top-{EXPECTED_K}: {gpu_l47:?}");
    if cpu_l47 == gpu_l47 {
        eprintln!(
            "  H(ii) FALSIFIED — CPU and GPU pick the SAME 8 experts at L47 with router cos=0.9926"
        );
        eprintln!(
            "  → L47 cliff is POST-ROUTING. Next investigation: per-expert FfnSwigl capture at L47."
        );
    } else {
        let cpu_only: Vec<u32> = cpu_l47
            .iter()
            .filter(|x| !gpu_l47.contains(x))
            .copied()
            .collect();
        let gpu_only: Vec<u32> = gpu_l47
            .iter()
            .filter(|x| !cpu_l47.contains(x))
            .copied()
            .collect();
        eprintln!("  H(ii) CONFIRMED — CPU expert_only={cpu_only:?} GPU expert_only={gpu_only:?}");
        eprintln!(
            "  → L47 cliff is ROUTING DIVERGENCE. Fix space: deterministic tie-breaking | fp64 gate softmax | reorder-stable top-k."
        );
    }

    // This is a PROBE — print the verdict, do not assert. The verdict
    // drives PR-3f+ (fix selection).
}
