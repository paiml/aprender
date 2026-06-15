//! PMAT-OXIDE-Q4K-001: cuda-oxide Q4K GEMV backend parity tests.
//!
//! 3-way bit-exact-within-Q4K-eps parity:
//!   oxide-PTX  vs  TiledQ4KGemv hand-PTX  vs  CPU `fused_q4k_parallel_matvec`.
//!
//! The oxide kernel is a pre-generated `.target sm_121` PTX asset, so these
//! tests are GATED on compute capability >= 120 (Blackwell sm_120+). On sm_89
//! (RTX 4090) or any non-Blackwell device they return early (skip) so CI on
//! non-Blackwell hosts is not failed.
//!
//! Run on gx10 (GB10, sm_121):
//!   cargo test -p aprender-serve --features cuda --test q4k_gemv_oxide_parity \
//!       -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]
#![allow(clippy::needless_range_loop)]

use realizar::cuda::CudaExecutor;
use realizar::quantize::{fused_q4k_parallel_matvec, QK_K};

/// Q4_K super-block size in bytes: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144
const Q4K_BLOCK_BYTES: usize = 144;

/// Minimum compute capability for the oxide backend (Blackwell sm_120+).
const OXIDE_MIN_CC: u32 = 120;

/// Build synthetic Q4_K weights (mirrors tests/q4k_gemv_parity.rs).
/// d=1.0, dmin=0.0, scales=1, mins=0, all quantized nibbles=1.
fn create_test_q4k_weights(out_dim: usize, in_dim: usize) -> Vec<u8> {
    assert!(
        in_dim.is_multiple_of(QK_K),
        "in_dim must be multiple of 256"
    );

    let super_blocks_per_row = in_dim / QK_K;
    let row_bytes = super_blocks_per_row * Q4K_BLOCK_BYTES;
    let total_bytes = out_dim * row_bytes;

    let mut data = vec![0u8; total_bytes];

    for row in 0..out_dim {
        for sb in 0..super_blocks_per_row {
            let sb_offset = row * row_bytes + sb * Q4K_BLOCK_BYTES;

            // d = 1.0 (f16 0x3C00)
            data[sb_offset] = 0x00;
            data[sb_offset + 1] = 0x3C;
            // dmin = 0.0
            data[sb_offset + 2] = 0x00;
            data[sb_offset + 3] = 0x00;
            // scales[0..3] = 1, mins[0..3] = 0
            for i in 0..4 {
                data[sb_offset + 4 + i] = 1;
                data[sb_offset + 4 + 4 + i] = 0;
            }
            // packed scales[8..11] low nibble = 1
            for i in 0..4 {
                data[sb_offset + 4 + 8 + i] = 0x01;
            }
            // qs[0..127] = 0x11 (both nibbles = 1)
            for i in 0..128 {
                data[sb_offset + 16 + i] = 0x11;
            }
        }
    }
    data
}

/// Q4K GPU-vs-CPU tolerance band. The GPU dequant (f16 scales -> f32) differs
/// from the CPU reference at the LSB, so the per-element error is bounded by a
/// COMBINED absolute + relative band: |a - b| <= abs_tol + rel_tol * |b|.
///
/// Q4_K quantization is ~5% element-wise (per realizar CLAUDE.md), and on
/// near-zero elements the catastrophic-cancellation absolute floor dominates.
/// The load-bearing check is oxide-vs-TILED (both GPU, same dequant) which is
/// bit-exact; this band is only the looser GPU-vs-CPU sanity bound.
fn q4k_cpu_within(a: f32, b: f32) -> bool {
    let abs_tol = 5e-2; // near-zero floor (f16 dequant LSB scatter)
    let rel_tol = 6e-2; // ~6% Q4K element-wise (5% quant + f16 headroom)
    (a - b).abs() <= abs_tol + rel_tol * b.abs()
}

/// oxide-vs-tiled tolerance: both are GPU kernels doing the SAME Q4K dequant,
/// so they must agree to near bit-exactness. Allow only f32-rounding-order eps.
fn oxide_tiled_within(a: f32, b: f32) -> bool {
    let abs_tol = 1e-4;
    let rel_tol = 1e-4;
    (a - b).abs() <= abs_tol + rel_tol * b.abs()
}

/// Returns the executor + its compute capability, or `None` if CUDA init fails.
fn make_executor() -> Option<CudaExecutor> {
    match CudaExecutor::new(0) {
        Ok(exec) => Some(exec),
        Err(e) => {
            eprintln!("CUDA init failed (cannot run oxide parity test): {e:?}");
            None
        },
    }
}

/// Skip helper: returns the device cc, or None (with a printed skip) if the
/// device is not Blackwell sm_120+.
fn require_blackwell(exec: &CudaExecutor) -> Option<u32> {
    let cc = exec.compute_capability();
    if cc < OXIDE_MIN_CC {
        eprintln!(
            "SKIP: oxide Q4K backend requires sm_120+ (cc>={OXIDE_MIN_CC}); device cc={cc}. \
             This is expected on RTX 4090 (sm_89)."
        );
        None
    } else {
        Some(cc)
    }
}

#[test]
fn test_oxide_module_loads_and_entry_resolves() {
    // Smoke: the embedded PTX must load and the `q4k_matvec` entry must resolve.
    // Driven indirectly through a 1-row GEMV; a load/entry failure surfaces as
    // an Err here. Skips gracefully on non-Blackwell.
    let Some(mut exec) = make_executor() else {
        return;
    };
    if require_blackwell(&exec).is_none() {
        return;
    }

    let in_dim = 256;
    let out_dim = 1;
    let weights = create_test_q4k_weights(out_dim, in_dim);
    let input = vec![1.0f32; in_dim];
    let mut out = vec![0.0f32; out_dim];

    exec.q4k_gemv_oxide(&weights, &input, &mut out, out_dim as u32, in_dim as u32)
        .expect("oxide PTX must load + launch (entry `q4k_matvec`)");
    assert!(out[0].is_finite(), "oxide output must be finite");
    println!("oxide module loaded; out[0]={}", out[0]);
}

#[test]
fn test_oxide_vs_cpu_parity_synthetic() {
    let Some(mut exec) = make_executor() else {
        return;
    };
    if require_blackwell(&exec).is_none() {
        return;
    }

    let in_dim = 512; // 2 super-blocks
    let out_dim = 8;
    let weights = create_test_q4k_weights(out_dim, in_dim);
    let input: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) / 6.0).collect();

    let cpu = fused_q4k_parallel_matvec(&weights, &input, in_dim, out_dim).expect("cpu q4k");

    // Existing GPU path (tiled hand-PTX) — the load-bearing parity reference.
    let mut tiled = vec![0.0f32; out_dim];
    exec.q4k_gemv(&weights, &input, &mut tiled, out_dim as u32, in_dim as u32)
        .expect("tiled q4k");

    let mut oxide = vec![0.0f32; out_dim];
    exec.q4k_gemv_oxide(&weights, &input, &mut oxide, out_dim as u32, in_dim as u32)
        .expect("oxide q4k");

    println!("=== Oxide vs Tiled vs CPU (synthetic, in={in_dim}, out={out_dim}) ===");
    for i in 0..out_dim {
        println!(
            "  [{i}] CPU={:.6} TILED={:.6} OXIDE={:.6}",
            cpu[i], tiled[i], oxide[i]
        );
        // Load-bearing: oxide must match the existing GPU kernel near-exactly.
        assert!(
            oxide_tiled_within(oxide[i], tiled[i]),
            "oxide/tiled mismatch at {i}: oxide={}, tiled={}",
            oxide[i],
            tiled[i]
        );
        // Sanity: oxide within Q4K GPU-vs-CPU band of the CPU reference.
        assert!(
            q4k_cpu_within(oxide[i], cpu[i]),
            "oxide/cpu out of Q4K band at {i}: oxide={}, cpu={}",
            oxide[i],
            cpu[i]
        );
    }
}

#[test]
fn test_oxide_three_way_parity_real_weights() {
    use realizar::gguf::{MappedGGUFModel, GGUF_TYPE_Q4_K};
    use std::path::Path;

    let Some(mut exec) = make_executor() else {
        return;
    };
    if require_blackwell(&exec).is_none() {
        return;
    }

    // Try a few candidate model paths (gx10 may store models elsewhere).
    let candidates = [
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "/tmp/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    ];
    let Some(model_path) = candidates.iter().find(|p| Path::new(p).exists()) else {
        eprintln!(
            "SKIP: no Q4_K GGUF model found in candidate paths; synthetic test covers parity"
        );
        return;
    };

    let mapped = MappedGGUFModel::from_path(model_path).expect("open GGUF");
    let data = mapped.data();
    let tensor_info = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.qtype == GGUF_TYPE_Q4_K && t.name.contains("attn_q"))
        .expect("Q4_K attn_q tensor");

    let out_dim = tensor_info.dims[0] as usize;
    let in_dim = tensor_info.dims[1] as usize;
    let super_blocks_per_row = in_dim / QK_K;
    let tensor_size = out_dim * super_blocks_per_row * Q4K_BLOCK_BYTES;
    let tensor_start = mapped.model.tensor_data_start + tensor_info.offset as usize;
    let weight_data = &data[tensor_start..tensor_start + tensor_size];

    let input: Vec<f32> = (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();

    // CPU reference
    let cpu = fused_q4k_parallel_matvec(weight_data, &input, in_dim, out_dim).expect("cpu");

    // Tiled hand-PTX (existing path) via the sync q4k_gemv
    let mut tiled = vec![0.0f32; out_dim];
    exec.q4k_gemv(
        weight_data,
        &input,
        &mut tiled,
        out_dim as u32,
        in_dim as u32,
    )
    .expect("tiled gpu");

    // Oxide PTX
    let mut oxide = vec![0.0f32; out_dim];
    exec.q4k_gemv_oxide(
        weight_data,
        &input,
        &mut oxide,
        out_dim as u32,
        in_dim as u32,
    )
    .expect("oxide gpu");

    println!(
        "=== 3-way parity: {} ({}x{}) ===",
        tensor_info.name, out_dim, in_dim
    );
    let check: Vec<usize> = (0..10.min(out_dim))
        .chain(out_dim.saturating_sub(10)..out_dim)
        .collect();

    let mut max_abs_ot = 0.0f32;
    for &i in &check {
        let abs_ot = (oxide[i] - tiled[i]).abs();
        max_abs_ot = max_abs_ot.max(abs_ot);
        println!(
            "  [{i}] CPU={:.5} TILED={:.5} OXIDE={:.5} |oxide-tiled|={:.3e}",
            cpu[i], tiled[i], oxide[i], abs_ot
        );
        // LOAD-BEARING regression: oxide must match the existing GPU tiled
        // kernel near bit-exactly (both do identical Q4K dequant on the GPU).
        assert!(
            oxide_tiled_within(oxide[i], tiled[i]),
            "oxide/tiled mismatch at {i}: oxide={}, tiled={}, |diff|={}",
            oxide[i],
            tiled[i],
            abs_ot
        );
        // Sanity: oxide within the Q4K GPU-vs-CPU band of the CPU reference.
        // (Near-zero elements have large *relative* but tiny *absolute* error;
        // the combined band handles both, matching the existing tiled kernel.)
        assert!(
            q4k_cpu_within(oxide[i], cpu[i]),
            "oxide/cpu out of Q4K band at {i}: oxide={}, cpu={}",
            oxide[i],
            cpu[i]
        );
    }
    println!("max |oxide-tiled| = {max_abs_ot:.3e} (must be ~0: oxide ≡ tiled on GPU)");
}

#[test]
fn test_oxide_state_isolation() {
    // Run the same GEMV twice; identical output proves y is zeroed each launch
    // (no atomic-add accumulation leak across calls).
    let Some(mut exec) = make_executor() else {
        return;
    };
    if require_blackwell(&exec).is_none() {
        return;
    }

    let in_dim = 256;
    let out_dim = 4;
    let weights = create_test_q4k_weights(out_dim, in_dim);
    let input: Vec<f32> = (0..in_dim).map(|i| (i % 7) as f32 * 0.25).collect();

    let mut r1 = vec![0.0f32; out_dim];
    let mut r2 = vec![0.0f32; out_dim];
    exec.q4k_gemv_oxide(&weights, &input, &mut r1, out_dim as u32, in_dim as u32)
        .expect("oxide run 1");
    exec.q4k_gemv_oxide(&weights, &input, &mut r2, out_dim as u32, in_dim as u32)
        .expect("oxide run 2");

    for i in 0..out_dim {
        assert_eq!(
            r1[i], r2[i],
            "state leak at {i}: first={}, second={} (y not zeroed?)",
            r1[i], r2[i]
        );
    }
    println!("state isolation OK: {:?}", r1);
}
