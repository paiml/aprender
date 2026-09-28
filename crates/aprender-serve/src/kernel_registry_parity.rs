//! #4539 KREG-001 AC-3: kernel parity receipts — the measurement a row's `tolerance` points at.
//!
//! A receipt runs one registered CPU kernel on seeded random super-blocks and compares every output
//! with an f64 dot product over the SAME weights dequantized by the crate's `dequantize_*`. That
//! oracle is IN-TREE: it shares the block decoding with the kernel, so it catches the fused/SIMD dot
//! drifting from the scalar decode, not a decode that is wrong in both. The receipt says so
//! (`oracle_independent: false`); an independent llama.cpp oracle is the follow-up.
//!
//! Two tests:
//! - `emit_parity_receipts` (ignored) measures and writes the receipts. Run it on the host whose
//!   arch the receipt admits: `KREG_RECEIPT_OUT=evidence/kreg/parity KREG_GIT_SHA=<sha> cargo test
//!   -p aprender-serve --lib emit_parity_receipts -- --ignored`.
//! - `committed_parity_receipts_hold_on_this_host` re-measures every committed receipt with its own
//!   seed and workload and refuses an error above the receipt's `tolerance_rel`, so a receipt is a
//!   falsifiable claim about the code in the tree, not a file that was once true.

use super::*;
use crate::quantize::{
    dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, fused_q4k_parallel_matvec,
    fused_q5k_parallel_matvec, fused_q6k_parallel_matvec, with_fp32_activations, QK_K,
};
use rand::{Rng, SeedableRng};

pub(super) const SCHEMA: &str = "kernel-parity-receipt/v1";
const ORACLE: &str = "in_tree_dequant_f64";

/// The kernels this harness can measure, with their super-block size and the entry point the
/// registry names. A kernel absent here has no receipt path yet.
struct Kernel {
    id: &'static str,
    source_fn: &'static str,
    block_bytes: usize,
    /// Fill one super-block with random quants and a finite, positive scale.
    fill: fn(&mut rand::rngs::StdRng, &mut [u8]),
    dequant: fn(&[u8]) -> Result<Vec<f32>>,
    matvec: fn(&[u8], &[f32], usize, usize) -> Result<Vec<f32>>,
}

fn f16_bytes(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_bits().to_le_bytes()
}

/// Random quants; `d`/`dmin` at `[0..4]`, the layout Q4_K and Q5_K share.
fn fill_dmin_first(rng: &mut rand::rngs::StdRng, b: &mut [u8]) {
    rng.fill(&mut b[4..]);
    b[0..2].copy_from_slice(&f16_bytes(rng.random_range(0.002..0.02)));
    b[2..4].copy_from_slice(&f16_bytes(rng.random_range(0.0..0.01)));
}

/// Random quants and signed scales; `d` is the trailing f16 at `[208..210]`.
fn fill_q6_k(rng: &mut rand::rngs::StdRng, b: &mut [u8]) {
    rng.fill(&mut b[..208]);
    b[208..210].copy_from_slice(&f16_bytes(rng.random_range(0.0005..0.005)));
}

const KERNELS: &[Kernel] = &[
    Kernel {
        id: "cpu.matvec.q4_k",
        source_fn: "fused_q4k_parallel_matvec",
        block_bytes: 144,
        fill: fill_dmin_first,
        dequant: dequantize_q4_k,
        matvec: fused_q4k_parallel_matvec,
    },
    Kernel {
        id: "cpu.matvec.q5_k",
        source_fn: "fused_q5k_parallel_matvec",
        block_bytes: 176,
        fill: fill_dmin_first,
        dequant: dequantize_q5_k,
        matvec: fused_q5k_parallel_matvec,
    },
    Kernel {
        id: "cpu.matvec.q6_k",
        source_fn: "fused_q6k_parallel_matvec",
        block_bytes: 210,
        fill: fill_q6_k,
        dequant: dequantize_q6_k,
        matvec: fused_q6k_parallel_matvec,
    },
];

fn kernel(id: &str) -> &'static Kernel {
    KERNELS
        .iter()
        .find(|k| k.id == id)
        .unwrap_or_else(|| panic!("no parity harness for {id}"))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Workload {
    in_dim: usize,
    out_dim: usize,
    seed: u64,
    trials: usize,
}

/// K spans several super-blocks, so a per-block scale or offset bug shows; `out_dim` is past the
/// sequential/parallel split, so the rayon fan-out is the path measured.
const WORKLOAD: Workload = Workload {
    in_dim: 8 * QK_K,
    out_dim: 256,
    seed: 4539,
    trials: 4,
};

/// Worst error over every output of every trial, relative to the trial's largest |reference|.
/// Normalising by the row max, not per element, keeps a near-zero output from reading as 100%.
#[derive(Debug, Default, Clone, Copy)]
struct Measured {
    max_abs_err: f64,
    max_rel_err: f64,
}

fn measure(k: &Kernel, w: Workload, fp32_activations: bool) -> Measured {
    let mut rng = rand::rngs::StdRng::seed_from_u64(w.seed);
    let blocks_per_row = w.in_dim / QK_K;
    let mut m = Measured::default();
    for _ in 0..w.trials {
        let mut weights = vec![0u8; w.out_dim * blocks_per_row * k.block_bytes];
        for block in weights.chunks_exact_mut(k.block_bytes) {
            (k.fill)(&mut rng, block);
        }
        let x: Vec<f32> = (0..w.in_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let deq = (k.dequant)(&weights).expect("dequantize the random weights");
        let reference: Vec<f64> = deq
            .chunks_exact(w.in_dim)
            .map(|row| {
                row.iter()
                    .zip(&x)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum()
            })
            .collect();
        let run = || (k.matvec)(&weights, &x, w.in_dim, w.out_dim);
        let got = if fp32_activations {
            with_fp32_activations(run)
        } else {
            run()
        }
        .expect("run the kernel");
        assert_eq!(got.len(), w.out_dim, "{}: output length", k.id);
        let scale = reference.iter().fold(0.0f64, |a, r| a.max(r.abs()));
        assert!(
            scale > 0.0,
            "{}: an all-zero reference measures nothing",
            k.id
        );
        for (g, r) in got.iter().zip(&reference) {
            let abs = (f64::from(*g) - r).abs();
            assert!(abs.is_finite(), "{}: non-finite output", k.id);
            m.max_abs_err = m.max_abs_err.max(abs);
            m.max_rel_err = m.max_rel_err.max(abs / scale);
        }
    }
    m
}

/// Twice the measured error, rounded UP to one significant digit: a bound read off the
/// measurement with a stated margin, never typed from memory.
fn tolerance_from(measured: f64) -> f64 {
    let x = (2.0 * measured).max(f64::MIN_POSITIVE);
    let p = 10f64.powf(x.log10().floor());
    // The epsilon keeps 8e-7 / 1e-7 = 8.000…01 from ceiling to 9.
    ((x / p) - 1e-9).ceil() * p
}

fn host_arch() -> &'static str {
    std::env::consts::ARCH
}

fn isa_detected() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut v = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for (name, on) in [
            ("sse4.2", is_x86_feature_detected!("sse4.2")),
            ("avx", is_x86_feature_detected!("avx")),
            ("avx2", is_x86_feature_detected!("avx2")),
            ("fma", is_x86_feature_detected!("fma")),
            ("avx512f", is_x86_feature_detected!("avx512f")),
            ("avx512bw", is_x86_feature_detected!("avx512bw")),
            ("avx512vnni", is_x86_feature_detected!("avx512vnni")),
        ] {
            if on {
                v.push(name);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        for (name, on) in [
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            (
                "dotprod",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
        ] {
            if on {
                v.push(name);
            }
        }
    }
    v
}

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn receipt(k: &Kernel, row: &KernelRow) -> serde_json::Value {
    let served = measure(k, WORKLOAD, false);
    let fp32 = measure(k, WORKLOAD, true);
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .or_else(|| std::env::var("HOSTNAME").ok())
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .expect("a receipt names its host");
    let sha = std::env::var("KREG_GIT_SHA").expect("KREG_GIT_SHA: the commit that was measured");
    serde_json::json!({
        "schema": SCHEMA,
        "kernel_id": k.id,
        "source_fn": k.source_fn,
        "registry_precision": row.precision,
        "host": host,
        "host_arch": host_arch(),
        "isa_detected": isa_detected(),
        "build_identity": sha,
        "ts": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "oracle": ORACLE,
        "oracle_independent": false,
        "workload": {
            "in_dim": WORKLOAD.in_dim,
            "out_dim": WORKLOAD.out_dim,
            "seed": WORKLOAD.seed,
            "trials": WORKLOAD.trials,
        },
        "served": {"max_abs_err": served.max_abs_err, "max_rel_err": served.max_rel_err},
        "fp32_activations": {"max_abs_err": fp32.max_abs_err, "max_rel_err": fp32.max_rel_err},
        "tolerance_rel": tolerance_from(served.max_rel_err),
    })
}

#[test]
#[ignore = "writes receipts; run on the host whose arch it admits (see module doc)"]
fn emit_parity_receipts() {
    let out = repo_root().join(std::env::var("KREG_RECEIPT_OUT").expect("KREG_RECEIPT_OUT"));
    std::fs::create_dir_all(&out).expect("receipt dir");
    let r = registry().expect("registry");
    for k in KERNELS {
        let row = r
            .rows()
            .iter()
            .find(|row| row.kernel_id == k.id)
            .unwrap_or_else(|| panic!("{} is not a registry row", k.id));
        assert_eq!(
            row.source_fn, k.source_fn,
            "{}: harness and row name different fns",
            k.id
        );
        let doc = receipt(k, row);
        let path = out.join(format!("{}.json", k.id));
        let text = serde_json::to_string_pretty(&doc).expect("receipt json");
        std::fs::write(&path, text + "\n").expect("write receipt");
        eprintln!("{}: {}", path.display(), doc["served"]);
    }
}

fn committed() -> Vec<(serde_json::Value, serde_json::Value)> {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("../kernel-registry-receipts.json"))
            .expect("receipts json");
    doc["receipts"]
        .as_array()
        .expect("receipts")
        .iter()
        .map(|entry| {
            let path = entry["receipt"].as_str().expect("receipt path");
            let text = std::fs::read_to_string(repo_root().join(path))
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            let receipt = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
            (entry.clone(), receipt)
        })
        .collect()
}

/// FALSIFY-KREG-009: a committed receipt is the row's tolerance, agrees with its ratchet entry,
/// and still holds — its own seed and workload, re-measured here, stay within `tolerance_rel`.
#[test]
fn committed_parity_receipts_hold_on_this_host() {
    let r = registry().expect("registry");
    for (entry, rc) in committed() {
        let id = rc["kernel_id"].as_str().expect("kernel_id");
        let path = entry["receipt"].as_str().expect("receipt path");
        assert_eq!(rc["schema"], SCHEMA, "{path}: schema");
        assert_eq!(
            entry["kernel_id"], id,
            "{path}: ratchet entry names another kernel"
        );
        assert_eq!(
            entry["host_arch"], rc["host_arch"],
            "{path}: ratchet entry arch"
        );
        let row = r
            .rows()
            .iter()
            .find(|row| row.kernel_id == id)
            .unwrap_or_else(|| panic!("{path}: {id} is not a row"));
        assert_eq!(
            row.tolerance, path,
            "{id}: tolerance must point at its receipt"
        );
        assert_eq!(
            rc["registry_precision"],
            row.precision.as_str(),
            "{id}: precision drifted"
        );
        let wl = &rc["workload"];
        let dim = |k: &str| wl[k].as_u64().expect("workload field") as usize;
        let w = Workload {
            in_dim: dim("in_dim"),
            out_dim: dim("out_dim"),
            seed: wl["seed"].as_u64().expect("seed"),
            trials: dim("trials"),
        };
        let bound = rc["tolerance_rel"].as_f64().expect("tolerance_rel");
        let now = measure(kernel(id), w, false);
        assert!(
            now.max_rel_err <= bound,
            "{id}: max_rel_err {} > receipt tolerance {bound} ({path})",
            now.max_rel_err
        );
        // The row's `precision` is a claim about the served path, and the two measurements test
        // it: f32 activations serve exactly what the FP32 scope computes; a quantized precision
        // serves something measurably coarser. (#4539: q4_k declared f32 and served Q8_K.)
        let fp32 = measure(kernel(id), w, true);
        if row.precision == "f32" {
            assert!(
                now.max_rel_err <= fp32.max_rel_err,
                "{id}: precision=f32, yet the served path ({}) is coarser than FP32 ({})",
                now.max_rel_err,
                fp32.max_rel_err
            );
        } else {
            assert!(
                now.max_rel_err > 10.0 * fp32.max_rel_err,
                "{id}: precision={}, yet the served path ({}) is as exact as FP32 ({})",
                row.precision,
                now.max_rel_err,
                fp32.max_rel_err
            );
        }
    }
}

/// A row whose tolerance names a receipt has a ratchet entry for it: the path is never typed
/// without the measurement behind it.
#[test]
fn a_receipt_path_tolerance_has_a_ratchet_entry() {
    let r = registry().expect("registry");
    let entries: Vec<String> = committed()
        .iter()
        .map(|(e, _)| e["kernel_id"].as_str().expect("kernel_id").to_string())
        .collect();
    for row in r.rows() {
        if row.tolerance != "unmeasured" {
            assert!(
                entries.contains(&row.kernel_id),
                "{}: tolerance {} has no receipt entry",
                row.kernel_id,
                row.tolerance
            );
        }
    }
}

#[test]
fn tolerance_rounds_twice_the_measurement_up_to_one_digit() {
    for (m, want) in [(1.2e-4, 3e-4), (4.0e-7, 8e-7), (4.6e-3, 1e-2)] {
        let got = tolerance_from(m);
        assert!((got - want).abs() < want * 1e-9, "{m}: {got} != {want}");
    }
}
