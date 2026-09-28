//! #4539 KREG-001 AC-3: kernel parity receipts — the measurement a row's `tolerance` points at.
//!
//! A receipt runs one registered CPU kernel on seeded random super-blocks and compares every output
//! with an f64 dot product over the SAME weights dequantized by the crate's `dequantize_*`. That
//! oracle is IN-TREE: it shares the block decoding with the kernel, so it catches the fused/SIMD dot
//! drifting from the scalar decode, not a decode that is wrong in both. The receipt says so
//! (`oracle_independent: false`).
//!
//! The dequant-only and IQ rows are measured against an INDEPENDENT oracle instead: fixtures that
//! `scripts/kreg_ggufpy_oracle.py` wrote from llama.cpp's gguf-py (its own decoders and grid
//! tables) under `evidence/kreg/oracle/<TYPE>/`. Their receipts say `oracle_independent: true`.
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
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    fused_q4_0_q8_0_parallel_matvec, fused_q4k_parallel_matvec, fused_q5k_parallel_matvec,
    fused_q6k_parallel_matvec, fused_q8_0_q8_0_parallel_matvec, iq_parallel_matvec,
    quantize_activations_q8_0, with_fp32_activations, QK_K,
};
use rand::{Rng, SeedableRng};

pub(super) const SCHEMA: &str = "kernel-parity-receipt/v1";
const ORACLE: &str = "in_tree_dequant_f64";

/// The kernels this harness can measure, with their block size and the entry point the registry
/// names. A kernel absent here has no receipt path yet.
struct Kernel {
    id: &'static str,
    source_fn: &'static str,
    block_elems: usize,
    block_bytes: usize,
    workload: Workload,
    /// Fill one block with random quants and a finite, positive scale.
    fill: fn(&mut rand::rngs::StdRng, &mut [u8]),
    /// The activation quantization the kernel declares, as a round trip: the second oracle a
    /// quantized-precision row is checked against. `None` for f32 activations and for Q8_K, whose
    /// exact reference is the FP32 scope.
    act_quant: Option<fn(&[f32]) -> Vec<f32>>,
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

/// Random quants; `d` at `[0..2]`, the layout Q4_0 and Q8_0 share.
fn fill_d_first(rng: &mut rand::rngs::StdRng, b: &mut [u8]) {
    rng.fill(&mut b[2..]);
    b[0..2].copy_from_slice(&f16_bytes(rng.random_range(0.002..0.02)));
}

/// Activations through the crate's Q8_0 quantizer and back: 32-element blocks, one scale each.
fn q8_0_round_trip(x: &[f32]) -> Vec<f32> {
    let (scales, quants) = quantize_activations_q8_0(x);
    quants
        .iter()
        .enumerate()
        .take(x.len())
        .map(|(i, q)| f32::from(*q) * scales[i / 32])
        .collect()
}

/// Random quants and signed scales; `d` is the trailing f16 at `[208..210]`.
fn fill_q6_k(rng: &mut rand::rngs::StdRng, b: &mut [u8]) {
    rng.fill(&mut b[..208]);
    b[208..210].copy_from_slice(&f16_bytes(rng.random_range(0.0005..0.005)));
}

const KERNELS: &[Kernel] = &[
    Kernel {
        id: "cpu.matvec.q4_k",
        block_elems: QK_K,
        workload: WORKLOAD,
        act_quant: None,
        source_fn: "fused_q4k_parallel_matvec",
        block_bytes: 144,
        fill: fill_dmin_first,
        dequant: dequantize_q4_k,
        matvec: fused_q4k_parallel_matvec,
    },
    Kernel {
        id: "cpu.matvec.q5_k",
        block_elems: QK_K,
        workload: WORKLOAD,
        act_quant: None,
        source_fn: "fused_q5k_parallel_matvec",
        block_bytes: 176,
        fill: fill_dmin_first,
        dequant: dequantize_q5_k,
        matvec: fused_q5k_parallel_matvec,
    },
    Kernel {
        id: "cpu.matvec.q6_k",
        block_elems: QK_K,
        workload: WORKLOAD,
        act_quant: None,
        source_fn: "fused_q6k_parallel_matvec",
        block_bytes: 210,
        fill: fill_q6_k,
        dequant: dequantize_q6_k,
        matvec: fused_q6k_parallel_matvec,
    },
    Kernel {
        id: "cpu.matvec.q4_0",
        source_fn: "fused_q4_0_q8_0_parallel_matvec",
        block_elems: 32,
        block_bytes: 18,
        workload: WORKLOAD_Q8_0,
        fill: fill_d_first,
        act_quant: Some(q8_0_round_trip),
        dequant: dequantize_q4_0,
        matvec: fused_q4_0_q8_0_parallel_matvec,
    },
    Kernel {
        id: "cpu.matvec.q8_0",
        source_fn: "fused_q8_0_q8_0_parallel_matvec",
        block_elems: 32,
        block_bytes: 34,
        workload: WORKLOAD_Q8_0,
        fill: fill_d_first,
        act_quant: Some(q8_0_round_trip),
        dequant: dequantize_q8_0,
        matvec: fused_q8_0_q8_0_parallel_matvec,
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

/// K spans several super-blocks, so a per-block scale or offset bug shows; `out_dim` is at
/// `generic_parallel_matvec`'s split (256), so Q5_K and Q6_K take the rayon fan-out.
const WORKLOAD: Workload = Workload {
    in_dim: 8 * QK_K,
    out_dim: 256,
    seed: 4539,
    trials: 4,
};

/// The Q4_0/Q8_0 matvecs go parallel from 1024 rows (`PARALLEL_THRESHOLD`), the size real layers
/// run at, so that is the path measured.
const WORKLOAD_Q8_0: Workload = Workload {
    in_dim: 8 * QK_K,
    out_dim: 1024,
    seed: 4539,
    trials: 4,
};

/// Worst error over every output of every trial, relative to the trial's largest |reference|.
/// Normalising by the row max, not per element, keeps a near-zero output from reading as 100%.
#[derive(Debug, Default, Clone, Copy)]
struct Measured {
    max_abs_err: f64,
    max_rel_err: f64,
    /// The same error against an oracle fed the kernel's own quantized activations.
    quantized_act_rel_err: Option<f64>,
    /// KTEST-02 per-element margin under the row's EM-DOT model, worst over the trials.
    margin: Option<Margin>,
}

/// The worst KTEST-02 report over a measurement's trials (KTEST-001 §3.2).
#[derive(Debug, Clone, Copy, Default)]
struct Margin {
    max: f64,
    p999: f64,
    nmse: f64,
    pass: bool,
}

/// EM-DOT for these kernels: f32 accumulation over one row, stored as f32, no flush-to-zero.
fn em_dot(k: usize) -> aprender_kernel_oracle::ErrorModel {
    use aprender_kernel_oracle::Dtype;
    aprender_kernel_oracle::ErrorModel::Dot {
        k,
        acc: Dtype::F32,
        out: Dtype::F32,
        ftz: false,
    }
}

/// Judge one trial by margin. The oracle is fed the activations the kernel consumed (§0.4), so
/// quantization error is not charged to the kernel; the magnitudes are Σ|w|·|x| per output.
fn margin_of(got: &[f32], deq: &[f32], x_used: &[f32], oracle: &[f64], in_dim: usize) -> Margin {
    let mags: Vec<f64> = deq
        .chunks_exact(in_dim)
        .map(|row| {
            row.iter()
                .zip(x_used)
                .map(|(a, b)| f64::from(a.abs()) * f64::from(b.abs()))
                .sum()
        })
        .collect();
    let r = aprender_kernel_oracle::margin::judge_model(got, oracle, &mags, &em_dot(in_dim))
        .unwrap_or_else(|e| panic!("EM-DOT K={in_dim} refused: {e:?}"));
    Margin {
        max: r.max_margin,
        p999: r.p999_margin,
        nmse: r.nmse,
        pass: r.passed(),
    }
}

fn measure(k: &Kernel, w: Workload, fp32_activations: bool) -> Measured {
    let mut rng = rand::rngs::StdRng::seed_from_u64(w.seed);
    let blocks_per_row = w.in_dim / k.block_elems;
    let mut m = Measured::default();
    for _ in 0..w.trials {
        let mut weights = vec![0u8; w.out_dim * blocks_per_row * k.block_bytes];
        for block in weights.chunks_exact_mut(k.block_bytes) {
            (k.fill)(&mut rng, block);
        }
        let x: Vec<f32> = (0..w.in_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let deq = (k.dequant)(&weights).expect("dequantize the random weights");
        let oracle = |x: &[f32]| -> Vec<f64> {
            deq.chunks_exact(w.in_dim)
                .map(|row| {
                    row.iter()
                        .zip(x)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum()
                })
                .collect()
        };
        let reference = oracle(&x);
        let quantized_x = k.act_quant.map(|q| q(&x));
        let quantized_reference = quantized_x.as_ref().map(|qx| oracle(qx));
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
        let x_used = if fp32_activations {
            &x
        } else {
            quantized_x.as_ref().unwrap_or(&x)
        };
        let trial_oracle = if fp32_activations {
            &reference
        } else {
            quantized_reference.as_ref().unwrap_or(&reference)
        };
        let t = margin_of(&got, &deq, x_used, trial_oracle, w.in_dim);
        let prev = m.margin.unwrap_or(Margin {
            pass: true,
            ..Margin::default()
        });
        m.margin = Some(Margin {
            max: prev.max.max(t.max),
            p999: prev.p999.max(t.p999),
            nmse: prev.nmse.max(t.nmse),
            pass: prev.pass && t.pass,
        });
        if let Some(qr) = &quantized_reference {
            let worst = got
                .iter()
                .zip(qr)
                .fold(0.0f64, |a, (g, r)| a.max((f64::from(*g) - r).abs() / scale));
            m.quantized_act_rel_err = Some(m.quantized_act_rel_err.unwrap_or(0.0).max(worst));
        }
    }
    m
}

/// Above any f32 accumulation error at these K (measured: 2e-7..1.3e-6) and below any 8-bit
/// activation quantization (measured: 3.7e-3..5.6e-3), with an order of magnitude to each side.
const F32_CEILING: f64 = 1e-4;

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
fn receipt_header(id: &str, source_fn: &str, row: &KernelRow, oracle: &str) -> serde_json::Value {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .or_else(|| std::env::var("HOSTNAME").ok())
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .expect("a receipt names its host");
    let sha = std::env::var("KREG_GIT_SHA").expect("KREG_GIT_SHA: the commit that was measured");
    let device = std::iter::once(host_arch())
        .chain(isa_detected())
        .collect::<Vec<_>>()
        .join("+");
    let set = InputSet::from_tree(&repo_root(), row, "none", &device, oracle)
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    serde_json::json!({
        "schema": SCHEMA,
        "kernel_id": id,
        "source_fn": source_fn,
        "registry_precision": row.precision,
        "host": host,
        "host_arch": host_arch(),
        "isa_detected": isa_detected(),
        "build_identity": sha,
        "input_set": {
            "source_sha256": set.source_sha256,
            "row_sha256": set.row_sha256,
            "toolchain": set.toolchain,
            "driver": set.driver,
            "device": set.device,
            "oracle": set.oracle,
        },
        "input_set_hash": set.hash(),
        "ts": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
    })
}

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn receipt(k: &Kernel, row: &KernelRow) -> serde_json::Value {
    let served = measure(k, k.workload, false);
    let fp32 = measure(k, k.workload, true);
    let mut doc = receipt_header(k.id, k.source_fn, row, ORACLE);
    doc["oracle"] = ORACLE.into();
    doc["oracle_independent"] = false.into();
    doc["workload"] = serde_json::json!({
        "in_dim": k.workload.in_dim,
        "out_dim": k.workload.out_dim,
        "seed": k.workload.seed,
        "trials": k.workload.trials,
    });
    doc["served"] =
        serde_json::json!({"max_abs_err": served.max_abs_err, "max_rel_err": served.max_rel_err});
    doc["fp32_activations"] =
        serde_json::json!({"max_abs_err": fp32.max_abs_err, "max_rel_err": fp32.max_rel_err});
    doc["quantized_activation_oracle"] = served
        .quantized_act_rel_err
        .map(|e| serde_json::json!({"max_rel_err": e}))
        .into();
    doc["tolerance_rel"] = tolerance_from(served.max_rel_err).into();
    doc["margin"] = margin_json(served.margin);
    doc
}

/// The receipt's KTEST-02 block: the verdict is `max_margin ≤ 1` under the row's error model;
/// NMSE is recorded for comparison with llama.cpp and never decides.
fn margin_json(m: Option<Margin>) -> serde_json::Value {
    let m = m.expect("a measured receipt has a margin");
    serde_json::json!({
        "error_model": "EM-DOT",
        "acc": "f32",
        "max_margin": m.max,
        "p999_margin": m.p999,
        "nmse": m.nmse,
        "verdict": if m.pass { "pass" } else { "fail" },
    })
}

/// A row measured against the gguf-py fixtures: the kernel is run on the fixture's bytes.
struct FixtureKernel {
    id: &'static str,
    source_fn: &'static str,
    ggml_type: &'static str,
    /// `(ggml_type_id, weights, x, in_dim, out_dim) -> y`, the row's own served path.
    eval: fn(u32, &[u8], &[f32], usize, usize) -> Vec<f32>,
}

const GGUF_PY_ORACLE: &str = "gguf_py_dequant_f64";
const ORACLE_DIR: &str = "evidence/kreg/oracle";

/// The oracle part of a fixture receipt's input set: the oracle, its dir, llama.cpp commit and
/// every fixture file's sha256 (sorted by name), so a regenerated fixture makes the receipt stale.
fn fixture_oracle(ggml_type: &str, meta: &serde_json::Value) -> String {
    let shas: Vec<String> = meta["sha256"]
        .as_object()
        .map(|o| {
            let mut v: Vec<String> = o
                .iter()
                .map(|(f, h)| format!("{f}:{}", h.as_str().unwrap_or("?")))
                .collect();
            v.sort();
            v
        })
        .unwrap_or_default();
    format!(
        "{GGUF_PY_ORACLE}@{ORACLE_DIR}/{ggml_type}@{}@{}",
        meta["llama_cpp_commit"].as_str().unwrap_or("?"),
        shas.join(",")
    )
}

/// A dequant-only row serves `dequantize_*` followed by an f32 dot, row by row.
fn dequant_then_dot(
    dequant: fn(&[u8]) -> crate::error::Result<Vec<f32>>,
    w: &[u8],
    x: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let dense = dequant(w).expect("the fixture dequantizes");
    assert_eq!(dense.len(), in_dim * out_dim, "dequantized length");
    dense
        .chunks_exact(in_dim)
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn iq(t: u32, w: &[u8], x: &[f32], i: usize, o: usize) -> Vec<f32> {
    iq_parallel_matvec(t, w, x, i, o).expect("iq_parallel_matvec on the fixture")
}

macro_rules! dequant_row {
    ($id:literal, $f:ident, $t:literal) => {
        FixtureKernel {
            id: $id,
            source_fn: stringify!($f),
            ggml_type: $t,
            eval: |_, w, x, i, o| dequant_then_dot($f, w, x, i, o),
        }
    };
}

macro_rules! iq_row {
    ($id:literal, $t:literal) => {
        FixtureKernel {
            id: $id,
            source_fn: "iq_parallel_matvec_into",
            ggml_type: $t,
            eval: iq,
        }
    };
}

const FIXTURE_KERNELS: &[FixtureKernel] = &[
    dequant_row!("cpu.matvec.q4_1", dequantize_q4_1, "Q4_1"),
    dequant_row!("cpu.matvec.q5_0", dequantize_q5_0, "Q5_0"),
    dequant_row!("cpu.matvec.q5_1", dequantize_q5_1, "Q5_1"),
    dequant_row!("cpu.matvec.q2_k", dequantize_q2_k, "Q2_K"),
    dequant_row!("cpu.matvec.q3_k", dequantize_q3_k, "Q3_K"),
    iq_row!("cpu.matvec.iq2_xxs", "IQ2_XXS"),
    iq_row!("cpu.matvec.iq3_xxs", "IQ3_XXS"),
    iq_row!("cpu.matvec.iq4_nl", "IQ4_NL"),
    iq_row!("cpu.matvec.iq3_s", "IQ3_S"),
    iq_row!("cpu.matvec.iq2_s", "IQ2_S"),
    iq_row!("cpu.matvec.iq4_xs", "IQ4_XS"),
];

fn fixture_kernel(id: &str) -> &'static FixtureKernel {
    FIXTURE_KERNELS
        .iter()
        .find(|k| k.id == id)
        .unwrap_or_else(|| panic!("{id}: no fixture kernel"))
}

struct Fixture {
    meta: serde_json::Value,
    weights: Vec<u8>,
    x: Vec<f32>,
    reference: Vec<f64>,
}

/// Load a fixture and refuse one whose files are not the ones its `meta.json` hashed.
fn fixture(ggml_type: &str) -> Fixture {
    use sha2::{Digest, Sha256};
    let dir = repo_root().join(ORACLE_DIR).join(ggml_type);
    let read =
        |f: &str| std::fs::read(dir.join(f)).unwrap_or_else(|e| panic!("{ggml_type}/{f}: {e}"));
    let meta: serde_json::Value =
        serde_json::from_slice(&read("meta.json")).expect("fixture meta.json");
    assert_eq!(
        meta["oracle"], GGUF_PY_ORACLE,
        "{ggml_type}: fixture oracle"
    );
    assert_eq!(meta["ggml_type"], ggml_type, "{ggml_type}: fixture type");
    let mut load = |f: &str| {
        let b = read(f);
        assert_eq!(
            meta["sha256"][f].as_str(),
            Some(format!("{:x}", Sha256::digest(&b)).as_str()),
            "{ggml_type}/{f}: bytes are not the ones meta.json hashed"
        );
        b
    };
    let weights = load("weights.bin");
    let x = load("x.bin")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let reference = load("ref.bin")
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    Fixture {
        meta,
        weights,
        x,
        reference,
    }
}

/// The kernel's error against gguf-py, normalized by the largest |reference| as `measure` does.
fn measure_fixture(k: &FixtureKernel, f: &Fixture) -> Measured {
    let dim = |key: &str| f.meta[key].as_u64().expect("fixture dim") as usize;
    let (in_dim, out_dim) = (dim("in_dim"), dim("out_dim"));
    assert_eq!(f.x.len(), in_dim, "{}: x length", k.id);
    assert_eq!(f.reference.len(), out_dim, "{}: ref length", k.id);
    let type_id = f.meta["ggml_type_id"].as_u64().expect("ggml_type_id") as u32;
    let y = (k.eval)(type_id, &f.weights, &f.x, in_dim, out_dim);
    assert_eq!(y.len(), out_dim, "{}: output length", k.id);
    let scale = f.reference.iter().fold(0f64, |m, r| m.max(r.abs()));
    assert!(
        scale > 0.0,
        "{}: an all-zero reference measures nothing",
        k.id
    );
    let max_abs_err = y
        .iter()
        .zip(&f.reference)
        .map(|(a, r)| (f64::from(*a) - r).abs())
        .fold(0f64, f64::max);
    Measured {
        max_abs_err,
        max_rel_err: max_abs_err / scale,
        quantized_act_rel_err: None,
        margin: None,
    }
}

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn fixture_receipt(k: &FixtureKernel, row: &KernelRow) -> serde_json::Value {
    let f = fixture(k.ggml_type);
    let served = measure_fixture(k, &f);
    let m = &f.meta;
    let mut doc = receipt_header(k.id, k.source_fn, row, &fixture_oracle(k.ggml_type, m));
    doc["oracle"] = GGUF_PY_ORACLE.into();
    doc["oracle_independent"] = true.into();
    doc["oracle_fixture"] = serde_json::json!({
        "dir": format!("{ORACLE_DIR}/{}", k.ggml_type),
        "llama_cpp_commit": m["llama_cpp_commit"],
        "sha256": m["sha256"],
    });
    doc["workload"] = serde_json::json!({
        "in_dim": m["in_dim"], "out_dim": m["out_dim"], "seed": m["seed"], "trials": 1,
    });
    doc["served"] =
        serde_json::json!({"max_abs_err": served.max_abs_err, "max_rel_err": served.max_rel_err});
    doc["tolerance_rel"] = tolerance_from(served.max_rel_err).into();
    doc["margin"] = serde_json::json!({
        "verdict": "not_run",
        "reason": "the fixture has no |W|·|x| magnitudes and no reference on the activations the kernel quantizes (KTEST-001 §0.4); needs a fixture regen",
    });
    doc
}

#[test]
#[ignore = "writes receipts; run on the host whose arch it admits (see module doc)"]
fn emit_parity_receipts() {
    let out = repo_root().join(std::env::var("KREG_RECEIPT_OUT").expect("KREG_RECEIPT_OUT"));
    std::fs::create_dir_all(&out).expect("receipt dir");
    let r = registry().expect("registry");
    // KREG_EMIT=<id,id>: re-measure only these, so a new row does not re-stamp the others.
    let only = std::env::var("KREG_EMIT").ok();
    let wanted = |id: &str| {
        only.as_deref()
            .is_none_or(|o| o.split(',').any(|w| w == id))
    };
    let row_for = |id: &str, source_fn: &str| {
        let row = r
            .rows()
            .iter()
            .find(|row| row.kernel_id == id)
            .unwrap_or_else(|| panic!("{id} is not a registry row"));
        assert_eq!(
            row.source_fn, source_fn,
            "{id}: harness and row name different fns"
        );
        row
    };
    let docs = KERNELS
        .iter()
        .filter(|k| wanted(k.id))
        .map(|k| (k.id, receipt(k, row_for(k.id, k.source_fn))))
        .chain(
            FIXTURE_KERNELS
                .iter()
                .filter(|k| wanted(k.id))
                .map(|k| (k.id, fixture_receipt(k, row_for(k.id, k.source_fn)))),
        );
    for (id, doc) in docs {
        let path = out.join(format!("{id}.json"));
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
        assert!(
            rc["input_set_hash"]
                .as_str()
                .is_some_and(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit())),
            "{path}: no input_set_hash, so the release gate cannot judge it fresh"
        );
        if rc["oracle"] == GGUF_PY_ORACLE {
            assert_eq!(
                rc["margin"]["verdict"], "not_run",
                "{path}: a fixture receipt cannot claim a margin its fixture cannot support"
            );
            check_fixture_receipt(id, path, &rc, row);
            continue;
        }
        assert_eq!(rc["oracle"], ORACLE, "{path}: unknown oracle");
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
        assert_eq!(
            rc["margin"]["verdict"], "pass",
            "{path}: KTEST-02 margin verdict (max ≤ 1 under EM-DOT)"
        );
        let now_margin = now.margin.expect("measured margin");
        assert!(
            now_margin.pass,
            "{id}: max margin {} > 1 under EM-DOT on this host ({path})",
            now_margin.max
        );
        assert!(
            now.max_rel_err <= bound,
            "{id}: max_rel_err {} > receipt tolerance {bound} ({path})",
            now.max_rel_err
        );
        // The row's `precision` is a claim about the served path, and the two measurements test
        // it: f32 activations serve exactly what the FP32 scope computes; a quantized precision
        // serves something measurably coarser. (#4539: q4_k declared f32 and served Q8_K.)
        let fp32 = measure(kernel(id), w, true);
        match row.precision.as_str() {
            // Both halves: the FP32 scope reaches only Q4_K, so "no coarser than the scope" alone
            // would pass a Q8_0 kernel declared f32 (its scope run is quantized too).
            "f32" => assert!(
                now.max_rel_err <= fp32.max_rel_err && now.max_rel_err < F32_CEILING,
                "{id}: precision=f32, yet the served path ({}) is coarser than FP32 ({}) or \
                 than {F32_CEILING}",
                now.max_rel_err,
                fp32.max_rel_err
            ),
            "q8_k" => assert!(
                now.max_rel_err > 10.0 * fp32.max_rel_err,
                "{id}: precision=q8_k, yet the served path ({}) is as exact as FP32 ({})",
                now.max_rel_err,
                fp32.max_rel_err
            ),
            // Q8_0 has no FP32 scope to compare with, so the second oracle decides: the kernel
            // must track the quantized-activation oracle far closer than the f32 one.
            "q8_0" => {
                let q = now
                    .quantized_act_rel_err
                    .unwrap_or_else(|| panic!("{id}: precision=q8_0 and no act_quant oracle"));
                assert!(
                    q * 10.0 < now.max_rel_err,
                    "{id}: precision=q8_0, yet the served path is no closer to Q8_0 \
                     activations ({q}) than to f32 ones ({})",
                    now.max_rel_err
                );
            },
            p => panic!("{id}: no precision check for {p}"),
        }
    }
}

/// The gguf-py half of FALSIFY-KREG-009: the fixture is the one the receipt measured, the kernel
/// still lands within `tolerance_rel` of gguf-py, and `precision=f32` means f32-exact against it.
fn check_fixture_receipt(id: &str, path: &str, rc: &serde_json::Value, row: &KernelRow) {
    assert_eq!(
        rc["oracle_independent"], true,
        "{path}: a gguf-py receipt is independent"
    );
    let k = fixture_kernel(id);
    assert_eq!(rc["source_fn"], k.source_fn, "{path}: source_fn");
    let f = fixture(k.ggml_type);
    assert_eq!(
        rc["oracle_fixture"]["sha256"], f.meta["sha256"],
        "{path}: the fixture changed since the receipt was measured"
    );
    let bound = rc["tolerance_rel"].as_f64().expect("tolerance_rel");
    let now = measure_fixture(k, &f);
    assert!(
        now.max_rel_err <= bound,
        "{id}: max_rel_err {} > receipt tolerance {bound} against gguf-py ({path})",
        now.max_rel_err
    );
    match row.precision.as_str() {
        "f32" => assert!(
            now.max_rel_err < F32_CEILING,
            "{id}: precision=f32, yet it is {} from gguf-py (ceiling {F32_CEILING})",
            now.max_rel_err
        ),
        p => panic!("{id}: no gguf-py precision check for {p}"),
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
