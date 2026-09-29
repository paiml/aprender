//! #4539 KREG-001 AC-3 for `ops[]` rows: parity receipts for the per-forward op kernels.
//!
//! An op kernel runs on f32 activations, so there is no block format to decode: the oracle is the
//! op's own definition evaluated in f64 on the same inputs (`f64_definition`), written here from
//! the formula, not from the kernel. It shares no code with the kernel, so the receipt says
//! `oracle_independent: true`.
//!
//! The error models these rows declare (EM-RED for a norm's reduction followed by a rsqrt and a
//! scale, EM-ELEM for tanh) have no implemented bound in `aprender-kernel-oracle` yet, so the
//! receipt's margin block says `not_modelled` and decides nothing. What the receipt DOES claim is
//! checked: the committed receipt re-measures within its `tolerance_rel` on this host, and
//! `precision=f32` holds below [`F32_CEILING`].
//!
//! `emit_op_parity_receipts` (ignored) writes them, like `emit_parity_receipts`:
//! `KREG_RECEIPT_OUT=evidence/kreg/parity KREG_GIT_SHA=<sha> cargo test -p aprender-serve --lib
//! emit_op_parity_receipts -- --ignored`.

use super::*;

pub(super) const OP_ORACLE: &str = "f64_definition";

/// One `ops[]` row the harness measures.
pub(super) struct Op {
    id: &'static str,
    source_fn: &'static str,
    /// The definition the f64 oracle evaluates; part of the receipt's input set.
    formula: &'static str,
    workload: OpWorkload,
    /// Draw the inputs from `rng`, run the kernel, and return its output with the f64 oracle's.
    run: fn(&mut rand::rngs::StdRng, usize) -> (Vec<f32>, Vec<f64>),
}

#[derive(Debug, Clone, Copy)]
struct OpWorkload {
    n: usize,
    seed: u64,
    trials: usize,
}

const OP_WORKLOAD: OpWorkload = OpWorkload {
    n: 896,
    seed: 0x4539_0f00,
    trials: 8,
};

const EPS: f32 = 1e-6;

fn activations(rng: &mut rand::rngs::StdRng, n: usize, r: f32) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-r..r)).collect()
}

fn gains(rng: &mut rand::rngs::StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(0.5..1.5)).collect()
}

fn run_rmsnorm(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let (x, w) = (activations(rng, n, 2.0), gains(rng, n));
    let mut got = vec![0.0f32; n];
    crate::gguf::ops::rms_norm_into(&x, &w, EPS, &mut got);
    let ms = x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / n as f64;
    let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
    let want = x
        .iter()
        .zip(&w)
        .map(|(a, g)| f64::from(*a) * inv * f64::from(*g))
        .collect();
    (got, want)
}

fn run_layernorm(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    // An offset mean, so the centring the kernel does is exercised, not a no-op on ~0-mean input.
    let x: Vec<f32> = activations(rng, n, 2.0).iter().map(|v| v + 0.75).collect();
    let (w, b) = (gains(rng, n), activations(rng, n, 0.5));
    let mut got = vec![0.0f32; n];
    crate::gguf::ops::layer_norm_into(&x, &w, Some(&b), EPS, &mut got);
    let mean = x.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;
    let var = x
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let inv = 1.0 / (var + f64::from(EPS)).sqrt();
    let want = x
        .iter()
        .zip(w.iter().zip(&b))
        .map(|(a, (g, c))| (f64::from(*a) - mean) * inv * f64::from(*g) + f64::from(*c))
        .collect();
    (got, want)
}

fn run_gelu(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let x = activations(rng, n, 6.0);
    let mut got = x.clone();
    crate::gguf::ops::gelu(&mut got);
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let want = x
        .iter()
        .map(|v| {
            let v = f64::from(*v);
            0.5 * v * (1.0 + (c * (v + 0.044_715 * v.powi(3))).tanh())
        })
        .collect();
    (got, want)
}

pub(super) const OPS: &[Op] = &[
    Op {
        id: "cpu.rmsnorm.f32",
        source_fn: "rms_norm_into",
        formula: "rmsnorm:x*w/sqrt(mean(x^2)+eps)",
        workload: OP_WORKLOAD,
        run: run_rmsnorm,
    },
    Op {
        id: "cpu.layernorm.f32",
        source_fn: "layer_norm_into",
        formula: "layernorm:(x-mean)*w/sqrt(var+eps)+b",
        workload: OP_WORKLOAD,
        run: run_layernorm,
    },
    Op {
        id: "cpu.gelu.f32",
        source_fn: "gelu",
        formula: "gelu_tanh:0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))",
        workload: OP_WORKLOAD,
        run: run_gelu,
    },
];

pub(super) fn op(id: &str) -> &'static Op {
    OPS.iter()
        .find(|o| o.id == id)
        .unwrap_or_else(|| panic!("{id}: no op harness entry"))
}

/// The oracle part of an op receipt's input set: the oracle and the definition it evaluates.
pub(super) fn op_oracle(o: &Op) -> String {
    format!("{OP_ORACLE}:{}", o.formula)
}

#[derive(Debug, Default)]
struct OpMeasured {
    max_abs_err: f64,
    max_rel_err: f64,
}

fn measure_op(o: &Op, w: OpWorkload) -> OpMeasured {
    let mut rng = rand::rngs::StdRng::seed_from_u64(w.seed);
    let mut m = OpMeasured::default();
    for _ in 0..w.trials {
        let (got, want) = (o.run)(&mut rng, w.n);
        assert_eq!(got.len(), want.len(), "{}: output length", o.id);
        assert_eq!(got.len(), w.n, "{}: output length", o.id);
        let scale = want.iter().fold(0.0f64, |a, r| a.max(r.abs()));
        assert!(
            scale > 0.0,
            "{}: an all-zero reference measures nothing",
            o.id
        );
        for (g, r) in got.iter().zip(&want) {
            let abs = (f64::from(*g) - r).abs();
            assert!(abs.is_finite(), "{}: non-finite output", o.id);
            m.max_abs_err = m.max_abs_err.max(abs);
            m.max_rel_err = m.max_rel_err.max(abs / scale);
        }
    }
    m
}

fn op_row<'a>(r: &'a Registry, id: &str) -> &'a OpRow {
    r.ops()
        .iter()
        .find(|row| row.kernel_id == id)
        .unwrap_or_else(|| panic!("{id} is not an ops[] row"))
}

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn op_receipt(o: &Op, row: &OpRow) -> serde_json::Value {
    let served = measure_op(o, o.workload);
    let set = InputSet::from_op_tree(&repo_root(), row, "none", &device(), &op_oracle(o))
        .unwrap_or_else(|e| panic!("{}: {e}", o.id));
    let mut doc = receipt_header_for(o.id, o.source_fn, &row.precision, &set);
    doc["oracle"] = OP_ORACLE.into();
    doc["oracle_formula"] = o.formula.into();
    doc["oracle_independent"] = true.into();
    doc["workload"] = serde_json::json!({
        "n": o.workload.n,
        "seed": o.workload.seed,
        "trials": o.workload.trials,
    });
    doc["served"] =
        serde_json::json!({"max_abs_err": served.max_abs_err, "max_rel_err": served.max_rel_err});
    doc["tolerance_rel"] = tolerance_from(served.max_rel_err).into();
    doc["margin"] = serde_json::json!({
        "error_model": row.error_model,
        "verdict": "not_modelled",
        "why": "aprender-kernel-oracle has no bound for this error model yet; tolerance_rel decides",
    });
    doc
}

#[test]
#[ignore = "writes receipts; run on the host whose arch it admits (see module doc)"]
fn emit_op_parity_receipts() {
    let out = repo_root().join(std::env::var("KREG_RECEIPT_OUT").expect("KREG_RECEIPT_OUT"));
    std::fs::create_dir_all(&out).expect("receipt dir");
    let r = registry().expect("registry");
    let only = std::env::var("KREG_EMIT").ok();
    for o in OPS.iter().filter(|o| {
        only.as_deref()
            .is_none_or(|w| w.split(',').any(|w| w == o.id))
    }) {
        let row = op_row(&r, o.id);
        assert_eq!(
            row.source_fn, o.source_fn,
            "{}: harness and row name different fns",
            o.id
        );
        let doc = op_receipt(o, row);
        let path = out.join(format!("{}.json", o.id));
        let text = serde_json::to_string_pretty(&doc).expect("receipt json");
        std::fs::write(&path, text + "\n").expect("write receipt");
        eprintln!("{}: {}", path.display(), doc["served"]);
    }
}

/// Every `op_receipts` entry of the ratchet file with the receipt it points at.
pub(super) fn committed_ops() -> Vec<(serde_json::Value, serde_json::Value)> {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("../kernel-registry-receipts.json"))
            .expect("receipts json");
    doc["op_receipts"]
        .as_array()
        .expect("op_receipts")
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

/// FALSIFY-KREG-009 for `ops[]`: a committed op receipt names its row and oracle, and still holds
/// on this host — its own seed and workload, re-measured, stay within `tolerance_rel` — and a
/// `precision=f32` row is f32-exact against the f64 definition.
#[test]
fn committed_op_parity_receipts_hold_on_this_host() {
    let r = registry().expect("registry");
    let all = committed_ops();
    assert!(
        !all.is_empty(),
        "no op receipts: this test would pass measuring nothing"
    );
    for (entry, rc) in all {
        let id = rc["kernel_id"].as_str().expect("kernel_id");
        let path = entry["receipt"].as_str().expect("receipt path");
        assert_eq!(rc["schema"], SCHEMA, "{path}: schema");
        assert_eq!(
            entry["kernel_id"], id,
            "{path}: ratchet entry names another op"
        );
        assert_eq!(
            entry["host_arch"], rc["host_arch"],
            "{path}: ratchet entry arch"
        );
        let row = op_row(&r, id);
        let o = op(id);
        assert_eq!(
            row.tolerance, path,
            "{id}: tolerance must point at its receipt"
        );
        assert_eq!(
            rc["registry_precision"],
            row.precision.as_str(),
            "{id}: precision drifted"
        );
        assert_eq!(rc["oracle"], OP_ORACLE, "{path}: oracle");
        assert_eq!(
            rc["oracle_formula"], o.formula,
            "{path}: the oracle's definition changed"
        );
        assert_eq!(rc["oracle_independent"], true, "{path}: oracle_independent");
        let wl = &rc["workload"];
        let w = OpWorkload {
            n: wl["n"].as_u64().expect("n") as usize,
            seed: wl["seed"].as_u64().expect("seed"),
            trials: wl["trials"].as_u64().expect("trials") as usize,
        };
        let bound = rc["tolerance_rel"].as_f64().expect("tolerance_rel");
        let now = measure_op(o, w);
        assert!(
            now.max_rel_err <= bound,
            "{id}: max_rel_err {} > receipt tolerance {bound} ({path})",
            now.max_rel_err
        );
        match row.precision.as_str() {
            "f32" => assert!(
                now.max_rel_err < F32_CEILING,
                "{id}: precision=f32, yet the error against f64 is {} ≥ {F32_CEILING}",
                now.max_rel_err
            ),
            p => panic!("{id}: no precision check for {p}"),
        }
    }
}

#[test]
fn every_op_harness_entry_names_its_row() {
    let r = registry().expect("registry");
    for o in OPS {
        assert_eq!(op_row(&r, o.id).source_fn, o.source_fn, "{}", o.id);
    }
}
