//! `tensor_ab` — tensor-level GPU-vs-CPU A/B on REAL model bytes.
//!
//! The end-to-end cosine gate (`apr parity`) reports ONE number over a whole forward.
//! A kernel under development is wrong at the block level long before the model runs,
//! so this compares a SINGLE named tensor's GEMV: CPU reference vs GPU kernel, on the
//! same bytes, and names the FIRST differing row instead of one aggregate.
//!
//! Usage:
//!   cargo run --release --features cuda --example tensor_ab -- <model.gguf> <name-substr>
//!   cargo run --release --features cuda --example tensor_ab -- <model.gguf> --list
//!   ... --perturb        plant a fault in the GPU input so the comparison MUST go RED
//!
//! `--perturb` exists because a comparison that has never rejected a wrong answer is
//! not evidence about a right one. Run it before trusting a green.
//!
//! Exit: 0 match, 1 mismatch, 2 usage/not-found, 3 no GPU kernel for this qtype (the
//! correct pre-kernel answer for F16 / IQ4_XS until those land).

use realizar::apr::dequant::dequantize_f16 as apr_dequantize_f16;
use realizar::cuda::{BoundWeight, CudaExecutor, WeightQuantType};
use realizar::gguf::MappedGGUFModel;
use realizar::quantize::iq_dispatch::dequantize_iq_tensor;
use realizar::quantize::{
    dequantize_f16, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k, dequantize_q5_0,
    dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};
use trueno_gpu::driver::GpuBuffer;

fn qtype_name(q: u32) -> &'static str {
    match q {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        23 => "IQ4_XS",
        _ => "?",
    }
}

/// Bytes on disk for `n` elements of `qtype`. `None` = shape not modelled here.
fn tensor_bytes(qtype: u32, n: usize) -> Option<usize> {
    let sb = |per256: usize| n.checked_mul(per256).map(|v| v / 256);
    match qtype {
        0 => Some(n * 4),
        1 => Some(n * 2),
        2 => Some(n / 32 * 18),
        3 => Some(n / 32 * 20),
        6 => Some(n / 32 * 22),
        8 => Some(n / 32 * 34),
        12 => sb(144),
        13 => sb(176),
        14 => sb(210),
        23 => sb(136), // IQ4_XS: 136 B / 256 elems
        _ => None,
    }
}

/// CPU reference dequantisation — the SAME functions the product ships, never a re-derivation.
///
/// F16 has TWO public implementations (`quantize::dequantize_f16` and
/// `apr::dequant::dequantize_f16`). They are cross-checked against each other here rather
/// than one being trusted: if the two references disagree, the A/B is meaningless and we
/// want to know that before blaming a kernel.
fn cpu_dequant(qtype: u32, bytes: &[u8], n: usize) -> Option<Vec<f32>> {
    let v = match qtype {
        0 => Some(
            bytes
                .chunks_exact(4)
                .take(n)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>(),
        ),
        1 => {
            let a = dequantize_f16(bytes).ok()?;
            let b = apr_dequantize_f16(bytes, n);
            let m = a.len().min(b.len()).min(n);
            let bad = a[..m].iter().zip(&b[..m]).position(|(x, y)| {
                // NaN-aware: both non-finite at the same index is AGREEMENT. A plain
                // `(x-y).abs() > EPS` is false for NaN and an `all(<= EPS)` is false
                // for NaN, so the naive forms disagree with each other on NaN input
                // and one of them reports a reference conflict that does not exist.
                if !x.is_finite() && !y.is_finite() {
                    return false;
                }
                (x - y).abs() > f32::EPSILON
            });
            if let Some(i) = bad {
                eprintln!("REFERENCE DISAGREEMENT between the two public F16 dequantisers:");
                eprintln!(
                    "  quantize::dequantize_f16   len={} first6={:?}",
                    a.len(),
                    &a[..6.min(a.len())]
                );
                eprintln!(
                    "  apr::dequant::dequantize_f16 len={} first6={:?}",
                    b.len(),
                    &b[..6.min(b.len())]
                );
                eprintln!(
                    "  first differing index {i}: quantize={} apr={}",
                    a[i], b[i]
                );
                eprintln!("  (compared {m} elements; n={n})");
                std::process::exit(4);
            }
            Some(a)
        },
        2 => dequantize_q4_0(bytes).ok(),
        3 => dequantize_q4_1(bytes).ok(),
        6 => dequantize_q5_0(bytes).ok(),
        8 => dequantize_q8_0(bytes).ok(),
        12 => dequantize_q4_k(bytes).ok(),
        13 => dequantize_q5_k(bytes).ok(),
        14 => dequantize_q6_k(bytes).ok(),
        23 => dequantize_iq_tensor(qtype, bytes).ok(),
        _ => None,
    }?;
    Some(v)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    d / (na.sqrt() * nb.sqrt())
}

#[allow(clippy::too_many_lines)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: tensor_ab <model.gguf> <name-substr>|--list [--perturb]");
        std::process::exit(2);
    }
    let path = &args[1];
    let needle = &args[2];
    let perturb = args.iter().any(|a| a == "--perturb");

    let mapped = MappedGGUFModel::from_path(path).expect("open gguf");
    if needle == "--list" {
        for t in &mapped.model.tensors {
            println!(
                "{:<40} dims={:?} qtype={} ({})",
                t.name,
                t.dims,
                t.qtype,
                qtype_name(t.qtype)
            );
        }
        return;
    }

    let Some(t) = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name.contains(needle.as_str()))
    else {
        eprintln!("no tensor matching {needle}");
        std::process::exit(2);
    };
    if t.dims.len() != 2 {
        eprintln!(
            "{}: not 2-D (dims={:?}) — GEMV needs a matrix",
            t.name, t.dims
        );
        std::process::exit(2);
    }
    // TensorInfo.dims is stored REVERSED relative to GGUF `ne`. Verified against
    // token_embd.weight: raw dims=[248320, 2560] while ne=[n_embd=2560, n_vocab=248320]
    // (which is what `apr tensors` displays). So ne[0] — the fastest-varying axis, the
    // GEMV's input width — is dims[1], and the row count is dims[0]. Getting this
    // backwards silently transposes the matvec and every row "mismatches".
    let out_dim = usize::try_from(t.dims[0]).expect("out_dim");
    let in_dim = usize::try_from(t.dims[1]).expect("in_dim");
    let n_elem = in_dim * out_dim;
    let Some(nbytes) = tensor_bytes(t.qtype, n_elem) else {
        eprintln!(
            "qtype {} ({}) size not modelled",
            t.qtype,
            qtype_name(t.qtype)
        );
        std::process::exit(2);
    };
    println!("tensor : {}", t.name);
    println!("shape  : in={in_dim} out={out_dim} ({n_elem} elems)");
    println!(
        "qtype  : {} ({})  bytes={nbytes}",
        t.qtype,
        qtype_name(t.qtype)
    );

    // TensorInfo.offset is relative to the TENSOR DATA SECTION, not to the file, even
    // though its doc comment in gguf/types.rs says "Offset in the file". Reading it as a
    // file offset yields a slice from the wrong place that still dequantises "successfully"
    // into garbage — measured: 3016 non-finite outputs and max|d| 192 on a tensor whose
    // real range is [-0.2567, 0.1308] per `apr tensors --stats`. Both legs saw the SAME
    // wrong bytes, so they agreed with each other and the A/B looked merely noisy.
    let off = mapped
        .model
        .tensor_data_start
        .checked_add(usize::try_from(t.offset).expect("offset"))
        .expect("absolute offset");
    let Some(bytes) = mapped.tensor_slice(off, nbytes) else {
        eprintln!("tensor bytes lie outside the file (offset={off} len={nbytes})");
        std::process::exit(2);
    };

    // Deterministic input vector — a fixed, reproducible probe, not rand.
    let x: Vec<f32> = (0..in_dim)
        .map(|i| ((i % 17) as f32).mul_add(0.031, -0.25))
        .collect();

    // ---- CPU leg -------------------------------------------------------------
    let Some(w) = cpu_dequant(t.qtype, bytes, n_elem) else {
        eprintln!("no CPU reference dequant for qtype {}", t.qtype);
        std::process::exit(2);
    };
    if w.len() < n_elem {
        eprintln!("CPU dequant returned {} of {n_elem} elements", w.len());
        std::process::exit(2);
    }
    let mut cpu = vec![0f32; out_dim];
    for (r, o) in cpu.iter_mut().enumerate() {
        let row = &w[r * in_dim..(r + 1) * in_dim];
        *o = row.iter().zip(&x).map(|(a, b)| a * b).sum();
    }

    // ---- GPU leg -------------------------------------------------------------
    let Some(qt) = WeightQuantType::from_ggml_type(t.qtype) else {
        println!();
        println!(
            "GPU    : NO KERNEL for qtype {} ({}) — from_ggml_type returned None.",
            t.qtype,
            qtype_name(t.qtype)
        );
        println!("         This is the correct PRE-KERNEL answer, not a failure of this harness.");
        std::process::exit(3);
    };
    let mut ex = CudaExecutor::new(0).expect("cuda executor");
    ex.load_quantized_weights_with_type("ab", bytes, t.qtype)
        .expect("upload");
    let ptr = ex.get_quantized_weight_ptr("ab").expect("ptr");
    let bw = BoundWeight::bind(
        ptr,
        nbytes,
        qt,
        u32::try_from(out_dim).expect("out"),
        u32::try_from(in_dim).expect("in"),
    );
    let mut xi = x.clone();
    if perturb {
        // Plant the fault in the GPU leg ONLY, so a working comparison MUST report RED.
        xi[in_dim / 2] += 1.0;
        println!(
            "PERTURB: GPU input[{}] += 1.0 — this run MUST report MISMATCH",
            in_dim / 2
        );
    }
    let gin = GpuBuffer::from_host(ex.context(), &xi).expect("input buf");
    let gout = GpuBuffer::<f32>::new(ex.context(), out_dim).expect("out buf");
    ex.bound_gemv(&bw, &gin, &gout).expect("gemv");
    let mut gpu = vec![0f32; out_dim];
    gout.copy_to_host(&mut gpu).expect("readback");

    // ---- compare -------------------------------------------------------------
    let nan_cpu = cpu.iter().filter(|v| !v.is_finite()).count();
    let nan_gpu = gpu.iter().filter(|v| !v.is_finite()).count();
    // Tolerance is relative to the OUTPUT SCALE, not to each element: a per-element
    // max(|a|,1.0) band lets a large-magnitude row hide a large absolute error.
    let rms = (cpu
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / cpu.len().max(1) as f64)
        .sqrt();
    let tol = (1e-3 * rms).max(1e-4);
    let mut first_bad = None;
    let mut max_abs = 0f64;
    let mut n_bad = 0usize;
    for (i, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
        let d = f64::from((*a - *b).abs());
        if d > max_abs {
            max_abs = d;
        }
        if d > tol {
            n_bad += 1;
            if first_bad.is_none() {
                first_bad = Some((i, *a, *b, d));
            }
        }
    }
    let cos = cosine(&cpu, &gpu);
    println!();
    println!("rows   : {out_dim}");
    println!("rms    : {rms:.6}   tol={tol:.6} (1e-3 x rms)");
    println!("nonfin : cpu={nan_cpu} gpu={nan_gpu}");
    println!("badrows: {n_bad}");
    println!("cosine : {cos:.8}");
    println!("max|d| : {max_abs:.6}");
    if nan_cpu > 0 || nan_gpu > 0 {
        println!(
            "RESULT : NON-FINITE VALUES — cpu={nan_cpu} gpu={nan_gpu}; the comparison is void"
        );
        std::process::exit(1);
    }
    match first_bad {
        None => {
            println!("RESULT : MATCH");
            std::process::exit(0);
        },
        Some((i, a, b, d)) => {
            let blk = (i * in_dim) / 256;
            println!("RESULT : MISMATCH at row {i} (first of possibly many)");
            println!("         cpu={a:.6}  gpu={b:.6}  |d|={d:.6}");
            println!("         row {i} starts in super-block ~{blk}");
            std::process::exit(1);
        },
    }
}
