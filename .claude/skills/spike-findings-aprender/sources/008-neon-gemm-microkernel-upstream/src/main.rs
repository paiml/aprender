//! Spike 008 driver: correctness and timing of `trueno::blis::gemm_blis` on aarch64 against a plain
//! 8-accumulator loop and faer, on transformer-shaped and square GEMMs.
//!
//! Run from the spike directory:
//!   CARGO_TARGET_DIR=../../../target cargo run --release -- before   # scalar microkernel
//!   CARGO_TARGET_DIR=../../../target cargo run --release -- after    # NEON 8x6 microkernel
//! Writes `results-<label>.json` and prints Markdown to stdout (saved as RUN-OUTPUT-<label>.md).
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize, Clone)]
struct Row {
    shape: String,
    m: usize,
    n: usize,
    k: usize,
    gflop: f64,
    blis_ms: f64,
    blis_gfs: f64,
    gemm_ms: f64,
    gemm_gfs: f64,
    plain_ms: f64,
    plain_gfs: f64,
    faer_ms: f64,
    faer_gfs: f64,
    blis_err: f64,
    gemm_err: f64,
    plain_err: f64,
    faer_err: f64,
}

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn gen(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg(seed);
    (0..n).map(|_| r.next_f32()).collect()
}

/// f64 reference, row-major C = A(m×k) · B(k×n).
fn naive_f64(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p] as f64;
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j] as f64;
            }
        }
    }
    c
}

/// Spike-005 style dot product with 8 independent accumulators (LLVM vectorises this).
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let mut acc = [0.0f32; 8];
    for c in 0..chunks {
        for k in 0..8 {
            acc[k] += a[c * 8 + k] * b[c * 8 + k];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for i in chunks * 8..n {
        s += a[i] * b[i];
    }
    s
}

/// Plain loops over a pre-transposed B (n×k), the spike-005 `linear` path.
fn plain8(m: usize, n: usize, k: usize, a: &[f32], bt: &[f32], c: &mut [f32]) {
    for i in 0..m {
        let ar = &a[i * k..(i + 1) * k];
        for j in 0..n {
            c[i * n + j] = dot8(ar, &bt[j * k..(j + 1) * k]);
        }
    }
}

fn transpose(k: usize, n: usize, b: &[f32]) -> Vec<f32> {
    let mut bt = vec![0.0f32; n * k];
    for p in 0..k {
        for j in 0..n {
            bt[j * k + p] = b[p * n + j];
        }
    }
    bt
}

fn max_rel_err(c: &[f32], r: &[f64]) -> f64 {
    let scale = r.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    c.iter().zip(r).fold(0.0f64, |m, (x, y)| m.max((*x as f64 - y).abs())) / scale
}

/// Median wall-clock milliseconds over `reps` runs (after one warm-up).
fn median_ms<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f();
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    t[t.len() / 2]
}

fn main() {
    let label = std::env::args().nth(1).unwrap_or_else(|| "run".to_string());
    // (name, m, n, k): C[m×n] = A[m×k] · B[k×n]. Transformer rows are tokens (129 = 2048/16 + REG).
    let shapes: Vec<(&str, usize, usize, usize)> = vec![
        ("bolt-tiny proj 256→256", 129, 256, 256),
        ("bolt-tiny FF wi 256→1024", 129, 1024, 256),
        ("bolt-tiny FF wo 1024→256", 129, 256, 1024),
        ("bolt-small proj 512→512", 129, 512, 512),
        ("bolt-small FF wi 512→2048", 129, 2048, 512),
        ("bolt-small FF wo 2048→512", 129, 512, 2048),
        ("chronos-2 FF wi 768→3072", 129, 3072, 768),
        ("square 64", 64, 64, 64),
        ("square 128", 128, 128, 128),
        ("square 256", 256, 256, 256),
        ("square 512", 512, 512, 512),
        ("square 1024", 1024, 1024, 1024),
        ("odd 129×131×67", 129, 131, 67),
        ("odd 33×50×100", 33, 50, 100),
        ("odd 7×5×300", 7, 5, 300),
        ("k=1 64×64×1", 64, 64, 1),
        ("k=3 64×48×3", 64, 48, 3),
    ];
    println!("# Spike 008 — gemm_blis on aarch64 ({label})\n");
    println!("Machine: {}\n", std::process::Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default());
    println!("| shape | m | n | k | MFLOP | gemm_blis ms | GF/s | blis::gemm ms | GF/s | plain8 ms | GF/s | faer ms | GF/s | blis/plain | err blis | err gemm | err plain | err faer |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut rows = Vec::new();
    for (name, m, n, k) in shapes {
        let a = gen(m * k, 1);
        let b = gen(k * n, 2);
        let bt = transpose(k, n, &b);
        let r = naive_f64(m, n, k, &a, &b);
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let reps = ((2.0e9 / flops) as usize).clamp(5, 200);

        let mut c_blis = vec![0.0f32; m * n];
        let blis_ms = median_ms(reps, || {
            c_blis.iter_mut().for_each(|v| *v = 0.0);
            trueno::blis::gemm_blis(m, n, k, &a, &b, &mut c_blis, None).expect("gemm_blis");
        });
        let mut c_gemm = vec![0.0f32; m * n];
        let gemm_ms = median_ms(reps, || {
            c_gemm.iter_mut().for_each(|v| *v = 0.0);
            trueno::blis::gemm(m, n, k, &a, &b, &mut c_gemm).expect("gemm");
        });
        let mut c_plain = vec![0.0f32; m * n];
        let plain_ms = median_ms(reps, || plain8(m, n, k, &a, &bt, &mut c_plain));
        let fa = faer::Mat::<f32>::from_fn(m, k, |i, j| a[i * k + j]);
        let fb = faer::Mat::<f32>::from_fn(k, n, |i, j| b[i * n + j]);
        let mut fc = faer::Mat::<f32>::zeros(m, n);
        let faer_ms = median_ms(reps, || {
            faer::linalg::matmul::matmul(fc.as_mut(), faer::Accum::Replace, fa.as_ref(), fb.as_ref(), 1.0f32, faer::Par::Seq);
        });
        let c_faer: Vec<f32> = (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| fc[(i, j)]).collect();

        let gfs = |ms: f64| flops / ms / 1e6;
        let row = Row {
            shape: name.to_string(), m, n, k, gflop: flops / 1e9,
            blis_ms, blis_gfs: gfs(blis_ms), gemm_ms, gemm_gfs: gfs(gemm_ms),
            plain_ms, plain_gfs: gfs(plain_ms), faer_ms, faer_gfs: gfs(faer_ms),
            blis_err: max_rel_err(&c_blis, &r), gemm_err: max_rel_err(&c_gemm, &r),
            plain_err: max_rel_err(&c_plain, &r), faer_err: max_rel_err(&c_faer, &r),
        };
        println!(
            "| {} | {} | {} | {} | {:.1} | {:.3} | {:.1} | {:.3} | {:.1} | {:.3} | {:.1} | {:.3} | {:.1} | {:.2}× | {:.1e} | {:.1e} | {:.1e} | {:.1e} |",
            row.shape, m, n, k, flops / 1e6, row.blis_ms, row.blis_gfs, row.gemm_ms, row.gemm_gfs,
            row.plain_ms, row.plain_gfs, row.faer_ms, row.faer_gfs, row.blis_ms / row.plain_ms,
            row.blis_err, row.gemm_err, row.plain_err, row.faer_err
        );
        rows.push(row);
    }
    let worst = rows.iter().map(|r| r.blis_err.max(r.gemm_err)).fold(0.0f64, f64::max);
    println!("\nWorst gemm_blis / gemm relative error vs f64 reference: {worst:.2e} (bar: 1e-5)");
    let tf: Vec<&Row> = rows.iter().filter(|r| r.shape.starts_with("bolt-tiny")).collect();
    let sum = |f: fn(&Row) -> f64| tf.iter().map(|r| f(r)).sum::<f64>();
    println!("Bolt-tiny layer (proj + wi + wo): gemm_blis {:.2} ms, plain8 {:.2} ms, faer {:.2} ms", sum(|r| r.blis_ms), sum(|r| r.plain_ms), sum(|r| r.faer_ms));
    std::fs::write(format!("results-{label}.json"), serde_json::to_string_pretty(&rows).expect("json")).expect("write");
}
