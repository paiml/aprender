//! Performance canary: trueno WGPU vs Burn WGPU throughput
//!
//! usage: aprender-train-canary [--iters N] [--json PATH]
//!
//! Both sides are timed host-to-host: upload A and B, matmul, read C back.
//! Each size gets one synchronised warmup per side (shader compile, pipeline
//! cache) and then N timed iterations; the reported time is the MEDIAN, so a
//! single stall does not move the ratio. ratio = burn_ms / trueno_ms, so a
//! ratio above 1.0 means trueno is faster (#3174).

use std::time::Instant;

const SIZES: [(usize, usize, usize); 5] = [
    (4, 2560, 9728),   // seq_len=4, hidden→intermediate (training workload)
    (32, 2560, 9728),  // seq_len=32
    (128, 2560, 9728), // seq_len=128
    (4, 2560, 151936), // lm_head (small seq)
    (32, 2560, 4096),  // Q projection
];

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn args() -> Result<(usize, Option<String>), String> {
    let mut iters = 10;
    let mut json = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--iters" => {
                iters = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|&v: &usize| v > 0)
                    .ok_or("--iters needs a positive integer")?;
            }
            "--json" => json = Some(it.next().ok_or("--json needs a path")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok((iters, json))
}

fn run() -> Result<(), String> {
    use burn::backend::Wgpu;
    use burn::tensor::{Tensor, TensorData};
    type B = Wgpu;
    let dev = Default::default();
    let (iters, json) = args()?;

    eprintln!("=== Performance Canary: trueno vs Burn WGPU (median of {iters}) ===\n");

    let gpu = trueno::backends::gpu::GpuDevice::new().map_err(|e| format!("{e}"))?;
    let mut rows = Vec::new();

    for (m, k, n) in SIZES {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let mut c = vec![0.0f32; m * n];

        let burn_once = || {
            let ab = Tensor::<B, 2>::from_data(TensorData::new(a.clone(), [m, k]), &dev);
            let bb = Tensor::<B, 2>::from_data(TensorData::new(b.clone(), [k, n]), &dev);
            // to_data() blocks until the result is on the host: the timed unit.
            ab.matmul(bb).to_data()
        };

        // Warmup, synchronised on both sides so compile time stays out of the timings.
        gpu.matmul(&a, &b, &mut c, m, k, n)
            .map_err(|e| format!("{e}"))?;
        let _ = burn_once();

        let mut t_trueno = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            gpu.matmul(&a, &b, &mut c, m, k, n)
                .map_err(|e| format!("{e}"))?;
            t_trueno.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mut t_burn = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let _ = burn_once();
            t_burn.push(t.elapsed().as_secs_f64() * 1000.0);
        }

        let (trueno_ms, burn_ms) = (median(t_trueno), median(t_burn));
        let ratio = burn_ms / trueno_ms;
        let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        eprintln!(
            "  [{m}x{k}x{n}] trueno={trueno_ms:.1}ms ({:.1} GFLOP/s) burn={burn_ms:.1}ms ({:.1} GFLOP/s) ratio={ratio:.2}x",
            gflop / (trueno_ms / 1000.0),
            gflop / (burn_ms / 1000.0)
        );
        rows.push(format!(
            "    {{\"size\": \"{m}x{k}x{n}\", \"trueno_ms\": {trueno_ms:.3}, \"burn_ms\": {burn_ms:.3}, \"ratio\": {ratio:.4}}}"
        ));
    }

    eprintln!("\nratio > 1.0 = trueno faster, ratio < 1.0 = burn faster");
    if let Some(path) = json {
        let body = format!(
            "{{\n  \"iters\": {iters},\n  \"statistic\": \"median\",\n  \"rows\": [\n{}\n  ]\n}}\n",
            rows.join(",\n")
        );
        std::fs::write(&path, body).map_err(|e| format!("write {path}: {e}"))?;
    }
    Ok(())
}
