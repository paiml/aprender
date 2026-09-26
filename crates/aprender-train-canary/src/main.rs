//! Performance canary: trueno WGPU vs Burn WGPU throughput
//!
//! usage: aprender-train-canary [--iters N] [--json PATH]
//!
//! Both sides are timed host-to-host: upload A and B, matmul, read C back, on
//! the one hardware wgpu adapter present (the JSON's `gpu` names it). Burn's
//! owned input copies are made outside the clock.
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

/// Software rasterisers wgpu can enumerate; a canary on one of these measures a CPU.
const SOFTWARE: [&str; 3] = ["llvmpipe", "lavapipe", "swiftshader"];

/// The one hardware adapter both backends run on. More than one distinct hardware
/// adapter is an error: nothing could then say which one Burn's DiscreteGpu(0) is.
fn pick_adapter() -> Result<(u32, String), String> {
    let all = trueno::backends::gpu::GpuDevice::list_adapters();
    let hw: Vec<&(u32, String, String)> = all
        .iter()
        .filter(|(_, name, _)| !SOFTWARE.iter().any(|s| name.to_lowercase().contains(s)))
        .collect();
    let mut names: Vec<&str> = hw.iter().map(|(_, n, _)| n.as_str()).collect();
    names.dedup();
    match (hw.first(), names.len()) {
        (Some((idx, name, _)), 1) => Ok((*idx, name.clone())),
        _ => Err(format!(
            "need exactly one hardware wgpu adapter, found {all:?}"
        )),
    }
}

fn run() -> Result<(), String> {
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::Wgpu;
    use burn::tensor::{Tensor, TensorData};
    type B = Wgpu;
    let (iters, json) = args()?;

    let (adapter_index, adapter) = pick_adapter()?;
    eprintln!("=== Performance Canary: trueno vs Burn WGPU (median of {iters}) on {adapter} ===\n");

    let gpu = trueno::backends::gpu::GpuDevice::new_with_adapter_index(adapter_index)
        .map_err(|e| format!("{e}"))?;
    let dev = WgpuDevice::DiscreteGpu(0);
    let mut rows = Vec::new();

    for (m, k, n) in SIZES {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 13 + 7) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let mut c = vec![0.0f32; m * n];

        // Burn's from_data takes ownership, so each call needs its own host copy. The
        // copy is made BEFORE the clock starts: trueno reads the same slices in place,
        // and a 1.5 GB memcpy on the Burn side only would be measured as matmul.
        let burn_inputs = || {
            (
                TensorData::new(a.clone(), [m, k]),
                TensorData::new(b.clone(), [k, n]),
            )
        };
        let burn_once = |(da, db): (TensorData, TensorData)| {
            let ab = Tensor::<B, 2>::from_data(da, &dev);
            let bb = Tensor::<B, 2>::from_data(db, &dev);
            // to_data() blocks until the result is on the host: the timed unit.
            ab.matmul(bb).to_data()
        };

        // Warmup, synchronised on both sides so compile time stays out of the timings.
        gpu.matmul(&a, &b, &mut c, m, k, n)
            .map_err(|e| format!("{e}"))?;
        let _ = burn_once(burn_inputs());

        let mut t_trueno = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            gpu.matmul(&a, &b, &mut c, m, k, n)
                .map_err(|e| format!("{e}"))?;
            t_trueno.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mut t_burn = Vec::with_capacity(iters);
        for _ in 0..iters {
            let inputs = burn_inputs();
            let t = Instant::now();
            let _ = burn_once(inputs);
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
            "{{\n  \"gpu\": {adapter:?},\n  \"iters\": {iters},\n  \"statistic\": \"median\",\n  \"rows\": [\n{}\n  ]\n}}\n",
            rows.join(",\n")
        );
        std::fs::write(&path, body).map_err(|e| format!("write {path}: {e}"))?;
    }
    Ok(())
}
