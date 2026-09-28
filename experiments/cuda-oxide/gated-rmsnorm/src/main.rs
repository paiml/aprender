//! #3522 O-1 A/B: the cuda-oxide `gated_rmsnorm` against the hand PTX `gdn_gated_rmsnorm`, same data, same GPU.
//!
//! Parity is against an f64 CPU reference of the serve math; timing is the GPU-event median of 5x100 warm
//! launches, one launch per side per sample. The last line per shape is machine-read into the
//! `apr-kernel-receipt/v1` row: `RECEIPT head_dim=.. heads=.. cos=.. maxdiff=.. oxide_us=.. ptx_us=..`.
//!
//! Usage: `cargo oxide run -- [baseline-ptx-dir]` (default `baseline-ptx`); each shape needs
//! `gdn_gated_rmsnorm_h{head_dim}_n{heads}.<arch>.ptx` there, emitted by `emit-baseline`.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gated_rmsnorm_oxide::kernels;
use std::sync::Arc;

const EPS: f32 = 1e-6;
/// (head_dim, heads): Qwen3.5-0.8B first (the receipt row), then wider GDN layers for the record.
const SHAPES: [(usize, usize); 4] = [(128, 16), (128, 32), (128, 48), (128, 64)];
const WARMUP: usize = 20;
const ITERS: usize = 100;
const REPS: usize = 5;

fn inputs(n: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x = (0..n * d).map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0).collect();
    let g = (0..n * d).map(|i| ((i * 53 % 97) as f32 / 16.0) - 3.0).collect();
    let w = (0..d).map(|i| 0.5 + (i * 7 % 13) as f32 / 13.0).collect();
    (x, g, w)
}

fn cpu_reference(x: &[f32], g: &[f32], w: &[f32], d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for (cx, cg) in x.chunks_exact(d).zip(g.chunks_exact(d)) {
        let sq: f64 = cx.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let scale = 1.0 / (sq / d as f64 + f64::from(EPS)).sqrt();
        for i in 0..d {
            let gi = f64::from(cg[i]);
            let silu = gi / (1.0 + (-gi).exp());
            out.push((f64::from(cx[i]) * scale * f64::from(w[i]) * silu) as f32);
        }
    }
    out
}

fn parity(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut md) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(want) {
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        na += a * a;
        nb += b * b;
        md = md.max((a - b).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), md)
}

fn median_us(stream: &Arc<CudaStream>, mut launch: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        launch();
    }
    stream.synchronize().expect("sync");
    let flag = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut t: Vec<f64> = (0..REPS)
        .map(|_| {
            let start = stream.record_event(flag).expect("event");
            for _ in 0..ITERS {
                launch();
            }
            let end = stream.record_event(flag).expect("event");
            f64::from(start.elapsed_ms(&end).expect("elapsed")) * 1000.0 / ITERS as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "baseline-ptx".into());
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    let arch = format!("sm{major}{minor}");
    let module = kernels::load(&ctx).expect("oxide module");
    println!("gated_rmsnorm A/B on {arch}: oxide #[kernel] vs hand-PTX gdn_gated_rmsnorm, eps={EPS}");
    let mut failed = false;
    for (d, n) in SHAPES {
        let (x, g, w) = inputs(n, d);
        let want = cpu_reference(&x, &g, &w, d);
        let dx = DeviceBuffer::from_host(&stream, &x).expect("x");
        let dg = DeviceBuffer::from_host(&stream, &g).expect("gate");
        let dw = DeviceBuffer::from_host(&stream, &w).expect("weight");
        let mut dout = DeviceBuffer::<f32>::zeroed(&stream, n * d).expect("out");
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (d as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut oxide = || {
            // SAFETY: grid x block covers exactly n*d = dout.len() threads, one per element (the DisjointSlice
            // launch contract); every input is n*d or d long.
            unsafe { module.gated_rmsnorm(&stream, cfg, &dx, &dg, &dw, &mut dout, d as u32, EPS) }
                .expect("oxide launch");
        };
        oxide();
        let oxide_us = median_us(&stream, &mut oxide);
        let (cos_o, md_o) = parity(&dout.to_host_vec(&stream).expect("read"), &want);

        let path = format!("{dir}/gdn_gated_rmsnorm_h{d}_n{n}.{arch}.ptx");
        let ptx = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let hm = ctx.load_module_from_ptx_src(&ptx).expect("hand-PTX module");
        let func = hm.load_function("gdn_gated_rmsnorm").expect("gdn_gated_rmsnorm");
        let hout = DeviceBuffer::<f32>::zeroed(&stream, n * d).expect("out");
        let ptrs = [dx.cu_deviceptr(), dg.cu_deviceptr(), dw.cu_deviceptr(), hout.cu_deviceptr()];
        let hand = || {
            let mut p = ptrs;
            let mut params: [*mut std::ffi::c_void; 4] = [
                (&raw mut p[0]).cast(),
                (&raw mut p[1]).cast(),
                (&raw mut p[2]).cast(),
                (&raw mut p[3]).cast(),
            ];
            // SAFETY: the four u64 device pointers the hand PTX declares, each buffer n*d (weight d) long.
            unsafe {
                cuda_core::launch_kernel_on_stream(&func, (n as u32, 1, 1), (32, 1, 1), 0, &stream, &mut params)
            }
            .expect("hand-PTX launch");
        };
        hand();
        let (cos_h, md_h) = parity(&hout.to_host_vec(&stream).expect("read"), &want);
        let ptx_us = median_us(&stream, hand);

        let ok = cos_o >= 0.9999 && md_o < 1e-3;
        failed |= !ok;
        println!(
            "  h={d} n={n}: oxide {oxide_us:.3}us cos={cos_o:.7} md={md_o:.2e} | hand {ptx_us:.3}us cos={cos_h:.7} md={md_h:.2e} | ratio {:.3} {}",
            oxide_us / ptx_us,
            if ok { "PARITY" } else { "PARITY-FAIL" }
        );
        println!(
            "RECEIPT head_dim={d} heads={n} cos={cos_o:.9} maxdiff={md_o:.3e} oxide_us={oxide_us:.3} ptx_us={ptx_us:.3}"
        );
    }
    if failed {
        std::process::exit(1);
    }
}
